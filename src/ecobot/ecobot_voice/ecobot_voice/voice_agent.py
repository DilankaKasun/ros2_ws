"""EcoBot conversational agent: LiveKit Agents pipeline (STT -> LLM -> TTS)
with tool calls onto the arm/vision ROS topics.

LLM defaults to Gemma via LiveKit Inference (LLM_MODEL=google/gemma-4-31b-it)
— confirmed against LiveKit's docs (docs.livekit.io/agents/start/voice-ai/):
`from livekit.agents import inference; inference.LLM(model="provider/model")`,
billed through the existing LIVEKIT_API_KEY/SECRET, no separate GOOGLE_API_KEY
needed for this path.

STT defaults to a local Sinhala Whisper fine-tune (STT_MODEL=
seniruk/whisper-small-si) run via transformers — confirmed from the model
card as a standard transformers checkpoint, NOT a LiveKit Inference model
despite the provider/model-shaped name. See local_engines.py.

TTS defaults to Piper (fast, verified working, ARM-friendly). TTS_ENGINE=
omnivoice switches to k2-fsa/OmniVoice, which per its own model card is
self-hosted only (no cloud API despite this project's TTS_ENGINE=cloud
naming), needs a real CUDA GPU, and is UNVERIFIED end-to-end here — see
local_engines.py's synthesize_speech for the caveats before relying on it
on Jetson hardware in real time.

All credentials come from the environment (see ecobot_voice/.env.example).
This is a scaffold, not a stub: the pipeline wiring, tool definitions, and
ROS bridge are complete and exercised by ros_bridge's own IK/state logic,
but end-to-end connection to LiveKit has not been tested against a live
server in this environment, and the local_stt/local_tts adapters are
unverified against the installed livekit-agents plugin ABC (see their
module docstrings for how to check).
"""
import base64
import logging
import os
import site
import sys


def _fix_jetson_torch_cusparselt():
    """The local Sinhala STT (transformers) and OmniVoice TTS paths both
    need torch. NVIDIA's Jetson torch build (torch==...+nv24.08, confirmed
    installed here) depends on libcusparseLt.so.0, which is present under
    the pip-installed nvidia-cusparselt-cu12 package but NOT on the dynamic
    linker's search path — `import torch` fails with an ImportError despite
    the .so file existing on disk.

    Verified: setting LD_LIBRARY_PATH from *within* the same running process
    (os.environ mutation before a delayed `import torch`) does NOT fix this
    — glibc resolves LD_LIBRARY_PATH once at process start, not per dlopen().
    The fix has to be in place before the process starts, so this re-execs
    the interpreter once with the corrected environment. Safe to call
    unconditionally: no-ops if the path is already set or doesn't exist
    (e.g. on a non-Jetson dev machine without this package installed).
    """
    try:
        lib_dir = os.path.join(
            site.getusersitepackages(), 'nvidia', 'cusparselt', 'lib')
    except Exception:
        return
    if not os.path.isdir(lib_dir):
        return
    if not (sys.argv and os.path.isfile(sys.argv[0])):
        # Not a real script invocation (e.g. `python3 -c ...`, a REPL, or an
        # interactive import for testing) — sys.argv[0] won't replay through
        # execve. Skip; torch will raise its own ImportError if something
        # downstream actually needs it, which is clearer than an execve
        # failure here would be.
        return
    current = os.environ.get('LD_LIBRARY_PATH', '')
    if lib_dir in current.split(':'):
        return
    os.environ['LD_LIBRARY_PATH'] = f'{lib_dir}:{current}' if current else lib_dir
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


_fix_jetson_torch_cusparselt()

# Checked live against the installed livekit-agents (1.6.7): its CLI does
# NOT load .env itself (zero references to `dotenv` anywhere in the
# installed package, confirmed via grep) — contrary to what this module
# used to assume. Without this explicit call, LIVEKIT_URL/GOOGLE_API_KEY/etc.
# from .env never reach os.environ and worker startup fails with
# "ValueError: ws_url is required, or set LIVEKIT_URL environment variable".
#
# usecwd=True is required here, not optional: python-dotenv's find_dotenv()
# defaults to walking upward from the *importing module's* __file__ (via
# stack inspection), not from the cwd. Confirmed live — a bare
# load_dotenv() silently found nothing and left LIVEKIT_URL unset, because
# colcon copies this file into
# install/ecobot_voice/lib/python3.10/site-packages/ecobot_voice/ (ament_python
# installs are copies, not symlinks, even without --symlink-install) and
# .env sits nowhere near that tree. usecwd=True forces the cwd-based search
# instead, matching this package's documented run convention of `cd` into
# the ecobot_voice source directory before `ros2 run` (see .env.example).
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, function_tool, llm
from livekit.agents import inference
from livekit.plugins import silero

# livekit.plugins.* packages call Plugin.register_plugin() at their own
# import time, and that registration is only accepted during the framework's
# early main-thread "preload" phase (visible in the startup log as
# {"packages": ["livekit.plugins.silero", ...]}) — NOT lazily from inside
# entrypoint()/​_build_*(), which run in a per-job worker context. Confirmed
# live: `from livekit.plugins import google` inside _build_llm() crashed
# with "RuntimeError: Plugins must be registered on the main thread", while
# `silero` (imported here, at true top-level, same as it already was) is
# preloaded correctly. `livekit.agents.inference` is unaffected — it's a
# built-in module, not a registered Plugin, hence the working LLM_BACKEND=
# inference path in earlier testing. Guarded so an uninstalled optional
# plugin doesn't crash the agent for backends nobody selected.
try:
    from livekit.plugins import google
except ImportError:
    google = None
try:
    from livekit.plugins import deepgram
except ImportError:
    deepgram = None
try:
    from livekit.plugins import elevenlabs
except ImportError:
    elevenlabs = None

from .persona import INSTRUCTIONS
from .ros_bridge import RosBridge

logger = logging.getLogger('ecobot_voice')

# NOTE: every config value here is still read via os.environ.get() *inside*
# these functions rather than as module-level constants, even though
# load_dotenv() above now runs at import time. Kept this way because entry
# points other than main() (e.g. tests importing this module directly) may
# run before any .env is loaded at all — a module-level `X =
# os.environ.get(...)` would freeze in the fallback default in that case.
# Confirmed live: this exact bug made STT_MODEL default to "small" and
# download Systran/faster-whisper-small instead of the .env-configured
# seniruk/whisper-small-si.


def _build_llm():
    backend = os.environ.get('LLM_BACKEND', 'inference')
    if backend == 'inference':
        model = os.environ.get('LLM_MODEL', 'google/gemma-4-31b-it')
        return inference.LLM(model=model)
    if google is None:
        raise RuntimeError(
            'LLM_BACKEND=google but livekit-plugins-google is not installed')
    model = os.environ.get('ECOBOT_GEMINI_MODEL', 'gemini-3.1-flash-live-preview')
    return google.LLM(model=model)


def _build_stt():
    backend = os.environ.get('ECOBOT_STT_BACKEND', 'local')
    if backend == 'local':
        from .local_stt import LocalWhisperSTT
        return LocalWhisperSTT()
    if deepgram is None:
        raise RuntimeError(
            'ECOBOT_STT_BACKEND=deepgram but livekit-plugins-deepgram is not installed')
    model = os.environ.get('ECOBOT_DEEPGRAM_MODEL', 'nova-2')
    return deepgram.STT(model=model)


def _build_tts():
    # TTS_ENGINE is the single source of truth (piper | omnivoice | cloud |
    # elevenlabs) — piper/omnivoice both route through LocalPiperTTS, which
    # re-reads TTS_ENGINE itself inside local_engines.synthesize_speech() to
    # pick between them; that's an intentional two-layer read, not a
    # duplicate bug, since both layers read the same env var consistently.
    engine = os.environ.get('TTS_ENGINE', 'piper')

    if engine in ('piper', 'omnivoice'):
        from .local_tts import LocalPiperTTS
        return LocalPiperTTS()

    if engine == 'cloud':
        if google is None:
            raise RuntimeError(
                'TTS_ENGINE=cloud but livekit-plugins-google is not installed')
        # Google Cloud Text-to-Speech (a different API/product surface than
        # the Gemini Developer API used for the LLM — needs Cloud
        # Text-to-Speech enabled on the GCP project GOOGLE_API_KEY belongs
        # to; UNVERIFIED whether this key has that scope). si-LK is a real
        # documented Google Cloud TTS language; no specific voice_name is
        # pinned so Google picks its own default voice for that language.
        language = os.environ.get('ECOBOT_CLOUD_TTS_LANGUAGE', 'si-LK')
        voice_name = os.environ.get('ECOBOT_CLOUD_TTS_VOICE', '')
        kwargs = {'language': language}
        if voice_name:
            kwargs['voice_name'] = voice_name
        return google.TTS(**kwargs)

    if elevenlabs is None:
        raise RuntimeError(
            'TTS_ENGINE=elevenlabs but livekit-plugins-elevenlabs is not installed')
    voice_id = os.environ.get('ECOBOT_ELEVENLABS_VOICE_ID', '')
    if voice_id:
        return elevenlabs.TTS(voice_id=voice_id)
    return elevenlabs.TTS()

# livekit-agents' cli.run_app() and rclpy both parse sys.argv on their own
# terms (argparse vs. the --ros-args convention) and would collide if both
# saw the raw argv. main() strips the --ros-args portion for livekit before
# run_app() reads sys.argv, but rclpy still needs it — captured here first
# so RosBridge can hand it to rclpy.init() explicitly.
_ROS_ARGV = None


class EcobotAssistant(Agent):
    def __init__(self, bridge: RosBridge):
        super().__init__(instructions=INSTRUCTIONS)
        self._bridge = bridge

    @function_tool
    async def move_arm_to(self, x: float, y: float, z: float) -> str:
        """Move the wrist camera to a 3D point in meters, relative to the
        arm base (x=forward, y=left/right, z=up).

        Args:
            x: forward distance in meters
            y: sideways distance in meters (positive = left)
            z: height in meters
        """
        ok, message = self._bridge.move_arm_to(x, y, z)
        return message

    @function_tool
    async def home_arm(self) -> str:
        """Return the arm to its resting home position."""
        self._bridge.home_arm()
        return 'returning arm home'

    @function_tool
    async def start_plant_scan(self, x: float, y: float, z: float) -> str:
        """Begin an automatic multi-viewpoint scan (front/left/right/top) of
        an object at the given position, in meters relative to the arm base.

        Args:
            x: forward distance to the plant in meters
            y: sideways offset in meters
            z: height in meters
        """
        self._bridge.start_scan(x, y, z)
        return 'starting plant scan'

    @function_tool
    async def stop_scan(self) -> str:
        """Stop any in-progress scan and return the arm home."""
        self._bridge.stop_scan()
        return 'stopping scan'

    @function_tool
    async def look_at_wrist_camera(self) -> list:
        """Capture a fresh still image from the wrist-mounted camera so you
        can describe what the arm is currently looking at."""
        jpeg = self._bridge.get_wrist_frame_jpeg()
        if jpeg is None:
            return ['No recent wrist camera frame is available.']
        data_url = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')
        return [
            'Current wrist camera view:',
            llm.ImageContent(image=data_url),
        ]

    @function_tool
    async def get_robot_status(self) -> str:
        """Report the arm's current status, pose, and any active scan."""
        status = self._bridge.get_arm_status()
        scan = self._bridge.get_scanner_status()
        parts = [f"arm status: {status['status']}"]
        if status['pose']:
            x, y, z = status['pose']
            parts.append(f'arm tip at ({x:.2f}, {y:.2f}, {z:.2f})')
        if scan:
            parts.append(
                f"scan: {scan['status']} "
                f"(viewpoint {scan['viewpoint']}/{scan['total_viewpoints']})")
        return '; '.join(parts)

    @function_tool
    async def get_detected_objects(self) -> str:
        """List objects the main navigation camera currently detects,
        useful for deciding what to scan next."""
        detections = self._bridge.get_latest_detections()
        if not detections:
            return 'No objects currently detected.'
        names = [d.get('class', d.get('label', 'object')) for d in detections]
        return 'Detected: ' + ', '.join(names)


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    wrist_camera_topic = os.environ.get(
        'ECOBOT_WRIST_CAMERA_TOPIC', '/arm/camera/image_raw')
    bridge = RosBridge(wrist_camera_topic=wrist_camera_topic, ros_args=_ROS_ARGV)

    session = AgentSession(
        stt=_build_stt(),
        llm=_build_llm(),
        tts=_build_tts(),
        vad=silero.VAD.load(),
    )

    try:
        await session.start(
            agent=EcobotAssistant(bridge),
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
        await session.generate_reply(
            instructions='Greet whoever is nearby briefly and say you are ready to help.')
    finally:
        bridge.shutdown()


def main():
    global _ROS_ARGV
    logging.basicConfig(level=logging.INFO)

    _ROS_ARGV = list(sys.argv)
    from rclpy.utilities import remove_ros_args
    sys.argv = remove_ros_args(sys.argv)

    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == '__main__':
    main()
