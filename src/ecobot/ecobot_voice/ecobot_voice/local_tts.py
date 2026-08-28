"""livekit-agents TTS plugin adapter around Piper.

Verified against the installed livekit-agents version (checked by reading
livekit.agents.tts.ChunkedStream._run's real signature and copying the
push/initialize pattern used by the installed livekit-plugins-google TTS
ChunkedStream — see _run below).
"""
from livekit.agents import tts, utils

from .local_engines import synthesize_speech, PIPER_SAMPLE_RATE


class LocalPiperTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=PIPER_SAMPLE_RATE,
            num_channels=1,
        )

    def synthesize(self, text, *, conn_options=None):
        return _PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _PiperChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        pcm, sample_rate = await synthesize_speech(self._input_text)
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type='audio/pcm',
        )
        output_emitter.push(pcm)
