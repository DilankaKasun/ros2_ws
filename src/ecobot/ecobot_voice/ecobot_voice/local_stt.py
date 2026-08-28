"""livekit-agents STT plugin adapter around local Whisper inference
(local_engines.py — faster-whisper or a transformers HF checkpoint,
selected by STT_MODEL's shape).

Verified against the installed livekit-agents version: _recognize_impl's
signature and the SpeechEvent/SpeechData/SpeechEventType construction below
were checked with
    python3 -c "from livekit.agents import stt; import inspect; print(inspect.signature(stt.STT._recognize_impl))"
Re-check this if the installed livekit-agents version changes — this ABC
has moved between releases.
"""
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import NotGivenOr, APIConnectOptions

from .local_engines import transcribe_pcm16_async


class LocalWhisperSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )

    async def _recognize_impl(
        self,
        buffer: 'stt.AudioBuffer',
        *,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        combined = rtc.combine_audio_frames(buffer)
        text = await transcribe_pcm16_async(
            combined.data.tobytes(), combined.sample_rate)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or 'en')],
        )
