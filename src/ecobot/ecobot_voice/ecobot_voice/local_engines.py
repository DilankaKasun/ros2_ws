"""Local (on-Jetson) speech engines, kept independent of the livekit-agents
plugin ABC so they're testable and reusable even if the thin adapter classes
in local_stt.py/local_tts.py need small fixes for a specific livekit-agents
version.

STT: STT_MODEL selects the engine —
  - a Hugging Face repo id containing "/" (e.g. "seniruk/whisper-small-si",
    a community Whisper fine-tune) loads via transformers.pipeline, since
    such checkpoints are standard transformers format and are NOT loadable
    by faster-whisper without a separate ct2-transformers-converter step.
  - anything else (e.g. "small", "base") is treated as a faster-whisper
    model size (CTranslate2 — faster, but only for stock Whisper sizes).
TTS: TTS_ENGINE selects the engine — "piper" (default: fast, ARM-friendly,
  CLI-based, verified working) or "omnivoice" (k2-fsa/OmniVoice — UNVERIFIED
  here: it's a self-hosted diffusion-LM TTS model per its model card, no
  cloud API exists despite the project's TTS_ENGINE=cloud naming, needs a
  real CUDA GPU, and voice cloning needs ref_audio+ref_text, not just a text
  prompt. Likely too slow for real-time on Jetson-class hardware — kept as
  an explicit opt-in, not the default.

All config below is read via os.environ.get() *inside* functions, not as
module-level constants. Observed bug: livekit-agents' CLI loads .env via
python-dotenv during its own startup, which happens AFTER this module is
first imported (voice_agent.py imports it up top, before cli.run_app()
runs) — module-level `X = os.environ.get(...)` froze in the fallback
defaults regardless of what .env actually set. Confirmed live: STT_MODEL
defaulted to "small" and downloaded Systran/faster-whisper-small instead of
the configured seniruk/whisper-small-si.
"""
import asyncio
import os

# ---- STT ---------------------------------------------------------------

_faster_whisper_model = None
_faster_whisper_model_id = None
_hf_stt_pipeline = None
_hf_stt_pipeline_id = None


def _is_hf_repo_id(model_id: str) -> bool:
    return '/' in model_id


def _get_faster_whisper_model(model_id: str):
    global _faster_whisper_model, _faster_whisper_model_id
    if _faster_whisper_model is None or _faster_whisper_model_id != model_id:
        from faster_whisper import WhisperModel
        # ctranslate2 doesn't reliably ship prebuilt aarch64 CUDA wheels for
        # Jetson — device defaults to cpu for that reason.
        device = os.environ.get('ECOBOT_WHISPER_DEVICE', 'cpu')
        compute_type = os.environ.get('ECOBOT_WHISPER_COMPUTE_TYPE', 'int8')
        _faster_whisper_model = WhisperModel(
            model_id, device=device, compute_type=compute_type)
        _faster_whisper_model_id = model_id
    return _faster_whisper_model


def _get_hf_stt_pipeline(model_id: str):
    global _hf_stt_pipeline, _hf_stt_pipeline_id
    if _hf_stt_pipeline is None or _hf_stt_pipeline_id != model_id:
        from transformers import pipeline
        device = os.environ.get('ECOBOT_HF_STT_DEVICE', 'cpu')
        _hf_stt_pipeline = pipeline(
            'automatic-speech-recognition', model=model_id, device=device)
        _hf_stt_pipeline_id = model_id
    return _hf_stt_pipeline


def _resample_to_16k(audio, sample_rate: int):
    if sample_rate == 16000:
        return audio
    from scipy.signal import resample_poly
    import math
    g = math.gcd(16000, sample_rate)
    return resample_poly(audio, 16000 // g, sample_rate // g)


def transcribe_pcm16(pcm_bytes: bytes, sample_rate: int) -> str:
    """Blocking. Run off the event loop via asyncio.to_thread."""
    import numpy as np

    stt_model = os.environ.get('STT_MODEL', 'small')

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio = _resample_to_16k(audio, sample_rate)

    if _is_hf_repo_id(stt_model):
        pipe = _get_hf_stt_pipeline(stt_model)
        result = pipe({'array': audio, 'sampling_rate': 16000})
        return result['text'].strip()

    model = _get_faster_whisper_model(stt_model)
    segments, _info = model.transcribe(audio, language=None, vad_filter=False)
    return ' '.join(seg.text.strip() for seg in segments).strip()


async def transcribe_pcm16_async(pcm_bytes: bytes, sample_rate: int) -> str:
    return await asyncio.to_thread(transcribe_pcm16, pcm_bytes, sample_rate)


# ---- TTS ----------------------------------------------------------------

PIPER_SAMPLE_RATE = int(os.environ.get('ECOBOT_PIPER_SAMPLE_RATE', '22050'))

_omnivoice_model = None
_omnivoice_model_id = None


async def _synthesize_piper(text: str) -> tuple[bytes, int]:
    piper_model = os.environ.get('ECOBOT_PIPER_MODEL', '')
    if not piper_model:
        raise RuntimeError(
            'ECOBOT_PIPER_MODEL is not set — point it at a downloaded '
            'Piper .onnx voice file (see .env.example).')
    piper_bin = os.environ.get('ECOBOT_PIPER_BIN', 'piper')
    sample_rate = PIPER_SAMPLE_RATE

    proc = await asyncio.create_subprocess_exec(
        piper_bin, '--model', piper_model, '--output-raw',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(text.encode('utf-8'))
    if proc.returncode != 0:
        raise RuntimeError(f'piper failed: {stderr.decode(errors="replace")}')
    return stdout, sample_rate


def _get_omnivoice_model(model_id: str, device: str):
    global _omnivoice_model, _omnivoice_model_id
    if _omnivoice_model is None or _omnivoice_model_id != model_id:
        import torch
        from omnivoice import OmniVoice
        _omnivoice_model = OmniVoice.from_pretrained(
            model_id, device_map=device, dtype=torch.float16)
        _omnivoice_model_id = model_id
    return _omnivoice_model


def _synthesize_omnivoice_sync(text: str) -> tuple[bytes, int]:
    # Verified against the real package (extracted the wheel and read
    # omnivoice/cli/infer.py + models/omnivoice.py directly — the model
    # card's example only showed voice-clone mode). generate() supports
    # three modes: voice-clone (ref_audio+ref_text), voice-design (instruct
    # string, no reference audio — e.g. "female, moderate pitch", exactly
    # what the original plan described), or auto (neither set, model picks
    # a voice). Preferring instruct here since we don't have a reference
    # clip; falls through to auto if ECOBOT_OMNIVOICE_INSTRUCT is also unset.
    import numpy as np

    model_id = os.environ.get('TTS_MODEL', 'k2-fsa/OmniVoice')
    device = os.environ.get('ECOBOT_OMNIVOICE_DEVICE', 'cuda:0')
    language = os.environ.get('ECOBOT_OMNIVOICE_LANGUAGE', 'sinhala')
    instruct = os.environ.get('ECOBOT_OMNIVOICE_INSTRUCT', 'female, moderate pitch')
    ref_audio = os.environ.get('ECOBOT_OMNIVOICE_REF_AUDIO', '') or None
    ref_text = os.environ.get('ECOBOT_OMNIVOICE_REF_TEXT', '') or None

    model = _get_omnivoice_model(model_id, device)
    audios = model.generate(
        text=text,
        language=language or None,
        ref_audio=ref_audio,
        ref_text=ref_text,
        instruct=None if ref_audio else (instruct or None),
    )
    audio = audios[0]
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    return pcm16, model.sampling_rate


async def synthesize_speech(text: str) -> tuple[bytes, int]:
    """Returns (raw PCM16 mono bytes, sample_rate)."""
    tts_engine = os.environ.get('TTS_ENGINE', 'piper')
    if tts_engine == 'omnivoice':
        # UNVERIFIED end-to-end: written from the model card's example only,
        # not run against the actual package. Needs a CUDA GPU and is likely
        # too slow for real-time use on Jetson — see module docstring.
        return await asyncio.to_thread(_synthesize_omnivoice_sync, text)
    return await _synthesize_piper(text)
