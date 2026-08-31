"""Gemini vision wrapper for per-plant health assessment.

One call per plant: all captured wrist-camera viewpoint JPEGs go in
together with their labels (front/right/left/top), so the model can reason
across viewpoints (e.g. "yellowing visible from top but not front") rather
than getting four disconnected single-image opinions.

Never raises — a mission run should never crash because a network call or
a malformed response failed. Every failure mode degrades to a
health='unknown' result with an 'error' field set instead.
"""
import json
import logging
import os
import re

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

from google import genai
from google.genai import types

logger = logging.getLogger('ecobot_mission.gemini_client')

PROMPT = """You are a plant inspection assistant reviewing close-up photos
taken by a robot's wrist camera of ONE plant, from several viewpoints.

Report only what you can actually support. This matters more than
completeness: the report is read as fact, and a confident guess about a
plant someone is caring for is worse than a gap. So:

  * Omit any field you are not reasonably sure of. Leave it out of the
    JSON entirely rather than guessing, writing "unknown", or filling it
    with a plausible-sounding value.
  * Species facts (life span, sizes, growth rate, temperature, water)
    should only be given once you are confident of the identification. If
    you are not sure what the plant is, omit the whole species_data and
    mature_size sections.
  * Anything you can see in the photos — colour, condition, structure —
    you may report from the photos alone.
  * Never invent a measurement of THIS plant that the photos cannot show.

Respond with ONLY a single JSON object, no prose, no markdown fences.
Include only the keys you are confident about; every one is optional
except health, confidence and notes:

{
  "health": "healthy" | "stressed" | "diseased" | "dead" | "unknown",
  "confidence": <float 0.0-1.0>,
  "issues": [<short strings, e.g. "yellowing leaves", "pest damage">],
  "notes": "<one or two sentence plain-English summary>",

  "identification": {
    "common_name": "", "scientific_name": "", "family": "",
    "confidence_rating": "high" | "medium" | "low"
  },
  "visible_condition": {
    "overall_health": "", "leaf_colour_and_condition": "",
    "stem_and_foliage_structure": ""
  },
  "symptoms": [<what is visibly wrong, one short phrase each>],
  "treatments": [<practical steps for the symptoms above>],
  "species_data": {
    "life_span": "", "average_height": "", "average_diameter": "",
    "average_girth": "", "seed_size": "", "growth_rate": "",
    "optimal_temperature": "", "water_need": ""
  },
  "mature_size": {
    "height_range_and_progress": "", "diameter_progress": "",
    "life_cycle_timeline": ""
  },
  "habitat": {
    "climate_or_zones": "", "indoor_potted_context": ""
  }
}

If the photos do not clearly show a plant, or are too blurry or dark to
judge, set health to "unknown", say why in notes, and omit everything
else."""

_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


class GeminiClient:
    def __init__(self, api_key=None, model=None, timeout_s=20.0, max_retries=1):
        # GEMINI_API_KEY is the name the dashboard already uses; GOOGLE_API_KEY
        # is accepted too so existing setups keep working.
        api_key = (api_key
                   or os.environ.get('GEMINI_API_KEY')
                   or os.environ.get('GOOGLE_API_KEY'))
        if not api_key:
            raise RuntimeError(
                'GEMINI_API_KEY not set. Put it in '
                'ecobot_bringup/.env (see .env.example).')
        self._client = genai.Client(api_key=api_key)
        self._model = model or os.environ.get(
            'ECOBOT_MISSION_GEMINI_MODEL', 'gemini-3.1-pro-preview')
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def assess_plant(self, jpeg_images, labels=None):
        """jpeg_images: list[bytes]. Returns a result dict; never raises."""
        if not jpeg_images:
            return self._degraded('no images captured')

        parts = []
        for i, jpg in enumerate(jpeg_images):
            label = labels[i] if labels and i < len(labels) else f'view {i}'
            parts.append(types.Part.from_text(text=f'-- {label} --'))
            parts.append(types.Part.from_bytes(data=jpg, mime_type='image/jpeg'))
        parts.append(types.Part.from_text(text=PROMPT))

        last_err = None
        for _ in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=types.Content(role='user', parts=parts),
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        # The full report — identification, condition,
                        # symptoms, treatments, species facts — does not
                        # fit in 512, and a truncated reply parses as a
                        # failure rather than a partial answer.
                        max_output_tokens=2048,
                        response_mime_type='application/json',
                        http_options=types.HttpOptions(
                            timeout=int(self._timeout_s * 1000)),
                    ),
                 )
                return self._parse(response.text)
            except Exception as e:
                last_err = e
                logger.warning(f'gemini call attempt failed: {e}')
        return self._degraded(f'gemini call failed: {last_err}')

    def assess_plant_live(self, jpeg_images, labels=None):
        """Asynchronously connect via Gemini Live streaming session, send the images and PROMPT,
        and stream the response back. Falls back to standard assess_plant on error."""
        import asyncio
        try:
            return asyncio.run(self._assess_plant_live_async(jpeg_images, labels))
        except Exception as e:
            logger.warning(f'Gemini Live connection failed (falling back to standard assess_plant): {e}')
            return self.assess_plant(jpeg_images, labels)

    async def _assess_plant_live_async(self, jpeg_images, labels=None):
        from google.genai import types

        if not jpeg_images:
            return self._degraded('no images captured')

        # Construct the parts to send to the Live session
        parts = []
        for i, jpg in enumerate(jpeg_images):
            label = labels[i] if labels and i < len(labels) else f'view {i}'
            parts.append(types.Part.from_text(text=f'-- {label} --'))
            parts.append(types.Part.from_bytes(data=jpg, mime_type='image/jpeg'))
        parts.append(types.Part.from_text(text=PROMPT))

        # Setup Live Connect Config
        config = types.LiveConnectConfig(
            response_modalities=['TEXT'],
            system_instruction=types.Content(parts=[types.Part.from_text(text=PROMPT)]),
            generation_config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
            )
        )

        text_responses = []
        # Connect to the live session asynchronously
        async with self._client.aio.live.connect(model=self._model, config=config) as session:
            # Send the parts
            await session.send(input=parts, end_of_turn=True)

            # Receive the streamed responses
            async for response in session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts or []:
                            if part.text:
                                text_responses.append(part.text)
                    if server_content.turn_complete or server_content.generation_complete:
                        break

        full_text = "".join(text_responses)
        logger.info(f"Gemini Live session streamed complete response: {full_text}")
        return self._parse(full_text)

    def _parse(self, text):
        if not text:
            return self._degraded('empty response')
        try:
            data = json.loads(text)
        except Exception:
            # Model wrapped JSON in prose/markdown fences despite
            # response_mime_type — extract the first {...} block.
            m = _JSON_OBJECT_RE.search(text)
            if not m:
                return self._degraded(f'unparseable response: {text[:200]}')
            try:
                data = json.loads(m.group(0))
            except Exception:
                return self._degraded(f'unparseable response: {text[:200]}')

        result = {
            'health': str(data.get('health', 'unknown')),
            'confidence': float(data.get('confidence', 0.0)),
            'issues': list(data.get('issues', [])),
            'notes': str(data.get('notes', '')),
            'error': None,
        }

        # The optional sections. Anything the model left out, or filled
        # with an empty string or a placeholder, is dropped rather than
        # carried through — a report is easier to trust when a missing
        # fact is visibly missing instead of showing as "unknown".
        placeholder = {'', 'unknown', 'n/a', 'na', 'none', 'not sure',
                       'not applicable', '-'}

        def clean(value):
            if isinstance(value, dict):
                out = {k: clean(v) for k, v in value.items()}
                out = {k: v for k, v in out.items() if v not in (None, {}, [])}
                return out or None
            if isinstance(value, list):
                out = [clean(v) for v in value]
                out = [v for v in out if v not in (None, {}, [])]
                return out or None
            if value is None:
                return None
            text = str(value).strip()
            return None if text.lower() in placeholder else text

        for key in ('identification', 'visible_condition', 'species_data',
                    'mature_size', 'habitat', 'symptoms', 'treatments'):
            section = clean(data.get(key))
            if section:
                result[key] = section
        return result

    def _degraded(self, reason):
        return {
            'health': 'unknown', 'confidence': 0.0, 'issues': [],
            'notes': reason, 'error': reason,
        }
