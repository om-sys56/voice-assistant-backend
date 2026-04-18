# ============================================================
# services/text_to_speech.py
# Converts LLM text response into an audio file for playback
# Supports gTTS (free) and Google Cloud TTS (premium voices)
# ============================================================

import os
import uuid
from config import settings


async def synthesize_speech(text: str, language: str = None) -> str:
    """
    Main entry point for Text-to-Speech.
    Returns the file path of the generated audio file.

    Args:
        text: The text to convert to speech
        language: Language code like "en", "hi", "fr"

    Returns:
        File path to the generated MP3 audio file
    """
    lang = language or settings.DEFAULT_LANGUAGE

    # Make sure the temp folder exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    # Generate a unique filename for each audio response
    filename = f"response_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(settings.TEMP_DIR, filename)

    if settings.TTS_ENGINE == "gtts":
        await _synthesize_with_gtts(text, lang, output_path)
    elif settings.TTS_ENGINE == "google":
        await _synthesize_with_google_cloud(text, lang, output_path)
    else:
        raise ValueError(f"Unknown TTS engine: {settings.TTS_ENGINE}")

    return output_path


async def _synthesize_with_gtts(text: str, language: str, output_path: str):
    """
    Use gTTS (Google Text-to-Speech) - completely FREE.
    - No API key needed
    - Uses Google Translate's TTS under the hood
    - Supports 60+ languages
    - Slight robotic tone compared to premium APIs
    """
    try:
        from gtts import gTTS
        import asyncio

        # Create TTS object
        tts = gTTS(
            text=text,
            lang=language,
            slow=False  # Normal speed
        )

        # Save audio file (runs in thread pool to keep FastAPI async)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: tts.save(output_path))

    except Exception as e:
        raise RuntimeError(f"gTTS synthesis failed: {str(e)}")


async def _synthesize_with_google_cloud(text: str, language: str, output_path: str):
    """
    Use Google Cloud Text-to-Speech - PREMIUM quality.
    - Requires google_credentials.json
    - Neural voices (WaveNet, Studio) sound very natural
    - Pay per character after free tier
    """
    try:
        from google.cloud import texttospeech
        import asyncio

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

        # Create client
        client = texttospeech.TextToSpeechClient()

        # Set the text to synthesize
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Configure voice settings
        voice = texttospeech.VoiceSelectionParams(
            language_code=_get_google_lang_code(language),
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            # Optionally specify a neural voice name:
            # name="en-US-Neural2-A"
        )

        # Configure audio format
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,    # 1.0 = normal speed
            pitch=0.0,            # 0.0 = default pitch
        )

        # Run in executor (blocking call)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
        )

        # Write the audio content to file
        with open(output_path, "wb") as f:
            f.write(response.audio_content)

    except Exception as e:
        raise RuntimeError(f"Google Cloud TTS failed: {str(e)}")


def _get_google_lang_code(language: str) -> str:
    """Convert short language codes to BCP-47 format for Google APIs."""
    lang_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
        "ja": "ja-JP",
        "zh": "zh-CN",
        "ar": "ar-XA",
        "pt": "pt-BR",
        "ru": "ru-RU",
        "ko": "ko-KR",
        "it": "it-IT",
    }
    return lang_map.get(language, "en-US")
