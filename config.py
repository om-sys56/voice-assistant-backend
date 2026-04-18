# ============================================================
# config.py - Central configuration for the Voice Assistant
# Loads all settings from the .env file
# ============================================================

import os
from dotenv import load_dotenv

# Load the .env file into environment variables
load_dotenv()


class Settings:
    """
    Central settings class.
    All configuration lives here so we can change it in one place.
    """

    # --- API Keys ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # --- Engine Selections ---
    LLM_ENGINE: str = os.getenv("LLM_ENGINE", "gemini")        # "openai" or "gemini"
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "gtts")          # "gtts" or "google"

    # --- Model Names ---
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    # --- App Settings ---
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # --- Directories ---
    TEMP_DIR: str = "temp_audio"   # Folder to store temporary audio files


# Create a single shared instance
settings = Settings()

# Validate critical settings at startup
def validate_settings():
    """Check that the required API keys are present based on selected engines."""
    errors = []

    if settings.LLM_ENGINE == "gemini" and not settings.GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required when LLM_ENGINE=gemini")


    if errors:
        print("\n⚠️  Configuration Errors:")
        for e in errors:
            print(f"  ❌ {e}")
        print("\nPlease update your .env file and restart.\n")

    return len(errors) == 0
