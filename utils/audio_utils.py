# ============================================================
# utils/audio_utils.py
# Helper functions for audio file handling and cleanup
# ============================================================

import os
import uuid
import time
from config import settings


def save_upload_to_temp(audio_bytes: bytes, extension: str = "webm") -> str:
    """
    Save uploaded audio bytes to a temporary file.
    
    Args:
        audio_bytes: Raw audio data from the browser
        extension: File extension (webm, wav, mp3, ogg)
    
    Returns:
        Full path to the saved temp file
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Create unique filename to avoid collisions
    filename = f"input_{uuid.uuid4().hex}.{extension}"
    file_path = os.path.join(settings.TEMP_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    
    return file_path


def cleanup_file(file_path: str):
    """
    Delete a temporary audio file after use.
    Silently ignores errors (file may already be gone).
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Not critical if cleanup fails


def cleanup_old_files(max_age_seconds: int = 3600):
    """
    Remove temp audio files older than max_age_seconds.
    Call this periodically to prevent disk space buildup.
    
    Args:
        max_age_seconds: Files older than this will be deleted (default: 1 hour)
    """
    if not os.path.exists(settings.TEMP_DIR):
        return

    now = time.time()
    for filename in os.listdir(settings.TEMP_DIR):
        file_path = os.path.join(settings.TEMP_DIR, filename)
        try:
            # Check file age using last modified time
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)
        except Exception:
            pass
