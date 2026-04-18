# ============================================================
# main.py - FastAPI Application Entry Point
# Defines all API endpoints for the Voice Assistant
# ============================================================

import os
import sys

# Add backend directory to Python path for clean imports
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio

from config import settings, validate_settings
from services.llm_service import get_llm_response
from services.text_to_speech import synthesize_speech
from utils.audio_utils import save_upload_to_temp, cleanup_file, cleanup_old_files

# ============================================================
# App Initialization
# ============================================================

app = FastAPI(
    title="Voice Assistant API",
    description="Speech-to-Text → LLM → Text-to-Speech pipeline",
    version="1.0.0"
)

# Allow frontend (running on different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio files at /audio/<filename>
os.makedirs(settings.TEMP_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=settings.TEMP_DIR), name="audio")

# Serve frontend files at root
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ============================================================
# Pydantic Models (Request/Response shapes)
# ============================================================

class TextChatRequest(BaseModel):
    """Request body for text-based chat (no audio)."""
    message: str
    conversation_history: Optional[list] = []
    language: Optional[str] = "en"
    speak_response: Optional[bool] = True


class TextChatResponse(BaseModel):
    """Response from text chat endpoint."""
    user_message: str
    assistant_response: str
    audio_url: Optional[str] = None
    language: str


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Serve the frontend HTML page."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Voice Assistant API is running!", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Use this to verify the server is running and config is valid.
    """
    config_ok = validate_settings()
    return {
        "status": "ok" if config_ok else "config_error",
        "engines": {
            "stt": settings.STT_ENGINE,
            "llm": settings.LLM_ENGINE,
            "tts": settings.TTS_ENGINE,
        },
        "language": settings.DEFAULT_LANGUAGE,
    }



@app.post("/api/text-chat", response_model=TextChatResponse)
async def text_chat(request: TextChatRequest):
    """
    Text-based chat endpoint (no audio input).
    Useful for testing the LLM and TTS without microphone.
    """
    try:
        # --- Step 1: Get LLM response ---
        llm_response = await get_llm_response(
            request.message,
            request.conversation_history,
            request.language
        )

        audio_url = None

        # --- Step 2: Optionally synthesize speech ---
        if request.speak_response:
            output_path = await synthesize_speech(llm_response, request.language)
            audio_url = f"/audio/{os.path.basename(output_path)}"

        return TextChatResponse(
            user_message=request.message,
            assistant_response=llm_response,
            audio_url=audio_url,
            language=request.language,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/synthesize")
async def synthesize_only(
    text: str = Form(...),
    language: str = Form(default="en"),
):
    """
    Convert text to speech without LLM.
    Useful for testing TTS in isolation.
    """
    try:
        output_path = await synthesize_speech(text, language)
        audio_filename = os.path.basename(output_path)
        return {"audio_url": f"/audio/{audio_filename}", "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cleanup")
async def cleanup_temp_files():
    """Remove old temporary audio files (older than 1 hour)."""
    cleanup_old_files(max_age_seconds=3600)
    return {"message": "Cleanup complete"}


# ============================================================
# Startup / Shutdown Events
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Run when the server starts."""
    print("\n" + "="*50)
    print("🎙️  Voice Assistant API Starting...")
    print("="*50)
    validate_settings()
    print(f"✅ LLM Engine: {settings.LLM_ENGINE}")
    print(f"✅ TTS Engine: {settings.TTS_ENGINE}")
    print(f"✅ Default Language: {settings.DEFAULT_LANGUAGE}")
    print(f"🌐 API Docs: http://localhost:{settings.PORT}/docs")
    print(f"🌐 Frontend: http://localhost:{settings.PORT}/")
    print("="*50 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Run when the server shuts down - clean up temp files."""
    print("\n🧹 Cleaning up temporary files...")
    cleanup_old_files(max_age_seconds=0)  # Delete ALL temp files on shutdown


# ============================================================
# Run directly with: python main.py
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        PORT = int(os.getenv("PORT", 8000)),
        reload=True,    # Auto-reload on code changes (dev mode)
        log_level="info"
    )
