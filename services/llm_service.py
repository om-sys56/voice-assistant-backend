# ============================================================
# services/llm_service.py
# Sends user text to an LLM and gets an intelligent response
# Supports OpenAI GPT and Google Gemini
# ============================================================

from config import settings

# System prompt that defines the assistant's personality
SYSTEM_PROMPT = """You are a helpful, friendly, and concise voice assistant.
Since your responses will be converted to speech:
- Keep answers clear and conversational (2-4 sentences ideally)
- Avoid using bullet points, markdown, or special characters
- Speak naturally as if having a conversation
- If asked something complex, give the most useful summary
- Be warm, helpful, and direct"""


async def get_llm_response(
    user_message: str,
    conversation_history: list = None,
    language: str = None
) -> str:
    """
    Main entry point for LLM responses.
    Picks the right engine and returns the assistant's reply.

    Args:
        user_message: The transcribed text from the user
        conversation_history: List of past messages for context
            Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        language: Language code for response (optional)

    Returns:
        Assistant's response as a string
    """
    history = conversation_history or []
    lang = language or settings.DEFAULT_LANGUAGE

    # Add language instruction if not English
    system = SYSTEM_PROMPT
    if lang != "en":
        system += f"\n\nIMPORTANT: Respond in the same language as the user. Current language: {lang}"

    if settings.LLM_ENGINE == "openai":
        return await _get_openai_response(user_message, history, system)
    elif settings.LLM_ENGINE == "gemini":
        return await _get_gemini_response(user_message, history, system)
    else:
        raise ValueError(f"Unknown LLM engine: {settings.LLM_ENGINE}")


async def _get_openai_response(
    user_message: str,
    history: list,
    system_prompt: str
) -> str:
    """
    Get response from OpenAI's GPT models.
    Includes full conversation history for multi-turn dialogue.
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Build the messages array:
        # 1. System message (defines behavior)
        # 2. Past conversation (for context)
        # 3. New user message
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)  # Add conversation history
        messages.append({"role": "user", "content": user_message})

        # Make the API call
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=300,        # Keep responses concise for voice
            temperature=0.7,       # Balanced creativity
        )

        # Extract the response text
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"OpenAI LLM failed: {str(e)}")


async def _get_gemini_response(
    user_message: str,
    history: list,
    system_prompt: str
) -> str:
    """
    Get response from Google's Gemini models.
    Converts OpenAI-style history to Gemini format.
    """
    try:
        import google.generativeai as genai
        import asyncio

        # Configure Gemini with API key
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Create model with system instruction
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=system_prompt
        )

        # Convert history to Gemini's format (uses "parts" instead of "content")
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Start chat session with history
        chat = model.start_chat(history=gemini_history)

        # Run in executor since Gemini SDK is synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: chat.send_message(user_message)
        )

        return response.text.strip()

    except Exception as e:
        raise RuntimeError(f"Gemini LLM failed: {str(e)}")
