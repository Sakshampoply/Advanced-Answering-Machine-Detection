"""
AMD (Answering Machine Detection) Python Service
Uses Gemini Flash for real-time audio analysis.

This service:
1. Exposes a WebSocket endpoint for Twilio Media Streams audio
2. Analyzes audio chunks in real-time using Gemini Flash API
3. Makes AMD decisions (human/machine) and returns them immediately
4. Logs results back to Next.js backend for database storage
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import json
import asyncio
import base64
import logging
import struct
import io
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import tempfile

from google import genai
from google.genai import types
import httpx
from typing import Optional as _Optional

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # runtime optional
import websockets

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# Configuration
GEMINI_MODEL = "gemini-live-2.5-flash-preview"
DECISION_THRESHOLD_CONFIDENCE = 0.75

# Hugging Face Inference API config
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_ASR_MODEL = os.getenv("HF_ASR_MODEL", "distil-whisper/distil-small.en")
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "tiny.en")
USE_LOCAL_WHISPER_FALLBACK = os.getenv(
    "USE_LOCAL_WHISPER_FALLBACK", "true"
).lower() in ("1", "true", "yes")
PRELOAD_LOCAL_WHISPER = os.getenv("PRELOAD_LOCAL_WHISPER", "false").lower() in (
    "1",
    "true",
    "yes",
)
_whisper_model = None  # lazy-loaded local whisper model

# Global flag to skip remote HF calls after repeated 410s
HF_REMOTE_DISABLED = False

# AMD System Prompt
AMD_SYSTEM_PROMPT = """You are an advanced Answering Machine Detection (AMD) system.

Analyze audio and classify as HUMAN or MACHINE:
- HUMAN: Natural speech, immediate greeting ("hello", "hi"), conversational
- MACHINE: Robotic voice, long messages (>5 words), beep tones, IVR, voicemail

Respond with ONLY valid JSON (no markdown, no extra text):
{
    "decision": "human" | "machine" | "uncertain",
    "confidence": 0.0-1.0,
    "reason": "brief explanation (max 20 words)"
}

CRITICAL: Make decisions as soon as possible (~500-1000ms of audio).
"""

# Global state
active_calls = {}


def pcmu_to_wav(pcmu_data: bytes) -> bytes:
    """Convert raw PCMU audio to WAV format."""
    # PCMU parameters (Twilio Media Streams standard)
    sample_rate = 8000
    num_channels = 1

    # Create WAV header
    num_samples = len(pcmu_data)
    byte_rate = sample_rate * num_channels * 1  # 1 byte per sample for PCMU
    block_align = num_channels * 1

    wav_header = io.BytesIO()

    # RIFF header
    wav_header.write(b"RIFF")
    wav_header.write(struct.pack("<I", 36 + num_samples))
    wav_header.write(b"WAVE")

    # fmt subchunk
    wav_header.write(b"fmt ")
    wav_header.write(struct.pack("<I", 16))  # Subchunk1Size
    wav_header.write(struct.pack("<H", 7))  # AudioFormat (7 = PCMU/G711)
    wav_header.write(struct.pack("<H", num_channels))
    wav_header.write(struct.pack("<I", sample_rate))
    wav_header.write(struct.pack("<I", byte_rate))
    wav_header.write(struct.pack("<H", block_align))
    wav_header.write(struct.pack("<H", 8))  # BitsPerSample

    # data subchunk
    wav_header.write(b"data")
    wav_header.write(struct.pack("<I", num_samples))
    wav_header.write(pcmu_data)

    return wav_header.getvalue()


def pcmu_to_pcm16(pcmu_data: bytes) -> bytes:
    """Convert PCMU (G.711 μ-law) audio to PCM 16-bit.

    Gemini Live API expects PCM audio, not PCMU.
    """
    # Lookup table for PCMU to linear PCM conversion
    PCMU_TABLE = [
        -32124,
        -31100,
        -30076,
        -29052,
        -28028,
        -27004,
        -25980,
        -24956,
        -23932,
        -22908,
        -21884,
        -20860,
        -19836,
        -18812,
        -17788,
        -16764,
        -15996,
        -15484,
        -14972,
        -14460,
        -13948,
        -13436,
        -12924,
        -12412,
        -11900,
        -11388,
        -10876,
        -10364,
        -9852,
        -9340,
        -8828,
        -8316,
        -7932,
        -7676,
        -7420,
        -7164,
        -6908,
        -6652,
        -6396,
        -6140,
        -5884,
        -5628,
        -5372,
        -5116,
        -4860,
        -4604,
        -4348,
        -4092,
        -3900,
        -3772,
        -3644,
        -3516,
        -3388,
        -3260,
        -3132,
        -3004,
        -2876,
        -2748,
        -2620,
        -2492,
        -2364,
        -2236,
        -2108,
        -1980,
        -1884,
        -1820,
        -1756,
        -1692,
        -1628,
        -1564,
        -1500,
        -1436,
        -1372,
        -1308,
        -1244,
        -1180,
        -1116,
        -1052,
        -988,
        -924,
        -876,
        -844,
        -812,
        -780,
        -748,
        -716,
        -684,
        -652,
        -620,
        -588,
        -556,
        -524,
        -492,
        -460,
        -428,
        -396,
        -372,
        -356,
        -340,
        -324,
        -308,
        -292,
        -276,
        -260,
        -244,
        -228,
        -212,
        -196,
        -180,
        -164,
        -148,
        -132,
        -120,
        -112,
        -104,
        -96,
        -88,
        -80,
        -72,
        -64,
        -56,
        -48,
        -40,
        -32,
        -24,
        -16,
        -8,
        0,
        32124,
        31100,
        30076,
        29052,
        28028,
        27004,
        25980,
        24956,
        23932,
        22908,
        21884,
        20860,
        19836,
        18812,
        17788,
        16764,
        15996,
        15484,
        14972,
        14460,
        13948,
        13436,
        12924,
        12412,
        11900,
        11388,
        10876,
        10364,
        9852,
        9340,
        8828,
        8316,
        7932,
        7676,
        7420,
        7164,
        6908,
        6652,
        6396,
        6140,
        5884,
        5628,
        5372,
        5116,
        4860,
        4604,
        4348,
        4092,
        3900,
        3772,
        3644,
        3516,
        3388,
        3260,
        3132,
        3004,
        2876,
        2748,
        2620,
        2492,
        2364,
        2236,
        2108,
        1980,
        1884,
        1820,
        1756,
        1692,
        1628,
        1564,
        1500,
        1436,
        1372,
        1308,
        1244,
        1180,
        1116,
        1052,
        988,
        924,
        876,
        844,
        812,
        780,
        748,
        716,
        684,
        652,
        620,
        588,
        556,
        524,
        492,
        460,
        428,
        396,
        372,
        356,
        340,
        324,
        308,
        292,
        276,
        260,
        244,
        228,
        212,
        196,
        180,
        164,
        148,
        132,
        120,
        112,
        104,
        96,
        88,
        80,
        72,
        64,
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
    ]

    pcm_data = bytearray()
    for byte in pcmu_data:
        # Convert PCMU byte to 16-bit PCM
        pcm_value = PCMU_TABLE[byte]
        # Pack as little-endian 16-bit signed integer
        pcm_data.extend(struct.pack("<h", pcm_value))

    return bytes(pcm_data)


def pcm16_to_wav(pcm16_data: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw PCM16 mono data into a standard PCM WAV container."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm16_data)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # Subchunk1Size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm16_data)
    return buf.getvalue()


class CallSession:
    """Track state for each call."""

    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.decision = None
        self.confidence = 0.0
        self.audio_chunks = 0
        self.total_ms = 0
        self.buffer_pcm16 = bytearray()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events."""
    logger.info("AMD service starting...")
    # Optional: preload local whisper to reduce first-call latency
    global _whisper_model
    if (
        PRELOAD_LOCAL_WHISPER
        and USE_LOCAL_WHISPER_FALLBACK
        and WhisperModel is not None
    ):
        try:
            if _whisper_model is None:
                logger.info(
                    f"[HF] Preloading local Whisper model: {LOCAL_WHISPER_MODEL}"
                )
                _whisper_model = WhisperModel(
                    LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8"
                )
        except Exception as e:
            logger.warning(f"[HF] Failed to preload local Whisper: {e}")
    yield
    logger.info("Shutting down...")
    active_calls.clear()


app = FastAPI(
    title="AMD Service",
    description="Answering Machine Detection using Gemini Flash",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "ok",
        "service": "AMD",
        "model": GEMINI_MODEL,
        "active_calls": len(active_calls),
    }


@app.post("/predict")
async def predict_audio(audio: UploadFile = File(...)):
    """
    Process pre-recorded audio file (fallback for testing).

    Returns:
    {
        "label": "human" | "machine" | "uncertain",
        "confidence": 0.0-1.0,
        "reason": "explanation"
    }
    """
    try:
        audio_data = await audio.read()
        logger.info(f"Processing: {audio.filename} ({len(audio_data)} bytes)")

        # Determine MIME type
        mime_type = "audio/wav"
        if audio.filename.endswith(".mp3"):
            mime_type = "audio/mpeg"
        elif audio.filename.endswith(".ogg"):
            mime_type = "audio/ogg"

        # Use Gemini 2.0 Flash for batch processing
        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(
            [
                {"mime_type": mime_type, "data": audio_data},
                AMD_SYSTEM_PROMPT,
            ]
        )

        response_text = response.text.strip()
        logger.info(f"Response: {response_text}")

        # Extract JSON
        try:
            result = json.loads(response_text)
            decision = result.get("decision", "uncertain").lower()
            label = (
                "human"
                if decision == "human"
                else "machine" if decision == "machine" else "human"
            )

            return {
                "label": label,
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse: {response_text[:100]}")
            return {"label": "uncertain", "confidence": 0.5, "reason": "parse error"}

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/amd/{call_sid}")
async def websocket_amd(websocket: WebSocket, call_sid: str):
    """Real-time AMD using Gemini Live API."""
    await websocket.accept()
    session = CallSession(call_sid)
    active_calls[call_sid] = session

    logger.info(f"WebSocket opened for call: {call_sid}")

    decision_sent = False
    audio_buffer = bytearray()

    try:
        # Wait for first audio chunk before connecting to Gemini Live
        logger.info(f"Waiting for first audio chunk for call: {call_sid}")

        first_audio_received = False
        while not first_audio_received and not decision_sent:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "audio":
                    audio_b64 = message.get("data")
                    if audio_b64:
                        audio_chunk = base64.b64decode(audio_b64)
                        audio_buffer.extend(audio_chunk)
                        session.audio_chunks += 1
                        session.total_ms += len(audio_chunk) / (8000 * 2 / 1000)

                        logger.info(
                            f"Call {call_sid}: chunk {session.audio_chunks}, "
                            f"~{session.total_ms:.0f}ms audio"
                        )
                        first_audio_received = True
                        break

                elif msg_type == "end":
                    logger.info(f"Call ended before getting audio: {call_sid}")
                    return

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {call_sid}: {e}")

        if not first_audio_received:
            logger.warning(f"No audio received for call {call_sid}, closing")
            return

        # Now connect to Gemini Live API
        logger.info(f"Initializing Gemini Live API for call: {call_sid}")

        config = {
            "system_instruction": {"parts": [{"text": AMD_SYSTEM_PROMPT}]},
            "response_modalities": ["TEXT"],
        }

        async with client.aio.live.connect(
            model=GEMINI_MODEL, config=config
        ) as session_conn:
            logger.info("Gemini Live API session started.")

            # Send the first audio chunk we already received
            initial_chunk_pcm = pcmu_to_pcm16(bytes(audio_buffer))
            await session_conn.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            data=initial_chunk_pcm,
                            mime_type="audio/pcm;rate=8000",
                        )
                    ]
                )
            )
            logger.info(f"Sent first audio chunk to Gemini for {call_sid}")

            async def send_audio():
                nonlocal decision_sent
                try:
                    while not decision_sent:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        msg_type = message.get("type")

                        if msg_type == "audio":
                            audio_b64 = message.get("data")
                            if audio_b64:
                                audio_chunk = base64.b64decode(audio_b64)
                                pcm16_chunk = pcmu_to_pcm16(audio_chunk)
                                await session_conn.send(
                                    input=types.LiveClientRealtimeInput(
                                        media_chunks=[
                                            types.Blob(
                                                data=pcm16_chunk,
                                                mime_type="audio/pcm;rate=8000",
                                            )
                                        ]
                                    )
                                )
                            else:
                                logger.debug(f"No audio data in message for {call_sid}")
                        elif msg_type == "call_ended":
                            logger.info(f"Call ended message received for {call_sid}")
                            break
                        else:
                            logger.warning(
                                f"Unknown message type for {call_sid}: {msg_type}"
                            )
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"Upstream WebSocket closed for {call_sid}: {e}")
                except Exception as e:
                    logger.error(
                        f"Error in send_audio for {call_sid}: {e}", exc_info=True
                    )
                finally:
                    logger.info(f"send_audio task finished for {call_sid}")

            async def receive_responses():
                nonlocal decision_sent
                try:
                    logger.info(f"Starting to receive Gemini responses for {call_sid}")
                    text_buffer = ""

                    def _clean_and_append(txt: str) -> None:
                        nonlocal text_buffer
                        # Strip common fences and prefixes the model may include
                        s = txt.strip()
                        if s.startswith("```"):
                            # remove surrounding code fences
                            parts = s.split("```")
                            if len(parts) > 1:
                                s = parts[1]
                            s = s.strip()
                            if s.startswith("json\n"):
                                s = s[5:]
                            s = s.strip()
                        # Append to rolling buffer
                        text_buffer += s

                    def _extract_json_candidate(s: str) -> str | None:
                        # Heuristic: take from first '{' to last '}'
                        start = s.find("{")
                        end = s.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            return s[start : end + 1]
                        return None

                    async for response in session_conn.receive():
                        if decision_sent:
                            break

                        sc = getattr(response, "server_content", None)
                        if sc and sc.model_turn:
                            for part in sc.model_turn.parts or []:
                                if getattr(part, "text", None) and not decision_sent:
                                    logger.info(
                                        f"Gemini response for {call_sid}: {part.text}"
                                    )
                                    _clean_and_append(part.text)

                                    candidate = _extract_json_candidate(text_buffer)
                                    if candidate:
                                        try:
                                            result = json.loads(candidate)
                                            decision = result.get(
                                                "decision", "uncertain"
                                            )
                                            confidence = float(
                                                result.get("confidence", 0.5)
                                            )
                                            reason = result.get("reason", "")

                                            session.decision = decision
                                            session.confidence = confidence

                                            logger.info(
                                                f"AMD Decision {call_sid}: {decision} "
                                                f"(confidence: {confidence:.2%})"
                                            )

                                            await websocket.send_json(
                                                {
                                                    "type": "analysis_complete",
                                                    "result": decision,
                                                    "confidence": confidence,
                                                    "reason": reason,
                                                }
                                            )
                                            decision_sent = True
                                        except json.JSONDecodeError:
                                            # Not yet a full JSON; continue accumulating
                                            pass

                        # If server indicates end of turn, try final parse
                        if (
                            not decision_sent
                            and sc
                            and getattr(sc, "turn_complete", False)
                        ):
                            candidate = _extract_json_candidate(text_buffer)
                            if candidate:
                                try:
                                    result = json.loads(candidate)
                                    decision = result.get("decision", "uncertain")
                                    confidence = float(result.get("confidence", 0.5))
                                    reason = result.get("reason", "")

                                    session.decision = decision
                                    session.confidence = confidence

                                    logger.info(
                                        f"AMD Decision {call_sid}: {decision} "
                                        f"(confidence: {confidence:.2%})"
                                    )

                                    await websocket.send_json(
                                        {
                                            "type": "analysis_complete",
                                            "result": decision,
                                            "confidence": confidence,
                                            "reason": reason,
                                        }
                                    )
                                    decision_sent = True
                                except Exception as e:
                                    logger.error(f"Failed to parse final response: {e}")
                                    await websocket.send_json(
                                        {
                                            "type": "error",
                                            "error": "Failed to parse analysis",
                                        }
                                    )
                                    decision_sent = True

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error receiving responses: {e}", exc_info=True)

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_responses())

            try:
                await asyncio.gather(send_task, receive_task)
            except Exception as e:
                logger.error(f"Error in gather: {e}", exc_info=True)
                if not send_task.done():
                    send_task.cancel()
                if not receive_task.done():
                    receive_task.cancel()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_sid}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if call_sid in active_calls:
            del active_calls[call_sid]
        logger.info(f"Session cleaned up: {call_sid}")


async def hf_transcribe(wav_bytes: bytes) -> str:
    """Call Hugging Face Inference API (ASR) and return transcript text with fallbacks."""
    if not HF_API_TOKEN:
        logger.warning("HUGGINGFACE_API_TOKEN not set; returning empty transcript")
        return ""
    # If we've previously observed HF returning 410, skip remote and trigger local fallback
    global HF_REMOTE_DISABLED
    if HF_REMOTE_DISABLED:
        raise RuntimeError("HF remote disabled due to previous 410 responses")

    models_to_try = [
        HF_ASR_MODEL,
        "openai/whisper-tiny",
        "openai/whisper-base.en",
        "distil-whisper/distil-medium.en",
    ]

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "audio/wav"}
    timeout = httpx.Timeout(25.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        last_error = None
        for model_id in models_to_try:
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            try:
                r = await client.post(url, content=wav_bytes, headers=headers)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "text" in data:
                    return data["text"] or ""
                if (
                    isinstance(data, list)
                    and data
                    and isinstance(data[0], dict)
                    and "text" in data[0]
                ):
                    return data[0]["text"] or ""
                last_error = ValueError(
                    f"Unexpected ASR response format from {model_id}"
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                code = e.response.status_code
                logger.warning(f"[HF] Transcription error from {model_id}: {code}")
                if code == 410:
                    # Model deprecated or unavailable: disable further remote attempts
                    HF_REMOTE_DISABLED = True
                if code in (401, 403):
                    break
            except Exception as e:
                last_error = e
                logger.warning(f"[HF] Transcription exception from {model_id}: {e}")
        if last_error:
            raise last_error
        return ""


def local_transcribe_wav(wav_bytes: bytes) -> str:
    """Transcribe using local faster-whisper as a fallback. Returns transcript or empty string.

    Requires faster-whisper installed and ffmpeg available on PATH.
    """
    global _whisper_model
    if not USE_LOCAL_WHISPER_FALLBACK:
        return ""
    if WhisperModel is None:
        logger.warning("[HF] Local fallback requested but faster-whisper not installed")
        return ""
    try:
        if _whisper_model is None:
            logger.info(f"[HF] Loading local Whisper model: {LOCAL_WHISPER_MODEL}")
            _whisper_model = WhisperModel(
                LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8"
            )
        # Write wav to temp file for robust decoding
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            segments, _ = _whisper_model.transcribe(
                tmp.name, vad_filter=True, beam_size=1
            )
            text_parts = []
            for seg in segments:
                if getattr(seg, "text", None):
                    text_parts.append(seg.text)
            return " ".join(text_parts).strip()
    except Exception as e:
        logger.warning(f"[HF] Local whisper fallback failed: {e}")
        return ""


def classify_transcript(text: str) -> tuple[str, float, str]:
    """Simple heuristics for AMD based on transcript."""
    t = (text or "").strip().lower()
    if not t:
        return ("uncertain", 0.5, "no transcript yet")

    vm_patterns = [
        "leave a message",
        "leave your message",
        "after the tone",
        "at the tone",
        "record your message",
        "not available",
        "voicemail",
        "mailbox",
        "beep",
    ]
    if any(p in t for p in vm_patterns):
        return ("machine", 0.95, "voicemail phrase detected")

    words = t.split()
    if len(words) >= 8:
        return ("machine", 0.8, "long monologue early")

    if t.startswith("hello") or t.startswith("hi") or t.startswith("hey"):
        if len(words) <= 5:
            return ("human", 0.8, "short greeting")

    return ("uncertain", 0.55, "insufficient evidence")


@app.websocket("/ws/hf/{call_sid}")
async def websocket_hf(websocket: WebSocket, call_sid: str):
    """Real-time AMD via Hugging Face ASR + heuristics.

    Strategy: accumulate PCM16 audio; every ~1s send short WAV to ASR, classify.
    Stop once a confident decision is made or on end/close.
    """
    await websocket.accept()
    session = CallSession(call_sid)
    active_calls[call_sid] = session

    logger.info(f"[HF] WebSocket opened for call: {call_sid}")
    decision_sent = False
    last_checked_len = 0

    # Proactively warm up local whisper in background to reduce latency
    warmup_task: asyncio.Task | None = None
    if USE_LOCAL_WHISPER_FALLBACK and WhisperModel is not None:

        async def _warmup():
            try:
                global _whisper_model
                if _whisper_model is None:
                    logger.info(
                        f"[HF] Warming up local Whisper model: {LOCAL_WHISPER_MODEL}"
                    )
                    _whisper_model = WhisperModel(
                        LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8"
                    )
            except Exception as e:
                logger.warning(f"[HF] Warmup failed: {e}")

        warmup_task = asyncio.create_task(_warmup())

    async def analyze_loop():
        nonlocal decision_sent, last_checked_len
        try:
            while not decision_sent:
                await asyncio.sleep(1.0)
                if (
                    len(session.buffer_pcm16) - last_checked_len < 16000
                ):  # ~1s at 8kHz*2bytes
                    continue
                last_checked_len = len(session.buffer_pcm16)

                # Use up to first 6s for early decision
                max_bytes = 6 * 8000 * 2
                chunk = bytes(session.buffer_pcm16[:max_bytes])
                wav = pcm16_to_wav(chunk, 8000)
                try:
                    text = await hf_transcribe(wav)
                    logger.info(f"[HF] Transcript ({call_sid}): {text}")
                    # If remote returns empty, immediately try local fallback
                    if not text:
                        raise RuntimeError("empty transcript from remote")
                    decision, conf, reason = classify_transcript(text)
                    if decision != "uncertain" and conf >= 0.75:
                        try:
                            await websocket.send_json(
                                {
                                    "type": "analysis_complete",
                                    "result": decision,
                                    "confidence": conf,
                                    "reason": reason,
                                }
                            )
                        except Exception:
                            pass
                        decision_sent = True
                        break
                except Exception as e:
                    logger.warning(f"[HF] Transcription error: {e}")
                    # Local fallback
                    try:
                        text = await asyncio.to_thread(local_transcribe_wav, wav)
                        if text:
                            decision, conf, reason = classify_transcript(text)
                            if decision != "uncertain" and conf >= 0.75:
                                try:
                                    await websocket.send_json(
                                        {
                                            "type": "analysis_complete",
                                            "result": decision,
                                            "confidence": conf,
                                            "reason": reason,
                                        }
                                    )
                                except Exception:
                                    pass
                                decision_sent = True
                                break
                    except Exception as e2:
                        logger.warning(f"[HF] Local fallback error: {e2}")

        except asyncio.CancelledError:
            pass

    analyze_task = asyncio.create_task(analyze_loop())

    try:
        while not decision_sent:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue
            msg_type = message.get("type")
            if msg_type == "audio":
                audio_b64 = message.get("data")
                if audio_b64:
                    audio_chunk = base64.b64decode(audio_b64)
                    pcm16 = pcmu_to_pcm16(audio_chunk)
                    session.buffer_pcm16.extend(pcm16)
            elif msg_type == "end" or msg_type == "call_ended":
                break

        # Final attempt if no decision yet
        if not decision_sent and session.buffer_pcm16:
            wav = pcm16_to_wav(bytes(session.buffer_pcm16), 8000)
            try:
                text = await hf_transcribe(wav)
                if not text:
                    raise RuntimeError("empty transcript from remote")
                decision, conf, reason = classify_transcript(text)
                # Even if confidence is low, emit a final decision so downstream can close gracefully
                final_decision = (
                    decision
                    if conf >= 0.75
                    else (decision if decision != "uncertain" else "uncertain")
                )
                final_conf = (
                    conf
                    if conf >= 0.75
                    else (conf if decision != "uncertain" else 0.55)
                )
                try:
                    await websocket.send_json(
                        {
                            "type": "analysis_complete",
                            "result": final_decision,
                            "confidence": final_conf,
                            "reason": reason,
                        }
                    )
                except Exception:
                    pass
                decision_sent = True
            except Exception as e:
                logger.error(f"[HF] Final transcription error: {e}")
                # Try local fallback once more
                try:
                    text = await asyncio.to_thread(local_transcribe_wav, wav)
                    if text is not None:
                        decision, conf, reason = classify_transcript(text)
                        # Emit whatever best-effort decision we have
                        final_decision = (
                            decision
                            if conf >= 0.75
                            else (decision if decision != "uncertain" else "uncertain")
                        )
                        final_conf = (
                            conf
                            if conf >= 0.75
                            else (conf if decision != "uncertain" else 0.55)
                        )
                        try:
                            await websocket.send_json(
                                {
                                    "type": "analysis_complete",
                                    "result": final_decision,
                                    "confidence": final_conf,
                                    "reason": reason,
                                }
                            )
                        except Exception:
                            pass
                        decision_sent = True
                        return
                except Exception as e2:
                    logger.warning(f"[HF] Final local fallback error: {e2}")
                try:
                    await websocket.send_json(
                        {
                            "type": "analysis_complete",
                            "result": "uncertain",
                            "confidence": 0.5,
                            "reason": "hf error",
                        }
                    )
                except Exception:
                    pass
                decision_sent = True

    except WebSocketDisconnect:
        logger.info(f"[HF] WebSocket disconnected: {call_sid}")
    except Exception as e:
        logger.error(f"[HF] WebSocket error: {e}", exc_info=True)
    finally:
        if not analyze_task.done():
            analyze_task.cancel()
        if warmup_task and not warmup_task.done():
            warmup_task.cancel()
        active_calls.pop(call_sid, None)
        logger.info(f"[HF] Session cleaned up: {call_sid}")


# Back-compat alias: some callers may still hit /ws/amd/hf/{call_sid}
@app.websocket("/ws/amd/hf/{call_sid}")
async def websocket_hf_alias(websocket: WebSocket, call_sid: str):
    await websocket_hf(websocket, call_sid)
