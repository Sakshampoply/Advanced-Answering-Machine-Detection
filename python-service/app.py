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

from google import genai
from google.genai import types
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


class CallSession:
    """Track state for each call."""

    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.decision = None
        self.confidence = 0.0
        self.audio_chunks = 0
        self.total_ms = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events."""
    logger.info("AMD service starting...")
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
