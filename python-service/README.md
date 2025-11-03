# Gemini Flash AMD Python Service

This is a FastAPI service that uses Google's Gemini 2.5 Flash API to detect if audio contains a human voice or an automated/machine response.

## Setup

1. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file and add your Gemini API key:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Running Locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

## API Endpoints

### POST /predict

Sends an audio file for analysis.

**Request:**

- Form data with `audio` file (WAV, MP3, or OGG)

**Response:**

```json
{
  "label": "human" | "machine",
  "confidence": 0.85,
  "reason": "Human voice with natural greeting",
  "audio_type": "greeting"
}
```

### GET /health

Health check endpoint.

## Docker

Build the image:

```bash
docker build -t amd-python-service .
```

Run the container:

```bash
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key amd-python-service
```

## Testing with curl

```bash
curl -X POST -F "audio=@test.wav" http://localhost:8000/predict
```
