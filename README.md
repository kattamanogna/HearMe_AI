# HearMe_AI

HearMe_AI is a **multimodal mental health AI chatbot** project scaffold that combines:
- Text emotion + intent understanding
- Audio emotion recognition
- Face emotion recognition
- A fusion engine to make a final empathetic prediction
- FastAPI backend endpoints for serving inference

> This repository is a starter template with function stubs and comments describing what to implement next.

## Project Structure

```text
HearMe_AI/
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes.py
│       ├── core/
│       │   └── config.py
│       ├── main.py
│       └── schemas.py
├── models/
│   ├── text_emotion_intent/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── audio_emotion/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── infer.py
│   └── face_emotion/
│       ├── preprocess.py
│       ├── train.py
│       └── infer.py
├── fusion/
│   └── engine.py
├── utils/
│   ├── io.py
│   └── logger.py
├── scripts/
│   └── train_all.py
├── data/
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Train Models

Run individual training scripts after preparing datasets:

```bash
python models/text_emotion_intent/train.py --train-path data/text/train.jsonl --output-dir artifacts/text_model --epochs 3
python -c "from models.audio_emotion.train import train_audio_emotion_model; train_audio_emotion_model('data/audio', 'artifacts/audio_model')"
python -c "from models.face_emotion.train import train_face_emotion_model; train_face_emotion_model('data/face', 'artifacts/face_model')"
```

Or run the combined orchestrator:

```bash
python scripts/train_all.py
```

## Run Backend

From the repository root:

```bash
uvicorn backend.app.main:app --reload
```

Then test:

```bash
curl http://127.0.0.1:8000/api/v1/health
```


## API Examples

### Text prediction

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text":"I feel hopeful today"}'
```

### Audio prediction (file upload)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-audio \
  -F "file=@/path/to/sample.wav"
```

### Face prediction (file upload)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-face \
  -F "file=@/path/to/face.jpg"
```

### Multimodal analysis (base64 payloads)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text":"I am stressed but hopeful",
    "audio_bytes":"UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
    "face_base64":"/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }'
```

The `/analyze` response now includes `response_text`, a safe empathetic message generated from
the fused emotion and input text.

### Optional HuggingFace response generation

By default, chat responses use deterministic templates per emotion. You can enable lightweight
HuggingFace text generation by installing `transformers` and setting:

```bash
export ENABLE_HF_CHAT_RESPONSE=1
export HF_CHAT_MODEL=sshleifer/tiny-gpt2
```

If model loading fails, the backend automatically falls back to template responses with the same
safety filter.

## Next Implementation Steps

- Replace placeholder model logic in all `train.py` and `infer.py` modules.
- Add real preprocessing pipelines and dataset schemas.
- Add unit tests for API routes and model utility functions.
- Add safety filters and escalation logic for mental-health-critical scenarios.
