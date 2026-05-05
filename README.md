ekko_asr_service

Minimal one-shot ASR service for `ekko` voice messages.

Interfaces:

- `GET /health`
- `POST /asr/transcribe`

Run:

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 19001 --reload
```

Prerequisites:

- Python 3.9+
- System `ffmpeg` available in `PATH`

This service uses `faster-whisper` for local one-shot ASR.

Environment:

```env
EKKO_ASR_SERVICE_TOKEN=
EKKO_ASR_MODEL_SIZE=large-v3
EKKO_ASR_DEVICE=auto
EKKO_ASR_COMPUTE_TYPE=auto
EKKO_ASR_DEFAULT_LANGUAGE=zh
EKKO_ASR_BEAM_SIZE=5
EKKO_ASR_VAD_FILTER=true
```
