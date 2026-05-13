ekko_asr_service

Minimal queued ASR service for `ekko` voice messages.

Interfaces:

- `POST /asr/transcribe/queue`

Run:

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 19001 --reload
```

Prerequisites:

- Python 3.9+
- System `ffmpeg` available in `PATH`

This service stores uploaded audio as temporary local files, queues only metadata, runs `faster-whisper`, and pushes results back to `ekko` by callback.

Environment:

```env
EKKO_ASR_SERVICE_TOKEN=
EKKO_ASR_MODEL_PATH=/data/models/faster-whisper/large-v3
EKKO_ASR_DEVICE=auto
EKKO_ASR_COMPUTE_TYPE=auto
EKKO_ASR_DEFAULT_LANGUAGE=zh
EKKO_ASR_BEAM_SIZE=5
EKKO_ASR_VAD_FILTER=true
EKKO_ASR_QUEUE_MAX_SIZE=0
EKKO_ASR_CALLBACK_TIMEOUT_SECONDS=30
EKKO_ASR_TEMP_AUDIO_DIR=./tmp_audio
```
