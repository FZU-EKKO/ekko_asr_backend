ekko_asr_service

Current interfaces:

- `POST /asr/transcribe`
  - one-shot HTTP transcription
- `WS /asr/stream`
  - streaming ASR interface for `ekko`

Streaming message flow:

1. `start_session`
```json
{
  "type": "start_session",
  "session_id": 1,
  "user_id": "1234567",
  "sample_rate": 16000,
  "channels": 1,
  "sample_width": 2,
  "language": "zh"
}
```

2. `audio_chunk`
```json
{
  "type": "audio_chunk",
  "audio_base64": "..."
}
```

3. `end_stream`
```json
{
  "type": "end_stream"
}
```

Streaming result events:

- `session_started`
- `speech_start`
- `partial_result`
- `final_result`
- `stream_closed`
- `error`

Internal pipeline:

- per-user VAD state
- utterance task scheduler
- batch-oriented inference worker
- final/partial events pushed back to `ekko`

Run:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

If `ekko` sets `EKKO_ASR_REMOTE_TOKEN`, set the same value to `EKKO_ASR_SERVICE_TOKEN` here.
