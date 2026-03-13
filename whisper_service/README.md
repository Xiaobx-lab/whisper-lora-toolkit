# Whisper LoRA ASR Service

Ray Serve + FastAPI service for the LoRA fine-tuned Whisper-large-v3-turbo model with true batch inference and encoder caching optimization.

## Quick Start

```bash
conda activate asr_trainer

# 1. Edit config (set adapter_path, adjust batch_size, replicas, etc.)
vim whisper_service/config/whisper_config.yaml

# 2. Start service
bash whisper_service/deploy.sh start

# 3. Check status
bash whisper_service/deploy.sh status

# 4. Transcribe a file
python whisper_service/asr_client.py path/to/audio.wav

# 5. Stop service
bash whisper_service/deploy.sh stop
```

## API Endpoints

### `GET /healthz`

Health check.

```bash
curl http://127.0.0.1:8009/healthz
# ok
```

### `POST /asr`

Main ASR endpoint. Accepts PCM16 base64-encoded audio.

**Request:**
```json
{
  "audio_data": "<PCM16 base64 string>",
  "sample_rate": 16000
}
```

**Response:**
```json
{
  "text": "transcribed text here",
  "language": "zh"
}
```

**Error response:**
```json
{
  "error": "audio_data_missing"
}
```

### `POST /infer`

Legacy endpoint (compatible with whipserZoo).

**Request:**
```json
{
  "action": "infer",
  "audio": "<PCM16 base64 string>"
}
```

## Client Usage

### Command line

```bash
# Transcribe a single file
python whisper_service/asr_client.py audio.wav

# Specify service URL
python whisper_service/asr_client.py audio.mp3 --url http://127.0.0.1:8009

# Concurrent stress test (triggers batch inference)
python whisper_service/test_client.py audio.wav --n 20
```

### Python library

```python
from whisper_service.asr_client import transcribe_file, transcribe, health_check

# Check service
assert health_check("http://127.0.0.1:8009")

# Transcribe a file (wav/mp3/flac/ogg/m4a/...)
result = transcribe_file("audio.wav")
print(result["text"])      # "transcribed text"
print(result["language"])  # "zh"

# Or encode audio yourself and call transcribe()
import base64
with open("audio_pcm16.raw", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
result = transcribe(b64, sample_rate=16000)
```

### curl

```bash
# Encode audio to PCM16 base64 (requires ffmpeg + base64)
AUDIO_B64=$(ffmpeg -i audio.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 - 2>/dev/null | base64 -w0)

# Call the API
curl -X POST http://127.0.0.1:8009/asr \
  -H "Content-Type: application/json" \
  -d "{\"audio_data\": \"$AUDIO_B64\", \"sample_rate\": 16000}"
```

## Configuration

Edit `whisper_service/config/whisper_config.yaml`:

```yaml
model:
  base_model_id: "/path/to/whisper-large-v3-turbo"
  adapter_path: "/path/to/lora/adapter/best"   # null = vanilla Whisper
  language: null                                 # null = auto-detect, or "zh"/"en"/...
  task: "transcribe"                             # "transcribe" or "translate"

deployment:
  num_replicas: 1           # Number of Ray Serve replicas
  gpu_per_replica: null     # null = auto (1.0 / num_replicas)

batch:
  max_batch_size: 8         # Max requests per batch
  batch_wait_timeout_s: 0.01

generation:
  max_new_tokens: 440       # Max output tokens
  num_beams: 1              # 1 = greedy decoding
```

## Resource Usage

Measured on NVIDIA H200 with `whisper-large-v3-turbo` + LoRA adapter (bfloat16):

| Component | Memory |
|-----------|--------|
| Model weights (base + LoRA) | ~1.6 GB |
| Inference overhead (batch_size=8) | ~0.8 GB |
| **Total per replica** | **~2.5 GB** |

## File Structure

```
whisper_service/
├── config/
│   └── whisper_config.yaml      # Service configuration
├── utils/
│   ├── __init__.py
│   └── config_loader.py         # YAML config loader
├── whisper_lora_server.py       # Ray Serve + FastAPI server
├── asr_client.py                # Client library & CLI example
├── test_client.py               # Concurrent stress test
├── test_batch_standalone.py     # Standalone batch inference test (no Ray)
├── deploy.sh                    # Service management (start/stop/restart/status)
├── logs/
│   └── service.log              # Runtime logs
└── README.md
```
