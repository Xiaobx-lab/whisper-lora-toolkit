# Whisper LoRA ASR Service

Ray Serve + FastAPI service for the LoRA fine-tuned Whisper-large-v3-turbo model with true batch inference and encoder caching optimization.

## Architecture

```
Client (HTTP)
    |
    v
Ray Serve Proxy (:8009)
    |
    v
@serve.batch (groups N concurrent requests)
    |
    v
WhisperLoRAASR._batched_infer()
    |
    +-- Encoder (runs ONCE per batch, output cached)
    |
    +-- Language Detection (reuses encoder output, no redundant encoder pass)
    |
    +-- Decoder / model.generate() (reuses encoder output, true batch decoding)
    |
    v
JSON response per request
```

Key optimizations vs the reference project (whipserZoo):
- **Encoder caching**: encoder runs once per batch; the output is reused for both language detection and decoding. The reference project runs the encoder twice (once for language detection, once for transcription).
- **True batch inference**: concurrent requests are grouped by `@serve.batch` into a single `model.generate()` call.
- **LoRA adapter support**: loads fine-tuned adapters via PEFT with custom language token support (e.g. `<|nan|>` for Minnan).

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

- `audio_data` (required): mono PCM16 audio, base64-encoded.
- `sample_rate` (optional, default 16000): sample rate of the input audio. If not 16000, the server will **auto-resample** to 16kHz via torchaudio. Supported rates: 8000, 22050, 44100, 48000, etc.

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

Possible error values: `audio_data_missing`, `decode_or_resample_error`, `service_not_ready`, `invalid_request`.

### `POST /infer`

Legacy endpoint (compatible with whipserZoo).

**Request:**
```json
{
  "action": "infer",
  "audio": "<PCM16 base64 string>"
}
```

Response format is the same as `/asr`. Sample rate is fixed at 16000.

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
  adapter_path: "/path/to/lora/adapter/best"   # null = vanilla Whisper (no adapter)
  local_files_only: true
  language: null                                 # null = auto-detect, or "zh"/"en"/...
  task: "transcribe"                             # "transcribe" or "translate"

deployment:
  num_replicas: 1           # Number of Ray Serve replicas
  gpu_per_replica: null     # null = auto (1.0 / num_replicas)
  max_ongoing_requests: 100

batch:
  max_batch_size: 8         # Max requests per batch
  batch_wait_timeout_s: 0.01  # Max wait to form a batch (seconds)

generation:
  max_new_tokens: 440       # Max output tokens (leave room for decoder prefix)
  num_beams: 1              # 1 = greedy decoding

server:
  port: 8009
```

## Performance

Measured on NVIDIA H200, `whisper-large-v3-turbo` + LoRA adapter, bfloat16.

### Latency (single request, GPU exclusive)

| Stage | Time |
|-------|------|
| Encoder forward | ~50ms |
| Language detection | ~10ms |
| Decoder generate (avg) | ~200ms |
| **Total inference** | **~260ms** |
| + HTTP overhead | ~310ms |

### Throughput (batch inference)

| Mode | Per request | Throughput |
|------|------------|------------|
| Sequential (no batching) | ~310ms | ~3 req/s |
| 8 concurrent (batch=8) | ~145ms | ~7 req/s |
| 10 concurrent (batch=10) | ~70ms | ~14 req/s |

### Resource Usage

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
├── asr_client.py                # Client library & CLI tool
├── test_client.py               # Concurrent stress test
├── test_batch_standalone.py     # Standalone batch inference test (no Ray)
├── deploy.sh                    # Service management (start/stop/restart/status)
├── logs/
│   └── service.log              # Runtime logs
└── README.md
```
