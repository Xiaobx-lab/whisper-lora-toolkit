# config_stage2_nan_token.py
# DEPRECATED EXPERIMENT: retained for reference only.
# Use train.py with configs/config_multilingual_nan_token.py instead.
# Stage 2: Language token embedding calibration on balanced data.
# Loads the stage-1 LoRA adapter, freezes everything, and only trains
# the ~100 language-token embedding rows to fix language detection bias.

# ===== Stage 1 checkpoint =====
STAGE1_ADAPTER_PATH = "whisper-turbo-multilingual-nan-token/latest/best"

# ===== Base model (auto-detected from adapter if omitted) =====
MODEL_NAME = "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

# ===== Data =====
TRAIN_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/mix_stage2_train.jsonl"
VALID_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/benchmark/benchmark_v0_0_1.jsonl"
OUTPUT_DIR = "whisper-turbo-multilingual-nan-token-stage2"
RUN_NAME = "stage2_lang_calibration"

# ===== Training hyperparams =====
EPOCHS = 5
TRAIN_BS = 16
EVAL_BS = 16
GRAD_ACCUM = 1
LR = 1e-5
WARMUP_STEPS = 100
MAX_LABEL_LEN = 448

EVAL_STEPS = 500
SAVE_STEPS = 1000
EVAL_SAMPLES = 100
LANG_EVAL_SAMPLES = 500
NUM_WORKERS = 4
SEED = 42
CLIP_NORM = 1.0

# ===== Precision =====
USE_BF16 = True
USE_GRAD_CHECKPOINT = False

# ===== Whisper behaviour =====
FORCE_DECODER_PROMPT = False
FORCE_LANGUAGE = "zh"
FORCE_TASK = "transcribe"

DEFAULT_LANGUAGE = "zh"
