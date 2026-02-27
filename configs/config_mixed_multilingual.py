# config_mixed_multilingual.py
# 混合多语种微调配置 (nan 74% + zh 11% + en 8% + ja 6% + yue 1%)

# ===== 路径 =====
MODEL_NAME = "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
TRAIN_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/mixed_train.jsonl"
VALID_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/our_data/valid.jsonl"
OUTPUT_DIR = "whisper-turbo-multilingual-lora"
RUN_NAME = "multilingual_mixed"

# ===== 训练超参 =====
EPOCHS = 10
TRAIN_BS = 8
EVAL_BS = 8
GRAD_ACCUM = 1
LR = 1e-4
WARMUP_STEPS = 500
MAX_LABEL_LEN = 448

EVAL_STEPS = 10000
SAVE_STEPS = 10000
EVAL_SAMPLES = 100
NUM_WORKERS = 4
SEED = 42
CLIP_NORM = 1.0

# ===== 精度/显存 =====
USE_BF16 = True
USE_GRAD_CHECKPOINT = False

# ===== Whisper 行为 =====
FREEZE_ENCODER = False
FORCE_DECODER_PROMPT = False   # 多语种训练，让模型自己学语言检测
FORCE_LANGUAGE = "zh"          # FORCE_DECODER_PROMPT=True 时使用
FORCE_TASK = "transcribe"

# FORCE_DECODER_PROMPT=False 时生效：
# 每条 JSONL 数据需要有 "language" 字段 (如 "zh", "en", "ja", "yue")
# 没有 language 字段的样本会用 DEFAULT_LANGUAGE 兜底
DEFAULT_LANGUAGE = "zh"
# 将不被 Whisper 支持的语言码映射到支持的语言码
LANGUAGE_MAPPING = {"nan": "zh", "minnan": "zh"}

# ===== LoRA =====
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"
