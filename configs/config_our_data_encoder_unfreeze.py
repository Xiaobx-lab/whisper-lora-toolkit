# config_our_data_encoder_unfreeze.py
# 实验：解冻 encoder，对比 FREEZE_ENCODER=True 的效果

# ===== 路径 =====
MODEL_NAME = "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
TRAIN_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/our_data/train.jsonl"
VALID_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/our_data/valid.jsonl"
OUTPUT_DIR = "whisper-turbo-minnan-zh-lora-encoder-unfreeze"
RUN_NAME = "minnan_zh_encoder_unfreeze"

# ===== 训练超参 =====
EPOCHS = 5           # 先跑少一点观察
TRAIN_BS = 8
EVAL_BS = 8
GRAD_ACCUM = 1
LR = 1e-4
WARMUP_STEPS = 500   # 稍微多一点 warmup
MAX_LABEL_LEN = 448

EVAL_STEPS = 5000    # 降低 eval 频率
SAVE_STEPS = 10000   # 降低保存频率
EVAL_SAMPLES = 200   # validation 采样数量
NUM_WORKERS = 4
SEED = 42
CLIP_NORM = 1.0

# ===== 精度/显存 =====
USE_BF16 = True
USE_GRAD_CHECKPOINT = False

# ===== Whisper 行为 =====
FREEZE_ENCODER = False  # 解冻 encoder，让 encoder 的 LoRA 和原始权重都训练
FORCE_DECODER_PROMPT = True
FORCE_LANGUAGE = "zh"
FORCE_TASK = "transcribe"

# ===== LoRA =====
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"
