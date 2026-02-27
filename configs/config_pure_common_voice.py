# config.py
# 所有配置都在这里改（暂时用字符串/常量，避免 argparse）

# ===== 路径 =====
MODEL_NAME = "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
TRAIN_JSONL = "common_voice_nan_tw_data/train.jsonl"
VALID_JSONL = "common_voice_nan_tw_data/valid.jsonl"
OUTPUT_DIR = "whisper-turbo-minnan-zh-lora"
RUN_NAME = "minnan_zh"

# ===== 训练超参 =====
EPOCHS = 10
TRAIN_BS = 2
EVAL_BS = 2
GRAD_ACCUM = 1  # 数据少时设为1，否则梯度不会更新
LR = 1e-4
WARMUP_STEPS = 200
MAX_LABEL_LEN = 448

EVAL_STEPS = 500     # 0 表示不按 steps eval，仅每个 epoch eval
SAVE_STEPS = 500     # 0 表示不按 steps 保存
NUM_WORKERS = 4
SEED = 42
CLIP_NORM = 1.0

# ===== 精度/显存 =====
USE_BF16 = True      # H200 建议 True  
USE_GRAD_CHECKPOINT = False  # LoRA训练参数少，不需要gradient checkpointing

# ===== Whisper 行为 =====
FREEZE_ENCODER = True
FORCE_DECODER_PROMPT = True  # 如果你确定输出一定是“中文 transcribe”，可设 True
FORCE_LANGUAGE = "zh"
FORCE_TASK = "transcribe" # "transcribe"

# ===== LoRA =====
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# HuggingFace推荐的Whisper LoRA target modules
# 包含更多attention层通常效果更好
TARGET_MODULES = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"
# TARGET_MODULES = "q_proj,k_proj"
