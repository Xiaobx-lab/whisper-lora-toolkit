# config_multilingual_nan_token.py
# 实验：为闽南话新增独立语言标签 <|nan|>，而非映射到 zh
# 新 token 的 embedding 用 zh + yue 的均值初始化；训练时同时更新
# 新 token 与已有语言 token 的 embedding（见 UNFREEZE_LANG_EMBEDDINGS）

# ===== 路径 =====
MODEL_NAME = "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
TRAIN_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/mixed_train_v2.jsonl"
VALID_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/our_data/valid.jsonl"
OUTPUT_DIR = "whisper-turbo-multilingual-nan-token"
RUN_NAME = "multilingual_nan_token_v2"

# ===== 训练超参 =====
EPOCHS = 10
TRAIN_BS = 16
EVAL_BS = 16
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
FORCE_DECODER_PROMPT = False
FORCE_LANGUAGE = "zh"
FORCE_TASK = "transcribe"

DEFAULT_LANGUAGE = "zh"
# "minnan" 仍映射到 "nan"，但 "nan" 不再映射到 "zh"，而是用新增的 <|nan|> token
LANGUAGE_MAPPING = {"minnan": "nan"}

# ===== 自定义语言 token =====
# 新增 <|nan|> token 到词表，embedding 用 zh + yue 的均值初始化
ADD_CUSTOM_LANG_TOKENS = ["nan"]
CUSTOM_LANG_TOKEN_INIT = {"nan": ["zh", "yue"]}
# 解冻原有 50 个语言 token 的 embedding 行，使其能跟随 LoRA 表征变化一起训练
# 非语言 token 的 embedding 仍然冻结
UNFREEZE_LANG_EMBEDDINGS = True

# ===== LoRA =====
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"
