# myWhisperTrainer

基于 Hugging Face Transformers 与 PEFT 的 Whisper LoRA 训练、评测和推理工具集。项目主要面向中文、粤语、闽南语及多语种语音识别，支持自定义语言 token、两阶段训练、CER 评测，以及基于 Ray Serve 的批量推理服务。

## 功能概览

- 使用 LoRA 微调 Whisper，降低训练显存与参数量开销
- 支持冻结或解冻 Whisper encoder
- 支持按样本指定语言的多语种训练
- 支持新增自定义语言 token，例如闽南语 `<|nan|>`
- 支持只校准语言 token embedding 的第二阶段训练
- 自动记录训练日志、TensorBoard 指标、配置快照和最佳 checkpoint
- 提供 CER、多语种语言检测和模型对比评测工具
- 提供 Ray Serve + FastAPI 批量推理服务

## 项目结构

```text
myWhisperTrainer/
├── train.py                  # 第一阶段 LoRA 训练入口
├── train_stage2.py           # 第二阶段语言 token embedding 校准
├── asrDataLoader.py          # JSONL 数据集与 Whisper collator
├── custom_lang.py            # 自定义语言 token 与 embedding 处理
├── whisper_peft.py           # Whisper 的 PEFT/LoRA 适配
├── configs/                  # 不同实验的训练配置
├── benchmark/                # CER 与语言检测评测
├── data_processing/          # 数据下载、转换、混合和校验脚本
├── scripts/                  # 数据统计及模型对比工具
├── whisper_service/          # Ray Serve 推理服务与客户端
├── run_nohup.sh              # 后台运行第一阶段训练
└── run_stage2.sh             # 后台运行第二阶段训练
```

模型权重、训练语料、日志、checkpoint 和评测输出默认不会提交到 Git。使用前需自行准备对应资源。

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU 与匹配的 CUDA 环境（推荐）
- FFmpeg（部分音频处理或命令行示例需要）

安装训练依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如需使用 TensorBoard：

```bash
pip install tensorboard
```

如需运行推理服务，再安装：

```bash
pip install "ray[serve]" fastapi pyyaml requests
```

> PyTorch 和 torchaudio 应根据本机 CUDA 版本安装。具体安装命令请参考 PyTorch 官方说明。

## 数据格式

训练集和验证集使用 JSONL 格式，每行至少包含 `audio` 与 `text`：

```json
{"audio": "/path/to/audio.wav", "text": "对应的转写文本"}
```

多语种训练可增加 `language` 字段：

```json
{"audio": "/path/to/audio.wav", "text": "对应的转写文本", "language": "zh"}
```

音频会在加载时转换为 16 kHz 单声道。`language` 可在配置中的 `LANGUAGE_MAPPING` 里进一步映射。

## 配置训练任务

训练参数集中在 `configs/` 下的 Python 配置文件中。建议复制一个最接近需求的配置再修改：

```bash
cp configs/config.py configs/config_local.py
```

至少需要确认以下项目：

```python
MODEL_NAME = "/path/to/whisper-model"
TRAIN_JSONL = "/path/to/train.jsonl"
VALID_JSONL = "/path/to/valid.jsonl"
OUTPUT_DIR = "outputs/my-run"
RUN_NAME = "my_run"
```

仓库中的部分配置保留了原实验环境的绝对路径，无法直接在其他机器使用，请先替换为本机路径。

常用训练参数包括：

- `TRAIN_BS`、`EVAL_BS`：训练和验证 batch size
- `GRAD_ACCUM`：梯度累积步数
- `LR`、`WARMUP_STEPS`、`EPOCHS`：优化器与训练周期
- `USE_BF16`：是否使用 BF16
- `FREEZE_ENCODER`：是否冻结 encoder
- `LORA_R`、`LORA_ALPHA`、`LORA_DROPOUT`：LoRA 参数
- `FORCE_DECODER_PROMPT`：是否强制指定语言和任务
- `ADD_CUSTOM_LANG_TOKENS`：需要新增的语言 token

## 第一阶段：LoRA 训练

前台运行：

```bash
python train.py --config configs/config_local.py
```

限制验证样本数量：

```bash
python train.py \
  --config configs/config_local.py \
  --eval-samples 200
```

`--eval-samples 0` 表示使用完整验证集。

每次训练会在 `OUTPUT_DIR` 下创建带时间戳的独立目录，并保存：

- 实际使用的配置快照
- TensorBoard 日志
- 定期 checkpoint
- 验证集 CER 最优的 `best/` adapter
- 指向最近一次训练的 `latest` 符号链接

查看 TensorBoard：

```bash
tensorboard --logdir outputs/my-run
```

## 自定义语言 token

项目支持向 Whisper 词表增加语言 token。以下示例新增闽南语 `<|nan|>`，并以中文和粤语 token embedding 的均值初始化：

```python
LANGUAGE_MAPPING = {"minnan": "nan"}
ADD_CUSTOM_LANG_TOKENS = ["nan"]
CUSTOM_LANG_TOKEN_INIT = {"nan": ["zh", "yue"]}
UNFREEZE_LANG_EMBEDDINGS = True
```

可参考 [`configs/config_multilingual_nan_token.py`](configs/config_multilingual_nan_token.py)。

## 第二阶段：语言 embedding 校准

第二阶段会加载第一阶段生成的 LoRA adapter，冻结 encoder、decoder 和 LoRA 权重，只训练语言 token 的 embedding。它适用于修正多语种数据不均衡导致的语言检测偏差。

先编辑 `configs/config_stage2_nan_token.py`，确认第一阶段 adapter、数据集及输出路径，然后运行：

```bash
python train_stage2.py --config configs/config_stage2_nan_token.py
```

## 模型评测

使用 LoRA adapter 运行多语种 benchmark：

```bash
python benchmark/run_benchmark_lora.py \
  --adapter_path /path/to/adapter/best \
  --benchmark_path /path/to/benchmark.jsonl \
  --output_dir benchmark/results_local
```

对于需要强制语言的模型，可指定语言与任务：

```bash
python benchmark/run_benchmark_lora.py \
  --adapter_path /path/to/adapter/best \
  --benchmark_path /path/to/benchmark.jsonl \
  --output_dir benchmark/results_local \
  --force_language zh \
  --force_task transcribe
```

评测会输出逐条转写结果、整体 CER、按语言统计和语言检测结果。

## 推理服务

`whisper_service/` 提供基于 Ray Serve + FastAPI 的推理服务，支持：

- LoRA adapter 与自定义语言 token
- 并发请求的真实 batch 推理
- encoder 输出复用
- 自动语言检测
- PCM16 Base64 HTTP 接口

快速启动：

```bash
# 先修改模型与 adapter 路径
vim whisper_service/config/whisper_config.yaml

bash whisper_service/deploy.sh start
bash whisper_service/deploy.sh status
python whisper_service/asr_client.py /path/to/audio.wav
bash whisper_service/deploy.sh stop
```

完整配置和 API 文档见 [`whisper_service/README.md`](whisper_service/README.md)。

## 后台训练脚本

`run_nohup.sh` 和 `run_stage2.sh` 中包含原实验环境的绝对路径。使用前需修改 `BASE_DIR` 和 Conda 初始化路径。

```bash
./run_nohup.sh start configs/config_local.py
./run_nohup.sh status

./run_stage2.sh start configs/config_stage2_nan_token.py
./run_stage2.sh status
```

## 注意事项

- 不要把模型权重、原始音频、训练数据、密钥或运行日志提交到 Git。
- 数据集许可证与隐私要求需由使用者自行确认。
- 当前训练配置主要针对 GPU 环境；CPU 可用于小规模调试，但不适合实际训练。
- 推理和评测前，请确保基础模型与 adapter 来自相互兼容的训练版本。
