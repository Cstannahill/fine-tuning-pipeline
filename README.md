# Unsloth Fine-tuning Pipeline

A production-grade, modular pipeline for fine-tuning Large Language Models using LoRA and QLoRA with Unsloth. Designed for efficiency, ease of use, and extensibility.

## 🌟 Features

- **LoRA & QLoRA Support** – Parameter-efficient fine-tuning with automatic quantization handling
- **Config-Driven Architecture** – YAML-based configuration with sensible defaults, no scripting needed
- **Rich CLI Experience** – Beautiful terminal UI with real-time progress, ETA, and training metrics
- **Modular Design** – Clean separation of concerns for easy extension and maintenance
- **Dataset Registry** – Pre-configured dataset library inspired by LlamaFactory (100+ datasets)
- **Comprehensive Evaluation** – Built-in metrics: perplexity, BLEU, ROUGE, and custom evaluators
- **Multiple Output Formats** – Save as adapters, merged models, or quantized variants
- **HuggingFace Integration** – Direct Hub integration for model pushing and dataset loading
- **Multi-Format Dataset Support** – JSON, JSONL, CSV, TXT, and Parquet datasets
- **Automatic Dataset Configuration** – Inspect and register new datasets with automatic format detection

---

## 📁 Project Structure

```
fine-tuning/
├── main.py                     # Main training entry point
├── setup.sh                    # Setup script for WSL/Linux
├── requirements.txt            # Production dependencies
├── dataset_info.json          # Dataset registry with 100+ pre-configured datasets
├── Dockerfile                 # Container configuration
│
├── configs/
│   ├── default.yaml           # Default training configuration
│   └── ...
│
├── src/
│   ├── config.py              # Configuration manager
│   ├── data.py                # Dataset loading and preprocessing
│   ├── model.py               # Model initialization with LoRA/QLoRA
│   ├── trainer.py             # Training orchestration (transforms, HF Trainer)
│   ├── evaluator.py           # Evaluation metrics and callbacks
│   ├── merger.py              # Adapter merging utilities
│   ├── dataset_registry.py    # Dataset registry system
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── progress.py        # Rich progress tracking
│
├── tools/
│   ├── dataset_tool.py        # Browse and manage dataset registry
│   ├── dataset_inspector.py   # Inspect and register local datasets
│   ├── merge_adapter.py       # Standalone adapter merging utility
│   └── README.md              # Tools documentation
│
├── docs/
│   ├── DATASET_REGISTRY.md    # Dataset registry guide
│   ├── TRAINING_CONFIG.md     # Configuration reference
│   └── ...
│
├── data/                       # Sample datasets
│   ├── alpaca_en_demo.json
│   ├── alpaca_zh_demo.json
│   ├── bluemoon_sharegpt.jsonl
│   └── ...
│
├── outputs/                    # Training outputs (auto-created)
├── logs/                       # Training logs (auto-created)
└── unsloth_compiled_cache/    # Unsloth compilation cache
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** (16GB+ VRAM recommended for QLoRA, 24GB+ for LoRA)
- **CUDA 11.8+** (if using GPU)
- **50GB+ free disk space** (for models and datasets)

### Installation

#### Option 1: WSL/Linux (Recommended)

```bash
# Clone and navigate to project directory
git clone https://github.com/Cstannahill/fine-tuning-pipeline.git
cd fine-tuning

# Create virtual environment
python -m venv .venv

# Or using uv
uv venv .venv --python python3.12 --seed

# Activate environment
source venv/bin/activate
```

#### Option 2: Windows (Native)

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Docker

```bash
# Build image
docker build -t fine-tuning .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/app fine-tuning bash
```

### Basic Training in 3 Steps

#### 1. Verify Setup

```bash
# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# List available datasets
python tools/dataset_tool.py list
```

#### 2. Configure Training

Edit `configs/default.yaml`:

```yaml
model:
  name: "microsoft/phi-2" # Or any HuggingFace model
  load_in_4bit: true # Use 4-bit quantization (QLoRA)

dataset:
  dataset_name: "alpaca_en" # Example: Alpaca English dataset
  max_samples: null # Use all samples

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
  output_dir: "./outputs"
```

#### 3. Start Training

```bash
python main.py --config configs/default.yaml
```

Monitor training with real-time progress, loss curves, and estimated completion time.

---

## 🛠️ Tools & Utilities

### Dataset Management

#### 1. Dataset Registry Browser (`dataset_tool.py`)

Explore and manage datasets from the pre-configured registry:

```bash
# List all available datasets
python tools/dataset_tool.py list

# Show detailed info about a dataset
python tools/dataset_tool.py info alpaca_en

# Generate config snippet for a dataset
python tools/dataset_tool.py config alpaca_en

# Search datasets
python tools/dataset_tool.py search "chat"

# Filter by type
python tools/dataset_tool.py list --type sft
python tools/dataset_tool.py list --type ranking
```

**Supported Dataset Types:**

- **SFT** – Supervised fine-tuning (instruction, input, output)
- **Ranking** – Preference learning (DPO, KTO, RLHF)
- **Chat** – Conversation datasets (ShareGPT format)
- **Multimodal** – Vision and audio support

#### 2. Dataset Inspector (`dataset_inspector.py`) ⭐ NEW

Inspect local dataset files and automatically register them in the registry:

```bash
# Inspect a local dataset (JSON, JSONL, CSV, TXT, Parquet)
python tools/dataset_inspector.py data/my_dataset.jsonl

# Preview sample records and inferred configuration
# Output shows:
#   - File type and record count
#   - Auto-detected format (alpaca, sharegpt, text)
#   - Sample records with formatting
#   - Inferred column mapping

# Register the dataset in dataset_info.json
python tools/dataset_inspector.py data/my_dataset.jsonl --register --name "my_custom_dataset"

# Overwrite existing registry entry
python tools/dataset_inspector.py data/my_dataset.jsonl --register --name "my_dataset" --overwrite

# Adjust sample size for inspection
python tools/dataset_inspector.py data/large_dataset.jsonl --sample-size 10
```

**Supported Formats:**

- JSON arrays and objects
- JSONL (one record per line)
- CSV with headers
- TXT (one record per line)
- Parquet files

**Auto-Detection:**

- Alpaca format (instruction, input, output)
- ShareGPT format (messages array)
- Simple text format (single text field)

### Model Management

#### 3. Adapter Merger (`merge_adapter.py`) ⭐ NEW

Merge trained LoRA adapters with base models into standalone, deployable models:

```bash
# Basic merge
python tools/merge_adapter.py \
  microsoft/phi-2 \
  outputs/20260322_141603/checkpoints/checkpoint-400 \
  outputs/merged_model

# With custom dtype
python tools/merge_adapter.py \
  microsoft/phi-2 \
  outputs/adapter_model \
  outputs/merged_model \
  --dtype bfloat16

# With custom device mapping
python tools/merge_adapter.py \
  microsoft/phi-2 \
  outputs/adapter_model \
  outputs/merged_model \
  --device-map cpu

# Allow custom model code
python tools/merge_adapter.py \
  your-org/custom-model \
  outputs/adapter_model \
  outputs/merged_model \
  --trust-remote-code
```

**Options:**

- `--dtype` – `auto` (default), `float32`, `float16`, `bfloat16`
- `--device-map` – `auto` (default), `cpu`, `cuda`, or specific mapping
- `--trust-remote-code` – Allow custom model implementations from Hub

**Output:**

- Merged model with all weights integrated
- Tokenizer and processor (if applicable)
- Sidecar files (config, READMEs, etc.)

---

## 📊 Training Modes

### Full Training Pipeline

```bash
python main.py --config configs/default.yaml
```

Executes: Data load → Model init → Training → Evaluation → Merging

### Evaluation Only

```bash
python main.py --config configs/default.yaml --eval-only
```

Evaluate an existing checkpoint or adapter without training.

### Merging Only

```bash
python main.py --config configs/default.yaml --merge-only
```

Merge trained adapters without re-training (useful for post-processing).

### Training with Resume

```bash
python main.py --config configs/default.yaml --resume-from-checkpoint outputs/20260322_141603/checkpoints/checkpoint-400
```

Resume training from a specific checkpoint.

---

## 📁 Configuration Guide

### Model Selection

```yaml
model:
  name: "microsoft/phi-2" # HuggingFace model ID (example)
  max_seq_length: 2048 # Context window
  load_in_4bit: true # 4-bit quantization (QLoRA)

  # LoRA configuration
  lora:
    r: 16 # LoRA rank
    lora_alpha: 32 # LoRA alpha (scaling)
    lora_dropout: 0.05 # Dropout for LoRA layers
    bias: "none" # Bias handling: none, all, lora_only
    target_modules: ["q_proj", "v_proj"] # Modules to apply LoRA to
    task_type: "CAUSAL_LM" # Task type for PEFT
```

### Dataset Configuration

**Using Registry (Recommended):**

```yaml
dataset:
  dataset_name: "alpaca_en" # From dataset_info.json
  max_samples: 1000 # Optional: limit samples
```

**Using HuggingFace Dataset:**

```yaml
dataset:
  name: "yahma/alpaca-cleaned"
  split: "train"
  subset: null # For multi-split datasets
```

**Using Local Files:**

```yaml
dataset:
  name: "local_dataset"
  data_path: "data/my_dataset.jsonl" # Local file path
  split: "train"
```

See [`docs/DATASET_REGISTRY.md`](docs/DATASET_REGISTRY.md) for full configuration options.

### Training Hyperparameters

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler_type: "linear"
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Checkpointing
  save_steps: 100
  eval_steps: 100
  save_total_limit: 3
  resume_from_checkpoint: null

  # Mixed precision
  fp16: false
  bf16: true # Recommended for A100/H100
```

See [`docs/TRAINING_CONFIG.md`](docs/TRAINING_CONFIG.md) for comprehensive reference.

---

## 📊 Output Structure

After training completes:

```
outputs/20260322_141603/
├── config.yaml                  # Saved training configuration
├── training.log                 # Detailed training logs
│
├── checkpoints/
│   ├── checkpoint-100/          # Intermediate checkpoints
│   ├── checkpoint-200/
│   └── checkpoint-400/
│
├── adapter_model/               # Final LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── merged_model/                # Merged model (if enabled)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.model
│   └── generation_config.json
│
└── evaluation_results.json      # Metrics (perplexity, BLEU, ROUGE)
```

---

## 🔮 Using Trained Models

### Using the Adapter (Memory Efficient)

```python
from unsloth import FastLanguageModel

# Load the fine-tuned adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/YYYYMMDD_HHMMSS/adapter_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Prepare for inference
FastLanguageModel.for_inference(model)

# Generate text
inputs = tokenizer(
    "Your prompt here",
    return_tensors="pt",
    truncation=True
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using the Merged Model (Standard)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model
model = AutoModelForCausalLM.from_pretrained(
    "outputs/YYYYMMDD_HHMMSS/merged_model",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/YYYYMMDD_HHMMSS/merged_model"
)

# Generate text
inputs = tokenizer("Your prompt", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Push to HuggingFace Hub

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/20260322_141603/adapter_model",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Save and push merged model
model.push_to_hub_merged(
    "username/my-model",
    tokenizer,
    private=False,
    token="hf_...",
)
```

---

## 📈 Monitoring & Evaluation

### Real-Time Metrics

The CLI displays:

- Training/eval loss (batch-level)
- Learning rate schedule
- GPU memory usage
- Tokens per second (throughput)
- Estimated time remaining

### Evaluation Metrics

After training, `evaluation_results.json` contains:

```json
{
  "perplexity": 8.543,
  "bleu": 0.342,
  "rouge1": 0.456,
  "rouge2": 0.312,
  "rougeL": 0.398
}
```

### Custom Evaluators

Extend `src/evaluator.py` to add custom metrics:

```python
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, predictions, references):
        # Your custom metric logic
        return {"custom_metric": score}
```

---

## 🔧 Advanced Usage

### Multi-GPU Training

The pipeline automatically detects and uses all available GPUs:

```bash
# Uses all GPUs automatically
python main.py --config configs/default.yaml

# Or explicitly with environment variables
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config configs/default.yaml
```

### Custom Datasets

1. **Prepare your data** (JSON/JSONL/CSV/TXT/Parquet format)
2. **Inspect with dataset_inspector.py:**
   ```bash
   python tools/dataset_inspector.py data/my_data.jsonl --sample-size 10
   ```
3. **Register in dataset_info.json:**
   ```bash
   python tools/dataset_inspector.py data/my_data.jsonl --register --name my_dataset
   ```
4. **Use in training config:**
   ```yaml
   dataset:
     dataset_name: "my_dataset"
   ```

### Custom Models

Supported model architectures:

- Llama (Meta)
- Mistral (Mistral AI)
- Qwen (Alibaba)
- Gemma (Google)
- Phi (Microsoft)
- Falcon (TII)
- And any HuggingFace causal LM

### Distributed Training (DDP)

```bash
torchrun --nproc_per_node=4 main.py --config configs/default.yaml
```

---

## 📚 Documentation

- [`docs/DATASET_REGISTRY.md`](docs/DATASET_REGISTRY.md) – Complete dataset registry guide
- [`docs/TRAINING_CONFIG.md`](docs/TRAINING_CONFIG.md) – All configuration options
- [`tools/README.md`](tools/README.md) – Tool-specific documentation
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) – 5-minute getting started guide

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

1. Reduce `per_device_train_batch_size` (start with 1)
2. Enable `gradient_accumulation_steps`
3. Use `load_in_4bit: true` for QLoRA
4. Reduce `max_seq_length`

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

### CUDA/GPU Issues

```bash
# Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# Check memory
nvidia-smi

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Dataset Format Issues

Use `dataset_inspector.py` to diagnose:

```bash
python tools/dataset_inspector.py your_file.jsonl --sample-size 20
```

Shows detected format, sample records, and inferred configuration.

### Model Loading Errors

- **`TrustRemoteCode` error** – Add `--trust-remote-code` flag to merge_adapter.py
- **Quantization issues** – Verify GPU supports the quantization method (requires NVIDIA H100/A100)
- **Memory mapping** – Use `--device-map cpu` for loading on CPU

---

## 📋 System Requirements

| Component  | Minimum | Recommended |
| ---------- | ------- | ----------- |
| GPU Memory | 12GB    | 24GB+       |
| RAM        | 16GB    | 32GB+       |
| Storage    | 50GB    | 100GB+      |
| CUDA       | 11.8    | 12.0+       |
| Python     | 3.8     | 3.10+       |

### Tested Combinations

- ✅ A100, H100 (80GB) – Full fine-tuning with large batch sizes
- ✅ RTX 4090 (24GB) – QLoRA with batch size 2-4
- ✅ RTX 3090 (24GB) – QLoRA with batch size 1-2
- ✅ T4 (16GB) – QLoRA with batch size 1, gradient accumulation

---

## 📦 Release Notes

### v1.0.0 (Current)

**New Features:**

- ✨ Dataset Inspector tool for automatic dataset registration
- ✨ Standalone Adapter Merger utility
- 🎯 Improved CLI with better error handling
- 📊 Expanded evaluation metrics
- 🔧 Better configuration validation

**Improvements:**

- Better progress tracking
- Faster data loading with caching
- Improved memory management
- Enhanced logging

For detailed changelog, see [releases](https://github.com/example/releases).

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Issues** – GitHub Issues for bug reports and features
- **Discussions** – GitHub Discussions for questions
- **Documentation** – See `docs/` directory for detailed guides

---

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) – Fast LoRA fine-tuning
- [HuggingFace Transformers](https://huggingface.co/transformers/) – Model libraries
- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) – Dataset registry inspiration
- [PEFT](https://github.com/huggingface/peft) – Parameter-efficient fine-tuning

---

**Happy fine-tuning! 🚀**
