# Tools & Utilities Documentation

This directory contains standalone utility scripts for managing datasets and models in the fine-tuning pipeline.

## Overview

| Tool | Purpose | Status |
|------|---------|--------|
| `dataset_tool.py` | Browse, search, and inspect the dataset registry | ✅ Core |
| `dataset_inspector.py` | **NEW** – Inspect local datasets and auto-register | ✨ New |
| `merge_adapter.py` | **NEW** – Merge LoRA adapters into standalone models | ✨ New |

---

## 📚 Dataset Tool (`dataset_tool.py`)

Browse and manage the pre-configured dataset registry (100+ datasets from `dataset_info.json`).

### Usage

```bash
# List all available datasets
python tools/dataset_tool.py list

# Show detailed information about a dataset
python tools/dataset_tool.py info alpaca_en

# Generate YAML config snippet for a dataset
python tools/dataset_tool.py config alpaca_en

# Search datasets by keyword
python tools/dataset_tool.py search "chat"

# Filter by dataset type
python tools/dataset_tool.py list --type sft         # Supervised fine-tuning
python tools/dataset_tool.py list --type ranking     # Preference learning (DPO, etc)
```

### Dataset Registry Contents

The registry includes datasets in several categories:

- **General Instruction Following** – alpaca, wikipedia, wikitext
- **Chat & Conversation** – ShareGPT, OpenAssistant, SlimOrcaDedicated
- **Code** – CodeAlpaca, Evol-Code-v2
- **Reasoning** – MATH, GSM8K, reasoning datasets
- **Multimodal** – LLaVA, MLLM datasets with images/videos
- **Domain-Specific** – Medical, legal, scientific datasets
- **Preference Learning** – DPO, KTO, RLHF preference pairs

### Example Output

```
Available Datasets (142 total)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ Name                   ┃ Source              ┃ Format    ┃ Type  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ alpaca_en              │ HuggingFace         │ alpaca    │ SFT   │
│ alpaca_zh              │ HuggingFace         │ alpaca    │ SFT   │
│ bluemoon_sharegpt      │ Local               │ sharegpt  │ SFT   │
│ claude-opus-reasoning  │ HuggingFace         │ sharegpt  │ SFT   │
│ gpt-5-2-reasoning      │ HuggingFace         │ sharegpt  │ SFT   │
│ dpo_en_demo            │ Local               │ alpaca    │ DPO   │
│ kto_en_demo            │ Local               │ alpaca    │ KTO   │
│ mllm_demo              │ Local               │ alpaca    │ SFT   │
└────────────────────────┴─────────────────────┴───────────┴───────┘
```

---

## 🔍 Dataset Inspector (`dataset_inspector.py`) ⭐ NEW

Automatically inspect local dataset files (JSON, JSONL, CSV, TXT, Parquet) and register them for use in training.

### Purpose

- **Identify dataset format** – Automatically detects Alpaca, ShareGPT, or plain text formats
- **Preview data** – Shows sample records for verification
- **Infer configuration** – Generates registry entry with correct column mapping
- **Register datasets** – Adds custom datasets to `dataset_info.json` for easy reuse

### Usage

#### Basic Inspection

Display information about a local dataset:

```bash
python tools/dataset_inspector.py data/my_dataset.jsonl
```

**Output shows:**
- File path and type
- Total number of records
- Sample records (first 5 by default)
- Auto-detected format (alpaca, sharegpt, text)
- Inferred column configuration

#### Register Dataset

Add the dataset to `dataset_info.json`:

```bash
python tools/dataset_inspector.py data/my_dataset.jsonl --register --name my_dataset
```

The dataset becomes available immediately in training configs:

```yaml
dataset:
  dataset_name: "my_dataset"
```

#### Advanced Options

```bash
# Adjust sample size for inspection
python tools/dataset_inspector.py data/large_dataset.jsonl --sample-size 10

# Use custom registry file location
python tools/dataset_inspector.py data/my_dataset.jsonl \
  --registry custom_dataset_info.json \
  --register \
  --name my_dataset

# Overwrite existing dataset entry
python tools/dataset_inspector.py data/my_dataset.jsonl \
  --register \
  --name existing_dataset \
  --overwrite

# Inspect without registering
python tools/dataset_inspector.py data/my_dataset.jsonl --sample-size 20
```

### Supported Formats

| Format | Extension | Example |
|--------|-----------|---------|
| **JSON Array** | `.json` | `[{...}, {...}]` |
| **JSON Lines** | `.jsonl` | One JSON object per line |
| **CSV** | `.csv` | With header row |
| **Plain Text** | `.txt` | One record per line |
| **Parquet** | `.parquet` | Arrow format |

### Auto-Detection

The tool automatically detects and configures:

#### Alpaca Format
Expects: `instruction`, `input`, `output` fields

```json
{
  "instruction": "What is 2+2?",
  "input": "",
  "output": "4"
}
```

Generates config:
```yaml
formatting: alpaca
columns:
  instruction: instruction
  input: input
  output: output
```

#### ShareGPT Format
Expects: `conversations` or `messages` field with role/content pairs

```json
{
  "conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there"}
  ]
}
```

Generates config:
```yaml
formatting: sharegpt
columns:
  messages: conversations
tags:
  message_key: conversations
  role_key: from
  content_key: value
```

#### Text Format
Falls back to simple text field

```json
{"text": "Some training text"}
```

Or plain text file with one line per record.

### Example Workflow

```bash
# Step 1: Prepare your dataset
# (You have: data/my_custom_data.jsonl with 50,000 conversations)

# Step 2: Inspect format
python tools/dataset_inspector.py data/my_custom_data.jsonl --sample-size 3

# Output shows:
# - Format: ShareGPT detected ✓
# - Total records: 50,000
# - Sample records displayed
# - Inferred columns and tags

# Step 3: Register if format looks good
python tools/dataset_inspector.py data/my_custom_data.jsonl \
  --register \
  --name my_conversations

# Step 4: Use in training
# In configs/default.yaml:
# dataset:
#   dataset_name: "my_conversations"

# Step 5: Start training!
python main.py --config configs/default.yaml
```

---

## 🔗 Adapter Merger (`merge_adapter.py`) ⭐ NEW

Convert trained LoRA adapters into standalone, deployment-ready models by merging them with the base model.

### Purpose

- **Production Deployment** – Merge adapters into complete models
- **Model Sharing** – Create single-file models for distribution
- **Inference Optimization** – Merged models have better inference performance
- **Compatibility** – Standard HuggingFace format (no PEFT required for inference)

### Usage

#### Basic Merge

Merge base model with adapter:

```bash
python tools/merge_adapter.py \
  meta-llama/llama-2-7b \
  outputs/20260322_141603/adapter_model \
  outputs/merged_model
```

#### Merge from Local Checkpoint

```bash
python tools/merge_adapter.py \
  ./base_models/mistral-7b-v0.3 \
  ./outputs/20260322_141603/checkpoints/checkpoint-400 \
  ./outputs/merged_model
```

#### Advanced Options

```bash
# Specify output dtype (bfloat16 for A100/H100)
python tools/merge_adapter.py \
  meta-llama/llama-2-7b \
  outputs/adapter_model \
  outputs/merged_model \
  --dtype bfloat16

# Load on CPU to save GPU memory (slower)
python tools/merge_adapter.py \
  meta-llama/llama-2-7b \
  outputs/adapter_model \
  outputs/merged_model \
  --device-map cpu

# Allow custom model implementations
python tools/merge_adapter.py \
  custom-architecture-model \
  outputs/adapter_model \
  outputs/merged_model \
  --trust-remote-code

# Combine options
python tools/merge_adapter.py \
  meta-llama/llama-2-7b \
  outputs/adapter_model \
  outputs/merged_model \
  --dtype float32 \
  --device-map cpu \
  --trust-remote-code
```

### Arguments

```
positional arguments:
  base_model_path           Path or HuggingFace model ID for base model
  adapter_path              Path to LoRA adapter directory
  output_path               Directory to save merged model

options:
  --dtype {auto,bfloat16,float16,float32}
                            Torch dtype while loading base model
                            Default: auto (infers from model config)
  --device-map DEVICE_MAP   Transformers device_map value
                            Default: auto
                            Options: auto, cpu, cuda, cpu,0, cuda:0, etc
  --trust-remote-code       Allow custom model code from repository
```

### Output Structure

The merged model directory contains all necessary files for deployment:

```
outputs/merged_model/
├── config.json                      # Model configuration
├── model.safetensors               # Model weights (safe format)
│   (or pytorch_model.bin for other formats)
├── tokenizer.model                 # Tokenizer weights
├── tokenizer.json                  # Tokenizer config
├── tokenizer_config.json           # More tokenizer config
├── special_tokens_map.json         # Special tokens
├── generation_config.json          # Generation defaults
├── README.md                       # Hub README (if present)
└── preprocessor_config.json        # For multimodal models
```

### Model Compatibility

The merger automatically handles:

- **Causal LM** – Mistral, Llama, Qwen, Gemma, Phi, Falcon, etc.
- **Vision-Language** – Models with image inputs (LLaVA, etc.)
- **Seq2Seq** – Models with encoder-decoder (T5, BART, etc.)
- **Processors** – Special processors for multimodal models

The appropriate model loader is auto-detected based on model architecture.

### Example Workflow

```bash
# Step 1: Train a model
python main.py --config configs/default.yaml

# Step 2: Check training outputs
ls outputs/20260322_141603/
# → adapter_model/, checkpoints/, config.yaml, etc.

# Step 3: Merge the adapter
python tools/merge_adapter.py \
  unsloth/mistral-7b-v0.3 \
  outputs/20260322_141603/adapter_model \
  outputs/20260322_141603/merged_model

# Step 4: Test the merged model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('outputs/20260322_141603/merged_model')
tokenizer = AutoTokenizer.from_pretrained('outputs/20260322_141603/merged_model')
print('✓ Merged model loads successfully!')
"

# Step 5: Push to HuggingFace Hub
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('outputs/20260322_141603/merged_model')
tokenizer = AutoTokenizer.from_pretrained('outputs/20260322_141603/merged_model')
model.push_to_hub('username/my-finetuned-model')
tokenizer.push_to_hub('username/my-finetuned-model')
"

# Done! Your model is now deployable as a standalone unit.
```

### Performance Notes

- **Merging Time** – Usually 5-15 minutes depending on model size
- **Disk Space** – Merged model = adapter + base model (full model size)
- **GPU Memory** – Required for efficient merging (can use --device-map cpu as fallback)
- **Inference Speed** – Merged models are ~10-15% faster than using adapters

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA OOM during merge | Use `--device-map cpu` (slower but works) |
| Model loading fails | Add `--trust-remote-code` for custom models |
| Tokenizer not found | Ensure adapter dir has `special_tokens_map.json` |
| Wrong model loader | Auto-detection handles 99% of cases; if not, check model config |

---

## 💡 Tips & Tricks

### Working with Large Datasets

```bash
# For 100M+ records, sample before full processing
python tools/dataset_inspector.py huge_dataset.jsonl \
  --sample-size 5 \
  --register \
  --name huge_dataset_sampled
```

### Combining Multiple Datasets

Use the registry to create a "composite" entry, or merge files before registration:

```bash
# Merge multiple JSONL files
cat data/file1.jsonl data/file2.jsonl data/file3.jsonl > data/combined.jsonl

# Then inspect and register
python tools/dataset_inspector.py data/combined.jsonl \
  --register \
  --name combined_dataset
```

### Debugging Model Issues

```bash
# Test merge without GPU overhead
python tools/merge_adapter.py \
  model_id \
  adapter_path \
  output_path \
  --device-map "cpu" \
  --dtype float32
```

---

## 📋 Tool Dependencies

All tools require only standard packages from `requirements.txt`:

- `transformers` – Model loading and utilities
- `peft` – LoRA adapter handling (merge_adapter)
- `datasets` – For parquet support (dataset_inspector)
- `rich` – Beautiful CLI output

No additional installation needed beyond the main setup.

---

## 🔗 Integration with Main Pipeline

### Using Registered Datasets

```yaml
# In configs/default.yaml
dataset:
  dataset_name: "alpaca_en"  # From dataset_tool.py
  # or
  dataset_name: "my_custom"  # From dataset_inspector.py
```

### Using Merged Models

```yaml
# For inference in main pipeline --eval-only
model:
  name: "outputs/merged_model"
  load_in_4bit: false  # Merged models don't need quantization
```

---

