# Dataset Registry System - Summary

## What Changed?

I've implemented a LlamaFactory-style dataset registry system that eliminates the need to manually configure dataset formats, columns, and templates every time you switch datasets.

## Key Benefits

✅ **No more manual configuration** - Just use `dataset_name: "alpaca_en"`
✅ **100+ pre-configured datasets** - All from your dataset_info.json
✅ **Easy dataset discovery** - CLI tool to browse and search
✅ **Consistent formatting** - Pre-tested configurations
✅ **Quick switching** - Change one line to try a different dataset
✅ **Backwards compatible** - Old configs still work

## How It Works

### Old Way (Manual)

```yaml
dataset:
  name: "yahma/alpaca-cleaned"
  split: "train"
  instruction_field: "instruction"
  input_field: "input"
  output_field: "output"
  prompt_template: |
    Below is an instruction that describes a task...
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:
    {output}
```

### New Way (Registry)

```yaml
dataset:
  dataset_name: "alpaca_en" # Everything else is automatic!
```

## Quick Start

### 1. Browse Available Datasets

```bash
python tools/dataset_tool.py list
```

### 2. Get Dataset Info

```bash
python tools/dataset_tool.py info alpaca_en
```

### 3. Update Your Config

```yaml
dataset:
  dataset_name: "alpaca_en" # Change this line only!
  max_samples: 10000 # Optional overrides
```

### 4. Train

```bash
python main.py --config configs/default.yaml
```

## Available Datasets

Your `dataset_info.json` includes:

- **52 Instruction datasets** (alpaca, belle, codealpaca, mathinstruct, etc.)
- **15 Conversation datasets** (sharegpt, ultrachat, lmsys_chat, etc.)
- **12 Ranking/DPO datasets** (ultrafeedback, nectar, dpo_mix, etc.)
- **8 Multimodal datasets** (llava, pokemon_cap, etc.)
- **11 Pretraining datasets** (wikipedia, refinedweb, fineweb, etc.)
- **Your custom datasets** (rp, rp2, rpc, nsfw_reddit, etc.)

## CLI Tool Commands

```bash
# List all datasets
python tools/dataset_tool.py list

# Filter by type
python tools/dataset_tool.py list --type sft
python tools/dataset_tool.py list --type ranking
python tools/dataset_tool.py list --type multimodal

# Search datasets
python tools/dataset_tool.py search alpaca

# Get detailed info
python tools/dataset_tool.py info sharegpt4

# Generate config snippet
python tools/dataset_tool.py config alpaca_en

# Add custom dataset
python tools/dataset_tool.py add my_dataset data/my_data.json \
    --format alpaca \
    --columns instruction=question output=answer
```

## Makefile Shortcuts

```bash
make datasets              # List all datasets
make datasets-sft          # List SFT datasets only
make datasets-search QUERY=alpaca  # Search datasets
```

## Example Configurations

### Quick Test (5 minutes)

```yaml
model:
  name: "unsloth/gemma-2b"
  load_in_4bit: true

dataset:
  dataset_name: "alpaca_en_demo"
  max_samples: 1000

training:
  num_train_epochs: 1
```

### Production Quality

```yaml
model:
  name: "unsloth/mistral-7b-v0.3"
  load_in_4bit: true

dataset:
  dataset_name: "alpaca_gpt4_en"

lora:
  r: 32
  lora_alpha: 64

training:
  num_train_epochs: 5
  learning_rate: 1.0e-4
```

### Your Custom Dataset

```yaml
dataset:
  dataset_name: "rp" # From your dataset_info.json
```

## Adding Your Datasets

### Option 1: CLI (Easiest)

```bash
python tools/dataset_tool.py add my_data data/my_file.jsonl \
    --format alpaca \
    --columns instruction=inst output=resp
```

### Option 2: Edit dataset_info.json

```json
{
  "my_dataset": {
    "file_name": "my_data.jsonl",
    "formatting": "alpaca",
    "columns": {
      "instruction": "question",
      "output": "answer"
    }
  }
}
```

### Option 3: HuggingFace Dataset

```json
{
  "my_hf_dataset": {
    "hf_hub_url": "username/dataset-name",
    "formatting": "sharegpt"
  }
}
```

## File Structure

```
unsloth-finetuning/
├── dataset_info.json       # Dataset registry (your file!)
├── configs/
│   ├── default.yaml        # Updated with registry support
│   └── examples/           # Example configs
├── src/
│   ├── dataset_registry.py # Registry implementation
│   ├── data.py            # Updated data manager
│   └── ...
├── tools/
│   └── dataset_tool.py    # CLI tool
├── DATASET_REGISTRY.md    # Full documentation
└── README.md              # Updated with registry info
```

## Migration Guide

### If You Have Existing Configs

**Option 1: Keep using them** - They still work!

**Option 2: Migrate to registry:**

1. Find your dataset: `python tools/dataset_tool.py search <name>`
2. Get config: `python tools/dataset_tool.py config <dataset_name>`
3. Update your config file
4. Remove old format specifications

## Supported Formats

- ✅ Alpaca (instruction-input-output)
- ✅ ShareGPT (multi-turn conversations)
- ✅ Plain text
- ✅ DPO/RLHF (ranking pairs)
- ✅ KTO (preference labels)
- ✅ Multimodal (images, videos, audio)
- ✅ Tool calling

## Example: Switching Datasets

```yaml
# Try English Alpaca
dataset:
  dataset_name: "alpaca_en"

# Switch to Chinese
dataset:
  dataset_name: "alpaca_zh"

# Try conversations
dataset:
  dataset_name: "sharegpt4"

# Try code
dataset:
  dataset_name: "codealpaca"

# Try math
dataset:
  dataset_name: "mathinstruct"

# Try your data
dataset:
  dataset_name: "rp"
```

Just change one line and you're good to go!

## Troubleshooting

### "Dataset not found in registry"

```bash
# List available datasets
python tools/dataset_tool.py list

# Search for similar
python tools/dataset_tool.py search <partial_name>
```

### "Column not found"

```bash
# Check dataset configuration
python tools/dataset_tool.py info <dataset_name>

# Update dataset_info.json if needed
```

### Using Legacy Config

Simply don't use `dataset_name` - use the old format:

```yaml
dataset:
  name: "username/dataset"
  instruction_field: "instruction"
  # etc...
```

## Best Practices

1. ✅ Use `dataset_name` for known datasets
2. ✅ Add your datasets to registry for easy reuse
3. ✅ Test with `max_samples: 1000` first
4. ✅ Use `python tools/dataset_tool.py info` to verify config
5. ✅ Keep dataset_info.json in version control

## Next Steps

1. **Explore datasets**: `python tools/dataset_tool.py list`
2. **Try different datasets**: Just change `dataset_name` in config
3. **Add your datasets**: Use the CLI tool or edit JSON
4. **Share configs**: Dataset registry makes configs portable

## Resources

- `DATASET_REGISTRY.md` - Full documentation
- `configs/examples/` - Example configurations
- `tools/dataset_tool.py` - CLI tool
- Your `dataset_info.json` - All available datasets

---

**The registry system is fully integrated and ready to use!** 🚀

Simply update your config to use `dataset_name` instead of manual configuration, and you're all set.
