# 🚀 Quick Start Guide

Get up and running with fine-tuning in **5 minutes**.

---

## ⚡ Installation (2 min)

### On WSL/Linux (Recommended)

```bash
# Clone or navigate to project
cd fine-tuning

# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv + installs deps)
./setup.sh

# Activate environment
source venv/bin/activate
```

### On Windows

```bash
python -m venv venv
venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🎯 First Training (3 min)

### 1. Verify GPU Setup

```bash
python -c "import torch; print('GPU Ready:', torch.cuda.is_available())"
```

### 2. Start Training

```bash
# Train with default config (example model on Alpaca dataset)
python main.py --config configs/default.yaml
```

Done! Watch the real-time progress as training starts.

### 3. Check Results

```bash
# Your trained model is in:
ls outputs/YYYYMMDD_HHMMSS/
# → adapter_model/        (LoRA adapter, ~50MB)
# → config.yaml          (training config)
# → checkpoints/         (intermediate saves)
```

---

## 🛠️ Common Next Steps

### Want to Use a Different Dataset?

```bash
# 1. List available datasets
python tools/dataset_tool.py list

# 2. Edit config/default.yaml
dataset:
  dataset_name: "alpaca_zh"  # or any from the list

# 3. Retrain
python main.py --config configs/default.yaml
```

### Want to Use Your Own Data?

```bash
# 1. Inspect your dataset (JSON/JSONL/CSV/TXT/Parquet)
python tools/dataset_inspector.py data/my_data.jsonl

# 2. Register it
python tools/dataset_inspector.py data/my_data.jsonl \
  --register --name my_dataset

# 3. Use in training
# Edit configs/default.yaml:
dataset:
  dataset_name: "my_dataset"

# 4. Train!
python main.py --config configs/default.yaml
```

### Want to Merge Your Model for Deployment?

```bash
# Create a standalone, deployable model
python tools/merge_adapter.py \
  microsoft/phi-2 \
  outputs/YYYYMMDD_HHMMSS/adapter_model \
  outputs/YYYYMMDD_HHMMSS/merged_model

# Now use it anywhere:
# model = AutoModelForCausalLM.from_pretrained("outputs/.../merged_model")
```

---

## 📝 Configuration Templates

Choose based on your GPU and time constraints.

### Quick Test (< 20 min)

```yaml
model:
  name: "microsoft/phi-2" # Small, fast model for testing
  load_in_4bit: true

dataset:
  dataset_name: "alpaca_en"
  max_samples: 100 # Small sample for quick testing

training:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
```

### Standard Training (2-4 hours)

```yaml
model:
  name: "microsoft/phi-2" # Or any suitable model
  load_in_4bit: true

dataset:
  dataset_name: "alpaca_en"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
```

### High Quality (8+ hours)

```yaml
model:
  name: "microsoft/phi-2"
  load_in_4bit: true

lora:
  r: 32
  lora_alpha: 64

training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  learning_rate: 1.0e-4
  gradient_accumulation_steps: 4
```

---

## 🧪 Test Your Trained Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/YYYYMMDD_HHMMSS/adapter_model",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🐛 Quick Troubleshooting

| Problem                | Solution                                                               |
| ---------------------- | ---------------------------------------------------------------------- |
| **CUDA Out of Memory** | Reduce `per_device_train_batch_size` to 1, or use `load_in_4bit: true` |
| **GPU not found**      | Run: `nvidia-smi` to verify GPU drivers                                |
| **Dataset not found**  | Ensure file path is correct, or use `dataset_tool.py list`             |
| **Slow training**      | Reduce `max_seq_length` or use smaller model                           |
| **Model won't merge**  | Add `--trust-remote-code` flag to merge_adapter.py                     |

---

## 📚 For More Details

- **Full Guide** → [`README.md`](README.md)
- **Tools Documentation** → [`tools/README.md`](tools/README.md)
- **Configuration Reference** → [`docs/TRAINING_CONFIG.md`](docs/TRAINING_CONFIG.md)
- **Dataset Registry** → [`docs/DATASET_REGISTRY.md`](docs/DATASET_REGISTRY.md)
- **Contributing** → [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 🎉 You're Ready!

That's it! You now have:

✅ Environment set up  
✅ First model training  
✅ Tools for managing datasets and models

**Next:** Customize your config and train your own models!

---

**Happy fine-tuning! 🚀**
