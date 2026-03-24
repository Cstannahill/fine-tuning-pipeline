# Configuration Parameters Reference Guide

A comprehensive guide to understanding how each parameter affects training quality, speed, and memory usage.

**Legend:**

- 🎯 **Quality**: How it affects model performance
- ⚡ **Speed**: How it affects training time
- 💾 **Memory**: How it affects VRAM/RAM usage
- 📊 **Recommended Values**: Suggested settings for different scenarios

---

## Model Configuration

### `max_seq_length: 2048`

**What it does:** Maximum number of tokens the model can process at once

🎯 **Quality Impact:**

- **Higher** (4096, 8192): Better for long-context tasks, can learn from longer examples
- **Lower** (512, 1024): May truncate important context, limiting learning
- **Sweet spot**: Match your data's typical length

⚡ **Speed Impact:**

- **Higher**: MUCH slower (quadratic relationship - 2x length = 4x slower)
- **Lower**: Faster training
- **Impact**: 2048 → 4096 can be 3-4x slower per step

💾 **Memory Impact:**

- **Higher**: MUCH more VRAM (quadratic - 2x length ≈ 4x memory)
- **Lower**: Less VRAM required
- **Impact**: 2048 → 4096 might require 3-4x more VRAM

📊 **Recommended:**

```yaml
# Chat/General: 2048
# Long documents: 4096
# Code: 2048-4096
# Short tasks: 512-1024
```

---

## LoRA Configuration

### `r: 16` (LoRA Rank)

**What it does:** Number of trainable parameters in LoRA layers. Higher = more capacity.

🎯 **Quality Impact:**

- **r=4-8**: Minimal adaptation, good for small tweaks
- **r=16-32**: Good balance, works for most tasks
- **r=64-128**: Maximum adaptation, best quality for complex tasks
- **Rule**: Higher rank = better quality, but diminishing returns after 32-64

⚡ **Speed Impact:**

- **Higher**: Slightly slower (10-20% slower at r=64 vs r=8)
- **Lower**: Marginally faster
- **Impact**: Not the bottleneck usually

💾 **Memory Impact:**

- **Higher**: More VRAM (but still efficient compared to full fine-tuning)
- **Lower**: Less VRAM
- **Scale**: r=8 uses ~50% memory of r=16

📊 **Recommended:**

```yaml
# Quick experiments: r=8
# Standard training: r=16-32
# Maximum quality: r=64-128
# Complex tasks (coding, reasoning): r=32-64
```

### `lora_alpha: 16`

**What it does:** Scaling factor for LoRA weights. Often set equal to rank.

🎯 **Quality Impact:**

- **alpha = r**: Standard, balanced learning
- **alpha > r** (e.g., 32): Stronger LoRA influence, more aggressive adaptation
- **alpha < r** (e.g., 8): Weaker influence, more conservative
- **Formula**: Effective learning rate = `(alpha / r) * learning_rate`

⚡ **Speed Impact:** None

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Standard: lora_alpha = r (16)
# Aggressive learning: lora_alpha = 2*r (32)
# Conservative: lora_alpha = r/2 (8)
```

### `lora_dropout: 0.0`

**What it does:** Randomly drops LoRA weights during training for regularization

🎯 **Quality Impact:**

- **0.0**: No regularization, might overfit on small datasets
- **0.05-0.1**: Good regularization, prevents overfitting
- **>0.1**: Too much regularization, might underfit
- **Use when**: Dataset < 10k examples

⚡ **Speed Impact:**

- **0.0**: Fastest
- **>0**: Slightly slower (5-10% slower)

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Large dataset (>10k): 0.0
# Medium dataset (1k-10k): 0.05
# Small dataset (<1k): 0.1
# Overfitting observed: 0.05-0.1
```

### `target_modules`

**What it does:** Which model layers get LoRA adapters

🎯 **Quality Impact:**

- **More modules**: Better adaptation, higher quality
- **Attention only** (q,k,v,o): Good for most tasks
- **+ FFN** (gate,up,down): Better for complex reasoning
- **All modules**: Maximum quality

⚡ **Speed Impact:**

- **More modules**: Slightly slower (linear relationship)
- **Attention only**: Fastest
- **Impact**: +FFN adds ~20% training time

💾 **Memory Impact:**

- **More modules**: More VRAM (proportional to module count)
- **Impact**: +FFN adds ~30% memory usage

📊 **Recommended:**

```yaml
# Fast/Standard:
- "q_proj"
- "k_proj"
- "v_proj"
- "o_proj"

# Best quality (recommended):
- "q_proj"
- "k_proj"
- "v_proj"
- "o_proj"
- "gate_proj"
- "up_proj"
- "down_proj"
```

### `use_gradient_checkpointing: true`

**What it does:** Trades computation for memory by recomputing activations

🎯 **Quality Impact:** None (exactly the same results)

⚡ **Speed Impact:**

- **true**: 20-30% slower
- **false**: Faster but needs more VRAM

💾 **Memory Impact:**

- **true**: 30-50% less VRAM required
- **false**: More VRAM needed
- **Critical**: Often the difference between fitting or not

📊 **Recommended:**

```yaml
# Limited VRAM (<16GB): true
# Plenty of VRAM (>24GB): false
# Training crashes OOM: true
```

### `use_rslora: false` (Rank-Stabilized LoRA)

**What it does:** Normalizes LoRA weights for better stability

🎯 **Quality Impact:**

- **false**: Standard behavior
- **true**: More stable training, can use higher learning rates
- **Better for**: High rank (r>32) or unstable training

⚡ **Speed Impact:** Negligible

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Standard (r≤32): false
# High rank (r>32): true
# Training unstable: true
```

---

## Training Configuration

### `num_train_epochs: 1`

**What it does:** How many times to go through the entire dataset

🎯 **Quality Impact:**

- **1 epoch**: Minimal learning, underfitting likely
- **3-5 epochs**: Standard, good balance
- **>5 epochs**: Risk of overfitting on small datasets
- **Rule**: More data = fewer epochs needed

⚡ **Speed Impact:**

- **Linear**: 3 epochs = 3x longer than 1 epoch

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Large dataset (>50k): 1-2 epochs
# Medium dataset (10k-50k): 3 epochs
# Small dataset (1k-10k): 5-10 epochs
# Tiny dataset (<1k): 10-20 epochs
```

### `per_device_train_batch_size: 2`

**What it does:** Number of examples processed together per GPU

🎯 **Quality Impact:**

- **Larger** (8, 16): Smoother gradients, more stable training
- **Smaller** (1, 2): Noisier gradients, can help escape local minima
- **Sweet spot**: As large as memory allows

⚡ **Speed Impact:**

- **Larger**: MUCH faster (better GPU utilization)
- **Smaller**: Slower
- **Scale**: Batch 8 is ~3-4x faster than batch 2

💾 **Memory Impact:**

- **Larger**: More VRAM (linear relationship)
- **Smaller**: Less VRAM
- **Critical**: Primary VRAM control

📊 **Recommended:**

```yaml
# Start high, reduce if OOM:
# 12GB VRAM: 2-4
# 16GB VRAM: 4-8
# 24GB VRAM: 8-16
# 48GB VRAM: 16-32
```

### `gradient_accumulation_steps: 4`

**What it does:** Simulates larger batch size by accumulating gradients

🎯 **Quality Impact:**

- **Effective batch = batch_size × accumulation_steps**
- **Higher**: Similar to larger batch (stabler training)
- **Sweet spot**: Aim for effective batch of 16-32

⚡ **Speed Impact:**

- **Higher**: Slower per step (no parallelization benefit)
- **Trade-off**: Updates less frequently
- **Note**: 2 steps of batch 4 ≈ same time as 1 step of batch 8

💾 **Memory Impact:** Minimal (only stores gradients)

📊 **Recommended:**

```yaml
# Calculate effective batch:
# per_device_train_batch_size × gradient_accumulation_steps

# Target effective batch 16-32:
# VRAM limited (batch=2): accumulation=8-16
# VRAM okay (batch=4): accumulation=4-8
# VRAM plenty (batch=8): accumulation=2-4
```

### `learning_rate: 2.0e-4`

**What it does:** How much to update weights each step

🎯 **Quality Impact:**

- **Too high** (>5e-4): Unstable, diverges, poor quality
- **Too low** (<1e-5): Slow learning, underfitting
- **Sweet spot**: 1e-4 to 3e-4 for most cases
- **Critical**: Single most important hyperparameter

⚡ **Speed Impact:**

- **Higher**: Learns faster (fewer epochs needed)
- **Lower**: Learns slower (more epochs needed)
- **Note**: Must balance with quality

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Conservative/Safe: 1.0e-4
# Standard (recommended): 2.0e-4
# Aggressive: 3.0e-4 to 5.0e-4
# Large models (>7B): 1.0e-4
# Small models (<3B): 3.0e-4

# With RSLoRA: Can use 2-3x higher
# Signs it's too high: Loss spikes, NaN, instability
# Signs it's too low: Loss barely decreases
```

### `weight_decay: 0.01`

**What it does:** L2 regularization to prevent overfitting

🎯 **Quality Impact:**

- **0.0**: No regularization, may overfit
- **0.01**: Standard, good default
- **0.1**: Strong regularization, may underfit
- **Use when**: Overfitting observed

⚡ **Speed Impact:** Negligible

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Large dataset: 0.0-0.01
# Medium dataset: 0.01
# Small dataset: 0.01-0.05
# Overfitting: 0.05-0.1
```

### `warmup_ratio: 0.03`

**What it does:** Gradually increases learning rate at start (3% of steps)

🎯 **Quality Impact:**

- **0.0**: No warmup, may be unstable at start
- **0.03-0.1**: Smooth start, more stable
- **>0.1**: Slow start, may waste time
- **Benefit**: Prevents early instability

⚡ **Speed Impact:**

- **Higher**: Slower initial learning
- **Lower**: Faster but less stable

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Standard: 0.03 (3% of steps)
# Short training (<1000 steps): 0.05-0.1
# Long training (>10k steps): 0.01-0.03
# Unstable starts: 0.05-0.1
```

### `lr_scheduler_type: "linear"`

**What it does:** How learning rate changes during training

🎯 **Quality Impact:**

- **"linear"**: Gradual decrease, good for most cases
- **"cosine"**: Smooth decrease, often better quality
- **"constant"**: No change, simpler but may overfit
- **"cosine"** often wins for quality

⚡ **Speed Impact:** None

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Best quality: "cosine"
# Standard: "linear"
# Experimentation: "constant"
# Short training: "constant"
```

### `optim: "adamw_8bit"`

**What it does:** Optimizer type and precision

🎯 **Quality Impact:**

- **"adamw_8bit"**: Slightly lower quality than full precision
- **"adamw_torch"**: Full precision, best quality
- **"paged_adamw_8bit"**: Same as 8bit but better memory
- **Difference**: Usually negligible (<1%)

⚡ **Speed Impact:**

- **"adamw_8bit"**: Fastest
- **"adamw_torch"**: Slightly slower
- **"paged_adamw_8bit"**: Similar to 8bit

💾 **Memory Impact:**

- **"adamw_8bit"**: 50% less VRAM than full precision
- **"adamw_torch"**: Uses most VRAM
- **"paged_adamw_8bit"**: Best for limited VRAM

📊 **Recommended:**

```yaml
# Limited VRAM: "paged_adamw_8bit"
# Standard: "adamw_8bit"
# Maximum quality: "adamw_torch"
```

### `fp16: false / bf16: false`

**What it does:** Precision for training computations

🎯 **Quality Impact:**

- **fp16**: Can have numerical instability
- **bf16**: Better stability than fp16, near fp32 quality
- **fp32** (both false): Best quality, most stable
- **bf16** requires Ampere+ GPUs (RTX 30xx+)

⚡ **Speed Impact:**

- **bf16/fp16**: 2x faster on modern GPUs
- **fp32**: Slower but more compatible

💾 **Memory Impact:**

- **bf16/fp16**: 50% less VRAM
- **fp32**: More VRAM

📊 **Recommended:**

```yaml
# RTX 30xx/40xx (Ampere+): bf16=true
# Older GPUs: fp16=true
# Stability issues: Both false (fp32)
# Auto-detect: Let system choose (current default)
```

### `max_grad_norm: 1.0`

**What it does:** Clips gradients to prevent explosion

🎯 **Quality Impact:**

- **1.0**: Standard, prevents instability
- **0.5**: More conservative, very stable
- **>1.0**: Less clipping, might be unstable
- **Critical**: Prevents training crashes

⚡ **Speed Impact:** Negligible

💾 **Memory Impact:** None

📊 **Recommended:**

```yaml
# Standard: 1.0
# Unstable training: 0.5
# Large models: 0.5-1.0
# Small models: 1.0-2.0
```

### `group_by_length: true`

**What it does:** Groups similar-length examples together

🎯 **Quality Impact:** None (same training)

⚡ **Speed Impact:**

- **true**: 10-30% faster (less padding waste)
- **false**: Slower but simpler

💾 **Memory Impact:**

- **true**: More efficient VRAM usage
- **false**: More padding = wasted VRAM

📊 **Recommended:**

```yaml
# Always: true (unless debugging)
```

### `dataloader_num_workers: 4`

**What it does:** Parallel threads for data loading

🎯 **Quality Impact:** None

⚡ **Speed Impact:**

- **0**: Slowest (single-threaded)
- **2-4**: Good balance
- **>4**: Diminishing returns, may waste CPU
- **Note**: Only helps if data loading is bottleneck

💾 **Memory Impact:**

- **Higher**: More RAM (small increase)

📊 **Recommended:**

```yaml
# Standard: 4
# CPU limited: 2
# Fast SSD: 8
# Slow storage: 2
```

### `save_steps: 100`

**What it does:** How often to save checkpoints

🎯 **Quality Impact:** None (just saves more often)

⚡ **Speed Impact:**

- **Lower**: Slower (more frequent saves)
- **Higher**: Faster (saves less often)
- **Impact**: Usually negligible unless very low

💾 **Memory Impact:**

- **Lower**: More disk space (more checkpoints)
- **save_total_limit**: Controls total checkpoints kept

📊 **Recommended:**

```yaml
# Short training (<1000 steps): 50-100
# Medium training: 100-500
# Long training (>10k steps): 500-1000
```

---

## Quick Reference Tables

### 🎯 Quality Priority (Speed/Memory secondary)

```yaml
model:
  max_seq_length: 4096 # Match your data

lora:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.0
  target_modules: [all 7 modules]
  use_gradient_checkpointing: true # If needed for memory
  use_rslora: true

training:
  num_train_epochs: 5
  per_device_train_batch_size: 1 # Increase if possible
  gradient_accumulation_steps: 32 # Effective batch = 32
  learning_rate: 1.0e-4 # Conservative
  lr_scheduler_type: "cosine"
  optim: "adamw_torch" # Full precision
  fp16: false
  bf16: false # Or true if available
```

### ⚡ Speed Priority (Quality acceptable)

```yaml
model:
  max_seq_length: 1024 # Or minimum needed

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.0
  target_modules: [q, k, v, o only] # Skip FFN
  use_gradient_checkpointing: false # If VRAM allows
  use_rslora: false

training:
  num_train_epochs: 1
  per_device_train_batch_size: 16 # Maximum possible
  gradient_accumulation_steps: 1
  learning_rate: 3.0e-4 # Faster learning
  lr_scheduler_type: "constant"
  optim: "adamw_8bit"
  bf16: true # If Ampere+
  group_by_length: true
```

### 💾 Memory Constrained (<12GB VRAM)

```yaml
model:
  max_seq_length: 1024

lora:
  r: 8
  lora_alpha: 8
  target_modules: [q, k, v, o only]
  use_gradient_checkpointing: true # Critical!

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  optim: "paged_adamw_8bit"
  bf16: true # or fp16
```

### 🎯 Balanced (Recommended Starting Point)

```yaml
model:
  max_seq_length: 2048

lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.0
  target_modules: [all 7]
  use_gradient_checkpointing: true
  use_rslora: false

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4 # Effective = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"
  bf16: true
```

---

## Common Scenarios

### "Training is too slow"

1. **Reduce `max_seq_length`** (biggest impact)
2. **Increase `per_device_train_batch_size`**
3. **Reduce `num_train_epochs`**
4. **Disable `use_gradient_checkpointing`** (if VRAM allows)
5. **Reduce `target_modules`** (remove FFN layers)
6. **Enable `bf16` or `fp16`**
7. **Reduce `r` (LoRA rank)**

### "Running out of memory"

1. **Enable `use_gradient_checkpointing`** (biggest impact)
2. **Reduce `per_device_train_batch_size`**
3. **Reduce `max_seq_length`**
4. **Use `optim: "paged_adamw_8bit"`**
5. **Enable `bf16` or `fp16`**
6. **Reduce `r` (LoRA rank)**
7. **Reduce `target_modules`**

### "Model not learning well"

1. **Increase `num_train_epochs`**
2. **Increase `learning_rate`** (try 3e-4 to 5e-4)
3. **Increase `r` (LoRA rank)** to 32 or 64
4. **Add FFN to `target_modules`**
5. **Increase effective batch size** (batch × accumulation)
6. **Try `lr_scheduler_type: "cosine"`**
7. **Enable `use_rslora`** if rank > 32

### "Training is unstable (loss spikes, NaN)"

1. **Reduce `learning_rate`** (try 1e-4)
2. **Reduce `max_grad_norm`** to 0.5
3. **Increase `warmup_ratio`** to 0.1
4. **Disable fp16** (use fp32 or bf16)
5. **Increase `gradient_accumulation_steps`**
6. **Enable `use_rslora`**

---

## Pro Tips

1. **Start with the balanced config** above and adjust based on your needs
2. **Monitor GPU utilization** - if < 80%, can increase batch size
3. **Watch for overfitting** - if train loss << eval loss, reduce epochs or add regularization
4. **Effective batch size** should be 16-32 for most tasks
5. **Learning rate** is the most important hyperparameter - tune this first
6. **Use `bf16`** if you have Ampere+ GPU (RTX 30xx/40xx)
7. **Gradient checkpointing** is free quality - only costs speed
8. **LoRA rank** has diminishing returns above 64
