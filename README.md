# GPT-2

Exploring GPT-2 model architecture and training with PyTorch and Hugging Face Transformers.

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/realgio95/gpt-2.git
cd gpt-2
```

### 2. Create and activate virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install ipykernel  # For Jupyter notebook support
```

### 4. Download training data

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -OutFile "input.txt"
```

**Linux/macOS:**
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## 📁 Project Structure

```
gpt-2/
├── play.ipynb        # Interactive exploration of GPT-2 weights
├── train_gpt2.py     # Training script (WIP)
├── requirements.txt  # Python dependencies
└── README.md
```

## 📓 Notebooks

### play.ipynb

Explore GPT-2's pretrained weights:
- Load GPT-2 (124M) from Hugging Face
- Visualize positional embeddings as heatmaps
- Analyze embedding dimensions across positions

## 🛠️ Requirements

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- Matplotlib

## � Ubuntu/WSL Setup (with GPU support)

For faster training with `torch.compile()` and Triton, use Ubuntu via WSL:

### 1. Install WSL and Ubuntu
```powershell
wsl --install -d Ubuntu
```
Restart your computer, then launch Ubuntu from Start menu and create a username/password.

### 2. Update Ubuntu packages
```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Install Python and pip
```bash
sudo apt install python3 python3-pip python3-venv -y
```

### 4. Install CUDA toolkit for WSL
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4 -y
```

### 5. Access the repo from WSL
```bash
cd /mnt/c/repos/gpt-2
```

### 6. Create virtual environment
```bash
python3 -m venv .venv-linux
source .venv-linux/bin/activate
```

### 7. Install PyTorch with CUDA
```bash
pip install torchvision torch --index-url https://download.pytorch.org/whl/cu124
```

### 8. Install dependencies
```bash
pip install tiktoken transformers triton
```

### 9. Verify GPU access
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 🎯 Complete Training Workflow

Follow these steps in order to train GPT-2 from scratch and evaluate results:

### Step 1: Download FineWeb-Edu Dataset
```bash
# In WSL/Linux terminal
cd /mnt/c/repos/gpt-2
source .venv-linux/bin/activate
pip install datasets tqdm
python fineweb.py
```
⏱️ This takes several hours. Continue to Step 2 while waiting.

### While FineWeb Downloads...

While the dataset downloads, you can prepare and test the training pipeline:

1. **Review `train_gpt2.py`** - Familiarize yourself with hyperparameters (lines ~497-527)
2. **Open `play.ipynb`** - Run the exploration cells (Cells 1-25) to understand GPT-2 architecture
3. **Test with Shakespeare** - If FineWeb isn't ready, training falls back to `input.txt` automatically
4. **Prepare HellaSwag** - The evaluation runs automatically every 250 steps during training

Once FineWeb finishes downloading, you're ready for the full training run.

### Step 2: Run Training
```bash
# Single GPU
python train_gpt2.py

# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```
Training logs are written to `log/log.txt`. Checkpoints saved every 5000 steps.

### Step 3: Monitor HellaSwag Progress
The training script automatically evaluates HellaSwag every 250 steps. Watch for output like:
```
step  250 | hellaswag accuracy: 0.2567
step  500 | hellaswag accuracy: 0.2834
...
```

### Step 4: Visualize Results
After training completes, open `play.ipynb` and run Cell 26:
```python
# This cell parses log/log.txt and plots:
# - Loss curves (train/val) vs GPT-2 baseline
# - HellaSwag accuracy vs GPT-2/GPT-3 baselines
```

### Step 5: Generate Text with Trained Model
Run the final cells in `train_gpt2.py` or use the notebook to generate text:
```python
# Example output after training:
# > Hello, I'm a language model, and I can help you understand...
```

### Quick Reference Commands
```bash
# Activate environment (WSL)
cd /mnt/c/repos/gpt-2 && source .venv-linux/bin/activate

# Check training progress
tail -f log/log.txt

# View last HellaSwag result
grep "hella" log/log.txt | tail -5

# Resume from checkpoint (manual - modify train_gpt2.py)
# Load: checkpoint = torch.load("log/model_05000.pt")
```

## 📦 Downloading FineWeb-Edu Dataset

The FineWeb-Edu dataset is a high-quality 10B token educational web corpus from HuggingFace. To download and tokenize it:

### 1. Install dataset dependencies
```bash
pip install datasets tqdm
```

### 2. Run the download script
```bash
python fineweb.py
```

This will:
- Download the `sample-10BT` split from HuggingFace (~15-20GB download)
- Tokenize all documents using GPT-2's BPE tokenizer
- Save ~100 shards of 100M tokens each to `edu_fineweb10B/`
- First shard becomes validation, rest are training

**Note:** This takes several hours depending on your internet and CPU speed. The script uses multiprocessing for faster tokenization.

### Output files:
```
edu_fineweb10B/
├── edufineweb_val_000000.npy    # Validation shard (~200MB)
├── edufineweb_train_000001.npy  # Training shard 1
├── edufineweb_train_000002.npy  # Training shard 2
└── ...                          # ~100 shards total
```

### Quick test without FineWeb
If you don't want to download the full dataset, the training script will automatically fall back to `input.txt` (Shakespeare) for testing.

## 🧠 Reproducing GPT-3: What Weights to Set

GPT-3 uses the **exact same transformer block architecture** as GPT-2 (decoder-only, pre-norm, causal attention, MLP with GELU). The only things that change are **scale** and **context window**.

### Architecture differences vs GPT-2

| Parameter      | GPT-2 (124M) | GPT-3 Small (125M) | GPT-3 175B   |
|----------------|--------------|--------------------|--------------|
| `block_size`   | **1024**     | **2048**           | **2048**     |
| `vocab_size`   | 50257        | 50257              | 50257        |
| `n_layer`      | 12           | 12                 | 96           |
| `n_head`       | 12           | 12                 | 96           |
| `n_embd`       | 768          | 768                | 12288        |

The key insight: **double the context window** (`block_size = 2048`) and **scale up** `n_layer`, `n_head`, and `n_embd`.

### All GPT-3 model sizes (Brown et al., 2020 – Table 2)

| Preset           | Approx params | `n_layer` | `n_head` | `n_embd` | `block_size` |
|------------------|---------------|-----------|----------|----------|--------------|
| `gpt3-small`     | 125M          | 12        | 12       | 768      | 2048         |
| `gpt3-medium`    | 350M          | 24        | 16       | 1024     | 2048         |
| `gpt3-large`     | 760M          | 24        | 16       | 1536     | 2048         |
| `gpt3-xl`        | 1.3B          | 24        | 16       | 2048     | 2048         |
| `gpt3-2.7b`      | 2.7B          | 32        | 32       | 2560     | 2048         |
| `gpt3-6.7b`      | 6.7B          | 32        | 32       | 4096     | 2048         |
| `gpt3-13b`       | 13B           | 40        | 40       | 5120     | 2048         |
| `gpt3-175b`      | 175B          | 96        | 96       | 12288    | 2048         |

All sizes use `vocab_size = 50257` (same GPT-2 BPE tokenizer).

### How to instantiate a GPT-3 model

```python
# Option 1 – use the named preset factory (recommended)
model = GPT.from_gpt3_preset("gpt3-small")

# Option 2 – build the config manually
from train_gpt2 import GPT, GPTConfig
config = GPTConfig(
    block_size=2048,   # GPT-3 uses a 2048-token context window
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
)
model = GPT(config)
```

### Matching training hyperparameters (from the paper)

| Model size | Batch size (tokens) | `max_lr` | `warmup_steps`* |
|------------|---------------------|----------|-----------------|
| Small      | 0.5M                | 6.0e-4   | 715             |
| Medium     | 0.5M                | 3.0e-4   | 715             |
| Large      | 0.5M                | 2.5e-4   | 715             |
| XL         | 1M                  | 2.0e-4   | 1430            |
| 2.7B       | 1M                  | 1.6e-4   | 1430            |
| 6.7B       | 2M                  | 1.2e-4   | 2860            |
| 13B        | 2M                  | 1.0e-4   | 2860            |
| 175B       | 3.2M                | 6.0e-5   | 4577            |

\* `warmup_steps` estimated as 375M warmup tokens ÷ batch_size_tokens.

All GPT-3 sizes use:
- **Optimizer**: AdamW with `betas=(0.9, 0.95)`, `eps=1e-8`
- **Weight decay**: 0.1 on 2-D parameters (same as GPT-2 training here)
- **LR schedule**: cosine decay to `min_lr = 0.1 × max_lr`
- **Mixed precision**: bfloat16 recommended

---

## 📊 Visualizing Training Results

After training completes, you can analyze the results using the `play.ipynb` notebook (Cell 26):

### Prerequisites
1. **Training must be complete** - The log file `log/log.txt` is created during training
2. **FineWeb dataset downloaded** - Required for HellaSwag evaluation during training
3. **HellaSwag evaluation ran** - The training script evaluates HellaSwag every 250 steps

### Options for training configuration

| Option | Description |
|--------|-------------|
| `use_compile = True` | Enable `torch.compile()` for faster training (Linux only) |
| `use_compile = False` | Disable compile to allow HellaSwag eval and text generation |
| Skip val loss check | Comment out validation loss section (see below) |
| Without DDP | Run `python train_gpt2.py` instead of `torchrun` for single GPU |

### Skipping validation loss check

To skip validation loss evaluation, comment out lines ~575-592 in `train_gpt2.py`:

```python
    # once in a while evaluate our validation loss
    # if step % 250 == 0 or last_step:
    #     model.eval()
    #     val_loader.reset()
    #     with torch.no_grad():
    #         val_loss_accum = 0.0
    #         val_loss_steps = 20
    #         for _ in range(val_loss_steps):
    #             x, y = val_loader.next_batch()
    #             x, y = x.to(device), y.to(device)
    #             with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #                 logits, loss = model(x, y)
    #             loss = loss / val_loss_steps
    #             val_loss_accum += loss.detach()
    #     if ddp:
    #         dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    #     if master_process:
    #         print(f"validation loss: {val_loss_accum.item():.4f}")
    #         with open(log_file, "a") as f:
    #             f.write(f"{step} val {val_loss_accum.item():.4f}\n")
```

### Tunable Hyperparameters

Key hyperparameters in `train_gpt2.py` (lines ~497-527) that you can adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 19073 | Training steps (~1 epoch over 10B tokens). **Increase 3x for better HellaSwag** |
| `max_lr` | 6e-4 | Maximum learning rate |
| `min_lr` | 6e-5 | Minimum learning rate (0.1 × max_lr) |
| `warmup_steps` | 715 | Linear warmup steps (~375M tokens) |
| `total_batch_size` | 524288 | Tokens per optimizer step (~0.5M) |
| `B` | 16 | Micro batch size (per GPU) |
| `T` | 1024 | Sequence length (context window) |
| `weight_decay` | 0.1 | Weight decay for 2D params (line ~548) |

**Example: Training 3x longer**
```python
max_steps = 19073 * 3  # ~57k steps, 3 epochs over 10B tokens
```

This significantly improves HellaSwag accuracy (observed ~5-10% improvement over baseline).

### Model Checkpointing

The training script saves model checkpoints every 5000 steps (configurable). Checkpoints are saved to `log/model_{step:05d}.pt`:

```python
if step > 0 and (step % 5000 == 0 or last_step):
    # optionally write model checkpoints
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': raw_model.state_dict(),
        'config': raw_model.config,
        'step': step,
        'val_loss': val_loss_accum.item()
    }
    # you might also want to add optimizer.state_dict() and
    # rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)
```

**To change checkpoint frequency**, modify `step % 5000` to your preferred interval.

### Viewing the plots
Open `play.ipynb` and run Cell 26 to see:
- **Loss curves** - Training and validation loss vs GPT-2 baseline
- **HellaSwag accuracy** - Model accuracy vs GPT-2/GPT-3 baselines

```python
# The cell parses log/log.txt and plots:
# - Panel 1: Train/Val loss with GPT-2 baseline (3.2924 for 124M)
# - Panel 2: HellaSwag accuracy vs GPT-2 (29.4%) and GPT-3 (33.7%) baselines
```

## Running with Multiple GPUs (DDP)

To train on multiple GPUs using Distributed Data Parallel:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

- `--standalone`: Single-node training (no multi-machine setup)
- `--nproc_per_node=8`: Number of GPUs to use (adjust based on your setup)

This will automatically:
- Spawn 8 processes (one per GPU)
- Each GPU processes different data batches
- Gradients are averaged across all GPUs
- Reduces `grad_accum_steps` by 8× to maintain the same total batch size

## Results
I got validation loss down to 1.9 from 10 over training of 14 hours!

## �📝 License

MIT
