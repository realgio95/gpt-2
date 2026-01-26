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

### 10. Run training
```bash
python train_gpt2.py
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
## �📝 License

MIT
