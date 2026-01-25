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

## �📝 License

MIT
