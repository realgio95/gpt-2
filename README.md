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

## 📝 License

MIT
