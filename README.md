# GPT-2

A project for exploring and training GPT-2 models using PyTorch and Hugging Face Transformers.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Jupyter kernel for notebooks

```bash
pip install ipykernel
```

## Requirements

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)


# Run this to 
python.exe -m pip install --upgrade pip 

To change the notebook kernel in VS Code:

Click the kernel name in the top-right corner of the notebook (e.g., "Python 3.11.9" or "base")

A dropdown menu appears with options:

Python Environments - Shows all detected Python interpreters including .venv
Jupyter Kernel - Shows registered Jupyter kernels
Existing Jupyter Server - Connect to a remote server
Select Python Environments → Choose your .venv (.venv)

Keyboard shortcut: Press Ctrl+Shift+P and type "Notebook: Select Notebook Kernel"

If .venv doesn't appear, try:

Reload VS Code (Ctrl+Shift+P → "Developer: Reload Window")
Ensure ipykernel is installed in the venv (already done earlier)