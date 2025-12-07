# llm-lab Setup on RunPods (using A40 as an example)

Complete step-by-step guide for setting up the `llm-lab` environment on a fresh RunPods A40 GPU pod.

**GPU**: NVIDIA A40 (48GB VRAM)
**Base Image**: Typically Ubuntu with CUDA pre-installed
**Working Directory**: `/workspace/`

---

## Prerequisites

- RunPods account with A40 pod deployed
- SSH or web terminal access to the pod
- GitHub access configured (for cloning private repos)

---

## Step 1: Install Miniforge (Mamba + Conda)

Miniforge provides both `mamba` and `conda`. Mamba is a faster drop-in replacement for conda.

```bash
# Navigate to a temporary location
cd /tmp

# Download the latest Miniforge installer
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Run the installer (non-interactive mode)
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# Clean up installer
rm Miniforge3-$(uname)-$(uname -m).sh
```

---

## Step 2: Initialize Shell for Conda/Mamba

This step is **critical** — without it, `mamba activate` will fail.

```bash
# Initialize conda for bash
~/miniforge3/bin/conda init bash

# Reload shell configuration
source ~/.bashrc
```

**Verify installation:**

```bash
mamba --version
conda --version
```

You should see version numbers for both.

---

## Step 3: (Optional) Disable Auto-Activation of Base

If you don't want `(base)` to activate automatically on every login:

```bash
conda config --set auto_activate_base false
```

---

## Step 4: Clone the llm-lab Repository

```bash
cd /workspace

# Clone via HTTPS (if no SSH key configured)
git clone https://github.com/YOUR_USERNAME/llm-lab.git

# OR clone via SSH (if SSH key is configured)
git clone git@github.com:YOUR_USERNAME/llm-lab.git

cd llm-lab
```

**Note**: Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 5: Create the llm-lab Environment

```bash
cd /workspace/llm-lab

# Create environment from environment.yml
mamba env create -f environment.yml
```

This will:

- Create a conda environment named `llm-lab`
- Install Python 3.11
- Install PyTorch, transformers, peft, trl, bitsandbytes, etc.
- Install dev tools (pytest, black, ruff, etc.)

**Expected time**: 3-10 minutes depending on network speed.

---

## Step 6: Activate the Environment

```bash
mamba activate llm-lab
```

> **Note**: If you encounter a shell initialization error, run:
>
> ```bash
> /root/miniforge3/bin/mamba shell init --shell bash --root-prefix=/root/miniforge3
> source ~/.bashrc
> mamba activate llm-lab
> ```

**Verify activation:**

```bash
# Should show the llm-lab environment path
which python

# Should show Python 3.11.x
python --version

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Step 7: Install llm-lab in Editable Mode

The `environment.yml` includes `-e .` which should install the package, but verify:

```bash
cd /workspace/llm-lab

# Install with dev dependencies
pip install -e ".[dev]"
```

---

## Step 8: Register Jupyter Kernel (for Notebooks)

This allows VSCode and Jupyter to detect the `llm-lab` environment:

```bash
python -m ipykernel install --user --name llm-lab --display-name "Python (llm-lab)"
```

---

## Step 9: (Optional) Setup Pre-commit Hooks

For development with automatic code formatting:

```bash
pre-commit install
```

---

## Step 10: Verify Setup

```bash
# Run tests
pytest

# Check key imports
python -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
python -c "from peft import LoraConfig; print('peft OK')"
python -c "from trl import SFTTrainer; print('trl OK')"
python -c "import bitsandbytes; print('bitsandbytes OK')"
```

---

## Quick Reference Commands

| Task                   | Command                                                      |
| ---------------------- | ------------------------------------------------------------ |
| Activate environment   | `mamba activate llm-lab`                                     |
| Deactivate environment | `mamba deactivate`                                           |
| List environments      | `mamba env list`                                             |
| Update environment     | `mamba env update -f environment.yml`                        |
| Remove environment     | `mamba env remove -n llm-lab`                                |
| Check GPU              | `nvidia-smi`                                                 |
| Check PyTorch GPU      | `python -c "import torch; print(torch.cuda.is_available())"` |

---

## Troubleshooting

### "mamba activate" fails with shell initialization error

Run:

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate llm-lab
```

Or re-initialize:

```bash
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### VSCode doesn't detect the environment

1. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Enter path: `~/miniforge3/envs/llm-lab/bin/python`

Or reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"

### Environment creation fails

Try with conda instead:

```bash
conda env create -f environment.yml
```

### CUDA not available in PyTorch

Verify CUDA is installed on the pod:

```bash
nvidia-smi
nvcc --version
```

If PyTorch was installed without CUDA support, reinstall:

```bash
mamba activate llm-lab
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## One-Liner Setup Script

For convenience, you can run this entire setup as a script:

```bash
#!/bin/bash
set -e

echo "=== Step 1: Installing Miniforge ==="
cd /tmp
wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
rm Miniforge3-$(uname)-$(uname -m).sh

echo "=== Step 2: Initializing shell ==="
~/miniforge3/bin/conda init bash
source ~/.bashrc

echo "=== Step 3: Disabling base auto-activation ==="
conda config --set auto_activate_base false

echo "=== Step 4: Cloning llm-lab ==="
cd /workspace
git clone https://github.com/YOUR_USERNAME/llm-lab.git || echo "Repo already exists"
cd llm-lab

echo "=== Step 5: Creating environment ==="
eval "$(mamba shell hook --shell bash)"
mamba env create -f environment.yml

echo "=== Step 6: Activating and installing ==="
mamba activate llm-lab
pip install -e ".[dev]"

echo "=== Step 7: Registering Jupyter kernel ==="
python -m ipykernel install --user --name llm-lab --display-name "Python (llm-lab)"

echo "=== Step 8: Installing pre-commit ==="
pre-commit install

echo "=== Setup Complete ==="
echo "Run 'mamba activate llm-lab' to activate the environment."
```

Save as `/workspace/setup_runpods.sh` and run with `bash setup_runpods.sh`.

---

## Related Documents

- [A30 Setup Notes](./llm-lab-on-runpods-A30.md)
- [Shell Initialization Issues](./working-with-llm-lab-pod-A30.md)

---

## Setting Up SSH Connection for GitHub

To use `git push` and `git pull` without entering credentials, configure SSH authentication.

### 1. Generate SSH Key on the Pod

Create a unique key pair on your RunPod instance.

1. Connect to the pod with Windsurf/VS Code
2. Open the Terminal (<kbd>Ctrl</kbd> + <kbd>`</kbd> or **Terminal > New Terminal**)
3. Generate the key pair (press Enter for default location; consider setting a passphrase):

```bash
ssh-keygen -t ed25519 -C "runpod-a40-git-key"
```

This creates:
- Private key: `/root/.ssh/id_ed25519`
- Public key: `/root/.ssh/id_ed25519.pub`

### 2. Retrieve the Public Key

Display and copy the public key content:

```bash
cat /root/.ssh/id_ed25519.pub
```

Copy the entire output (starts with `ssh-ed25519 AAA...` and ends with your comment).

### 3. Add Key to GitHub

1. Go to [GitHub](https://github.com) and log in
2. Navigate to: **Profile picture** → **Settings** → **SSH and GPG keys**
3. Click **New SSH key**
4. Fill in:
   - **Title**: Descriptive name (e.g., `RunPod A40`)
   - **Key**: Paste the public key string
5. Click **Add SSH key**

### 4. Test and Clone Your Repository

Test the connection:

```bash
ssh -T git@github.com
```

You should see a message confirming successful authentication.

Clone your repository using the SSH URL:

```bash
cd /workspace
git clone git@github.com:yourusername/your-repo-name.git
```

You can now use `git push` and `git pull` without entering credentials.

---

## Setting Up the Git Repository

After SSH authentication is configured, set up your repository.

### Option A: Clone a Fresh Repository

If the repo doesn't exist on the pod yet:

```bash
cd /workspace
git clone git@github.com:pleiadian53/llm-lab.git
cd llm-lab
```

### Option B: Configure an Existing Repository

If the repo was cloned via HTTPS and you want to switch to SSH:

```bash
cd /workspace/llm-lab

# Check current remote URL
git remote -v

# Change from HTTPS to SSH
git remote set-url origin git@github.com:pleiadian53/llm-lab.git

# Verify the change
git remote -v
```

### Configure Git Identity

Set your name and email for commits:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Verify Setup

```bash
cd /workspace/llm-lab
git status
git pull origin main
```

> **Note**: Run git commands from inside `/workspace/llm-lab`, not `/workspace`. The `/workspace` directory is not a git repository.

### First Commit with Pre-commit Hooks

If pre-commit hooks are installed (Step 9), your first commit may take longer as hooks initialize. If hooks auto-fix files, re-run:

```bash
git add -A
git commit -m "your message"
```

To bypass hooks temporarily (e.g., if mypy has pre-existing errors):

```bash
git commit --no-verify -m "your message"
```

---

## Appendix: Local SSH Config Example

On your **local machine** (not the pod), you can configure `~/.ssh/config` for easier connections:

```
Host runpod-llm
    # UPDATE: Use IP from "SSH over exposed TCP" section in RunPod dashboard
    HostName 45.135.57.174
    # UPDATE: Use Port from "SSH over exposed TCP" section
    Port 10440
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

Then connect with: `ssh runpod-llm`
