# Hybrid Colab-SSH Development Setup
**Date:** November 30, 2025
**Environment:** MacBook Pro M1 (Local) <-> Google Colab Pro (Remote)

## 1. Overview
We have established a "Hybrid" workflow. We use the local Windsurf/Cursor IDE to write code, but the code actually resides on Google Drive and executes on Google Colab's GPU runtime.

## 2. Prerequisites (Local M1 Mac)
* **Cloudflare Tunnel:** Installed via Homebrew to allow the SSH connection.
    * Command: `brew install cloudflare/cloudflare/cloudflared`
* **VS Code Extension:** "Remote - SSH" installed.
* **SSH Config:** Located at `~/.ssh/config` with specific `ProxyCommand` for M1 architecture.

## 3. The Colab "Boot" Script
Every time a new Colab runtime is started, this script must be run in the first cell to initialize the environment:

```python
import os
from google.colab import drive
from colab_ssh import launch_ssh_cloudflared

# 1. Mount Drive (Browser Auth required first)
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Configure Persistent Cache (Saves LLMs to Drive)
CACHE_DIR = "/content/drive/MyDrive/llm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR

# 3. Launch SSH Tunnel
# Note: Install colab_ssh if missing: !pip install colab_ssh --quiet
launch_ssh_cloudflared(password="my_password_here")
```



---

