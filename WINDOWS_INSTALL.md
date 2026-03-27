# Windows Installation Guide

The `opengradient` package requires a C compiler
to build its native dependencies. Windows does not
have one by default.

## Step 1 — Enable WSL

Open PowerShell as Administrator and run:

    wsl --install

Restart your PC when prompted.

## Step 2 — Install Python and uv inside WSL

Open the Ubuntu app and run:

    sudo apt update && sudo apt install -y python3 curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

## Step 3 — Install SDK

    uv add opengradient

## Step 4 — Verify

    uv run python3 -c "import opengradient; print('Ready!')"

## Common Errors

- Visual C++ 14.0 required → Use WSL instead
- wsl: command not found → Update Windows 10 to Build 19041+
- WSL stuck → Enable Virtualization in BIOS
- uv: command not found → Run: source $HOME/.local/bin/env
