# Windows Installation Guide

The `opengradient` package requires a C compiler to build its native dependencies. Windows does not have one by default.

## Step 1 — Enable WSL

Open PowerShell as Administrator and run:

    wsl --install

Restart your PC when prompted.

## Step 2 — Install Python inside WSL

    sudo apt update && sudo apt install -y python3 python3-pip python3-venv

## Step 3 — Create virtual environment

    python3 -m venv og-env
    source og-env/bin/activate

## Step 4 — Install SDK

    pip install opengradient

## Step 5 — Verify

    python3 -c "import opengradient; print('Ready!')"

## Common Errors

- Visual C++ 14.0 required → Use WSL instead
- wsl: command not found → Update Windows 10 to Build 19041+
- WSL stuck → Enable Virtualization in BIOS
