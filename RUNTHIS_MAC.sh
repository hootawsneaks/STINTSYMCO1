#!/bin/bash
echo "May God be with ye."

if [ -d ".venv" ]; then
    source .venv/bin/activate
    uv pip install --pre torch torchvision torchaudio
    uv pip install ultralytics jupyter
    echo "Enter the venv with: source .venv/bin/activate"
else
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
    fi
    uv venv
    source .venv/bin/activate
    uv pip install --pre torch torchvision torchaudio
    uv pip install ultralytics jupyter albumentations
    echo "Enter the venv with: source .venv/bin/activate"
fi