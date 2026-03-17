#!/bin/bash

# Infinidev Installer — builds a single-file executable via PyInstaller
# and installs it to ~/.local/bin/infinidev

set -e

BASE_DIR="$HOME/.infinidev"
BIN_DIR="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Building Infinidev single-file executable..."

mkdir -p "$BASE_DIR"
mkdir -p "$BIN_DIR"

# --- Ensure build dependencies ---
if command -v uv &> /dev/null; then
    echo "    Using uv..."
    uv sync
    # Install pyinstaller into the project venv if missing
    uv pip install pyinstaller 2>/dev/null || true
    PYTHON="uv run python"
    PIP="uv pip"
    PYINSTALLER="uv run pyinstaller"
else
    echo "    Using pip..."
    python3 -m pip install --quiet -e . pyinstaller
    PYTHON="python3"
    PIP="python3 -m pip"
    PYINSTALLER="pyinstaller"
fi

# --- Build with PyInstaller ---
echo "==> Running PyInstaller (--onefile)..."

$PYINSTALLER \
    --onefile \
    --name infinidev \
    --clean \
    --noconfirm \
    --paths "$SCRIPT_DIR/src" \
    --hidden-import infinidev.tools.file \
    --hidden-import infinidev.tools.git \
    --hidden-import infinidev.tools.shell \
    --hidden-import infinidev.tools.web \
    --hidden-import infinidev.tools.knowledge \
    --hidden-import infinidev.prompts.tech \
    --hidden-import infinidev.prompts.tech.python \
    --hidden-import infinidev.prompts.tech.rust \
    --hidden-import infinidev.prompts.tech.typescript \
    --hidden-import infinidev.prompts.tech.javascript \
    --hidden-import infinidev.config.tech_detection \
    --hidden-import infinidev.cli.tui \
    --hidden-import textual \
    --hidden-import click \
    --hidden-import litellm \
    --hidden-import httpx \
    --hidden-import trafilatura \
    --hidden-import pydantic \
    --hidden-import pydantic_settings \
    --hidden-import crewai \
    --hidden-import crewai.tools \
    --hidden-import ollama \
    --hidden-import chromadb \
    --hidden-import prompt_toolkit \
    --collect-all textual \
    --collect-data crewai \
    --collect-data litellm \
    --collect-data trafilatura \
    --distpath "$SCRIPT_DIR/dist" \
    --workpath "$SCRIPT_DIR/build" \
    --specpath "$SCRIPT_DIR" \
    "$SCRIPT_DIR/src/infinidev/cli/main.py"

# --- Install ---
if [ ! -f "$SCRIPT_DIR/dist/infinidev" ]; then
    echo "ERROR: Build failed — dist/infinidev not found."
    exit 1
fi

cp "$SCRIPT_DIR/dist/infinidev" "$BIN_DIR/infinidev"
chmod +x "$BIN_DIR/infinidev"

# --- Cleanup build artifacts ---
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/infinidev.spec"

echo ""
echo "==> Installed to $BIN_DIR/infinidev"
echo "    Make sure $BIN_DIR is in your PATH."
echo "    Run 'infinidev' from any directory."
