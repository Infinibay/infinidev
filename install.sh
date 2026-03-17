#!/bin/bash

# Infinidev Installer
# Installs infinidev as an isolated CLI tool via uv tool install.
# After install, run 'infinidev' from any directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is required but not installed."
    echo "Install it from: https://docs.astral.sh/uv/getting-started/"
    exit 1
fi

echo "==> Installing Infinidev..."
uv tool install --force "$SCRIPT_DIR"

echo ""
echo "==> Done! Run 'infinidev' from any directory."
echo "    If the command is not found, make sure ~/.local/bin is in your PATH:"
echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
