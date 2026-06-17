#!/usr/bin/env bash
#
# Build the Python engine as a standalone sidecar binary for packaging.
#
# Output: desktop/src-tauri/binaries/infinidev-server-<target-triple>
# (the name Tauri's `externalBin` expects; it copies it next to the app
#  executable, where src-tauri/src/lib.rs looks for it in release builds).
#
# This is a STARTING POINT. The engine pulls in heavy native dependencies
# (chromadb + onnxruntime, tree-sitter grammars, litellm). PyInstaller can
# bundle them but the exact --collect-all / hidden-import set often needs
# per-platform tuning. Build on the OS you are targeting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$REPO_ROOT/desktop/src-tauri/binaries"
ENTRY="$SCRIPT_DIR/sidecar_entry.py"

# Target triple matches what Tauri's externalBin expects (the Rust host triple).
TRIPLE="$(rustc -Vv | sed -n 's/^host: //p')"
NAME="infinidev-server-$TRIPLE"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

echo "Building $NAME → $OUT_DIR"

uv run --extra serve --with pyinstaller pyinstaller \
  --onefile --noconfirm --clean \
  --name "$NAME" \
  --distpath "$OUT_DIR" \
  --workpath "$REPO_ROOT/desktop/.pyinstaller-build" \
  --specpath "$REPO_ROOT/desktop/.pyinstaller-build" \
  --collect-all chromadb \
  --collect-all onnxruntime \
  --collect-all litellm \
  --collect-all tokenizers \
  --collect-all tree_sitter \
  --collect-submodules tree_sitter_python \
  --collect-submodules tree_sitter_javascript \
  --collect-submodules tree_sitter_typescript \
  --collect-submodules infinidev \
  --copy-metadata infinidev \
  --copy-metadata litellm \
  "$ENTRY"

echo "Done: $OUT_DIR/$NAME"
echo
echo "Next: add  \"externalBin\": [\"binaries/infinidev-server\"]  under"
echo "\"bundle\" in src-tauri/tauri.conf.json, then run: npm run tauri:build"
