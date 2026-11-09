#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch image with CUDA preinstalled).
# The working tree is rsynced to /workspace/Sen4Sam (no git clone needed) and the data zip to
# /workspace/sen4sam_data.zip (processed/ + manifest.csv at its root). Run:
#     bash /workspace/Sen4Sam/scripts/runpod_setup.sh
# Safe to re-run: every step is skipped when already done.
set -euo pipefail
REPO=${REPO:-/workspace/Sen4Sam}
DATA_ZIP=${DATA_ZIP:-/workspace/sen4sam_data.zip}
PY=${PYTHON:-python}

if [ ! -d "$REPO" ]; then
  echo "[setup] $REPO missing, cloning"
  git clone https://github.com/ZagonTM/Sen4Sam "$REPO"
fi
cd "$REPO"

echo "[setup] python: $(command -v $PY)"
$PY -c "import torch; print('[setup] torch', torch.__version__, '| cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# --- deps, without touching the preinstalled torch -----------------------------------------
# Pin the installed torch/torchvision versions as constraints so nothing upgrades/downgrades them.
CONSTRAINTS=$(mktemp)
$PY -m pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio)(==| @)' > "$CONSTRAINTS" || true
echo "[setup] pinned: $(tr '\n' ' ' < "$CONSTRAINTS")"

pipi() {   # install into the interpreter that has torch: pip first, then uv --system as fallback
  if $PY -m pip install -q -c "$CONSTRAINTS" "$@"; then return 0; fi
  if $PY -m pip install -q --break-system-packages -c "$CONSTRAINTS" "$@"; then return 0; fi
  if command -v uv >/dev/null; then
    uv pip install --python "$(command -v $PY)" --system -c "$CONSTRAINTS" "$@" && return 0
    uv pip install --python "$(command -v $PY)" --system --break-system-packages -c "$CONSTRAINTS" "$@" && return 0
  fi
  echo "[setup] install failed: $*"; return 1
}

pipi numpy pandas scipy pillow matplotlib requests tifffile tqdm "hydra-core>=1.3.2" "iopath>=0.1.10"
if ! $PY -c "import torchvision" 2>/dev/null; then
  echo "[setup] WARNING: torchvision missing; sam2 needs it. Install a torchvision build matching torch $($PY -c 'import torch;print(torch.__version__)') manually."
fi
if ! $PY -c "import sam2" 2>/dev/null; then
  # SAM2_BUILD_CUDA=0: skip the optional CUDA extension; --no-deps: never let it pull another torch.
  SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 pipi --no-deps --no-build-isolation \
      "git+https://github.com/facebookresearch/sam2.git"
fi

# --- checkpoint ----------------------------------------------------------------------------
mkdir -p checkpoints data
CKPT=checkpoints/sam2.1_hiera_small.pt
if [ ! -s "$CKPT" ]; then
  URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
  if command -v wget >/dev/null; then wget -q -O "$CKPT.part" "$URL"; else curl -fsSL -o "$CKPT.part" "$URL"; fi
  mv "$CKPT.part" "$CKPT"
fi
echo "[setup] checkpoint: $(du -h "$CKPT" | cut -f1)"

# --- data ----------------------------------------------------------------------------------
n_patches() { ls data/processed 2>/dev/null | wc -l | tr -d ' '; }
if [ "$(n_patches)" -lt 2664 ] || [ ! -f data/manifest.csv ]; then
  test -f "$DATA_ZIP" || { echo "[setup] Upload $DATA_ZIP first"; exit 1; }
  command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip; }
  TMP=/workspace/_sen4sam_unzip
  rm -rf "$TMP"; mkdir -p "$TMP"
  unzip -q -o "$DATA_ZIP" -d "$TMP"
  # find the manifest and the directory that actually holds the *_rgb.png files (handles
  # processed/, data/processed/ and processed/processed/ layouts)
  MAN=$(find "$TMP" -name manifest.csv -not -path '*/__MACOSX/*' | head -1)
  RGB=$(find "$TMP" -name '*_rgb.png' -not -path '*/__MACOSX/*' | head -1)
  [ -n "$MAN" ] && [ -n "$RGB" ] || { echo "[setup] zip lacks manifest.csv or *_rgb.png"; exit 1; }
  rm -rf data/processed
  mv "$(dirname "$RGB")" data/processed
  mv "$MAN" data/manifest.csv
  rm -rf "$TMP"
fi
echo "[setup] patch files: $(n_patches) (expect 2664) | manifest: $(test -f data/manifest.csv && echo ok || echo MISSING) | split: $(test -f results/split.csv && echo ok || echo MISSING)"

# --- smoke check ---------------------------------------------------------------------------
$PY -c "import torch,sam2;print(torch.cuda.is_available())"
$PY -c "import sys; sys.path.insert(0,'src'); from sam2_auto_segment import build_model; m=build_model(); print('[setup] model ok on', next(m.parameters()).device)"
echo "[setup] done"
