#!/usr/bin/env bash
# PHM2010 download via Kaggle mirror (CC0). Requires Kaggle credentials:
#   1) kaggle.com -> Account -> Create New API Token -> saves kaggle.json
#   2) mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# Fallback: IEEE DataPort (institutional login):
#   https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge
# QIT-CEMC dataset (source pool + hyperprior): https://doi.org/10.6084/m9.figshare.27323346
#   -> unzip + unrar into data/raw/scidata/
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=data/raw/phm2010
mkdir -p "$RAW"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "[i] installing kaggle CLI..."
  python3 -m pip install --quiet kaggle
fi
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "[!] ~/.kaggle/kaggle.json not found — follow the steps in this script's header." >&2
  exit 1
fi

kaggle datasets download rabahba/phm-data-challenge-2010 -p "$RAW" --unzip
echo "[ok] downloaded to $RAW"
find "$RAW" -maxdepth 2 -type d | head -20
