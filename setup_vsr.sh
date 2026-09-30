#!/usr/bin/env bash
# Installs the Spanish visual speech recognition model on a GPU machine
# (tested on a RunPod PyTorch template). Run once:   bash setup_vsr.sh
#
# Model: Ma, Petridis & Pantic (2022), "Visual speech recognition for multiple
# languages in the wild". Model licence: NON-COMMERCIAL / comparative use only.
# Fine for evaluation, NOT for production.
set -euo pipefail
cd "$(dirname "$0")"

echo "== System packages"
apt-get update -qq && apt-get install -y -qq ffmpeg git unzip curl >/dev/null

echo "== Model code"
[ -d vsr ] || git clone -q https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages vsr
python -c "import torch, torchaudio" 2>/dev/null || pip install -q torchaudio
pip install -q -r vsr/requirements.txt
pip install -q "mediapipe==0.10.14" gdown opencv-python-headless

# Links from the model's README (bit.ly -> Google Drive)
download() { # $1 = bit.ly link, $2 = destination .zip
  local url; url=$(curl -sIL -o /dev/null -w '%{url_effective}' "$1")
  # gdown >= 5 understands Drive links directly; older versions need --fuzzy
  gdown -q "$url" -O "$2" 2>/dev/null || gdown -q --fuzzy "$url" -O "$2"
}

echo "== Visual model weights (Spanish, ~186 MB)"
[ -f /tmp/vsr_es.zip ] || download https://bit.ly/34MjWBW /tmp/vsr_es.zip
echo "== Language model (Spanish, ~180 MB)"
[ -f /tmp/lm_es.zip ] || download https://bit.ly/3rppyJN /tmp/lm_es.zip

python -m lipreading.vsr.place_weights vsr /tmp/vsr_es.zip /tmp/lm_es.zip

echo
echo "Done. Quick test of the model:"
echo "  python -m lipreading.vsr.evaluate --zip YOUR_PACKAGE.zip --vsr ./vsr --no-llm"
