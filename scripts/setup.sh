#!/bin/bash
# EDAP environment setup
# Usage: bash setup.sh [--minimal]

set -e

MINIMAL=false
[ "$1" = "--minimal" ] && MINIMAL=true

echo "Setting up EDAP environment..."
echo "$(date)"

# conda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
fi

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda create -n edap python=3.12 -y
conda activate edap

# python deps
pip install --upgrade pip
pip install torch==2.5.1 transformers==4.44.0 accelerate==0.33.0
pip install datasets==2.21.0
pip install wandb tqdm

# optional: flash-attn
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not available, skipping"

echo "Checking install..."
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

if [ "$MINIMAL" = true ]; then
    echo "Minimal setup done. Download model/data manually:"
    echo "  huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./models/qwen2.5-7b"
    echo "  huggingface-cli download miii/ConFiQA --local-dir ./ConFiQA"
    echo "  python scripts/convert_confiqa.py"
    exit 0
fi

# model
mkdir -p ./models/qwen2.5-7b
if [ -f "./models/qwen2.5-7b/config.json" ]; then
    echo "Model already exists, skipping."
else
    echo "Downloading Qwen2.5-7B (this may take a while)..."
    huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./models/qwen2.5-7b
fi

# data — ConFiQA raw files (3 JSONs from miii/ConFiQA on HF)
mkdir -p ./ConFiQA
if [ -f "./ConFiQA/ConFiQA-QA.json" ]; then
    echo "ConFiQA raw data already exists, skipping."
else
    echo "Downloading ConFiQA raw data..."
    huggingface-cli download miii/ConFiQA --local-dir ./ConFiQA
fi

echo ""
echo "Setup complete."
echo "  conda activate edap"
echo "  python scripts/convert_confiqa.py    # convert raw data → training format"
echo "  python src/train.py --dry_run        # quick test"
echo "  python src/train.py                  # full training"
echo "  python src/evaluate.py --baseline greedy --max_samples 500"
