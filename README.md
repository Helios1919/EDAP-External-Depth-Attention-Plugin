# EDAP: External Depth Attention Plugin

**Plug-in cross-depth attention modules for resolving residual stream conflicts in frozen LLMs.**

When a language model's parametric (memorized) knowledge contradicts external context (e.g., retrieved documents in RAG), LLMs often default to their internal memory, causing hallucinations. EDAP inserts lightweight, trainable multi-head attention plugins at transformer block boundaries that learn to calibrate trust across depth — explicitly weighting how much to rely on representations from shallow vs. deep layers for a given query.

## Architecture

```
Frozen Qwen2.5-7B (28 layers → 4 blocks of 7)
          │
Block 0 → [EDAP₀] → r'₀ ─┐
Block 1 → [EDAP₁] → r'₁ ─┤
Block 2 → [EDAP₂] → r'₂ ─┼─→ Trainable LM Head → Answer
Block 3 → [EDAP₃] → r'₃ ─┘
```

Each EDAP plugin: 8-head cross-depth attention with 3× LayerNorm (input, key, output) and learnable depth position embeddings.

| Component | Params |
|-----------|--------|
| EDAP plugins (4×) | 206M |
| LM head (unfrozen) | 545M |
| **Total trainable** | **751M (9.9% of backbone)** |
| Qwen2.5-7B backbone | 7.6B (frozen) |

## Quick Start

```bash
# 1. Setup environment (conda env + model + raw data)
bash scripts/setup.sh
conda activate edap

# 2. Prepare data (official ConFiQA → training format)
python scripts/convert_confiqa.py

# 3. Quick debug run (100 samples, 3 steps)
python src/train.py --dry_run

# 4. Full training (A100 defaults: batch=8, grad_accum=2)
python src/train.py

#   V100 fallback:
#   python src/train.py --batch_size 2 --grad_accum 8

#   Skip validation split (train on all data):
#   python src/train.py --val_split 0

# 5. Evaluate all methods (runs on both NQ-Swap & ConFiQA, 500 samples each)
python src/evaluate.py --baseline greedy  --max_samples 500
python src/evaluate.py --baseline cad     --max_samples 500
python src/evaluate.py --baseline dola    --max_samples 500
python src/evaluate.py --checkpoint ./checkpoints/edap_best.pt --max_samples 500
```

**Model download**: `setup.sh` downloads Qwen2.5-7B to `./models/qwen2.5-7b`. If you already have the model elsewhere, pass `--model_path /path/to/model` to `train.py` and `evaluate.py`.

### Training Notes

- **Full-sequence loss**: Uses teacher-forcing over all answer tokens (not single-token), fixing the loss→0 overfitting problem
- **Validation split**: Default 20% held out (`--val_split 0.2`); val loss logged per epoch for overfitting monitoring
- **Early stopping**: Default patience=2 epochs (`--early_stop_patience 2`); saves best checkpoint only
- **Label smoothing**: 0.1 default for regularization (`--label_smoothing 0.1`)
- **Dropout**: 0.1 in EDAP attention (`--edap_dropout 0.1`)
- **Auto dtype**: Detects bf16 support (A100) vs fp16 fallback (V100)
- **Checkpointing**: Saved after every epoch to `--output_dir` (default `./checkpoints`; on AutoDL auto-redirects to `/root/autodl-tmp/checkpoints`)

### Evaluation Notes

- All methods (EDAP, Greedy, CAD, DoLa) use **multi-token greedy generation** — not single-token argmax — for a fair comparison
- Exact match (EM) is computed after normalizing whitespace and punctuation
- Results broken down by `correct_source` (context vs memory) and saved as JSON
- **Evaluates both NQ-Swap and ConFiQA** automatically; NQ-Swap is auto-downloaded from HuggingFace
- Use `--max_samples N` to limit eval size (default 0 = all)
- Checkpoint config (n_heads, n_blocks, dropout) auto-detected from saved state

## Requirements

- Python 3.12+
- **A100 40GB / 80GB recommended** (native bf16, batch_size=8)
- V100 32GB works but needs manual override: `--batch_size 2 --grad_accum 8`
- See `environment.yml` or `requirements.txt` for Python dependencies

## Data

### ConFiQA (training + eval)

1. Download raw data from HuggingFace:
   ```bash
   huggingface-cli download miii/ConFiQA --local-dir ./ConFiQA
   ```
   (produces `ConFiQA/ConFiQA-QA.json`, `ConFiQA/ConFiQA-MR.json`, `ConFiQA/ConFiQA-MC.json`)
2. Convert to unified training format:
   ```bash
   python scripts/convert_confiqa.py
   ```
   (produces `data/confiqa/confiqa_train.json`)

### NQ-Swap (eval only)

Auto-downloaded from HuggingFace (`pminervini/NQ-Swap`) on first evaluation run. Cached at `data/nqswap/nqswap_dev.json`.

> **Network note**: If HuggingFace is inaccessible (e.g. mainland China), set `export HF_ENDPOINT=https://hf-mirror.com` before running downloads.

## Project Structure

```
EDAP/
├── src/
│   ├── edap_plugin.py      # EDAP plugin class + factory
│   ├── train.py            # Training script
│   ├── data_utils.py       # ConFiQA dataset + NQ-Swap loader
│   ├── evaluate.py         # Evaluation (greedy, CAD, DoLa, EDAP)
│   └── report.py           # Full comparison report generator
├── scripts/
│   ├── setup.sh            # One-click environment setup
│   └── convert_confiqa.py  # Convert official ConFiQA splits to training format
├── environment.yml         # Conda environment spec
├── requirements.txt        # Pip dependencies
└── README.md
```

## Baselines

| Method | Mechanism | Paper |
|--------|-----------|-------|
| Greedy | No intervention | — |
| CAD | Output logit contrast | Shi et al., 2023 |
| DoLa | Internal layer contrast (ICLR) | Chuang et al., 2024 |

## Key Design Decisions

- **Multi-plugin chain**: 4 plugins at block boundaries for incremental calibration
- **Multi-head (H=8)**: dimension split across heads for nuanced attention
- **Three LayerNorms**: Input (depth magnitude), Key (attention stability), Output (residual preservation)
- **Zero-init W_O**: Plugin starts as identity mapping, learns to deviate only where needed
- **Unfrozen lm_head**: Required for gradient pathway through frozen backbone
- **Ablation**: `--shuffle_depth` flag randomizes source ordering to verify depth dimension matters (optional ablation experiment)

## Results (ConFiQA-trained EDAP, Qwen2.5-7B backbone)

Evaluation on 200 samples per dataset, multi-token greedy decoding:

| Method | ConFiQA EM | NQ-Swap EM |
|--------|-----------|------------|
| Greedy (no intervention) | — | — |
| CAD | — | — |
| DoLa | — | — |
| **EDAP** | — | — |

> Run `python src/evaluate.py --baseline <method> --max_samples 500` to reproduce.
> Each JSON includes `em`, `em_prefix` (catches correct answers that fail to stop), and `em_by_source` (breakdown by context/memory).

## License

MIT
