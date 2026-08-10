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

Each EDAP plugin: 4-head cross-depth attention with 3× LayerNorm (input, key, output) and learnable depth position embeddings.

| Component | Params |
|-----------|--------|
| EDAP plugins (4×) | 206M |
| LM head (unfrozen) | 545M |
| **Total trainable** | **751M (9.9% of backbone)** |
| Qwen2.5-7B backbone | 7.6B (frozen) |

## Quick Start

```bash
# 1. Setup environment
bash scripts/setup.sh
conda activate edap

# 2. Prepare data (official ConFiQA → training format)
python scripts/convert_confiqa.py

# 3. Quick debug run (100 samples, 3 steps)
python src/train.py --dry_run

# 4. Full training
python src/train.py                      # EDAP
python src/train.py --shuffle_depth      # EDAP-random (control)

# 5. Evaluate baselines
python src/evaluate.py --baseline greedy
python src/evaluate.py --baseline cad
python src/evaluate.py --baseline dola

# 6. Evaluate trained EDAP
python src/evaluate.py --checkpoint ./checkpoints/edap_epoch3.pt

# 7. Generate full comparison report
python src/report.py \
    --edap_ckpt checkpoints/edap_epoch3.pt \
    --edap_random_ckpt checkpoints/edap_random_epoch3.pt
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (V100 32GB minimum; A100 40GB recommended)
- See `environment.yml` or `requirements.txt` for Python dependencies

## Data

- **ConFiQA**: converts official ConFiQA QA/MR/MC splits into unified training format. Official data goes in `ConFiQA/`; run `python scripts/convert_confiqa.py` to produce `data/confiqa/confiqa_train.json`
- **NQ-Swap**: auto-downloaded from HuggingFace (`pminervini/NQ-Swap`) for evaluation

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
| EDAP-random | Cross-depth (shuffled control) | This work |

## Key Design Decisions

- **Multi-plugin chain**: 4 plugins at block boundaries for incremental calibration
- **Multi-head (H=4)**: dimension split across heads for nuanced attention
- **Three LayerNorms**: Input (depth magnitude), Key (attention stability), Output (residual preservation)
- **Zero-init W_O**: Plugin starts as identity mapping, learns to deviate only where needed
- **Unfrozen lm_head**: Required for gradient pathway through frozen backbone
- **EDAP-random control**: Same architecture with shuffled depth order — proves depth dimension matters

## License

MIT
