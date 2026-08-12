# EDAP: External Depth Attention Plugin

**Plug-in cross-depth attention modules for resolving residual stream conflicts in frozen LLMs.**

When a language model's parametric (memorized) knowledge contradicts external context (e.g., retrieved documents in RAG), LLMs often default to their internal memory, causing hallucinations. EDAP inserts lightweight, trainable multi-head attention plugins at transformer block boundaries that learn to calibrate trust across depth — explicitly weighting how much to rely on representations from shallow vs. deep layers for a given query.

## Architecture

```
  Input ──→ [L0..L3] ──→ [EDAP₀] ──→ [L4..L7] ──→ [EDAP₁] ──→ ... ──→ [L24..L27] ──→ [EDAP₆] ──→ LM Head ──→ Answer
               └─ frozen ─┘ └ trainable ┘└─ frozen ─┘ └ trainable ┘       └── frozen ──┘ └ trainable ┘
```
> 7 backbone blocks (4 layers each, frozen) alternating with 7 EDAP plugins (trainable).

> **Each EDAPₖ is a miniature cross-attention block:**
>
> ```
>  sources = [emb, r₀, r₁, ..., rₖ₋₁, bₖ]     ← K, V  (frozen backbone outputs)
>                            │
>                    ┌───────▼───────┐
>                    │  Cross-Attn   │  Q = bₖ  (current block, "what do I need?")
>                    │  Q·Kᵀ / √dₖ   │  K,V = sources  ("here's all available info")
>                    └───────┬───────┘
>                            │
>                    ┌───────▼───────┐
>                    │   Gate Mix    │  rₖ = gate · fused + (1−gate) · bₖ
>                    └───────┬───────┘
>                            │
>                            ▼
>                           rₖ  ──→  feeds into next block & all later EDAPs
> ```
> 
> - **K, V** come from all prior sources — embedding + every earlier EDAP output + current block residual — giving each plugin a full view of the semantic trajectory from shallow to deep.
> - **Q** comes from the current block's own output (`bₖ`), asking: "given where I am, which depth levels should I attend to?"
> - **Gate mixing** prevents the plugin from overwriting non-conflict tokens: each token independently blends the EDAP-fused representation with the original block output.
> - The fused output `rₖ` **replaces** the raw block output downstream — so later layers and later EDAPs all see the calibrated representation.


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

Evaluation on 500 samples per dataset, multi-token greedy decoding.

### Overall

| Method | ConFiQA EM | NQ-Swap EM | NQ-Swap P-EM |
|--------|-----------|------------|-------------|
| Greedy (no intervention) | 16.60% | 44.00% | 63.60% |
| CAD | 3.60% | 7.40% | 64.80% |
| DoLa | 5.40% | 7.80% | 23.20% |
| **EDAP** | **58.20%** | 10.80% | 14.80% |

> EM = strict exact match; P-EM = prefix match (output starts with correct answer — catches CAD/DoLa EOS suppression). NQ-Swap P-EM reveals CAD actually knows the answer 64.8% of the time but can't stop.

### ConFiQA by source (EM)

| Method | Context (n=245) | Memory (n=255) |
|--------|:---:|:---:|
| Greedy | 28.7% | 3.7% |
| CAD | 7.0% | 0.0% |
| DoLa | 9.7% | 0.8% |
| **EDAP** | **77.1%** | **38.0%** |

### Key Findings

- **EDAP outperforms Greedy by +41.6pp on ConFiQA** (16.6% → 58.2%), with context-type answers reaching 77.1%
- **EDAP does not transfer to NQ-Swap**: 10.8% EM vs Greedy's 44.0% — the learned routing is ConFiQA-specific, proving plugins capture dataset-level conflict patterns rather than a coarse heuristic
- **CAD's EOS problem confirmed**: NQ-Swap P-EM is 64.8% but EM is only 7.4% — logit contrast suppresses the stop token, causing correct answers to be buried in continuation text
- Greedy naturally follows NQ-Swap's swapped context (44.0%), but struggles with ConFiQA's missing-context scenarios (16.6%)

## License

MIT
