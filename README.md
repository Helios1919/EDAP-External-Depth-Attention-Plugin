# EDAP: External Depth Attention Plugin

**Plug-in cross-depth attention modules for resolving residual stream conflicts in frozen LLMs.**

When a language model's parametric (memorized) knowledge contradicts external context (e.g., retrieved documents in RAG), LLMs often default to their internal memory, causing hallucinations. EDAP inserts lightweight, trainable multi-head attention plugins at transformer block boundaries that learn to calibrate trust across depth — explicitly weighting how much to rely on representations from shallow vs. deep layers for a given query.

## Architecture

```
  Input ──→ [L0..L3] ──→ [EDAP₀] ──→ [L4..L7] ──→ [EDAP₁] ──→ ... ──→ [L24..L27] ──→ [EDAP₆] ──→ LM Head ──→ Answer
               └─ frozen ─┘ └ trainable ┘                                 └── frozen ──┘ └ trainable ┘
```
> Qwen2.5-7B (28 layers) → 7 blocks (4 layers each, frozen) alternating with 7 EDAP plugins (trainable).  
> Block boundaries configurable via `--edap_blocks` or `--block_layers`.

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
# 1. Setup
bash scripts/setup.sh && conda activate edap

# 2. Prepare data
python scripts/convert_confiqa.py

# 3. Dry run (100 samples, 3 steps)
python src/train.py --dry_run

# 4. Full training (A100-40GB)
python src/train.py --freeze_lm_head

#   Recommended flags:
#   --freeze_lm_head       prevent 545M lm_head from memorizing dataset biases
#   --batch_size 2 --grad_accum 8   for A100-80GB
#   --edap_noise 0                  disable exposure-bias noise
#   --val_split 0                   train on all data

# 5. Evaluate baselines
python src/evaluate.py --baseline greedy  --max_samples 500
python src/evaluate.py --baseline cad     --max_samples 500
python src/evaluate.py --baseline dola    --max_samples 500
python src/evaluate.py --checkpoint /root/autodl-tmp/checkpoints/edap_best.pt --max_samples 500

# 6. Generate comparison report
python src/report.py --edap_ckpt /root/autodl-tmp/checkpoints/edap_best.pt \
                      --edap_random_ckpt /root/autodl-tmp/checkpoints/edap_random_best.pt
```

**Model download**: `setup.sh` downloads Qwen2.5-7B to `./models/qwen2.5-7b`. If you already have the model elsewhere, pass `--model_path /path/to/model` to `train.py` and `evaluate.py`.

### Training Notes

- **Teacher forcing over full answer sequences** (not single-token classification)
- **Exposure bias mitigation**: Gaussian noise on EDAP sources during training (`--edap_noise 0.02`, default) to bridge the teacher-forcing → autoregressive gap; set `--edap_noise 0` to disable
- **Target-entropy regularization** (`--lambda_entropy 0.05`): penalizes attention distributions that collapse to a single source or become uniform
- **Gate mean regularization** (`--lambda_gate_reg 0.01`): pulls each plugin's *mean* gate toward 0.5, preventing EDAP from being globally bypassed (gate→0) or hard-replacing the backbone (gate→1), while leaving per-token variance free so the gate keeps discriminative power
- **Stratified validation split**: 20% held out per `correct_source` type (`--val_split 0.2`); early stopping patience=2
- **Label smoothing**: 0.1 default (`--label_smoothing 0.1`)
- **`--freeze_lm_head`** (recommended): freezes the 545M lm_head and inserts a small trainable bottleneck, forcing EDAP to learn meaningful routing instead of letting the lm_head memorize dataset biases
- Checkpoints saved to `--output_dir` (auto-redirects to `/root/autodl-tmp/checkpoints` on AutoDL)

### Evaluation Notes

- All methods use **multi-token greedy generation** for fair comparison
- **DoLa** uses the original dynamic premature-layer selection: at every decoding step it picks the layer with max JS-divergence against the final layer (every 2nd layer, mapped to Qwen2.5-7B's 28 layers), then contrasts `(1+α)·final − α·premature` (`--dola_early_exit=-1` default, `--dola_alpha=1.0`). Pass `--dola_early_exit <layer>` for the static single-layer variant
- Exact match (EM) and prefix-EM reported; prefix-EM catches models that know the answer but can't stop (common with CAD/DoLa)
- Results broken down by `correct_source` (context / memory)
- **Evaluates both NQ-Swap and ConFiQA** automatically; NQ-Swap auto-downloaded from HuggingFace
- Checkpoint config (n_heads, n_blocks, dropout) auto-detected from saved state

## Requirements

- Python 3.12+
- **A100 40GB / 80GB recommended** (native bf16, default batch_size=1 grad_accum=16)
- V100 32GB works but needs manual override: `--batch_size 1 --grad_accum 16` (fp16)
- See `environment.yml` or `requirements.txt` for Python dependencies

## Data

### ConFiQA (training + eval)

1. Download raw data: `huggingface-cli download miii/ConFiQA --local-dir ./ConFiQA`
2. Convert + decontaminate: `python scripts/convert_confiqa.py`
   - Produces `data/confiqa/confiqa_train.json` and `data/confiqa/confiqa_test.json` (80/20 stratified split)
   - Detects and masks answer leaks in counterfactual contexts (replaces leaked spans with `[MASK]`)

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

- **Delta-mode K**: attention keys computed from incremental block differences (`s_i − s_{i−1}`), giving higher contrast than raw cumulated vectors. Learnable baseline for source 0 prevents magnitude asymmetry.
- **Per-token gated mixing**: each token independently blends EDAP-fused output with original block output via a learned sigmoid gate (input LayerNorm-ed, gate initialized near 0.5 with an O(1) logit scale to avoid saturation), preventing the plugin from overwriting non-conflict tokens.
- **Progressive source count**: EDAP₀ sees 2 sources (emb + block₀), EDAP₆ sees 8 — shallow plugins make simple decisions, deep plugins have full trajectory visibility.
- **Zero-init W_O**: plugin starts as identity mapping, learns to deviate only where needed.
- **Shared K/V across plugins** (`--shared_kv`): reduces parameter count ~1/3.
- **`--freeze_lm_head`**: inserts a trainable d→d bottleneck before the frozen lm_head, preventing the 545M classification head from memorizing dataset biases and forcing EDAP to learn meaningful routing.
- **Exposure bias noise** (`--edap_noise`): Gaussian perturbation of EDAP sources during training, bridging the gap between teacher-forcing (GT hidden states) and autoregressive generation.
- **Target-entropy regularization**: keeps cross-depth attention informative without collapsing to single-source or uniform routing.
- **Ablations**: `--shuffle_depth` (randomize block order), `--no_delta`, `--no_gate`, `--no_flip_augmentation`.

## Results

Run `python src/report.py --edap_ckpt <path> --edap_random_ckpt <path>` after training to generate a full comparison report with:

- EM / Prefix-EM breakdown by method (EDAP, EDAP-random, Greedy, CAD, DoLa) and dataset (ConFiQA, NQ-Swap)
- Per-source-type analysis (context vs. memory)
- Cross-depth attention heatmaps (EDAP vs. EDAP-random)
- Failure case analysis (EDAP-wrong / Greedy-right reversals)
- Success criteria checklist (against `prototype-experiment.md`)

Results are saved to `results/` as JSON + Markdown report + attention heatmap.

## License

MIT
