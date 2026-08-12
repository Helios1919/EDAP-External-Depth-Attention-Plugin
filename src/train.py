"""EDAP training on ConFiQA with Qwen2.5-7B backbone."""

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from edap_plugin import create_edap_plugins, edap_forward
from data_utils import ConFiQADataset, create_dataloader


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
parser.add_argument("--model_path", default="./models/qwen2.5-7b")
parser.add_argument("--data_path", default="./data/confiqa/confiqa_train.json")
parser.add_argument("--output_dir", default="./checkpoints",
                    help="Default auto-uses /root/autodl-tmp/ if available")
parser.add_argument("--log_dir", default="./logs")
parser.add_argument("--edap_blocks", type=int, default=7,
                    help="Number of EDAP plugins (= block boundaries); 7 → every 4 layers")
parser.add_argument("--edap_heads", type=int, default=8)
parser.add_argument("--edap_dropout", type=float, default=0.1,
                    help="Dropout rate in EDAP plugins")
parser.add_argument("--shared_kv", action="store_true",
                    help="Share W_K/W_V across plugins (reduces params ~1/3)")
parser.add_argument("--block_layers", type=str, default=None,
                    help="Comma-separated block boundary layer indices, e.g. '3,7,11,15,19,23,27'. "
                         "If not set, auto-computed evenly given --edap_blocks.")
parser.add_argument("--no_delta", action="store_true",
                    help="Disable delta attention (use cumulated K instead)")
parser.add_argument("--no_gate", action="store_true",
                    help="Disable gated mixing (hard-replace residual)")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1,
                    help="Per-step batch; A100-40GB safe with 1 (model ~38GB). Increase if using 80GB.")
parser.add_argument("--grad_accum", type=int, default=16,
                    help="effective batch = batch_size * grad_accum (default 16)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Unified learning rate (used for both EDAP and lm_head unless overridden)")
parser.add_argument("--lr_edap", type=float, default=None,
                    help="Learning rate for EDAP plugins (defaults to --lr)")
parser.add_argument("--lr_lm_head", type=float, default=None,
                    help="Learning rate for lm_head (defaults to --lr)")
parser.add_argument("--freeze_lm_head", action="store_true",
                    help="Freeze lm_head and insert a trainable d→d bottleneck; "
                         "prevents the 545M lm_head from memorising dataset biases "
                         "and forces EDAP to learn meaningful routing (recommended).")
parser.add_argument("--warmup_steps", type=int, default=450)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing for cross-entropy (0 = off)")
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of training data held out for validation")
parser.add_argument("--early_stop_patience", type=int, default=2,
                    help="Stop if val loss doesn't improve for N epochs (0 = off)")
parser.add_argument("--lambda_entropy", type=float, default=0.05,
                    help="Target-entropy regularisation on cross-depth attention (0 = off). "
                         "Penalises deviation from target entropy ln(min(N,3)), "
                         "preventing both attention collapse (too low) and uniform routing (too high).")
parser.add_argument("--lambda_gate_reg", type=float, default=0.01,
                    help="Gate L2 regularisation toward 0.5: prevents gate from collapsing "
                         "to 0 (bypass EDAP) or 1 (hard-replace backbone) (0 = off)")
parser.add_argument("--flip_augmentation", action="store_true",
                    help="Enable flipped counterfactual augmentation (opt-in)")
parser.add_argument("--edap_noise", type=float, default=0.02,
                    help="Gaussian noise std on EDAP sources during training; "
                         "mitigates exposure bias from teacher forcing "
                         "(0 = off; recommended 0.01-0.05)")
parser.add_argument("--dataset_types", type=str, default=None,
                    help="Comma-separated types to keep, e.g. 'counterfactual,context_required'. "
                         "If not set, all types are used. Useful for two-phase training.")
parser.add_argument("--shuffle_depth", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--wandb", action="store_true", default=False)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# V100 lacks bf16; fall back to fp16. A100/H100 use bf16.
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Auto-detect data disk for checkpoints (avoid filling system disk)
if args.output_dir == "./checkpoints" and os.path.isdir("/root/autodl-tmp"):
    args.output_dir = "/root/autodl-tmp/checkpoints"

print(f"Device: {device}")
print(f"Compute dtype: {COMPUTE_DTYPE}")
print(f"Mode: {'EDAP-random' if args.shuffle_depth else 'EDAP'}")
print(f"Dry run: {args.dry_run}")

if args.wandb:
    mode_str = "edap_random" if args.shuffle_depth else "edap"
    wandb.init(
        project="edap-prototype",
        name=f"{mode_str}_{args.edap_blocks}blk_{args.edap_heads}h",
        config=vars(args),
    )

# -- model ----------------------------------------------------------

print("Loading model...")
load_src = args.model_path if Path(args.model_path).exists() else args.model_name
model = AutoModelForCausalLM.from_pretrained(
    load_src, torch_dtype=COMPUTE_DTYPE,
    device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(load_src or args.model_name)
tokenizer.pad_token = tokenizer.eos_token

# keep lm_head trainable so EDAP has a gradient pathway;
# everything else frozen
print("Freezing backbone...")
for name, p in model.named_parameters():
    p.requires_grad = "lm_head" in name

# Optional: freeze lm_head too and insert a small trainable bottleneck.
# This prevents the 545M lm_head from memorising dataset biases and
# forces EDAP to learn meaningful cross-depth routing.
lm_head_bottleneck = None
if args.freeze_lm_head:
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False
    lm_head_bottleneck = nn.Linear(
        model.config.hidden_size, model.config.hidden_size, bias=False,
    ).to(device).to(COMPUTE_DTYPE)
    nn.init.normal_(lm_head_bottleneck.weight, mean=0.0, std=0.02)
    print(f"lm_head FROZEN — inserted trainable {model.config.hidden_size}→{model.config.hidden_size} bottleneck "
          f"({sum(p.numel() for p in lm_head_bottleneck.parameters()):,} params)")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable / total: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# -- resolve block boundaries -----------------------------------------

n_total_layers = len(model.model.layers)
if args.block_layers is not None:
    BLOCK_EXITS = [int(x) for x in args.block_layers.split(",")]
else:
    # auto-compute evenly-spaced exits: e.g. 4 blocks → [6,13,20,27]
    step = n_total_layers // args.edap_blocks
    BLOCK_EXITS = [step * (i + 1) - 1 for i in range(args.edap_blocks)]

for lidx in BLOCK_EXITS:
    assert 0 <= lidx < n_total_layers, f"Block exit {lidx} out of range (0-{n_total_layers-1})"
print(f"Block exits: {BLOCK_EXITS}")

# -- EDAP plugins ---------------------------------------------------

edap_plugins = create_edap_plugins(
    d_model=model.config.hidden_size,
    n_heads=args.edap_heads,
    n_blocks=args.edap_blocks,
    dropout=args.edap_dropout,
    shared_kv=args.shared_kv,
).to(device).to(COMPUTE_DTYPE)

n_edap_params = sum(p.numel() for p in edap_plugins.parameters())
print(f"EDAP params: {n_edap_params:,}")

# -- data -----------------------------------------------------------

print("Loading data...")
train_dataset = ConFiQADataset(
    data_path=args.data_path, split="train",
    max_samples=100 if args.dry_run else None,
    augment_counterfactual=args.flip_augmentation,
    tokenizer=tokenizer, max_seq_length=args.max_seq_length, seed=42,
)

# ---- validation split (stratified by correct_source) ----
if args.val_split > 0:
    from collections import defaultdict
    # group indices by correct_source for stratified split
    source_to_indices = defaultdict(list)
    for i, s in enumerate(train_dataset.samples):
        src = s.get("correct_source", "unknown")
        source_to_indices[src].append(i)

    train_indices, val_indices = [], []
    rng = torch.Generator().manual_seed(42)
    for src, indices in source_to_indices.items():
        n_val_src = max(1, int(len(indices) * args.val_split))
        perm = torch.randperm(len(indices), generator=rng).tolist()
        val_indices.extend([indices[p] for p in perm[:n_val_src]])
        train_indices.extend([indices[p] for p in perm[n_val_src:]])
        print(f"  {src}: {len(indices) - n_val_src} train / {n_val_src} val")

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    train_loader = create_dataloader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_indices)}  Val: {len(val_indices)}")
else:
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    print(f"Train samples: {len(train_dataset)} (no val split)")

# -- optimizer -------------------------------------------------------

lr_edap = args.lr_edap if args.lr_edap is not None else args.lr
lr_lm_head = args.lr_lm_head if args.lr_lm_head is not None else args.lr
print(f"LR: edap={lr_edap:.1e}  lm_head={lr_lm_head:.1e}")

edap_params = list(edap_plugins.parameters())
lm_head_params = [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad]
all_trainable = edap_params + lm_head_params

param_groups = [
    {"params": edap_params, "lr": lr_edap},
    {"params": lm_head_params, "lr": lr_lm_head},
]
if lm_head_bottleneck is not None:
    bottleneck_params = list(lm_head_bottleneck.parameters())
    all_trainable += bottleneck_params
    param_groups.append({"params": bottleneck_params, "lr": lr_edap})
    print(f"Bottleneck params: {sum(p.numel() for p in bottleneck_params):,}")

optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
total_steps = len(train_loader) * args.epochs // args.grad_accum

# linear warmup → cosine decay
def _lr_lambda(step):
    if step < args.warmup_steps:
        return step / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, _lr_lambda)

print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} steps/epoch\n")
global_step = 0
best_val_loss = float("inf")
patience_counter = 0
best_epoch = 0
model.eval()  # no dropout in frozen backbone

for epoch in range(args.epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        # EDAP-interleaved forward (backbone frozen, EDAP gets gradients)
        logits, all_weights, all_gates = edap_forward(
            model, input_ids, attn_mask, edap_plugins,
            BLOCK_EXITS, COMPUTE_DTYPE,
            shuffle_depth=args.shuffle_depth,
            delta_mode=not args.no_delta,
            gate_mode=not args.no_gate,
            edap_noise=args.edap_noise,
            collect_weights=True,
            lm_head_bottleneck=lm_head_bottleneck,
        )  # logits: [B, S, V]; all_weights: list of [B, S, H, N]

        # ---- entropy regularisation ----
        if args.lambda_entropy > 0:
            entropy_loss = torch.tensor(0.0, device=device)
            for w in all_weights:
                # w: [B, S, H, N]
                ent = -(w * torch.log(w + 1e-8)).sum(dim=-1)  # [B, S, H]
                N = w.size(-1)
                target_H = math.log(min(N, 3))  # encourage focusing on ~3 sources
                entropy_loss = entropy_loss + ((ent - target_H) ** 2).mean()
            entropy_loss = entropy_loss / len(all_weights)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        # ---- gate regularisation ----
        if args.lambda_gate_reg > 0:
            gate_reg = torch.tensor(0.0, device=device)
            for g in all_gates:
                gate_reg = gate_reg + ((g - 0.5) ** 2).mean()
            gate_reg = gate_reg / len(all_gates)
        else:
            gate_reg = torch.tensor(0.0, device=device)

        # Next-token prediction shift
        # Cast to fp32 for CE — bf16 logits can overflow in log_softmax
        # when bottleneck amplifies hidden states, producing inf→NaN.
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=args.label_smoothing,
        )
        loss = (ce_loss
                + args.lambda_entropy * entropy_loss
                + args.lambda_gate_reg * gate_reg) / args.grad_accum

        # NaN guard: if CE overflows despite fp32 (extremely rare), or if
        # entropy/gate regularisation produces NaN, skip this micro-batch
        # entirely to prevent gradient corruption from spreading.
        # NOTE: do NOT zero_grad() here — if NaN occurs mid-cycle, valid
        # gradients from earlier micro-batches should be preserved.
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] NaN/Inf loss at epoch {epoch+1} step {step+1} — skipping micro-batch")
            epoch_loss += float('nan')
            global_step += 1
            continue

        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            # Clip gradients; skip step if any gradient is non-finite
            # (belt-and-suspenders — fp32 CE should prevent this, but
            #  EDAP delta attention can still produce NaN grads in bf16.)
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                print(f"[WARN] Non-finite grad norm at epoch {epoch+1} step {step+1} — skipping optimizer step")
            optimizer.zero_grad()

        epoch_loss += loss.item() * args.grad_accum
        global_step += 1

        if global_step % 50 == 0:
            avg_loss = epoch_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]

            # Gate stats: detect if gate collapses to 0 or 1
            gate_mean = torch.stack([g.mean() for g in all_gates]).mean().item()
            gate_std = torch.stack([g.std() for g in all_gates]).mean().item()

            # Attention entropy per plugin: detect routing collapse
            attn_ents = []
            for w in all_weights:
                ent = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean().item()
                attn_ents.append(ent)
            avg_attn_ent = sum(attn_ents) / len(attn_ents)

            print(f"Epoch {epoch+1} step {global_step} | loss {avg_loss:.4f} lr {lr:.2e} | gate μ={gate_mean:.3f} σ={gate_std:.3f} | attn_H={avg_attn_ent:.3f} | ent={entropy_loss.item():.4f} gate_reg={gate_reg.item():.4f}")
            if args.wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/gate_mean": gate_mean,
                    "train/gate_std": gate_std,
                    "train/attn_entropy": avg_attn_ent,
                    "train/ce_loss": ce_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "train/gate_reg": gate_reg.item(),
                }, step=global_step)

        if args.dry_run and global_step >= 3:
            print("Dry run done.")
            break

    if args.dry_run:
        break

    print(f"=== Epoch {epoch+1} done, avg loss {epoch_loss / len(train_loader):.4f} ===")

    # ---- validation ----
    if val_loader is not None:
        model.eval()
        edap_plugins.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attn_mask = batch["attention_mask"].to(device)

                logits = edap_forward(
                    model, input_ids, attn_mask, edap_plugins,
                    BLOCK_EXITS, COMPUTE_DTYPE, shuffle_depth=args.shuffle_depth,
                    delta_mode=not args.no_delta,
                    gate_mode=not args.no_gate,
                    collect_weights=False,
                    lm_head_bottleneck=lm_head_bottleneck,
                )  # [B, S, V] — backbone inside no_grad
                # Cast to fp32 for stable CE (mirrors training fix)
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=args.label_smoothing,
                )
                if torch.isfinite(loss):
                    val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"=== Val loss: {val_loss:.4f} ===")
        if args.wandb:
            wandb.log({"val/loss": val_loss}, step=global_step)

        # ---- early stopping ----
        if args.early_stop_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                print(f"Val loss did not improve ({patience_counter}/{args.early_stop_patience})")

        # ---- save checkpoint ----
        if val_loader is not None:
            is_best = (val_loss == best_val_loss)
        else:
            # No validation split — save every epoch
            is_best = True
        if is_best:
            os.makedirs(args.output_dir, exist_ok=True)
            tag = "edap_random" if args.shuffle_depth else "edap"
            ckpt_path = Path(args.output_dir) / f"{tag}_best.pt"
            # Remove old best checkpoint if exists
            for old in Path(args.output_dir).glob(f"{tag}_best*.pt"):
                old.unlink()
            ckpt_dict = {
                "epoch": epoch + 1,
                "edap_plugins": edap_plugins.state_dict(),
                "val_loss": val_loss,
                "global_step": global_step,
                "config": vars(args),
            }
            # Save trainable lm_head params (or bottleneck if lm_head is frozen)
            if lm_head_bottleneck is not None:
                ckpt_dict["lm_head_bottleneck"] = lm_head_bottleneck.state_dict()
            else:
                ckpt_dict["lm_head"] = {n: p.clone() for n, p in model.named_parameters()
                            if "lm_head" in n and p.requires_grad}
            torch.save(ckpt_dict, ckpt_path)
            print(f"Best checkpoint -> {ckpt_path} (val_loss={val_loss:.4f})")

            # Also save a resume checkpoint with optimizer state
            resume_path = Path(args.output_dir) / f"{tag}_resume.pt"
            resume_dict = {
                "epoch": epoch + 1,
                "edap_plugins": edap_plugins.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
                "global_step": global_step,
                "config": vars(args),
            }
            if lm_head_bottleneck is not None:
                resume_dict["lm_head_bottleneck"] = lm_head_bottleneck.state_dict()
            else:
                resume_dict["lm_head"] = {n: p.clone() for n, p in model.named_parameters()
                            if "lm_head" in n and p.requires_grad}
            torch.save(resume_dict, resume_path)

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

        edap_plugins.train()

if args.wandb:
    wandb.finish()
print("Done.")
