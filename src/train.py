"""EDAP training on ConFiQA with Qwen2.5-7B backbone."""

import os
import math
import argparse
from pathlib import Path

import torch
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
parser.add_argument("--edap_blocks", type=int, default=4)
parser.add_argument("--edap_heads", type=int, default=8)
parser.add_argument("--edap_dropout", type=float, default=0.1,
                    help="Dropout rate in EDAP plugins")
parser.add_argument("--block_layers", type=str, default=None,
                    help="Comma-separated block boundary layer indices, e.g. '6,13,20,27'. "
                         "If not set, auto-computed evenly from 28 layers given --edap_blocks.")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8,
                    help="A100 default 8; reduce to 2 for V100-32GB")
parser.add_argument("--grad_accum", type=int, default=2,
                    help="effective batch = batch_size * grad_accum (16 for A100)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Unified learning rate (used for both EDAP and lm_head unless overridden)")
parser.add_argument("--lr_edap", type=float, default=None,
                    help="Learning rate for EDAP plugins (defaults to --lr)")
parser.add_argument("--lr_lm_head", type=float, default=None,
                    help="Learning rate for lm_head (defaults to --lr)")
parser.add_argument("--warmup_steps", type=int, default=450)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing for cross-entropy (0 = off)")
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of training data held out for validation")
parser.add_argument("--early_stop_patience", type=int, default=2,
                    help="Stop if val loss doesn't improve for N epochs (0 = off)")
parser.add_argument("--no_flip_augmentation", action="store_true",
                    help="Disable flipped counterfactual augmentation (ablation)")
parser.add_argument("--shuffle_depth", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--wandb", action="store_true", default=False)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# V100 不支持 bf16，自动回退到 fp16；A100/H100 则用 bf16
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
).to(device).to(COMPUTE_DTYPE)

n_edap_params = sum(p.numel() for p in edap_plugins.parameters())
print(f"EDAP params: {n_edap_params:,}")

# -- data -----------------------------------------------------------

print("Loading data...")
train_dataset = ConFiQADataset(
    data_path=args.data_path, split="train",
    max_samples=100 if args.dry_run else None,
    augment_counterfactual=not args.no_flip_augmentation,
    tokenizer=tokenizer, max_seq_length=args.max_seq_length, seed=42,
)

# ---- validation split ----
if args.val_split > 0:
    n_val = int(len(train_dataset) * args.val_split)
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = create_dataloader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Train: {n_train}  Val: {n_val}")
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

optimizer = AdamW([
    {"params": edap_params, "lr": lr_edap},
    {"params": lm_head_params, "lr": lr_lm_head},
], lr=args.lr, weight_decay=args.weight_decay)
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

        # EDAP-interleaved forward: backbone runs in no_grad, EDAP
        # fuses at each block boundary and injects back into residual stream
        logits = edap_forward(
            model, input_ids, attn_mask, edap_plugins,
            BLOCK_EXITS, COMPUTE_DTYPE, shuffle_depth=args.shuffle_depth,
        )  # [B, S, V]
        # Shift: predict token[t] from hidden[t-1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=args.label_smoothing,
        )
        loss = loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * args.grad_accum
        global_step += 1

        if global_step % 50 == 0:
            avg_loss = epoch_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1} step {global_step} | loss {avg_loss:.4f} lr {lr:.2e}")
            if args.wandb:
                wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)

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
                )  # [B, S, V] — edap_forward uses no_grad internally for backbone
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=args.label_smoothing,
                )
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

        # ---- save checkpoint (only best model, no optimizer to save space) ----
        is_best = (val_loss == best_val_loss)
        if is_best and val_loader is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            tag = "edap_random" if args.shuffle_depth else "edap"
            ckpt_path = Path(args.output_dir) / f"{tag}_best.pt"
            # Remove old best checkpoint if exists
            for old in Path(args.output_dir).glob(f"{tag}_best*.pt"):
                old.unlink()
            torch.save({
                "epoch": epoch + 1,
                "edap_plugins": edap_plugins.state_dict(),
                "lm_head": {n: p.clone() for n, p in model.named_parameters()
                            if "lm_head" in n and p.requires_grad},
                "val_loss": val_loss,
                "global_step": global_step,
                "config": vars(args),
            }, ckpt_path)
            print(f"Best checkpoint -> {ckpt_path} (val_loss={val_loss:.4f})")

            # Also save a resume checkpoint with optimizer state
            resume_path = Path(args.output_dir) / f"{tag}_resume.pt"
            torch.save({
                "epoch": epoch + 1,
                "edap_plugins": edap_plugins.state_dict(),
                "lm_head": {n: p.clone() for n, p in model.named_parameters()
                            if "lm_head" in n and p.requires_grad},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
                "global_step": global_step,
                "config": vars(args),
            }, resume_path)

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

        edap_plugins.train()

if args.wandb:
    wandb.finish()
print("Done.")
