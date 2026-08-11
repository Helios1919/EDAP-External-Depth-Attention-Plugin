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

from edap_plugin import create_edap_plugins
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
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8,
                    help="A100 default 8; reduce to 2 for V100-32GB")
parser.add_argument("--grad_accum", type=int, default=2,
                    help="effective batch = batch_size * grad_accum (16 for A100)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_steps", type=int, default=450)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing for cross-entropy (0 = off)")
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of training data held out for validation")
parser.add_argument("--early_stop_patience", type=int, default=2,
                    help="Stop if val loss doesn't improve for N epochs (0 = off)")
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

# -- hooks: grab hidden states at block boundaries -------------------

# Qwen2.5-7B has 28 layers, split into 4 blocks of 7
BLOCK_EXITS = [6, 13, 20, 27]
block_exits: list = []


def _capture_hook(module, inp, out):
    # out is a tuple for most HF implementations
    block_exits.append(out[0].detach())


hooks = []
for lidx in BLOCK_EXITS:
    h = model.model.layers[lidx].register_forward_hook(_capture_hook)
    hooks.append(h)
print(f"{len(hooks)} hooks registered at layers {BLOCK_EXITS}")

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
    augment_counterfactual=True,
    tokenizer=tokenizer, max_seq_length=args.max_seq_length,
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

edap_params = list(edap_plugins.parameters())
lm_head_params = [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad]
all_trainable = edap_params + lm_head_params

optimizer = AdamW(all_trainable, lr=args.lr, weight_decay=args.weight_decay)
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

        # forward through frozen backbone, hooks collect block exits
        block_exits.clear()
        with torch.no_grad():
            emb = model.model.embed_tokens(input_ids)
            _ = model.model(inputs_embeds=emb, attention_mask=attn_mask)

        # EDAP chain
        r_prev = [emb.detach().to(COMPUTE_DTYPE)]
        for r_blk, plug in zip(block_exits, edap_plugins):
            sources = r_prev + [r_blk.to(COMPUTE_DTYPE)]
            r_fused, _ = plug(sources, shuffle_depth=args.shuffle_depth)
            r_prev.append(r_fused)

        # Full-sequence teacher-forcing loss over all answer tokens
        # r_prev[-1]: [B, S, d] → lm_head → [B, S, V]
        logits = model.lm_head(r_prev[-1])  # [B, S, vocab]
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

                block_exits.clear()
                emb = model.model.embed_tokens(input_ids)
                _ = model.model(inputs_embeds=emb, attention_mask=attn_mask)

                r_prev = [emb.detach().to(COMPUTE_DTYPE)]
                for r_blk, plug in zip(block_exits, edap_plugins):
                    sources = r_prev + [r_blk.to(COMPUTE_DTYPE)]
                    r_fused, _ = plug(sources, shuffle_depth=args.shuffle_depth)
                    r_prev.append(r_fused)

                logits = model.lm_head(r_prev[-1])
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

        edap_plugins.train()

for h in hooks:
    h.remove()
if args.wandb:
    wandb.finish()
print("Done.")
