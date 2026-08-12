"""Evaluation for EDAP and baselines on knowledge conflict datasets."""

import os
import json
import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from edap_plugin import create_edap_plugins
from data_utils import ConFiQADataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def parse_args():
    p = argparse.ArgumentParser(description="EDAP Evaluation")
    p.add_argument("--model_path", default="./models/qwen2.5-7b")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--shuffle_depth", action="store_true")
    p.add_argument("--data_path", default="./data/confiqa/confiqa_train.json")
    p.add_argument("--nq_swap_path", default="./data/nqswap/nqswap_dev.json")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--baseline", default=None, choices=["greedy", "cad", "dola"])
    return p.parse_args()


def load_eval_data(path, max_samples=0):
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # If the file contains NQ-Swap raw fields, remap them
        if data and "sub_context" in data[0]:
            data = [{
                "question": s["question"],
                "context": s["sub_context"],
                "correct_answer": s["sub_answer"][0] if isinstance(s["sub_answer"], list) else s["sub_answer"],
                "correct_source": "context",
                "original_answer": s.get("org_answer", [""])[0] if isinstance(s.get("org_answer", ""), list) else s.get("org_answer", ""),
            } for s in data]
        # If ConFiQA raw fields (answer instead of correct_answer), remap
        elif data and "answer" in data[0] and "correct_answer" not in data[0]:
            data = [{
                "question": s.get("question", ""),
                "context": s.get("context", ""),
                "correct_answer": s.get("answer", ""),
                "correct_source": s.get("correct_source", "unknown"),
            } for s in data]
    elif path.lower() in ("nq_swap", "nq-swap", "nq_swap_dev"):
        # auto-download from HuggingFace
        from data_utils import load_nq_swap
        data = load_nq_swap(cache_path="./data/nqswap/nqswap_dev.json")
    else:
        ds = ConFiQADataset(path, augment_counterfactual=False)
        data = [{
            "question": s.get("question", ""),
            "context": s.get("context", ""),
            "correct_answer": s.get("answer", ""),
            "correct_source": s.get("correct_source", "unknown"),
        } for s in ds.samples]
    if max_samples > 0:
        data = data[:max_samples]
    return data


def normalize_answer(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_metrics(pred_text, gt_text):
    """Compute both full EM and prefix EM.

    Prefix EM catches cases where the model outputs the correct answer
    but fails to stop (common with CAD/DoLa logit contrast suppressing EOS).
    """
    pred = normalize_answer(pred_text)
    gt = normalize_answer(gt_text)
    em = int(pred == gt)
    em_prefix = int(pred.startswith(gt) and len(gt) > 0)
    return pred, gt, em, em_prefix


# ---------------------------------------------------------------
# evaluation runners

def _generate_edap_answer(model, tokenizer, edap_plugins, prompt, max_new=32):
    """Generate full answer through the EDAP-modified forward path.

    Uses greedy decode with incremental EDAP: each new token goes through
    the frozen backbone + EDAP chain, and lm_head on the last position.
    """
    block_exits = []

    def _hook(m, inp, out):
        block_exits.append(out[0].detach())

    handles = []
    for i in [6, 13, 20, 27]:
        handles.append(model.model.layers[i].register_forward_hook(_hook))

    generated = []
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    for _ in range(max_new):
        block_exits.clear()
        with torch.no_grad():
            emb = model.model.embed_tokens(input_ids)
            _ = model.model(inputs_embeds=emb, attention_mask=None)

        r_prev = [emb.detach().to(COMPUTE_DTYPE)]
        for r_blk, plug in zip(block_exits, edap_plugins):
            sources = r_prev + [r_blk.to(COMPUTE_DTYPE)]
            r_fused, _ = plug(sources, shuffle_depth=False)
            r_prev.append(r_fused)

        next_logits = model.lm_head(r_prev[-1][0, -1, :])
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated.append(next_id.item())

        if next_id.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    for h in handles:
        h.remove()

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_edap(samples, model, tokenizer, edap_plugins, shuffle_depth=False,
             return_attn=False):
    """Evaluate EDAP by generating full answers, not single-token argmax."""

    results = []
    for s in tqdm(samples, desc="EDAP"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        pred_text = _generate_edap_answer(model, tokenizer, edap_plugins, prompt)

        gt = s.get("correct_answer", "")
        pred_norm, gt_norm, em, em_prefix = compute_metrics(pred_text, gt)
        results.append({
            "pred": pred_norm,
            "gt": gt_norm,
            "correct_source": s.get("correct_source", "unknown"),
            "em": em,
            "em_prefix": em_prefix,
        })

    if return_attn:
        # Attention stats collection not yet implemented for generation mode;
        # fall back to returning results only.
        pass
    return results


def _build_attn_summary(attn_stats, n_plugins):
    """Convert incremental (sum, sum_sq, count) tuples to {mean, std} dicts."""
    # aggregate across source types for per-plugin view
    plugin_sums = [None] * n_plugins
    plugin_sqs = [None] * n_plugins
    plugin_cnts = [0.0] * n_plugins

    per_source_type = {}

    for (pi, stype), (s_sum, s_sq, cnt) in attn_stats.items():
        mean = (s_sum / cnt).tolist()
        variance = (s_sq / cnt - (s_sum / cnt) ** 2).clamp(min=0.0)
        std = variance.sqrt().tolist()

        if stype not in per_source_type:
            per_source_type[stype] = {}
        per_source_type[stype][f"edap_{pi}"] = {"mean": mean, "std": std, "n": int(cnt)}

        # accumulate for overall per-plugin
        if plugin_sums[pi] is None:
            plugin_sums[pi] = s_sum.clone()
            plugin_sqs[pi] = s_sq.clone()
        else:
            plugin_sums[pi] += s_sum
            plugin_sqs[pi] += s_sq
        plugin_cnts[pi] += cnt

    per_plugin = {}
    for pi in range(n_plugins):
        if plugin_sums[pi] is not None:
            mean = (plugin_sums[pi] / plugin_cnts[pi]).tolist()
            variance = (plugin_sqs[pi] / plugin_cnts[pi] -
                        (plugin_sums[pi] / plugin_cnts[pi]) ** 2).clamp(min=0.0)
            std = variance.sqrt().tolist()
            per_plugin[f"edap_{pi}"] = {"mean": mean, "std": std, "n": int(plugin_cnts[pi])}
        else:
            per_plugin[f"edap_{pi}"] = {"mean": [], "std": [], "n": 0}

    return {
        "per_plugin": per_plugin,
        "per_source_type": per_source_type,
    }


def _generate_greedy(model, tokenizer, prompt, max_new=32):
    """Greedy decode using frozen Qwen backbone (no EDAP)."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_greedy(samples, model, tokenizer):
    results = []
    for s in tqdm(samples, desc="Greedy"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        pred = _generate_greedy(model, tokenizer, prompt)
        gt = s.get("correct_answer", "")
        pred_norm, gt_norm, em, em_prefix = compute_metrics(pred, gt)
        results.append({
            "pred": pred_norm,
            "gt": gt_norm,
            "correct_source": s.get("correct_source", "unknown"),
            "em": em,
            "em_prefix": em_prefix,
        })
    return results


def _generate_cad_answer(model, tokenizer, prompt_ctx, prompt_no_ctx, max_new=32, alpha=1.0):
    """Greedy decode with CAD logit contrasting at each step."""
    input_ids_ctx = tokenizer(prompt_ctx, return_tensors="pt")["input_ids"].to(model.device)
    input_ids_no = tokenizer(prompt_no_ctx, return_tensors="pt")["input_ids"].to(model.device)
    generated = []

    for _ in range(max_new):
        with torch.no_grad():
            out_ctx = model(input_ids_ctx)
            out_no = model(input_ids_no)
            logits = (1 + alpha) * out_ctx.logits[0, -1, :] - alpha * out_no.logits[0, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_id.item())
        if next_id.item() == tokenizer.eos_token_id:
            break
        input_ids_ctx = torch.cat([input_ids_ctx, next_id.unsqueeze(0)], dim=1)
        input_ids_no = torch.cat([input_ids_no, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_cad(samples, model, tokenizer, alpha=1.0):
    """CAD (Shi et al. 2023): generation with logit contrasting."""
    results = []
    for s in tqdm(samples, desc="CAD"):
        prompt_ctx = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        prompt_no_ctx = f"Question: {s['question']}\n\nAnswer:"
        pred = _generate_cad_answer(model, tokenizer, prompt_ctx, prompt_no_ctx, alpha=alpha)
        gt = s.get("correct_answer", "")
        pred_norm, gt_norm, em, em_prefix = compute_metrics(pred, gt)
        results.append({
            "pred": pred_norm,
            "gt": gt_norm,
            "correct_source": s.get("correct_source", "unknown"),
            "em": em,
            "em_prefix": em_prefix,
        })
    return results


def run_dola(samples, model, tokenizer, early_exit=13):
    """DoLa (Chuang et al., ICLR 2024) — generation with layer contrasting."""

    early_layer = model.model.layers[early_exit]

    def _early_hook(m, inp, out):
        _early_hook.state = out[0].detach()

    _early_hook.state = None
    handle = early_layer.register_forward_hook(_early_hook)

    results = []
    for s in tqdm(samples, desc="DoLa"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        generated = []

        for _ in range(32):
            _early_hook.state = None
            with torch.no_grad():
                out = model(input_ids)
                logits_final = out.logits[0, -1, :]

            if _early_hook.state is not None:
                early_hidden = _early_hook.state[0, -1, :].to(COMPUTE_DTYPE)
                early_hidden = model.model.norm(early_hidden)
                logits_early = model.lm_head(early_hidden.to(COMPUTE_DTYPE)).float()
                logits = logits_final.float() - logits_early
            else:
                logits = logits_final

            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

        pred = tokenizer.decode(generated, skip_special_tokens=True)

        gt = s.get("correct_answer", "")
        pred_norm, gt_norm, em, em_prefix = compute_metrics(pred, gt)
        results.append({
            "pred": pred_norm,
            "gt": gt_norm,
            "correct_source": s.get("correct_source", "unknown"),
            "em": em,
            "em_prefix": em_prefix,
        })

    handle.remove()
    return results


def summarize(results, method, ds_name, output_dir):
    if not results:
        return
    em = sum(r["em"] for r in results) / len(results) * 100
    em_prefix = sum(r.get("em_prefix", 0) for r in results) / len(results) * 100
    print(f"\n{method} on {ds_name}: EM = {em:.2f}% | Prefix-EM = {em_prefix:.2f}%")

    by_source = {}
    by_source_prefix = {}
    for r in results:
        src = r["correct_source"]
        by_source.setdefault(src, []).append(r["em"])
        by_source_prefix.setdefault(src, []).append(r.get("em_prefix", 0))

    em_by_source = {}
    for src in sorted(by_source):
        src_em = sum(by_source[src]) / len(by_source[src]) * 100
        src_em_prefix = sum(by_source_prefix[src]) / len(by_source_prefix[src]) * 100
        em_by_source[src] = {
            "em": round(src_em, 2),
            "em_prefix": round(src_em_prefix, 2),
            "n": len(by_source[src]),
        }
        print(f"  {src}: EM = {src_em:.2f}% | Prefix-EM = {src_em_prefix:.2f}% ({len(by_source[src])} samples)")

    out_path = Path(output_dir) / f"{ds_name}_{method}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "method": method, "dataset": ds_name,
            "em": em, "em_prefix": em_prefix,
            "n_samples": len(results),
            "em_by_source": em_by_source,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------
# main

if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_path if Path(args.model_path).exists() else args.model_name
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=COMPUTE_DTYPE,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path or args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    edap_plugins = None
    if args.baseline is None and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        n_heads = cfg.get("edap_heads", 8)
        n_blocks = cfg.get("edap_blocks", 4)
        dropout = cfg.get("edap_dropout", 0.1)
        print(f"Checkpoint config: n_heads={n_heads}, n_blocks={n_blocks}, dropout={dropout}")

        edap_plugins = create_edap_plugins(
            d_model=model.config.hidden_size,
            n_heads=n_heads, n_blocks=n_blocks, dropout=dropout,
        ).to(device).to(COMPUTE_DTYPE)

        edap_plugins.load_state_dict(ckpt["edap_plugins"])
        for name, p in model.named_parameters():
            if "lm_head" in name and name in ckpt.get("lm_head", {}):
                p.data.copy_(ckpt["lm_head"][name])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # --- load eval data ---
    eval_sets = {}
    # try local JSON first, then HF download
    nq_path = args.nq_swap_path
    try:
        eval_sets["NQ-Swap"] = load_eval_data(nq_path, args.max_samples)
    except Exception as e:
        print(f"  [warn] NQ-Swap not available: {e}")
    try:
        eval_sets["ConFiQA"] = load_eval_data(args.data_path, args.max_samples)
    except Exception as e:
        print(f"  [warn] ConFiQA not available: {e}")

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name, samples in eval_sets.items():
        if not samples:
            continue
        print(f"\n{'='*50}\n{ds_name} ({len(samples)} samples)\n{'='*50}")

        if args.baseline:
            runners = {
                "greedy": run_greedy,
                "cad": run_cad,
                "dola": run_dola,
            }
            res = runners[args.baseline](samples, model, tokenizer)
            method = args.baseline
        else:
            shuffle = args.shuffle_depth
            res = run_edap(samples, model, tokenizer, edap_plugins, shuffle_depth=shuffle)
            method = "edap_random" if shuffle else "edap"

        summarize(res, method, ds_name, args.output_dir)

    print("\nDone.")
