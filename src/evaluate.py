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


# ---------------------------------------------------------------
# evaluation runners

def run_edap(samples, model, tokenizer, edap_plugins, shuffle_depth=False,
             return_attn=False):
    block_exits = []

    def _hook(m, inp, out):
        block_exits.append(out[0].detach())

    exits = [6, 13, 20, 27]  # Qwen2.5-7B: 28 layers / 4 blocks
    handles = []
    for i in exits:
        handles.append(model.model.layers[i].register_forward_hook(_hook))

    # incremental attention stats: keyed by (plugin_idx, source_type)
    # each entry: (sum, sum_of_squares, count)
    attn_stats = {} if return_attn else None

    results = []
    for s in tqdm(samples, desc="EDAP"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)

        block_exits.clear()
        with torch.no_grad():
            emb = model.model.embed_tokens(inp["input_ids"])
            _ = model.model(inputs_embeds=emb, attention_mask=inp["attention_mask"])

        r_prev = [emb.to(COMPUTE_DTYPE)]
        for pi, (r_blk, plug) in enumerate(zip(block_exits, edap_plugins)):
            sources = r_prev + [r_blk.to(COMPUTE_DTYPE)]
            r_fused, attn_w = plug(sources, shuffle_depth=shuffle_depth)
            r_prev.append(r_fused)

            if return_attn:
                # reduce [1, S, H, n_sources] → [n_sources] (mean over B,S,H)
                w = attn_w.detach().mean(dim=(0, 1, 2)).cpu()
                stype = s.get("correct_source", "unknown")
                key = (pi, stype)
                if key not in attn_stats:
                    attn_stats[key] = (torch.zeros_like(w),
                                       torch.zeros_like(w),
                                       0.0)
                s_sum, s_sq, cnt = attn_stats[key]
                attn_stats[key] = (s_sum + w, s_sq + w * w, cnt + 1.0)

        logits = model.lm_head(r_prev[-1][0, -1, :])
        pred_id = torch.argmax(logits, dim=-1)
        pred = tokenizer.decode([pred_id.item()])

        gt = s.get("correct_answer", "")
        em = int(normalize_answer(pred) == normalize_answer(gt))
        results.append({
            "pred": normalize_answer(pred),
            "gt": normalize_answer(gt),
            "correct_source": s.get("correct_source", "unknown"),
            "em": em,
        })

    for h in handles:
        h.remove()

    if return_attn:
        # compile incremental stats into summary dict
        summary = _build_attn_summary(attn_stats, len(edap_plugins))
        return results, summary
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


def run_greedy(samples, model, tokenizer):
    results = []
    for s in tqdm(samples, desc="Greedy"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp)
            pred_id = torch.argmax(out.logits[0, -1, :], dim=-1)
        pred = tokenizer.decode([pred_id.item()])
        gt = s.get("correct_answer", "")
        results.append({
            "pred": normalize_answer(pred),
            "gt": normalize_answer(gt),
            "correct_source": s.get("correct_source", "unknown"),
            "em": int(normalize_answer(pred) == normalize_answer(gt)),
        })
    return results


def run_cad(samples, model, tokenizer, alpha=1.0):
    """CAD (Shi et al. 2023): logit_CAD = (1+α) * logit_ctx+q − α * logit_q"""
    results = []
    for s in tqdm(samples, desc="CAD"):
        prompt_ctx = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        prompt_no_ctx = f"Question: {s['question']}\n\nAnswer:"

        with torch.no_grad():
            out_ctx = model(**tokenizer(prompt_ctx, return_tensors="pt").to(model.device))
            out_null = model(**tokenizer(prompt_no_ctx, return_tensors="pt").to(model.device))
            logits = (1 + alpha) * out_ctx.logits[0, -1, :] - alpha * out_null.logits[0, -1, :]
            pred_id = torch.argmax(logits, dim=-1)
        pred = tokenizer.decode([pred_id.item()])

        gt = s.get("correct_answer", "")
        results.append({
            "pred": normalize_answer(pred),
            "gt": normalize_answer(gt),
            "correct_source": s.get("correct_source", "unknown"),
            "em": int(normalize_answer(pred) == normalize_answer(gt)),
        })
    return results


def run_dola(samples, model, tokenizer, early_exit=13):
    """DoLa (Chuang et al., ICLR 2024) via logit contrasting.

    Gets logits from an early layer (~midpoint for Qwen2.5-7B's 28 layers)
    by patching the hidden state through lm_head.  Subtracts early logits
    from final logits to amplify mature layer knowledge.
    """
    early_layer = model.model.layers[early_exit]

    def _early_hook(m, inp, out):
        _early_hook.state = out[0].detach()

    _early_hook.state = None
    handle = early_layer.register_forward_hook(_early_hook)

    results = []
    for s in tqdm(samples, desc="DoLa"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)

        _early_hook.state = None
        with torch.no_grad():
            out = model(**inp)
            logits_final = out.logits[0, -1, :]

        if _early_hook.state is not None:
            early_hidden = _early_hook.state[0, -1, :].float()
            early_hidden = early_hidden.to(logits_final.dtype)
            early_hidden = model.model.norm(early_hidden)
            logits_early = model.lm_head(early_hidden)
            logits = logits_final - logits_early
        else:
            logits = logits_final

        pred_id = torch.argmax(logits, dim=-1)
        pred = tokenizer.decode([pred_id.item()])

        gt = s.get("correct_answer", "")
        results.append({
            "pred": normalize_answer(pred),
            "gt": normalize_answer(gt),
            "correct_source": s.get("correct_source", "unknown"),
            "em": int(normalize_answer(pred) == normalize_answer(gt)),
        })

    handle.remove()
    return results


def summarize(results, method, ds_name, output_dir):
    if not results:
        return
    em = sum(r["em"] for r in results) / len(results) * 100
    print(f"\n{method} on {ds_name}: EM = {em:.2f}%")

    by_source = {}
    for r in results:
        src = r["correct_source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(r["em"])
    for src, ems in sorted(by_source.items()):
        print(f"  {src}: EM = {sum(ems)/len(ems)*100:.2f}% ({len(ems)} samples)")

    out_path = Path(output_dir) / f"{ds_name}_{method}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "method": method, "dataset": ds_name,
            "em": em, "n_samples": len(results), "results": results,
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
        edap_plugins = create_edap_plugins(
            d_model=model.config.hidden_size, n_heads=4, n_blocks=4,
        ).to(device).to(COMPUTE_DTYPE)

        ckpt = torch.load(args.checkpoint, map_location=device)
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
