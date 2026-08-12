"""Generate comparison report for EDAP prototype experiment.

Runs all 5 methods (Greedy, CAD, DoLa, EDAP, EDAP-random) across both
datasets (NQ-Swap, ConFiQA), then produces a comprehensive markdown report
with success-criteria verdicts, attention analysis, and failure-case review.

Usage:
    python src/report.py \
        --edap_ckpt checkpoints/edap_epoch3.pt \
        --edap_random_ckpt checkpoints/edap_random_epoch3.pt \
        --model_path ./models/qwen2.5-7b

    # quick smoke-test (10 samples only):
    python src/report.py ... --max_samples 10

    # skip methods already evaluated (resume):
    python src/report.py ... --resume
"""

import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edap_plugin import create_edap_plugins
from evaluate import (
    run_greedy, run_cad, run_dola, run_edap,
    summarize, normalize_answer, load_eval_data,
)


# --- CLI ---

def parse_args():
    p = argparse.ArgumentParser(description="EDAP Prototype Report Generator")
    p.add_argument("--edap_ckpt", required=True,
                   help="Path to trained EDAP checkpoint")
    p.add_argument("--edap_random_ckpt", required=True,
                   help="Path to trained EDAP-random checkpoint")
    p.add_argument("--model_path", default="./models/qwen2.5-7b")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    p.add_argument("--data_path", default="./data/confiqa/confiqa_test.json")
    p.add_argument("--nq_swap_path", default="./data/nqswap/nqswap_dev.json")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit eval samples (0 = all)")
    p.add_argument("--resume", action="store_true",
                   help="Skip methods whose result JSON already exists")
    p.add_argument("--no_heatmap", action="store_true",
                   help="Skip attention heatmap generation")
    return p.parse_args()


# --- helpers ---

def _result_path(output_dir, dataset, method):
    safe = dataset.replace("-", "_")
    return Path(output_dir) / f"{safe}_{method}.json"


def _load_json_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- evaluation ---

METHODS = ["greedy", "cad", "dola", "edap", "edap_random"]


def load_or_run_method(method, dataset_name, samples, args, model, tokenizer,
                       edap_plugins, edap_random_plugins):
    """Run a single method on a single dataset, with resume support."""
    out_path = _result_path(args.output_dir, dataset_name, method)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume and out_path.exists():
        print(f"  [skip] {dataset_name}/{method} — JSON exists")
        return _load_json_results(out_path)

    print(f"  [run]  {dataset_name}/{method}")

    if method == "greedy":
        results = run_greedy(samples, model, tokenizer)
        attn_summary = None
    elif method == "cad":
        results = run_cad(samples, model, tokenizer)
        attn_summary = None
    elif method == "dola":
        results = run_dola(samples, model, tokenizer)
        attn_summary = None
    elif method == "edap":
        results, attn_summary = run_edap(
            samples, model, tokenizer, edap_plugins,
            shuffle_depth=False, return_attn=True,
        )
    elif method == "edap_random":
        results, attn_summary = run_edap(
            samples, model, tokenizer, edap_random_plugins,
            shuffle_depth=True, return_attn=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # save raw results
    summarize(results, method, dataset_name, args.output_dir)

    return {
        "method": method,
        "dataset": dataset_name,
        "n_samples": len(results),
        "results": results,
        "attn_summary": attn_summary,
    }


def run_all_evaluations(args, model, tokenizer, datasets, edap_plugins,
                        edap_random_plugins):
    """Run all method × dataset combinations and return collected results."""
    all_results = {}  # key: (dataset, method)
    attn_data = {}    # key: (dataset, method) — only for EDAP variants

    for ds_name, samples in datasets.items():
        if not samples:
            continue
        print(f"\n{'='*50}\n{ds_name} ({len(samples)} samples)\n{'='*50}")

        for method in METHODS:
            result = load_or_run_method(
                method, ds_name, samples, args, model, tokenizer,
                edap_plugins, edap_random_plugins,
            )

            all_results[(ds_name, method)] = result
            if result.get("attn_summary"):
                attn_data[(ds_name, method)] = result["attn_summary"]

    return all_results, attn_data


# --- success criteria ---

def evaluate_success_criteria(all_results, attn_data):
    """Check all success criteria against prototype-experiment.md."""

    def _em(ds, method):
        r = all_results.get((ds, method), {})
        res_list = r.get("results", [])
        if not res_list:
            return 0.0
        return sum(x["em"] for x in res_list) / len(res_list) * 100

    nq_edap = _em("NQ-Swap", "edap")
    nq_dola = _em("NQ-Swap", "dola")
    nq_random = _em("NQ-Swap", "edap_random")
    nq_greedy = _em("NQ-Swap", "greedy")
    nq_cad = _em("NQ-Swap", "cad")

    primary = {
        "edap_vs_dola": {
            "pass": nq_edap > nq_dola,
            "edap_em": round(nq_edap, 2),
            "baseline_em": round(nq_dola, 2),
            "delta": round(nq_edap - nq_dola, 2),
            "label": "EDAP > DoLa on NQ-Swap",
        },
        "edap_vs_random": {
            "pass": nq_edap > nq_random,
            "edap_em": round(nq_edap, 2),
            "baseline_em": round(nq_random, 2),
            "delta": round(nq_edap - nq_random, 2),
            "label": "EDAP > EDAP-random on NQ-Swap",
        },
    }
    primary_pass = all(c["pass"] for c in primary.values())

    secondary = {
        "edap_vs_greedy": {
            "pass": nq_edap > nq_greedy,
            "edap_em": round(nq_edap, 2),
            "baseline_em": round(nq_greedy, 2),
            "delta": round(nq_edap - nq_greedy, 2),
            "label": "EDAP > Greedy",
        },
        "edap_vs_cad": {
            "pass": nq_edap >= nq_cad,
            "edap_em": round(nq_edap, 2),
            "baseline_em": round(nq_cad, 2),
            "delta": round(nq_edap - nq_cad, 2),
            "label": "EDAP >= CAD",
        },
    }

    # soft criteria: attention analysis
    soft = {}
    edap_attn = attn_data.get(("NQ-Swap", "edap"), {})
    if edap_attn:
        per_plugin = edap_attn.get("per_plugin", {})
        # check non-uniformity: max - min mean weight across sources per plugin
        ranges = []
        for pname, pdata in per_plugin.items():
            means = pdata.get("mean", [])
            if len(means) >= 2:
                ranges.append(max(means) - min(means))
        avg_range = sum(ranges) / len(ranges) if ranges else 0.0
        soft["attn_nonuniform"] = {
            "pass": avg_range > 0.10,
            "observation": f"Average attention range per plugin: {avg_range:.3f} "
                           f"(> 0.10 → non-uniform)",
        }

        # check scene-awareness: different source types have different peak attention
        per_stype = edap_attn.get("per_source_type", {})
        peaks_by_type = {}
        for stype, plugins in per_stype.items():
            for pname, pdata in plugins.items():
                means = pdata.get("mean", [])
                if means:
                    peak = max(range(len(means)), key=lambda i: means[i])
                    peaks_by_type.setdefault(stype, []).append(peak)
        # scene-aware if at least two source types have different modal peaks
        modal_peaks = {st: max(set(peaks), key=peaks.count)
                       for st, peaks in peaks_by_type.items() if peaks}
        scene_aware = len(set(modal_peaks.values())) >= 2 if len(modal_peaks) >= 2 else False
        soft["scene_aware"] = {
            "pass": scene_aware,
            "observation": f"Modal attention peaks by source type: {modal_peaks}"
                           if modal_peaks else "Insufficient data",
        }
    else:
        soft["attn_nonuniform"] = {"pass": None, "observation": "No attention data"}
        soft["scene_aware"] = {"pass": None, "observation": "No attention data"}

    return {
        "primary": primary,
        "primary_pass": primary_pass,
        "secondary": secondary,
        "soft": soft,
    }


# --- report ---

def _em_cell(all_results, ds, method):
    r = all_results.get((ds, method), {})
    res_list = r.get("results", [])
    n = len(res_list)
    if n == 0:
        return "—"
    em = sum(x["em"] for x in res_list) / n * 100
    return f"{em:.1f}"


def build_comparison_table(all_results, datasets):
    lines = []
    ds_names = [d for d in datasets if datasets[d]]
    header = "| Method | " + " | ".join(ds_names) + " |"
    lines.append(header)
    sep = "|" + "|".join([" --- " for _ in range(len(ds_names) + 1)]) + "|"
    lines.append(sep)

    for method in METHODS:
        label = {"greedy": "Greedy", "cad": "CAD", "dola": "DoLa",
                 "edap": "**EDAP (ours)**", "edap_random": "EDAP-random"}[method]
        cells = [_em_cell(all_results, d, method) for d in ds_names]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_per_source_table(all_results, ds_name, method):
    r = all_results.get((ds_name, method), {}).get("results", [])
    if not r:
        return f"*No results for {method} on {ds_name}*"

    by_source = defaultdict(list)
    for x in r:
        by_source[x.get("correct_source", "unknown")].append(x["em"])

    lines = [f"| Source Type | EM (%) | N Samples |",
             "| --- | --- | --- |"]
    for st in sorted(by_source):
        ems = by_source[st]
        lines.append(f"| {st} | {sum(ems)/len(ems)*100:.1f} | {len(ems)} |")
    return "\n".join(lines)


def build_verdict_table(criteria_dict):
    lines = ["| Criterion | Status | EDAP EM | Baseline EM | Δ |",
             "| --- | --- | --- | --- | --- |"]
    for _, c in criteria_dict.items():
        status = "✅ PASS" if c["pass"] else "❌ FAIL"
        lines.append(
            f"| {c['label']} | {status} | "
            f"{c['edap_em']:.1f}% | {c['baseline_em']:.1f}% | "
            f"{'+' if c['delta']>=0 else ''}{c['delta']:.1f} |"
        )
    return "\n".join(lines)


def build_soft_table(soft_criteria):
    lines = ["| Criterion | Status | Observation |",
             "| --- | --- | --- |"]
    for name, c in soft_criteria.items():
        if c["pass"] is None:
            status = "⚪ N/A"
        elif c["pass"]:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        lines.append(f"| {name} | {status} | {c['observation']} |")
    return "\n".join(lines)


def build_attention_section(attn_data):
    """Build markdown showing per-plugin mean attention weights."""
    lines = []

    for label, key in [("EDAP", "edap"), ("EDAP-random", "edap_random")]:
        attn = attn_data.get(("NQ-Swap", key), {})
        per_plugin = attn.get("per_plugin", {}) if attn else {}
        if not per_plugin:
            lines.append(f"### {label}\n*No attention data available.*\n")
            continue

        lines.append(f"### {label} — Mean Attention Weights")
        # dynamic columns based on max n_sources
        max_src = max((len(v.get("mean", [])) for v in per_plugin.values()), default=0)
        header = "| Plugin | " + " | ".join(f"S{i}" for i in range(max_src)) + " |"
        lines.append(header)
        sep = "| --- |" + "|".join([" --- " for _ in range(max_src)]) + "|"
        lines.append(sep)

        for pi in sorted(per_plugin, key=lambda x: int(x.split("_")[1])):
            p = per_plugin[pi]
            means = p.get("mean", [])
            cells = []
            for i in range(max_src):
                if i < len(means):
                    cells.append(f"{means[i]:.3f}")
                else:
                    cells.append("—")
            lines.append("| " + pi + " | " + " | ".join(cells) + " |")
        lines.append("")

    # scene-aware breakdown
    edap_attn = attn_data.get(("NQ-Swap", "edap"), {})
    per_stype = edap_attn.get("per_source_type", {}) if edap_attn else {}
    if per_stype:
        lines.append("### EDAP — Attention by Source Type")
        for stype in sorted(per_stype):
            plugins = per_stype[stype]
            lines.append(f"\n**{stype}** (n={plugins.get('edap_0', {}).get('n', 0)} samples)")
            for pi_name in sorted(plugins, key=lambda x: int(x.split("_")[1])):
                p = plugins[pi_name]
                means = p.get("mean", [])
                if means:
                    peak_idx = max(range(len(means)), key=lambda i: means[i])
                    lines.append(
                        f"- {pi_name}: {[round(m, 3) for m in means]} "
                        f"→ peak at Source {peak_idx} ({means[peak_idx]:.3f})"
                    )
        lines.append("")

    return "\n".join(lines)


def analyze_failures(all_results, max_per_type=5):
    """Find EDAP failures where Greedy succeeded, grouped by source type."""
    lines = []

    for ds_name in ["NQ-Swap", "ConFiQA"]:
        edap_r = all_results.get((ds_name, "edap"), {}).get("results", [])
        greedy_r = all_results.get((ds_name, "greedy"), {}).get("results", [])
        if not edap_r or not greedy_r:
            continue

        n = min(len(edap_r), len(greedy_r))
        failures = defaultdict(list)
        for i in range(n):
            e = edap_r[i]
            g = greedy_r[i]
            # EDAP wrong (em=0) but Greedy right (em=1)
            if e["em"] == 0 and g["em"] == 1:
                stype = e.get("correct_source", "unknown")
                failures[stype].append({
                    "idx": i,
                    "correct_source": stype,
                    "gt": e["gt"],
                    "edap_pred": e["pred"],
                    "greedy_pred": g["pred"],
                })

        if not failures:
            lines.append(f"**{ds_name}**: No failure reversals found (EDAP wrong + Greedy right).\n")
            continue

        lines.append(f"### {ds_name}")
        for stype in sorted(failures):
            cases = failures[stype][:max_per_type]
            lines.append(f"\n**{stype}** ({len(failures[stype])} total, showing {len(cases)})")
            for c in cases:
                lines.append(
                    f"- `gt={c['gt']}` | EDAP→`{c['edap_pred']}` | "
                    f"Greedy→`{c['greedy_pred']}`"
                )
        lines.append("")

    if not any(
        all_results.get((ds, "edap"), {}).get("results") and
        all_results.get((ds, "greedy"), {}).get("results")
        for ds in ["NQ-Swap", "ConFiQA"]
    ):
        lines.append("*No EDAP or Greedy results available for failure analysis.*\n")

    return "\n".join(lines)


# --- heatmap ---

def generate_heatmap(attn_data, output_dir):
    """Generate attention heatmap: EDAP vs EDAP-random, per-plugin weights."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available — skipping heatmap")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, (label, key) in zip(axes, [("EDAP", "edap"), ("EDAP-random", "edap_random")]):
        attn = attn_data.get(("NQ-Swap", key), {})
        per_plugin = attn.get("per_plugin", {}) if attn else {}

        if not per_plugin:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue

        plugins_sorted = sorted(per_plugin, key=lambda x: int(x.split("_")[1]))
        max_src = max((len(per_plugin[p].get("mean", [])) for p in plugins_sorted),
                      default=0)

        # build matrix: [n_plugins × max_src], mask invalid positions
        matrix = []
        masks = []
        cell_text = []
        for pi_name in plugins_sorted:
            means = per_plugin[pi_name].get("mean", [])
            row = list(means) + [float("nan")] * (max_src - len(means))
            mask_row = [False] * len(means) + [True] * (max_src - len(means))
            matrix.append(row)
            masks.append(mask_row)
            cell_text.append([f"{v:.3f}" if not m else "" for v, m in zip(row, mask_row)])

        matrix = [[v for v in row] for row in matrix]
        masked = [[v if not masks[i][j] else float("nan")
                   for j, v in enumerate(row)] for i, row in enumerate(matrix)]

        # diverging colormap centered per row at 1/n_sources for that row
        # simpler: use RdBu_r with global vmin/vmax
        flat_vals = [v for row in masked for v in row if not math.isnan(v)]
        if flat_vals:
            vmin, vmax = min(flat_vals), max(flat_vals)
            # center at uniform = 1/n (approximate)
            uniform_approx = 1.0 / max_src if max_src > 0 else 0.25
            abs_max = max(abs(vmax - uniform_approx), abs(uniform_approx - vmin), 0.01)
            vmin_c, vmax_c = uniform_approx - abs_max, uniform_approx + abs_max
        else:
            vmin_c, vmax_c = 0, 1

        im = ax.imshow(masked, aspect="auto", cmap="RdBu_r",
                       vmin=vmin_c, vmax=vmax_c)

        # annotations
        for i in range(len(plugins_sorted)):
            for j in range(max_src):
                if not masks[i][j]:
                    val = matrix[i][j]
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold")

        ax.set_xticks(range(max_src))
        ax.set_xticklabels([f"Source {i}" for i in range(max_src)])
        ax.set_yticks(range(len(plugins_sorted)))
        ax.set_yticklabels(plugins_sorted)
        ax.set_title(f"{label}\nmean attention per plugin", fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, label="attention weight")

    out_path = Path(output_dir) / "attention-heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Heatmap saved → {out_path}")


# --- main ---

def write_report(args, all_results, attn_data, verdict):
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# EDAP Prototype Experiment Report\n")
    lines.append(f"**Generated**: {ts}  ")
    lines.append(f"**Model**: Qwen2.5-7B (bf16)  ")
    lines.append(f"**Checkpoints**:  ")
    lines.append(f"- EDAP: `{args.edap_ckpt}`  ")
    lines.append(f"- EDAP-random: `{args.edap_random_ckpt}`  \n")

    # ── 1. Success Criteria ──
    lines.append("## 1. Success Criteria Verdict\n")
    lines.append("### Primary (Hard Gates — both must pass)\n")
    lines.append(build_verdict_table(verdict["primary"]))
    overall = "✅ **ALL PRIMARY CRITERIA PASSED**" if verdict["primary_pass"] else \
              "❌ **PRIMARY CRITERIA NOT MET**"
    lines.append(f"\n**Overall Primary**: {overall}\n")

    lines.append("### Secondary (Soft Expectations)\n")
    lines.append(build_verdict_table(verdict["secondary"]))
    lines.append("")

    lines.append("### Soft Criteria\n")
    lines.append(build_soft_table(verdict["soft"]))
    lines.append("")

    # ── 2. Comparison Table ──
    lines.append("## 2. Main Comparison\n")
    datasets = {}
    for (ds, _) in all_results:
        datasets[ds] = True
    datasets = {d: True for d in sorted(datasets)}
    lines.append(build_comparison_table(all_results, datasets))
    lines.append("")

    # ── 3. Per-Source Breakdown ──
    lines.append("## 3. Per-Source Breakdown\n")
    for ds_name in datasets:
        lines.append(f"### {ds_name}\n")
        for method in ["edap", "greedy", "cad", "dola", "edap_random"]:
            lines.append(f"**{method}**")
            lines.append(build_per_source_table(all_results, ds_name, method))
            lines.append("")

    # ── 4. Attention Analysis ──
    lines.append("## 4. Attention Analysis\n")
    lines.append(build_attention_section(attn_data))

    heatmap_path = Path(args.output_dir) / "attention-heatmap.png"
    if heatmap_path.exists():
        lines.append(f"![Attention Heatmap](attention-heatmap.png)\n")

    # ── 5. Failure Cases ──
    lines.append("## 5. Failure Case Analysis\n")
    lines.append(
        "*Cases where EDAP predicted incorrectly but vanilla Greedy "
        "got it right.  These highlight EDAP-specific regressions.*\n"
    )
    lines.append(analyze_failures(all_results))

    # ── 6. Raw Result Files ──
    lines.append("## 6. Raw Result Files\n")
    lines.append("| File | Method | Dataset | EM |")
    lines.append("| --- | --- | --- | --- |")
    for (ds_name, method), r in sorted(all_results.items()):
        res_list = r.get("results", [])
        n = len(res_list)
        em = sum(x["em"] for x in res_list) / n * 100 if n else 0.0
        fname = _result_path(args.output_dir, ds_name, method).name
        lines.append(f"| {fname} | {method} | {ds_name} | {em:.1f}% |")
    lines.append("")

    lines.append("---\n*Report generated by `src/report.py`*")

    report_text = "\n".join(lines)
    report_path = Path(args.output_dir) / "prototype-report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved → {report_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load model
    model_path = args.model_path if Path(args.model_path).exists() else args.model_name
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path or args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load EDAP checkpoints
    print("Loading EDAP plugins...")
    ckpt = torch.load(args.edap_ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    n_heads = cfg.get("edap_heads", 8)
    n_blocks = cfg.get("edap_blocks", 4)
    dropout = cfg.get("edap_dropout", 0.1)
    print(f"  Checkpoint config: n_heads={n_heads}, n_blocks={n_blocks}")

    edap_plugins = create_edap_plugins(
        d_model=model.config.hidden_size,
        n_heads=n_heads, n_blocks=n_blocks, dropout=dropout,
    ).to(device).to(torch.bfloat16)
    edap_plugins.load_state_dict(ckpt["edap_plugins"])
    # restore lm_head
    for name, p in model.named_parameters():
        if "lm_head" in name and name in ckpt.get("lm_head", {}):
            p.data.copy_(ckpt["lm_head"][name])
    print(f"  EDAP loaded: {args.edap_ckpt}")

    ckpt_r = torch.load(args.edap_random_ckpt, map_location=device, weights_only=False)
    cfg_r = ckpt_r.get("config", {})
    n_heads_r = cfg_r.get("edap_heads", 8)
    n_blocks_r = cfg_r.get("edap_blocks", 4)
    dropout_r = cfg_r.get("edap_dropout", 0.1)

    edap_random_plugins = create_edap_plugins(
        d_model=model.config.hidden_size,
        n_heads=n_heads_r, n_blocks=n_blocks_r, dropout=dropout_r,
    ).to(device).to(torch.bfloat16)
    edap_random_plugins.load_state_dict(ckpt_r["edap_plugins"])
    print(f"  EDAP-random loaded: {args.edap_random_ckpt}")

    # load eval data
    print("Loading evaluation data...")
    datasets = {}
    nq_path = args.nq_swap_path
    try:
        datasets["NQ-Swap"] = load_eval_data(nq_path, args.max_samples)
        print(f"  NQ-Swap: {len(datasets['NQ-Swap'])} samples")
    except Exception as e:
        print(f"  [warn] NQ-Swap not available: {e}")

    try:
        confiqa = load_eval_data(args.data_path, args.max_samples)
        datasets["ConFiQA"] = confiqa
        print(f"  ConFiQA: {len(confiqa)} samples")
    except Exception as e:
        print(f"  [warn] ConFiQA not available: {e}")

    # run all evaluations
    print("\nRunning evaluations...")
    all_results, attn_data = run_all_evaluations(
        args, model, tokenizer, datasets, edap_plugins, edap_random_plugins,
    )

    # compute verdict
    print("\nEvaluating success criteria...")
    verdict = evaluate_success_criteria(all_results, attn_data)

    # write report
    write_report(args, all_results, attn_data, verdict)

    # heatmap
    if not args.no_heatmap and attn_data:
        print("Generating heatmap...")
        generate_heatmap(attn_data, args.output_dir)

    # final summary to stdout
    print("\n" + "=" * 50)
    print("REPORT SUMMARY")
    print("=" * 50)
    for name, c in verdict["primary"].items():
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['label']}: EDAP={c['edap_em']}% vs {c['baseline_em']}%")
    print(f"\nOverall primary: {'PASS' if verdict['primary_pass'] else 'FAIL'}")
    print(f"Full report: {Path(args.output_dir) / 'prototype-report.md'}")


if __name__ == "__main__":
    main()
