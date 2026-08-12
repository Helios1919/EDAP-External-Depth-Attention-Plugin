"""将 ConFiQA 官方原始 JSON 转换为 data_utils 需要的统一格式。

每个原始样本产生两条：
  - context_required: 用 orig_context, 答案在上下文中
  - counterfactual:   用 cf_context (含错误事实), 正确答案靠记忆

去污染: 检测反事实上下文中是否包含正确答案子串，
        若泄露则标记并尝试替换为 MASK token。
"""

import json
import random
import re
from pathlib import Path
from typing import Tuple

SRC_DIR = Path(__file__).parent.parent / "ConFiQA"
OUT_DIR = Path(__file__).parent.parent / "data" / "confiqa"
TRAIN_DST = OUT_DIR / "confiqa_train.json"
TEST_DST = OUT_DIR / "confiqa_test.json"
SEED = 42
TEST_RATIO = 0.2  # 80/20 train/test split

files = ["ConFiQA-QA.json", "ConFiQA-MR.json", "ConFiQA-MC.json"]

samples = []
leak_warnings = []

def _check_answer_leak(context: str, orig_answer: str) -> Tuple[str, bool]:
    """Check whether orig_answer (the memory-truth answer) appears in the
    counterfactual context.  If it does, flag it and replace occurrences
    with [MASK] to prevent the model from extracting the answer directly
    from the context (shortcut learning).

    Returns (cleaned_context, leaked_bool).
    """
    answer_lower = orig_answer.strip().lower()
    context_lower = context.lower()
    if not answer_lower or len(answer_lower) < 3:
        return context, False
    # Escape for regex but keep word boundaries flexible
    escaped = re.escape(answer_lower)
    if not re.search(escaped, context_lower):
        return context, False
    # Replace all case-insensitive occurrences with [MASK]
    cleaned = re.sub(escaped, "[MASK]", context, flags=re.IGNORECASE)
    return cleaned, True

for fname in files:
    with open(SRC_DIR / fname, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for item in raw:
        # ---- context_required ----
        samples.append({
            "type": "context_required",
            "question": item["question"],
            "context": item["orig_context"],
            "answer": item["orig_answer"],
            "context_answer": item["orig_answer"],
            "correct_source": "context",
            "config": fname.replace("ConFiQA-", "").replace(".json", ""),
        })
        # ---- counterfactual ----
        cf_context = item["cf_context"]
        cf_context, leaked = _check_answer_leak(cf_context, item["orig_answer"])
        if leaked:
            leak_warnings.append(
                f"[LEAK] {fname} | Q: {item['question'][:60]}... | "
                f"ans='{item['orig_answer']}' found in cf_context"
            )
        samples.append({
            "type": "counterfactual",
            "question": item["question"],
            "context": cf_context,
            "answer": item["orig_answer"],
            "context_answer": item["cf_answer"],
            "correct_source": "memory",
            "context_correct": item["orig_context"],
            "config": fname.replace("ConFiQA-", "").replace(".json", ""),
            "_answer_leaked": leaked,
        })

random.seed(SEED)
random.shuffle(samples)

# 按类型分层切分 train/test
cc_samples = [s for s in samples if s["type"] == "counterfactual"]
cr_samples = [s for s in samples if s["type"] == "context_required"]
ci_samples = [s for s in samples if s["type"] == "context_irrelevant"]

train_samples, test_samples = [], []
for group in [cc_samples, cr_samples, ci_samples]:
    n_test = max(1, int(len(group) * TEST_RATIO))
    test_samples.extend(group[:n_test])
    train_samples.extend(group[n_test:])

random.shuffle(train_samples)
random.shuffle(test_samples)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(TRAIN_DST, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, ensure_ascii=False, indent=2)
with open(TEST_DST, "w", encoding="utf-8") as f:
    json.dump(test_samples, f, ensure_ascii=False, indent=2)

# 统计
from collections import Counter
for name, subset in [("train", train_samples), ("test", test_samples)]:
    cnt = Counter(s["type"] for s in subset)
    cfg = Counter(s["config"] for s in subset)
    leaked_n = sum(1 for s in subset if s.get("_answer_leaked"))
    print(f"✅ {name}: {len(subset)} 条  →  type {dict(cnt)}  config {dict(cfg)}")
    if leaked_n:
        print(f"   ⚠️  Answer leak in cf_context: {leaked_n} samples (replaced with [MASK])")
if leak_warnings:
    print(f"\n⚠️  {len(leak_warnings)} counterfactual samples had answer leaks:")
    for w in leak_warnings[:10]:
        print(f"   {w}")
    if len(leak_warnings) > 10:
        print(f"   ... and {len(leak_warnings) - 10} more")
