"""将 ConFiQA 官方原始 JSON 转换为 data_utils 需要的统一格式。

每个原始样本产生两条：
  - context_required: 用 orig_context, 答案在上下文中
  - counterfactual:   用 cf_context (含错误事实), 正确答案靠记忆
"""

import json
import random
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "ConFiQA"
OUT_DIR = Path(__file__).parent.parent / "data" / "confiqa"
TRAIN_DST = OUT_DIR / "confiqa_train.json"
TEST_DST = OUT_DIR / "confiqa_test.json"
SEED = 42
TEST_RATIO = 0.2  # 80/20 train/test split

files = ["ConFiQA-QA.json", "ConFiQA-MR.json", "ConFiQA-MC.json"]

samples = []

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
        samples.append({
            "type": "counterfactual",
            "question": item["question"],
            "context": item["cf_context"],
            "answer": item["orig_answer"],
            "context_answer": item["cf_answer"],
            "correct_source": "memory",
            "context_correct": item["orig_context"],
            "config": fname.replace("ConFiQA-", "").replace(".json", ""),
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
    print(f"✅ {name}: {len(subset)} 条  →  type {dict(cnt)}  config {dict(cfg)}")
