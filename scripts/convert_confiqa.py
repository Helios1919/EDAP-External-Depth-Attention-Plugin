"""将 ConFiQA 官方原始 JSON 转换为 data_utils 需要的统一格式。

每个原始样本产生两条：
  - context_required: 用 orig_context, 答案在上下文中
  - counterfactual:   用 cf_context (含错误事实), 正确答案靠记忆
"""

import json
import random
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "ConFiQA"
DST = Path(__file__).parent.parent / "data" / "confiqa" / "confiqa_train.json"
SEED = 42

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

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

# 统计
from collections import Counter
cnt = Counter(s["type"] for s in samples)
cfg = Counter(s["config"] for s in samples)
print(f"✅ 写入 {DST}  →  {len(samples)} 条样本")
print(f"   type 分布: {dict(cnt)}")
print(f"   config 分布: {dict(cfg)}")
