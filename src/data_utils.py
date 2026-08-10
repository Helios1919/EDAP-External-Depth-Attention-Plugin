"""ConFiQA data loading and NQ-Swap construction."""

import json
import os
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


_PROMPT = "{context}\n\nQuestion: {question}\n\nAnswer:"


class ConFiQADataset(Dataset):

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        augment_counterfactual: bool = True,
        tokenizer=None,
        max_seq_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        # Default to local converted ConFiQA dataset
        if data_path is None:
            data_path = str(Path(__file__).parent.parent / "data" / "confiqa" / "confiqa_train.json")

        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            print(f"ConFiQA: loaded {len(raw)} samples from {data_path}")
        else:
            print(f"Local ConFiQA not found at {data_path}, trying HF download...")
            ds = load_dataset("RajMaheshwari/ConFiQA", "QA", split=split)
            raw = [dict(item) for item in ds]

        if max_samples:
            raw = raw[:max_samples]

        cr = [s for s in raw if s.get("type") == "context_required"]
        ci = [s for s in raw if s.get("type") == "context_irrelevant"]
        cc = [s for s in raw if s.get("type") == "counterfactual"]

        self.samples.extend(cc[:2000])
        self.samples.extend(cr[:1500])
        self.samples.extend(ci[:1500])

        if augment_counterfactual:
            for s in cc[:2000]:
                # only flip samples that have a correct-context variant
                if "context_correct" not in s or "context_answer" not in s:
                    continue
                flipped = dict(s)
                flipped["context"] = s["context_correct"]
                flipped["answer"] = s["context_answer"]
                flipped["correct_source"] = "context"
                flipped["_flipped"] = True
                self.samples.append(flipped)

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = _PROMPT.format(context=s["context"], question=s["question"])
        answer = s.get("answer", "")

        if self.tokenizer is None:
            return {
                "prompt": prompt,
                "answer": answer,
                "correct_source": s.get("correct_source", "unknown"),
            }

        full_text = prompt + " " + answer
        tokens = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_length,
            padding="max_length", return_tensors="pt",
        )
        prompt_tok = self.tokenizer(
            prompt, truncation=True, max_length=self.max_seq_length,
        )
        p_len = len(prompt_tok["input_ids"])

        labels = tokens["input_ids"].clone()
        labels[0, :p_len] = -100

        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": labels[0],
            "prompt_len": p_len,
        }


def load_nq_swap(cache_path=None, max_samples=None):
    """Load NQ-Swap from HuggingFace (pminervini/NQ-Swap).

    NQ-Swap: Natural Questions with entity-substituted contexts.
    The model should trust the context (sub_answer) because the user
    intentionally provided the swapped entity.

    If ``cache_path`` is given and the file exists, loads from local JSON.
    Otherwise downloads from HF and optionally caches.

    Returns list of dicts with keys:
        question, context, correct_answer, correct_source,
        original_answer, context_answer
    """
    if cache_path and Path(cache_path).exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"NQ-Swap: {len(data)} samples loaded from {cache_path}")
        if max_samples:
            data = data[:max_samples]
        return data

    print("Downloading NQ-Swap from pminervini/NQ-Swap ...")
    ds = load_dataset("pminervini/NQ-Swap", split="dev")

    out = []
    for item in ds:
        out.append({
            "question": item["question"],
            "context": item["sub_context"],
            "correct_answer": item["sub_answer"],
            "correct_source": "context",
            "original_answer": item["org_answer"],
            "context_answer": item["sub_answer"],
        })

    print(f"NQ-Swap: {len(out)} samples downloaded")

    if cache_path:
        os.makedirs(Path(cache_path).parent, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Cached to {cache_path}")

    if max_samples:
        out = out[:max_samples]
    return out


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def create_dataloader(dataset, batch_size=4, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )


if __name__ == "__main__":
    import sys
    # Quick test: load ConFiQA and print stats
    ds = ConFiQADataset()
    print(f"Total samples: {len(ds)}")
    cc = sum(1 for s in ds.samples if s.get("type") == "counterfactual" and not s.get("_flipped"))
    cr = sum(1 for s in ds.samples if s.get("type") == "context_required" and not s.get("_flipped"))
    ci = sum(1 for s in ds.samples if s.get("type") == "context_irrelevant")
    flipped = sum(1 for s in ds.samples if s.get("_flipped"))
    print(f"  CC: {cc}, CR: {cr}, CI: {ci}, Flipped: {flipped}")
    print(f"Sample: {ds[0]}")
