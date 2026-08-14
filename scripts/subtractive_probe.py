"""Zero-training subtractive-direction probe.

Question: is the direction (h_shallow - h_deep) — the "trust context, suppress
memory" direction — natively aligned with flipping a counterfactual answer? No
training, no plugin. We just inject  delta = alpha * (h_shallow - h_deep)  after
the final RMSNorm and check how often the argmax flips to the target token.

Control: a random direction with the same magnitude, to show the gain is from the
direction, not the magnitude.
"""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "/root/EDAP/src")
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
DT = torch.bfloat16
DATA = "/root/EDAP/data/confiqa/confiqa_test.json"
N = 300
SHALLOW = 3      # layer 3 output (block-0 exit) ~ context
DEEP = 27        # layer 27 output (last block) ~ memory

model = AutoModelForCausalLM.from_pretrained(
    "/root/EDAP/models/qwen2.5-7b", torch_dtype=DT, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/root/EDAP/models/qwen2.5-7b")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
norm = model.model.norm

data = json.load(open(DATA))
cc = [s for s in data if s.get("type") == "counterfactual"][:N]

ALPHAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]


def first_token(text):
    ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


# precompute per-sample hidden states (shallow/deep at last pos)
rows = []
with torch.no_grad():
    for s in cc:
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :]          # final, pre-norm
        hs = out.hidden_states[SHALLOW + 1][0, -1, :]
        hd = out.hidden_states[DEEP + 1][0, -1, :]
        t_gt = first_token(s["answer"])
        if t_gt is None:
            continue
        hn = norm(h)                                  # post-norm final
        ns = norm(hs)                                 # post-norm shallow
        nd = norm(hd)                                 # post-norm deep
        rows.append((hn, ns, nd, t_gt))

print(f"usable counterfactual samples: {len(rows)}")

torch.manual_seed(0)
rand_dir = torch.randn(model.config.hidden_size, device=device)
rand_dir = rand_dir / rand_dir.norm()

print(f"\n{'alpha':>6} | {'subtractive flip%':>18} | {'random-dir flip%':>17}")
for a in ALPHAS:
    flip_sub = flip_rnd = 0
    for hn, ns, nd, t_gt in rows:
        hn = hn.to(DT); ns = ns.to(DT); nd = nd.to(DT)
        d_sub = ns - nd
        d_sub = d_sub / (d_sub.norm() + 1e-8)
        lg_sub = model.lm_head(hn + a * d_sub)
        lg_rnd = model.lm_head(hn + a * rand_dir.to(DT))
        if lg_sub.argmax().item() == t_gt:
            flip_sub += 1
        if lg_rnd.argmax().item() == t_gt:
            flip_rnd += 1
    print(f"{a:>6} | {100*flip_sub/len(rows):>17.1f}% | {100*flip_rnd/len(rows):>16.1f}%")
