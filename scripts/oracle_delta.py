"""Oracle-delta feasibility probe.

Question: under a *fully frozen* backbone + *frozen* lm_head, can a residual-stream
delta injected at the final block push the model to output the target token? And how
large must that delta be, relative to the hidden state itself?

Two oracles, from most optimistic to most realistic:
  1. Oracle-Linear  (closed form, injected AFTER final RMSNorm):
       logits = W @ h_norm + b  is linear in h_norm, so the minimum-L2 delta that
       flips current argmax t_cur -> target t_gt has a closed form:
         d = W[t_gt] - W[t_cur],  margin = logits[t_cur] - logits[t_gt],
         delta = margin * d / ||d||^2.
       Report ||delta|| / ||h_norm||. This is the *best possible case* for a
       hidden-space intervention.
  2. Oracle-Residual (gradient search, injected BEFORE final RMSNorm):
       optimize a delta on the raw last-layer output so that, after RMSNorm +
       lm_head, the target token becomes the argmax. Report ||delta|| / ||h||.
       This is where EDAP actually injects, and RMSNorm is nonlinear here.

Interpretation:
  - If Oracle-Linear already needs a large relative perturbation (>~0.3), then even
    the most optimistic hidden-space intervention cannot flip the frozen lm_head:
    the right fix is to intervene at the *logit* level (CAD/DoLa) or unfreeze lm_head.
  - If Oracle-Linear is small but Oracle-Residual is large, RMSNorm is the obstacle:
    injecting before the norm is the wrong position.
"""
import sys, os, json, torch
import torch.nn.functional as F
sys.path.insert(0, "/root/EDAP/src")
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
COMPUTE_DTYPE = torch.bfloat16
DATA = "/root/EDAP/data/confiqa/confiqa_test.json"
N = 300
MARGIN = 1.0  # logit margin to guarantee a clean flip

model = AutoModelForCausalLM.from_pretrained(
    "/root/EDAP/models/qwen2.5-7b", torch_dtype=COMPUTE_DTYPE, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/root/EDAP/models/qwen2.5-7b")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

norm_fn = model.model.norm
W = model.lm_head.weight            # [V, d]
b = model.lm_head.bias              # None or [V]
V, d = W.shape
eps = getattr(model.model.norm, "variance_epsilon", 1e-6)

# float32 copies for the residual (gradient) oracle
W32 = W.float().detach()
b32 = b.float().detach() if b is not None else None
nw32 = model.model.norm.weight.float().detach()


def first_token(text):
    # Answer follows "Answer:" with a leading space, so encode " <text>" to get
    # the BPE token that actually starts the answer (ĠLondon, not London).
    ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def rmsnorm(x, w, e):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e) * w


data = json.load(open(DATA))
cc = [s for s in data if s.get("type") == "counterfactual"][:N]

lin_rels = []
res_rels = []
n_correct = 0
n_ctx = 0
n_other = 0
n_flip_fail = 0

for s in cc:
    prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    final_h = out.hidden_states[-1][0, -1, :]        # norm 前 [d]
    h_norm = norm_fn(final_h)                        # norm 后 [d]
    logits = model.lm_head(h_norm)                   # [V]
    t_cur = logits.argmax().item()
    t_gt = first_token(s["answer"])
    t_ctx = first_token(s.get("context_answer", ""))

    if t_gt is None:
        continue

    # --- vanilla behaviour ---
    if t_cur == t_gt:
        n_correct += 1
        continue
    if t_ctx is not None and t_cur == t_ctx:
        n_ctx += 1
    else:
        n_other += 1

    # --- Oracle-Linear (closed form, norm 后) ---
    dvec = (W[t_gt] - W[t_cur]).float()
    margin = (logits[t_cur] - logits[t_gt]).float().item() + 1e-3
    dnorm2 = (dvec @ dvec).item()
    delta_lin = (margin / dnorm2) * dvec
    lin_rels.append((delta_lin.norm() / h_norm.float().norm()).item())

    # --- Oracle-Residual (gradient search, norm 前) ---
    h0 = final_h.float().detach().clone()
    delta = torch.zeros(d, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=60,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        h = rmsnorm(h0 + delta, nw32, eps)
        lg = h @ W32.T + (b32 if b32 is not None else 0.0)
        target = lg[t_gt]
        others = lg.clone(); others[t_gt] = -1e9
        worst = others.max()
        loss = F.relu(worst - target + MARGIN) + 1e-4 * (delta.pow(2).sum())
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)

    with torch.no_grad():
        h = rmsnorm(h0 + delta, nw32, eps)
        lg = h @ W32.T + (b32 if b32 is not None else 0.0)
        flipped = lg.argmax().item() == t_gt
        rel = (delta.norm() / h0.norm()).item()
    if flipped:
        res_rels.append(rel)
    else:
        n_flip_fail += 1

n_total = len(cc)
print("=" * 60)
print(f"counterfactual samples: {n_total}")
print(f"  vanilla already correct (argmax == answer): {n_correct} ({100*n_correct/n_total:.1f}%)")
print(f"  vanilla outputs context_answer (misled):    {n_ctx} ({100*n_ctx/n_total:.1f}%)")
print(f"  vanilla outputs other token:                {n_other} ({100*n_other/n_total:.1f}%)")
print("-" * 60)


def dist(name, vals):
    if not vals:
        print(f"{name}: n=0")
        return
    vals = sorted(vals)
    import statistics
    print(f"{name}: n={len(vals)}  "
          f"min={vals[0]:.4f}  med={statistics.median(vals):.4f}  "
          f"mean={statistics.mean(vals):.4f}  p90={vals[int(0.9*len(vals))]:.4f}  "
          f"max={vals[-1]:.4f}")


print("Required relative perturbation ||delta|| / ||h||  (smaller = easier):")
dist("  Oracle-Linear (after norm, closed-form best-case)", lin_rels)
dist("  Oracle-Residual (before norm, gradient search) ", res_rels)
print(f"  residual-oracle flip failures: {n_flip_fail}")
print("=" * 60)
