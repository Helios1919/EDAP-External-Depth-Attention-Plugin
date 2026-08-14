"""Logit-Router EDAP: learnable per-depth logit subtraction (minimal probe).

Core idea (vs old EDAP which mixed hidden states):
    logits = (1 + lam) * logits_final  -  lam * sum_ell w_ell * logits_ell
where w = softmax(MHA-router(x)) is a *content-dependent* weight over reference
layers, and lam is a global contrast strength. The subtraction is done in LOGIT
(unembedding) space where the "trust context / suppress memory" direction is
naturally aligned (verified by oracle-delta). The MHA only *scores* which depths
hold the conflicting memory — it does NOT rewrite hidden states.

What this probe answers:
  1. Does learnable multi-layer logit subtraction beat vanilla (and approach DoLa)?
  2. Are the learned weights w *content-dependent* (do they differ between
     counterfactual vs context_required samples)?  -> the actual novelty claim.

Frozen: backbone + lm_head. Trainable: MHA router (W_Q, W_K) + scalar lam.
"""
import sys, os, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/root/EDAP/src")
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
DT = torch.bfloat16
MODEL = "/root/EDAP/models/qwen2.5-7b"
TRAIN = "/root/EDAP/data/confiqa/confiqa_train.json"
TEST = "/root/EDAP/data/confiqa/confiqa_test.json"
REF_LAYERS = [3, 7, 11, 15, 19, 23, 27]   # block exits -> hidden_states[i+1]
N_TRAIN = 4000
N_TEST = 300
EPOCHS = 2
LR = 1e-3


class LogitRouter(nn.Module):
    """Multi-head scoring over depth: q = final-layer state, K = each ref layer."""
    def __init__(self, d_model, n_heads, n_ref):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_ref = n_ref
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        # init small so routing starts near-uniform
        nn.init.normal_(self.W_Q.weight, 0.0, 0.02)
        nn.init.normal_(self.W_K.weight, 0.0, 0.02)

    def forward(self, q, K):
        # q: [B, d]  (final layer, last position, post-norm)
        # K: [B, n_ref, d]
        B = q.shape[0]
        Q = self.W_Q(q).view(B, self.n_heads, self.d_head)          # [B,H,dh]
        Kk = self.W_K(K).view(B, self.n_ref, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        scores = torch.einsum('bhd,bhrd->bhr', Q, Kk) / (self.d_head ** 0.5)  # [B,H,n_ref]
        w = scores.softmax(-1).mean(1)                              # [B,n_ref]
        return w


def get_logits(model, ids):
    """Final + per-ref-layer logits at the LAST position, plus post-norm hiddens."""
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states                      # list, hs[i] = layer i-1 output (hs[0]=emb)
    norm = model.model.norm
    lm = model.lm_head
    last = hs[-1][:, -1, :]                      # [B,d] layer 27, pre-norm
    q = norm(last)                               # [B,d] post-norm -> router query
    logits_final = lm(q)                         # [B,V]
    K = torch.stack([norm(hs[l + 1][:, -1, :]) for l in REF_LAYERS], dim=1)  # [B,n_ref,d]
    logits_ref = torch.stack([lm(norm(hs[l + 1][:, -1, :])) for l in REF_LAYERS], dim=1)  # [B,n_ref,V]
    return q, K, logits_final, logits_ref


def first_token_ids(texts, tokenizer):
    ids = [tokenizer(" " + t, add_special_tokens=False)["input_ids"][0] for t in texts]
    return torch.tensor(ids, device=device)


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DT, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    d = model.config.hidden_size
    n_ref = len(REF_LAYERS)
    router = LogitRouter(d, n_heads=8, n_ref=n_ref).to(device).to(DT)
    lam = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
    opt = torch.optim.AdamW(list(router.parameters()) + [lam], lr=LR, weight_decay=0.01)

    train = [s for s in json.load(open(TRAIN)) if s.get("type") == "counterfactual"][:N_TRAIN]
    print(f"train counterfactual: {len(train)}")

    router.train()
    for ep in range(EPOCHS):
        tot = 0.0; n = 0
        for s in train:
            prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            tgt = torch.tensor([tokenizer(" " + s["answer"], add_special_tokens=False)["input_ids"][0]],
                               device=device)
            q, K, logits_final, logits_ref = get_logits(model, ids)
            w = router(q, K)                                   # [1,n_ref]
            ref = (w.unsqueeze(-1) * logits_ref).sum(1)        # [1,V]
            lam_c = torch.clamp(lam, 0.0, 3.0)
            logits = (1 + lam_c) * logits_final - lam_c * ref  # [1,V]
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep}: avg loss {tot/n:.4f}  lam={lam.item():.3f}")

    # ---- eval: single-token flip on counterfactual test ----
    router.eval()
    test = [s for s in json.load(open(TEST)) if s.get("type") == "counterfactual"][:N_TEST]
    flip_vanilla = flip_router = 0
    w_cc = torch.zeros(n_ref, device=device)
    with torch.no_grad():
        for s in test:
            prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            tgt = tokenizer(" " + s["answer"], add_special_tokens=False)["input_ids"][0]
            q, K, logits_final, logits_ref = get_logits(model, ids)
            if logits_final.argmax(-1).item() == tgt:
                flip_vanilla += 1
            w = router(q, K); w_cc += w[0]
            lam_c = torch.clamp(lam, 0.0, 3.0)
            ref = (w.unsqueeze(-1) * logits_ref).sum(1)
            logits = (1 + lam_c) * logits_final - lam_c * ref
            if logits.argmax(-1).item() == tgt:
                flip_router += 1
    print(f"\n=== single-token accuracy on counterfactual test (N={len(test)}) ===")
    print(f"vanilla : {100*flip_vanilla/len(test):.1f}%")
    print(f"router  : {100*flip_router/len(test):.1f}%  (lam={lam.item():.2f})")
    print(f"avg w over depth: {[round(x, 3) for x in (w_cc/len(test)).tolist()]}")
    print(f"REF_LAYERS     : {REF_LAYERS}")

    # ---- content-dependence: counterfactual vs context_required ----
    cr = [s for s in json.load(open(TEST)) if s.get("type") == "context_required"][:N_TEST]
    def avg_w(samples):
        acc = torch.zeros(n_ref, device=device)
        with torch.no_grad():
            for s in samples:
                prompt = f"{s['context']}\n\nQuestion: {s['question']}\n\nAnswer:"
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                q, K, _, _ = get_logits(model, ids)
                acc += router(q, K)[0]
        return acc / len(samples)
    if cr:
        w_cr = avg_w(cr)
        print(f"\ncontent-dependence check (avg w):")
        print(f"counterfactual  : {[round(x,3) for x in (w_cc/len(test)).tolist()]}")
        print(f"context_required: {[round(x,3) for x in w_cr.tolist()]}")


if __name__ == "__main__":
    main()
