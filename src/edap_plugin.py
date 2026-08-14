"""EDAP: cross-depth attention at block boundaries for knowledge conflict."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class EDAPPlugin(nn.Module):

    def __init__(
        self, n_sources: int, d_model: int = 3584, n_heads: int = 4,
        dropout: float = 0.1,
        W_K_shared: Optional[nn.Linear] = None,
        W_V_shared: Optional[nn.Linear] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sources = n_sources
        self.d_model = d_model
        self.n_heads = n_heads
        d_head = d_model // n_heads
        d_total = d_head * n_heads

        self.W_Q = nn.Linear(d_model, d_total, bias=False)
        self.W_O = nn.Linear(d_total, d_model, bias=False)

        # K/V can be shared across plugins — only one copy of params
        self.W_K = W_K_shared if W_K_shared is not None else nn.Linear(d_model, d_total, bias=False)
        self.W_V = W_V_shared if W_V_shared is not None else nn.Linear(d_model, d_total, bias=False)
        self._owns_KV = (W_K_shared is None)  # track ownership for state_dict

        self.norm_in = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_total)
        self.norm_out = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.depth_embed = nn.Parameter(torch.randn(n_sources, d_total) * 0.1)

        # Prevent magnitude asymmetry in delta mode: K_0 = W_K(emb - baseline)
        # instead of W_K(emb), making all K's relative rather than dominated
        # by the first source.
        self.baseline = nn.Parameter(torch.zeros(d_model))

        # Per-token soft gate: how much EDAP-fused output vs original block output.
        # The raw input [sources[-1], r_out] has RMS ~ 20 and 2*d = 7168 dims, so a
        # W_gate std of 0.02 would give a gate logit std of ~38.5 -> sigmoid 100%
        # saturates to 0/1 (hard per-token switch, unstable residual). Fix: LayerNorm
        # the gate input and scale W_gate so the logit std is ~O(1).
        self.norm_gate = nn.LayerNorm(d_model * 2)
        self.W_gate = nn.Linear(d_model * 2, 1)
        nn.init.normal_(self.W_gate.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.W_gate.bias)

        # init W_O near zero so the plugin starts as identity
        nn.init.normal_(self.W_O.weight, mean=0.0, std=1e-5)

    def forward(
        self,
        sources: list,
        shuffle_depth: bool = False,
        delta_mode: bool = False,
        edap_noise: float = 0.0,
    ):
        """Run cross-depth attention.

        sources: list of [B, S, d_model], last element is current block output.
        shuffle_depth: if True, permute source order for control experiment.
        delta_mode: if True, build K from incremental deltas (high contrast);
                    V always uses cumulated sources.
        edap_noise: if > 0 and in training mode, add scaled Gaussian noise
                    to sources before MHA. Mitigates exposure bias from
                    teacher-forcing (training sees GT hidden states, but
                    inference sees generated-token hidden states).

        Returns (r_out, weights, gate):
            r_out:  [B, S, d_model] fused output via residual + delta.
            weights: [B, S, H, N] attention weights over sources.
            gate:   [B, S, 1] per-token mixing gate.
        """
        # ---- exposure-bias noise (train only) -----------------------------
        if edap_noise > 0 and self.training:
            sources = [
                s + torch.randn_like(s) * edap_noise
                for s in sources
            ]
        B, S, _ = sources[-1].shape
        N = len(sources)

        R = torch.stack(sources, dim=0)  # [N, B, S, d] — cumulated for V

        # -- delta K ---------------------------------------------------------
        if delta_mode and N > 1:
            R_K_list = [sources[0] - self.baseline]
            for i in range(1, N):
                R_K_list.append(sources[i] - sources[i - 1])
            R_K = torch.stack(R_K_list, dim=0)  # [N, B, S, d]
        else:
            R_K = R

        # -- shuffle -----------------------------------------------------------
        if shuffle_depth and N > 1:
            perm = torch.randperm(N - 1, device=R.device)
            prev = R[:-1][perm]
            R = torch.cat([prev, R[-1:]], dim=0)
            prev_K = R_K[:-1][perm]
            R_K = torch.cat([prev_K, R_K[-1:]], dim=0)
            depth_embed = torch.cat(
                [self.depth_embed[:-1][perm], self.depth_embed[-1:]], dim=0
            )
        else:
            depth_embed = self.depth_embed

        # -- MHA --------------------------------------------------------------
        R_norm = self.norm_in(R)
        R_K_norm = self.norm_in(R_K)

        Q = self.W_Q(R_norm[-1])                 # [B, S, d_total]
        K = self.W_K(R_K_norm)                   # [N, B, S, d_total]
        K = K + depth_embed.unsqueeze(1).unsqueeze(2)
        K = self.norm_k(K)
        V = self.W_V(R_norm)

        d_head = self.d_model // self.n_heads
        K = K.view(N, B, S, self.n_heads, d_head).permute(1, 2, 3, 0, 4)  # [B,S,H,N,d]
        V = V.view(N, B, S, self.n_heads, d_head).permute(1, 2, 3, 0, 4)
        Q = Q.view(B, S, self.n_heads, d_head).unsqueeze(3)               # [B,S,H,1,d]

        scale = d_head ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, V).squeeze(3)  # [B,S,H,d]
        out = out.reshape(B, S, -1)

        delta = self.W_O(self.norm_out(self.dropout(out)))
        r_out = sources[-1] + delta

        # -- gate -------------------------------------------------------------
        gate_input = self.norm_gate(
            torch.cat([sources[-1], r_out], dim=-1)
        )  # [B, S, 2*d]
        gate = torch.sigmoid(self.W_gate(gate_input))          # [B, S, 1]

        return r_out, weights.squeeze(3), gate


def create_edap_plugins(d_model=3584, n_heads=4, n_blocks=4, dropout=0.1,
                         shared_kv: bool = False):
    """Build one EDAP plugin per block boundary.

    Plugin i sees: embedding + all previous calibrated residuals + current block.
    So n_sources goes 2, 3, 4, 5, ... for n_blocks blocks.

    When shared_kv=True, W_K and W_V are shared across all plugins, reducing
    parameter count by ~1/3 and encouraging depth-consistent retrieval patterns.
    """
    d_head = d_model // n_heads
    d_total = d_head * n_heads
    W_K_shared = nn.Linear(d_model, d_total, bias=False) if shared_kv else None
    W_V_shared = nn.Linear(d_model, d_total, bias=False) if shared_kv else None

    return nn.ModuleList([
        EDAPPlugin(
            n_sources=i + 2, d_model=d_model, n_heads=n_heads, dropout=dropout,
            W_K_shared=W_K_shared, W_V_shared=W_V_shared,
        )
        for i in range(n_blocks)
    ])


def edap_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    edap_plugins: nn.ModuleList,
    block_exits: List[int],
    compute_dtype: torch.dtype,
    shuffle_depth: bool = False,
    delta_mode: bool = True,
    gate_mode: bool = True,
    edap_noise: float = 0.0,
    collect_weights: bool = False,
    lm_head_bottleneck: Optional[nn.Module] = None,
):
    """Run frozen backbone interleaved with EDAP at block boundaries.

    Forward flow (example with N blocks):

        emb → [L0..e0]  → b0 → EDAP0(emb, b0) = r0
                                  │ gated mix: gate·r0 + (1-gate)·b0
                                  ▼
               [e0+1..e1] → b1 → EDAP1(emb, r0, b1) = r1
                                  │ gated mix
                                  ▼
               ...
                                  ▼
               [e_{N-2}+1..e_{N-1}] → b_{N-1} → EDAP_{N-1}(...) = r_{N-1}
                                  → lm_head

    All backbone layers run under torch.no_grad(); EDAP plugins retain gradients.
    Gated mixing prevents EDAP from over-modifying non-conflict tokens.
    Delta-mode K gives higher contrast source weights (prevents routing collapse).

    Args:
        model:          Qwen2ForCausalLM with frozen backbone.
        input_ids:      [B, S] token ids.
        attention_mask: [B, S] or None.
        edap_plugins:   ModuleList of EDAPPlugin, one per block boundary.
        block_exits:    layer indices that mark block boundaries.
        compute_dtype:  torch.bfloat16 / float16 for EDAP computation.
        shuffle_depth:  randomise source order (ablation).
        delta_mode:     use incremental deltas for K (default True).
        gate_mode:      use learnable per-token gate mixing (default True).
        collect_weights: if True, also return cross-depth attention weights.

    Returns:
        logits:  [B, S, V] prediction logits from lm_head.
        (optionally) all_weights: list of [B, S, H, N] per-plugin attention.
    """
    device = input_ids.device
    B, S = input_ids.shape

    # --- build block ranges ---
    block_ranges = []
    prev_end = -1
    for exit_layer in block_exits:
        block_ranges.append((prev_end + 1, exit_layer))
        prev_end = exit_layer

    # --- embedding & mask ---
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        cache_position = torch.arange(S, device=device)
        causal_mask = model.model._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=hidden_states,
            cache_position=cache_position,
            past_key_values=None,
            output_attentions=False,
        )
        position_ids = cache_position.unsqueeze(0).expand(B, -1)

    emb = hidden_states.detach().to(compute_dtype)
    fused_outputs: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []
    all_gates: List[torch.Tensor] = []
    current = hidden_states

    # --- interleaved blocks + EDAP ---
    for blk_idx, (start, end) in enumerate(block_ranges):
        # 1. Run this block's transformer layers (no grad)
        with torch.no_grad():
            for layer_idx in range(start, end + 1):
                current = model.model.layers[layer_idx](
                    current,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                )[0]

        # 2. Capture block exit (detached from graph)
        block_out = current.detach().to(compute_dtype)

        # 3. EDAP cross-depth fusion (gradients flow through EDAP params)
        sources: List[torch.Tensor] = [emb] + fused_outputs + [block_out]
        fused, weights, gate = edap_plugins[blk_idx](
            sources, shuffle_depth=shuffle_depth, delta_mode=delta_mode,
            edap_noise=edap_noise,
        )
        fused_outputs.append(fused)
        if collect_weights:
            all_weights.append(weights)
            all_gates.append(gate)

        # 4. Gated mixing or hard replacement for intermediate blocks;
        #    the last block's gate is applied below before lm_head.
        if blk_idx < len(edap_plugins) - 1:
            if gate_mode:
                current = (gate * fused + (1 - gate) * block_out).to(current.dtype)
            else:
                current = fused.to(current.dtype)

    # --- lm_head ---
    hidden = fused_outputs[-1]
    if gate_mode:
        # Apply the last EDAP plugin's gate to the final fused output.
        # Previously the last gate was computed but never used, wasting its params.
        hidden = (gate * hidden + (1 - gate) * block_out).to(hidden.dtype)
    # Final RMSNorm: the vanilla Qwen forward applies model.model.norm before
    # lm_head. Skipping it fed hidden states at RMS≈9.2 (vs the ~5 the frozen
    # lm_head was trained on), exploding CE from ~10 → ~144.
    hidden = model.model.norm(hidden.to(hidden.dtype))
    if lm_head_bottleneck is not None:
        hidden = lm_head_bottleneck(hidden.to(compute_dtype))
    logits = model.lm_head(hidden)

    if collect_weights:
        return logits, all_weights, all_gates
    return logits
