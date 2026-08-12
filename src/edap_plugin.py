"""EDAP: cross-depth attention at block boundaries for knowledge conflict."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EDAPPlugin(nn.Module):

    def __init__(self, n_sources: int, d_model: int = 3584, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sources = n_sources
        self.d_model = d_model
        self.n_heads = n_heads
        d_head = d_model // n_heads
        d_total = d_head * n_heads

        self.W_Q = nn.Linear(d_model, d_total, bias=False)
        self.W_K = nn.Linear(d_model, d_total, bias=False)
        self.W_V = nn.Linear(d_model, d_total, bias=False)
        self.W_O = nn.Linear(d_total, d_model, bias=False)

        self.norm_in = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_total)
        self.norm_out = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.depth_embed = nn.Parameter(torch.randn(n_sources, d_total) * 0.1)

        # init W_O near zero so the plugin starts as identity
        nn.init.normal_(self.W_O.weight, mean=0.0, std=1e-5)

    def forward(self, sources: list, shuffle_depth: bool = False):
        """Run cross-depth attention.

        sources: list of [B, S, d_model], last element is current block.
        shuffle_depth: if True, permute source order for control experiment.
        """
        B, S, _ = sources[-1].shape
        N = len(sources)

        R = torch.stack(sources, dim=0)  # [N, B, S, d]

        if shuffle_depth and N > 1:
            perm = torch.randperm(N - 1, device=R.device)
            prev = R[:-1][perm]
            R = torch.cat([prev, R[-1:]], dim=0)
            depth_embed = torch.cat(
                [self.depth_embed[:-1][perm], self.depth_embed[-1:]], dim=0
            )
        else:
            depth_embed = self.depth_embed

        R = self.norm_in(R)

        Q = self.W_Q(R[-1])  # [B, S, d_total]
        K = self.W_K(R)      # [N, B, S, d_total]
        K = K + depth_embed.unsqueeze(1).unsqueeze(2)
        K = self.norm_k(K)
        V = self.W_V(R)

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

        return r_out, weights.squeeze(3)


def create_edap_plugins(d_model=3584, n_heads=4, n_blocks=4, dropout=0.1):
    """Build one EDAP plugin per block boundary.

    Plugin i sees: embedding + all previous calibrated residuals + current block.
    So n_sources goes 2, 3, 4, 5 for 4 blocks.
    """
    return nn.ModuleList([
        EDAPPlugin(n_sources=i + 2, d_model=d_model, n_heads=n_heads, dropout=dropout)
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
):
    """Run frozen backbone interleaved with EDAP at block boundaries.

    EDAP-fused residual *replaces* the hidden state flowing into the next block,
    unlike the original post-hoc design where EDAP ran only after all 28 layers.

    Forward flow (example 4 blocks on 28 layers):

        emb → [L0..L6]  → b0 → EDAP0(emb, b0) = r0
                                  │ inject r0 as residual
                                  ▼
               [L7..L13] → b1 → EDAP1(emb, r0, b1) = r1
                                  │ inject r1 as residual
                                  ▼
               [L14..L20] → b2 → EDAP2(emb, r0, r1, b2) = r2
                                  │ inject r2 as residual
                                  ▼
               [L21..L27] → b3 → EDAP3(emb, r0, r1, r2, b3) = r3 → lm_head

    All backbone layers run under torch.no_grad(); EDAP plugins retain gradients.

    Args:
        model:          Qwen2ForCausalLM with frozen backbone.
        input_ids:      [B, S] token ids.
        attention_mask: [B, S] or None.
        edap_plugins:   ModuleList of EDAPPlugin, one per block boundary.
        block_exits:    layer indices that mark block boundaries, e.g. [6,13,20,27].
        compute_dtype:  torch.bfloat16 / float16 for EDAP computation.
        shuffle_depth:  randomise source order (ablation).

    Returns:
        logits:  [B, S, V] prediction logits from lm_head.
    """
    device = input_ids.device
    B, S = input_ids.shape

    # ---- build block ranges ------------------------------------------------
    block_ranges = []
    prev_end = -1
    for exit_layer in block_exits:
        block_ranges.append((prev_end + 1, exit_layer))
        prev_end = exit_layer

    # ---- embedding & mask --------------------------------------------------
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
    current = hidden_states  # requires_grad=False from frozen embed

    # ---- interleaved blocks + EDAP -----------------------------------------
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

        # 3. EDAP cross-depth fusion  (gradients flow through EDAP params)
        sources: List[torch.Tensor] = [emb] + fused_outputs + [block_out]
        fused, _weights = edap_plugins[blk_idx](
            sources, shuffle_depth=shuffle_depth,
        )
        fused_outputs.append(fused)

        # 4. Inject fused residual as input to the NEXT block
        if blk_idx < len(edap_plugins) - 1:
            current = fused.to(current.dtype)

    # ---- lm_head -----------------------------------------------------------
    logits = model.lm_head(fused_outputs[-1])
    return logits
