"""EDAP: cross-depth attention at block boundaries for knowledge conflict."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDAPPlugin(nn.Module):

    def __init__(self, n_sources: int, d_model: int = 3584, n_heads: int = 4):
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

        self.depth_embed = nn.Parameter(torch.randn(n_sources, d_total) * 0.02)

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

        delta = self.W_O(self.norm_out(out))
        r_out = sources[-1] + delta

        return r_out, weights.squeeze(3)


def create_edap_plugins(d_model=3584, n_heads=4, n_blocks=4):
    """Build one EDAP plugin per block boundary.

    Plugin i sees: embedding + all previous calibrated residuals + current block.
    So n_sources goes 2, 3, 4, 5 for 4 blocks.
    """
    return nn.ModuleList([
        EDAPPlugin(n_sources=i + 2, d_model=d_model, n_heads=n_heads)
        for i in range(n_blocks)
    ])
