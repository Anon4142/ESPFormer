import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class MultiHeadVanillaAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=8, d_k=64, dropout=0.):
        super().__init__()
        inner_dim = d_k * num_heads
        project_out = not (num_heads == 1 and d_k == num_hidden)
        self.heads = num_heads
        self.scale = d_k ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(num_hidden, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, num_hidden),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        if attn_mask is not None:
            # If mask is 2D (batch_size, seq_len), expand it to 3D (batch_size, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).expand(-1, x.size(1), -1)
            #Attn_mask is usually (batch_size, seq_len, seq_len) upon input
            # Unsqueeze it for the heads dimension to match dots: (batch_size, 1, seq_len, seq_len)
            dots = dots.masked_fill(attn_mask.unsqueeze(1), float('-inf')) #Broadcasts, saves memory

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn
