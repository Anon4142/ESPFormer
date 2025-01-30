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
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(num_hidden, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, num_hidden),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (b, h, n, d) x (b, h, d, n) -> (b, h, n, n)
        attn = self.attend(dots)  #softmax over last dimension
        out = torch.matmul(attn, v)
        # Merge heads back: (b, h, n, d) -> (b, n, h*d)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        return out, attn #embeddings and attention weights
