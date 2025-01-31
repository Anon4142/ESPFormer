import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import ot 
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from attentions.esp import Esp



from util import sample_and_knn_group


class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GammaAggregator(nn.Module):
    def __init__(self, mode="mean", num_heads=None, num_slices=None, temperature=0.1):
        super(GammaAggregator, self).__init__()
        self.mode = mode
        self.temperature = temperature

        if mode == "learnable_manual":
            if num_heads is None or num_slices is None:
                raise ValueError("num_heads and num_slices must be provided for learnable_manual mode.")
            self.weights = nn.Parameter(torch.randn(num_heads, num_slices))  # Shape: [H, L]
        elif mode == "closed_form":
            pass 
        elif mode not in ["mean", "learnable_manual"]:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, Gamma, x=None, y=None):
        if self.mode == "mean":
            return torch.mean(Gamma, dim=2)  # Mean over slices
        elif self.mode == "learnable_manual":
            return self.learnable_manual_solution(Gamma)
        elif self.mode == "closed_form":
            return self.closed_form_solution(Gamma, x, y)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.mode}")

    def learnable_manual_solution(self, Gamma):

        normalized_weights = F.softmax(self.weights, dim=-1)  # Shape: [H, L]
        normalized_weights = normalized_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: [1, H, L, 1, 1]
        Gamma_weighted = Gamma * normalized_weights
        return Gamma_weighted.sum(dim=2)

    def closed_form_solution(self, Gamma, x=None, y=None):

        if x is None or y is None:
            raise ValueError("For closed_form mode, x and y must be provided.")
        cost = torch.cdist(x, y, p=2)
        swds = (cost.unsqueeze(2) * Gamma).sum(dim=(-1, -2)) 
        min_swds = swds.min(dim=-1, keepdim=True).values  
        exp_swds = torch.exp(-self.temperature * (swds - min_swds)) 
        weights = exp_swds / exp_swds.sum(dim=-1, keepdim=True)  
        Gamma_weighted = Gamma * weights.unsqueeze(-1).unsqueeze(-1)
        return Gamma_weighted.sum(dim=2)  # Sum over slices, shape: [B, H, N, N]


class SoftSort_p2(torch.nn.Module):
    def __init__(self, tau=1e-3, hard=False):
        super(SoftSort_p2, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        scores = scores.transpose(3,2) # Shape: B x H x L x N
        scores = scores.unsqueeze(-1)  # Shape: B x H x L x N x 1
        sorted_scores, _ = scores.sort(dim=3, descending=False )  # Shape: B x H x L x N x 1
        pairwise_diff = ((scores.transpose(4, 3) - sorted_scores) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(dim=-1)
        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat 
        return P_hat.squeeze(-1)

class SA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., interp=None, learnable=True, agg_mode="closed_form", temperature=.1, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.esp = Esp(d_in=dim_head, heads=heads, interp=interp, agg_mode=agg_mode, temperature=temperature)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = self.esp(q * self.scale, k * self.scale)
        attn = attn * attn.shape[-1]
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_xyz, new_feature


class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512, 256]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg2 = SG(s=samples[1], in_channels=256, out_channels=256)
    
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
        _, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]

        return features2


class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x


if __name__ == '__main__':
    """
    Please be careful to excute the testing code, because
    it may cause the GPU out of memory.
    """
    
    pc = torch.rand(32, 3, 1024).to('cuda')

    # testing for Embedding
    embedding = Embedding().to('cuda')
    out = embedding(pc)
    print("Embedding output size:", out.size())

    # testing for SA
    sa = SA(channels=out.size(1)).to('cuda')
    out = sa(out)
    print("SA output size:", out.size())

    # testing for SG
    coords = torch.rand(32, 1024, 3).to('cuda')
    features = torch.rand(32, 64, 1024).to('cuda')
    sg = SG(512, 128, 128).to('cuda')
    new_coords, out = sg(features, coords)
    print("SG output size:", new_coords.size(), out.size())

    # testing for NeighborEmbedding
    ne = NeighborEmbedding().to('cuda')
    out = ne(pc)
    print("NeighborEmbedding output size:", out.size())

    # testing for OA
    oa = OA(256).to('cuda')
    out = oa(out)
    print("OA output size:", out.size())