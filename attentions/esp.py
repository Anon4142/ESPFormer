import torch 
import numpy as np
import ot 
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        sorted_scores, _ = scores.sort(dim=3, descending=True)  # Shape: B x H x L x N x 1
        pairwise_diff = ((scores.transpose(4, 3) - sorted_scores) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(dim=-1)
        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat 
        return P_hat.squeeze(-1)



class Esp(nn.Module):
    def __init__(self, d_in, heads=8, tau=1e-3, interp=None, agg_mode="closed_form", temperature=.1):
        super(Esp, self).__init__()

        self.agg_mode = agg_mode
        self.softsort = SoftSort_p2(tau=tau)
        self.interp = interp
        self.aggregator = GammaAggregator(
            mode=agg_mode, 
            num_heads=heads, 
            num_slices=d_in if agg_mode == "learnable_manual" else None, 
            temperature=temperature
        )
            
    def forward(self, X, Y):
        B, H, N, L = X.shape
    
        Pu = self.softsort(X)  
        Pv = self.softsort(Y)  
    
        # Compute Gamma
        if self.interp is None:
            Gamma = Pu.transpose(-1, -2) @ Pv  # Shape: [B, H, L, N, N]
        else:
            interp_expanded = self.interp.unsqueeze(0).unsqueeze(0).repeat(B, X_line.shape[-1], 1, 1).to(X.device)
            Pu = Pu.unsqueeze(-1) if Pu.shape[-1] == 1 else Pu
            Pv = Pv.unsqueeze(-1) if Pv.shape[-1] == 1 else Pv
            Gamma = Pu.transpose(-1, -2) @ interp_expanded @ Pv
    
        Gamma_hat = self.aggregator(Gamma, x=X, y=Y) if self.agg_mode=="closed_form" else self.aggregator(Gamma)
        return Gamma_hat

class EspAttention(nn.Module):
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


#####
#Attention with key padding mask
class EspGraph(nn.Module):
    def __init__(self, d_in, heads=8, tau=1e-3, interp=None, agg_mode="closed_form", temperature=10):
        super(EspGraph, self).__init__()

        self.agg_mode = agg_mode
        self.softsort = SoftSort_p2(tau=tau)
        self.interp = interp
        self.aggregator = GammaAggregator(
            mode=agg_mode, 
            num_heads=heads, 
            num_slices=d_in if agg_mode == "learnable_manual" else None, 
            temperature=temperature
        )
            
    def forward(self, X, Y, mask=None):
        
        B, H, N, L = X.shape

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, L)
            fill_value = 1e+18
            X = torch.where(expanded_mask, X, torch.full_like(X, fill_value))
            Y = torch.where(expanded_mask, Y, torch.full_like(Y, fill_value))
        
        Pu = self.softsort(X)  
        Pv = self.softsort(Y)  
    
        # Compute Gamma
        if self.interp is None:
            Gamma = Pu.transpose(-1, -2) @ Pv  # Shape: [B, H, L, N, N]
        else:
            interp_expanded = self.interp.unsqueeze(0).unsqueeze(0).repeat(B, X_line.shape[-1], 1, 1).to(X.device)
            Pu = Pu.unsqueeze(-1) if Pu.shape[-1] == 1 else Pu
            Pv = Pv.unsqueeze(-1) if Pv.shape[-1] == 1 else Pv
            Gamma = Pu.transpose(-1, -2) @ interp_expanded @ Pv
    
        Gamma_hat = self.aggregator(Gamma, x=X, y=Y) if self.agg_mode=="closed_form" else self.aggregator(Gamma)
        return Gamma_hat


class EspGraphAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., interp=None, learnable=True, agg_mode="closed_form", temperature=10, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.esp = EspGraph(d_in=dim_head, heads=heads, interp=interp, agg_mode=agg_mode, temperature=temperature)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        if mask is not None:
            if mask.dim() == 4:  # Case: (B, H, N, N)
                reduced_mask = mask.any(dim=1).any(dim=2)  # Reduce over heads and query dimensions
            elif mask.dim() == 3:  # Case: (B, N, N)
                reduced_mask = mask.any(dim=1)  # Reduce over query dimensions
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected 3D or 4D tensor.")
        else:
            reduced_mask = None
        attn = self.esp(q * self.scale, k * self.scale, reduced_mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn
