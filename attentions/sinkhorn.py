import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# and from https://github.com/dfdazac/wassdistance/blob/master/layers.py

class SinkhornDistance(nn.Module):

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c):

        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()

        if mu.dim() < 2:
            mu = mu.view(-1, 1)

        if nu.dim() < 2:
            nu = nu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Stopping criterion
        thresh = 1e-12

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) -
                                torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
                # v = v.detach().requires_grad_(False)
                # v[v == float('inf')] = 0.0
                # v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                print('breaking')
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))


        return pi, C, U, V

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps



    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    

class SinkAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., max_iter=3, eps=1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_iter = max_iter
        self.sink = SinkhornDistance(eps=eps, max_iter=max_iter)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #
        dots_former_shape = dots.shape
        dots = dots.view(-1, dots_former_shape[2], dots_former_shape[3])
        attn = self.sink(dots)[0]
        attn = attn * attn.shape[-1]
        attn = attn.view(dots_former_shape)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn
    

