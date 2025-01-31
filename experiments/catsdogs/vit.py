import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from attentions.esp import EspAttention  # Import new attention mechanisms
from attentions.dif import MultiHeadDifAttention
from attentions.vanilla import MultiHeadVanillaAttention
from attentions.sinkhorn import SinkAttention

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 attention_type='learnable', max_iter=1, eps=1, interp=None):
        """
        :param attention_type: 'frozen' for FEspAttention, 'learnable' for LEspAttention
        :param interp: Interpolation matrix for LEspAttention
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        #attention_cls = MultiHeadVanillaAttention #VANILLA ATTENTION
        attention_cls = MultiHeadDifAttention #DIFFERENTIAL ATTENTION
        attention_cls = SinkAttention #SINKFORMER
        #attention_cls = EspAttention

        learnable = False if attention_type=='frozen' else True

        #seq_len is for DifAttention # of patches = 224^2/16^2 + 1 (+1 for CLS token)
        seq_length = 197
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, attention_cls(dim, num_heads=heads, attn_drop=dropout)),
                # PreNorm(dim, attention_cls(dim, heads=heads)), 
                PreNorm(dim, attention_cls(num_hidden=dim, num_heads=heads, seq_len=seq_length, d_k=dim_head)), #for DIFFERENTIAL
                #PreNorm(dim, attention_cls(dim, heads = heads, dim_head = dim_head, dropout = dropout, max_iter=max_iter, eps=eps)), #FOR SINKFORMER
                #PreNorm(dim, attention_cls(num_hidden=dim, num_heads=heads, d_k=dim_head)), #for VANILLA

                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
    ]))


    def forward(self, x):
        attn_weights = []
        for attn, ff in self.layers:
            attn_x, attn_matrix = attn(x) 
            x = attn_x + x
            x = ff(x) + x
            attn_weights.append(attn_matrix.cpu().detach().numpy())
        return x, attn_weights

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 attention_type='learnable', interp=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                       attention_type=attention_type, interp=interp)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        trans_x, attn_weights = self.transformer(x)
        x = trans_x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)