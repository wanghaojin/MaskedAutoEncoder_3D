import torch
import torch.nn as nn
from utils import to_pair, Transformer

class ViT(nn.Module):
    def __init__(
        self, image_size, patch_size, 
        num_classes=1000, dim=1024, depth=6, num_heads=8, mlp_dim=2048,
        pool='cls', channels=3, dim_per_head=64, dropout=0., embed_dropout=0.
    ):
        super().__init__()

        img_h, img_w = to_pair(image_size)
        self.patch_h, self.patch_w = to_pair(patch_size)
        assert not img_h % self.patch_h and not img_w % self.patch_w
        num_patches = (img_h // self.patch_h) * (img_w // self.patch_w)

        assert pool in {'cls', 'mean'}
        
        patch_dim = channels * self.patch_h * self.patch_w
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(p=embed_dropout)

        self.transformer = Transformer(
            dim, mlp_dim, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        b, c, img_h, img_w = x.shape
        assert not img_h % self.patch_h and not img_w % self.patch_w
        
        num_patches = (img_h // self.patch_h) * (img_w // self.patch_w)
        patches = x.view(
            b, c, 
            img_h // self.patch_h, self.patch_h, 
            img_w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        tokens = self.patch_embed(patches)
        tokens = torch.cat([self.cls_token.repeat(b, 1, 1), tokens], dim=1)
        tokens += self.pos_embed[:, :(num_patches + 1)]
        tokens = self.dropout(tokens)

        enc_tokens = self.transformer(tokens)
        pooled = enc_tokens[:, 0] if self.pool == 'cls' else enc_tokens.mean(dim=1)
        logits = self.mlp_head(pooled)
        return logits