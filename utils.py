import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = net
    
    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        b, l, d = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).reshape(b, l, -1)
        out = self.out(z)
        return out

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ffn(x)
        return x

class TIFImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_files = [
            f for f in os.listdir(data_dir) 
            if f.lower().endswith('.tif')
        ]
        self.image_files.sort()  
        
        print(f"Found {len(self.image_files)} TIF images in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = image.convert('L')
        if self.transform:
            image = self.transform(image)
        return image, 0 

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform 
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        data, _ = self.subset[idx]  
        if not isinstance(data, torch.Tensor):
            data = data.convert('L')

        if self.transform:
            data = self.transform(data)
        return data, 0 

def calculate_dataset_stats(dataloader):
    print("Calculating...")
    mean = 0.
    std = 0.
    total_images = 0
    
    pbar = tqdm(dataloader, desc="Calculating the mean and std")
    for images, _ in pbar:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)  
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    return mean, std

def interpolate_pos_embed(model, checkpoint_model):
    # 如果后续需要改变图像尺寸再使用，此处保留
    if 'pos_embed' not in checkpoint_model:
        return

    pos_embed_ckpt = checkpoint_model['pos_embed']
    pos_embed_current = model.encoder.pos_embed
    embedding_dim = pos_embed_ckpt.shape[-1]

    num_extra_tokens = 1
    orig_num_patches = pos_embed_ckpt.shape[1] - num_extra_tokens
    new_num_patches = pos_embed_current.shape[1] - num_extra_tokens

    if orig_num_patches == new_num_patches:
        return

    orig_size = int(orig_num_patches**0.5)
    new_size = int(new_num_patches**0.5)
    print(f"Interpolating pos_embed from {orig_size}x{orig_size} to {new_size}x{new_size}")

    cls_pos_embed = pos_embed_ckpt[:, :num_extra_tokens, :]
    pos_tokens = pos_embed_ckpt[:, num_extra_tokens:, :]

    pos_tokens = pos_tokens.view(1, orig_size, orig_size, embedding_dim).permute(0,3,1,2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
    )
    pos_tokens = pos_tokens.permute(0,2,3,1).reshape(1, new_size*new_size, embedding_dim)
    new_pos_embed = torch.cat((cls_pos_embed, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed