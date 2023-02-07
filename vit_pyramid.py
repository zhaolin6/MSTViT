import random
import numpy as np
import torch
import cv2
import math
from torch import nn
from skimage import transform
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, w=0, h=0):
        return self.fn(self.norm(x))


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, stride=1):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., channels = 10, num_patches=10):
        super().__init__()
        self.net = nn.Sequential(
            DEPTHWISECONV(channels, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=channels, kernel_size=1)
        )

    def forward(self, x):
        b, d, c = x.shape
        w = int(math.sqrt(d))
        x1 = rearrange(x, 'b (w h) c -> b c w h', w=w, h=w)
        x1 = self.net(x1)
        x1 = rearrange(x1, 'b c w h -> b (w h) c')
        x = x + x1
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, w=0, h=0):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, channels, dim, depth, heads, dim_head, mlp_dim, dropout = 0., num_patches=10):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.index = 0
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, channels=channels, num_patches=num_patches)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (num_patches ), dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, dim, depth, heads, dim_head, mlp_dim, dropout, num_patches=num_patches)

        self.small_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=3, p2=3),
            nn.Linear(3*3*channels, dim),
        )
        self.small_transformer = Transformer(dim, dim, depth, heads, dim_head, mlp_dim, dropout, num_patches=25)
        self.small_pos_embedding = nn.Parameter(torch.randn(1, 5*5 + 1, dim))
        self.small_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #####################################################

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(3400),
            nn.Linear(3400, num_classes)
        )
        ####### 梯形
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU())
        self.conv21 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU())
        self.conv22 = nn.Sequential(nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU())
        self.conv31 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU()
                                    )
        self.conv32 = nn.Sequential(nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(),)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, img, test=0):

        if len(img.shape) == 5: img = img.squeeze()
        # bands_ori = torch.mean(torch.mean(img, dim=2), dim=2)
        bands_ori = img

        img1 = self.conv11(img)
        img2 = self.conv21(img)
        img2 = self.conv22(torch.cat((img2, img1), dim=1))
        img3 = self.conv31(img)
        img3 = self.conv32(torch.cat((img3, img2), dim=1))
        img = self.pool(img3)
        img = self.conv4(img)

        # bands_3d2d = torch.mean(torch.mean(img, dim=2), dim=2)
        bands_3d2d = img

        spatial = self.to_patch_embedding(img)
        b, n, c = spatial.shape
        spatial = spatial + self.pos_embedding[:, :n]
        spatial = self.dropout(spatial)
        spatial = self.transformer(spatial)

        # bands_big = torch.mean(spatial, dim=2)
        bands_big = spatial

        small = self.small_to_patch_embedding(img)
        b, n, c = small.shape
        small = small + self.small_pos_embedding[:, :n]
        small = self.dropout(small)
        small = self.small_transformer(small)

        # bands_small = torch.mean(small, dim=2)
        bands_small = small

        x = torch.cat((spatial, small), dim=1)

        bands_all = torch.cat((bands_small, bands_big), dim=1)

        x = x.contiguous().view(x.shape[0], -1)
        x = self.to_latent(x)
        if test:
            return self.mlp_head(x), [bands_ori.flatten(1).cpu(), bands_3d2d.flatten(1).cpu(), bands_big.flatten(1).cpu(), bands_small.flatten(1).cpu(), bands_all.flatten(1).cpu()]
        else:
            return self.mlp_head(x)