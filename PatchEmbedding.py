import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size).to(device))
        self.projection = nn.Sequential(
            # Break the image into s1 x s2 patches and flatten them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]  # batch_size
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand cls_token for the batch
        x = self.projection(x)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class PatchEmbedding_CNN(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size).to(device))
        self.projection = nn.Sequential(
            # Use a convolutional layer for better performance
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]  # batch_size
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand cls_token for the batch
        x = self.projection(x)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
