from torch import nn
import torch
import torchvision


"""
Adapted from https://github.com/facebookresearch/barlowtwins
"""

class BarlowTwins(nn.Module):
    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # projector
        self.projector = projector

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        return z1, z2





def _off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def BarlowLoss(z1, z2, w):
    cross_correlation = z1.T @ z2

    on_diag = torch.diagonal(cross_correlation).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(cross_correlation).pow_(2).sum()
    loss = on_diag +  w * off_diag
    return loss

