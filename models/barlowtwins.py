from torch import nn
import torch
import torchvision
import pytorch_lightning as pl
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

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


class BarlowTwinsLit(pl.LightningModule):
    def __init__(self, backbone, config:dict):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(config['input_size'], config['hidden_size'], config['output_size'])
        self.criterion = BarlowTwinsLoss(gather_distributed=True)
        self.lr = config['lr']
        self.optim = config['optimizer']
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim = 1)
        return self.projection_head(x)
    
    def training_step(self, batch, batch_index):
         (x1, x2),_,_ = batch
         z1 = self.forward(x1)
         z2 = self.forward(x2)
         loss = self.criterion(z1,z2)
         return loss
    
    def configure_optimizers(self):
         optimizer = self.optim(self.parameters(), lr = self.lr)
         return optimizer


