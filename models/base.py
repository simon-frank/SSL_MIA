import torchvision.models as models
from torch import nn
import torch
import pytorch_lightning as pl


class Base50(nn.Module):
  def __init__(self, ReadoutHead):
    super().__init__()
    self.resnet50 = models.resnet50(pretrained=True)
    self.resnet50.eval() #wie laufzeit optimieren
    self.readoutHead = ReadoutHead

  def forward(self,x):
    x = self.readoutHead(self.resnet50(x))
    return x

class Base18(nn.Module):
    def __init__(self, ReadoutHead):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.eval() #wie laufzeit optimieren
        self.readoutHead = ReadoutHead
    
    def forward(self,x):
        x = self.readoutHead(self.resnet18(x))
        return x
    


class Base(pl.LightningModule):
  def __init__(self, backbone, ReadoutHead, config: dict):
    super().__init__()
    self.lr = config['finetuning']['lr']
    self.optim = config['optimizer']
    self.readoutHead = ReadoutHead
    self.backbone = backbone
    self.loss = nn.functional.cross_entropy

  def forward(self, x):
    x = self.readoutHead(self.backbone(x))
    return x
  
  def training_step(self, batch, batch_index):
    x,y = batch
    y_hat = self.forward(x)
    loss = self.loss(y_hat,y)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def configure_optimizers(self):
     optimizer = torch.optim.Adam(self.readoutHead.parameters(), lr=self.lr)
     return optimizer
