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
    self.trainall = config['finetuning']['trainall']
    self.readoutHead = ReadoutHead
    self.backbone = backbone
    self.loss = nn.functional.cross_entropy

  def forward(self, x):
    x = self.readoutHead(self.backbone(x))
    return x
  

  def calc_accuracy(self, y_hat, y):
    _, predicted_labels = torch.max(y_hat, dim=1)
    
    # Compare predicted labels with ground truth labels
    correct = (predicted_labels == y).sum().item()
    total = y.size(0)
    
    # Calculate accuracy
    accuracy = correct / total * 100.0
    return accuracy

  def validation_step(self, batch, batch_idx):
    # Validation logic
    x, y = batch
    y_hat = self.forward(x)
    loss = self.loss(y_hat, y)
    self.log('val_loss', loss, prog_bar=True)  # Log the validation loss
    self.log('val_acc', self.calc_accuracy(y_hat, y), prog_bar=True)  # Log the validation accuracy
    return loss
  
  def training_step(self, batch, batch_index):
    x,y = batch
    y_hat = self.forward(x)
    loss = self.loss(y_hat,y)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def configure_optimizers(self):
    if self.trainall:
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    else:
      optimizer = torch.optim.Adam(self.readoutHead.parameters(), lr=self.lr)
    return optimizer
