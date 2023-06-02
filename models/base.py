import torchvision.models as models
from torch import nn

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