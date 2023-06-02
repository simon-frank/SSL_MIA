from torch import nn
import torch.nn.functional as F

class ReadoutHead(nn.Module):
  def __init__(self, hiddenLayers,inputDim, outputDim):
    super().__init__()
    layers =[]
    prevDim = inputDim
    for i in range(hiddenLayers-1):
      nextDim = int(prevDim/2)
      layers.append(nn.Linear(prevDim,nextDim))
      layers.append(nn.BatchNorm1d(nextDim))
      layers.append(nn.ReLU(inplace=True))
      prevDim = nextDim
    layers.append(nn.Linear(prevDim, outputDim))
    self.MLP = nn.Sequential(*layers)

  def forward(self,x):
    x = self.MLP(x)
    x = F.softmax(x, dim=1)
    return x