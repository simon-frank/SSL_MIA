from torch import nn
import torch.nn.functional as F


"""
minimalistic readout head
"""
class ReadoutHead(nn.Module):
  def __init__(self, inputDim, outputDim):
    super().__init__()
    self.MLP = nn.Linear(inputDim, outputDim)

  def forward(self,x):
    x = self.MLP(x)
    x = F.softmax(x, dim=1)
    return x
  
