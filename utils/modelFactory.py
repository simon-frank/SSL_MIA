import torch.nn as nn
from models.barlowtwins import BarlowTwinsLit
import torchvision

"""
Factory for creating models used for the pretrained training
"""


def createModel(config: dict)-> nn.Module:
    #define backbone
    if config["pretraining"]["method"]== 'BarlowTwins':
        if config["pretraining"]["method"]["backbone"]["name"] == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True)
            backbone.fc = nn.Identity()
            return BarlowTwinsLit(backbone, config)

    else:
        raise ValueError("No valid model name given")