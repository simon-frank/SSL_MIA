import torch.nn as nn
from models.barlowtwins import BarlowTwinsLit
import torchvision

"""
Factory for creating models used for the pretrained training
"""


def loadModel(config: dict)-> nn.Module:
    #define backbone
    if config["pretraining"]["method"]["name"]== 'BarlowTwins':
        if config["pretraining"]["method"]["backbone"]["name"] == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True)
            backbone.fc = nn.Identity()
            return BarlowTwinsLit.load_from_checkpoint(config["evaluation"]["modelpath"], backbone=backbone, config=config)

    else:
        raise ValueError("No valid model name given")

def createModel(config: dict)-> nn.Module:
    #define backbone
    if config["pretraining"]["method"]["name"]== 'BarlowTwins':
        if config["pretraining"]["method"]["backbone"]["name"] == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True)
            backbone.fc = nn.Identity()
            return BarlowTwinsLit(backbone, config)

    else:
        raise ValueError("No valid model name given")