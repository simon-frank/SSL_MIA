import torch.nn as nn
from models.barlowtwins import BarlowTwinsLit
from models.vicregl import VicRegL
import torchvision
from nn.readoutHead import ReadoutHead
from models.base import Base
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
    elif config["pretraining"]["method"]["name"]== 'VicRegL':
        backbone = None
        if config["pretraining"]["method"]["backbone"]["name"] == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True)
        elif config["pretraining"]["method"]["backbone"]["name"] == 'resnet50': 
            backbone = torchvision.models.resnet50(zero_init_residual=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        return VicRegL.load_from_checkpoint(config["evaluation"]["modelpath"], backbone = backbone, config = config)
    else:
        raise ValueError("No valid model name given")

def createModel(config: dict)-> nn.Module:
    #define backbone
    if config["pretraining"]["method"]["name"]== 'BarlowTwins':
        if config["pretraining"]["method"]["backbone"]["name"] == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True)
            backbone.fc = nn.Identity()
            return BarlowTwinsLit(backbone, config)
    elif config["pretraining"]["method"]["name"]== 'VicRegL':
        backbone = None
        if config["pretraining"]["method"]["backbone"]["name"] == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True)
        elif config["pretraining"]["method"]["backbone"]["name"] == 'resnet50':
            backbone = torchvision.models.resnet50(zero_init_residual=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        return VicRegL(backbone, config)

    else:
        raise ValueError("No valid model name given")
    

def createFinetuningModel(config)->nn.Module:
    # get pretrained model
    if config['finetuning']['pretrained']:
        backbone = torchvision.models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        backbone.eval()
    else:
        model = loadModel(config)
        model.eval()
        backbone = model.backbone
        if config['usedMethod'] == 'VicRegL':
            backbone = nn.Sequential(model.backbone,
                                    model.average_pool,
                                    nn.Flatten(start_dim = 1))
    # freeze backbone
    finetuningModel = Base(backbone, ReadoutHead(512, config['finetuning']['output_size']), config)
    return finetuningModel

def loadFinetuningModel(config)-> nn.Module:
    backbone = loadModel(config).backbone
    model = Base.load_from_checkpoint(config["finetuning"]["modelpath"], backbone= backbone,ReadoutHead = ReadoutHead(512, config['finetuning']['output_size']),config=config)
    model.eval()
    return model