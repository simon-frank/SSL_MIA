import torch
import torch.nn as nn

from models.barlowtwins import BarlowTwinsLit
import torchvision

import pytorch_lightning as pl
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.data import LightlyDataset


def main():

    config =  {'data': 'data/',
               'lr': 0.06,
               'optimizer': torch.optim.SGD,
               'epochs': 10,
               'batch_size:': 8,
               'input_size': 512,
               'hidden_size': 2048,
               'output_size': 2048,
               'img_size': 224
               }

    #define backbone
    backbone = torchvision.models.resnet18(zero_init_residual=True)
    backbone.fc = nn.Identity()

    # create model
    model = BarlowTwinsLit(backbone, config)

    MIMA = torchvision.datasets.ImageFolder(config['data'])

    collate_fn = MultiViewCollate()
    transform = SimCLRTransform(input_size=config['img_size'])

    dataset = LightlyDataset.from_torch_dataset(MIMA, transform= transform)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size:'],
            collate_fn = collate_fn, 
            num_workers=4, 
            shuffle = True)
    
    
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = pl.Trainer(
        max_epochs = 10,
        devices='auto',
        accelerator=accelerator 
    )

    trainer.fit(model= model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()