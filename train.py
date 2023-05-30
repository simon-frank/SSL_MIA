import torch
import torch.nn as nn

from models.barlowtwins import BarlowTwinsLit
from utils.data import get_data
import torchvision
from mimeta import  MIMeta


import pytorch_lightning as pl
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.data import LightlyDataset


def main():

    config =  {'data': '/graphics/scratch2/datasets/practical_course/MIMeta/data',
              'seed': 12345,
               'lr': 0.03,
               'optimizer': torch.optim.SGD,
               'epochs': 100,
               'batch_size:': 192,
               'input_size': 512,
               'hidden_size': 2048,
               'output_size': 2048,
               'img_size': 224,
               'splits': [0.8,0.1,0.1],
               'domain': 'Peripheral Blood Cells',
               'task': 'cell class',
               'transform': SimCLRTransform
               }

    #define backbone
    backbone = torchvision.models.resnet18(zero_init_residual=True)
    backbone.fc = nn.Identity()

    # create model
    model = BarlowTwinsLit(backbone, config)

    # load data
    config['transform'] = config['transform'](input_size=config['img_size'])

    train, val, test = get_data(config)
    collate_fn = MultiViewCollate()

    dataloader = torch.utils.data.DataLoader(
            train,
            batch_size=config['batch_size:'],
            collate_fn = collate_fn, 
            num_workers=4, 
            shuffle = True)
    
    
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = pl.Trainer(
        max_epochs = config['epochs'],
        devices='auto',
        accelerator=accelerator 
    )

    trainer.fit(model= model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()
