import yaml
import os
import torch
import torch.nn as nn

from models.barlowtwins import BarlowTwinsLit
from utils.data import get_data_pretraining, load_config
from utils.modelFactory import createModel
#from utils.custommultiviewcollatefunction import CustomMultiViewCollateFunction
import torchvision
from mimeta import  MIMeta


from lightly.loss import BarlowTwinsLoss

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.data import LightlyDataset


def main():

    
    # Load config file
    config = load_config('config.yaml')

    # create model
    model = createModel(config)

    # get data for pretraining
    train, val, test = get_data_pretraining(config)
    collate_fn = MultiViewCollate()
    dataloader = torch.utils.data.DataLoader(
            train,
            batch_size=config['batch_size'],
            collate_fn = collate_fn, 
            num_workers=4, 
            shuffle = True)

    # Create a ModelCheckpoint callback
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath='path/to/save/directory',
    #    filename='model_{epoch}-{val_loss:.2f}',  # Customize the filename pattern
    #    save_top_k=5,  # Set the number of models to save
    #    mode='min',  # 'min' or 'max' depending on the metric being tracked
    #    monitor='val_loss',  # Metric to monitor for saving models
    #)

    save_path = os.path.join(config['savedmodel']['path'], config['savemodel']['name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path)


    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs = config['epochs'],
        devices='auto',
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        log_every_n_steps=15,
    )
    trainer.fit(model= model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()
