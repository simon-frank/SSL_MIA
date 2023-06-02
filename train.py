import yaml
import os
import torch
import torch.nn as nn

from models.barlowtwins import BarlowTwinsLit
from utils.data import get_data, load_config
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

    #define backbone
    backbone = torchvision.models.resnet18(zero_init_residual=True)
    backbone.fc = nn.Identity()

    # create model
    model = BarlowTwinsLit(backbone, config)

    # load data
    #config['transform'] = config['transform'](input_size=config['img_size'])

    train, val, test = get_data(config)
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

    save_path = os.path.join(config['model']['path'], config['model']['name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path
        
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = BarlowTwinsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)


    print("Starting Training")
    for epoch in range(10):
        total_loss = 0
        for (x0, x1), _, _ in dataloader:
            print(x0.shape)
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")



    #trainer = pl.Trainer(
    #    max_epochs = config['epochs'],
    #    devices='auto',
    #    accelerator=accelerator,
    #    callbacks=[checkpoint_callback],
    #    log_every_n_steps=8,
    #)
#
    #trainer.fit(model= model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()
