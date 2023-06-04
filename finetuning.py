from utils.modelFactory import createFinetuningModel
from utils.data import get_data_pretraining, load_config, get_data_finetuning
from utils.data import load_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os 

def main():

    # Load config file
    config = load_config('config.yaml')

    # create model
    model = createFinetuningModel(config)


    train, val, test = get_data_finetuning(config)

    save_path = os.path.join(config['savedmodel']['path'], config['finetuning']['name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path
        save_top_k=5,  # Set the number of models to save
    mode='min',  # 'min' or 'max' depending on the metric being tracked
    monitor='val_loss',)


    dataloader_training = torch.utils.data.DataLoader(
            train,
            64,
            shuffle = True)
    dataloader_val= torch.utils.data.DataLoader(
            val,
            64,
            shuffle = True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs = config['epochs'],
        devices='auto',
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        log_every_n_steps=15,
    )
    trainer.fit(model= model, train_dataloaders=dataloader_training, val_dataloaders=dataloader_val)

if __name__ == '__main__':
    main()  