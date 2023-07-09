from utils.modelFactory import createModel, loadModel
import torch
from utils.data import get_data_pretraining, load_config, get_data_finetuning
from models.barlowtwins import BarlowTwinsLit
import numpy as np
def main():
    
    # Load config file
    config = load_config('config.yaml')

    # create model
    model = loadFinetuningModel(config)

    #print(model)

    path = config["evaluation"]["modelpath"]
    #model = BarlowTwinsLit.load_from_checkpoint(config["evaluation"]["modelpath"])

    model.eval()

    train, val, test = get_data_finetuning(config)

    backbone = model.backbone
    print(len(train))
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
            train,
            64,
            shuffle = False)

    # get the feature vectors

    featureVectors = []
    gt = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch in dataloader:
        (x, y) = batch
        x = x.to(device)
        y = y.to(device)
        featureVectors.append(backbone(x).detach().cpu().numpy())
        gt.append(y.detach().cpu().numpy())
    print(featureVectors[0].shape)
    print(featureVectors[-1].shape)
    featureVectors  = np.concatenate(featureVectors, axis=0)
    gt = np.concatenate(gt, axis=0)
    print(featureVectors.shape)
    print(gt.shape)


    #using test set
    # do something with the model
    # create the feature vectors with the model, cluster them and calculate a metric
    pass


if __name__ == '__main__':
    main()