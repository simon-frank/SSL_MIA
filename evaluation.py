from utils.model import createModel
import torch
from utils.data import get_data_pretraining, load_config, get_data_finetuning

def main():
    
    # Load config file
    config = load_config('config.yaml')

    # create model
    model = createModel(config)

    model = model.load_from_checkpoint(config["evaluation"]["modelpath"])

    model.eval()

    train, val, test = get_data_finetuning(config)

    backbone = model.backbone

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
            train,
            64,
            num_workers=4,
            shuffle = False)

    # get the feature vectors

    featureVectors = []
    y = []
    for batch in dataloader:
        x, y = batch
        featureVectors.append(backbone(x))
        y.append(y)
    featureVectors = torch.cat(featureVectors, dim=0)
    y = torch.cat(y, dim=0)
    print(featureVectors.shape)
    print(y.shape)


    #using test set
    # do something with the model
    # create the feature vectors with the model, cluster them and calculate a metric
    pass


if __name__ == '__main__':
    main()