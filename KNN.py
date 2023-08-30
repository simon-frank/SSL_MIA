mport warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from utils.modelFactory import createModel,createFinetuningModel, loadFinetuningModel
    from utils.data import get_data_pretraining, load_config, get_data_finetuning, load_config, calculate_label_counts
    from models.barlowtwins import BarlowTwinsLit
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    import torch
    import os
    import re
    from annoy import AnnoyIndex
    import numpy as np
    import pandas as pd
    
def main():
    # Load config file
    config = load_config('config.yaml')
    
    local_place = os.path.abspath( os.path.join(os.path.dirname(__file__)))
    path_to_model = "/home/franksim/SSL_MIA/trained_models/barlowpretrained_finetuningall_0_1" 
    #path_to_model = local_place+"/trained_models" 

    #iterate through given folder and subfolders and store full filename (includeing path)
    #for all files of the type ckp
    modelnames = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_model):
        for filename in filenames:
            if filename.endswith('.ckpt'):
                modelnames[filename] = os.sep.join([dirpath, filename])

    cuda_yes = torch.cuda.is_available()

    for key,value in modelnames.items():

        Finetuned = False
        # Here the models are loaded
        try:
        #Loading of not-finetuned models    
            model = createModel(config)
            checkpoint = torch.load(value)
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
        #Loading of finetuned Model
            model = createFinetuningModel(config)
            checkpoint = torch.load(value)
            model.load_state_dict(checkpoint['state_dict'])
            Finetuned = True

        if cuda_yes:
            model.cuda()
        train, val, test = get_data_finetuning(config)
        model.eval()

        f=len(test)

        backbone = model.backbone
        #readouthead = model.readouthead

        dataloader= torch.utils.data.DataLoader(
                test,
                1,
                shuffle = False)
          
        featureVectors = []
        gt = []
        model_predictions = []
        true_predictions = []
        criterion = torch.nn.CrossEntropyLoss()
        device =torch.device("cuda" if cuda_yes else "cpu")#torch.device('cpu') 

        with torch.no_grad():
            for batch in dataloader:
                (inputs, targets) = batch
                
                inputs,targets = inputs.to(device), targets.to(device)

                featureVectors.append(backbone(inputs).detach().cpu().numpy())
                gt.append(targets.detach().cpu().numpy())

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                model_predictions.append(int(predicted))
                true_predictions.append(int(targets))

        featureVectors  = np.concatenate(featureVectors, axis=0)
        dim0 = featureVectors[0].shape
        gt = np.concatenate(gt, axis=0)

        
        #get number of items in largest class, to use as input for nearest neighbors
        num_of_instants = calculate_label_counts(test)


        #print(f"size of featureVectors{featureVectors.shape}\n class of fV: {type(featureVectors)}")
        knn_data = AnnoyIndex(f, metric='euclidean')
        for i in range(f):
            v = featureVectors[:,i]
            knn_data.add_item(i,v)

        #Given the 11 classes of the dataset
        knn_data.build(11)
        no_ending_name = re.sub(".ckpt","",key)
        knn_data.save(no_ending_name+".ann")
        print(key)
        distance_array = np.ones(shape=(len(test),len(test)))
        for i in range(0,f):
            for j in range (0,f):
                distance_array[i,j] = knn_data.get_distance(i,j)
        pd_distance_array = pd.DataFrame(distance_array, columns= model_predictions, index= true_predictions)
        pd_distance_array.to_csv(no_ending_name+".csv")

        

if __name__ == '__main__':
    main()