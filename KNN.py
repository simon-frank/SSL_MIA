from utils.modelFactory import createFinetuningModel, loadFinetuningModel
from utils.data import get_data_pretraining, load_config, get_data_finetuning, load_config, calculate_label_counts
from models.barlowtwins import BarlowTwinsLit
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import os
import sys
import annoy

def main():
    # Load config file
    config = load_config('config.yaml')
    
    local_place = os.path.abspath( os.path.join(os.path.dirname(__file__)))

    path_to_model = local_place+"/trained_models" #"graphics/scratch2/datasets/practical_course/MIMeta/trained_models/"

    modelnames = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_model):
        for filename in filenames:
            modelnames[filename] = os.sep.join([dirpath, filename])
    for models in modelnames:
        
        model_path = models.values()
        # Here the models are loaded

        model=ModelCheckpoint(dirpath=model_path)


        train, val, test = get_data_finetuning(config)

        model.eval()
        backbone = model.backbone
        readouthead = model.readouthead

        dataloader_val= torch.utils.data.DataLoader(
                test,
                64,
                shuffle = False)
          
        featureVectors = []
        gt = []
        model_predictions = []
        true_predictions = []
        criterion = torch.nn.CrossEntropyLoss()
        
        device =torch.device('cpu') #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for batch in enumerate(dataloader_val):
                (inputs, targets) = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                featureVectors.append(backbone(inputs).detach().cpu().numpy())
                gt.append(targets.detach().cpu().numpy())

                dim0 = featureVectors[0].shape
                dim1 = featureVectors[-1].shape

                featureVectors  = np.concatenate(featureVectors, axis=0)
                gt = np.concatenate(gt, axis=0)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                model_predictions.append(predicted)
                true_predictions.append(targets)
                #print(featureVectors.shape)
                #print(gt.shape)
        
        #get number of items in largest class, to use as input for nearest neighbors
        #num_of_instants = calculate_label_counts(test)
        knn_data = AnnoyIndex(featureVectors)
        knn_data.build(11)
        knn_data.save(models+".ann")

        distance_array = np.ones(shape=(dim0,dim0))
        for i in range(0,dim0):
            for j in range (0,dim0):
                distance_array[i,j] = knn_data.get_distance(i,j)


        

if __name__ == '__main__':
    main()