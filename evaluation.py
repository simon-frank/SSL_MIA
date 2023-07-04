from utils.modelFactory import createModel, loadModel,  loadFinetuningModel
import torch
from utils.data import get_data_pretraining, load_config, get_data_finetuning
from models.barlowtwins import BarlowTwinsLit
import pytorch_lightning as pl
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class KNNModule(pl.LightningModule):
    def __init__(self, vectors, k):
        super(KNNModule, self).__init__()
        self.vectors = vectors
        self.k = k

    def forward(self, x):
        return self.knn(x)

    def knn(self, x):
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(self.vectors)
        distances, indices = nbrs.kneighbors(x)
        return distances, indices
        
organ_slices_dict = {
    0: "heart",
    1: "left lung",
    2: "right lung",
    3: "liver",
    4: "spleen",
    5: "pancreas",
    6: "left kidney",
    7: "right kidney",
    8: "bladder",
    9: "left femoral head",
    10: "right femoral head"
}

 # adipose (ADI); background (BACK); debris (DEB); lymphocytes (LYM); mucus (MUC); smooth muscle (MUS); normal colon mucosa (NORM); cancer-associated stroma (STR); colorectal adenocarcinoma epithelium (TUM)

colorectal_cancer_dictionary = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "NORM", # normal colon mucosa 
    7: "STR", # cancer-associated stroma (STR)
    8: "TUM" # colorectal adenocarcinoma epithelium (TUM)
}

def main():
    
    # Load config file
    config = load_config('config.yaml')
    if config['evaluation']['domain'] in ['organs_axial','organs_coronal','organs_sagittal']:
        print("organ slices evaluation")
        dict = organ_slices_dict
        no_classes = 11
    if config['evaluation']['domain']== 'Colorectal Cancer':
        print("crc evaluation")
        dict = colorectal_cancer_dictionary
        no_classes = 9

    # create model
    model = loadModel(config) #loadFinetuningModel(config)

    print(model)

    path = config["evaluation"]["modelpath"]
    #model = BarlowTwinsLit.load_from_checkpoint(config["evaluation"]["modelpath"])

    model.eval()

    train, val, test = get_data_finetuning(config)

    backbone = model.backbone
    print(len(test))

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
            test,
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

    #print(featureVectors[0].shape)
    #print(featureVectors[-1].shape)
    featureVectors  = np.concatenate(featureVectors, axis=0)
    gt = np.concatenate(gt, axis=0)
    #print(featureVectors.shape)
    #print(gt.shape)

    print("before TSNE")
    tsne = TSNE(n_components=2, random_state=42)
    
    print("mid TSNE")
    featureVec_tsne = tsne.fit_transform(featureVectors)
    #print(f"featureVec_tsne {featureVec_tsne.shape}")
    
    print("after TSNE")
    #nbrs = NearestNeighbors(n_neighbors=no_classes, algorithm='auto').fit(featureVectors)
    #query_vector = np.arange(featureVectors.shape[1])  
    #distances, indices = nbrs.kneighbors([query_vector])

    #print(distances.shape)
    #print(indices.shape)
    #X = np.array(featureVectors)

    # Get the nearest neighbor vectors
    #nearest_vectors = X[indices.flatten()]

    # Create a scatter plot
    fig, ax = plt.subplots()
    #ax.scatter(featureVec_tsne[:, 0], featureVec_tsne[:, 1], label='image features', color='blue')
    #ax.scatter(nearest_vectors[:, 0], nearest_vectors[:, 1], label='Nearest Neighbors', color='red')
    #ax.scatter(query_vector[0], query_vector[1], label='Query Vector', color='green', marker='x')
    
    # Add labels and legend
    plt.title("TSNE Projection of extracted feature vectors\n CRC data with Barlow Twins")
    ax.set_xlabel('TSNE dim 1')
    ax.set_ylabel('TSNE dim 2')
    
    unique_labels = np.unique(gt)
    for label in unique_labels:
        plt.scatter(
            featureVec_tsne[gt == label, 0],
            featureVec_tsne[gt == label, 1],
            label=dict[label],#f'Class {label}',
            edgecolors='black',
            linewidths=1,
            alpha=0.7,
            marker='o',
        )
    ax.legend(fontsize='small',loc='upper right')
    plt.savefig('new legend.png')
    


if __name__ == '__main__':
    main()