from utils.modelFactory import createFinetuningModel, loadFinetuningModel
from utils.data import get_data_pretraining, load_config, get_data_finetuning, load_config, calculate_label_counts, save_performance
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # Load config file
    config = load_config('config.yaml')



    # create model
    model = loadFinetuningModel(config)


    train, val, test = get_data_finetuning(config)


    dataloader_val= torch.utils.data.DataLoader(
            test,
            64,
            shuffle = False)
    # create confusion matrix
    num_classes = config['finetuning']['output_size']
    confusion_matrix = np.zeros((num_classes, num_classes))


    print('Class balance:{0}'.format(calculate_label_counts(test)))
    test_loss = 0.0
    total = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch_idx, (inputs, targets) in enumerate(dataloader_val):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Calculate the accuracy
            _, predicted = torch.max(outputs, 1)
            for true_label, predicted_label in zip(targets, predicted):
                confusion_matrix[true_label][predicted_label] += 1
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(dataloader_val)
    accuracy = 100.0 * correct / total

    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}")
    
    normalized_confusion_matrix = confusion_matrix/ confusion_matrix.sum(axis=1)[:, np.newaxis]  
    # Plot the normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(1,num_classes+1))
    plt.yticks(tick_marks, range(1,num_classes+1))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()

    save_directory = os.path.dirname(config['finetuning']['modelpath'])
    save_performance(test_loss, accuracy, save_directory)
    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_directory,'confusion_matrix.png'))
if __name__ == '__main__':
    main()