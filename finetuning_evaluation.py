from utils.modelFactory import createFinetuningModel, loadFinetuningModel
from utils.data import get_data_pretraining, load_config, get_data_finetuning,\
 load_config, calculate_label_counts, calc_label_counts, confusion_matrix, print_confusion_matrix, plot_confusion_matrix
import torch

from tqdm import tqdm


def main():
    # Load config file
    config = load_config('config.yaml')

    # create model
    model = loadFinetuningModel(config)


    train, val, test = get_data_finetuning(config)


    dataloader_val= torch.utils.data.DataLoader(
            test,
            64,
            shuffle = False,
            num_workers= 8)
    
    print('Class balance:{0}'.format(calc_label_counts(dataloader_val)))
    test_loss = 0.0
    total = 0
    correct = 0
    pred = torch.Tensor([])
    targs = torch.Tensor([])
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for inputs, targets in tqdm(dataloader_val, desc = 'Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Calculate the accuracy
            _, predicted = torch.max(outputs, 1)

            pred = torch.hstack([pred, predicted.detach().cpu()])
            targs = torch.hstack([targs, targets.detach().cpu()])
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(dataloader_val)
    accuracy = 100.0 * correct / total
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}")
    print()
    print_confusion_matrix(confusion_matrix(targs, pred))
    plot_confusion_matrix(confusion_matrix(targs, pred))

if __name__ == '__main__':
    main()