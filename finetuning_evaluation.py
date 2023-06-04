from utils.modelFactory import createFinetuningModel
from utils.data import get_data_pretraining, load_config, get_data_finetuning, load_config, loadFinetuningModel
import torch


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
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(dataloader_val)
    accuracy = 100.0 * correct / total

    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}")
if __name__ == '__main__':
    main()