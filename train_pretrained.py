## training
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from models.base import Base18
from nn.readoutHead import ReadoutHead

def main():

    # Define the transforms for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set a random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Create an instance of the CIFAR10 dataset
    dataset = CIFAR10(root='.', train=True, transform=transform, download=True)


    # Define the ratio of training and test data
    train_ratio = 0.05  # 80% for training, 20% for testing
    test_ratio = 0.01

    # Calculate the sizes of training and test subsets
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    not_used = len(dataset) - train_size - test_size


    # Randomly split the dataset into training and test subsets
    train_dataset, test_dataset,_  = random_split(dataset, [train_size, test_size, not_used])

    # Create a data loader for batching and shuffling the dataset
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of your model and loss function
    model = Base18(ReadoutHead(3,1000,10))
    loss_fn = nn.CrossEntropyLoss()

    # Set up the optimizer
    learning_rate = 0.001
    optimizer = optim.Adam(model.readoutHead.parameters(), lr=learning_rate)

    # Set the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0

        # Iterate over the dataset
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

        eval_running_loss = 0.0
        eval_correct_predictions = 0
        # Iterate over the dataset
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)


            # Update statistics
            eval_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            eval_correct_predictions += (predicted == labels).sum().item()

        # Print epoch statistics
        epoch_loss = running_loss / len(train_dataloader)
        accuracy = correct_predictions / len(train_dataset) * 100
        print(f'Epoch [{epoch+1}/{num_epochs}] Train-Loss: {epoch_loss:.4f} Train-Accuracy: {accuracy:.2f}%')
        # Print epoch statistics
        epoch_loss = eval_running_loss / len(test_dataloader)
        accuracy = eval_correct_predictions / len(test_dataset) * 100
        print(f'Epoch [{epoch+1}/{num_epochs}] Test-Loss: {epoch_loss:.4f} Test-Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
	main()