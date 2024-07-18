import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
train_csv = pd.read_csv("train-metadata.csv")

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

            # Check for missing images
        missing_images = [f for f in self.data_frame['isic_id'] if not os.path.exists(os.path.join(img_dir, f + '.jpg'))]
        if missing_images:
            print(f"Warning: {len(missing_images)} images are missing")
            # Optionally, remove missing images from the dataframe
            self.data_frame = self.data_frame[~self.data_frame['isic_id'].isin(missing_images)]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,self.data_frame.iloc[idx]['isic_id'])
        img_name = img_name + '.jpg'

        if not os.path.exists(img_name):
            print(f"Image not found: {img_name}")
        
        image = Image.open(img_name).convert('RGB')
        
        label = self.data_frame.iloc[idx]['target']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Define the transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# File paths
csv_file_path = 'train-metadata.csv'
image_file_path = 'train-image/image/'

# Load the dataset
full_dataset = SkinCancerDataset(csv_file_path, image_file_path, transform=transform)
print(f"Full dataset size: {len(full_dataset)}")

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=full_dataset.data_frame.target)
print(f"Train set size: {len(train_indices)}")
print(f"Validation set size: {len(val_indices)}")

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)



# Calculate class weights for imbalanced data
train_labels = [full_dataset.data_frame.iloc[idx, 1] for idx in train_indices]
class_counts = pd.Series(train_labels).value_counts().sort_index()
class_weights = 1. / class_counts
samples_weights = [class_weights[label] for label in train_labels]


# Create a sampler for the training set
sampler = WeightedRandomSampler(samples_weights, num_samples=len(train_indices), replacement=True)

# Create the DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in val_loader: {len(val_loader)}")


print(f"Length of full dataset: {len(full_dataset)}")
print(f"Length of train dataset: {len(train_dataset)}")
print(f"Length of train indices: {len(train_indices)}")
print(f"Length of sampler: {len(sampler)}")

try:
    for inputs, labels in train_loader:
        print(f" inputs shape {inputs.shape}, labels shape {labels.shape}")
except Exception as e:
    print(f"Error occurred ")
    print(str(e))


# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(100*100*3, 128)  # Adjust based on your image size and channels
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 100*100*3)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Instantiate the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# ... (previous code remains the same)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):

    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):

            print(i)

            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Print batch progress
            if i % 10 == 0:  # Print every 10 batches
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save the model
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    print('Finished Training')

def main():
    # Set device to MPS
    print("starts here")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model, loss function, and optimizer
    model = SimpleNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()