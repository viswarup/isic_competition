import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import h5py
import numpy as np
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the custom dataset class
    def __init__(self, csv_file_name, img_dir, transform=None):
        self.csv_file = csv_file_name
        self.data_frame = pd.read_csv(self.csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1]

        if self.transform and callable(self.transform):
            augmented_image = self.transform(image=np.array(image))['image']
            return augmented_image, torch.tensor(label)
    

# Define the transforms with data augmentation using albumentations
transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Rotate(limit=20),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    A.Resize(100, 100),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# File paths
csv_file_path = 'train-metadata.csv'
img_dir_path = 'train-image/image/'

# Load the datasets
full_dataset = SkinCancerDataset(csv_file_path, img_dir_path, transform=transform)

# Split the combined dataset into training and validation sets
train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=[full_dataset[i][1] for i in range(len(full_dataset))])

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# Calculate class weights for imbalanced data
train_labels = [full_dataset[i][1] for i in train_indices]
class_counts = pd.Series(train_labels).value_counts().sort_index()
class_weights = 1. / class_counts
samples_weights = [class_weights[label] for label in train_labels]

# Create a sampler for the training set
sampler = WeightedRandomSampler(samples_weights, num_samples=len(train_indices), replacement=True)

# Create the DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Define the Vision Transformer model using timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
model.head = nn.Sequential(
    nn.Linear(model.head.in_features, 1),
    nn.Sigmoid()
)

# Set device to MPS or GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = model.to(device)

# Compile the model with torch.compile for performance optimization
model = torch.compile(model)

# Instantiate the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mixed precision training scaler
scaler = GradScaler()

# Define the training loop with mixed precision
def train_model(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# Define the validation loop
def validate_model(model, val_loader, criterion, device):
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
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    return val_loss, accuracy

# Main training procedure
num_epochs = 10
train_losses = []
val_losses = []
accuracies = []

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, accuracy = validate_model(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'skin_cancer_vit_model.pth')

# Load the test data from the HDF5 file
test_file_path = 'path_to_your_test_image_hdf5_file.h5'
test_labels_csv = 'path_to_your_test_labels_csv_file.csv'
test_df = pd.read_csv(test_labels_csv)

class TestSkinCancerDataset(Dataset):
    def __init__(self, hdf5_file, csv_file, transform=None):
        self.hdf5_file = hdf5_file
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            img_name = self.data_frame.iloc[idx, 0]
            image = f[img_name][()]
        image = Image.fromarray(image).convert('RGB')
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return image, label

test_dataset = TestSkinCancerDataset(test_file_path, test_labels_csv, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Evaluate the model on the test set
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted.cpu().tolist())

# Generate the confusion matrix and calculate accuracy
conf_matrix = confusion_matrix(all_labels, all_predictions)
accuracy = accuracy_score(all_labels, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
plt.show()
