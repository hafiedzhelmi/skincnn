import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define data transformations for training and validation (to be used in DataLoader)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define a simple transform for displaying the unprocessed images
unprocessed_transform = transforms.Compose([
    transforms.ToTensor()
])

# Define the path to the dataset
data_dir = r"C:\Users\hafiedz\Downloads\archive (1)\skin-disease-datasaet"

# Create datasets
image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
    for x in ['train', 'val']
}

# Load unprocessed images for visualization
unprocessed_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=unprocessed_transform)
    for x in ['train', 'val']
}

# Create dataloaders
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
    for x in ['train', 'val']
}

# Get class names
class_names = image_datasets['train'].classes

# Function to display a grid of images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # Un-normalize if necessary
    inp = np.clip(inp, 0, 1)  # Clip to [0, 1] for display
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=8)
    plt.axis('off')

# 1. Visualize one sample image per class (unprocessed)
def visualize_samples_per_class(unprocessed_datasets):
    num_classes = len(class_names)
    
    fig, axs = plt.subplots(1, num_classes, figsize=(15, 5))
    
    for idx, class_name in enumerate(class_names):
        # Find one image of the class
        class_idx = unprocessed_datasets['train'].class_to_idx[class_name]
        sample_idx = [i for i in range(len(unprocessed_datasets['train'])) if unprocessed_datasets['train'].imgs[i][1] == class_idx][0]
        image, label = unprocessed_datasets['train'][sample_idx]

        axs[idx].imshow(image.permute(1, 2, 0))
        axs[idx].set_title(class_name, fontsize=8)
        axs[idx].axis('off')
    
    plt.show()

# 2. Visualize a batch from the shuffled train DataLoader (processed)
def visualize_train_dataloader(dataloader):
    inputs, classes = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title="Training Data Batch")

# 3. Visualize a batch from the validation DataLoader (processed)
def visualize_val_dataloader(dataloader):
    inputs, classes = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title="Validation Data Batch")

# Create a figure with subplots for all visualizations
#plt.figure(figsize=(15, 15))

# 1. Visualize one sample image per class (unprocessed)
plt.subplot(3, 1, 1)
visualize_samples_per_class(unprocessed_datasets)

# 2. Visualize a batch from the shuffled train DataLoader (processed)
plt.subplot(3, 1, 2)
visualize_train_dataloader(dataloaders['train'])

# 3. Visualize a batch from the validation DataLoader (processed)
plt.subplot(3, 1, 3)
visualize_val_dataloader(dataloaders['val'])

plt.tight_layout()
plt.show()
