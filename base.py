import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from torchsummary import summary

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your data transformations and data loading code remains the same

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
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

data_dir = r"C:\Users\hafiedz\Downloads\archive (1)\skin-disease-datasaet"

image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

model = models.resnet18(pretrained=True)

# Move the model to the device (GPU or CPU)
model = model.to(device)

# Freeze all the parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer to match the number of classes in your dataset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Only the final layer (fc) should be trainable
for param in model.fc.parameters():
    param.requires_grad = True

# Ensure the final layer is on the same device
model.fc = model.fc.to(device)

# Example: Move your input tensor to the device (replace this with your actual input tensor)
# input_tensor = input_tensor.to(device)

# Print the model summary
summary(model, (3, 224, 224))
