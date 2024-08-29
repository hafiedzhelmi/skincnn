import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os

# Define the class names (11 classes)
class_names = [
    'BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 
    'FU-ringworm', 'H-healthy-feet', 'H-healthy-hands', 'H-healthy-skin', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 
    'VI-shingles'
]

# Define the transformations for the validation dataset
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the validation dataset
data_dir = r"C:\Users\hafiedz\Downloads\archive (1)\skin-disease-datasaet"
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the model
model = models.resnet18(pretrained=False)

# Ensure to match the architecture used during training
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Adjusting to 11 classes

# Load the trained model weights
model.load_state_dict(torch.load('model4.pth', map_location=torch.device('cpu')))
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Collect predictions and labels
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()

# Print classification report (precision, recall, F1-score per class)
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:\n")
print(report)
