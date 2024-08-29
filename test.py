from torchvision import datasets
import os

data_dir = r"C:\Users\hafiedz\Downloads\archive (1)\skin-disease-datasaet"
dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'))  # or 'val'

class_to_idx = dataset.class_to_idx
print(class_to_idx)
