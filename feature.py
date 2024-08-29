import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2  # Assuming you're using OpenCV to load images

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Define the transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming 224x224 as the desired size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Assuming the image is read using OpenCV
img = cv2.imread(r'C:\Users\hafiedz\Downloads\archive (1)\skin-disease-datasaet\train\BA-impetigo\91_BA-impetigo (16).jpg')

# Convert the OpenCV image (NumPy array) to a PIL image
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Apply the transformations
image = preprocess(image)

# Unsqueeze to add a batch dimension
image = image.unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Define the specific layers to visualize based on your summary
conv_layers = [
    model.conv1,             # First convolutional layer
    model.layer1[0].conv1,   # Conv layer from first residual block
    model.layer1[0].conv2,   # Conv layer from first residual block
    model.layer2[0].conv1,   # Conv layer from second residual block
    model.layer3[0].conv1,   # Conv layer from third residual block
    model.layer4[0].conv1    # Conv layer from fourth residual block
]

# Function to pass the image through the layers and collect the outputs
def get_resnet_outputs(model, image, conv_layers):
    outputs = []
    x = image
    for layer in conv_layers:
        x = layer(x)
        outputs.append(x)
    return outputs

# Move the image and model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
image = image.to(device)

# Get the outputs from the defined layers
outputs = get_resnet_outputs(model, image, conv_layers)

# Visualize the feature maps
for num_layer, output in enumerate(outputs):
    plt.figure(figsize=(30, 30))
    # For ResNet, outputs can have multiple sub-layers, so we need to handle them
    if len(output.shape) == 4:  # [batch_size, num_filters, height, width]
        layer_viz = output[0, :, :, :]  # Take the first image in the batch
        print(f"Layer {num_layer} output shape: {layer_viz.shape}")

        for i in range(min(layer_viz.shape[0], 64)):  # Show up to 64 filters
            plt.subplot(8, 8, i + 1)
            plt.imshow(layer_viz[i].cpu().detach().numpy(), cmap='gray')
            plt.axis("off")

        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./layer_{num_layer}_feature_maps.png")
        plt.close()
