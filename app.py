import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps, UnidentifiedImageError

# Define the class names
class_names = [
    'BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 
    'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles'
]

# Descriptions and remedies for each class
descriptions = {
    'BA-cellulitis': "A bacterial infection of the skin and underlying tissues that causes redness, swelling, and pain.",
    'BA-impetigo': "A highly contagious bacterial skin infection, common in children, characterized by red sores.",
    'FU-athlete-foot': "A fungal infection that affects the skin on the feet, causing itching, scaling, and redness.",
    'FU-nail-fungus': "A fungal infection affecting the toenails or fingernails, causing thickening and discoloration.",
    'FU-ringworm': "A fungal infection of the skin or scalp, presenting as a red circular rash with clearer skin in the center.",
    'PA-cutaneous-larva-migrans': "A parasitic skin infection caused by hookworm larvae, leading to a winding, itchy rash.",
    'VI-chickenpox': "A viral infection causing an itchy rash and red spots or blisters all over the body.",
    'VI-shingles': "A reactivation of the chickenpox virus in the body, causing a painful rash typically on one side of the body."
}

remedies = {
    'BA-cellulitis': "Seek medical attention for antibiotics to treat the infection. Keep the affected area clean and elevated.",
    'BA-impetigo': "Use antibiotic ointments or oral antibiotics as prescribed by a doctor. Keep the affected area clean.",
    'FU-athlete-foot': "Apply antifungal creams or powders, and keep your feet clean and dry. Avoid walking barefoot in communal areas.",
    'FU-nail-fungus': "Use antifungal medications, either topical or oral, as prescribed by a healthcare provider. Keep nails trimmed and clean.",
    'FU-ringworm': "Apply antifungal creams and keep the affected area clean and dry. Avoid sharing personal items.",
    'PA-cutaneous-larva-migrans': "Consult a doctor for antiparasitic medication like albendazole or ivermectin. Avoid walking barefoot in areas where hookworms are common.",
    'VI-chickenpox': "Rest and stay hydrated. Use calamine lotion or oatmeal baths to reduce itching. Antiviral medication may be prescribed in some cases.",
    'VI-shingles': "Consult a doctor for antiviral medications. Pain relief may include over-the-counter painkillers or prescription medications."
}

# Load the pre-trained ResNet-18 model and modify it to match your number of classes
model = models.resnet18(pretrained=False)
# Modify the fully connected layer to match the saved model's architecture
model.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.fc.in_features, len(class_names))
)

# Load the state dict
model.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocessing function
def preprocess_image(image):
    try:
        # Convert to RGB if the image is not in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Ensure the image is resized properly while maintaining aspect ratio
        transform = transforms.Compose([
            transforms.Resize(256),  # Resize smallest side to 256 while maintaining aspect ratio
            transforms.CenterCrop(224),  # Crop the center 224x224 region
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])
        
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit UI
st.title("Skin Condition Classification with ResNet-18")
st.markdown("""
Welcome to the skin condition classification app. 
Upload an image of a skin condition, and the model will predict the condition based on the trained ResNet-18 architecture.
""")

uploaded_file = st.file_uploader("Upload an image of the skin condition", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        input_tensor = preprocess_image(image)
        
        if input_tensor is not None:
            # Model inference
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, dim=0)
            
            predicted_disease = class_names[predicted_class.item()]
            st.subheader(f"It looks like you have **{predicted_disease}**")
            st.subheader(f"Confidence: ")
            st.write(f"{confidence.item()* 100:.2f}%")
            st.subheader(f"What is it? ")
            st.write(f"{descriptions[predicted_disease]}")
            st.subheader(f" What should you do?")
            st.write(f"{remedies[predicted_disease]}")
            
            # Display all class probabilities
            st.subheader("Class Probabilities:")
            for idx, prob in enumerate(probabilities):
                st.write(f"{class_names[idx]}: {prob.item() * 100:.2f}%")
        else:
            st.error("The image could not be processed. Please try uploading a different image.")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a .png, .jpg, or .jpeg file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.sidebar.title("Model Information")
st.sidebar.markdown(f"""
- **Model Architecture**: ResNet-18
- **Number of Classes**: {len(class_names)}
- **Classes and Descriptions**:
""")

# Display descriptions in the sidebar
for class_name in class_names:
    st.sidebar.markdown(f"**{class_name}:** {descriptions[class_name]}")
