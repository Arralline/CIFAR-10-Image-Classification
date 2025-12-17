import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the path to the saved model
MODEL_PATH = './cifar10_transfer_learning_model.pth'

# Load the pre-trained ResNet18 model structure and modify its final layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Load the trained weights
# Ensure the model file exists, if not, it should have been saved in a previous step
if not os.path.exists(MODEL_PATH):
    # This part should ideally be handled before deploying, but as a fallback
    # if the model object is still in memory, it can be saved.
    # In a typical Hugging Face Space, the model file would be pushed separately.
    print(f"Warning: Model file {MODEL_PATH} not found. Ensure it is available in the Space repository.")
    # For the purpose of generating the app.py, we assume the model file will be present

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode

# Define the transformations (must match training/testing transformations)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image_gradio(img: Image.Image):
    # Preprocess the image
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Move the input to the same device as the model
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Create a dictionary of class names and their probabilities
    prediction_results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

    return prediction_results

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_image_gradio,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier (Transfer Learning)",
    description="Upload an image among this list [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck] to classify it into one of the 10 CIFAR-10 categories using a fine-tuned ResNet18 model.",
    examples=[]
)

if __name__ == "__main__":
    demo.launch()
