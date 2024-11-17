import torch
from train_model import ImprovedCNN  # Adjust the import if your model class is named differently
from torchvision import transforms
from PIL import Image

# Step 1: Load the trained model
model = ImprovedCNN()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Define the class names in the same order as in your dataset
class_names = ["cats", "dogs"]  # Adjust according to your classes

# Step 2: Define the image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Step 3: Load and preprocess the image
def predict(image_path):
    image = Image.open(image_path)  # Load the image
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Disable gradient calculation
        output = model(image)  # Get model predictions
        print(f"Raw output scores: {output}")  # Print output scores

        _, predicted = torch.max(output, 1)  # Get the predicted class index

    return class_names[predicted.item()]  # Map index to class name

# Example usage:
cat_image_path = r"C:\Users\ASUS\OneDrive\Desktop\443.jpg"  # Provide the path to your cat image
predicted_class = predict(cat_image_path)

if predicted_class == "cats":
    print("It's a cat!")
elif predicted_class == "dogs":
    print("It's a dog!")
else:
    print("Unknown class")
