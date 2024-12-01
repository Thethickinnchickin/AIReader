import os
import numpy as np
import tensorflow as tf
import cv2  # OpenCV for image processing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the fine-tuned model
model = load_model('models/bird_classifier_finetuned.h5')

# Load class labels from classes.txt
def load_class_labels(class_labels_file):
    """Load class labels from the provided file."""
    class_names = []
    with open(class_labels_file, 'r') as f:
        for line in f:
            class_names.append(line.strip())  # Add each class name to the list
    return class_names

# Load the class names (species labels)
class_names = load_class_labels('classes.txt')  # Path to your classes.txt file

# Image preprocessing function used during training (same as in fine-tuning)
def preprocess_image(image_path):
    """Preprocess image to the required format for fine-tuned model."""
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    # Load image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Failed to load image '{image_path}'.")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by the model
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image (same as during training)
    return img_array

def classify_image(image_data):
    """Use the model to classify the image and return the predicted label."""
    predictions = model.predict(image_data)  # Get predictions from the model
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the index of the highest prediction
    return predicted_class, predictions[0][predicted_class]  # Return class index and its confidence score

def get_label_from_index(predicted_class, confidence, class_names):
    """Map the predicted index to a human-readable label."""
    # Get the class label from the index
    predicted_label = class_names[predicted_class]  # Using the class_names list
    return predicted_label, confidence  # Return the label and confidence score

# Example usage
image_path = "ST.jpg"  # Your image path
image_data = preprocess_image(image_path)

if image_data is not None:
    predicted_class, confidence = classify_image(image_data)
    predicted_label, confidence = get_label_from_index(predicted_class, confidence, class_names)

    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")
