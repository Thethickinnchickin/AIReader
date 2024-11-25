import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing
import requests
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model from Keras
model = MobileNetV2(weights='imagenet')

# ImageNet class labels from a JSON file or online API
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels = requests.get(LABELS_URL).json()
labels = {int(key): value[1] for key, value in labels.items()}

def preprocess_image(frame):
    """Preprocess image to the required format for MobileNetV2."""
    # Convert frame to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to the input size expected by MobileNetV2
    img = cv2.resize(img, (224, 224))
    # Convert to array and expand dimensions (batch size of 1)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image for MobileNetV2
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(image_data):
    """Use the model to classify the image and return the predicted label."""
    predictions = model.predict(image_data)  # Get predictions from the model
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode top 3 predictions
    return decoded_predictions

def get_label_from_index(decoded_predictions):
    """Map the predicted index to a human-readable label."""
    top_label = decoded_predictions[0]  # Get the top prediction
    return top_label[1], top_label[2]  # Return label and its confidence score

# Load the video file (replace 'video.mp4' with your file path)
video_path = "N.mp4"  # Update with the correct path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame or end of video reached.")
        break
    
    # Preprocess the captured frame
    image_data = preprocess_image(frame)

    # Predict the label
    decoded_predictions = classify_image(image_data)
    predicted_label, confidence = get_label_from_index(decoded_predictions)

    # Display the prediction label in the frame
    if confidence > 0.6:
        # Show the bird's name in red on the upper-left corner
        cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # Default prediction label in green on the upper-left corner
        cv2.putText(frame, f"Predicted: {predicted_label} ({confidence*100:.2f}%)", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    # Show the frame
    cv2.imshow('Video Feed', frame)

    # Wait for 1 ms before moving to the next frame; exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
