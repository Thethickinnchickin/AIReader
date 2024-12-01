import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import threading

# Convert .h5 model to .tflite (if you don't already have the tflite model)
def convert_model():
    model = tf.keras.models.load_model('models/bird_classifier_finetuned.h5')

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open('models/bird_classifier_finetuned.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted to .tflite format.")

# Uncomment this if you haven't already converted the model
#convert_model()

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="models/bird_classifier_finetuned.tflite")
interpreter.allocate_tensors()

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
def preprocess_image(frame):
    """Preprocess image to the required format for fine-tuned model."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by the model
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image (same as during training)
    return img_array

def classify_image(image_data):
    """Use the model to classify the image and return the predicted label."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    # Get the output prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the index of the highest prediction
    return predicted_class, predictions[0][predicted_class]  # Return class index and its confidence score

def get_label_from_index(predicted_class, confidence, class_names):
    """Map the predicted index to a human-readable label."""
    predicted_label = class_names[predicted_class]  # Using the class_names list
    return predicted_label, confidence  # Return the label and confidence score

# Load YOLO object detection model (use pre-trained YOLO weights and config)
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Assuming 'bird' is class ID 14 (adjust based on your YOLO class list like coco.names)
bird_class_id = 14  # Update this to match the index for 'bird' in your YOLO class file
confidence_threshold = 0.5  # Confidence threshold for valid detections
nms_threshold = 0.4  # Threshold for Non-Maximum Suppression

# Load the video file (MP4)
video_path = '4.mp4'  # Path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Skip every n-th frame (for example, 3)
skip_frames = 1
frame_count = 0

# Threading function to process each frame and classify birds
def process_frame(frame):
    height, width, channels = frame.shape
    # Perform YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    # Initialize lists for NMS
    boxes = []
    confidences = []

    # Iterate over each detected object
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider detections with a high confidence and class of "bird"
            if confidence > confidence_threshold and class_id == bird_class_id:
                # Get bounding box coordinates
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:  # Ensure indices is not empty
        for i in indices.flatten():  # Correct way to iterate over indices
            x, y, w, h = boxes[i]
            confidence = confidences[i]

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Classify the object inside the bounding box (bird detection)
            cropped_frame = frame[y:y + h, x:x + w]

            if cropped_frame.size > 0:  # Ensure cropped_frame is not empty
                image_data = preprocess_image(cropped_frame)

                if image_data is not None:
                    predicted_class, confidence = classify_image(image_data)
                    predicted_label, confidence = get_label_from_index(predicted_class, confidence, class_names)

                    # Display the current prediction for each bird
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label_text = f"{predicted_label}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x, y - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

# Start processing the video frames in a loop
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no frame is read (end of video)

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Use threading to process the frame
    thread = threading.Thread(target=process_frame, args=(frame,))
    thread.start()
    thread.join()  # Wait for the thread to finish processing

    # Display the video feed with the predicted bounding boxes and labels
    cv2.imshow('Bird Detection with Bounding Boxes', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

