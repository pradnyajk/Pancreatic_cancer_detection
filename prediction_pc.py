"""
Program to use PC-ResNet50 model for classifying images with preprocessing steps.

Package requirements:
    a) python = 3.9 
    b) numpy
    c) pandas
    d) tensorflow=2.11.0
    e) keras = 2.11.0
    f) scikit-learn
    g) scipy
    h) opencv-python

Input: 
    a) generated_model.h5 file
    b) Folder named 'Images' with images to be classified
Output: Prediction file
"""

import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# Define class labels (update as needed)
class_labels = ['Cancer', 'Normal', 'Pancreatitis']

# Function to preprocess an image
def preprocess_image(image_path, target_size=(240, 240)):
    """
    Preprocesses the image: Cropping, resizing, applying CLAHE, and converting to 3-channel.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create a binary mask for segmentation
    _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)  # Largest contour
        x, y, w, h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]  # Crop to ROI
    
    # Resize to match model input size
    img = cv2.resize(img, target_size)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Adjust brightness and contrast
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)  # Contrast=1.0, Brightness=-50
    
    # Normalize the image
    img = img / 255.0  # Normalize to [0,1]
    
    # Convert grayscale to RGB (duplicate single channel into 3)
    img_rgb = np.stack((img,) * 3, axis=-1)
    
    return img_rgb

# Function to load model and make predictions
def cnn_prediction(image_folder_path, model_path, target_size=(240, 240, 3)):
    """
    Loads the trained CNN model and makes predictions on images in the folder.
    """
    # Load the pre-trained model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Get all image file paths
    image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) 
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    print(f"Found {len(image_paths)} images. Preprocessing...")

    # Preprocess images and prepare for prediction
    images = []
    filenames = []
    for image_path in image_paths:
        img = preprocess_image(image_path, target_size[:2])
        images.append(img)
        filenames.append(os.path.basename(image_path))

    images = np.array(images)  # Convert list to numpy array

    print("Images preprocessed. Running predictions...")

    # Make predictions
    predictions = model.predict(images)
    
    # Process predictions
    results = []
    for filename, probs in zip(filenames, predictions):
        predicted_index = np.argmax(probs)  # Get the index of the highest probability
        predicted_label = class_labels[predicted_index]  # Get the corresponding class label
        
        # Store results (filename, predicted class, probabilities)
        results.append([filename, predicted_label] + probs.tolist())

    return results

# Main execution block
if __name__ == "__main__":
    # Automatically detect paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    image_folder = os.path.join(script_dir, "Images")  # Image folder inside script directory
    model_folder = os.path.join(script_dir, "Model")  # Model folder inside script directory
    
    # Automatically find the model file
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.h5')]
    if not model_files:
        raise FileNotFoundError("No model file found in the 'Model' folder!")
    model_path = os.path.join(model_folder, model_files[0])  # Use the first model found
    
    # Run predictions
    predictions = cnn_prediction(image_folder, model_path)

    # Save results to CSV in the script directory
    output_csv_path = os.path.join(script_dir, "predictions.csv")
    df = pd.DataFrame(predictions, columns=["Filename", "Predicted_Class"] + class_labels)
    df.to_csv(output_csv_path, index=False)

    print(f"\nPredictions saved to {output_csv_path}")

    # Print results
    for row in predictions:
        print(f"Prediction for {row[0]}: {row[1]}, Probabilities: {row[2:]}")
