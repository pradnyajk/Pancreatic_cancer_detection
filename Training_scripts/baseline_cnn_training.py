# -*- coding: utf-8 -*-
"""

@author: Wisdom
"""
# Import required libraries

import os
import cv2
import visualkeras
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD


# Setup GPU Memory Growth to prevent memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Path where results will be stored
path = "path_to_result"

# Function to load images from folder and their corresponding labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                if img_path.endswith(('.jpg', '.jpeg', '.png')):
                    image = cv2.imread(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(subfolder)  # Folder name is used as the label
    return images, labels

# Define paths to your training and test folders
training_folder = "path_to_train_folder"
test_folder = "path_to_test_folder"

# Load images and labels from training and test folders
train_images, train_labels = load_images_from_folder(training_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# Print the number of training and test images
print("Number of train images:", len(train_images))
print("Number of test images:", len(test_images))

# Create directories to store results, graphs, and model
result = os.path.join(path, "result")
if not os.path.exists(result):
    os.mkdir(result)
graph = os.path.join(path, "graph")
if not os.path.exists(graph):
    os.mkdir(graph)
model_dir = os.path.join(path, "model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Convert images to numpy arrays for easier processing
train_images = np.array(train_images)
test_images = np.array(test_images)

# Split the training set into training and validation sets (80% training, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Normalize the image pixel values to [0, 1] range for faster convergence
x_train = x_train / 255.0
x_val = x_val / 255.0
test_images = test_images / 255.0

# Encode string labels into integers using LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
test_labels = label_encoder.transform(test_labels)

# One-hot encoding of labels for multi-class classification
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=3)

# Model creation function (for KerasClassifier to use in GridSearchCV)
def create_model(learning_rate=0.001, optimizer='SGD'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  
    ])
    model.compile(optimizer=tf.keras.optimizers.get(optimizer)(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Use KerasClassifier to wrap the Keras model for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Grid Search for hyperparameter tuning
param_grid = {
    'batch_size': [64, 100, 128, 132],
    'epochs': [10, 100, 300, 500],
    'learning_rate': [0.01, 0.001, 0.0001],
    'optimizer': ['SGD', 'Adam']
}

# GridSearchCV with 10-fold cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)

# Print best hyperparameters
print("Best Hyperparameters: ", grid_result.best_params_)

# Evaluate model on validation set
best_model = grid_result.best_estimator_.model
val_loss, val_accuracy = best_model.evaluate(x_val, y_val)
print('Validation loss: {}, Validation accuracy: {}'.format(val_loss, val_accuracy * 100))

# Test the model using the test set and evaluate its performance
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels_one_hot)
print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy * 100))

# Predict labels for the test set
test_predictions = best_model.predict(test_images)
test_pred_labels = np.argmax(test_predictions, axis=1)

# Classification report
class_report = classification_report(test_labels, test_pred_labels, target_names=label_encoder.classes_)
print("Classification Report:\n", class_report)

# ROC Curve and AUC for each class
test_labels_bin = label_binarize(test_labels, classes=np.arange(3))  # Assuming 3 classes
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
plt.legend(loc="lower right")
plt.savefig(graph + '/roc-curve-multiclass.png')
plt.show()

# Save Classification Report to file
class_report_file = os.path.join(result, 'classification_report.txt')
with open(class_report_file, 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)

# Save training history to a CSV file
hist_df = pd.DataFrame(grid_result.best_estimator_.model.history.history)
hist_csv_file = os.path.join(result, 'history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Confusion matrix
predicted_classes = np.argmax(test_predictions, axis=1)
cm = confusion_matrix(test_labels, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(graph + '/confusion-matrix.png')
plt.show()


