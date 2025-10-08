# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 09:58:38 2025

@author: Wisdom
"""

# Importing required libraries
import os
import cv2
import visualkeras
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD

# Set the path for storing results
path = "path_to_result"
num_epochs = 300
batch_size = 64


# Function to load images from folder
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
                        labels.append(subfolder)
    return images, labels

# Define paths to your training and test folders
training_folder = "path_to_train_folder"
test_folder = "path_to_test_folder"


# Load training and test images and labels
train_images, train_labels = load_images_from_folder(training_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# Print the number of training and test images
print("Number of train images:", len(train_images))
print("Number of test images:", len(test_images))

# Create directories to store results, graphs, and model
result = path + "/result"
if not os.path.exists(result):
    os.mkdir(result)
graph = path + "/graph"
if not os.path.exists(graph):
    os.mkdir(graph)
model_dir = path + "/model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Convert lists to NumPy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)
print("Original train_images shape:", train_images.shape)
print("Original test_images shape:", test_images.shape)

# Normalize the datasets
train_images = train_images / 255.0
test_images = test_images / 255.0

# Label encoding for integer labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Load the pre-trained VGG-19 model without the top classification layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(240, 240, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(70, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)  

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a custom learning rate
custom_learning_rate = 0.001
sgd = SGD(lr=custom_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a GridSearch for tuning the learning rate and momentum
param_grid = {
    'batch_size': [64,100, 128, 132],
    'epochs': [10, 100, 300, 500],
    'optimizer__learning_rate': [0.01, 0.001, 0.0001, 0.00001],
    'optimizer__momentum': [0.8, 0.9]
}

# Use StratifiedKFold for 10-fold cross-validation on training data
skf = StratifiedKFold(n_splits=10)

# Initialize arrays for storing scores
train_accuracy_scores = []
val_accuracy_scores = []

# Loop over each fold for cross-validation
for train_index, val_index in skf.split(train_images, train_labels):
    x_train_fold, x_val_fold = train_images[train_index], train_images[val_index]
    y_train_fold, y_val_fold = train_labels[train_index], train_labels[val_index]
    
    # Fit the model on the current fold
    history = model.fit(x_train_fold, y_train_fold, epochs=num_epochs, verbose=1, validation_data=(x_val_fold, y_val_fold))
    
    # Evaluate on the validation fold
    val_loss, val_accuracy = model.evaluate(x_val_fold, y_val_fold)
    train_accuracy_scores.append(history.history['accuracy'][-1])
    val_accuracy_scores.append(val_accuracy)

    print(f'Fold validation accuracy: {val_accuracy * 100:.2f}%')

# Save the final model after cross-validation
model.save(os.path.join(model_dir, 'cnn_model.h5'))

# Test the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy * 100))

# Predict labels for the test set
test_predictions = model.predict(test_images)
test_pred_labels = np.argmax(test_predictions, axis=1)

# Calculate precision, recall, and F1 score on the test set
precision = precision_score(test_labels, test_pred_labels, average='weighted')
recall = recall_score(test_labels, test_pred_labels, average='weighted')
f1 = f1_score(test_labels, test_pred_labels, average='weighted')
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Compute ROC curve and ROC AUC for each class
test_labels_bin = label_binarize(test_labels, classes=np.arange(3))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  # Assuming you have 3 classes
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
plt.legend(loc="lower right")
plt.savefig(graph + '/roc-curve-multiclass.png')
plt.show()

# Classification report
class_report = classification_report(test_labels, test_pred_labels, target_names=label_encoder.classes_)
print("Classification report:\n", class_report)
class_report_file = os.path.join(result, 'classification_report.txt')
with open(class_report_file, 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)

# Save training history
hist_df = pd.DataFrame(history.history)
hist_csv_file = result + '/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Plot training and validation accuracy
plt.plot(range(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Epoch')
plt.legend()
plt.savefig(graph + '/accuracy-plot.png')
plt.show()

# Plot training and validation loss
plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training loss')
plt.plot(range(1, num_epochs + 1), history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch')
plt.legend()
plt.savefig(graph + '/loss-plot.png')
plt.show()

# Generate predicted labels
predicted_labels = model.predict(test_images)
predicted_classes = np.argmax(predicted_labels, axis=1)

# Actual labels
actual_labels = test_labels

# Create confusion matrix
cm = confusion_matrix(actual_labels, predicted_classes)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(graph + '/confusion-matrix.png')

plt.show()
