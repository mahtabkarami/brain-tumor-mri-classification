import matplotlib.pyplot as plt

# IMPORTANT: Please ensure that cell 94f2f8d6 (Model Training) was successfully
# executed in the current session immediately before running this cell. This
# will define the 'history' object required for plotting.

# Get the training history data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Training and Validation Loss
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("Training and validation metrics plotted successfully.")


import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Path to the saved model file
model_path = 'best_model.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Trained model file '{model_path}' not found. Please ensure the model training cell (94f2f8d6) was successfully executed and the model was saved.")
else:
    # 1. Load the best trained model
    best_model = load_model(model_path)

    # 2. Evaluate the loaded model on the test_generator
    # Ensure test_generator is defined from previous steps
    if 'test_generator' not in locals() and 'test_generator' not in globals():
        print("Error: 'test_generator' not found. Please ensure the data preprocessing cell has been executed.")
    else:
        test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)

        # 3. Print the test loss and test accuracy
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf # Required for to_categorical
import os
from tensorflow.keras.models import load_model # Required for load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Required for ImageDataGenerator

# --- Re-instantiate necessary variables from previous cells for robustness ---
# Define target image dimensions and batch size (from cell cb2ee6dd)
img_width, img_height = 224, 224
batch_size = 32

# Dynamic determination of dataset_path (from cell 55530f57 or 0563068b)
base_dataset_path = 'dataset'

# --- Add a check for the existence of base_dataset_path ---
if not os.path.exists(base_dataset_path):
    print(f"Error: Dataset directory '{base_dataset_path}' not found.")
    print("Please ensure the dataset has been downloaded and extracted by running the relevant cells.")
    # Set flags to prevent further operations that depend on the dataset
    dataset_found = False
    test_generator = None
    best_model = None
else:
    dataset_found = True
    subdirs = [d for d in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, d))]
    if len(subdirs) == 1 and subdirs[0] == 'brain-tumor-mri-dataset': # Check for specific subdir name
        dataset_path = os.path.join(base_dataset_path, subdirs[0])
    else:
        dataset_path = base_dataset_path # Assume 'Training'/'Testing' are directly under 'dataset'

    test_dir = os.path.join(dataset_path, 'Testing')

    # Re-create test_datagen and test_generator (from cell cb2ee6dd)
    test_datagen = ImageDataGenerator(
        rescale=1./255                  # Normalize pixel values to [0, 1]
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Ensure the best_model is loaded
    model_path = 'best_model.h5'
    best_model = None # Initialize to None
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Cannot perform ROC/AUC calculation.")

# ---------------------------------------------------------------------------

if dataset_found and test_generator and best_model:
    # 1. Obtain true class labels and predicted probabilities for the entire test set
    test_generator.reset()
    steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    y_pred_proba = best_model.predict(test_generator, steps=steps, verbose=1)
    y_true_roc = test_generator.classes

    # 2. Convert y_true_roc into a one-hot encoded format
    n_classes = len(test_generator.class_indices)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true_roc, num_classes=n_classes)

    # Get class names in the order of their indices
    class_names = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda item: item[1])]

    # 3. Calculate FPR, TPR, and AUC for each class (one-vs-rest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

    # Plot the random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (area = 0.50)')

    # Add plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print("ROC/AUC curves calculated and plotted successfully.")
else:
    print("Skipping ROC/AUC calculation due to missing dataset, model or test generator setup.")