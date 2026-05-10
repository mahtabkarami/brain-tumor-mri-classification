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


import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import load_img, img_to_array

# Ensure best_model is loaded
if 'best_model' not in locals() and 'best_model' not in globals():
    model_path = 'best_model.h5'
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Cannot perform prediction.")
        # Exit or handle the error appropriately

# Ensure test_generator is defined and class mappings are available
if 'test_generator' in locals() or 'test_generator' in globals():
    class_indices = test_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    print("Error: 'test_generator' not found. Cannot determine class names.")
    idx_to_class = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'} # Fallback

# Re-collect all test image paths, if not already available or if for a fresh run
if 'all_test_image_paths' not in locals() or not all_test_image_paths:
    all_test_image_paths = []
    if 'test_dir' in locals() or 'test_dir' in globals():
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_test_image_paths.append(os.path.join(class_path, img_name))
    else:
        print("Error: 'test_dir' not found. Cannot select random image.")

if all_test_image_paths and ('best_model' in locals() or 'best_model' in globals()):
    # Randomly select an image
    random_image_path = random.choice(all_test_image_paths)

    # Load the selected image and preprocess it
    img = load_img(random_image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Use the model to predict
    predictions = best_model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = idx_to_class[predicted_class_idx]

    # Extract true class name from the image path
    true_class_name = os.path.basename(os.path.dirname(random_image_path))

    # Display the image and results
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"True: {true_class_name}, Predicted: {predicted_class_name}")
    plt.axis('off')
    plt.show()

    print(f"Random image selected: {random_image_path}")
    print(f"True Label: {true_class_name}")
    print(f"Predicted Label: {predicted_class_name}")
    print(f"Prediction Probabilities: {predictions[0]}")
else:
    print("Cannot proceed with prediction: Model or test images not available.")

# Print the overall test accuracy (assuming it's already calculated and stored)
if 'test_accuracy' in locals() or 'test_accuracy' in globals():
    print(f"\nOverall Test Accuracy: {test_accuracy:.4f}")
else:
    print("\nOverall Test Accuracy not found. Please ensure model evaluation was performed.")


import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import load_img, img_to_array

# Ensure best_model is loaded
# If best_model is not in memory, it needs to be loaded again
if 'best_model' not in locals() and 'best_model' not in globals():
    model_path = 'best_model.h5'
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Cannot perform prediction.")
        # Exit or handle the error appropriately

# 1. Create a dictionary that maps class indices to class names
# The test_generator.class_indices gives class_name -> index
# We need index -> class_name for display purposes
if 'test_generator' in locals() or 'test_generator' in globals():
    class_indices = test_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    print("Error: 'test_generator' not found. Cannot determine class names.")
    # Fallback or error handling
    idx_to_class = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'} # Assuming standard order

# 2. Get all test image paths (Corrected logic)
all_test_image_paths = []
if 'test_dir' in locals() or 'test_dir' in globals():
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_test_image_paths.append(os.path.join(class_path, img_name))
    print(f"Found {len(all_test_image_paths)} images in the test set.")
else:
    print("Error: 'test_dir' not found. Cannot select random image.")

if all_test_image_paths and ('best_model' in locals() or 'best_model' in globals()):
    # 3. Randomly select an image
    random_image_path = random.choice(all_test_image_paths)

    # 4. Load the selected image and preprocess it
    # img_width and img_height are defined in previous cells
    img = load_img(random_image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)

    # 5. Normalize pixel values and expand dimensions
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # 6. Use the model to predict
    predictions = best_model.predict(img_array)

    # 7. Determine predicted class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = idx_to_class[predicted_class_idx]

    # 8. Extract true class name from the image path
    true_class_name = os.path.basename(os.path.dirname(random_image_path))

    # 9. Display the image and results
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"True: {true_class_name}, Predicted: {predicted_class_name}")
    plt.axis('off')
    plt.show()

    print(f"Random image selected: {random_image_path}")
    print(f"True Label: {true_class_name}")
    print(f"Predicted Label: {predicted_class_name}")
    print(f"Prediction Probabilities: {predictions[0]}")
else:
    print("Cannot proceed with prediction: Model or test images not available.")



import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.utils import load_img, img_to_array

# Ensure best_model is loaded
if 'best_model' not in locals() and 'best_model' not in globals():
    model_path = 'best_model.h5'
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Cannot identify incorrect predictions.")
        # Handle error or exit

# Ensure test_generator is defined and class mappings are available
if 'test_generator' in locals() or 'test_generator' in globals():
    class_indices = test_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    print("Error: 'test_generator' not found. Cannot determine class names for incorrect predictions.")
    # Fallback or error handling
    idx_to_class = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'} # Assuming standard order

# Re-collect all test image paths, if not already available or if for a fresh run
if 'all_test_image_paths' not in locals() or not all_test_image_paths:
    all_test_image_paths = []
    if 'test_dir' in locals() or 'test_dir' in globals():
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_test_image_paths.append(os.path.join(class_path, img_name))
    else:
        print("Error: 'test_dir' not found. Cannot gather test image paths.")

incorrect_predictions = []
num_incorrect_to_display = 5 # Number of incorrect predictions to visualize

if all_test_image_paths and ('best_model' in locals() or 'best_model' in globals()):
    # Take a sample of the test images to avoid processing the entire dataset if it's too large
    # and speed up identification of incorrect predictions for visualization.
    sample_size = min(200, len(all_test_image_paths)) # Adjust sample size as needed
    sampled_image_paths = random.sample(all_test_image_paths, sample_size)

    print(f"Checking {len(sampled_image_paths)} sampled test images for incorrect predictions...")

    for img_path in sampled_image_paths:
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = best_model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = idx_to_class[predicted_class_idx]

        true_class_name = os.path.basename(os.path.dirname(img_path))

        if predicted_class_name != true_class_name:
            incorrect_predictions.append({
                'image_path': img_path,
                'true_label': true_class_name,
                'predicted_label': predicted_class_name
            })

        if len(incorrect_predictions) >= num_incorrect_to_display:
            break

    print(f"Found {len(incorrect_predictions)} incorrect predictions (max {num_incorrect_to_display} requested).")

    # Display incorrect predictions
    plt.figure(figsize=(15, 5 * len(incorrect_predictions)))
    for i, incorrect in enumerate(incorrect_predictions):
        plt.subplot(len(incorrect_predictions), 3, i*3 + 1)
        img = load_img(incorrect['image_path'])
        plt.imshow(img)
        plt.title(f"True: {incorrect['true_label']}\nPred: {incorrect['predicted_label']}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

else:
    print("Cannot visualize incorrect predictions: Model or test images not available.")

