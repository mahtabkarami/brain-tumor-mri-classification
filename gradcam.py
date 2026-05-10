import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Libraries imported successfully.")


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import for test_generator

# --- Re-instantiate necessary variables from previous cells for robustness ---
# Define target image dimensions and batch size (from cell cb2ee6dd)
img_width, img_height = 224, 224
batch_size = 32

# Construct the full path to the testing directory (from cell cb2ee6dd)
# Assuming dataset_path is 'dataset/brain-tumor-mri-dataset' as determined previously
# Need to determine the actual dataset_path if not globally set after previous runs

# Dynamic determination of dataset_path (from cell 55530f57 or 0563068b)
base_dataset_path = 'dataset'
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

# ---------------------------------------------------------------------------

# 2. Ensure the best_model is loaded
model_path = 'best_model.h5'
best_model = None # Initialize to None
if os.path.exists(model_path):
    best_model = load_model(model_path)
    print(f"Loaded best model from {model_path}")
else:
    print(f"Error: '{model_path}' not found. Cannot perform full evaluation.")

# 3. test_generator is now re-instantiated, proceed with evaluation if model is loaded
if test_generator and best_model:
    # Get class names from the generator
    class_names = list(test_generator.class_indices.keys())
    # We need index -> class_name for display purposes
    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}

    # 4. Reset the test_generator to ensure predictions start from the beginning
    test_generator.reset()

    # 5. Obtain the model's predictions for the entire test set
    # Calculate steps based on total samples and batch size and cast to int
    steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    predictions = best_model.predict(test_generator, steps=steps, verbose=1)

    # 6. Convert these predictions (probabilities) into class labels
    y_pred = np.argmax(predictions, axis=1)

    # 7. Extract the true class labels from the test_generator
    y_true = test_generator.classes

    # 8. Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 9. Generate a classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # 10. Visualize the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    print("Confusion Matrix and Classification Report generated successfully.")
else:
    print("Skipping confusion matrix and classification report due to missing model or test generator setup.")













import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 # OpenCV for image manipulation
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# --- Re-instantiate necessary variables from previous cells for robustness ---
# Define target image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Dynamic determination of dataset_path
base_dataset_path = 'dataset'

if not os.path.exists(base_dataset_path):
    print(f"Error: Dataset directory '{base_dataset_path}' not found.")
    print("Please ensure the dataset has been downloaded and extracted by running the relevant cells.")
    dataset_found = False
    test_generator = None
    best_model = None
else:
    dataset_found = True
    subdirs = [d for d in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, d))]
    if len(subdirs) == 1 and subdirs[0] == 'brain-tumor-mri-dataset':
        dataset_path = os.path.join(base_dataset_path, subdirs[0])
    else:
        dataset_path = base_dataset_path

    test_dir = os.path.join(dataset_path, 'Testing')

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    model_path = 'best_model.h5'
    best_model = None
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Cannot perform Grad-CAM visualization.")

# Ensure all_test_image_paths is collected
all_test_image_paths = []
if dataset_found and os.path.exists(test_dir):
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_test_image_paths.append(os.path.join(class_path, img_name))

# Map class indices to names
idx_to_class = {v: k for k, v in test_generator.class_indices.items()} if test_generator else None
# ---------------------------------------------------------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Get the gradients of the name_of_the_output_neuron with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Get the value of the gradients by performing global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    # with respect to the selected class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization, normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), preds[0].numpy()

def display_gradcam(img_path, heatmap, original_img, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + original_img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

print("Grad-CAM utility functions defined.")


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 # OpenCV for image manipulation
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# --- Re-instantiate necessary variables from previous cells for robustness ---
# Define target image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Dynamic determination of dataset_path
base_dataset_path = 'dataset'

if not os.path.exists(base_dataset_path):
    print(f"Error: Dataset directory '{base_dataset_path}' not found.")
    print("Please ensure the dataset has been downloaded and extracted by running the relevant cells.")
    dataset_found = False
    test_generator = None
    best_model = None
else:
    dataset_found = True
    subdirs = [d for d in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, d))]
    if len(subdirs) == 1 and subdirs[0] == 'brain-tumor-mri-dataset':
        dataset_path = os.path.join(base_dataset_path, subdirs[0])
    else:
        dataset_path = base_dataset_path

    test_dir = os.path.join(dataset_path, 'Testing')

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    model_path = 'best_model.h5'
    best_model = None
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        # Explicitly build the model after loading to ensure model.input is set
        # Use the expected input shape for building
        best_model.build(input_shape=(None, img_width, img_height, 3)) # (None for batch size)
        print(f"Loaded best model from {model_path} and built it.")
    else:
        print(f"Error: '{model_path}' not found. Cannot perform Grad-CAM visualization.")

# Ensure all_test_image_paths is collected
all_test_image_paths = []
if dataset_found and os.path.exists(test_dir):
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_test_image_paths.append(os.path.join(class_path, img_name))

# Map class indices to names
idx_to_class = {v: k for k, v in test_generator.class_indices.items()} if test_generator else None
# ---------------------------------------------------------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    # Use model.layers[0].input to explicitly get the input tensor from the first layer
    grad_model = tf.keras.models.Model(
        inputs=model.layers[0].input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.layers[-1].output]
    )

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        # Pass the watched tensor through the grad_model
        # Since model.layers[0].input is a single tensor, call grad_model with a single tensor
        last_conv_layer_output, preds = grad_model(img_tensor)

        # Explicitly watch the output of the convolutional layer
        # This ensures gradients can be computed with respect to it.
        tape.watch(last_conv_layer_output)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Handle case where grads might still be None (e.g., if there's no path)
    if grads is None:
        # Fallback or raise an error
        print(f"Warning: Gradients are None for layer {last_conv_layer_name}. Cannot generate heatmap.")
        # Returning an array of zeros ensures the rest of the code can proceed without error
        return np.zeros(last_conv_layer_output.shape[1:3]), preds[0].numpy()

    # Get the value of the gradients by performing global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    # with respect to the selected class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization, normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), preds[0].numpy()

def display_gradcam(img_path, heatmap, original_img, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + original_img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img



if best_model and dataset_found and all_test_image_paths and idx_to_class:
    # Get the name of the last convolutional layer from the model summary
    last_conv_layer_name = 'conv2d_5'

    # Select N random images to display Grad-CAM
    num_images_to_display = 5
    random_image_paths = random.sample(all_test_image_paths, min(num_images_to_display, len(all_test_image_paths)))

    plt.figure(figsize=(15, 5 * len(random_image_paths)))

    for i, img_path in enumerate(random_image_paths):
        # Load and preprocess the image
        original_img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(original_img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Extract true class name
        true_class_name = os.path.basename(os.path.dirname(img_path))

        # Generate Grad-CAM heatmap and predictions
        heatmap, predictions = make_gradcam_heatmap(img_array, best_model, last_conv_layer_name)

        # Only display if heatmap was successfully generated (not all zeros)
        if np.any(heatmap):
            predicted_class_idx = np.argmax(predictions)
            predicted_class_name = idx_to_class[predicted_class_idx]
            confidence = predictions[predicted_class_idx]

            # Display original image with heatmap
            gradcam_img = display_gradcam(img_path, heatmap, np.array(original_img))

            plt.subplot(len(random_image_paths), 2, 2 * i + 1)
            plt.imshow(original_img)
            plt.title(f"Original\nTrue: {true_class_name}")
            plt.axis('off')

            plt.subplot(len(random_image_paths), 2, 2 * i + 2)
            plt.imshow(gradcam_img)
            plt.title(f"Grad-CAM\nPred: {predicted_class_name} ({confidence:.2f})")
            plt.axis('off')
        else:
            print(f"Skipping Grad-CAM for {img_path} due to missing gradients.")

    plt.tight_layout()
    plt.show()

    print(f"Grad-CAM visualizations for {len(random_image_paths)} random images displayed.")
else:
    print("Skipping Grad-CAM visualization due to missing model, dataset, test images, or class mappings.")



import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Ensure best_model is loaded if not already in memory
if 'best_model' not in locals() and 'best_model' not in globals():
    model_path = 'best_model.h5'
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Please ensure the model training cell (94f2f8d6) was successfully executed and the model was saved.")
        best_model = None

if best_model:
    best_model.summary()
else:
    print("Model could not be loaded, so summary cannot be displayed.")


import random
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

# Ensure best_model is loaded if not already in memory
if 'best_model' not in locals() and 'best_model' not in globals():
    model_path = 'best_model.h5'
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        print(f"Loaded best model from {model_path}")
    else:
        print(f"Error: '{model_path}' not found. Cannot proceed with classification analysis.")
        best_model = None

# Ensure img_width, img_height, all_test_image_paths, idx_to_class are defined
# (These should be available from previous cells, but re-checking for robustness)
if not all(var in globals() or var in locals() for var in ['img_width', 'img_height', 'all_test_image_paths', 'idx_to_class']):
    print("Error: Missing one or more required variables (img_width, img_height, all_test_image_paths, idx_to_class). Please ensure previous cells have been executed.")
    best_model = None # Prevent execution if dependencies are missing

correctly_classified_images_per_class = {
    class_name: [] for class_name in idx_to_class.values()
}

images_per_class_limit = 3 # Number of correctly classified images to collect per class

if best_model and all_test_image_paths and idx_to_class:
    print("Identifying correctly classified images for each class...")

    # Shuffle paths to ensure random selection for correct classifications
    random.shuffle(all_test_image_paths)

    for img_path in all_test_image_paths:
        # Check if we have collected enough images for all classes
        if all(len(v) >= images_per_class_limit for v in correctly_classified_images_per_class.values()):
            break

        # Load and preprocess the image
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Extract true class name
        true_class_name = os.path.basename(os.path.dirname(img_path))

        # Make prediction
        predictions = best_model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = idx_to_class[predicted_class_idx]
        confidence_score = predictions[0][predicted_class_idx]

        # If correctly classified and we haven't reached the limit for this class
        if predicted_class_name == true_class_name and \
           len(correctly_classified_images_per_class[true_class_name]) < images_per_class_limit:
            correctly_classified_images_per_class[true_class_name].append({
                'image_path': img_path,
                'true_label': true_class_name,
                'predicted_label': predicted_class_name,
                'confidence_score': confidence_score
            })

    print("Summary of collected correctly classified images per class:")
    for class_name, images in correctly_classified_images_per_class.items():
        print(f"  - {class_name}: {len(images)} images")
else:
    print("Skipping identification of correctly classified images due to missing model or other dependencies.")


import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import load_img, img_to_array

# Ensure all necessary components are available
if not all(var in globals() or var in locals() for var in [
    'best_model', 'img_width', 'img_height', 'idx_to_class',
    'correctly_classified_images_per_class', 'make_gradcam_heatmap', 'display_gradcam'
]):
    print("Error: Missing one or more required variables/functions for Grad-CAM visualization. Please ensure previous cells have been executed.")
else:
    print("Generating Grad-CAM visualizations for correctly classified images...")

    # Get the name of the last convolutional layer from the model summary
    # Assuming 'conv2d_5' is the last convolutional layer as observed in model.summary()
    last_conv_layer_name = 'conv2d_5'

    # Determine the total number of images to display
    total_images_to_display = sum(len(images) for images in correctly_classified_images_per_class.values())

    plt.figure(figsize=(15, 5 * total_images_to_display))
    plot_idx = 0

    for class_name, images_list in correctly_classified_images_per_class.items():
        print(f"Displaying Grad-CAM for correctly classified '{class_name}' images...")
        for image_info in images_list:
            plot_idx += 1
            img_path = image_info['image_path']
            true_label = image_info['true_label']
            predicted_label = image_info['predicted_label']
            confidence = image_info['confidence_score']

            # Load and preprocess the image
            original_img = load_img(img_path, target_size=(img_width, img_height))
            img_array = img_to_array(original_img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Generate Grad-CAM heatmap and predictions
            heatmap, predictions_all = make_gradcam_heatmap(img_array, best_model, last_conv_layer_name)

            # Only display if heatmap was successfully generated (not all zeros)
            if np.any(heatmap):
                gradcam_img = display_gradcam(img_path, heatmap, np.array(original_img))

                # Original Image subplot
                plt.subplot(total_images_to_display, 2, 2 * (plot_idx - 1) + 1)
                plt.imshow(original_img)
                plt.title(f"Original ({class_name})\nTrue: {true_label}")
                plt.axis('off')

                # Grad-CAM Image subplot
                plt.subplot(total_images_to_display, 2, 2 * (plot_idx - 1) + 2)
                plt.imshow(gradcam_img)
                plt.title(f"Grad-CAM (Predicted: {predicted_label} ({confidence:.2f}))")
                plt.axis('off')
            else:
                print(f"Skipping Grad-CAM for {img_path} due to missing gradients or failed heatmap generation.")

    plt.tight_layout()
    plt.show()

    print(f"Grad-CAM visualizations for {total_images_to_display} correctly classified images displayed.")

