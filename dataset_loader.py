
# Install the Kaggle library
!pip install -q kaggle

# Create a directory for Kaggle credentials
!mkdir -p ~/.kaggle

# Upload your Kaggle API token (kaggle.json)
from google.colab import files
files.upload()

# Move the uploaded kaggle.json to the .kaggle directory
!mv kaggle.json ~/.kaggle/

# Set permissions for the kaggle.json file
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# The downloaded file name is typically the dataset name with .zip appended
dataset_zip_file = 'brain-tumor-mri-dataset.zip'

import zipfile
import os

output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(dataset_zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"'{dataset_zip_file}' extracted to '{output_dir}' successfully.")


import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the base path to the extracted dataset directory
base_dataset_path = 'dataset'

# Check if there's a single subdirectory within 'dataset' (common for Kaggle zips)
# and adjust the dataset_path to point to that subdirectory
subdirs = [d for d in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, d))]
if len(subdirs) == 1:
    dataset_path = os.path.join(base_dataset_path, subdirs[0])
else:
    dataset_path = base_dataset_path # Assume 'Training'/'Testing' are directly under 'dataset' if no single subdir

print(f"Using dataset path: {dataset_path}")

# Dictionary to store image counts per class (e.g., 'Training', 'Testing')
class_counts = {}

# Dictionary to store paths of sample images for visualization, grouped by top-level class (e.g., 'Training')
sample_images = {}

# Iterate through the main subdirectories (e.g., 'Training', 'Testing') within the determined dataset path
for main_class_name in os.listdir(dataset_path):
    main_class_dir = os.path.join(dataset_path, main_class_name)
    if os.path.isdir(main_class_dir):
        images_in_main_class = []
        # Iterate through the sub-class directories (e.g., 'glioma', 'meningioma')
        for sub_class_name in os.listdir(main_class_dir):
            sub_class_dir = os.path.join(main_class_dir, sub_class_name)
            if os.path.isdir(sub_class_dir):
                images_in_sub_class = [
                    os.path.join(sub_class_dir, img)
                    for img in os.listdir(sub_class_dir)
                    if img.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                images_in_main_class.extend(images_in_sub_class)

        class_counts[main_class_name] = len(images_in_main_class)

        # Select a few random images for visualization from the current main class
        if len(images_in_main_class) > 0:
            sample_images[main_class_name] = random.sample(images_in_main_class, min(3, len(images_in_main_class))) # Get up to 3 samples per class

# Print the total number of images found and the count of images per class
total_images = sum(class_counts.values())
print(f"Total number of images found: {total_images}")
print("Image distribution per class:")
for class_name, count in class_counts.items():
    print(f"- {class_name}: {count} images")

# Visualization of sample images
print("\nSample Images:")
for class_name, samples in sample_images.items():
    if samples:
        print(f"Displaying samples for: {class_name}")
        plt.figure(figsize=(12, 4))
        for i, img_path in enumerate(samples):
            plt.subplot(1, len(samples), i + 1)
            img = mpimg.imread(img_path)
            plt.imshow(img, cmap='gray') # Assuming MRI images are grayscale
            # Extract the actual tumor type (sub-class name) from the path for the title
            tumor_type = os.path.basename(os.path.dirname(img_path))
            plt.title(f"{class_name}: {tumor_type}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


import os

# Assuming dataset_path is already defined from the previous cell
# If not, for reproducibility in a fresh run, it would be:
# base_dataset_path = 'dataset'
# subdirs = [d for d in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, d))]
# if len(subdirs) == 1:
#     dataset_path = os.path.join(base_dataset_path, subdirs[0])
# else:
#     dataset_path = base_dataset_path

# Initialize dictionary to store image counts for each specific tumor type
tumor_type_counts = {}

# Iterate through the main directories (e.g., 'Training', 'Testing')
for main_split_name in os.listdir(dataset_path):
    main_split_dir = os.path.join(dataset_path, main_split_name)
    if os.path.isdir(main_split_dir):
        # Initialize a sub-dictionary for the current main split
        tumor_type_counts[main_split_name] = {}
        # Iterate through the subdirectories (tumor types) within the main split
        for tumor_type_name in os.listdir(main_split_dir):
            tumor_type_dir = os.path.join(main_split_dir, tumor_type_name)
            if os.path.isdir(tumor_type_dir):
                # Count image files in the current tumor type directory
                images_in_tumor_type = [
                    img for img in os.listdir(tumor_type_dir)
                    if img.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                tumor_type_counts[main_split_name][tumor_type_name] = len(images_in_tumor_type)

# Print a summary of the image distribution per tumor type within each split
print(f"Detailed image distribution per class and tumor type:")
total_dataset_images = 0
for main_split, tumor_types in tumor_type_counts.items():
    print(f"\n--- {main_split} ---")
    split_total = 0
    for tumor_type, count in tumor_types.items():
        print(f"- {main_split} - {tumor_type}: {count} images")
        split_total += count
    print(f"Total for {main_split}: {split_total} images")
    total_dataset_images += split_total

print(f"\nGrand Total of images in dataset: {total_dataset_images}")




import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define target image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Construct the full path to the training and testing directories
train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Testing')

# Data Augmentation and Preprocessing for Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Normalize pixel values to [0, 1]
    rotation_range=20,              # Randomly rotate images by up to 20 degrees
    zoom_range=0.2,                 # Randomly zoom images by 20%
    width_shift_range=0.2,          # Randomly shift images horizontally by 20%
    height_shift_range=0.2,         # Randomly shift images vertically by 20%
    shear_range=0.2,                # Apply shear transformation
    horizontal_flip=True,           # Randomly flip images horizontally
    fill_mode='nearest'             # Strategy for filling in new pixels created by transformations
)

# Preprocessing for Test Data (only rescaling, no augmentation)
test_datagen = ImageDataGenerator(
    rescale=1./255                  # Normalize pixel values to [0, 1]
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Data generators created successfully.")
print(f"Training classes: {train_generator.class_indices}")
print(f"Test classes: {test_generator.class_indices}")