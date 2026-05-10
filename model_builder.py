from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# Define the input shape for the model
input_shape = (img_width, img_height, 3) # 3 for RGB images
num_classes = train_generator.num_classes

# Initialize the Sequential model
model = Sequential()

# Add Input layer explicitly as the first layer to avoid UserWarning
model.add(Input(shape=input_shape))

# Convolutional Block 1
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from convolutional layers
model.add(Flatten())

# Dense layers (classification head)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) # Dropout layer to reduce overfitting
model.add(Dense(num_classes, activation='softmax')) # Output layer with softmax for multi-class classification

# Print the model summary
model.summary()

print("Model architecture defined successfully.")


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001), # Using Adam optimizer with a default learning rate
    loss='categorical_crossentropy',   # Suitable for multi-class classification
    metrics=['accuracy']               # Monitor accuracy during training
)

# Define Callbacks
# EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=10,             # Stop if validation accuracy doesn't improve for 10 epochs
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# ModelCheckpoint to save the best model during training
model_checkpoint = ModelCheckpoint(
    filepath='best_model.h5',      # Path to save the model file
    monitor='val_accuracy',      # Monitor validation accuracy
    save_best_only=True,         # Save only the best model
    mode='max',                  # Save when 'val_accuracy' is maximized
    verbose=1                    # Log when a new best model is saved
)

# List of callbacks to be used during training
callbacks = [early_stopping, model_checkpoint]

print("Model compiled and callbacks defined successfully.")

epochs = 50 # You can adjust this number based on your computational resources and convergence
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=callbacks
)

print("Model training initiated.")

