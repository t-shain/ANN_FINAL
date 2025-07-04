import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define constants
IMAGE_SIZE = (224, 224)  # Common size for many models (can adjust)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = ...  # This will be automatically determined from folders

# Path to your dataset (should contain subfolders for each class)
train_data_dir = '/kaggle/input/garbage-classification/Garbage classification/Garbage classification/'


# Create data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # for multi-class classification
    shuffle=True,
    subset='training'  # Only needed if using validation_split
)

# Create test data generator - typically only rescaling, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=10,            
    class_mode='categorical',  # same as training
    shuffle=False  
    subset='testing'# important for consistent evaluation
)

# Get the number of classes from the generator
NUM_CLASSES = len(train_generator.class_indices)
print(f"Found {NUM_CLASSES} classes: {train_generator.class_indices}")

# Build a simple CNN model
model = Sequential([
    Conv2D(20, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(40, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(80, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
)

# Save the model
model.save('garbage-classifier-custom.h5')

# Option 1: Simple evaluation
results = model.evaluate(test_generator)
print(f"Evaluation results - Loss: {results[0]:.4f}, Accuracy: {results[1]:.2%}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have a trained model and generators set up
# train_generator and validation_generator created with flow_from_directory()

def plot_confusion_matrix(generator, model):
    # Reset generator to ensure we start from the beginning
    generator.reset()
    
    # Get all predictions
    predictions = model.predict(generator, steps=len(generator), verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = generator.classes
    class_labels = list(generator.class_indices.keys())
    
    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    return cm

# Usage example:
cm = plot_confusion_matrix(train_generator, model)
