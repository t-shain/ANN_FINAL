import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    batch_size=1,            
    class_mode='categorical',  # same as training
    shuffle=False  
    subset='testing'
)

# Get the number of classes from the generator
NUM_CLASSES = len(train_generator.class_indices)
print(f"Found {NUM_CLASSES} classes: {train_generator.class_indices}")

## generate X,y training and testing data from the generators
def generator_to_train_test_arrays(generator, test_size=0.2, random_state=None):
    """
    Convert a directory flow generator into separate train/test arrays.
    
    Args:
        generator: A generator created with flow_from_directory
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Arrays containing the split data
    """
    # Get all data from the generator
    X = []
    y = []
    
    # Calculate number of batches needed to get all data
    num_batches = int(np.ceil(generator.samples / generator.batch_size))
    
    for i in range(num_batches):
        batch_X, batch_y = next(generator)
        X.append(batch_X)
        y.append(batch_y)
        
        # Reset generator if we've reached the end
        if (i + 1) * generator.batch_size >= generator.samples:
            generator.reset()
    
    # Concatenate all batches
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = generator_to_train_test_arrays(train_generator)



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

# train the model
history = model.fit(x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=None,
    shuffle=True,
    steps_per_epoch= len(X_train) // BATCH_SIZE,)

# Save the model
model.save('garbage-classifier-custom.h5')

# Option 1: Simple evaluation
results = model.evaluate(X_test, y_test, batch_size=64)
print(f"Evaluation results - Loss: {results[0]:.4f}, Accuracy: {results[1]:.2%}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('garbage-classifier-custom.h5')
# 1. Get predictions (use X_test if you want validation performance)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # If y_train is one-hot encoded

# 2. Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# 3. Plot with Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_generator.class_indices.keys(), 
            yticklabels=train_generator.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Training Data)')
plt.show()
