
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from config import *

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

base_model = VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                   include_top=False,
                   weights='imagenet')

print("Layer Trainable Status:")
for i, layer in enumerate(base_model.layers):
    print(f"{i:2d}. Layer Name: {layer.name:30s} | Trainable: {layer.trainable}")

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=EPOCHS)
print("Training completed. Saving the model...")
model.save(FINETUNED_MODEL_SAVE_PATH)

print("Print accuracies and losses")
train_losses = history.history['loss']
train_accuracies = history.history['accuracy']

print("Train Losses:", train_losses)
print("Train Accuracies:", train_accuracies)

epochs = range(1, len(train_losses) + 1)

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'o-', label='Train Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'o-', label='Train Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/shir/Desktop/ANN_FINAL/fine_tuning_plots.png')
print(f"Model saved to {FINETUNED_MODEL_SAVE_PATH}")
print("Fine-tuning completed successfully.")


