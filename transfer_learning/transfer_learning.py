
import shutil
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from ..config import *

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset():
    random.seed(SEED)
    create_dir(TRAIN_DIR)
    create_dir(TEST_DIR)

    for class_name in os.listdir(ORIGINAL_DATASET_DIR):
        class_path = os.path.join(ORIGINAL_DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        create_dir(train_class_dir)
        create_dir(test_class_dir)

        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)

if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
    split_dataset()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

base_model = VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                   include_top=False,
                   weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=EPOCHS)
model.save(MODEL_SAVE_PATH)
