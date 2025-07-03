import os

ORIGINAL_DATASET_DIR = "/Users/shir/Desktop/ANN_FINAL/dataset"
SPLIT_DATASET_DIR = "/Users/shir/Desktop/ANN_FINAL/split_dataset"
TRAIN_DIR = os.path.join(SPLIT_DATASET_DIR, "train")
TEST_DIR = os.path.join(SPLIT_DATASET_DIR, "test")
MODEL_SAVE_PATH = "/Users/shir/Desktop/ANN_FINAL/vgg16_model.h5"

SPLIT_RATIO = 0.8
SEED = 42
IMG_HEIGHT = 384
IMG_WIDTH = 512
BATCH_SIZE = 32
EPOCHS = 5
