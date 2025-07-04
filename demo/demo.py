import argparse
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import os

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] 
MODEL_SAVE_PATH = "/Users/shir/Desktop/ANN_FINAL/models/vgg16_model.h5"
IMG_HEIGHT = 384
IMG_WIDTH = 512

def download_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Error downloading image: {e}")


def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(model, image_tensor):
    predictions = model.predict(image_tensor)
    class_idx = np.argmax(predictions)
    return CLASS_NAMES[class_idx], predictions[0][class_idx]

def main():
    parser = argparse.ArgumentParser(description="Classify an image using a fine-tuned VGG16 model.")
    parser.add_argument("image_path", help="Path to the local image file")
    args = parser.parse_args()

    print(f"[INFO] Loading image from: {args.image_path}")
    image_tensor = load_and_preprocess_image(args.image_path)

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    print("[INFO] Predicting...")
    label, confidence = predict(model, image_tensor)

    print(f"\nPrediction: {label} ({confidence * 100:.2f}%)")

if __name__ == "__main__":
    main()