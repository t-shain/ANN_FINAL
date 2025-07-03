import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from config import IMG_HEIGHT, IMG_WIDTH, ORIGINAL_DATASET_DIR

IMG_SIZE = (224, 224)  # VGG16 expects 224x224 input size
TOP_N = 15  # Top predictions per class
model = VGG16(weights='imagenet')

# Iterate over each material class
classes = sorted(os.listdir(ORIGINAL_DATASET_DIR))
for cls in classes:
    class_dir = os.path.join(ORIGINAL_DATASET_DIR, cls)
    if not os.path.isdir(class_dir):
        continue
    
    image_paths = [
        os.path.join(class_dir, file)
        for file in os.listdir(class_dir)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    print(f"\nProcessing class '{cls}' with {len(image_paths)} images")
    
    imagenet_labels = []
    for img_path in tqdm(image_paths, desc=f"Classifying {cls}"):
        try:
            img = image.load_img(img_path, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            top_label = decode_predictions(preds, top=1)[0][0][1]
            imagenet_labels.append(top_label)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            continue
    
    counter = Counter(imagenet_labels)
    most_common = counter.most_common(TOP_N)
    labels, values = zip(*most_common) if most_common else ([], [])
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top {TOP_N} ImageNet Predictions for '{cls}'")
    plt.ylabel("Count")
    plt.xlabel("ImageNet Class Labels")
    plt.tight_layout()
    save_path = f"/Users/shir/Desktop/ANN_FINAL/imagenet_histogram_{cls}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Saved histogram for {cls} to {save_path}")
