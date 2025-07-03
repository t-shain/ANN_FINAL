import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import FINETUNED_MODEL_SAVE_PATH, MODEL_SAVE_PATH
import tensorflow as tf
from config import TEST_DIR, SEED, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

models = [FINETUNED_MODEL_SAVE_PATH, MODEL_SAVE_PATH]

for model in models:
    print(f"Evaluating model: {model}")
    model = tf.keras.models.load_model(model)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    y_pred_probs = model.predict(test_ds)  
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    cm = confusion_matrix(y_true, y_pred)

    class_names = test_ds.class_names 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_finetuned.png')