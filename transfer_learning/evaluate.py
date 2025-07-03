
import tensorflow as tf
from ..config import MODEL_SAVE_PATH, TRAIN_DIR, TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, SEED 
import matplotlib.pyplot as plt

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

loaded_test_loss, loaded_test_acc = loaded_model.evaluate(test_ds)

print(f"Model test accuracy: {loaded_test_acc:.4f}")
print(f"Model test loss: {loaded_test_loss:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.axhline(y=test_acc, color='r', linestyle='--', label='Test Accuracy')
plt.title('Train vs Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('Train vs Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the plots
plt.savefig('/Users/shir/Desktop/ANN_FINAL/training_plots.png')