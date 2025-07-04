
import tensorflow as tf
from config import *
import matplotlib.pyplot as plt

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, 
    shuffle=False  
)

loaded_model = tf.keras.models.load_model(FINETUNED_MODEL_SAVE_PATH)

print("Evaluate...")
test_losses, test_acc = loaded_model.evaluate(test_ds)

print(f"Model test loss: {test_losses:.4f}")
print(f"Model test accuracy: {test_acc:.4f}")

# Loss and accuracy values for VGG transfer learning (without fine-tuning)
#train_losses = [2.1663, 0.6999, 0.5349, 0.4618, 0.3947]
#train_accuracies = [0.4633, 0.7351, 0.7948, 0.8309, 0.8534]

train_losses = [2.230436086654663, 1.3060144186019897, 1.0521023273468018, 0.8838033676147461, 0.7557089328765869]
train_accuracies = [0.2872709333896637, 0.4814264476299286, 0.5898959636688232, 0.6726102232933044, 0.7211490869522095]

epochs = range(1, len(train_losses) + 1)

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'o-', label='Train Loss')
plt.axhline(y=test_losses, color='r', linestyle='--', label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'o-', label='Train Accuracy')
plt.axhline(y=test_acc, color='r', linestyle='--', label='Test Loss')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('/Users/shir/Desktop/ANN_FINAL/fine_tuned_training_plots.png')