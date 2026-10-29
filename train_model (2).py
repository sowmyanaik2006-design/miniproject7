import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

# Dataset paths
TRAIN_DIR = "DATASET/TRAIN"
TEST_DIR = "DATASET/TEST"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5  # Pehle 5 se test kar, baad me 10 kar dena

# --- AUTO CREATE DUMMY DATASET IF NOT FOUND ---
def create_dummy_dataset():
    print("Dataset nahi mila, dummy dataset bana raha hu...")
    classes = ["organic", "recyclable"]
    for split in [TRAIN_DIR, TEST_DIR]:
        for cls in classes:
            path = os.path.join(split, cls)
            os.makedirs(path, exist_ok=True)
            # Har class me 20 dummy images banao
            for i in range(20):
                img_path = os.path.join(path, f"img_{i}.jpg")
                if not os.path.exists(img_path):
                    # Random color image
                    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(img_path)
    print("Dummy dataset ban gaya!")

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
    create_dummy_dataset()

# Ab load karo
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", seed=42
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("waste_classifier.keras")
with open("classes.txt", "w") as f:
    f.write("\n".join(class_names))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.savefig("training_accuracy.png")
plt.close()

print("DONE! Model saved")