import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import os

IMG_DIR = "data/images/"
IMG_SIZE = 96
BATCH_SIZE = 32

# --- read the labels ---
filenames = []
labels = []
with open("data/labels.csv") as f:
    for line in f:
        name, label = line.strip().split(",")
        filenames.append(IMG_DIR + name)
        labels.append(int(label))

# --- split into train and validation ---
data = list(zip(filenames, labels))
random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.8)
train_data = data[:split]
val_data = data[split:]
print("train:", len(train_data), "val:", len(val_data))

# --- image loading ---
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

def make_ds(data):
    paths = [d[0] for d in data]
    labs = [d[1] for d in data]
    ds = tf.data.Dataset.from_tensor_slices((paths, labs))
    ds = ds.map(load_image)
    ds = ds.batch(BATCH_SIZE)
    return ds

train_ds = make_ds(train_data)
val_ds = make_ds(val_data)

# --- the model (with augmentation + dropout) ---
model = keras.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# --- train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# --- plot ---
os.makedirs("outputs", exist_ok=True)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("After regularisation")
plt.savefig("outputs/after_regularisation.png")
plt.show()