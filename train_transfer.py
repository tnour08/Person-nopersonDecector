import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix

IMG_DIR = "data/images/"
IMG_SIZE = 96
BATCH_SIZE = 32

#read labels
data = []
with open("data/labels.csv") as f:
    for line in f:
        name, label = line.strip().split(",")
        data.append((IMG_DIR + name, int(label)))

random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.8)
train_data = data[:split]
val_data = data[split:]
print("train:", len(train_data), "val:", len(val_data))

#image loading, MobileNetV2 preprocessing
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = keras.applications.mobilenet_v2.preprocess_input(img)
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

base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

#train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# plot training vs validation loss
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("outputs/loss_curve.png")
plt.show()

# --- metrics with scikit-learn ---
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend((preds > 0.5).astype(int).flatten())
    y_true.extend(labels.numpy())

print("\n--- metrics on validation set ---")
print("precision:", round(precision_score(y_true, y_pred), 3))
print("recall:", round(recall_score(y_true, y_pred), 3))
print("confusion matrix:")
print(confusion_matrix(y_true, y_pred))