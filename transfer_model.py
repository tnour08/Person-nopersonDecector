import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 96

# load MobileNetV2 pre-trained on ImageNet, without its top classifier
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,        # drop its 1000-class ImageNet head
    weights="imagenet",       # keep the pre-trained weights
)

# freeze the base so its learned features aren't overwritten during training
base.trainable = False

# build our own model on top of the frozen base
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()