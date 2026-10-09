import tensorflow as tf

IMG_DIR = "data/images/"
IMG_SIZE = 96
BATCH_SIZE = 32

# read the label file we just made
filenames = []
labels = []
with open("data/labels.csv") as f:
    for line in f:
        name, label = line.strip().split(",")
        filenames.append(IMG_DIR + name)
        labels.append(int(label))

# a function that loads and preprocesses one image
def load_image(path, label):
    data = tf.io.read_file(path)
    image = tf.io.decode_jpeg(data, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0                # scale pixels from 0-255 to 0-1
    return image, label

# build the dataset
ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
ds = ds.map(load_image)
ds = ds.batch(BATCH_SIZE)

# check one batch
for images, labels in ds.take(1):
    print("batch of images:", images.shape)
    print("batch of labels:", labels.shape)
    print("pixel range:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))