# Per[README.md](https://github.com/user-attachments/files/33257477/README.md)
Person Detector: Binary Image Classification with a CNN and MobileNetV2

A binary image classifier that predicts whether a **person is present** in an image. It was built step by step as a lightweight model aimed at **edge deployment**. The project goes from a custom CNN trained from scratch, to a regularised CNN, to transfer learning with a pre-trained **MobileNetV2** backbone. Validation accuracy rose from **~60% to ~82%**.

**Tech stack:** Python · TensorFlow / Keras · OpenCV · scikit-learn · pycocotools · Matplotlib

---

## Highlights

- **End-to-end pipeline:** data download, labelling, `tf.data` input pipeline, training, evaluation.
- **No 25 GB download:** the project uses the COCO annotations to pick a **balanced ~4,600-image subset** and downloads only those images.
- **Custom CNN from scratch,** with overfitting diagnosed from the train/val curves and reduced using **dropout and data augmentation**.
- **Transfer learning** with a frozen ImageNet-pretrained **MobileNetV2** base and a small custom classification head.
- **Evaluation** with accuracy, precision, recall and a confusion matrix.

---

## Results

| Stage | Model | Val accuracy |
|---|---|---|
| 1. Baseline | Custom 3-block CNN | ~60% (peaks ~62%, then falls as the model overfits) |
| 2. Regularised | CNN + augmentation + dropout | ~64% (train and val now track each other) |
| 3. Transfer learning | Frozen MobileNetV2 + custom head | **~82%** |

### 1. Baseline: overfitting
Training accuracy climbs to ~87%, but validation accuracy peaks around epoch 3 and then declines. The model is memorising the training set.

![Before regularisation](outputs/before_regularisation.png)

### 2. With augmentation and dropout
`RandomFlip`, `RandomRotation` and `Dropout(0.5)` close the gap between training and validation accuracy. Both curves rise together, but a small CNN trained from scratch on ~3,700 images reaches its limit in the mid-60s.

![After regularisation](outputs/after_regularisation.png)

### 3. Transfer learning: loss curves
With a frozen MobileNetV2 base, validation loss drops quickly and levels off around 0.40. These curves were used to diagnose convergence and tune the learning rate.

![Training vs validation loss](outputs/loss_curve.png)

---

## Dataset

The data comes from the [COCO 2017](https://cocodataset.org/) **val2017** split (5,000 images):

- **Person:** images with at least one `person` annotation, label `1`.
- **No person:** all other images, label `0`.
- The classes are balanced by taking the same number from each: 2 × 2,307 = **4,614 images**.
- The split is 80/20 train/validation with a fixed seed (`random.seed(42)`): about 3,691 train and 923 validation images.
- Images are resized to **96 × 96**, a small input size suited to edge devices.

### Annotations

Place the COCO annotation files in `annotations/`:

| File | Used for |
|---|---|
| `instances_val2017.json` | **Required.** Selects person/no-person images, creates labels, profiles boxes, draws sample boxes |
| `person_keypoints_val2017.json` | Optional, for exploration (person keypoints) |
| `captions_val2017.json` | Optional, for exploration (image captions) |
| `*_train2017.json` | Not used by the current scripts |

> The `*_train2017.json` files are larger than GitHub's 100 MB file limit, so `.gitignore` excludes them. Download them from [cocodataset.org](https://cocodataset.org/#download) (`annotations_trainval2017.zip`) if you need them.

---

## Project structure

```
.
├── annotations/                 # COCO annotation JSON files
├── data/
│   ├── images/                  # downloaded subset (git-ignored, created by download_data.py)
│   └── labels.csv               # filename,label (created by make_labels.py)
├── outputs/                     # saved training plots
│
├── load_images.py               # OpenCV basics: grayscale, crop, resize
├── visualize_samples.py         # draws COCO person bounding boxes on sample images
├── profile_data.py              # histograms of person box widths, heights and scales
├── download_data.py             # downloads the balanced person / no-person subset
├── make_labels.py               # writes data/labels.csv and checks class balance
├── pipeline.py                  # tf.data pipeline: load, decode, resize, normalise, batch
├── model.py                     # custom CNN architecture (summary only)
├── train_baseline.py            # stage 1: CNN with no regularisation
├── train_regularized.py         # stage 2: CNN + augmentation + dropout
├── transfer_model.py            # MobileNetV2 transfer model (summary only)
├── train_transfer.py            # stage 3: train MobileNetV2 model, compute metrics
└── requirements.txt
```

---

## Getting started

### 1. Install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

Put `instances_val2017.json` in `annotations/`, then run:

```bash
python download_data.py   # downloads ~4,600 images into data/images/
python make_labels.py     # writes data/labels.csv and prints the class balance
```

### 3. (Optional) Explore the data

```bash
python visualize_samples.py   # sample images with person bounding boxes
python profile_data.py        # box size and scale distributions
python pipeline.py            # checks one batch's shape and pixel range
```

### 4. Train

```bash
python train_baseline.py      # stage 1, saves outputs/before_regularisation.png
python train_regularized.py   # stage 2, saves outputs/after_regularisation.png
python train_transfer.py      # stage 3, saves outputs/loss_curve.png and prints metrics
```

`train_transfer.py` prints precision, recall and the confusion matrix on the validation set.

---

## Model details

### Custom CNN (stages 1 and 2)

```
Input 96×96×3
[RandomFlip(horizontal), RandomRotation(0.1)]   ← stage 2 only
Conv2D(16, 3×3, ReLU) → MaxPool
Conv2D(32, 3×3, ReLU) → MaxPool
Conv2D(64, 3×3, ReLU) → MaxPool
Flatten
[Dropout(0.5)]                                  ← stage 2 only
Dense(64, ReLU)
Dense(1, sigmoid)
```

- Optimiser: Adam · Loss: binary cross-entropy · Pixels scaled to [0, 1].

### MobileNetV2 transfer model (stage 3)

```
Input 96×96×3
MobileNetV2 (ImageNet weights, include_top=False, frozen)
GlobalAveragePooling2D
Dropout(0.3)
Dense(1, sigmoid)
```

- MobileNetV2 is built for mobile and embedded devices: it uses depthwise-separable convolutions and has few parameters, which suits edge deployment.
- Inputs use `mobilenet_v2.preprocess_input`, which scales pixels to [-1, 1].
- Optimiser: Adam. The learning rate was tuned using the loss curves: `train_transfer.py` uses `1e-3` and `transfer_model.py` uses `1e-4`.
- Only the classification head is trained. The backbone stays frozen.

---

## Key lessons

- **Read the curves, not just the final number.** The baseline's training accuracy kept rising while validation accuracy fell. That gap is the clearest sign of overfitting.
- **Regularisation trades training accuracy for generalisation.** Augmentation and dropout lowered training accuracy but made validation accuracy track it.
- **Pretrained features beat more tuning on small data.** With ~3,700 training images, ImageNet features gave a larger gain than any change to the small CNN.

---

## Future work

- [ ] Fine-tune the top MobileNetV2 layers with a low learning rate.
- [ ] Add `shuffle`, `cache` and `prefetch` to the `tf.data` pipeline.
- [ ] Use a separate test set (for example, a balanced subset of train2017) for an unbiased final score.
- [ ] Add callbacks: `EarlyStopping`, `ModelCheckpoint`.
- [ ] Export to **TensorFlow Lite** with int8 quantisation and benchmark on an edge device such as a Raspberry Pi.
- [ ] Add an inference script for a single image or webcam feed (OpenCV).

---

## Acknowledgements

- [COCO dataset](https://cocodataset.org/), Lin et al., *Microsoft COCO: Common Objects in Context* (2014). The annotations are licensed CC BY 4.0.
- [MobileNetV2](https://arxiv.org/abs/1801.04381), Sandler et al. (2018), via `tf.keras.applications`.
