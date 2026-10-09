import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

coco = COCO("annotations/instances_val2017.json")

person_id = coco.getCatIds(catNms=["person"])[0]
person_imgs = coco.getImgIds(catIds=[person_id])

widths = []
heights = []
areas = []

for img_id in person_imgs:
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_id])
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        box = ann["bbox"]
        w = box[2]
        h = box[3]
        widths.append(w)
        heights.append(h)
        areas.append(w * h)

print("total boxes:", len(areas))

scales = []
for a in areas:
    scales.append(np.sqrt(a))

print("median scale:", np.median(scales))

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(widths, bins=40)
plt.title("box widths")

plt.subplot(1, 3, 2)
plt.hist(heights, bins=40)
plt.title("box heights")

plt.subplot(1, 3, 3)
plt.hist(scales, bins=40)
plt.title("object scale")

plt.show()