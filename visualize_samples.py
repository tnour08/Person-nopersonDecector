import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

coco = COCO("annotations/instances_val2017.json")

person_id = coco.getCatIds(catNms=["person"])[0]
person_imgs = coco.getImgIds(catIds=[person_id])

samples = []
for img_id in person_imgs:
    info = coco.loadImgs(img_id)[0]
    path = "data/images/" + info["file_name"]
    if os.path.exists(path):
        samples.append((img_id, path))
    if len(samples) == 12:
        break

plt.figure(figsize=(15, 10))

for i in range(len(samples)):
    img_id = samples[i][0]
    path = samples[i][1]
    image = cv2.imread(path)

    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_id])
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        box = ann["bbox"]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.subplot(3, 4, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.show()