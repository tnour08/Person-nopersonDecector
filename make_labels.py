import os
from pycocotools.coco import COCO

coco = COCO("annotations/instances_val2017.json")

person_id = coco.getCatIds(catNms=["person"])[0]
person_imgs = set(coco.getImgIds(catIds=[person_id]))

# go through the images we actually downloaded
img_dir = "data/images"
rows = []

for fname in os.listdir(img_dir):
    if not fname.endswith(".jpg"):
        continue
    # the filename is the image id with leading zeros, for example 000000391895.jpg
    img_id = int(fname.replace(".jpg", ""))
    label = 1 if img_id in person_imgs else 0
    rows.append(fname + "," + str(label))

# save to a text file, one "filename,label" per line
with open("data/labels.csv", "w") as f:
    f.write("\n".join(rows))

print("wrote", len(rows), "labels")

# quick sanity check on the balance
labels = [int(r.split(",")[1]) for r in rows]
print("person:", sum(labels), "no person:", len(labels) - sum(labels))