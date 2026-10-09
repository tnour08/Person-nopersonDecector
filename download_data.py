import os
import requests
from pycocotools.coco import COCO

coco = COCO("annotations/instances_val2017.json")

person_id = coco.getCatIds(catNms=["person"])[0]
person_imgs = coco.getImgIds(catIds=[person_id])

all_imgs = coco.getImgIds()
no_person_imgs = list(set(all_imgs) - set(person_imgs))

n = min(len(person_imgs), len(no_person_imgs))
subset = person_imgs[:n] + no_person_imgs[:n]

os.makedirs("data/images", exist_ok=True)

for img_id in subset:
    info = coco.loadImgs(img_id)[0]
    path = "data/images/" + info["file_name"]

    if os.path.exists(path):
        continue

    r = requests.get(info["coco_url"])
    f = open(path, "wb")
    f.write(r.content)
    f.close()

    print("saved", info["file_name"])
