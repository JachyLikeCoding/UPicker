import os, json
import random
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import numpy as np
from pycocotools.coco import COCO
from coord_io import BoundBox, read_star_file
from preprocess import image_read, read_width_height, save_image


transform_0 = A.Compose([
                    A.RandomSizedBBoxSafeCrop(width=1024, height=1024, erosion_rate=0.2),
                    A.HorizontalFlip(always_apply=True),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

transform_1 = A.Compose([
                    A.VerticalFlip(always_apply=True),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

transform_2 = A.Compose([
                    A.RandomSizedBBoxSafeCrop(width=1024, height=1024, erosion_rate=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

transforms = [transform_0, transform_1, transform_2]


def make_augment_coco_dataset(root_path, image_path, num):
    dataset = {'categories': [], 'images': [], 'annotations': []}
    classes = ['particle']
    # Establishing the correspondence between class labels and IDs
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    anno_id = 0

    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    catIds = coco.getCatIds(catNms=['particle'])
    imgIds = coco.getImgIds(catIds=catIds)
    print(len(imgIds))

    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        image_name = img['file_name']
        image_width, image_height = img['width'], img['height']
        print(image_name)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=[1], iscrowd=None)
        anns = coco.loadAnns(annIds)
        bboxes = []
        for j in range(len(anns)):
            box = anns[j]['bbox'] # xmin, ymin, width, height
            if box[0]>0 and box[1]>0 and box[0]+box[2]< image_width and box[1]+box[3]< image_height: 
                bboxes.append(anns[j]['bbox'])

        image = cv2.imread(os.path.join(image_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        category_ids = [1]*len(bboxes)

        for t in range(0,3): 
            tmp_bboxes = bboxes
            transform = transforms[t]
            transformed = transform(image=image, bboxes=tmp_bboxes, category_ids=category_ids)
            width, height = transformed['image'].shape[0], transformed['image'].shape[1]
            # print("width:", width, "   height:", height)

            dataset['images'].append({'file_name': f'{t}_'+image_name,
                                    'id': i + t*len(imgIds),
                                    'width': width,
                                    'height': height})
            cv2.imwrite(f'{save_path}{t}_{image_name}',transformed['image'])
            tmp_bboxes = transformed['bboxes']

            for box in tmp_bboxes:
                x_min, y_min, w, h = box
                x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
                # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
                anno_id += 1
                dataset['annotations'].append({
                    'area': w * h,
                    'bbox': [x_min, y_min, w, h],
                    'category_id': 1,  # particle class
                    'id': anno_id,
                    'image_id': i + t*len(imgIds),
                    'iscrowd': 0,
                    'segmentation': []
                })

    # save the results
    json_name = os.path.join(root_path, f'annotations/instances_train_augment_{num}.json')
    print("json_name:", json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    num = 5
    dataset = 'EMPIAR10389'
    image_path = f'data/{dataset}/train_{num}/'
    save_path = f'data/{dataset}/train_augment_{num}/'
    root_path = f'data/{dataset}/'
    annFile = f'data/{dataset}/annotations/instances_train_{num}.json'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    make_augment_coco_dataset(root_path, image_path, num)