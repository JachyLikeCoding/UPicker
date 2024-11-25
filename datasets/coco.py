# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.misc import get_local_rank, get_local_size

from .torchvision_datasets import CocoDetection as TvCocoDetection


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, no_cats=False, filter_pct=-1, seed=42):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.no_cats = no_cats
        self.filter_pct = filter_pct
        self.seed = seed
        
    def filter_objects(self, ann_ids):
        num_objects = len(ann_ids)
        num_keep = int(num_objects * self.filter_pct)
        rng = np.random.default_rng(self.seed)
        # print('\n[DEBUG] >>filter percent, num_keep = ', num_keep)

        # 检查 ann_ids 的长度是否足够
        if len(ann_ids) < num_keep:
            print(f"Warning: ann_ids length ({len(ann_ids)}) is less than num_keep ({num_keep}). Adjusting num_keep to {len(ann_ids)}.")
            num_keep = len(ann_ids)
        if num_keep > 0:
            selected_objects = rng.choice(ann_ids, size=num_keep, replace=False)
            return selected_objects
        else:
            return ann_ids

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        ann_ids = [ann['id'] for ann in target]
        filtered_ann_ids = self.filter_objects(ann_ids)
        filtered_target = [ann for ann in target if ann['id'] in filtered_ann_ids] 
        
        # print('target:', len(ann_ids), '    filtered_target:', len(filtered_ann_ids))
        
        image_id = self.ids[idx]
        filtered_target = {'image_id': image_id, 'annotations': filtered_target}
        img, filtered_target = self.prepare(img, filtered_target)
        
       
        if self._transforms is not None:
            img, filtered_target = self._transforms(img, filtered_target)
        
        if self.no_cats:
            print('[debug] no_cats = ', self.no_cats)
            filtered_target['labels'][:] = 1
        
        return img, filtered_target
    

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes, keep = preprocess_xywh_boxes(boxes, h, w)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def preprocess_xywh_boxes(boxes, h, w):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    return boxes, keep


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    normalize_single = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,], [0.229,])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



def make_coco_transforms_single(image_set):
    normalize_single = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,], [0.229,])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize_single,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([1024], max_size=1333),
            normalize_single,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    mode = 'instances'

    if args.dataset_file == "coco":
        root = Path(args.coco_path)
        print('build dataset, root = ',root)
        assert root.exists(), f'provided path {root} does not exist'
        PATHS = {
            "train": (root / "train", root / "annotations" / f'{mode}_train.json'),
            "val": (root / "val", root / "annotations" / f'{mode}_val.json'),
        }
        img_folder, ann_file = PATHS[image_set]

    if 'pretrain' in args.dataset:
        root = Path(os.path.join(os.path.join(args.data_root, args.dataset_file)))
        print('build dataset, root = ',root)
        assert root.exists(), f'provided path {root} does not exist'
        if args.filter_num < 0:
            PATHS = {
                "train": (root / "pretrain", None),
                "val": (root / "val", None),
            }
        else:
            PATHS = {
                "train": (root / f"pretrain_{args.filter_num}", None),
                "val": (root / "val", None),
            }
        img_folder, ann_file = PATHS[image_set]
        
    elif 'EMPIAR' in args.dataset_file:
        root = Path(os.path.join(os.path.join(args.data_root, args.dataset_file)))
        print('build dataset, root = ',root)
        assert root.exists(), f'provided path {root} does not exist'
        if args.filter_num < 0: # full dataset
            PATHS = {
                "train": (root / "train", root / "annotations" / f'{mode}_train.json'),
                "val": (root / "val", root / "annotations" / f'{mode}_val.json'),
            }
        else: # filter_num > 0 means select filter_num micrographs from the full training dataset
            PATHS = {
                "train": (root / f"train_{args.filter_num}", root / "annotations" / f'{mode}_train_{args.filter_num}.json'),
                "val": (root / "val", root / "annotations" / f'{mode}_val.json'),
            }
        # select
        if args.augment == True:
            if args.filter_num < 0:
                PATHS = {
                    "train": (root / f"train_augment", root / "annotations" / f'{mode}_train_augment.json'),
                    "val": (root / "val", root / "annotations" / f'{mode}_val.json'),
                }
            else:
                PATHS = {
                    "train": (root / f"train_augment_{args.filter_num}", root / "annotations" / f'{mode}_train_augment_{args.filter_num}.json'),
                    "val": (root / "val", root / "annotations" / f'{mode}_val.json'),
                }
        img_folder, ann_file = PATHS[image_set]
    else:
        root = Path(os.path.join(os.path.join(args.data_root, args.dataset_file)))
        print('build dataset, root = ',root)
        assert root.exists(), f'provided path {root} does not exist'
        PATHS = {
            "train": (root / "train", None),
            "val": (root / "val", None),
        }
        # raise ValueError(f"unknown {args.dataset_file}")
        img_folder, ann_file = PATHS[image_set]

    # add no_cats and filter_pct
    no_cats = False

    if 'pretrain' in args.dataset:
        no_cats = True

    filter_pct = -1
    if image_set == 'train' and args.filter_pct > 0:
        filter_pct = args.filter_pct
    print('no_cats: ', no_cats)
    print(' > ann_file: ', ann_file)

    dataset = CocoDetection(img_folder, ann_file, 
                            transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),
                            no_cats=no_cats, filter_pct=filter_pct, seed=args.seed)
    return dataset
