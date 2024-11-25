# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
pre-training dataset which implements random query patch detection.
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
import datasets.transforms as T
from torchvision.transforms import transforms
from PIL import ImageFilter
import random
import cv2
from util.box_ops import crop_bbox
from cryoEM.coord_io import read_star_file_topk, read_eman_boxfile


def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh


class SelfDet(Dataset):
    """
    SelfDet is a dataset class which implements LoG query patch detection.
    It crops patches based on LoG filter as queries from the given image with the corresponding bounding box.
    The format of the bounding box is same to COCO.
    """

    def __init__(self, root, detection_transform, query_transform, cache_dir=None, max_prop=50, box_width=200, strategy='log'):
        super(SelfDet, self).__init__()
        self.strategy = strategy
        self.cache_dir = cache_dir
        self.query_transform = query_transform
        self.root = root
        self.max_prop = max_prop
        self.detection_transform = detection_transform
        self.files = []
        self.box_width = box_width
        self.dist2 = -np.log(np.arange(1, 301) / 301) / 10
        max_prob = (-np.log(1 / 1001)) ** 4

        for (troot, _, files) in os.walk(root, followlinks=True):
            for f in files:
                if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png','mrc']:
                    path = os.path.join(troot, f)
                    self.files.append(path)
                else:
                    continue
        print(f'num of files:{len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_path = self.files[item]
        if img_path.endswith('.mrc'):
            from cryoEM.preprocess import read_mrc, read_width_height
            img = read_mrc(img_path)
            w, h = read_width_height(img_path)
        else:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
        # print('[SelfDet] ---- img path: ', img_path)

        if self.strategy == 'topk': # selective search has randomness and without caching, the results are better.
            boxes = selective_search(img, h, w, res_size=128)
            boxes = boxes[:self.max_prop]
        elif self.strategy == 'mc':
            boxes = self.load_from_cache(item, img, h, w)
            boxes_indicators = np.where(np.random.binomial(1, p=self.dist2[:len(boxes)]))[0]
            boxes = boxes[boxes_indicators]
        elif self.strategy == "random":
            # boxes = self.load_from_cache(random.choice(range(self.files)), None, None, None) # relies on cache for now
            # boxes = boxes[:self.max_prop]
            boxes = random_crop_boxes(img, h, w, diam_min=self.box_width * 0.8, diam_max=self.box_width *1.2, nums_patches=self.max_prop)
        elif self.strategy == 'log':
            boxes = log_autopick_boxes(img_path, output_path='AutoPick/', box_width=self.box_width, topk=self.max_prop) # read from the LoG autopick star file
        elif self.strategy == 'edgebox':
            boxes = get_edge_boxes(img_path, self.max_prop)
        else:
            raise ValueError("No such strategy")

        # # uncomment for debug: visualize image and patches
        # from util.plot_utils import plot_results
        # from matplotlib import pyplot as plt
        # plt.figure()
        # # boxes = selective_search(img, h, w, res_size=128)
        # plot_results(img_path, np.array(img), np.zeros(30), boxes[:30], plt.gca(), norm=False)
        # # plt.show()

        if len(boxes) < 2:
            return self.__getitem__(random.randint(0, len(self.files) - 1))

        patches = [img.crop([b[0], b[1], b[2], b[3]]) for b in boxes]
        # print('patches shape:', np.array(patches).shape)
        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        target['patches'] = torch.stack([self.query_transform(p) for p in patches], dim=0)
        target['boxes'] = torch.tensor(boxes)
        target['iscrowd'] = torch.zeros(len(target['boxes']))
        target['area'] = target['boxes'][..., 2] * target['boxes'][..., 3]
        target['labels'] = torch.ones(len(target['boxes'])).long()
        target['image_id'] = torch.tensor(item)
        img, target = self.detection_transform(img, target)
        if len(target['boxes']) < 2:
            return self.__getitem__(random.randint(0, len(self.files) - 1))
        # print('img shape:',  np.array(img).shape)
        # print('target size:',  np.array(target).shape)
        return img, target

    def load_from_cache(self, item, img, h, w):
        fn = self.files[item].split('/')[-1].split('.')[0] + '.npy'
        fp = os.path.join(self.cache_dir, fn)
        try:
            with open(fp, 'rb') as f:
                boxes = np.load(f)
        except FileNotFoundError:
            boxes = selective_search(img, h, w, res_size=None)
            with open(fp, 'wb') as f:
                np.save(f, boxes)
        return boxes


def selective_search(img, h, w, res_size=128):
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('float32')
    print('Total Number of Region Proposals: {}'.format(len(boxes)))
    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes



def get_edge_boxes(img_path, topk, algo_edgedet = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz') if os.path.exists('model.yml.gz') else None):
    img = cv2.imread(img_path)
    edges = algo_edgedet.detectEdges(img.astype(np.float32) / 255.0)
    orimap = algo_edgedet.computeOrientation(edges)
    edges = algo_edgedet.edgesNms(edges, orimap)
    algo_edgeboxes = cv2.ximgproc.createEdgeBoxes(alpha=0.6, beta=0.4, maxBoxes=100, minBoxArea=20000)
    algo_edgeboxes.setMaxBoxes(topk)
    boxes_xywh, scores = algo_edgeboxes.getBoundingBoxes(edges, orimap)

    if scores is None:
        boxes_xywh, scores = np.array([[0, 0.0, img.shape[1], img.shape[0]]]), np.ones((1, ))
    results = []
    for box in boxes_xywh:
        results.append((box[0], box[1], box[0]+box[2], box[1]+box[3]))
    results = np.array(results, dtype=np.float32)
    # print(f'get {len(results)} edge boxes')
    return results



def log_autopick_boxes(img_path, output_path='AutoPick/', box_width=200, topk=50):
    # print('[log_autopick_boxes] topk = ', topk)
    cur_path = os.getcwd()
    img_name = img_path.split('/')[-1]
    img_dir = os.path.dirname(img_path)
    os.chdir(img_dir)

    star_path = os.path.join(cur_path+'/'+os.path.split(img_dir)[0] + '/micrographs/' + output_path, img_name[:-4] + '_autopick.star')
    if os.path.exists(star_path):
        boxes = read_star_file_topk(star_path, box_width, topk)
        # print('debug: top k = ', topk)

    star_path = os.path.join(cur_path+'/'+os.path.split(img_dir)[0] + '/micrographs/' + output_path, img_name[:-4] + '_autopick.box')
    if os.path.exists(star_path): 
        boxes = read_eman_boxfile(star_path, topk)

    # print(len(boxes))
    os.chdir(cur_path)
    results = []
    for box in boxes:
        results.append((box.x, box.y, box.x+box.w, box.y+box.h))
    boxes = np.array(results, dtype=np.float32)

    return boxes


def random_crop_boxes(img, h, w, diam_min=200, diam_max=250, nums_patches=50):
    boxes = []
    for i in range(nums_patches):
        patchsize = np.random.randint(diam_min, diam_max)
        box_x = np.random.randint(0, h-patchsize)
        box_y = np.random.randint(0, w-patchsize)
        boxes.append((box_x, box_y, box_x + patchsize, box_y + patchsize))
    return boxes



def make_self_det_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # The image of ImageNet is relatively small.
    scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

    if image_set == 'train' or image_set == 'pretrain':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=600),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=600),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=600),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_query_transforms(image_set):
    if image_set == 'train' or image_set == 'pretrain':
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if image_set == 'val':
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    raise ValueError(f'unknown {image_set}')


def build_selfdet(image_set, args, p):
    return SelfDet(p, detection_transform=make_self_det_transforms(image_set), 
                    query_transform=get_query_transforms(image_set),
                    cache_dir=args.cache_path,
                    max_prop=args.max_prop, 
                    box_width=args.box_width,
                    strategy=args.strategy)


# just for debugging
# if __name__ == '__main__':
#     img_path = 'data/EMPIAR10028/micrographs/'
#     for image in os.listdir(img_path):
#         if image.endswith('.mrc'):
#             log_autopick_boxes(os.path.join(img_path, image))
