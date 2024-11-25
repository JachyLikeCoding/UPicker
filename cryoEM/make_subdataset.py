'''Make coco sub-dataset for training with N images'''
import json
import time
import shutil
from collections import defaultdict
import json
import os
from pathlib import Path


class COCO:
    def __init__(self, annotation_file=None, origin_img_dir=""):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.origin_dir = origin_img_dir
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def build(self, tarDir=None, tarFile='./new_train.json', N=100):
        load_json = {'images': [], 'annotations': [], 'categories': [], 'type': 'instances'}
        # if not Path(tarDir).exists():
        #     Path(tarDir).mkdir()
        os.makedirs(tarDir, exist_ok=True)
        for i in self.imgs:
            if(N == 0):
                break
            tic = time.time()
            img = self.imgs[i]
            load_json['images'].append(img)
            fname = os.path.join(tarDir, img['file_name'])
            anns = self.imgToAnns[img['id']]
            for ann in anns:
                load_json['annotations'].append(ann)
            if not os.path.exists(fname):
                shutil.copy(self.origin_dir+'/'+img['file_name'], tarDir)
            print('copy {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))
            N -= 1
        for i in self.cats:
            load_json['categories'].append(self.cats[i])
        with open(tarFile, 'w+') as f:
            json.dump(load_json, f, indent=4)


def make_subdataset(origin_img_dir, origin_annotation_dir, save_img_dir, save_annotation_dir, subsize=50):
    coco = COCO(origin_annotation_dir, origin_img_dir)
    coco.build(save_img_dir, save_annotation_dir, subsize)


if __name__ == "__main__":
    coco = COCO('data/EMPIAR10028/annotations/instances_train.json',
                origin_img_dir='data/EMPIAR10028/train')               # full data path
    coco.build('data/EMPIAR10028/train_50', 
                'data/EMPIAR10028/annotations/instances_train_50.json', 50)  # subset save path
