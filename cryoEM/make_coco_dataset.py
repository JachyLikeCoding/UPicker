import os
import csv
import cv2
import json
import argparse
import numpy as np
from coord_io import BoundBox, read_star_file, read_eman_boxfile, read_txt_file, read_star_file_topk, read_csv_file
from preprocess import image_read, read_width_height, save_image


def get_args_parser():
    parser = argparse.ArgumentParser('UPicker', add_help=False)
    parser.add_argument('--coco_path', default='./data/EMPIAR10028/', type=str)
    parser.add_argument('--images_path', default='./data/EMPIAR10028/micrographs/processed/', type=str)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--box_width', default=200, type=int)
    parser.add_argument('--bin', default=1, type=int)
    parser.add_argument('--topk', default=50, type=int)
    parser.add_argument('--split', default=0.8, type=float)
    parser.add_argument('--ifsplit', action='store_true', help="If images are split in patches.")
    parser.add_argument('--ifmask', action='store_true', help="If filter autopick boxes with mask.")
    return parser


def make_coco_dataset(root_path, image_path, box_width=200, phase='train', ifsplit=False, ifmask=False, split=0.0, topk=50):
    """Make coco-style dataset. """
    if not os.path.exists(os.path.join(root_path, phase)):
        os.makedirs(os.path.join(root_path, phase))

    dataset = {'categories': [], 'images': [], 'annotations': []}
    classes = ['particle']
    # Establishing the correspondence between class labels and IDs
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # not split dataset
    if not ifsplit:
        # Retrieve image names from the images folder
        indexes = [f for f in os.listdir(image_path) if os.path.isfile(image_path + f)]

        # split the training and testing dataset
        split = int(len(indexes) * split)

        for index in indexes:
            if index.startswith(".") or os.path.isdir(image_path + index):
                indexes.remove(index)
        print(f"There are totally {len(indexes)} micrographs in the {root_path}.")

        if phase == 'train':
            indexes = [line for i, line in enumerate(indexes) if i < split]
        elif phase == 'val':
            indexes = [line for i, line in enumerate(indexes) if i >= split]
        elif phase == 'pretrain':
            indexes = indexes

        print(f"There are totally {len(indexes)} micrographs in the {phase}, {root_path}.")
        for index in indexes:
            if index.endswith('.mrc'):
                image = image_read(f'{image_path}{index}')
                if not np.issubdtype(image.dtype, np.float32):
                    image = image.astype(np.float32)
                save_image(image, f'{root_path}/{phase}/{index[:-4]}.png')
            elif index.endswith(('.jpg', '.png')):
                os.system(f"cp {image_path}/{index} {root_path}/{phase}/")
            else:
                raise Exception(f"{image_path}/{index} is not supported image format.")
    else:
        # for split data
        print('......Make dataset for split data......')
        print(os.path.join(root_path, phase))
        indexes = [f for f in os.listdir(os.path.join(root_path, phase))]
        print(f'there are {len(indexes)} split images.')

    anno_id = 0

    # read image width and height , read bounding boxes
    for k, index in enumerate(indexes):
        if ifsplit:
            width, height = read_width_height(os.path.join(root_path, phase, index))
        else:
            width, height = read_width_height(os.path.join(image_path) + index)

        print("width:", width, "   height:", height)
        dataset['images'].append({'file_name': index,
                                  'id': k,
                                  'width': width,
                                  'height': height})
        print(index)
        boxes = []
        
        # read box file or star file
        if index.endswith(("jpg", "png", "mrc")):
            if args.bin == 1:
                if phase == 'pretrain':
                    # read box file
                    if ifmask:
                        box_file_path = os.path.join(root_path, 'micrographs/AutoPick_filtered/') + index[:-4] + '_autopick.star'
                    else:
                        box_file_path = os.path.join(root_path, 'micrographs/AutoPick/') + index[:-4] + '_autopick.star'
                    if os.path.exists(box_file_path):
                        print(box_file_path)
                        boxes = read_star_file_topk(box_file_path, box_width=box_width, k=topk)

                    # read star file
                    if ifmask:
                        box_file_path = os.path.join(root_path, 'micrographs/AutoPick_filtered/') + index[:-4] + '_autopick.box'
                    else:
                        box_file_path = os.path.join(root_path, 'micrographs/AutoPick/') + index[:-4] + '_autopick.box'
                    if os.path.exists(box_file_path):
                        print(box_file_path)
                        boxes = read_eman_boxfile(box_file_path, topk)
                else:
                    box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.star'
                    print(box_file_path)
                    if os.path.exists(box_file_path):
                        print('[debug] read star file....')
                        boxes = read_star_file(box_file_path, box_width=box_width)
                    box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.box'
                    if os.path.exists(box_file_path):
                        print('[debug] read box file....')
                        boxes = read_eman_boxfile(box_file_path)
                    box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.csv'
                    if os.path.exists(box_file_path):
                        print('[debug] read csv file....')
                        boxes = read_csv_file(box_file_path, image_height=height, box_width=box_width)
                    box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.txt'
                    if os.path.exists(box_file_path):
                        print('[debug] read txt file....')
                        boxes = read_txt_file(box_file_path, box_width=box_width)
            else:
                if phase == 'pretrain':
                    box_file_path = os.path.join(root_path, 'micrographs/downsample{0}/AutoPick/'.format(args.bin)) + index[:-4] + '_autopick.star'
                    print(box_file_path)
                    boxes = read_star_file_topk(box_file_path, box_width=box_width, k=topk)
                else:
                    box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.star'
                    if os.path.exists(box_file_path):
                        boxes = read_star_file(box_file_path, box_width=box_width)
                    box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.box'
                    if os.path.exists(box_file_path):
                        boxes = read_eman_boxfile(box_file_path)
                    box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.csv'
                    if os.path.exists(box_file_path):
                        boxes = read_csv_file(box_file_path, height, box_width=box_width)
                    box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.txt'
                    if os.path.exists(box_file_path):
                        boxes = read_txt_file(box_file_path)

        for box in boxes:
            box_width = int(box.w)
            box_height = int(box.h)
            box_xmin = int(box.x)
            # box_ymin = height - (int(box.y) + box_height)
            box_ymin = int(box.y)

            anno_id += 1
            dataset['annotations'].append({
                'area': box.w * box.h,
                'bbox': [box_xmin, box_ymin, box.w, box.h],
                'category_id': 1,  # particle class
                'id': anno_id,
                'image_id': k,
                'iscrowd': 0,
                'segmentation': []
            })

    # Folder to save the results
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path, 'annotations/instances_{}.json'.format(phase))
    print("json_name:", json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


def make_sub_coco_dataset(root_path, image_path, box_width=200, phase='train', split=0.0, subsize=50):
    """Make coco-style dataset. """
    if not os.path.exists(os.path.join(root_path, phase)):
        os.makedirs(os.path.join(root_path, phase))

    dataset = {'categories': [], 'images': [], 'annotations': []}
    classes = ['particle']
    # Establishing the correspondence between class labels and IDs
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # Retrieve image names from the images folder
    indexes = [f for f in os.listdir(image_path)]
    print(f"There are totally {len(indexes)} micrographs in the {root_path}.")

    # generate subset of subsize micrographs
    indexes = np.random.choice(indexes, size=subsize)
    print(f"Randomly choose {len(indexes)} micrographs.")
    print(indexes)

    # split the training and testing dataset
    split = int(len(indexes) * split)

    for index in indexes:
        if index.startswith(".") or os.path.isdir(image_path + index):
            indexes.remove(index)

    if phase == 'train':
        indexes = [line for i, line in enumerate(indexes) if i < split]
    elif phase == 'val':
        indexes = [line for i, line in enumerate(indexes) if i >= split]
    elif phase == 'pretrain':
        indexes = indexes

    for index in indexes:
        if index.endswith('.mrc'):
            image = image_read(f'{image_path}{index}')
            if not np.issubdtype(image.dtype, np.float32):
                image = image.astype(np.float32)
            save_image(image, f'{root_path}/{phase}/{index[:-4]}.png')
        elif index.endswith(('.jpg', '.png')):
            os.system(f"cp {image_path}/{index} {root_path}/{phase}/")
        else:
            raise Exception(f"{image_path}/{index} is not supported image format.")

    anno_id = 0

    # read image width and height , read bounding boxes
    for k, index in enumerate(indexes):
        width, height = read_width_height(os.path.join(image_path) + index)
        print("width:", width, "   height:", height)
        dataset['images'].append({'file_name': index[:-4] + '.png',
                                  'id': k,
                                  'width': width,
                                  'height': height})
        print(index)
        boxes = []
        # read box file or star file
        if index.endswith(("jpg", "png", "mrc")):
            if args.bin == 1:
                box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.star'
                print(box_file_path)
                if os.path.exists(box_file_path):
                    print('[debug] read star file....')
                    boxes = read_star_file(box_file_path, box_width=box_width)
                box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.box'
                if os.path.exists(box_file_path):
                    print('[debug] read box file....')
                    boxes = read_eman_boxfile(box_file_path)
                box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.csv'
                if os.path.exists(box_file_path):
                    print('[debug] read csv file....')
                    boxes = read_csv_file(box_file_path, height, box_width=box_width)
            else:
                box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.star'
                if os.path.exists(box_file_path):
                    boxes = read_star_file(box_file_path, box_width=box_width)
                box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.box'
                if os.path.exists(box_file_path):
                    boxes = read_eman_boxfile(box_file_path)
                box_file_path = os.path.join(root_path, 'annots/downsample{0}/'.format(args.bin)) + index[:-4] + '.csv'
                if os.path.exists(box_file_path):
                    boxes = read_csv_file(box_file_path, height, box_width=box_width)

        for box in boxes:
            box_width = int(box.w)
            box_height = int(box.h)
            box_xmin = int(box.x)
            # box_ymin = height - (int(box.y) + box_height)
            box_ymin = int(box.y)

            anno_id += 1
            dataset['annotations'].append({
                'area': box.w * box.h,
                'bbox': [box_xmin, box_ymin, box.w, box.h],
                'category_id': 1,  # particle class
                'id': anno_id,
                'image_id': k,
                'iscrowd': 0,
                'segmentation': []
            })

    # Folder to save the results
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path, 'annotations/instances_{}.json'.format(phase))
    print("json_name:", json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


def analysis_json(annFile):
    from pycocotools.coco import COCO
    coco=COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds()) 
    cat_nms=[cat['name'] for cat in cats] 

    print("{:<15} {:<5}     {:<10}".format('classname', 'imgnum', 'bboxnum'))
    print('---------------------------------')
    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=[cat_name])
        imgId = coco.getImgIds(catIds=catId)
        annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))
        print('Average particle num per image:', len(annId)/len(imgId))



def make_coco_result(image_path, result_path, box_width=200):
    """Make coco-style dataset of prediction results. """
    dataset = {'categories': [], 'images': [], 'annotations': []}
    classes = ['particle']
    # Establishing the correspondence between class labels and IDs
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # Retrieve image names from the images folder
    indexes = [f for f in os.listdir(result_path) if os.path.isfile(result_path + f)]

    for index in indexes:
        if index.startswith(".") or os.path.isdir(result_path + index):
            indexes.remove(index)
    print(f"There are totally {len(indexes)} prediction result file in the {result_path}.")

    anno_id = 0

    # read image width and height , read bounding boxes
    for k, index in enumerate(indexes):
        print('index: ', index)
        width, height = read_width_height(os.path.join(image_path) + index[:-5] + '.png')
        print("width:", width, "   height:", height)
        dataset['images'].append({'file_name': index[:-5] + '.png',
                                  'id': k,
                                  'width': width,
                                  'height': height})
        print(index)
        boxes = []
        # read box file or star file
    
        box_file_path = os.path.join(result_path, index)
        print(box_file_path)
        
        if os.path.exists(box_file_path):
            print('[debug] read star file....')
            boxes = read_star_file(box_file_path, box_width=box_width)
        
        for box in boxes:
            box_width = int(box.w)
            box_height = int(box.h)
            box_xmin = int(box.x)
            # box_ymin = height - (int(box.y) + box_height)
            box_ymin = int(box.y)

            anno_id += 1
            dataset['annotations'].append({
                'area': box.w * box.h,
                'bbox': [box_xmin, box_ymin, box.w, box.h],
                'category_id': 1,  # particle class
                'id': anno_id,
                'image_id': k,
                'iscrowd': 0,
                'segmentation': []
            })

    json_name = os.path.join(result_path, 'result.json')
    print("json_name:", json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
        
         

def main(args):
    print(args)
    make_coco_dataset(args.coco_path, args.images_path, box_width=args.box_width, phase=args.phase, ifsplit=args.ifsplit, ifmask=args.ifmask, split=args.split, topk=args.topk)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cryo-coco dataset preperation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

# python make_coco_dataset.py