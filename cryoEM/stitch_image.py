"""
stitch the split images and annotations.
"""

import os
import numpy as np
import cv2
import csv
import argparse
from PIL import Image
from util.box_utils import nms
from coord_io import read_star_file, read_csv_file, read_eman_boxfile



def get_args_parser():
    parser = argparse.ArgumentParser('Stitch images and responding annotations.', add_help=False)

    parser.add_argument('--split_num', default=2, type=int)
    parser.add_argument('--ext', default='.png', type=str)
    parser.add_argument('--gap', default=120, type=int)
    parser.add_argument('--patches_path', type=str)
    parser.add_argument('--annots_path', type=str)
    parser.add_argument('--full_images_path', type=str)
    parser.add_argument('--output_path', type=str)

    return parser


def stitch_image(path, out_path, prefix, gap=100, num_yx=2, ext='.jpg'):
    if not os.path.exists(path):
        raise FileNotFoundError("The full image path is not available.")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if prefix is None:
        raise FileNotFoundError("No images to be stitch. Check your full images path.")

    filenames = sorted(os.listdir(path))
    subimages = [f for f in filenames if f.startswith(prefix) and f.endswith(('jpg','png'))]
    print('prefix:', prefix)
    print("path:", path)
    print("image nums:", len(subimages))
    print('[begin]:')
    if len(subimages) != num_yx * num_yx:
        print("The parameters of the composite image and the requested number cannot be matched!")
        return
    
    i = 0
    list_a = []

    # *step 1: combines the images into columns with a single argument, num_yx, and a number of rows of images for each column
    for subimage in subimages:
        i += 1 # i用于计数
        t = (i - 1) // num_yx # t用于换列
        im = Image.open(os.path.join(path, subimage))
        im_array = np.array(im)
        im_array = np.flipud(im_array)

        if (i - 1) % num_yx == 0:
            list_a.append(im_array)
        else:
            list_a[t] = np.concatenate((list_a[t], im_array[2 * gap:, :]), axis=0)
        print(f"list_a[{t}].shape", list_a[t].shape)

    # *step 2: After composing columns, you need to concatenate them all together
    for j in range(len(list_a) - 1):
        list_a[0] = np.concatenate((list_a[0], (list_a[j + 1])[:, 2 * gap:]), axis=1)
        print(f"list_a[0].shape", list_a[0].shape)

    list_a[0] = np.flipud(list_a[0])
    im_save = Image.fromarray(np.uint8(list_a[0]))
    im_save.save(out_path + prefix + ext)
    print("finished")


def stitch_annotations(annots_path, patches_path, out_path, prefix, gap=100, num_yx=2, ext='.jpg'):
    """
        annots_path: path of the patch annotations
        out_path: path of the output merged annotations
        prefix: the prefix of the image, to find the sub-images annotations
        num_yx: the number of patches for each row and column
        gap: the gap of patches
    """
    if not os.path.exists(annots_path):
        raise FileNotFoundError("The patch image annotations path is not available.")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if prefix is None:
        raise FileNotFoundError("No images to be stitch. Check your full images path.")

    filenames = sorted(os.listdir(annots_path))
    sub_annots = [f for f in filenames if f.startswith(prefix) and f.endswith(('.star','.box','.csv'))]

    print("annots nums:", len(sub_annots))
    if len(sub_annots) != num_yx * num_yx:
        print('The parameters of the composite image and the requested number cannot be matched!')
        return
        # raise ValueError("The parameters of the composite image and the requested number cannot be matched.！")

    nms_boxes, nms_scores = [], []
    for annot in sub_annots:
        top = annot[:-5].split('_')[-1]
        left = annot[:-5].split('_')[-2]
        top, left = int(top), int(left)
        print('top=', top, ' , left=', left)
        if top == 0:
            top = 1928
        elif top == 1928:
            top = 0
        patch_path = patches_path + annot[:-5] + ext
        annot = annots_path + annot
        if os.path.exists(patch_path):
            print('patch_path----', patch_path)
            img = cv2.imread(patch_path, -1)
            img_h, img_w = img.shape[:2]
            print('annot----', annot)
            if os.path.getsize(annot) == 0:
                print(annot, " has no bbox.")
            else:
                if annot.endswith('.star'):
                    boxes = read_star_file(annot, box_width=gap)
                elif annot.endswith('.box'):
                    boxes = read_eman_boxfile(annot)
                elif annot.endswith('.csv'):
                    boxes = read_csv_file(annot, img_h)
                print('len(boxes):', len(boxes))

                for box in boxes:
                    box_xmin = int(box.x) + left
                    box_ymin = int(box.y) + top
                    nms_scores.append(box.c)
                    nms_box = [box_xmin, box_ymin, box_xmin + box.w, box_ymin + box.h]
                    nms_boxes.append(nms_box)
        else:
            print('patch_path not exists: ', patch_path)
            continue

    path = out_path + 'saved_box_files/'
    if not os.path.exists(path):
        os.mkdir(path)

    path = path + prefix + '.star'
    nms_boxes = np.array(nms_boxes)
    print("boxes count before nms processing: ",nms_boxes.shape)
    # NMS deletes overlap boxes

    if len(nms_boxes) > 1:
        picked_boxes, picked_scores = nms(nms_boxes, nms_scores, threshold=0.4)
        print("boxes count after nms processing: ", picked_boxes.shape)
    else:
        picked_boxes = nms_boxes
        picked_scores = nms_scores

    file = open(path, 'w')
    boxwriter = csv.writer(
        file, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
    )
    boxwriter.writerow([])
    boxwriter.writerow(["data_"])
    boxwriter.writerow([])
    boxwriter.writerow(["loop_"])
    boxwriter.writerow(["_rlnCoordinateX #1 "])
    boxwriter.writerow(["_rlnCoordinateY #2 "])
    boxwriter.writerow(["_rlnClassNumber #3 "])
    boxwriter.writerow(["_rlnAnglePsi #4"])
    boxwriter.writerow(["_rlnAutopickFigureOfMerit #5"])

    with open(path, "w") as boxfile:
        # boxwriter = csv.writer(boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE)
        if len(picked_boxes) > 0:
            for index, bb in enumerate(picked_boxes):
                # boxwriter.writerow([bb[0], img_h * num_yx - (num_yx - 1) * gap - bb[1], gap, gap])
                boxwriter.writerow([bb[0]+gap/2, bb[1]+gap/2, 0, 0.0, picked_scores[index]])


def main(args):
    split_num = args.split_num
    gap = args.gap
    patches_path = args.patches_path
    annots_path = args.annots_path
    full_images_path = args.full_images_path
    output_path = args.output_path
    ext = args.ext

    # the names of full images are the prefix of patches.
    prefix_names = [prefix for prefix in os.listdir(full_images_path)
                    if os.path.isfile(os.path.join(full_images_path, prefix))]

    for p in prefix_names:
        prefix, suffix = os.path.splitext(p)
        print(prefix, suffix)  # test   .py
        stitch_image(patches_path, output_path, prefix, gap, split_num, ext)
        stitch_annotations(annots_path, patches_path, output_path, prefix, gap, split_num, ext)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stitch images and corresponding annotations.', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
