"""
Remove the false positive results on ice and carbon region and image edge
"""

import micrograph_cleaner as mce
import cv2
import numpy as np
import os
import argparse
from cryoEM.read_image import image_read
from cryoEM.coord_io import read_eman_boxfile, read_star_file, write_star_file
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('Clean boxes in mask.', add_help=False)
    parser.add_argument('--image_path', default='/media/feng/2480CDB880CD90AA/Upicker_data/empiar10081-all/micrographs', type=str)
    parser.add_argument('--boxsize', default=150, type=int)
    parser.add_argument('--model', default='/home/feng/UPicker/defaultModel.keras', type=str)
    parser.add_argument('--threshold', default=0.1, type=float, help='Clean boxes which is < threshold on mask. 0-bad region, 1-ideal region')
    parser.add_argument('--withmask', action='store_true', help="if has masks already." )
    return parser


def clean_edge_boxes(pre_boxes, h, w):
    box_clean = []
    edge_box_num = 0
    for pre_box in pre_boxes:
        x2,y2,w2,h2 = pre_box.x, pre_box.y, pre_box.w, pre_box.h
        if x2 < 0 or y2 < 0 or x2 + w2 > w or y2 + h2 > h:
            edge_box_num += 1
        else:
            box_clean.append(pre_box)
    
    print(f'There are {edge_box_num} edge boxes')
    return box_clean


def clean_micrograph(image_path, boxsize=200, model_path='/home/feng/UPicker/defaultModel.keras'):
    micrograph = image_read(image_path)

    # By default, the mask predictor will try load the model at
    # "~/.local/share/micrograph_cleaner_em/models/"
    if model_path:
        deepLearningModelFname = model_path
    else:
        deepLearningModelFname = "~/.local/share/micrograph_cleaner_em/models/defaultModel.keras"
     
    with mce.MaskPredictor(boxsize, deepLearningModelFname=deepLearningModelFname, gpus=[0]) as mp:
        mask = mp.predictMask(micrograph)  # by default, mask is float32 numpy array

    return mask


def save_mask(image_path, mask):
    name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.dirname(image_path)
    save_path = save_path + '/mask/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # mask_name_mrc = save_path + name_without_ext + '_mask.mrc'
    mask_name_jpg = save_path + name_without_ext + '_mask.jpg'
    # mask_name_png = save_path + name_without_ext + '_mask.png'
    # mask_name_txt = save_path + name_without_ext + '_mask.txt'

    # Then write the mask as a txt file, mrc file and jpg file
    # np.savetxt(mask_name_txt, mask)

    # with mrcfile.new(mask_name_mrc, overwrite=True) as maskFile:
    #     maskFile.set_data(mask.astype(np.half))  # as float
    # mask[:120, :] = 1
    # mask[-120:, :] = 1
    # mask[:, :120] = 1
    # mask[:, -120:] = 1
    cv2.imwrite(mask_name_jpg, mask * 255)



def display_compare(name, image_path, ext='.jpg'):
    name_without_ext = os.path.splitext(os.path.basename(name))[0]
    mask_name = image_path + "/mask/" + name_without_ext + '_mask' + ext
    print('mask name: ', mask_name)
    image_name = image_path + name_without_ext + ext
    original = image_read(name)
    mask = image_read(mask_name)

    # display and save compare results.
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(original, 'gray')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, 'gray')
    plt.title('mask')
    plt.savefig(f"{image_path}/mask/" + name_without_ext + "_compare" + ext)


def delete_box_in_mask(mask, boxes, scores, threshold=0.1):
    delete_indexes = []
    delete_boxes = []
    if_leave = []
    for i, box in enumerate(boxes):
        # [box_xmin, box_ymin, box_xmax, box_ymax]
        box_center_x = int((box[0]+box[2])*0.5)
        box_center_y = int((box[1]+box[3])*0.5)
        # print(box)
        # print(box_center_x, box_center_y)
        if 0 < box_center_y < mask.shape[0] and 0 < box_center_x < mask.shape[1]:
            if mask[box_center_y][box_center_x] > threshold * 255:
                delete_indexes.append(i)
                if_leave.append(False)
                delete_boxes.append(box)
            else:
                if_leave.append(True)
        else:
            delete_indexes.append(i)
            if_leave.append(False)
            delete_boxes.append(box)
    print(f'--------Delete {len(delete_indexes)} boxs')
    print(len(if_leave))
    boxes_cleaned = [boxes[i] for i in range(len(boxes)) if if_leave[i]]
    saved_scores = [scores[i] for i in range(len(scores)) if if_leave[i]]
    
    return boxes_cleaned, saved_scores, delete_boxes


def filter_boxes(root_path, box_width, threshold=0.2):
    log_path = f'{root_path}/AutoPick'
    mask_path = f'{root_path}/mask'
    output_log_path = f'{root_path}/AutoPick_filtered'
    output_delete_path = f'{root_path}/AutoPick_deleted'

    if not os.path.exists(output_log_path):
        os.makedirs(output_log_path)
    if not os.path.exists(output_delete_path):
        os.makedirs(output_delete_path)

    logs = [f for f in os.listdir(log_path) if f.endswith(('_autopick.star','_autopick.box'))]

    for log in logs:
        print('log: ', log)
        img_name = os.path.splitext(log)[0][:-9]+ '_mask.jpg'
        mask_name = os.path.join(mask_path, img_name)
        mask = image_read(mask_name)
        mask = np.flipud(mask)


        if log.endswith('.box'):
            boxes = read_eman_boxfile(os.path.join(log_path, log))
        else:
            boxes = read_star_file(os.path.join(log_path, log), box_width)

        boxes_cleaned, box_deleted = filter_log_boxes_in_mask(mask, boxes, threshold)

        write_star_file(os.path.join(output_log_path, log), boxes_cleaned)
        write_star_file(os.path.join(output_delete_path, log), box_deleted)



def filter_log_boxes_in_mask(mask, boxes, threshold=0.2):
    delete_indexes = []
    delete_boxes = []
    if_leave = []
    for i, box in enumerate(boxes):
        # [box_xmin, box_ymin, box_xmax, box_ymax]
        box_center_x = int(box.x + box.w * 0.5)
        box_center_y = int(box.y + box.h * 0.5)

        if 0 < box_center_y < mask.shape[0] and 0 < box_center_x < mask.shape[1]:
            if mask[box_center_y][box_center_x] > threshold * 255:
                delete_indexes.append(i)
                if_leave.append(False)
                delete_boxes.append(box)
            else:
                if_leave.append(True)
        else:
            delete_indexes.append(i)
            if_leave.append(False)
            delete_boxes.append(box)

    print(f'--------Delete {len(delete_indexes)} boxs')
    # print(len(if_leave))
    boxes_cleaned = [boxes[i] for i in range(len(boxes)) if if_leave[i]]
    
    return boxes_cleaned, delete_boxes


def main(args):
    boxsize = args.boxsize
    images_path = args.image_path

    if not args.withmask:
        for image in os.listdir(images_path):
            image = os.path.join(images_path, image)
            print('image: ', image)
            
            if image.endswith(".mrc"):
                mask = clean_micrograph(image, boxsize=boxsize)
                save_mask(image, mask)
                print('save mask: ', image)
                # display_compare(image, image_path)

    filter_boxes(images_path, box_width=boxsize)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cleans boxes in mask.', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)