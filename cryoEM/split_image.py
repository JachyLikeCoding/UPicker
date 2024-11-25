"""
Raw image training data were cropped to generate patches of fixed size.
"""
import cv2
import os
import csv
import numpy as np
import glob
import argparse
from coord_io import read_star_file, write_star_file, read_eman_boxfile, read_csv_file
from preprocess import image_read, image_write, save_image


def get_args_parser():
    parser = argparse.ArgumentParser('Split images and responding annotations.', add_help=False)

    parser.add_argument('--split_num', default=2, type=int)
    parser.add_argument('--dirsrc', default='data/EMPIAR10096', type=str)
    parser.add_argument('--dirdst', default='data/EMPIAR10096/split', type=str)
    parser.add_argument('--ext', default='.png', type=str, choices=['.png', '.jpg', '.mrc'])
    parser.add_argument('--gap', default=100, type=int)
    parser.add_argument('--wo_preprocessed', action='store_true', help="If images have been preprocessed." )
    parser.add_argument('--iou_thresh', default=0.4, type=float)
    parser.add_argument('--ifannots', action='store_false', help="If images have annotations." )
    parser.add_argument('--phase', default='train', type=str)
    return parser


def iou(BBGT, imgRect):
    """
    Calculate the ratio between the intersection of each BBGT and the rectangular area of the image block and the area of the BBGT itself. 
    The ratio ranges from 0 to 1
    input: BBGT: n bboxes, n*4, [xmin,ymin,xmax,ymax], type: np.array
          imgRect: 裁剪的图像块在原图上的位置, [xmin,ymin,xmax,ymax], type:np.array
    return: The iou of each box and image patch (not the real iou), returns the size n, type: np.array
    """
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])
    wh = np.maximum(right_bottom - left_top, 0)
    inter_area = wh[:, 0] * wh[:, 1]
    iou = inter_area / ((BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]))
    return iou


def split(imgname, dirsrc, dirdst, split_num, gap, iou_thresh, ext, phase, wo_preprocess=False):
    """
    split images with annotation files.
    imgname:   Name of the image to crop (with extension)
    dirsrc:    For cutting image save directory on a directory, the default image and labeling files in a folder, 
            image under the images, labeled under labelTxt, annotations file format for each row a gt, format for xmin, ymin, xmax, ymax, class
    dirdst:    The previous directory where the cropped image is saved. There are images,labelTxt two directories to save the cut images or txt files
               The format of the saved image and txt file name is 'oriname_min_ymin.png(.txt),(xmin,ymin)' Is the upper-left coordinate of the cropped image on the original image,
    subsize:   The size of the cropped image, square by default
    gap:       The width at which images in adjacent rows or columns overlap is set to the width of the Bbox by default
    iou_thresh:BBGT less than this threshold will not be saved in the txt of the corresponding image 
                (if the image is too marginal or has no intersection with the image).
    ext:       The format in which the image is saved; the default is png.
    """
    if phase == 'train':
        path = os.path.join(os.path.join(dirsrc, 'train/'), imgname)
    elif phase == 'val':
        path = os.path.join(os.path.join(dirsrc, 'val/'), imgname)
    elif phase == 'pretrain':
        if wo_preprocess:
            path = os.path.join(os.path.join(dirsrc, 'micrographs/'), imgname)
        else:
            path = os.path.join(os.path.join(dirsrc, 'micrographs/processed/'), imgname)
    print(path)

    img = image_read(path)
    img_h, img_w = img.shape[:2]
    subsize_h, subsize_w = img_h // int(split_num) + gap, img_w // int(split_num) + gap
    print('subsize_h = ', subsize_h, ' ,subsize_w = ', subsize_w)
    BBGT = []

    if phase == 'pretrain':
        box_file_path = os.path.join(dirsrc, 'micrographs/AutoPick_filtered/') + imgname[:-4] + '_autopick.star'
        print(box_file_path)
        if os.path.exists(box_file_path):
            boxes = read_star_file(box_file_path, box_width=gap)

    else:
        # read box file
        box_file_path = os.path.join(dirsrc, 'annots/') + imgname[:-4] + '.box'
        if os.path.exists(box_file_path):
            boxes = read_eman_boxfile(box_file_path)
        # read star file
        box_file_path = os.path.join(dirsrc, 'annots/') + imgname[:-4] + '.star'
        if os.path.exists(box_file_path):
            boxes = read_star_file(box_file_path, box_width=gap)
        box_file_path = os.path.join(dirsrc, 'annots/') + imgname[:-4] + '.csv'
        if os.path.exists(box_file_path):
            boxes = read_csv_file(box_file_path, img_h, box_width=gap)

    for box in boxes:
        box_width = int(box.w)
        box_height = int(box.h)
        box_xmin = int(box.x)
        box_ymin = img_h - (int(box.y) + box_height)
        box_xmax = box_xmin + box_width
        box_ymax = box_ymin + box_height
        BBGT.append([box_xmin, box_ymin, box_xmax, box_ymax])
        # BBGT.append([box_xmin, img_h - (box_ymin + box.h), box.w, box.h])  # box.x, box.y, box.w, box.h
    BBGT = np.array(BBGT)

    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize_h >= img_h:
            reachbottom = True
            top = max(img_h - subsize_h, 0)
        while not reachright:
            if left + subsize_w >= img_w:
                reachright = True
                left = max(img_w - subsize_w, 0)
            imgsplit = img[top:min(top + subsize_h, img_h), left:min(left + subsize_w, img_w)]
            if imgsplit.shape[:2] != (subsize_h, subsize_w):
                template = np.zeros((subsize_h, subsize_w, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template

            imgsplit = np.flip(imgsplit, 0)


            if phase == 'pretrain':
                print('Save path: ', os.path.join(os.path.join(dirdst, 'micrographs'),
                                     imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext))
                save_image(imgsplit, os.path.join(os.path.join(dirdst, 'micrographs'),
                                     imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), mi=None, ma=None)
                save_image(imgsplit, os.path.join(os.path.join(dirdst, phase),
                                     imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), mi=None, ma=None)
            else:
                # image_write(os.path.join(os.path.join(dirdst, phase),
                #                      imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), imgsplit)
                print('Save path: ', os.path.join(os.path.join(dirdst, 'micrographs'),
                                     imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext))
                save_image(imgsplit, os.path.join(os.path.join(dirdst, phase),
                                     imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), mi=None, ma=None)

            imgrect = np.array([left, top, left + subsize_w, top + subsize_h]).astype('float32')
            ious = iou(BBGT[:, :4].astype('float32'), imgrect)
            BBpatch = BBGT[ious > iou_thresh]
            print("bbox number: ", len(BBpatch))

            # if phase == 'pretrain':
            #     path = os.path.join(os.path.join(dirdst, 'micrographs/AutoPick'),
            #                     imgname.split('.')[0] + '_' + str(left) + '_' + str(img_h - subsize_h - top) + '_autopick.box')
            # else:
            #     path = os.path.join(os.path.join(dirdst, 'annots'),
            #                     imgname.split('.')[0] + '_' + str(left) + '_' + str(img_h - subsize_h - top) + '.box')
            
            if phase == 'pretrain':
                path = os.path.join(os.path.join(dirdst, 'micrographs/AutoPick'),
                                imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + '_autopick.box')
                print('save annotation: ', path)
            else:
                path = os.path.join(os.path.join(dirdst, 'annots'),
                                imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + '.box')
                print('save annotation: ', path)


            with open(path, "w") as boxfile:
                boxwriter = csv.writer(
                    boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
                )
                for bb in BBpatch:  # [box_xmin, box_ymin, box_xmax, box_ymax]
                    boxheight = bb[3] - bb[1]
                    boxwidth = bb[2] - bb[0]
                    xmin = int(bb[0]) - left
                    ymin = subsize_h - (int(bb[1]) - top) - boxheight
                    xmax = int(bb[2]) - left
                    ymax = subsize_h - (int(bb[3]) - top)
                    # [box.x, box.y, box.w, box.h], box.x, box,y = lower left corner
                    # boxwriter.writerow([xmin, subsize_h - (ymin + boxheight), boxwidth, boxheight])
                    boxwriter.writerow([xmin, ymin, boxwidth, boxheight])
            left += subsize_w - gap
        top += subsize_h - gap


def split_autopick_annots(imgname, dirsrc, dirdst, split_num, gap, iou_thresh, ext, wo_preprocessed):
    if wo_preprocessed:
        path = os.path.join(os.path.join(dirsrc, 'micrographs/'), imgname)
    else:
        path = os.path.join(os.path.join(dirsrc, 'micrographs/processed/'), imgname)
    print(path)

    img = image_read(path)
    img_h, img_w = img.shape[:2]
    subsize_h, subsize_w = img_h // int(split_num) + gap, img_w // int(split_num) + gap
    BBGT = []

    # read box file
    box_file_path = os.path.join(dirsrc, 'micrographs/AutoPick/') + imgname[:-4] + '.box'

    if os.path.exists(box_file_path):
        print('box_file_path: ', box_file_path)
        boxes = read_eman_boxfile(box_file_path)

    # read star file
    box_file_path = os.path.join(dirsrc, 'micrographs/AutoPick/') + imgname[:-4] + '_autopick.star'

    if os.path.exists(box_file_path):
        print('box_file_path: ', box_file_path)
        boxes = read_star_file(box_file_path, box_width=gap)
    print(len(boxes))

    for box in boxes:
        box_width = int(box.w)
        box_height = int(box.h)
        box_xmin = int(box.x)
        box_ymin = img_h - (int(box.y) + box_height)
        box_xmax = box_xmin + box_width
        box_ymax = box_ymin + box_height
        BBGT.append([box_xmin, box_ymin, box_xmax, box_ymax])
        # BBGT.append([box_xmin, img_h - (box_ymin + box.h), box.w, box.h])  # box.x, box.y, box.w, box.h
    BBGT = np.array(BBGT)

    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize_h >= img_h:
            reachbottom = True
            top = max(img_h - subsize_h, 0)
        while not reachright:
            if left + subsize_w >= img_w:
                reachright = True
                left = max(img_w - subsize_w, 0)
            imgsplit = img[top:min(top + subsize_h, img_h), left:min(left + subsize_w, img_w)]
            if imgsplit.shape[:2] != (subsize_h, subsize_w):
                template = np.zeros((subsize_h, subsize_w, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template

            print(imgsplit)

            if not np.issubdtype(imgsplit.dtype, np.float32):
                imgsplit = imgsplit.astype(np.float32)

            mean = np.mean(imgsplit)
            sd = np.std(imgsplit)

            imgsplit = (imgsplit - mean) / sd
            imgsplit[imgsplit > 3] = 3
            imgsplit[imgsplit < -3] = -3

            # image_write(os.path.join(os.path.join(dirdst, 'micrographs'),
            #                          imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + '.png'), imgsplit)

            imgrect = np.array([left, top, left + subsize_w, top + subsize_h]).astype('float32')
            ious = iou(BBGT[:, :4].astype('float32'), imgrect)
            BBpatch = BBGT[ious > iou_thresh]
            print("bbox number: ", len(BBpatch))

            path = os.path.join(os.path.join(dirdst, 'micrographs/AutoPick/'),
                                imgname.split('.')[0] + '_' + str(left) + '_' + str(img_h - subsize_h - top) + '.box')

            with open(path, "w") as boxfile:
                boxwriter = csv.writer(
                    boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
                )
                for bb in BBpatch:  # [box_xmin, box_ymin, box_xmax, box_ymax]
                    boxheight = bb[3] - bb[1]
                    boxwidth = bb[2] - bb[0]
                    xmin = int(bb[0]) - left
                    ymin = int(bb[1]) - top
                    xmax = int(bb[2]) - left
                    ymax = int(bb[3]) - top
                    # [box.x, box.y, box.w, box.h], box.x, box,y = lower left corner
                    boxwriter.writerow([xmin, subsize_h - (ymin + boxheight), boxwidth, boxheight])
            left += subsize_w - gap
        top += subsize_h - gap


def split_only_images(imgname, dirsrc, dirdst, split_num=2, gap=100, ext='.png'):
    img = cv2.imread(os.path.join(dirsrc, imgname), -1)
    img_h, img_w = img.shape[:2]
    print(imgname, img_w, img_h)
    subsize_h, subsize_w = img_h // int(split_num) + gap, img_w // int(split_num) + gap

    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize_h >= img_h:
            reachbottom = True
            top = max(img_h - subsize_h, 0)
        while not reachright:
            if left + subsize_w >= img_w:
                reachright = True
                left = max(img_w - subsize_w, 0)
            imgsplit = img[top:min(top + subsize_h, img_h), left:min(left + subsize_w, img_w)]
            if imgsplit.shape[:2] != (subsize_h, subsize_w):
                template = np.zeros((subsize_h, subsize_w, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template
            name = os.path.splitext(imgname)[0]
            if name.endswith('_mask'):
                cv2.imwrite(os.path.join(dirdst, name[:-5] + '_' + str(left) + '_' + str(top) + '_mask'+ ext), imgsplit)
            else:
                cv2.imwrite(os.path.join(dirdst, name + '_' + str(left) + '_' + str(top) + ext), imgsplit)
            left += subsize_w - gap
        top += subsize_h - gap


def split_train_val_data(dirsrc, dirdst, split_num=2, gap=100, iou_thresh=0.4, ext='.png', phase=None):
    """
    split images with annotation files.
    """
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'micrographs')):
        os.mkdir(os.path.join(dirdst, 'micrographs'))
    if not os.path.exists(os.path.join(dirdst, 'annots')):
        os.mkdir(os.path.join(dirdst, 'annots'))
    if not os.path.exists(os.path.join(dirdst, 'val')):
        os.mkdir(os.path.join(dirdst, 'val'))
    if not os.path.exists(os.path.join(dirdst, 'train')):
        os.mkdir(os.path.join(dirdst, 'train'))


    imglist = glob.glob(f'{dirsrc}/{phase}/*{ext}')
    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist]

    for imgname in imgnameList:
        if imgname.endswith(("png","jpg","jpeg")):
            split(imgname, dirsrc, dirdst, split_num, gap, iou_thresh, ext, phase)


def split_pretrain_data(dirsrc, dirdst, split_num=2, gap=100, iou_thresh=0.4, ext='.png', wo_preprocessed=False, phase=None):
    """
    split images with annotation files.
    """
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'micrographs')):
        os.mkdir(os.path.join(dirdst, 'micrographs'))
    if not os.path.exists(os.path.join(dirdst, 'annots')):
        os.mkdir(os.path.join(dirdst, 'annots'))
    if not os.path.exists(os.path.join(dirdst, 'micrographs/AutoPick')):
        os.mkdir(os.path.join(dirdst, 'micrographs/AutoPick'))
        
    if wo_preprocessed:
        imglist = glob.glob(f'{dirsrc}/micrographs/*{ext}')
    else:
        imglist = glob.glob(f'{dirsrc}/micrographs/processed/*{ext}')

    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist]
    print(imgnameList)
    for imgname in imgnameList:
        if imgname.endswith(("png","jpg","jpeg","mrc")):
            split(imgname, dirsrc, dirdst, split_num, gap, iou_thresh, ext, phase, wo_preprocessed)


def split_test_images(dirsrc, dirdst, split_num=2, gap=100, iou_thresh=0.5, ext='.jpg'):
    """
    split test images without annotation files.
    """
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)

    imglist = glob.glob(f'{dirsrc}/*{ext}')
    
    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist]
    for imgname in imgnameList:
        if imgname.endswith(("png","jpg","jpeg")):
            split_only_images(imgname, dirsrc, dirdst, split_num, gap, ext)


def main(args):
    # split train and val images
    split_num = args.split_num
    gap = args.gap
    ext = args.ext
    iou_thresh = args.iou_thresh
    wo_preprocessed = args.wo_preprocessed
    dirsrc = args.dirsrc
    dirdst = args.dirdst
    phase = args.phase

    if args.ifannots:
        if phase == 'pretrain':
            print('[debug] split pretrain images')
            split_pretrain_data(dirsrc, dirdst, split_num, gap, iou_thresh, ext, wo_preprocessed, phase)
        elif phase in (('train', 'val')):
            print('[debug] split train and val images')
            split_train_val_data(dirsrc, dirdst, split_num, gap, iou_thresh, ext, phase)
    else:
        print('[debug] split test images')
        split_test_images(dirsrc, dirdst, split_num, gap, ext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split images and corresponding annotations which containing too much particles.', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
