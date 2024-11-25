import os
import sys
import csv
import pandas as pd
import numpy as np
from PIL import Image

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class BoundBox:
    """
    Bounding box of a particle.
    """

    def __init__(self, x, y, w, h, c=None, classes=None):
        """
        creates a bounding box.
        :param x: x coordinate of particle center.
        :param y: y coordinate of particle center.
        :param w: width of box
        :param h: height of box
        :param c: confidence of the box
        :param classes: class of the bounding box object
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c
        self.classes = classes
        self.meta = {}
        self.label = -1
        self.score = -1
        self.info = None

    def get_label(self):
        """

        :return: class with highest probability
        """
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        """

        :return: probability of the class
        """
        self.score = self.classes[self.get_label()]

        return self.score
    
    def get_bbox(self):
        """
        :return: (x, y, w, h) tuple
        """
        return (self.x, self.y, self.w, self.h)
    
    

def write_box(path, boxes, write_star=True):
    """
    Write box or star files.
    :param path: filepath or filename of the box file to write.
    :param boxes: boxes to write
    :param write_star: if true, a star file will be written.
    :return: None
    """
    if write_star:
        path = path[:-3] + 'star'
        write_star_file(path, boxes)
    else:
        write_eman_boxfile(path, boxes)


def write_star_file(path, boxes):
    with open(path, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow([])
        boxwriter.writerow(["data_"])
        boxwriter.writerow([])
        boxwriter.writerow(["loop_"])
        boxwriter.writerow(["_rlnCoordinateX #1 "])
        boxwriter.writerow(["_rlnCoordinateY #2 "])
        boxwriter.writerow(["_rlnClassNumber #3 "])
        boxwriter.writerow(["_rlnAnglePsi #4"])
        boxwriter.writerow(["_rlnScore #5"])
        for box in boxes:
            boxwriter.writerow([box.x + box.w / 2, box.y + box.h / 2, 0, 0.0, 0.0])


def write_eman_boxfile(path, boxes):
    with open(path, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        for box in boxes:
            # box.x, box,y = lower left corner
            boxwriter.writerow([box.x, box.y, box.w, box.h])


def write_cbox_file(path, boxes):
    with open(path, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        for box in boxes:
            est_w = box.meta["boxsize_estimated"][0]
            est_h = box.meta["boxsize_estimated"][1]
            boxwriter.writerow([box.x, box.y, box.w, box.h, est_w, est_h])
            
            
def write_txt_file(path, boxes, bin):
    with open(path, "w") as boxfile:
        for box in boxes:
            boxfile.write(f"{int(box.x//bin)} {int(box.y//bin)}\n")


def get_star_file_header(file_name):
    """
    load the header information of star file.
    :param file_name:
    :return: list of head names, rows that are occupied by the header.
    """
    start_header = False
    header_names = []
    idx = None

    with open(file_name, "r") as read:
        for idx, line in enumerate(read.readlines()):
            if line.startswith("_"):
                if start_header:
                    header_names.append(line.strip().split()[0])
                else:
                    start_header = True
                    header_names.append(line.strip().split()[0])
            elif start_header:
                break
    if not start_header:
        raise IOError(f"No header information found in {file_name}")

    return header_names, idx


def read_eman_boxfile(path, topk=-1):
    """
    Read a box file in EMAN box format.
    :param path: the path of box file
    :return: List of bounding boxes.
    """
    boxes = []
    if os.path.getsize(path) == 0:
        print(path, " has no bbox.")
    else:
        boxreader = np.atleast_2d(np.genfromtxt(path))
        boxes = [BoundBox(x=box[0], y=box[1], w=box[2], h=box[3]) for box in boxreader]

    if topk > 0:
        boxes = boxes[:topk]

    return boxes

# read cryosegnet box file
def read_box_file(path, image_height, box_width=200):
    boxes = []
 
    if os.path.getsize(path) == 0:
        print(path, " has no bbox.")
    else:
        with open(path, 'r') as file:
            lines = file.readlines()[1:]  # Skip header line
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # Skip invalid rows
                x = float(parts[1])
                y = float(parts[2])
                boxes.append(BoundBox(x=x- box_width / 2, y=y- box_width / 2, w=box_width, h=box_width))
    return boxes



def read_txt_file(path, box_width):
    boxreader = np.atleast_2d(np.genfromtxt(path))
    boxes = []
    for box in boxreader:
        bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width)
        boxes.append(bound_box)
    return boxes


def read_star_file(path, box_width, score_thresh=-1.0):
    header_names, skip_indices = get_star_file_header(path)
    boxreader = np.atleast_2d(np.genfromtxt(path, skip_header=skip_indices))
    if boxreader.size == 0 or boxreader.shape[1] < 2:
        print(f"No valid box coordinates found in {path}")
        return []

    boxes = []
    if score_thresh == -1.0:
        for box in boxreader:
            bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width, c=box[-1])
            boxes.append(bound_box)
    elif path.endswith("_autopick.star"): # for relion
        for box in boxreader:
            # print(box)
            if box[2] > score_thresh: #score
                bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width, c=box[-1])
                boxes.append(bound_box)
    else:  # for topaz
        for box in boxreader:
            # print(box)
            if box[-1] > score_thresh: #score
                bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width, c=box[-1])
                boxes.append(bound_box)
    return boxes


# read topk boxes from star file, for pre-training
def read_star_file_topk(path, box_width, k=50):
    header_names, skip_indices = get_star_file_header(path)
    boxreader = np.atleast_2d(np.genfromtxt(path, skip_header=skip_indices))
    boxes = []
    for box in boxreader:
        bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width, c=box[-1])
        boxes.append(bound_box)
    results = []
    for box in boxes:
        results.append((box.x, box.y, box.w, box.h, box.c))
    boxes = np.array(results, dtype=np.float32)
    boxes = boxes[np.lexsort(-boxes.T[-1,None])]
    boxes = boxes[:k]
    tmp = []
    for box in boxes: #x, y, w, h,_
        bound_box = BoundBox(x=box[0], y=box[1], w=box_width, h=box_width, c=box[-1])
        tmp.append(bound_box)
    return tmp


# just for test how label percent affect the prediction results.
def read_percent_star_file(path, box_width, percent=100):
    from random import sample
    header_names, skip_indices = get_star_file_header(path)
    boxreader = np.atleast_2d(np.genfromtxt(path, skip_header=skip_indices))
    boxes = []
    for box in boxreader:
        bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width)
        boxes.append(bound_box)
    box_num = int(len(boxes) * percent * 0.01)
    print(f'Before sample: {len(boxes)} boxes total.')
    boxes = sample(boxes,  box_num)
    print(f'After sample: {len(boxes)} boxes are chosen.')


def read_csv_file(path, image_height, box_width=200):
    boxreader = pd.read_csv(path)
    boxes = []
    for index, row in boxreader.iterrows():
        x, y, diameter = int(row['X-Coordinate']), int(row['Y-Coordinate']), int(row['Diameter'])
        if box_width > 0:
            diameter = box_width
        bound_box = BoundBox(x=x-diameter/2, y=image_height-(y+diameter/2), w=diameter, h=diameter)
        boxes.append(bound_box)
    return boxes


def star_file_bin(path, box_width=200, bin=1):
    boxes = read_star_file(path, box_width=box_width)
    # print(os.path.dirname(path))
    bin_path = os.path.join(os.path.dirname(path), f'downsample{bin}')
    print('bin path: ', bin_path)
    os.makedirs(bin_path, exist_ok=True)
    for box in boxes:
        box.x = box.x / bin
        box.y = box.y / bin
        box.w = box.w / bin
        box.h = box.h / bin
    print('Write downsampled star file: ', os.path.join(bin_path, os.path.basename(path)))
    write_star_file(os.path.join(bin_path, os.path.basename(path)).replace('_autopick',''), boxes)


def csv_2_star(path, image_height, box_width=200):
    prefix = os.path.splitext(path)[0]
    boxes = read_csv_file(path, image_height, box_width=200)
    write_star_file(prefix + '.star', boxes)
    