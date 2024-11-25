import os
import csv
from coord_io import read_star_file, read_csv_file
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import glob

def annots_to_topaz(root_path, box_width, image_height):
    '''
    Merge all star files into one topaz txt file. Prepare for topaz picking.
    '''
    annots_path = root_path + 'annots/'
    
    for root, dirs, files in os.walk(annots_path):
        print(files)
        star_files = [f for f in files]
    write_txt = root_path + 'topaz_particles.txt'
    
    with open(write_txt, "w") as boxfile:
        boxwriter = csv.writer(boxfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONE)
        boxwriter.writerow(['image_name', 'x_coord', 'y_coord'])
        for star_file in star_files:
            print(star_file)
            if star_file.endswith('.star'):
                boxes = read_star_file(annots_path + star_file, box_width)
            elif star_file.endswith('.csv'):
                boxes = read_csv_file(annots_path + star_file, image_height, box_width)
            for _, box in enumerate(boxes):
                boxwriter.writerow([star_file[:-4], box.x + box_width*0.5, box.y + box_width*0.5])


# count the number of x in the lst
def countX(lst, x):
    return lst.count(x)


def split_file(data, coord_path):
    print(len(data))
    list_1 = []
    for index,row in data.iterrows():
        image = str(row['image_name'])
        image = str(row['image_name'])
        list_1.append(image)
    print(len(list_1))
    print(list_1)
    # 相同id连续出现数据放同一文件中，视为该id出现一次，列表2用于id出现次数
    list_2 = []
    list_2.append(list_1[0])

    file_name = coord_path + str(list_1[0]) + '.star'
    print(file_name)
    file = open(file_name, 'w')
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

    for i in range(0, len(list_1) - 1):
        if list_1[i] == list_1[i + 1]:
            boxwriter.writerow([data.loc[i + 1]['x_coord'], data.loc[i + 1]['y_coord'], 0, 0.0, data.loc[i + 1]['score']])
        else:
            list_2.append(list_1[i + 1])
            num = countX(list_2, list_1[i + 1])

            file_name = coord_path + list_1[i + 1] + '.star'
            file = open(file_name, 'w')
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
            # file.write(data[i + 1].split()[1:])
            boxwriter.writerow([data.loc[i + 1]['x_coord'], data.loc[i + 1]['y_coord'], 0, 0.0, data.loc[i + 1]['score']])
            # file.write('\r\n')

    boxwriter.writerow([data.loc[len(list_1) - 1]['x_coord'], data.loc[len(list_1) - 1]['y_coord'], 0, 0.0, 0.0])
    file.close()


def topaz_to_annots(root_path, topaz_path, box_width):
    '''
    split topaz txt file into star files. Prepare for making dataset for training.
    '''
    annots_path = root_path + 'topaz/'
    ## load the labeled particles
    labeled_particles = pd.read_csv(topaz_path, sep='\t')
    # print(labeled_particles)
    split_file(labeled_particles, annots_path)


if __name__ == '__main__':
    root_path = '/home/feng/UPicker/data/EMPIAR-cryoPPP10096/'
    box_size = 80
    image_height = 4096
    
    # topaz_to_annots(root_path, box_size)
    # topaz_to_annots('/home/feng/UPicker/comparison_results/EMPIAR10590/', '/home/feng/Topaz_projects/data/EMPIAR-10590/predicted_particles_all_upsampled.txt', 158)
    # topaz_to_annots('/home/feng/UPicker/comparison_results/EMPIAR10028/', 
    #                 '/home/feng/topaz_projects/data/EMPIAR-10028/topaz/predicted_particles_all_upsampled.txt', 200)
    # annots_to_topaz(root_path, box_size, image_height)
    
    topaz_to_annots('/home/feng/UPicker/comparison_results/EMPIAR10532/', '/home/feng/Topaz_projects/data/EMPIAR-10532/predicted_particles_all_upsampled.txt', 90)