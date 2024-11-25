import os
import csv
from preprocess import find_image_annot_pairs_by_dir


# count the number of x in the lst
def countX(lst, x):
    return lst.count(x)


def split_starfile(star_file_path, coord_path):
    # read the raw star file
    with open(star_file_path, "r", encoding='gb18030', errors='ignore') as f:
        data = [line.rstrip('\n') for line in f]
        print(data[18])
        data = data[18:]
        print(len(data))
        list_1 = []
        for e in range(0, len(data) - 1):
            image = data[e].split()[9]  # image name
            image_pure = image.split('/')[-1]
            image_pure_without_mrc = image_pure[:-5]
            print(image_pure_without_mrc)
            list_1.append(image_pure_without_mrc)
        print(len(list_1))

        list_2 = []
        list_2.append(list_1[0])

        file_name = coord_path + list_1[0] + '.star'
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
                boxwriter.writerow([data[i + 1].split()[10], data[i + 1].split()[11], 0, 0.0, 0.0])
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
                boxwriter.writerow([data[i + 1].split()[10], data[i + 1].split()[11],0, 0.0, 0.0])
                # file.write('\r\n')
    print('end!')



if __name__ == "__main__":
    star_file_path = "data/EMPIAR10081/data.star"
    file_path = "data/EMPIAR10081/"
    annot_path = "data/EMPIAR10081/annots_all/"
    annot_path_sub = 'data/EMPIAR10081/annots/'
    img_dir = file_path + "micrographs/"

    if not os.path.exists(annot_path):
        os.makedirs(annot_path)
    if not os.path.exists(annot_path_sub):
        os.makedirs(annot_path_sub)

    split_starfile(star_file_path, annot_path)
    # os.system("rename 's/_particles_shiny//' data/EMPIAR10081/annots/*.star")

    pairs = find_image_annot_pairs_by_dir(annot_path, img_dir)
    print(len(pairs))

    for img, annot in pairs:
        print(annot)
        os.system(f"cp {annot} {annot_path_sub}")

    if not os.path.exists(file_path + 'annots/'):
        os.makedirs(file_path + 'annots/')