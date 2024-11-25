import os,sys

def find_image_annot_pairs(annotations, images):
    import difflib
    img_names = list(map(os.path.basename, images))
    img_anno_pairs = []
    for ann in annotations:
        ann_without_ext = os.path.splitext(os.path.basename(ann))[0]
        cand_list = [i for i in img_names if ann_without_ext in i]
        try:
            cand_list_no_ext = list(map(os.path.basename, cand_list))
            corresponding_img_path = difflib.get_close_matches(ann_without_ext, cand_list_no_ext, n=1, cutoff=0)[0]
            corresponding_img_path = cand_list[cand_list_no_ext.index(corresponding_img_path)]
        except IndexError:
            print("Cannot find corresponding image file for ", ann, '- Skipped.')
            continue
        index_image = img_names.index(corresponding_img_path)
        img_anno_pairs.append((images[index_image], ann))
    return img_anno_pairs


def find_image_annot_pairs_by_dir(ann_dir, img_dir):
    if not os.path.exists(ann_dir):
        print("Annotation folder does not exist:", ann_dir, "Please check your config file.")
        sys.exit(1)
    if not os.path.exists(img_dir):
        print("Your image folder does not exist:", ann_dir, "Please check your config file.")
        sys.exit(1)

    img_files = []
    for root, directories, filenames in os.walk(img_dir, followlinks=True):
        for filename in filenames:
            if filename.endswith(("jpg", "png", "mrc", "mrcs", "tif", "tiff")) and not filename.startswith("."):
                img_files.append(os.path.join(root, filename))

    # Read annotations
    annotations = []
    for root, directories, filenames in os.walk(ann_dir, followlinks=True):
        for ann in sorted(filenames):
            if ann.endswith(("star", "box", "txt")) and not filename.startswith("."):
                annotations.append(os.path.join(root, ann))
    img_annot_pairs = find_image_annot_pairs(annotations, img_files)

    return img_annot_pairs


if __name__ == '__main':
    ann_dir = ''
    img_dir = 'data/EMPIAR10081/micrographs'
    find_image_annot_pairs_by_dir(ann_dir, img_dir)