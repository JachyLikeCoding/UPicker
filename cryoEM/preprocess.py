import mrcfile
import math
import torch
import cv2
import time
import numpy as np
import os, sys
import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, data, filters, img_as_ubyte
from PIL import Image
from imageio import imsave
# from util.plot_utils import plot_prediction
# from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from coord_io import read_star_file_topk, read_star_file
from scipy import misc, signal


def get_args_parser():
    parser = argparse.ArgumentParser('UPicker', add_help=False)
    parser.add_argument('--images', default='./data/EMPIAR10028/micrographs/', type=str, help='The folder of micrographs to be preprocessed.')
    parser.add_argument('--output_dir', default='./data/EMPIAR10028', type=str, help='The output folder.')
    parser.add_argument('--bin', default=1, type=int, help='Downsample bin.')
    parser.add_argument('--box_width', default=200, type=int, help='The box width. Usually choose 1.5 * particle diameter.')
    parser.add_argument('--denoise', default='gaussian', type=str, choices=['gaussian', 'bifilter', 'lowpass', 'weiner'], help='The denoise filter. Default is bifilter.')
    parser.add_argument('--noequal', action='store_true', help='If need histogram equalization.')
    parser.add_argument('--ifready', action='store_true', help='If the micrographs have been preprocessed.')
    parser.add_argument('--mode', default='train', type=str, choices=['train','test'], help='If mode is test, no autopick schedule.')
    return parser


def image_read(image_path):
    image_path = str(image_path)
    if image_path.endswith(('.png', '.jpg')):
        try:
            # img = imageio.imread(image_path, mode='L')
            img = Image.open(image_path)
            img = np.array(img, copy=False)
            img = np.flipud(img)
            img = img.astype(np.uint8)
        except ValueError:
            sys.exit("Image" + str(image_path) + " is not valid.")
    elif image_path.endswith(('.mrc', 'mrcs')):
        img = read_mrc(image_path)
    elif image_path.endswith(("tif", "tiff")):
        img = read_tiff(image_path)
    else:
        raise Exception(image_path + 'is not supported image format.')
    
    if img.dtype != np.uint8:
        img = normalize_image(img)
    return img


def normalize_image(image, mi=None, ma=None):
    """Normalize the image data to the range [0, 255]."""
    if mi is None:
        mi = np.min(image)
    if ma is None:
        ma = np.max(image)
    image = (image - mi) / (ma - mi) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def read_width_height(image_path):
    width, height = 0, 0
    if image_path.endswith(("tif", "tiff", "jpg", "png")):
        im = Image.open(image_path)
        width, height = [int(i) for i in im.size]
    elif image_path.endswith(("mrc", "mrcs")):
        with mrcfile.mmap(image_path, permissive=True, mode='r') as mrc:
            width = mrc.header.ny
            height = mrc.header.nx
    return int(width), int(height)


def read_mrc(image_path):
    with mrcfile.open(image_path, permissive=True) as mrc:
            image_data = mrc.data
    return image_data


def read_tiff(image_path):
    image = Image.open(image_path)
    fp = image.fp
    image.load()
    fp.close()
    image = np.array(image, copy=False)
    image = (image - image.mean()) / image.std()
    return image


def mrc_2_png(image_path, output_path, suffix=None):
    image_data = read_mrc(image_path)
    if suffix is None:
        suffix = ''
    output_path = output_path + image_path.split('/')[-1] + suffix + '.plt.png'
    # plt.imsave(output_path, image_data, cmap='gray')
    io.imsave(output_path, image_data)


def is_single_channel(image_path):
    if image_path.endswith('.mrc', '.mrcs'):
        with mrcfile.mmap(image_path, permissive=True, mode='r+') as mrc:
            if mrc.header.nz > 1:
                return False
    return True


def quantize(x, mi, ma, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
    r = ma - mi
    print(f'Quantizing image with min: {mi}, max: {ma}, range: {r}')
    x = 255 * (x - mi) / r
    x = np.round(x).astype(dtype)
    print(f'Quantized image min: {x.min()}, max: {x.max()}')
    return x


def unquantize(x, mi, ma, dtype=np.float32):
    '''convert quantized image array back to approximate unquantized values.'''
    x = x.astype(dtype)
    y = x * (ma - mi) / 255 + mi
    return y


def image_write(image_path, image):
    if image_path.endswith((".jpg", '.png')):
        # if isinstance(image, np.ndarray) and image.dtype == np.float32:
        # # 将浮点型数组的值缩放到 [0, 255]
        #     image = (image * 255).astype(np.uint8)
        # image = Image.fromarray(image).convert('L')
        imageio.imwrite(image_path, image)
        # io.imsave(image_path, image)
    elif image_path.endswith((".tif", '.tiff')):
        image = np.float32(image)
        imageio.imwrite(image_path, image)
    elif image_path.endswith((".mrc", ".mrcs")):
        image = np.flipud(image)
        with mrcfile.new(image_path, overwrite=True) as mrc:
            mrc.set_data(np.float32(image))


def save_image(image, path, mi, ma, f=None):
    if f is None:
        f = os.path.splitext(path)[1]
        f = f[1:]  # remove the period
    else:
        path = path + '.' + f

    if f == 'mrc':
        save_mrc(image, path)
    elif f == 'tiff' or f == 'tif':
        save_tiff(image, path)
    elif f == 'png':
        save_png(image, path, mi=mi, ma=ma)
    elif f == 'jpg' or f == 'jpeg':
        print('save jpeg: ', path)
        print(image.dtype)
        print(image.max(), image.min())
        save_jpeg(image, path, mi=mi, ma=ma)


def save_mrc(image, image_path):
    with mrcfile.new(image_path, overwrite=True) as mrc:
        mrc.set_data(np.float32(image))


def save_png(image, image_path, mi, ma):
    # byte encode the image
    im = Image.fromarray(quantize(image, mi, ma))
    # im = Image.fromarray(quantize(image))
    # im = Image.fromarray(image)
    im.save(image_path, 'png')


def save_jpeg(image, image_path, mi, ma):
    image = Image.fromarray(image)

    if image.mode == 'F':
        image = image.convert('L')
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(image_path, 'JPEG')


def save_tiff(image, image_path):
    im = Image.fromarray(image)
    im.save(image_path, 'tiff')


def imadjust(image, low_in, high_in, low_out, high_out, gamma, c):
    w, h = image.shape
    result = np.zeros((w, h)).astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    print('\np1, p99:\n', p1, p99)
    img_out = np.clip(image, p1, p99)
    img_out = (((img_out - p1) /
                (p99 - p1))**gamma) * (high_out - low_out) + low_out
    img_out = c * img_out * 255
    img_out = img_out.astype(np.uint8)

    return img_out


def bi_filter(image, d=0):
    denoised = cv2.bilateralFilter(image, d, 100, 5)
    return denoised

def gaussian_blur(image):
    # Apply Gaussian Blur to denoise
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised

def weiner_filter(image):
    if image.dtype == np.uint8:
        image = np.float32(image / 255.0)
    denoised = signal.wiener(image, (5, 5))
    return denoised


def lowpass_filter(image):
    b, a = signal.butter(8, 0.2, 'lowpass')
    denoised = signal.filtfilt(b, a, image)
    return denoised


def equal_hist(image):
    equalized = cv2.equalizeHist(image)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
    enhanced = clahe.apply(equalized)

    return enhanced


def downsample(image, factor=1, shape=None):
    """Downsample a given 2D image using fourier transform"""
    if shape is None:
        m, n = image.shape[-2:]
        m = int(m / factor)
        n = int(n / factor)
        shape = (m, n)

    F = np.fft.rfft2(image)

    m, n = shape
    A = F[..., 0:m // 2, 0:n // 2 + 1]
    B = F[..., -m // 2:, 0:n // 2 + 1]
    F = np.concatenate([A, B], axis=0)

    ## scale the signal from down-sampling
    a = n * m
    b = image.shape[-2] * image.shape[-1]
    F *= (a / b)

    f = np.fft.irfft2(F, s=shape)

    return f.astype(image.dtype)


def flatten(image_path):
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    os.system(f"ib.micrographs_flatten -i {image_dir}")
    flatten_mrc = read_mrc(image_dir + '/flattened/' + image_name)

    return flatten_mrc


def log_autopick(img_path,
                 output_path='AutoPick/',
                 diam_min=200,
                 diam_max=250):
    print('img_path: ', img_path)
    cur_path = os.getcwd()
    img_name = img_path.split('/')[-1]
    img_dir = os.path.dirname(img_path)
    print('img_dir: ', img_dir)
    os.chdir(img_dir)
    os.system(f'pwd')
    print(
        f"relion_autopick --i {img_name} --odir {output_path} --pickname autopick --LoG --LoG_diam_min {diam_min} --LoG_diam_max {diam_max} --shrink 0 --lowpass 20 --LoG_adjust_threshold 0.2"
    )
    os.system(
        f"relion_autopick --i {img_name} --odir {output_path} --pickname autopick --LoG --LoG_diam_min  {diam_min} --LoG_diam_max {diam_max} --shrink 0 --lowpass 20 --LoG_adjust_threshold 0.2"
    )
    
    star_path = os.path.join(output_path, img_name[:-4] + '_autopick.star')
    print('star_path: ', star_path)
    boxes = read_star_file(star_path, box_width=(diam_max + diam_min) / 2)
    # boxes = read_star_file_topk(star_path, box_width=250, k=30)
    print(len(boxes))
    os.chdir(cur_path)
    results = []
    for box in boxes:
        results.append((box.x, box.y, box.x + box.w, box.y + box.h))
    results = np.array(results)
    return results


def log_autopick_list(img_dir, output_path, diam_min, diam_max):
    image_count = 0
    for image in os.listdir(img_dir):
        if image.endswith('.mrc'):
            image_count += 1
            print('LoG AUTOPICK for region proposals ================================', image)
            img_path = os.path.join(img_dir, image)
            log_autopick(img_path, output_path, diam_min, diam_max)
    return image_count


def selective_search(img, h, w, res_size=512):
    img_det = np.array(img)
    img_det = cv2.cvtColor(img_det, cv2.COLOR_GRAY2BGR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    fast = False
    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    boxes = ss.process().astype('float32')
    print('\nTotal Number of Region Proposals: {}'.format(len(boxes)))
    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]

    return boxes


def preprocess_image(image_path, bin=1, filter='bifilter', equal=True):
    image = image_read(image_path)
    print(f'Original image: {image_path}, shape: {image.shape}, dtype: {image.dtype}')
    # denoise
    if filter == 'bifilter':
        image = bi_filter(image)
    elif filter == 'weiner':
        image = weiner_filter(image)
    elif filter == 'lowpass':
        image = lowpass_filter(image)
    elif filter == 'gaussian':
        image = gaussian_blur(image)
    else:
        print('No denoise process...')
        pass
    # enhance
    if equal:
        image = equal_hist(image)
        print(f'Equalized histogram: {image_path}, shape: {image.shape}, dtype: {image.dtype}')
    else:
        print('No equal hist process...')
        pass
    # downsample
    if bin != 1:
        image = downsample(image, bin)
        print(f'Downsampled image: {image_path}, shape: {image.shape}, dtype: {image.dtype}')

    return image



def preprocess_image_list(images_dir, output_dir, bin, filter, equal, ifready):
    if not os.path.exists(output_dir + '/processed'):
        os.makedirs(output_dir + '/processed')

    for filename in os.listdir(images_dir):
        img_path = os.path.join(images_dir, filename)
        if filename.endswith(('.mrc', '.tif', '.png', '.jpg')):
            image = image_read(img_path)
            print(f'Read image: {filename}, shape: {image.shape}, dtype: {image.dtype}')
            out_path = output_dir + "processed/" + filename[:-4]+ '.jpg'
            print("write: ", out_path)
            if ifready:
                save_image(image, out_path)
            else:
                preprocessed_image = preprocess_image(os.path.join(images_dir, filename),
                                                  bin=bin,
                                                  filter=filter,
                                                  equal=equal)
                save_image(preprocessed_image, out_path, mi=None, ma=None)
                # image_write(out_path, preprocessed_image)

    # save the same bin annotations
    if bin != 1:
        annot_dir =  os.path.dirname(os.path.dirname(images_dir)) + '/annots'
        print('annot_dir: ', annot_dir)
        for filename in os.listdir(annot_dir):
            if filename.endswith('.star'):
                from cryoEM.coord_io import star_file_bin
                star_file_bin(os.path.join(annot_dir, filename),
                              box_width=args.box_width / bin,
                              bin=bin)


def tif2png(imagesDirectory=None):
    distDirectory = os.path.dirname(imagesDirectory)
    distDirectory = os.path.join(distDirectory, "png")  # 要存放png格式的文件夹路径
    try:
        os.mkdir(distDirectory)
    except OSError:
        print("The dictionary exists")
    for imageName in os.listdir(imagesDirectory):
        if imageName.endswith('.tif'):
            imagePath = os.path.join(imagesDirectory, imageName)
            image = Image.open(imagePath)
            data = np.array(image, dtype=np.float32)
            distImagePath = os.path.join(distDirectory, imageName[:-4] +
                                         '.png')  # 更改图像后缀为.png，与原图像同名
            image.save(distImagePath)

from PIL import Image

def png_to_jpeg(input_path, output_path):
    # Open the PNG image
    # with Image.open(input_path) as img:
    #     # If the image has an alpha channel, convert it to RGB
    #     if img.mode == 'RGBA':
    #         img = img.convert('RGB')
    #     # Save the image as JPEG
    #     img.save(output_path, 'JPEG')
    
    with Image.open(input_path) as img:
        # Convert image to grayscale if it is not already
        if img.mode != 'L':
            img = img.convert('L')
        img = img.convert('RGB')  # Ensure the image is in RGB mode
        img.save(output_path, 'JPEG')

        

def find_image_annot_pairs(annotations, images):
    print('annotations:', annotations)
    import difflib
    img_names = list(map(os.path.basename, images))
    img_anno_pairs = []
    for ann in annotations:
        # print('ann:----', ann)
        ann_without_ext = os.path.splitext(os.path.basename(ann))[0]
        cand_list = [i for i in img_names if ann_without_ext in i]
        try:
            cand_list_no_ext = list(map(os.path.basename, cand_list))
            corresponding_img_path = difflib.get_close_matches(
                ann_without_ext, cand_list_no_ext, n=1, cutoff=0)[0]
            corresponding_img_path = cand_list[cand_list_no_ext.index(
                corresponding_img_path)]
        except IndexError:
            # print("Cannot find corresponding image file for ", ann, '- Skipped.')
            continue
        index_image = img_names.index(corresponding_img_path)
        img_anno_pairs.append((images[index_image], ann))
    return img_anno_pairs


def find_image_annot_pairs_by_dir(ann_dir, img_dir):
    if not os.path.exists(ann_dir):
        print("Annotation folder does not exist:", ann_dir,
              "Please check your config file.")
        sys.exit(1)
    if not os.path.exists(img_dir):
        print("Your image folder does not exist:", img_dir,
              "Please check your config file.")
        sys.exit(1)

    img_files = []
    for root, directories, filenames in os.walk(img_dir, followlinks=True):
        for filename in filenames:
            if filename.endswith((".jpg", ".png", ".mrc", ".tif",
                                  ".tiff")) and not filename.startswith("."):
                img_files.append(os.path.join(root, filename))

    # Read annotations
    annotations = []
    for root, directories, filenames in os.walk(ann_dir, followlinks=True):
        for ann in sorted(filenames):
            if ann.endswith(
                (".star", ".box", ".txt")) and not filename.startswith("."):
                annotations.append(os.path.join(root, ann))
    img_annot_pairs = find_image_annot_pairs(annotations, img_files)
    # print('img_annot_pairs:\n', img_annot_pairs)
    return img_annot_pairs


def main(args):
    print(args)
    equal = not args.noequal
    print(f'equal: {equal}, ifready: {args.ifready}')

    preprocess_image_list(images_dir=args.images,
                          output_dir = args.output_dir,
                          bin=args.bin,
                          filter=args.denoise,
                          equal=equal,
                          ifready=args.ifready)

    start = time.time()
    if args.mode == 'test':
        pass
    else:
        image_count = log_autopick_list(img_dir=args.images,
                        output_path='AutoPick/',
                        diam_min=int(args.box_width * 0.88),
                        diam_max=int(args.box_width * 1.11))
        end = time.time()
        print('total log time: ', end-start)
        if image_count == 0:
            print('No image to autopick.')
        else:
            print('average cost time:', (end-start)/image_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocess script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)