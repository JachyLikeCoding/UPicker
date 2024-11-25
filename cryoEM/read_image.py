import mrcfile
import numpy as np
import sys
from skimage import io
from PIL import Image


def image_read(image_path):
    image_path = str(image_path)
    if image_path.endswith(('.png', '.jpg')):
        try:
            # img = imageio.imread(image_path, mode='L')
            img = Image.open(image_path)
            img = np.array(img, copy=False)
            img = np.flipud(img)
            # img = img.astype(np.uint8)
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
    mrc = mrcfile.open(image_path, permissive=True, mode='r+')
    if not mrc.is_single_image():
        raise Exception('Movie files are not supported')
        return None
    mrc_data = mrc.data
    mrc_data = np.squeeze(mrc_data)
    # mrc_data = np.flipud(mrc_data)
    return mrc_data


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
