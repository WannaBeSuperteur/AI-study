import cv2
import numpy as np
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils.utils import get_img_path


def change_brightness(img, ratio):

    """
    Brightness change for image

    Args:
        img   (np.array) : source image
        ratio (float)    : brightness change ratio (0.0 : black, 1.0 : original, 2.0 : white)
    """

    if ratio > 1.0:
        return (img * (2.0 - ratio) + 255.0 * (ratio - 1.0)).astype(int)
    elif ratio < 1.0:
        return (img * ratio).astype(int)

    return img


def augment_images(src_dir, dst_dir, ratio_range):

    """
    Crop + Resize images in specific source directory, then write them in destination directory

    Args:
        src_dir     (str)  : absolute path of source images
        dst_dir     (str)  : absolute path of crop + resized images
        ratio_range (list) : brightness change ratio range list for change_brightness() function (repeated)
    """

    img_names = os.listdir(src_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for idx, img_name in enumerate(img_names):
        img_array = np.fromfile(get_img_path(src_dir, img_name), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        brightness_ratio = ratio_range[idx % len(ratio_range)]
        image_augmented = change_brightness(image, brightness_ratio)

        dst_path = get_img_path(dst_dir, img_name)
        extension = os.path.splitext(dst_path)[1]
        result, encoded_img = cv2.imencode(extension, image_augmented)

        with open(dst_path, mode='w+b') as f:
            encoded_img.tofile(f)


if __name__ == '__main__':

    # expected test result :
    # [[ 75 120 165 210]
    #  [ 70 115 160 205]]

    print('==== test 1')
    print(change_brightness(img=np.array([[50, 100, 155, 205],
                                          [50, 100, 150, 200]]), ratio=1.1))

    # expected test result :
    # [[ 49  94 139 184]
    #  [ 45  90 135 180]]

    print('==== test 2')
    print(change_brightness(img=np.array([[55, 105, 155, 205],
                                          [50, 100, 150, 200]]), ratio=0.9))