import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils.utils import get_img_path


def crop_and_resize(src_dir, dst_dir, crop_ratio_vertical, crop_ratio_horizontal, dst_size):

    """
    Crop + Resize images in specific source directory, then write them in destination directory

    Args:
        src_dir               (str)   : absolute path of source images
        dst_dir               (str)   : absolute path of crop + resized images
        crop_ratio_vertical   (float) : vertical cropping ratio (0.0 - 1.0)
        crop_ratio_horizontal (float) : horizontal cropping ratio (0.0 - 1.0)
        dst_size              (int)   : destination image size (width/height before cropping)
    """

    img_names = os.listdir(src_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for img_name in img_names:
        img_array = np.fromfile(get_img_path(src_dir, img_name), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        img_height = image.shape[0]
        img_width = image.shape[1]
        crop_size_vertical = int(0.5 * img_height * crop_ratio_vertical)
        crop_size_horizontal = int(0.5 * img_width * crop_ratio_horizontal)

        if crop_size_vertical == 0 and crop_size_horizontal == 0:
            image_cropped = image
        elif crop_size_vertical == 0:
            image_cropped = image[:, crop_size_horizontal:-crop_size_horizontal]
        elif crop_size_horizontal == 0:
            image_cropped = image[crop_size_vertical:-crop_size_vertical, :]
        else:
            image_cropped = image[crop_size_vertical:-crop_size_vertical, crop_size_horizontal:-crop_size_horizontal]

        dst_height = int(dst_size * (1.0 - crop_ratio_vertical))
        dst_width = int(dst_size * (1.0 - crop_ratio_horizontal))
        image_resized = cv2.resize(image_cropped, (dst_width, dst_height))

        dst_path = get_img_path(dst_dir, img_name)
        extension = os.path.splitext(dst_path)[1]
        result, encoded_img = cv2.imencode(extension, image_resized)

        with open(dst_path, mode='w+b') as f:
            encoded_img.tofile(f)
