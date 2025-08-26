
from PIL import Image
import os


# 이미지 회전
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - img_path  (str)   : 회전 대상 이미지 경로
# - angle     (float) : 회전 각도 (도 단위)
# - save_path (str)   : 회전된 이미지 저장 경로

# Output:
# - save_path 에 회전된 이미지 저장됨

def rotate_image(img_path, angle, save_path):
    img = Image.open(img_path)
    img_rotated = img.rotate(angle, expand=True, fillcolor='white')
    img_rotated.save(save_path)


if __name__ == '__main__':
    rotate_image(img_path='scanned_images_dataset/train/Letter/508146503+-6507.jpg',
                 angle=-5,
                 save_path='test_rotate_minus_5_deg.png')

    rotate_image(img_path='scanned_images_dataset/train/Letter/508146503+-6507.jpg',
                 angle=25,
                 save_path='test_rotate_plus_25_deg.png')
