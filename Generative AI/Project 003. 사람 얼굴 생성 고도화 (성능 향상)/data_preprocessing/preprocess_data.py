import os
from pathlib import Path
from crop_and_resize import crop_and_resize

"""
Summary
 - 원본 데이터셋에 대한 crop + resize 및 augmentation 적용

Input
 - 다음 디렉토리에 있는 원본 데이터셋 전체
   - dataset/ThisPersonDoesNotExist
   - dataset/thispersondoesnotexist.10k
   
Output
 - cropped + resized images
   - resized/10k-images (from dataset/thispersondoesnotexist.10k)
   - resized/male       (from dataset/ThisPersonDoesNotExist/male)
   - resized/female     (from dataset/ThisPersonDoesNotExist/female)
 - augmented images
   - augmented/10k-images (from resized/10k-images)
   - augmented/male       (from resized/male)
   - augmented/female     (from resized/female) 
"""


def preprocess_data():
    current_dir = os.getcwd()
    project_home_dir = Path(current_dir).parent.absolute()

    src_dirs = ['thispersondoesnotexist.10k', 'ThisPersonDoesNotExist/female', 'ThisPersonDoesNotExist/male']
    dst_dirs = ['10k-images', 'female', 'male']

    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        crop_and_resize(src_dir=os.path.join(project_home_dir, 'dataset/' + src_dir),
                        dst_dir=os.path.join(project_home_dir, 'resized/' + dst_dir),
                        crop_ratio_vertical=0.0,
                        crop_ratio_horizontal=0.1875,
                        dst_size=128)


if __name__ == '__main__':
    preprocess_data()
