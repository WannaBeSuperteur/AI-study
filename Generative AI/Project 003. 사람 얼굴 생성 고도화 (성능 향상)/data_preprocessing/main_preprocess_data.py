import os
from pathlib import Path
from crop_and_resize import crop_and_resize
from augmentation import augment_images


"""
Summary
 - 원본 데이터셋에 대한 crop + resize 및 augmentation 적용

Input
 - 다음 디렉토리에 있는 원본 데이터셋 전체
   - dataset/original/ThisPersonDoesNotExist
   - dataset/original/thispersondoesnotexist.10k
   
Output
 - cropped + resized images
   - dataset/resized/10k-images (from dataset/original/thispersondoesnotexist.10k)
   - dataset/resized/male       (from dataset/original/ThisPersonDoesNotExist/male)
   - dataset/resized/female     (from dataset/original/ThisPersonDoesNotExist/female)
 - augmented images
   - dataset/augmented/10k-images (from dataset/resized/10k-images)
   - dataset/augmented/male       (from dataset/resized/male)
   - dataset/augmented/female     (from dataset/resized/female) 
"""


def preprocess_data():
    current_dir = os.getcwd()
    project_home_dir = Path(current_dir)  # 프로젝트 홈 디렉토리에서 실행하지 않으면 오류 발생

    if 'resized' in os.listdir(os.path.join(project_home_dir, 'dataset')):
        print('preprocessed data already exists')
        return

    src_dirs = ['thispersondoesnotexist.10k', 'ThisPersonDoesNotExist/female', 'ThisPersonDoesNotExist/male']
    dst_dirs = ['10k-images', 'female', 'male']

    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        crop_and_resize(src_dir=os.path.join(project_home_dir, 'dataset/original/' + src_dir),
                        dst_dir=os.path.join(project_home_dir, 'dataset/resized/' + dst_dir),
                        crop_ratio_vertical=0.0,
                        crop_ratio_horizontal=0.1875,
                        dst_size=128)


def augment_data():
    current_dir = os.getcwd()
    project_home_dir = Path(current_dir)  # 프로젝트 홈 디렉토리에서 실행하지 않으면 오류 발생

    if 'augmented' in os.listdir(os.path.join(project_home_dir, 'dataset')):
        print('augmented data already exists')
        return

    src_dst_dirs = ['10k-images', 'female', 'male']

    for directory in src_dst_dirs:
        augment_images(src_dir=os.path.join(project_home_dir, 'dataset/resized/' + directory),
                       dst_dir=os.path.join(project_home_dir, 'dataset/augmented/' + directory),
                       ratio_range=[0.84, 0.864, 0.888, 0.912, 0.936, 0.96,
                                    1.04, 1.064, 1.088, 1.112, 1.136, 1.16])


if __name__ == '__main__':
    preprocess_data()
    augment_data()
