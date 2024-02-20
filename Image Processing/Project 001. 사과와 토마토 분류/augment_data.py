import cv2
import os
import numpy as np
import shutil

RESIZE_DEST = 128


# cv2로 이미지 로딩
def load_imgs(is_train, img_class):
    train_or_test = 'train' if is_train else 'test'
    directory = f'archive/{train_or_test}/{img_class}'
    images = []
    
    imgs = os.listdir(directory)
    for img in imgs:
        file_name = directory + '/' + img
        image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        image_resized = cv2.resize(image, dsize=(RESIZE_DEST, RESIZE_DEST))
        images.append(image_resized)
        
    return np.array(images)


# 각 Augmentation 방법 실시
def flip_horizontal(image):
    return cv2.flip(image, 1)


def flip_vertical(image):
    return cv2.flip(image, 0)


def flip_all(image):
    return cv2.flip(cv2.flip(image, 0), 1)


def make_dir_and_imwrite(image_path, image):
    dir_path = '/'.join(image_path.split('/')[:2])
    try:
        os.makedirs(dir_path)
    except:
        pass # "파일이 이미 있으므로 만들 수 없습니다"
        
    cv2.imwrite(image_path, image)


# 각 이미지에 대해 augmentation 적용
def apply_augmentation(images, dir_name, prefix):
    for idx, image in enumerate(images):
        make_dir_and_imwrite(f'images/{dir_name}/{prefix}_{idx}.png', image)

        # horizontal flip
        h_flipped = flip_horizontal(image)

        # vertical filp
        v_flipped = flip_vertical(image)

        # both flip
        all_flipped = flip_all(image)

        make_dir_and_imwrite(f'images/{dir_name}/{prefix}_{idx}_hf.png', h_flipped)
        make_dir_and_imwrite(f'images/{dir_name}/{prefix}_{idx}_vf.png', v_flipped)
        make_dir_and_imwrite(f'images/{dir_name}/{prefix}_{idx}_af.png', all_flipped)


# 모든 이미지들에 대해 augmentation 적용하는 함수
def apply_augmentation_all():
    shutil.rmtree('images')
    apply_augmentation(images=train_tomatoes, dir_name='train', prefix='tomatoes')
    apply_augmentation(images=train_apples, dir_name='train', prefix='apples')


if __name__ == '__main__':
    train_tomatoes = load_imgs(is_train=True, img_class='tomatoes')
    train_apples = load_imgs(is_train=True, img_class='apples')

    print(train_tomatoes)
    print(np.shape(train_tomatoes))

    apply_augmentation_all()
