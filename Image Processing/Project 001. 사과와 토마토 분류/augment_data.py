import cv2
import os
import numpy as np

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
        
    return np.array(images) / 255.0


if __name__ == '__main__':
    train_tomatoes = load_imgs(is_train=True, img_class='tomatoes')

    print(train_tomatoes)
    print(np.shape(train_tomatoes))
