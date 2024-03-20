import os
import cv2

RESIZE_DEST = 120


# images 디렉토리 만들기
def make_images_dir():
    try:
        os.makedirs('images')
    except:
        print('"images" directory already exists.')


# archive/ImageNet-Mini/images 내부의 모든 디렉토리에 있는 이미지를 images 디렉토리에 저장
def save_all_images():
    imgs_dir_root = 'archive/ImageNet-Mini/images'
    imgs_dirs = os.listdir(imgs_dir_root)

    for dir_ in imgs_dirs:
        imgs_dir = imgs_dir_root + '/' + dir_
        imgs = os.listdir(imgs_dir)

        for img_name in imgs:
            image = cv2.imread(imgs_dir + '/' + img_name, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (RESIZE_DEST, RESIZE_DEST))
            cv2.imwrite('images/' + img_name, image)


if __name__ == '__main__':
    make_images_dir()
    save_all_images()
