import os
import cv2

RESIZE_DEST = 112


# images 디렉토리 만들기
def make_images_dir():
    try:
        os.makedirs('images')
    except:
        print('"images" directory already exists.')


# archive/ImageNet-Mini/images 내부의 모든 디렉토리에 있는 이미지를 images 디렉토리에 저장
def save_all_images():
    imgs_dir_root = 'archive/flowers'
    imgs_dirs = os.listdir(imgs_dir_root)

    for class_no, dir_ in enumerate(imgs_dirs):
        if dir_ == 'black_eyed_susan':
            imgs_dir = imgs_dir_root + '/' + dir_
            imgs = os.listdir(imgs_dir)[:-100]

            for img_name in imgs:
                image = cv2.imread(imgs_dir + '/' + img_name, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, (RESIZE_DEST, RESIZE_DEST))
                cv2.imwrite('images/class_' + str(class_no) + '_' + img_name, image)


if __name__ == '__main__':
    make_images_dir()
    save_all_images()
