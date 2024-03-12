import os
import cv2

DEST_IMG_SIZE = 120
CROPPED_EACH_SIZE = 32


# 디렉토리 생성
def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except:
        pass


# 이미지 리사이징 -> crop 후 저장
def resize_crop_and_save_image(src_path, dest_path):
    image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    image_cropped = image[CROPPED_EACH_SIZE:-CROPPED_EACH_SIZE, CROPPED_EACH_SIZE:-CROPPED_EACH_SIZE]
    image_resized = cv2.resize(image_cropped, (DEST_IMG_SIZE, DEST_IMG_SIZE))

    cv2.imwrite(dest_path, image_resized)


# 각 디렉토리 단위로 이미지 읽고 리사이징 -> crop 후 저장
def read_dir_and_resize_crop(dirs_map):
    for src_dir, dest_dir in dirs_map.items():
        src_imgs = os.listdir(src_dir)

        for img in src_imgs:
            src_path = src_dir + '/' + img
            dest_path = dest_dir + '/' + img
            
            resize_crop_and_save_image(src_path, dest_path)


if __name__ == '__main__':
    dirs_map = {
        'thispersondoesnotexist.10k': 'resized_images/first_dataset',
        'ThisPersonDoesNotExist/Female': 'resized_images/second_dataset_female',
        'ThisPersonDoesNotExist/Male': 'resized_images/second_dataset_male'
    }

    for _, dest_dir in dirs_map.items():
        create_dir(dest_dir)
    
    # 이미지 리사이징 및 crop 적용
    read_dir_and_resize_crop(dirs_map)
