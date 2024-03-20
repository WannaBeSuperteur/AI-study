import os
import cv2

RESIZE_DEST = 120


# 이미지에 대해 crop augmentation 적용
def crop(image, top=0, bottom=0, left=0, right=0):
    height = len(image)
    width = len(image[0])
    
    return image[top : height-bottom, left : width-right]


# 특정 image를 augmentation
def augment_image(image):
    try:
        cropped_image_0 = crop(image, top=75, left=75)
        cropped_image_1 = crop(image, top=75, right=75)
        cropped_image_2 = crop(image, bottom=75, left=75)
        cropped_image_3 = crop(image, bottom=75, right=75)

        return (cropped_image_0, cropped_image_1, cropped_image_2, cropped_image_3)
    
    except: # 가로 또는 세로 길이가 75 미만이어서 crop이 불가능하거나, 기타 등등의 오류 발생
        return None


# augment 된 이미지를 resize 후 저장
def resize_and_save(image, img_name):
    image = cv2.resize(image, (RESIZE_DEST, RESIZE_DEST))
    cv2.imwrite('images/' + img_name, image)


# archive/ImageNet-Mini/images 내부의 모든 디렉토리에 있는 이미지에 대해 augmentation 실시
def augment_all_images():
    imgs_dir_root = 'archive/ImageNet-Mini/images'
    imgs_dirs = os.listdir(imgs_dir_root)

    for dir_ in imgs_dirs:
        imgs_dir = imgs_dir_root + '/' + dir_
        imgs = os.listdir(imgs_dir)

        for img_name in imgs:
            image = cv2.imread(imgs_dir + '/' + img_name, cv2.IMREAD_UNCHANGED)

            try:
                (img_aug0, img_aug1, img_aug2, img_aug3) = augment_image(image)

                resize_and_save(img_aug0, img_name[:-5] + '_aug0.png')
                resize_and_save(img_aug1, img_name[:-5] + '_aug1.png')
                resize_and_save(img_aug2, img_name[:-5] + '_aug2.png')
                resize_and_save(img_aug3, img_name[:-5] + '_aug3.png')

            except: # 오류로 인해 augment_image(image) 의 결과가 None 인 경우
                pass


if __name__ == '__main__':
    augment_all_images()
