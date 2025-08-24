
import os
from PIL import Image, ImageEnhance

LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']


# 이미지 변환 (배경 색 및 글자 색을 검은색, 흰색에 가깝게 변환)
# Create Date: 2025.08.24
# Last Update Date: -

# Arguments:
# - img_dir    (str) : 이미지 데이터가 있는 디렉토리
# - split_mark (str) : split ('train', 'test') 을 나타내는 표시 ('training' or 'testing')

def convert_and_save_images(img_dir, split_mark):
    img_names = os.listdir(img_dir)
    img_paths = [f'{img_dir}/{img_name}' for img_name in img_names]

    for img_name, img_path in zip(img_names, img_paths):
        img = Image.open(img_path)
        new_img = ImageEnhance.Contrast(img).enhance(3.0)

        new_img_save_dir = img_dir.replace(f'/{split_mark}_data/', f'/{split_mark}_data_modified/')
        new_img_save_dir = str(new_img_save_dir)
        os.makedirs(new_img_save_dir, exist_ok=True)

        new_img_save_path = os.path.join(new_img_save_dir, str(img_name))
        new_img.save(new_img_save_path)


if __name__ == '__main__':
    print('data splitting ...')

    train_img_dirs = [f'standard_ocr_dataset/data/training_data/{letter}' for letter in LETTERS]
    test_img_dirs = [f'standard_ocr_dataset/data/testing_data/{letter}' for letter in LETTERS]

    for train_img_dir in train_img_dirs:
        convert_and_save_images(train_img_dir, split_mark='training')

    for test_img_dir in test_img_dirs:
        convert_and_save_images(test_img_dir, split_mark='testing')
