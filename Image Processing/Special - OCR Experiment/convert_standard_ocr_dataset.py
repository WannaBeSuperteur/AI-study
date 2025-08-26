
import os
import cv2
import numpy as np

LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']


# thresholding 된 이미지에서 가장자리 흰색 부분 제거
# Create Date: 2025.08.26
# Last Update Date: -

# Arguments:
# img (NumPy array) : thresholding 처리된 이미지

# Returns:
# cropped_img (NumPy array or None) : 흰색 가장자리 부분이 crop 처리된 이미지

def crop_thresholded_img(img):
    (height, width) = np.shape(img)
    top_y, bottom_y, left_x, right_x = None, None, None, None

    for y in range(height):
        if np.sum(img[y]) != 255 * width:
            top_y = y if top_y is None else min(top_y, y)
            bottom_y = y if bottom_y is None else max(bottom_y, y)

    for x in range(width):
        if np.sum(img[:, x]) != 255 * height:
            left_x = x if left_x is None else min(left_x, x)
            right_x = x if right_x is None else max(right_x, x)

    if top_y is None or left_x is None:
        return None

    if bottom_y - top_y < 8:
        return None

    cropped_img = img[top_y:bottom_y+1, left_x:right_x+1]
    return cropped_img


# 이미지 변환 (배경 색 및 글자 색을 검은색, 흰색에 가깝게 변환)
# Create Date: 2025.08.24
# Last Update Date: 2025.08.26
# - Contrast 강화 -> thresholding 방식으로 수정

# Arguments:
# - img_dir    (str) : 이미지 데이터가 있는 디렉토리
# - split_mark (str) : split ('train', 'test') 을 나타내는 표시 ('training' or 'testing')

def convert_and_save_images(img_dir, split_mark):
    img_names = os.listdir(img_dir)
    img_paths = [f'{img_dir}/{img_name}' for img_name in img_names]

    for img_name, img_path in zip(img_names, img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 1차 시도 (threshold = 127)
        img_thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_cropped = crop_thresholded_img(img_thresholded[1])

        # 2차 시도 (threshold = 170)
        if img_cropped is None:
            print(f'try failed with threshold=127 for {img_name}')
            img_thresholded = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
            img_cropped = crop_thresholded_img(img_thresholded[1])

        # 3차 시도 (threshold = 192)
        if img_cropped is None:
            print(f'try failed with threshold=170 for {img_name}')
            img_thresholded = cv2.threshold(img, 192, 255, cv2.THRESH_BINARY)
            img_cropped = crop_thresholded_img(img_thresholded[1])

        # 최종
        if img_cropped is not None:
            new_img_save_dir = img_dir.replace(f'/{split_mark}_data/', f'/{split_mark}_data_modified/')
            new_img_save_dir = str(new_img_save_dir)
            os.makedirs(new_img_save_dir, exist_ok=True)

            new_img_save_path = os.path.join(new_img_save_dir, str(img_name))
            cv2.imwrite(new_img_save_path, img_cropped)


if __name__ == '__main__':
    print('data converting ...')

    train_img_dirs = [f'standard_ocr_dataset/data/training_data/{letter}' for letter in LETTERS]
    test_img_dirs = [f'standard_ocr_dataset/data/testing_data/{letter}' for letter in LETTERS]

    print('')
    for train_img_dir in train_img_dirs:
        print(f'data converting for training images in {train_img_dir} ...')
        convert_and_save_images(train_img_dir, split_mark='training')

    print('')
    for test_img_dir in test_img_dirs:
        print(f'data converting for testing images in {test_img_dir} ...')
        convert_and_save_images(test_img_dir, split_mark='testing')
