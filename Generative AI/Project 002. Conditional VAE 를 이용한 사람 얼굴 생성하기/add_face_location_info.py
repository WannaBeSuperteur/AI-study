import pandas as pd
import numpy as np
import os
import cv2

IMG_SIZE = 120
IMG_CENTER = IMG_SIZE // 2

CENTER_X = 60
CENTER_Y = 68
CENTER_RADIUS = 8
CENTER_PIXELS_COUNT = (2 * CENTER_RADIUS) * (2 * CENTER_RADIUS)
EPSILON = 0.0001
MASK_SAMPLES_PER_DATASET = 100


def get_mean_and_std_of_center_pixels(center_pixels):
    center_pixels_ = np.reshape(center_pixels, (CENTER_PIXELS_COUNT, 3))
    center_pixels_rgb_ratio = [
        center_pixels_[:, 0] / (center_pixels_[:, 1] + EPSILON),  # blue / green
        center_pixels_[:, 0] / (center_pixels_[:, 2] + EPSILON),  # blue / red
        center_pixels_[:, 1] / (center_pixels_[:, 2] + EPSILON)   # green / red
    ]

    center_pixels_mean = np.mean(center_pixels_rgb_ratio, axis=1)
    center_pixels_std = np.std(center_pixels_rgb_ratio, axis=1)

    return center_pixels_mean, center_pixels_std


def create_mask(img, center_pixels_mean, center_pixels_std):
    B, G, R = cv2.split(img)
    ratio_bg = B / (G + EPSILON)
    ratio_br = B / (R + EPSILON)
    ratio_gr = G / (R + EPSILON)

    mask_bg = abs((ratio_bg - center_pixels_mean[0]) / (center_pixels_std[0] + EPSILON)) <= 1.7
    mask_br = abs((ratio_br - center_pixels_mean[1]) / (center_pixels_std[1] + EPSILON)) <= 1.7
    mask_gr = abs((ratio_gr - center_pixels_mean[2]) / (center_pixels_std[2] + EPSILON)) <= 1.7
    mask_all = mask_bg & mask_br & mask_gr
    mask_all = mask_all.astype(np.uint8) * 255

    return mask_all


def first_consecutive_5_white_after_5_blacks(arr):
    arr_size = len(arr)
    cur_blacks = 0
    cur_consecutive_whites = 0

    for i in range(arr_size):
        if arr[i] == 0:
            cur_blacks += 1

        if cur_blacks >= 5:
            if arr[i] > 0:
                cur_consecutive_whites += 1
                if cur_consecutive_whites >= 5:
                    return i
            else:
                cur_consecutive_whites = 0

    return arr_size


def get_face_location_info(img_mask):
    from_top = img_mask[:, IMG_CENTER]
    from_left = img_mask[IMG_CENTER]
    from_right = from_left[::-1]

    face_loc_from_top = first_consecutive_5_white_after_5_blacks(from_top)
    face_loc_from_left = first_consecutive_5_white_after_5_blacks(from_left)
    face_loc_from_right = first_consecutive_5_white_after_5_blacks(from_right)

    return {
        'face_location_top': [face_loc_from_top],
        'face_location_left': [face_loc_from_left],
        'face_location_right': [face_loc_from_right]
    }


def add_face_location_info(image_name, image_dir, face_location_info_df, face_location_info_file_path, save_mask=False):
    img_path = image_dir + '/' + image_name
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # (120, 120, 3)

    # distribution of central pixels (face)
    center_pixels = img[CENTER_Y-CENTER_RADIUS : CENTER_Y+CENTER_RADIUS, CENTER_X-CENTER_RADIUS : CENTER_X+CENTER_RADIUS]
    center_pixels_mean, center_pixels_std = get_mean_and_std_of_center_pixels(center_pixels)

    # create mask
    img_mask = create_mask(img, center_pixels_mean, center_pixels_std)

    # save mask
    if save_mask:
        cv2.imwrite(face_location_info_file_path, img_mask)

    # get face location info
    face_location_info = get_face_location_info(img_mask)
    face_location_info = pd.DataFrame(face_location_info)
    face_location_info_df = pd.concat([face_location_info, face_location_info_df])

    return face_location_info_df


def save_face_location_info_for_all_images():
    print('saving face location info start ...')

    face_location_info = pd.DataFrame()

    image_dirs = ['resized_images/first_dataset', 'resized_images/second_dataset_male',
                  'resized_images/second_dataset_female']

    # create face location info
    for image_dir in image_dirs:
        for idx, image_name in enumerate(os.listdir(image_dir)):
            face_location_info = add_face_location_info(
                image_name, image_dir,
                face_location_info_df=face_location_info,
                face_location_info_file_path=f'sample_face_location_info/{image_dir.split("/")[-1]}_{idx}.png',
                save_mask=(idx < MASK_SAMPLES_PER_DATASET)
            )

    # normalize face location info columns
    face_location_cols = ['face_location_top', 'face_location_left', 'face_location_right']
    image_count = len(face_location_info)

    for col in face_location_cols:
        column = face_location_info[col]
        bottom_5 = column.sort_values().iloc[int(0.05 * image_count)]
        top_10 = column.sort_values().iloc[int(0.9 * image_count)]
        print(f'column {col}, top 10% = {top_10}, bottom 5% = {bottom_5}')

        face_location_info[f'{col}_normalized'] = (column - bottom_5) / (top_10 - bottom_5)

    # concat into existing dataset
    condition_data = pd.read_csv('condition_data.csv', index_col=0)
    new_condition_data = pd.concat([condition_data, face_location_info], axis=1)
    new_condition_data.to_csv('condition_data.csv')

    print('finished')


def create_sample_face_location_info_dir():
    try:
        os.makedirs('sample_face_location_info')
    except:
        pass


if __name__ == '__main__':
    create_sample_face_location_info_dir()
    save_face_location_info_for_all_images()
