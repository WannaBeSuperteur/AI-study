import pandas as pd
import numpy as np
import os
import cv2

IMG_SIZE = 120
IMG_CENTER = IMG_SIZE // 2

CENTER_X = 60
CENTER_Y = 68
CENTER_RADIUS = 12
CENTER_PIXELS_COUNT = (2 * CENTER_RADIUS) * (2 * CENTER_RADIUS)
EPSILON = 0.0001
MASK_SAMPLES_PER_DATASET = 100
MAX_PIXEL_DIST_BASE = 5500


# (now unused) compute mean and std of center area pixel values
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


# get threshold distance for each position (y, x)
def get_threshold_dist(y, x):
    y_diff = max(0, abs(y - CENTER_Y) - 16)
    x_diff = max(0, abs(x - CENTER_X) - 16)

    return MAX_PIXEL_DIST_BASE / (y_diff + 12 * x_diff + 10)


# distance of pixel
def compute_dist(pixel0, pixel1):
    pixel0_ = np.array(pixel0, dtype=np.int32)
    pixel1_ = np.array(pixel1, dtype=np.int32)

    blue_diff = (pixel0_[0] - pixel1_[0]) * (pixel0_[0] - pixel1_[0])
    green_diff = (pixel0_[1] - pixel1_[1]) * (pixel0_[1] - pixel1_[1])
    red_diff = (pixel0_[2] - pixel1_[2]) * (pixel0_[2] - pixel1_[2])

    pixel_dist = blue_diff + green_diff + red_diff
    return pixel_dist


# run bfs to 'segment' face
def run_bfs(mask_all, y, x, img):
    cury = y
    curx = x
    curpixel = img[y][x]
    dist_thresh = get_threshold_dist(y, x)
    queue = []

    while True:
        for d in [1, 2]:
            if cury >= d and mask_all[cury - d][curx] == 0 and compute_dist(curpixel, img[cury - d][curx]) <= dist_thresh:
                mask_all[cury - d][curx] = 255
                queue.append([cury - d, curx])

            if cury < IMG_SIZE - d and mask_all[cury + d][curx] == 0 and compute_dist(curpixel, img[cury + d][curx]) <= dist_thresh:
                mask_all[cury + d][curx] = 255
                queue.append([cury + d, curx])

            if curx >= d and mask_all[cury][curx - d] == 0 and compute_dist(curpixel, img[cury][curx - d]) <= dist_thresh:
                mask_all[cury][curx - d] = 255
                queue.append([cury, curx - d])

            if curx < IMG_SIZE - d and mask_all[cury][curx + d] == 0 and compute_dist(curpixel, img[cury][curx + d]) <= dist_thresh:
                mask_all[cury][curx + d] = 255
                queue.append([cury, curx + d])

        if len(queue) == 0:
            return

        cury, curx = queue.pop(0)
        curpixel = img[cury][curx]
        dist_thresh = get_threshold_dist(cury, curx)


def create_mask(img):
    mask_all = np.zeros((IMG_SIZE, IMG_SIZE))

    for y in range(CENTER_Y - CENTER_RADIUS, CENTER_Y + CENTER_RADIUS):
        for x in range(CENTER_X - CENTER_RADIUS, CENTER_X + CENTER_RADIUS):
            mask_all[y][x] = 255
            run_bfs(mask_all, y, x, img)

    return mask_all


def first_consecutive_2_whites(arr):
    arr_size = len(arr)
    cur_consecutive_whites = 0

    for i in range(arr_size):
        if arr[i] > 0:
            cur_consecutive_whites += 1
            if cur_consecutive_whites >= 2:
                return i - 1
        else:
            cur_consecutive_whites = 0

    return arr_size


def get_face_location_info(img_mask):
    from_top = img_mask[:, IMG_CENTER]
    from_left = img_mask[IMG_CENTER]
    from_right = from_left[::-1]

    face_loc_from_top = first_consecutive_2_whites(from_top)
    face_loc_from_left = first_consecutive_2_whites(from_left)
    face_loc_from_right = first_consecutive_2_whites(from_right)

    return {
        'face_location_top': [face_loc_from_top],
        'face_location_left': [face_loc_from_left],
        'face_location_right': [face_loc_from_right]
    }


def add_face_location_info(image_name, image_dir, face_location_info_df, face_location_info_file_path, save_mask=False):
    img_path = image_dir + '/' + image_name
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # (120, 120, 3)

    # create mask
    img_mask = create_mask(img)

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
            if idx % 100 == 0:
                print(image_dir, idx)

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


# result :
"""
column face_location_top, top 10% = 38, bottom 5% = 7
column face_location_left, top 10% = 42, bottom 5% = 21
column face_location_right, top 10% = 41, bottom 5% = 22
"""
if __name__ == '__main__':
    create_sample_face_location_info_dir()
    save_face_location_info_for_all_images()
