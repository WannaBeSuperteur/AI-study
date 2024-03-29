import numpy as np
import math
import cv2


# compute hue
def compute_hue(x, y):
    if x == 0:
        if y > 0:
            return 90.0
        else:
            return 270.0

    # 제1, 2 사분면
    elif y >= 0:
        return np.arctan2(y, x) * (180.0 / math.pi)

    # 제3, 4 사분면
    else:
        return (np.arctan2(y, x) + 2.0 * math.pi) * (180.0 / math.pi)
    

# create HSV array from image
# saturation is proportion to max(R, G, B) - min(R, G, B)
def create_hsv_image(image, coord_x, coord_y, img_size):
    hsv_array = np.zeros((img_size, img_size, 3))
    
    for i in range(img_size):
        for j in range(img_size):
            i_ = i // 8
            j_ = j // 8
            
            hue = compute_hue(coord_x[i_][j_], coord_y[i_][j_])
            saturation = math.sqrt(coord_x[i_][j_] * coord_x[i_][j_] + coord_y[i_][j_] * coord_y[i_][j_])
            brightness = image[0][i][j]

            hsv_array[i][j][0] = hue
            hsv_array[i][j][1] = saturation
            hsv_array[i][j][2] = 255.0 * brightness

    hsv_array = np.array(hsv_array).astype(np.float32)
    return hsv_array


# greyscale image에서 (center_y, center_x) 좌표의 픽셀을 중심으로 하는 그 주변의 영역 추출
# 단, 이때 추출된 영역의 크기는 가로, 세로 길이가 각각 img_size 임
# greyscale image의 범위를 벗어난 경우 0으로 처리
def create_image(img_size, greyscale_image, center_y, center_x):
    h = len(greyscale_image)
    w = len(greyscale_image[0])
    
    result = np.zeros((img_size, img_size))

    for i in range(img_size):
        y = center_y - img_size // 2 + i
        
        for j in range(img_size):
            x = center_x - img_size // 2 + j
            
            if y >= 0 and x >= 0 and y < h and x < w:
                result[i][j] = greyscale_image[y][x]

    return result


# create level 1, level 2 and level 3 image
def create_lv_1_2_3_images(image_size, color_map_size, greyscale_image, i_start, j_start):
    
    # lv1 (56 x 56 -> 14 x 14)
    lv1_image = create_image(
        img_size=image_size // 2,
        greyscale_image=greyscale_image,
        center_y=i_start + 4, center_x=j_start + 4
    )

    lv1_image_resized = cv2.resize(lv1_image, (color_map_size, color_map_size))

    # lv2 (28 x 28 -> 14 x 14)
    lv2_image = create_image(
        img_size=image_size // 4,
        greyscale_image=greyscale_image,
        center_y=i_start + 4, center_x=j_start + 4
    )

    lv2_image_resized = cv2.resize(lv2_image, (color_map_size, color_map_size))

    # lv3 (14 x 14)
    lv3_image = create_image(
        img_size=image_size // 8,
        greyscale_image=greyscale_image,
        center_y=i_start + 4, center_x=j_start + 4
    )
    
    return lv1_image_resized, lv2_image_resized, lv3_image


# compute positional values between 0 and 1
# y/x position, distance from center / each 4 corner points
def compute_positional_values(i, j, color_map_size):
    pos_max = color_map_size - 1

    # y/x position
    y_position = i / pos_max
    x_position = j / pos_max

    # each 4 corner                    
    dist_corner_max = math.sqrt(2 * pos_max * pos_max)

    dist_top_left = math.sqrt(i * i + j * j) / dist_corner_max
    dist_top_right = math.sqrt(i * i + (pos_max - j) * (pos_max - j)) / dist_corner_max
    dist_bottom_left = math.sqrt((pos_max - i) * (pos_max - i) + j * j) / dist_corner_max
    dist_bottom_right = math.sqrt((pos_max - i) * (pos_max - i) + (pos_max - j) * (pos_max - j)) / dist_corner_max

    # center
    pos_center = pos_max / 2.0
    dist_center_max = dist_corner_max / 2.0
    dist_center = math.sqrt((i - pos_center) * (i - pos_center) + (j - pos_center) * (j - pos_center)) / dist_center_max 

    return [y_position, x_position, dist_top_left, dist_top_right, dist_bottom_left, dist_bottom_right, dist_center]
