import numpy as np
import math


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
