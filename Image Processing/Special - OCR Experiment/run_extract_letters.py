
import cv2
import numpy as np
import os


# 각 글자 영역의 top, bottom, left, right 좌표 계산 (BFS 기반 탐색)
# Create Date : 2025.08.26
# Last Update Date : -

# Arguments:
# - img     (NumPy array) : 각각의 글자 영역 추출 대상 이미지
# - start_y (int)         : 탐색 시작 y 좌표
# - start_x (int)         : 탐색 시작 x 좌표
# - visited (NumPy array) : BFS 기반 탐색용 방문 여부 체크 배열

# Returns:
# - location_info (dict) : {'top': int, 'bottom': int, 'left': int, 'right': int} 형식의 픽셀 좌표

def compute_boundary_of_letter(img, start_y, start_x, visited):
    current_y = start_y
    current_x = start_x
    queue = []
    location_info = {'top': current_y, 'bottom': current_y, 'left': current_x, 'right': current_x}

    while True:
        location_info['top'] = min(location_info['top'], current_y)
        location_info['bottom'] = max(location_info['bottom'], current_y)
        location_info['left'] = min(location_info['left'], current_x)
        location_info['right'] = max(location_info['right'], current_x)

        if not visited[current_y - 1][current_x] and current_y > 0 and img[current_y - 1][current_x] == 0:
            queue.append([current_y - 1, current_x])
            visited[current_y - 1][current_x] = True

        if not visited[current_y + 1][current_x] and current_y < np.shape(img)[0] and img[current_y + 1][current_x] == 0:
            queue.append([current_y + 1, current_x])
            visited[current_y + 1][current_x] = True

        if not visited[current_y][current_x - 1] and current_x > 0 and img[current_y][current_x - 1] == 0:
            queue.append([current_y, current_x - 1])
            visited[current_y][current_x - 1] = True

        if not visited[current_y][current_x + 1] and current_x < np.shape(img)[1] and img[current_y][current_x + 1] == 0:
            queue.append([current_y, current_x + 1])
            visited[current_y][current_x + 1] = True

        if len(queue) == 0:
            break

        next_pixel = queue.pop(0)
        current_y = next_pixel[0]
        current_x = next_pixel[1]

    return location_info


# 흑백 이미지로부터 각각의 글자 영역 추출
# Create Date : 2025.08.26
# Last Update Date : -

# Arguments:
# - img_path (str) : 각각의 글자 영역 추출 대상 이미지 경로

# Returns:
# - extracted_letters (data type TBU) : 추출된 글자들의 집합 (왼쪽 위부터 좌표 순서대로)

def extract_letters(img_path):
    extracted_letters_location_info = []
    extracted_letters = []

    # read image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(img)

    height = np.shape(img)[0]
    width = np.shape(img)[1]
    visited = np.zeros_like(img)

    # extract letter info
    for y in range(height):
        for x in range(width):
            if img[y][x] == 0 and not visited[y][x]:
                print(f'extracting : y={y}, x={x}')
                loc = compute_boundary_of_letter(img, y, x, visited)

                letter_height = loc['bottom'] - loc['top'] + 1
                letter_width = loc['right'] - loc['left'] + 1

                if 13 <= letter_height <= 50 and 10 <= letter_width <= 40 and 0.5 <= letter_width / letter_height <= 1:
                    letter_area = img[loc['top']:loc['bottom'] + 1, loc['left']:loc['right'] + 1]
                    location_info_dict = {
                        'letter_area': letter_area,
                        'location_info': loc
                    }
                    extracted_letters_location_info.append(location_info_dict)

    # sort letter info
    extracted_letters_location_info.sort(key=lambda x: x['location_info']['left'])
    extracted_letters_location_info.sort(key=lambda x: x['location_info']['top'])

    for loc in extracted_letters_location_info:
        extracted_letters.append(loc['letter_area'])

    return extracted_letters


if __name__ == '__main__':
    extracted_letters = extract_letters('test_black_white.png')

    os.makedirs('test_extract_letter', exist_ok=True)
    for idx, letter in enumerate(extracted_letters):
        letter_uint8 = letter.astype(np.uint8)
        cv2.imwrite(f'test_extract_letter/letter_{idx:04d}.png', letter_uint8)
