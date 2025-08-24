
import cv2


# 이미지 획득 (흑백 변환)
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - img_path  (str) : 획득 (흑백 변환) 대상 이미지 경로
# - save_path (str) : 흑백 변환 이미지 저장 경로

# Output:
# - save_path 에 흑백으로 변환된 이미지 저장됨

def acquire_image(img_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(save_path, img_thresholded[1])


if __name__ == '__main__':
    acquire_image(img_path='scanned_images_dataset/train/Letter/508146503+-6507.jpg', save_path='test_black_white.png')
