from augment_data import run_augmentation
import test
import train
import cv2
import os
import numpy as np

RESIZE_DEST = 128


# 학습 데이터 로딩
def load_train_data():
    train_images_name = os.listdir('images/train')
    train_input = []
    train_output = []
    valid_input = []
    valid_output = []

    for idx, name in enumerate(train_images_name):
        if idx % 500 == 0:
            print(idx)
            
        img = cv2.imread('images/train/' + name, cv2.IMREAD_UNCHANGED)
        original_img_id = int(name.split('.')[0].split('_')[1])

        # apple -> [1, 0], tomato -> [0, 1]
        if 'apples' in name:
            if original_img_id < 132:
                train_input.append(np.array(img) / 255.0)
                train_output.append([1, 0])
            else:
                valid_input.append(np.array(img) / 255.0)
                valid_output.append([1, 0])
            
        elif 'tomatoes' in name:
            if original_img_id < 104:
                train_input.append(np.array(img) / 255.0)
                train_output.append([0, 1])
            else:
                valid_input.append(np.array(img) / 255.0)
                valid_output.append([0, 1])

    train_input_return = np.array(train_input)
    train_output_return = np.array(train_output)
    valid_input_return = np.array(valid_input)
    valid_output_return = np.array(valid_output)

    print(f'shape of train input : {np.shape(train_input_return)}')
    print(f'shape of train output : {np.shape(train_output_return)}')
    print(f'shape of valid input : {np.shape(valid_input_return)}')
    print(f'shape of valid output : {np.shape(valid_output_return)}')
    
    return train_input_return, train_output_return, valid_input_return, valid_output_return


# 테스트 데이터 로딩
def load_test_data():
    test_apples_name = os.listdir('archive/test/apples')
    test_tomatoes_name = os.listdir('archive/test/tomatoes')
    test_input = []
    test_output = []

    for name in test_apples_name:
        apple_img = cv2.imread('archive/test/apples/' + name, cv2.IMREAD_UNCHANGED)
        apple_img = cv2.resize(apple_img, dsize=(RESIZE_DEST, RESIZE_DEST))
        
        test_input.append(np.array(apple_img) / 255.0)
        test_output.append([1, 0])

    for name in test_tomatoes_name:
        tomato_img = cv2.imread('archive/test/tomatoes/' + name, cv2.IMREAD_UNCHANGED)
        tomato_img = cv2.resize(tomato_img, dsize=(RESIZE_DEST, RESIZE_DEST))
        
        test_input.append(np.array(tomato_img) / 255.0)
        test_output.append([0, 1])

    test_input_return = np.array(test_input)
    test_output_return = np.array(test_output)

    print(f'shape of test input : {np.shape(test_input_return)}')
    print(f'shape of test output : {np.shape(test_output_return)}')
    
    return test_input_return, test_output_return


if __name__ == '__main__':

    # data augmentation 실시
    run_augmentation()

    # 학습, validation 및 테스트 데이터 로딩
    train_input, train_output, valid_input, valid_output = load_train_data()
    test_input, test_output = load_test_data()

    
