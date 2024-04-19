import tensorflow as tf
import numpy as np
from PIL import Image
import os

HIDDEN_DIMS = 152
NUM_IMGS_FOR_EACH_INFO = 3
BATCH_SIZE = 32


gender_to_img_name = {'male': 'M', 'female': 'F'}
hair_color_to_img_name = {'0.5': '5', '0.8': '8', '1.0': '1'}
mouth_to_img_name = {'0.0': '0', '0.5': '5', '1.0': '1'}
eyes_to_img_name = {'0.5': '5', '0.8': '8', '1.0': '1'}


# RGB <-> BGR
def convert_RGB_to_BGR(original_image):
    return original_image[:, :, ::-1]


# 생성된 cvae_decoder_model 테스트
def test_decoder(cvae_decoder, gender, hair_color, mouth, eyes, num):
    latent_space = np.random.normal(0.0, 1.0, size=(BATCH_SIZE, HIDDEN_DIMS))

    male_prob = 0.9999999 if gender == 'male' else 0.0000001
    female_prob = 0.9999999 if gender == 'female' else 0.0000001

    input_info_one = [male_prob, female_prob, float(hair_color), float(mouth), float(eyes),
                      np.random.uniform(),  # face location from top
                      np.random.uniform(),  # face location from left
                      np.random.uniform()]  # face location from right

    input_info = np.array([input_info_one for _ in range(BATCH_SIZE)])
    
    img = cvae_decoder([latent_space, input_info])
    img_np = np.array(img.numpy() * 255.0, dtype=np.uint8)
    img_np_rgb = Image.fromarray(convert_RGB_to_BGR(img_np[0]))

    img_name_gender = gender_to_img_name[gender]
    img_name_hair_color = hair_color_to_img_name[hair_color]
    img_name_mouth = mouth_to_img_name[mouth]
    img_name_eyes = eyes_to_img_name[eyes]

    img_name_info = img_name_gender + img_name_hair_color + img_name_mouth + img_name_eyes
    
    img_np_rgb.save(f'test_outputs/test_output_{img_name_info}_{num}.png')
    

# 모든 케이스에 대해 decoder model 테스트
def test_all_cases(cvae_decoder):
    for gender in ['male', 'female']:
        for hair_color in ['0.5', '0.8', '1.0']:
            for mouth in ['0.0', '0.5', '1.0']:
                for eyes in ['0.5', '0.8', '1.0']:
                    for num in range(NUM_IMGS_FOR_EACH_INFO):
                        test_decoder(cvae_decoder, gender, hair_color, mouth, eyes, num)


# test_outputs 디렉토리 생성
def create_test_outputs_dir():
    try:
        os.makedirs('test_outputs')
    except:
        pass


if __name__ == '__main__':
    cvae_decoder = tf.keras.models.load_model('cvae_decoder_model')
    create_test_outputs_dir()
    
    test_all_cases(cvae_decoder)
