import tensorflow as tf
import numpy as np
from PIL import Image

HIDDEN_DIMS = 40


# RGB <-> BGR
def convert_RGB_to_BGR(original_image):
    return original_image[:, :, ::-1]


# 생성된 cvae_decoder_model 모의 테스트
def mock_test_decoder(cvae_decoder):
    latent_space = np.random.normal(0.0, 1.0, size=(1, HIDDEN_DIMS))
    input_info = np.array([[0.000025, 0.999975, 0.99, 0.98, 0.99]])
    
    img = cvae_decoder([latent_space, input_info])
    img_np = np.array(img.numpy() * 255.0, dtype=np.uint8)

    print('result image (B,G,R) :')
    print(img_np)
    
    img_np_rgb = Image.fromarray(convert_RGB_to_BGR(img_np[0]))
    img_np_rgb.save(f'test_image_mock_test.png')


if __name__ == '__main__':
    
    # cvae_decoder_model 모의 테스트
    cvae_decoder = tf.keras.models.load_model('cvae_decoder_model')
    mock_test_decoder(cvae_decoder)
