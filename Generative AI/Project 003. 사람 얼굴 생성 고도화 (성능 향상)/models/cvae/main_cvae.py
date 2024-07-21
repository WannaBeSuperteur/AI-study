import tensorflow as tf
import numpy as np
import pandas as pd

from condition_input import merge_conditional_input_data
from cvae_model_utils import create_train_and_valid_data
from cvae_model_architecture import TRAIN_DATA_LIMIT
from cvae_model_architecture import train_cvae_model


def train_cvae():

    """
    Train NVAE-idea-based CVAE model
    """

    tf.compat.v1.disable_eager_execution()
    np.set_printoptions(suppress=True, linewidth=160)
    pd.set_option('display.max_columns', 16)

    # 학습 데이터 추출 (이미지 input + 해당 이미지의 class)
    train_input, train_info = create_train_and_valid_data(limit=TRAIN_DATA_LIMIT)

    print(f'\nshape of train input: {np.shape(train_input)}')
    print(f'\nshape of train info: {np.shape(train_info)}')

    # 학습 실시 및 모델 저장
    train_cvae_model(train_input, train_info)


if __name__ == '__main__':
    merge_conditional_input_data()
    train_cvae()