import os
import tensorflow as tf
import pandas as pd
from embedding_cbow import run_embedding


# 실행 시, "embedding_dataset.csv" 파일 필요
# 현재 시점 기준으로 window size = N = 4

if __name__ == '__main__':
    embedding_df = pd.read_csv('embedding_dataset.csv')
    emb_model = tf.keras.models.load_model('embedding_model')

    print('start')
    for _ in range(1000):
        test = run_embedding(0, emb_model.encoder)
    print('end')
