from generate_data import generate_data
from tokenize_data import tokenize_file_content
import train_embedding_model
import train_main_model
import os

LIMIT = None # train_data.csv 에서 실제로 이용할 학습 데이터 개수 (None이면 모든 데이터를 학습)

if __name__ == '__main__':

    tokens = tokenize_file_content('archive/human_chat.txt')
    file_list = os.listdir()

    # 학습 데이터 생성
    if 'train_data.csv' not in file_list:
        print('creating train data ...')
        generate_data(tokens)

    # 임베딩 모델 학습
    if 'embedding_model' not in file_list:
        print('training embedding model ...')
        train_embedding_model.run_all_process()

    # 메인 모델 학습
    if 'main_model' not in file_list:
        print('training main model ...')
        train_main_model.run_all_process(limit=LIMIT)
