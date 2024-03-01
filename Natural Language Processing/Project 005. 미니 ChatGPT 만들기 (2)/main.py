from generate_data import generate_data
from tokenize_data import tokenize_file_content
from train import run_all_process
import os

LIMIT = None # train_data.csv 에서 실제로 이용할 학습 데이터 개수 (None이면 모든 데이터를 학습)

if __name__ == '__main__':

    tokens = tokenize_file_content('archive/human_chat.txt')
    file_list = os.listdir()

    # 학습 데이터 생성
    if 'train_data.csv' not in file_list:
        print('creating train data ...')
        generate_data(tokens)

    # 메인 모델 학습
    if 'mini_chatgpt_model' not in file_list:
        print('training mini chatgpt model ...')
        run_all_process(limit=LIMIT)
