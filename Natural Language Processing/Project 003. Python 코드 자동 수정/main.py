import read_data as rd
import tokenizer
import pandas as pd
import embedding_cbow as ecbow
import os
from generate_dataset import get_or_create_main_model_train_data


# tokenize 하고, 그 텍스트를 공백으로 연결하여 반환
def tokenize_with_join(text):
    try:
        text = tokenizer.tokenize(text)
        return ' '.join(text)
    
    except:
        return None


# tokenize
def tokenize_converted_data():
    df = pd.read_csv('converted_data.csv', index_col=0)
    df['tokenized_code'] = df['code'].apply(lambda x: tokenize_with_join(x))
    df.to_csv('data_preprocessing_result.csv')


# Embedding Model 용으로, 데이터 읽고 model train 용 데이터 생성까지 진행
def generate_model_train_data():
    
    # Python code 원본 데이터 읽기
    rd.run_convert()

    # tokenize 실시
    tokenize_converted_data()

    # CBOW embedding 학습 실시
    ecbow.run_all_process()
    

if __name__ == '__main__':
    if 'embedding_dataset_for_cbow.csv' not in os.listdir():
        print('no embedding dataset for CBOW-like embedding model')
        generate_model_train_data()
        
    elif 'embedding_model' not in os.listdir():
        print('no embedding model')
        ecbow.train_cbow_like_model()

    else:
        print('embedding model prepared')

    # 메인 모델용 학습 데이터 로딩
    main_model_df = get_or_create_main_model_train_data()
    print(main_model_df)
