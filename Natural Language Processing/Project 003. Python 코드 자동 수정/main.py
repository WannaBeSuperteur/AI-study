import read_data as rd
import tokenizer
import pandas as pd
import embedding_cbow as ecbow
import os


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


# 데이터 읽고 model train 용 데이터 생성까지 진행
def generate_model_train_data():
    
    # Python code 원본 데이터 읽기
    rd.run_convert()

    # tokenize 실시
    tokenize_converted_data()

    # CBOW embedding 학습 실시
    ecbow.run_all_process()
    

if __name__ == '__main__':
    if 'embedding_dataset_for_cbow.csv' not in os.listdir():
        generate_model_train_data()
    else:
        ecbow.train_cbow_like_model()
    
