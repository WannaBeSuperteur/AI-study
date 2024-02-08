import read_data as rd
import embed_text as emb
import pandas as pd
import random


def read_dataset():
    try:
        df = pd.read_csv('train.csv', index_col=0)
    except:
        df = rd.read_data()
    return df


def read_train_embed():
    return pd.read_csv('train_embed.csv', index_col=0)


def add_embeddings(df, is_save=True):
    df_embed = pd.DataFrame(df)
    df_embed['embedding'] = df['sentence'].apply(lambda x: emb.embed_text(x))

    if is_save:
        df_embed.to_csv('train_embed.csv')

    return read_train_embed()


def convert_to_list(text):
    text = text.replace('[ ', '[')
    text = text.replace(' ]', ']')
    text = text.replace('  ', ',').replace(' ', ',')
    text = text.replace(',,,', ',').replace(',,', ',')
    return eval(text)


def run_experiment(df_embed, N=10000):
    rows = len(df_embed)
    result = pd.DataFrame()

    for experiment in range(N):
        if experiment % 500 == 0:
            print(experiment)

        # 문장을 추출할 row index 찾기
        while True:
            row_idx_0 = random.randint(0, rows-1)
            row_idx_1 = random.randint(0, rows-1)

            if row_idx_0 != row_idx_1:
                break

        emb_0 = convert_to_list(df_embed.iloc[row_idx_0]['embedding'])
        emb_1 = convert_to_list(df_embed.iloc[row_idx_1]['embedding'])
        cos_sim = emb.compute_cos_sim_arr(emb_0, emb_1)

        emoji_0 = df_embed.iloc[row_idx_0]['emoji']
        emoji_1 = df_embed.iloc[row_idx_1]['emoji']

        new_row = {'emoji_0': [emoji_0], 'emoji_1': [emoji_1], 'cos_sim': [cos_sim]}
        new_row = pd.DataFrame(new_row)
        result = pd.concat([result, new_row])

    result.to_csv('emoji_and_embedding.csv')
    

if __name__ == '__main__':

    # 데이터셋 읽기
    df = read_dataset()
    print(df)

    # 문장 임베딩
    try:
        df_embed = read_train_embed()
    except:
        df_embed = add_embeddings(df)
    print(df_embed)

    # 임베딩된 문장 2개 추출 및 결과 저장
    run_experiment(df_embed)
