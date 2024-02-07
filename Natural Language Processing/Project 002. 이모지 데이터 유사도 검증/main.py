import read_data as rd
import embed_text as emb
import pandas as pd


def read_dataset():
    try:
        df = pd.read_csv('train.csv', index_col=0)
    except:
        df = rd.read_data()
    return df


def add_embeddings(df, is_save=True):
    df_embed = pd.DataFrame(df)
    df_embed['embedding'] = df['sentence'].apply(lambda x: emb.embed_text(x))

    if is_save:
        df_embed.to_csv('train_embed.csv')

    return df_embed


if __name__ == '__main__':

    # 데이터셋 읽기
    df = read_dataset()
    print(df)

    # 문장 임베딩
    try:
        df_embed = pd.read_csv('train_embed.csv')
    except:
        df_embed = add_embeddings(df)
    print(df_embed)
