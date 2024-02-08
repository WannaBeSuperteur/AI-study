import read_data as rd
import embed_text as emb
import pandas as pd
import random
import os


def read_dataset():
    try:
        df = pd.read_csv('train.csv', index_col=0)
    except:
        df = rd.read_data()
    return df


def read_train_embed():
    return pd.read_csv('train_embed.csv', index_col=0)


# data에 embedding 컬럼 추가
def add_embeddings(df, is_save=True):
    df_embed = pd.DataFrame(df)
    df_embed['embedding'] = df['sentence'].apply(lambda x: emb.embed_text(x))

    if is_save:
        df_embed.to_csv('train_embed.csv')

    return read_train_embed()


# numpy.array 를 나타낸 텍스트를 list로 변환
def convert_to_list(text):
    text = text.replace('[ ', '[')
    text = text.replace(' ]', ']')
    text = text.replace('  ', ',').replace(' ', ',')
    text = text.replace(',,,', ',').replace(',,', ',')
    return eval(text)


# 임베딩된 문장 2개 추출 및 결과 저장
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


# 이모지 쌍별 cos 유사도 평균값 계산
def compute_avg_cos_sim():
    emoji_emb_df = pd.read_csv('emoji_and_embedding.csv', index_col=0)
    print(emoji_emb_df)

    emojis_set = set(list(emoji_emb_df['emoji_0']) + list(emoji_emb_df['emoji_1']))
    emojis = list(emojis_set)
    print(emojis)
    
    result = {}

    for idx, row in emoji_emb_df.iterrows():
        emoji_0 = row['emoji_0']
        emoji_1 = row['emoji_1']
        emoji_list = [emoji_0, emoji_1]
        emoji_list.sort()
        emoji_set = ' '.join(emoji_list)

        if emoji_set in result:
            result[emoji_set]['count'] += 1
            result[emoji_set]['total'] += row['cos_sim']
        else:
            result[emoji_set] = {'count': 1, 'total': row['cos_sim']}

    result_df = pd.DataFrame()

    for emoji_set in result:
        new_row = {
            'emoji_0': [emoji_set.split(' ')[0]],
            'emoji_1': [emoji_set.split(' ')[1]],
            'mean': [result[emoji_set]['total'] / result[emoji_set]['count']],
            'total': [result[emoji_set]['total']],
            'count': [result[emoji_set]['count']]
        }
        new_row = pd.DataFrame(new_row)
        result_df = pd.concat([result_df, new_row])

    result_df.sort_values(by=['emoji_0', 'emoji_1'], inplace=True)
    result_df.to_csv('avg_cos_sim.csv')
    

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
    if 'emoji_and_embedding.csv' not in os.listdir():
        run_experiment(df_embed)

    # 이모지 쌍별 cos 유사도 평균값 계산
    compute_avg_cos_sim()
