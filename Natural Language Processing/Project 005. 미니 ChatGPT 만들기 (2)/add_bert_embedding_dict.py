import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from embedding_helper import get_token_ids, cos_sim

# 본 프로젝트를 응용한 실제 서비스 시,
# 아래 모델을 사용자가 직접 다운받는 대신 bert_embedding_dict.csv 파일만을 이용할 수 있어야 함
bert_pretrained_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# BERT Embedding 계산
def get_bert_embedding(token):
    tokens = bert_tokenizer(token, return_tensors="pt", add_special_tokens=True)
    output = bert_pretrained_model(**tokens)
    hidden_states = output.last_hidden_state
    return hidden_states[:, 1, :].detach().numpy()


# BERT Embedding 목록을 bert_embedding_dict.csv 파일로 저장
def save_bert_embedding_dict(bert_embedding_dict):
    df = pd.DataFrame([])

    for token, embedding in bert_embedding_dict.items():
        if len(df) % 100 == 0:
            print(len(df))

        # add new row to DataFrame
        new_row = {'token': [token]}

        for i, e in enumerate(embedding[0]):
            new_row[f'emb_{i}'] = [e]

        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])

    df.to_csv('bert_embedding_dict.csv')
    

# 가장 가까운 BERT Embedding 찾기
def find_nearest_bert_embedding(token):
    token_embedding = get_bert_embedding(token)
    embedding_dict_df = pd.read_csv('bert_embedding_dict.csv', index_col=0)

    min_dist = 1000000.0
    min_dist_token = None

    for i in range(len(embedding_dict_df)):
        embedding = embedding_dict_df.iloc[i]['emb_0':]
        embedding = np.array(embedding.to_list())
        distance = np.linalg.norm(token_embedding - embedding) # Euclidean distance
        
        if distance < min_dist:
            min_dist = distance
            min_dist_token = embedding_dict_df.iloc[i]['token']

    return min_dist_token


# 가장 가까운 BERT Embedding 순위 출력 (token 이 주어졌을 때)
# 참고: Pandas Dataframe 대신 Numpy array를 이용하면 속도가 훨씬 빨라진다?
def find_nearest_bert_embedding_rank_for_token(token, bert_embedding_dict_np, embed_limit=128, verbose=False):
    token_embedding = get_bert_embedding(token)[0][:embed_limit]
    
    return find_nearest_bert_embedding_rank_for_embvec(
        embedding_vector=token_embedding,
        bert_embedding_dict_np=bert_embedding_dict_np,
        embed_limit=embed_limit,
        verbose=verbose
    )


# 가장 가까운 BERT Embedding 순위 출력 (embedding vector 가 주어졌을 때)
def find_nearest_bert_embedding_rank_for_embvec(embedding_vector, bert_embedding_dict_np, embed_limit=128, verbose=False, weight=None):
    result = []
    
    for i in range(len(bert_embedding_dict_np)):
        tkn = bert_embedding_dict_np[i][0]
        embedding = bert_embedding_dict_np[i][1:embed_limit+1]
        
        cos_similarity = cos_sim(embedding_vector, embedding, weight=weight) # Cosine similarity with element-wise weight
        result.append([tkn, cos_similarity ** 4.0])

    result.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f'10 nearest tokens based on BERT embedding (limit: {embed_limit}) :')
        for i in range(11):
            print(f'rank {i} : {result[i]}')

    return result


# BERT Embedding table 파일에서 token의 임베딩 찾기
def find_embedding_from_table(token, token_ids, bert_embedding_dict_np, embed_limit=128):
    token_id = token_ids[token]
    return bert_embedding_dict_np[token_id][1:embed_limit+1]


# BERT Embedding 목록 생성
def create_bert_embedding_dict(token_ids):
    bert_embedding_dict = {}
    cnt = 0
    
    for token, _ in token_ids.items():
        if cnt in [1, 3, 5, 10, 30, 50] or cnt % 100 == 0:
            print(cnt)
            
        bert_embedding_dict[token] = get_bert_embedding(token)
        cnt += 1

    return bert_embedding_dict


# BERT embedding table 생성의 전체 프로세스 실행
def run_all_process_bert_embedding_table():
    token_ids = get_token_ids()
    bert_embedding_dict = create_bert_embedding_dict(token_ids)
    save_bert_embedding_dict(bert_embedding_dict)

    # test
    test_tokens = ['review', 'snowman', 'four', 'five', 'six', 'seven']

    for test_token in test_tokens:
        nearest_token = find_nearest_bert_embedding(test_token)
        print(f'nearest token of {test_token} : {nearest_token}')


if __name__ == '__main__':
    run_all_process_bert_embedding_table()
