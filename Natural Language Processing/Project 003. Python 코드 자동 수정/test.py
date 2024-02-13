import tensorflow as tf
from tokenizer import tokenize, split_with_spaces
from generate_dataset import get_embeddings_for_each_idx
from embedding_cbow import get_vocab


# 코드를 tokenize 된 형태로 분리 가능한 형태로 수정
def convert_code(lines):
    lines = [x.split('#')[0] for x in lines]
    
    for i in range(len(lines)):
        if lines[i][-1] != '\n':
            lines[i] += '\n'
            
    lines = ''.join(lines)
    lines = lines.replace('\t', '    ')
    return lines


# ['if', 'print', 'elif', ...] -> {'if':0, 'print':1, 'elif':2, ...}
def create_vocab_map(vocab):
    vocab_map = {}

    for i in range(len(vocab)):
        vocab_map[vocab[i]] = i

    return vocab_map


# 각 token 의 vocab index 별 embedding 계산
def compute_embedding(emb_model):
    return get_embeddings_for_each_idx(emb_model)


# 코드를 쭉 읽으면서, 중간에 누락된 부분 찾기
window_size = 4
embedding_size = 16

def run_test(vocab_map, embeddings, splitted_lines, tokenized_lines, main_model):
    print('\n\n(nl) means new line\n')
    n = len(splitted_lines)

    for i in range(n - 2 * window_size + 1):

        # 입력 범위 확인
        tokens = tokenized_lines[i : i + (2 * window_size)]
        originals = splitted_lines[i : i + (2 * window_size)]

        # 입력 데이터 생성
        test_input_embeddings = []
        for i in range(2 * window_size):
            vocab_idx = vocab_map[tokens[i]]
            test_input_embeddings += list(embeddings[vocab_idx])

        test_input_size = (2 * window_size - 1) * embedding_size
        for i in range(test_input_size):
            test_input_embeddings[i] = test_input_embeddings[i + embedding_size] - test_input_embeddings[i]

        test_input_embeddings = test_input_embeddings[:test_input_size]

        # 메인 모델을 통한 테스트 및 결과 출력
        test_result = main_model([test_input_embeddings])[0][0]

        if test_result >= 0.8:
            before = ' '.join(originals[:window_size])
            after = ' '.join(originals[window_size:2*window_size])
            print(f'Something should be between "{before}" and "{after}" with prob {round(float(test_result), 6)}.')
    

if __name__ == '__main__':
    f = open('python_code.txt', 'r')
    lines = convert_code(f.readlines())
    f.close()

    print(lines)

    # 코드 토큰화
    splitted_lines = split_with_spaces(lines)
    tokenized_lines = tokenize(lines)

    assert len(splitted_lines) == len(tokenized_lines)

    # 각 token 별 임베딩 계산
    emb_model = tf.keras.models.load_model('embedding_model')
    vocab = get_vocab()
    vocab_map = create_vocab_map(vocab)
    embeddings = compute_embedding(emb_model)

    # main model을 이용하여 테스트
    main_model = tf.keras.models.load_model('main_model')
    
    run_test(vocab_map, embeddings, splitted_lines, tokenized_lines, main_model)
