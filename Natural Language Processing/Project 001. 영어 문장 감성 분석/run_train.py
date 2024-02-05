import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math

from preprocess_data import read_data, add_age_of_user_numeric_col
import nltk


def load_data():
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')

    add_age_of_user_numeric_col(train_data)
    add_age_of_user_numeric_col(test_data)

    return train_data, test_data


def remove_unused_columns(train_data, test_data):
    train_data.drop(columns=['textID', 'selected_text', 'Age of User'], inplace=True)
    test_data.drop(columns=['textID', 'Age of User'], inplace=True)


def make_one_hot_columns(data, columns):
    for column in columns:
        vcs_top10 = data[column].value_counts(sort=True, ascending=False).head(10)
        unique_vals = list(vcs_top10.index)
        unique_vals.sort() # train, test에서 one-hot item 순서가 고정되게

        for unique_val in unique_vals:
            new_col = column + '_' + unique_val
            data[new_col] = data[column].apply(
                lambda x: 1 if x == unique_val else 0
            )

        data.drop(columns=[column], inplace=True)


def apply_normalize(data, column, apply_log=False):
    if apply_log:
        data[column] = data[column].apply(lambda x: np.log(x + 1))
    
    mean = data[column].mean()
    std = data[column].std()

    data[column] = data[column].apply(lambda x: (x - mean) / std)


def remove_website_addresses(text):
    words = text.split(' ')
    words = list(filter(lambda x: 'http://' not in x and 'https://' not in x, words))
    return ' '.join(words)


def preprocess_text(text):
    text = remove_website_addresses(text)
    
    tokens = nltk.word_tokenize(text)
    tags = nltk.tag.pos_tag(tokens)

    tags = list(filter(lambda x: x[1] not in ['(', ')', ':'], tags))
    tags = [tag[0] for tag in tags]
    preprocessed_text = ' '.join(tags)

    return preprocessed_text


def apply_preprocess_text(data):
    data['text'] = data['text'].apply(lambda x: preprocess_text(x))


def preprocess_before_training(train_data, test_data):
    add_age_of_user_numeric_col(train_data)
    add_age_of_user_numeric_col(test_data)

    # 불필요한 컬럼 제거
    remove_unused_columns(train_data, test_data)

    # one-hot 처리
    one_hot_cols = ['Time of Tweet', 'Country']
    make_one_hot_columns(train_data, one_hot_cols)
    make_one_hot_columns(test_data, one_hot_cols)

    # normalize 처리
    numeric_cols = ['Age of User Numeric']
    numeric_log_cols = ['Population -2020', 'Land Area (Km)', 'Density (P/Km)']

    for numeric_col in numeric_cols:
        apply_normalize(train_data, numeric_col, apply_log=False)
        apply_normalize(test_data, numeric_col, apply_log=False)

    for numeric_log_col in numeric_log_cols:
        apply_normalize(train_data, numeric_log_col, apply_log=True)
        apply_normalize(test_data, numeric_log_col, apply_log=True)

    # target column (sentiment) 처리
    sentiment_mapping = {'positive': 1.0, 'neutral': 0.5, 'negative': 0.0}
    train_data['sentiment'] = train_data['sentiment'].apply(lambda x: sentiment_mapping[x])
    test_data['sentiment'] = test_data['sentiment'].apply(lambda x: sentiment_mapping[x])


def preprocess_text_in_data(train_data, test_data):
    apply_preprocess_text(train_data)
    apply_preprocess_text(test_data)


if __name__ == '__main__':
    train_data, test_data = load_data()
    print(train_data)
    print(test_data)

    preprocess_before_training(train_data, test_data)
    preprocess_text_in_data(train_data, test_data)

    print('after preprocessed:')
    print(train_data)
    print(test_data)
