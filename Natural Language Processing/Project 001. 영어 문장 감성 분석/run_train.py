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


if __name__ == '__main__':
    train_data, test_data = load_data()

    print(train_data)
    print(test_data)

    apply_preprocess_text(train_data)
    apply_preprocess_text(test_data)

    print('after preprocessed:')
    print(train_data)
    print(test_data)
