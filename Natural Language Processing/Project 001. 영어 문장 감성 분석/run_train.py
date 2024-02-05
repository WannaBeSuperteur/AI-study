import pandas as pd
import numpy as np
import math

from preprocess_data import read_data, add_age_of_user_numeric_col


def load_data():
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')

    add_age_of_user_numeric_col(train_data)
    add_age_of_user_numeric_col(test_data)

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = load_data()

    print(train_data)
    print(test_data)
