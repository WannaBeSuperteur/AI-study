import pandas as pd
import numpy as np
import math


# csv 파일 읽는 함수
def read_data(name):
    data = pd.read_csv(name, encoding='utf-8', encoding_errors='ignore')

    # column 목록 출력
    print(f'columns:\n{list(data.columns)}\n')
    
    return data.dropna()


# Age of User를 중앙값 (단, 0~20세는 15세, 70~100세는 80세) 을 이용하여 숫자로 변환
def convert_age_of_user(x):
    if x == '0-20':
        return 15

    elif x == '21-30':
        return 25

    elif x == '31-45':
        return 38

    elif x == '46-60':
        return 53

    elif x == '60-70':
        return 65

    elif x == '70-100':
        return 80


# 데이터의 Age of User의 Numeric 컬럼 추가
def add_age_of_user_numeric_col(data):
    data['Age of User Numeric'] = data['Age of User'].apply(
        lambda x: convert_age_of_user(x)
    )
    

# 데이터의 각 column의 기본 정보 (row 개수, null 개수 등) 출력
def print_basic_info(data, cols):
    for col in cols:
        dtype = data[col].dtypes
        
        print(f'\n[ column: {col} ]')
        print(f'row count  : {len(data[col])}')
        print(f'null count : {data[col].isnull().sum()}')
        print(f'data type  : {dtype}')

        if str(dtype) in ['int64', 'float64']:
            print(f'max        : {data[col].max()}')
            print(f'min        : {data[col].min()}')
            print(f'mean       : {data[col].mean()}')
            print(f'std        : {data[col].std()}')


# 데이터의 각 column의 각 one-hot class 별 데이터 개수 출력
def print_data_count(data, cols):
    for col in cols:
        print(f'\nvalue count of column {col} :')
        print(data[col].value_counts(sort=True, ascending=False))


# min, max 값을 이용하여 bin 만들기
def get_bins(numbers, is_log=False, bin_count=20):
    max_n = max(numbers)
    min_n = max(min(numbers), 0.01) # prevent divide-by-0 error
    
    if is_log:
        interval = pow(max_n / min_n, 1 / bin_count)
        result = [min_n * pow(interval, i) for i in range(bin_count + 1)]
    else:
        interval = (max_n - min_n) / bin_count
        result = [min_n + i * interval for i in range(bin_count + 1)]

    return result


# 데이터의 각 column의 값 분포 출력
def print_distribution(data, cols, is_log=False, bin_count=20):
    for col in cols:
        nums = data[col]
        bins = get_bins(nums.tolist(), is_log, bin_count)

        print(f'\nvalue distribution of column {col} :')
        for i in range(bin_count):
            min_n = bins[i]
            max_n = bins[i + 1]
            between_values = nums.between(min_n, max_n, inclusive='both').sum()
            print(f'between {round(min_n, 2)} and {round(max_n, 2)}: {between_values}')


# 데이터의 각 column의 상관계수 출력
def print_corr_coef(data, cols):
    print(f'\ncorrcoef between columns {cols} :')
    print(data[cols].corr())
    print('\n')


# EDA 하는 함수
def do_eda(data):

    # 나이를 숫자 값으로 변환한 컬럼 추가
    add_age_of_user_numeric_col(data)

    # 컬럼 지정
    one_hot_cols = ['Time of Tweet', 'Age of User', 'Country']
    numeric_cols = ['Age of User Numeric']
    numeric_log_cols = ['Population -2020', 'Land Area (Km)', 'Density (P/Km)']

    # EDA 대상 컬럼 :
    # sentiment, Time of Tweet, Age of User, Country, Population, Land Area, Density    
    eda_cols = one_hot_cols + numeric_cols

    # corr-coef 출력 대상 컬럼 :
    # Age of User, Population, Land Area, Density
    corr_coef_cols = numeric_cols + numeric_log_cols
    
    # 기본 정보 출력
    print_basic_info(data, eda_cols)
    
    # one-hot column들의 각 one-hot class 별 데이터 개수 출력
    print_data_count(data, one_hot_cols)

    # numeric column들의 값 분포 출력
    print_distribution(data, numeric_cols, is_log=False)
    print_distribution(data, numeric_log_cols, is_log=True)

    # numeric column들의 상관계수 출력
    print_corr_coef(data, corr_coef_cols)


if __name__ == '__main__':

    # 데이터 불러오고 출력하기
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')

    print(train_data)
    print(test_data)

    # 학습 데이터에 대해 EDA 실시
    do_eda(train_data)

    # 테스트 데이터의 Age of User의 Numeric 컬럼 추가
    add_age_of_user_numeric_col(train_data)
    add_age_of_user_numeric_col(test_data)

    print(train_data)
    print(test_data)
