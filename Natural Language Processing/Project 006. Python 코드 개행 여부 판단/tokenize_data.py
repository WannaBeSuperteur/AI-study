import pandas as pd
import numpy as np

INPUT_TOKENS = 10

import keyword

keywords = ['if', 'print', 'elif', 'else', 'for', 'in', 'while',
            'def', 'return', '+', '-', '*', '/', '%', '=', '>', '<', ':',
            '[', ']', '{', '}', '(', ')', ',',
            'range', 'try', 'except', 'break', 'continue', '(nl)',
            'True', 'False', 'and', 'or', 'not', 'class',
            'int', 'float', 'str', 'append', 'join', 'replace', 'split',
            'lambda', 'date', 'list', 'filter', 'split', 'set', 'dict',
            'numpy', 'np', 'pandas', 'pd', 'DataFrame', 'sum',
            'import', 'as', 'formatting_python', 'None'] + keyword.kwlist

keywords = list(set(keywords))


# code snippet 읽기
def read_code_snippets():
    code_snippets_data = pd.read_csv('code_snippets.csv', index_col=0)
    return code_snippets_data


# string을 tokenize 하는 함수
def tokenize_strings(text):
    string_ranges = []
    string_start = -1
    in_string = False
    current_str_mark = ' '

    str_tokens = {}

    for i in range(len(text)):
        if text[i] in ["'", '"']:
            if in_string and current_str_mark == text[i]:
                in_string = False
                current_str_mark = ' '
                string_ranges.append([string_start, i + 1])
                
            elif not in_string:
                in_string = True
                current_str_mark = text[i]
                string_start = i

    string_ranges = string_ranges[::-1]
    
    for string_range in string_ranges:
        string = text[string_range[0]+1:string_range[1]-1]
        if string not in str_tokens:
            str_tokens[string] = len(str_tokens) + 1
        replaced_string = f'mytxttkn{str_tokens[string]}'

        text = text[:string_range[0]] + replaced_string + text[string_range[1]:]

    return text


# 변수명을 var1, var2, ... 로 바꾸기
# 함수명을 func1, func2, ... 로 바꾸기
def convert_var_funcs(text_splitted):
    var_no = {}
    func_no = {}
    
    for i in range(len(text_splitted)):
        token = text_splitted[i]
        before_token = text_splitted[i-1]
        is_tokenized_text = (token[:8] == 'mytxttkn')
        
        if (token not in keywords) and not is_tokenized_text:

            # 함수명 바꾸기
            if before_token == 'def':
                if token not in func_no:
                    func_no[token] = len(func_no) + 1

                new_name = f'func{func_no[token]}'
                text_splitted[i] = new_name

            else:
                # 숫자 '0' -> (n) 으로 바꾸기
                if token == '0':
                    text_splitted[i] = '(n)'
                
                # 변수명 -> 변수명 바꾸기
                else:
                    if token not in var_no:
                        var_no[token] = len(var_no) + 1

                    new_name = f'var{var_no[token]}'
                    text_splitted[i] = new_name

    return ' '.join(text_splitted)


# 숫자 값을 '0' 으로 대체하기
def convert_nums(text_splitted):
    for i in range(len(text_splitted)):
        token = text_splitted[i]
        
        # 숫자임 -> '0' (임의의 숫자) 으로 바꾸기
        try:
            test = float(token)
            text_splitted[i] = '0'
        except:
            pass

    return ' '.join(text_splitted)


# tokenize할 각 요소들을 공백으로 구분
def split_with_spaces(text):
    text = text.replace(',', ' , ').replace('.', ' . ')
    text = text.replace('(', ' ( ').replace(')', ' ) ')
    text = text.replace('{', ' { ').replace('}', ' } ')
    text = text.replace('[', ' [ ').replace(']', ' ] ')
    text = text.replace(':', ' : ')
    text = text.replace('>=', ' >= ').replace('<=', ' <= ').replace('>', ' > ').replace('<', ' < ')
    text = text.replace('==', ' == ')
    text = text.replace('=', ' = ')
    text = text.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ').replace('%', ' % ')
    text = text.replace('f"', 'formatting_python "').replace("f'", "formatting_python '")

    text = text.replace('  ', ' ')
    text = text.replace('\n', '(nl)')
    
    return ' '.join(list(filter(lambda x: x != '', text.split(' '))))


# snippet 전처리
def preprocess_snippet(snippet):
    snippet = snippet.replace('\n', ' \n ')
    
    snippet = convert_nums(snippet.split(' '))
    snippet = split_with_spaces(snippet)
    snippet = tokenize_strings(snippet)
    snippet = convert_var_funcs(snippet.split(' '))
    
    return snippet


# tokenize된 line의 남는 공간을 '<null>' token으로 채우거나 truncate 하기
def convert_line_to_tokens(line_tokens, direction):

    # <null> token으로 채우기
    while len(line_tokens) < INPUT_TOKENS:
        if direction == 'start':
            line_tokens.append('<null>')
        else:
            line_tokens = ['<null>'] + line_tokens

    # truncate 하기
    if len(line_tokens) > INPUT_TOKENS:
        if direction == 'start':
            line_tokens = line_tokens[:INPUT_TOKENS]
        else:
            line_tokens = line_tokens[-INPUT_TOKENS:]

    return line_tokens


# 코드 전처리해서 입력 데이터셋으로 추가
def preprocess_code(code_snippet):
    input_ = []
    output_ = []

    # 코드 전처리
    try:
        preprocessed_snippet = preprocess_snippet(code_snippet)
    except:
        return None, None
    
    lines = preprocessed_snippet.split('(nl)')
    line_count = len(lines)

    for i in range(line_count):
        lines[i] = lines[i].strip()

    # 입력 데이터셋으로 추가
    for i in range(line_count - 3):
        before_line_1 = lines[i]
        before_line_2 = lines[i + 1]
        is_empty_line = None

        if lines[i + 2] == '' and i < line_count - 4:
            after_line_1 = lines[i + 3]
            after_line_2 = lines[i + 4]
            is_empty_line = True
            
        elif lines[i + 2] != '':
            after_line_1 = lines[i + 2]
            after_line_2 = lines[i + 3]
            is_empty_line = False
            
        # add to dataset
        if is_empty_line is not None:
            input_1 = convert_line_to_tokens(before_line_1.split(' '), direction='end')
            input_2 = convert_line_to_tokens(before_line_2.split(' '), direction='end')
            input_3 = convert_line_to_tokens(after_line_1.split(' '), direction='start')
            input_4 = convert_line_to_tokens(after_line_2.split(' '), direction='start')
            input_merged = input_1 + input_2 + input_3 + input_4

            output = [1.0] if is_empty_line else [0.0]

            input_.append(input_merged)
            output_.append(output)

    return input_, output_


# 데이터를 tokenize 하여 입력, 출력 데이터 추출
def tokenize_data(code_snippets):
    input_data = []
    output_data = []
    
    for idx, row in code_snippets.iterrows():
        code_snippet = row['code']
        input_, output_ = preprocess_code(code_snippet)

        if input_ is not None and output_ is not None:
            for i in range(len(input_)):
                input_data.append(input_[i])
                output_data.append(output_[i])

    return np.array(input_data), np.array(output_data)


# tokenize 된 데이터를 저장
def save_tokenized_data(input_data, output_data, file_name):
    all_data = np.concatenate([input_data, output_data], axis=1)
    all_data = pd.DataFrame(all_data)

    input_col_count = len(input_data[0])
    all_data.columns = [f'input_{x // INPUT_TOKENS}_{x % INPUT_TOKENS}' for x in range(input_col_count)] + ['output']

    print(f'\nentire data:\n{all_data}')

    all_data.to_csv(f'{file_name}.csv')


# train 시의 vocab set과 test 시의 vocab set을 matching 시키기 위해 vocab dic 저장
def save_vocab_dic(token_ids):
    result = pd.DataFrame()

    for token, idx in token_ids.items():
        new_row = {'token': [token], 'token_id': [idx]}
        new_row_df = pd.DataFrame(new_row)
        result = pd.concat([result, new_row_df])

    result.to_csv('vocab_map.csv')


# token ID로 변환 후 저장
def convert_into_token_id(input_data):
    token_ids = {}
    new_input_data = np.array(input_data)

    input_row_cnt = len(input_data)
    input_col_cnt = len(input_data[0])

    for i in range(input_row_cnt):
        for j in range(input_col_cnt):
            if input_data[i][j] not in token_ids:
                token_ids[input_data[i][j]] = len(token_ids)
                
            new_input_data[i][j] = token_ids[input_data[i][j]]

    # vocab dic 저장
    save_vocab_dic(token_ids)
            
    return new_input_data


if __name__ == '__main__':
    code_snippets = read_code_snippets()

    # tokenize 한 입력 데이터 저장
    input_data, output_data = tokenize_data(code_snippets)
    print(f'\ninput_data: {np.shape(input_data)}\n{input_data}')
    print(f'\noutput_data: {np.shape(output_data)}\n{output_data}')
    
    save_tokenized_data(input_data, output_data, file_name='train_data')

    # input data 를 token id 로 변환
    input_token_ids = convert_into_token_id(input_data)
    print(f'\ninput_token_ids: {np.shape(input_token_ids)}\n{input_token_ids}')
    
    save_tokenized_data(input_token_ids, output_data, file_name='train_data_token_id')

    
