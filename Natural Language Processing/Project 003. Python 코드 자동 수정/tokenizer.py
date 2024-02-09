import pandas as pd

keywords = ['if', 'print', 'elif', 'else', 'for', 'in', 'while',
            'def', 'return', '+', '-', '*', '/', '%', '=', '>', '<', ':',
            '[', ']', '{', '}', '(', ')', ',',
            'range', 'try', 'except', 'break', 'continue', '(nl)',
            'True', 'False', 'and', 'or', 'not', 'class',
            'int', 'float', 'str', 'append', 'join', 'replace', 'split',
            'lambda', 'date', 'list', 'filter', 'split', 'set', 'dict',
            'numpy', 'np', 'pandas', 'pd', 'DataFrame', 'sum']


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
    text = text.replace('\n', ' (nl) ')

    text = text.replace('  ', ' ')
    
    return list(filter(lambda x: x != '', text.split(' ')))


# 변수명을 var1, var2, ... 로 바꾸기
# 함수명을 func1, func2, ... 로 바꾸기
# 숫자 값을 (n) 으로 대체하기
def convert_var_func_nums(text_splitted):
    var_no = {}
    func_no = {}
    
    for i in range(1, len(text_splitted)):
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
                # 숫자임 -> (n) 으로 바꾸기
                try:
                    test = int(token)
                    text_splitted[i] = '(n)'

                # 숫자가 아님 -> 변수명 -> 변수명 바꾸기
                except:
                    if token not in var_no:
                        var_no[token] = len(var_no) + 1

                    new_name = f'var{var_no[token]}'
                    text_splitted[i] = new_name


def tokenize(text):

    # string을 먼저 tokenize 하기
    text = tokenize_strings(text)

    # 공백으로 구분하기
    text = text.replace('\t', '').replace('    ', '')
    text_splitted = split_with_spaces(text)

    # 변수명, 함수명, 숫자 값 바꾸기
    convert_var_func_nums(text_splitted)
    
    return text_splitted


# tokenize 결과 저장
def save_tokenized_texts(splitted_texts, csv_name):
    df = pd.DataFrame()

    for splitted_text in splitted_texts:
        text = ' '.join(splitted_text)
        new_row = pd.DataFrame({'text': [text]})
        df = pd.concat([df, new_row])

    df.to_csv(f'{csv_name}.csv')


if __name__ == '__main__':
    text = """
    a = 12
    b = 50

    def swap(v1, v2):
        temp = v1
        v1 = v2
        v2 = temp

        return [v2, v1]

    swap_test = swap(a, b)
    print(swap_test)

    c = a + b

    if c >= 100:
        d = c / (a + b)
    else:
        d = c * (a - b)

    text1 = 'test text'
    text2 = "test text"
    text3 = '"3" + "1" = "31"'
    text4 = "'3' + '1' = '31'"
    """

    splitted_text = tokenize(text)
    print(splitted_text)
    save_tokenized_texts([splitted_text], 'data_preprocessing_result_test')
