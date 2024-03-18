import pandas as pd


# 원본 데이터 읽기
def read_dataset():
    f = open('Python_code_data_modified.txt', 'r', encoding='utf-8')
    lines = [line.split('\n')[0] for line in f.readlines()]
    f.close()

    return lines


# 변환: 데이터를 각 코드별로 저장
def convert_into_data(lines):
    df = pd.DataFrame()
    code = ''

    for idx, line in enumerate(lines):
        is_code_finished_just_before = line.startswith('# Write') or line.startswith('# write') or idx == len(lines) - 1
        
        if idx > 0 and is_code_finished_just_before:
            code_df = pd.DataFrame({'code': [code]})
            df = pd.concat([df, code_df])
            code = ''
            
        else:
            except_comment = line.split('#')[0]
            is_importing = line.startswith('from ') or line.startswith('import ')
            
            if except_comment != '' and not is_importing:
                code += except_comment + '\n'

    df.to_csv('code_snippets.csv')


# 전체 실행
def run_convert():
    lines = read_dataset()
    convert_into_data(lines)


if __name__ == '__main__':
    run_convert()
