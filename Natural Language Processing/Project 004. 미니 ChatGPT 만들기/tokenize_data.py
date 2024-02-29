def tokenize(content):
    tokens = []
    
    for line in content:

        # 인코딩 및 개행문자 분리
        line = line.encode('unicode-escape').decode('utf-8')
        line = ': '.join(line.split(': ')[1:]).split('\\n')[0]

        # 문장 부호 및 축약형 처리
        line = line.replace(',', ' ,').replace('.', ' .')
        line = line.replace('!', ' !').replace('?', ' ?')
        line = line.replace("'m", " 'm").replace("'re", " 're").replace("'s", " 's")
        line = line.replace("'t", " 't").replace("'ve", " 've").replace("'d", " 'd")
        line = line.replace('"', '')

        # 소문자 처리
        line = line.lower()

        tokens += (line.split(' ') + ['<Person-Change>'])

    return tokens


def tokenize_file_content(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    return tokenize(lines)


if __name__ == '__main__':
    tokens = tokenize_file_content('archive/human_chat.txt')
    print(f'total tokens : {len(tokens)}')
    print(f'first 200 tokens :\n{tokens[:200]}')
