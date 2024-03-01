# 숫자 처리
def handle_numerics(line):
    tokens = line.split(' ')

    for i in range(len(tokens)):
        try:
            numeric = int(tokens[i])
            tokens[i] = '<number>'
        except:
            pass

    return ' '.join(tokens)


# 한 줄을 tokenize
def tokenize_line(line, ing_map, ly_map):

    # 소문자 처리
    line = line.lower()

    # 문장 부호 및 축약형 처리
    line = line.replace(',', ' ,').replace('.', ' .')
    line = line.replace('!', ' !').replace('?', ' ?')
    line = line.replace(' (', ' ( ').replace(') ', ' ) ')
    line = line.replace("'m", " 'm").replace("'re", " 're").replace("'s", " 's")
    line = line.replace("can't", "can 't").replace("n't", " n't").replace("'ve", " 've").replace("'d", " 'd")
    line = line.replace('"', '')

    # 자주 등장하는 -ing, -ly 처리
    for before, after in ing_map.items():
        line = line.replace(before, after)

    for before, after in ly_map.items():
        line = line.replace(before, after)

    # 숫자 처리
    line = handle_numerics(line)
    
    return line


def get_maps():
    ing_map = {
        'fascinating': 'fascinate -ing', 'coming': 'come -ing',
        'telling': 'tell -ing', 'dancing': 'dance -ing', 'interesting': 'interest -ing',
        'encouraging': 'encourage -ing', 'shopping': 'shop -ing',
        'having': 'have -ing', 'hiking': 'hike -ing', 'depressing': 'depress -ing',
        'going': 'go -ing', 'exciting': 'excite -ing', 'riding': 'ride -ing',
        'raining': 'rain -ing', 'surprising': 'surprise -ing', 'hunting': 'hunt -ing'
    }

    ly_map = {
        'really': 'real -ly', 'usually': 'usual -ly', 'deeply': 'deep -ly',
        'remarkably': 'remarkable -ly', 'oddly': 'odd -ly', 'actually': 'actual -ly',
        'especially': 'especial -ly', 'recently': 'recent -ly', 'probably': 'probable -ly'
    }

    return ing_map, ly_map


def tokenize(content):
    tokens = []
    ing_map, ly_map = get_maps()
    
    for line in content:

        # 인코딩 및 개행문자 분리
        line = line.encode('unicode-escape').decode('utf-8')
        line = ': '.join(line.split(': ')[1:]).split('\\n')[0]

        # tokenize 실시
        line = tokenize_line(line, ing_map, ly_map)

        tokens += (line.split(' ') + ['<person-change>'])

    return tokens


def tokenize_file_content(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    return tokenize(lines)


if __name__ == '__main__':
    tokens = tokenize_file_content('archive/human_chat.txt')
    print(f'total tokens          : {len(tokens)}')
    print(f'total tokens (unique) : {len(set(tokens))}')

    for idx, token in enumerate(tokens):
        if idx < 1000:
            print('index:', idx, 'token:', token)
