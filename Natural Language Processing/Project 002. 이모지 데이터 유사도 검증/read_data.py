import os
import pandas as pd

def read_data(is_save=True):
    f = open('train.txt', 'r')
    data = f.readlines()
    f.close()
    
    data = [x.split('\n')[0] for x in data]
    print(f'data size: {len(data)}')

    df = pd.DataFrame([])
    
    for line in data:
        sentence = ';'.join(line.split(';')[:-1])
        emoji = line.split(';')[-1]

        new_row = {'sentence': [sentence], 'emoji': [emoji]}
        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])

    if is_save:
        df.to_csv('train.csv')

    return df

if __name__ == '__main__':
    df = read_data()
    print(df)
