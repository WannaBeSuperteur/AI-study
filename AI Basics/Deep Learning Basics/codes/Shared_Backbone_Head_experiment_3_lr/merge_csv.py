import pandas as pd
import os


if __name__ == '__main__':
    csv_list = list(filter(lambda x: x.endswith('.csv'), os.listdir()))
    csv_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    merged_df_dict = {
        'learning_rate': [],
        'backbone': [],
        'case': [],
        'accuracy': []
    }
    column_names = ['backbone', 'case', 'accuracy']

    for csv_file in csv_list:
        df = pd.read_csv(csv_file)

        learning_rate = float(csv_file[:-4].split('_')[-1])
        merged_df_dict['learning_rate'] += [learning_rate for _ in range(len(df))]

        for column_name in column_names:
            merged_df_dict[column_name] += df[column_name].tolist()

    merged_df = pd.DataFrame(merged_df_dict)
    merged_df.to_csv('result_experiment_3_lr_merged.csv', index=False)

