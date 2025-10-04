import pandas as pd
import os


if __name__ == '__main__':
    csv_list = list(filter(lambda x: x.endswith('.csv'), os.listdir()))
    csv_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    merged_df_dict = {
        'train_dataset_size': [],
        'backbone': [],
        'case': [],
        'accuracy': []
    }
    column_names = ['backbone', 'case', 'accuracy']

    for csv_file in csv_list:
        df = pd.read_csv(csv_file)

        train_dataset_size = int(csv_file.split('.')[0].split('_')[-1])
        merged_df_dict['train_dataset_size'] += [train_dataset_size for _ in range(len(df))]

        for column_name in column_names:
            merged_df_dict[column_name] += df[column_name].tolist()

    merged_df = pd.DataFrame(merged_df_dict)
    merged_df.to_csv('result_experiment_3_merged.csv', index=False)

