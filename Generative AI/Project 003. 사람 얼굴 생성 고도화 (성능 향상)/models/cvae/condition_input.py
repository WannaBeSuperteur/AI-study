# TBU

import pandas as pd
import numpy as np

input_types = ['background', 'eyes', 'hair_color', 'head', 'mouth']


def merge_conditional_input_data():

    """
    File Input:
        models/data/all_output_background.csv : 'background' value for all final training images (for CVAE) by "Input Decision Model"
        models/data/all_output_eyes.csv       : 'eyes' value for all the final training images
        models/data/all_output_hair_color.csv : 'hair_color' value for all the final training images
        models/data/all_output_head.csv       : 'head' value for all the final training images
        models/data/all_output_mouth.csv      : 'mouth' value for all the final training images

    File Output:
        models/data/all_output.csv : merged all data above
    """

    merged_df = pd.DataFrame()

    for idx, input_type in enumerate(input_types):
        df = pd.read_csv(f'models/data/all_output_{input_type}.csv')

        if idx == 0:
            merged_df['image_paths'] = df['image_path']
        merged_df[input_type] = df[input_type]

    merged_df.to_csv('models/data/all_output.csv', index=False)

def get_conditional_input_data_from_merged_csv():
    """
    File Input:
        models/data/all_output.csv : merged all data above (by merge_conditional_input_data())

    Output:
        image_paths  (Pandas Series)    : image paths for all final training images (for CVAE)
        input_values (Pandas DataFrame) : conditional input values for all final training images
    """

    merged_df = pd.read_csv('models/data/all_output.csv')
    image_paths = merged_df['image_paths']
    input_values = merged_df[input_types]

    return image_paths, input_values


def get_conditional_input_data():
    merge_conditional_input_data()
    return get_conditional_input_data_from_merged_csv()