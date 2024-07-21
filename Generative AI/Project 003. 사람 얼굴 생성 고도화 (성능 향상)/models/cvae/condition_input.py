import pandas as pd


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