import pandas as pd
import numpy as np
import math
import tensorflow as tf

from run_train import define_data
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    train_data = pd.read_csv('train_data_final.csv', index_col=0)
    test_data = pd.read_csv('test_data_final.csv',index_col=0)
    (_, _, _, _, test_input, test_gt) = define_data(train_data, test_data)

    test_gt = np.array(test_gt)
    
    model = tf.keras.models.load_model('sentiment_model')
    test_output = model.predict(test_input)

    print('test input:')
    print(test_input)
    print('\ntest output:')
    print(test_output)
    print('\ntest ground truth:')
    print(test_gt)

    print(f'mean squared error: {mean_squared_error(test_output, test_gt)}')
    print(f'mean absolute error: {mean_absolute_error(test_output, test_gt)}')
    print(f'corrcoef: {np.corrcoef(test_output.flatten(), test_gt.flatten())[0][1]}')
    
