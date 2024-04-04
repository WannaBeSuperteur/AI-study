import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# modify file name if you want to test another csv file
file_name = 'regression_eyes_info_female.csv'

df = pd.read_csv(file_name, index_col=0)
df_arr = np.array(df).flatten()
mean_value = df_arr.mean()
mean_value_arr = np.full(len(df), mean_value)

print('\nfile name  :', file_name)
print('mean value :', mean_value)
print('mse        :', mean_squared_error(df_arr, mean_value_arr))
