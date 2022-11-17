import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data/primekg/kg.csv', usecols=["x_name", "relation", "y_name"])
data = data[["x_name", "relation", "y_name"]]
data_train, data_rem = train_test_split(data, train_size=0.8)
data_valid, data_test = train_test_split(data_rem, train_size=0.5)

np.savetxt('Data/primekg/train.txt', data_train.values, fmt='%s\t%s\t%s')
np.savetxt('Data/primekg/valid.txt', data_valid.values, fmt='%s\t%s\t%s')
np.savetxt('Data/primekg/test.txt', data_test.values, fmt='%s\t%s\t%s')
