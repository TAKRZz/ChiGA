# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/14 21:01
# @FileName : chiTest

from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split

from data.census import census_data
from data.bank import bank_data
from data.credit import credit_data
from utils.config import census, credit, bank

dataset = "census"

data_config = {"census": census, "credit": credit, "bank": bank}
d_config = data_config[dataset]  # replace

dataset_config = {"census": census_data, "credit": credit_data, "bank": bank_data}
ds_config = dataset_config[dataset]

print(dataset)
print(d_config.params)
print(d_config.input_bounds)
print(d_config.sens_name)
print(d_config.feature_name)
print(d_config.class_name)
print(d_config.categorical_features)
print(d_config.sensitive_param)

# X, Y, input_shape, nb_classes = ds_config()
#
# T = []
# for y in Y:
#     T.append(y[1])
# Y = np.array(T)
#
# RANDOM_STATE = 1
#
# print(Y[0:10])
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
#
# c = chi2(X_train, y_train)
# print(c)
# c = np.array(c[0])
#
# print(c.max())
# print(c.min())
# print(np.sum(c))
# print(c.mean())
#
# c1 = c / np.sum(c)
# print(c1)
# print(np.sum(c1))

print("")
print("")
inp = [1, 2, 3, 4, 6, 7, 8, 9, 2, 1, 3, 523, 213, 21421, 3]
# inp = [item for item in inp
arr = set()
print(tuple(inp))
arr.add(tuple(inp))
print(len(arr))

arr.add(tuple(inp))
print(len(arr))
