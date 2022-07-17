# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/15 20:08
# @FileName : resultTest
import time

import joblib
import numpy as np

from utils.config import census, credit, bank

dataset = "bank"
data_config = {"census": census, "credit": credit, "bank": bank}
config = data_config[dataset]  # replace
sensitive_param = config.sensitive_param
print(sensitive_param)
threshold = 0
input_bounds = config.input_bounds
print(input_bounds[sensitive_param - 1])
classifier_name = '../unfair_models/{}/MLP_unfair1.pkl'.format(dataset)  # replace


def isDisc(data, model, sens, bounds):
    print(bounds)
    min = bounds[0]
    max = bounds[1]
    print(min)
    print(max)
    res = np.zeros(shape=(data.shape[0]), dtype=int)
    for i in range(min, max + 1):
        for d in data:
            d[sens - 1] = i
        res += np.array(model.predict(data), dtype=int)
        # print(res.shape)
    return res


d = np.load("bank/1/local_samples_MLP_chiga_2_5.npy")
model = joblib.load(classifier_name)

# res = isDisc(data=d, model=model, sens=sensitive_param, bounds=input_bounds[sensitive_param - 1])
# # print(res[0:10])
#
# count = 0
# for r in range(res.shape[0]):
#     if res[r] != 0 and res[r] != (input_bounds[sensitive_param - 1][1] - input_bounds[sensitive_param - 1][0] + 1):
#         count += 1
#     else:
#         print(r, d[r])
# print(count)

for x in range(d.shape[0]):
    for y in range(x + 1, d.shape[0]):
        yes = True
        for i in range(d.shape[1]):
            if i != sensitive_param - 1 and d[x][i] != d[y][i]:
                yes = False
        if yes:
            print(d[x], d[y])
