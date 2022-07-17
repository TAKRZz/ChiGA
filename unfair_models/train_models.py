# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/14 20:25
# @FileName : train_models

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from utils.config import census, credit, bank
from sklearn.neural_network import MLPClassifier
from data.census import census_data
from data.bank import bank_data
from data.credit import credit_data
from sklearn.model_selection import train_test_split
import joblib

dataset = "credit"

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
# quit()
"""
    sex 9 [0, 1]
"""
model = MLPClassifier(activation='relu', alpha=0.0001, batch_size=64, beta_1=0.9,
                      beta_2=0.999, early_stopping=False, epsilon=1e-08,
                      hidden_layer_sizes=[128, 128, 128, 128, 128],
                      learning_rate='constant', learning_rate_init=0.01, max_iter=100,
                      momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                      power_t=0.5, random_state=None, shuffle=True, solver='adam',
                      tol=0.0001, validation_fraction=0.1, verbose=False,
                      warm_start=False)

# model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                                max_depth=None, max_features='auto', max_leaf_nodes=None,
#                                min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
#                                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
#                                oob_score=False, random_state=None, verbose=0,
#                                warm_start=False)

# model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#             decision_function_shape='ovr', degree=3, gamma='auto',
#             kernel='rbf', max_iter=-1, probability=True, random_state=None,
#             shrinking=True, tol=0.001, verbose=False)

X, Y, input_shape, nb_classes = ds_config()

T = []
for y in Y:
    T.append(y[1])
Y = np.array(T)

print(X.shape)
print(Y.shape)

print("Length of X is :", len(X))

# print(X)
# print(Y)
# print(input_shape)
# print(nb_classes)

RANDOM_STATE = 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
print(len(X_train))
print(len(X_test))
start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(model.score(X_test, y_test))

joblib.dump(model, "{}/MLP_unfair{}.pkl".format(dataset, RANDOM_STATE))

with open("./trainLog.txt", 'a') as fi:
    fi.write("----------------------------------------------------\n".format(model))
    fi.write("Model :{}\n".format(model))
    fi.write("Datasets:    {}\n".format(dataset))
    fi.write("Random_State : {}\n".format(RANDOM_STATE))
    fi.write("Score : {}\n".format(model.score(X_test, y_test)))
    fi.write("Saved as : \"model/{}/MLP_unfair{}.pkl\"\n".format(dataset, RANDOM_STATE))
    fi.write("takes : {}s\n".format(end - start))
    fi.write("----------------------------------------------------\n\n")
