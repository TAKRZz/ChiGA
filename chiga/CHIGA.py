# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/14 20:01
# @FileName : CHIGA

import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from utils.config import census, credit, bank
import joblib
from Genetic_Algorithm import GA
from data.census import census_data
from data.credit import credit_data
from data.bank import bank_data
from sklearn.feature_selection import chi2

import lime
from lime.lime_tabular import LimeTabularExplainer

# global 记录的是 可能歧视的输入 集

global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []

# tot_inputs 记录的是 所有产生的 输入 集
tot_inputs = set()
# 记录的是 sens 出现的 index ++
location = np.zeros(21)

"""
census: 9,1 for gender, age, 8 for race
credit: 9,13 for gender,age
bank: 1 for age
"""

dataset = "bank"
data_config = {"census": census, "credit": credit, "bank": bank}
config = data_config[dataset]  # replace
sensitive_param = 1
threshold = 0
input_bounds = config.input_bounds
print(input_bounds)
print(config.feature_name)
classifier_name = '../unfair_models/{}/MLP_unfair1.pkl'.format(dataset)  # replace
model = joblib.load(classifier_name)

'''
    LIME 局部解释器 （已经排序）
    选取 rank 靠前的
'''


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, iteration, params, input_bounds, sensitive_param):
        s = self.stepsize
        samples = []
        random.seed(time.time())
        while len(samples) < iteration:
            x = np.zeros(params)
            for i in range(params):
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
            x[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
            samples.append(x)
        return samples


def evaluate_local(inp):
    inp0 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = int(input_bounds[sensitive_param - 1][0])

    # inp0 = np.array(inp0)

    # print(tuple[inp0])
    # quit()
    tot_inputs.add(tuple(inp0))

    min_pre = 1
    max_pre = 0

    sum0 = 0
    sum1 = 0

    for val in range(config.input_bounds[sensitive_param - 1][0], config.input_bounds[sensitive_param - 1][1] + 1):

        # 进行一个替换
        inp1 = [int(i) for i in inp]
        inp1[sensitive_param - 1] = val

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out1 = model.predict(inp1)
        if out1 == 1:
            sum1 += 1
        else:
            sum0 += 1

        pre1 = model.predict_proba(inp1)

        if min_pre > pre1[0, 1]:
            min_pre = pre1[0, 1]

        if max_pre < pre1[0, 1]:
            max_pre = pre1[0, 1]

        # print(abs(pre0 - pre1)[0]
        # print(map(tuple, inp0))

    if (sum0 != 0) and (sum1 != 0) and (tuple(inp0) not in local_disc_inputs):
        local_disc_inputs.add(tuple(inp0))
        local_disc_inputs_list.append(inp0)

        return abs(max_pre - min_pre) + 0.5, 1

    return abs(max_pre - min_pre), 0


def xai_fair_testing(max_global, max_local):
    data_config = {"census": census, "credit": credit, "bank": bank}
    config = data_config[dataset]
    feature_names = config.feature_name
    class_names = config.class_name
    sens_name = config.sens_name[sensitive_param]
    params = config.params

    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    # prepare the testing data and model

    X, Y, input_shape, nb_classes = data[dataset]()

    # print(X[0:10])
    '''
        in Chi2, The inputs must be positive; 
    '''
    for b in range(len(input_bounds)):
        # print(input_bounds[b])
        # print(input_bounds[b][0])
        if input_bounds[b][0] < 0:
            for x in X:
                x[b] -= input_bounds[b][0]
    '''
        The Output should be 1-dimension
    '''
    T = []
    for y in Y:
        T.append(y[1])
    Y = np.array(T)

    RANDOM_STATE = 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

    chi2_value = chi2(X_train, y_train)
    chi2_value = np.array(chi2_value[0])
    '''
        # 对于 非歧视 样本， 更改权重更大的属性的值
    '''
    chi2_percent_positive = chi2_value / np.sum(chi2_value)
    '''
        # 对于  歧视 样本， 更改权重更小的属性的值
    '''
    chi2_percent_negative = (1 / chi2_percent_positive)
    chi2_percent_negative = chi2_percent_negative / np.sum(chi2_percent_negative)

    start = time.time()

    model_name = classifier_name.split("/")[-1].split("_")[0]
    # file_name = "aequitas_"+dataset+sensitive_param+"_"+model+""
    file_name = "chiga_{}_{}{}_{}_{}.txt".format(model_name, dataset, sensitive_param, int(max_global / 100),
                                                 int(max_local / 100))

    f = open(file_name, "a")
    f.write("iter:" + str(iter) + "-------------------------------------------------------" + "\n" + "\n")
    f.write("max_global : {}-------------max_local : {}---------\n\n".format(max_global, max_local))
    f.close()

    global_discovery = Global_Discovery()
    '''
    train_samples 随机输入集
    与训练数据无关
    '''
    train_samples = global_discovery(max_global, params, input_bounds, sensitive_param)
    train_samples = np.array(train_samples)

    np.random.shuffle(train_samples)

    print(train_samples.shape)

    seed = train_samples

    '''
        没有使用 X 输入数据集
        返回的是 可能歧视数据集
    '''

    print("Randomly Generate {} Samples".format(max_global))

    # print('Finish Searchseed')

    for inp in seed:
        inp0 = [int(i) for i in inp]
        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))
        tot_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs_list.append(inp0.tolist()[0])

    # print("Finished Global Search")
    # total
    # print('length of total input is:' + str(len(tot_inputs)))
    # potential
    # print('length of global discovery is:' + str(len(global_disc_inputs_list)))

    end = time.time()

    print('Total time:' + str(end - start))

    print("")
    print("Starting Local Search")

    '''
    ------------------------------------
              Genetic Algorithm
    ------------------------------------
    '''

    nums = global_disc_inputs_list
    DNA_SIZE = len(input_bounds)

    cross_rate = 0.9
    mutation = 0.05
    iteration = max_local
    ga = GA(nums=nums, bound=input_bounds, func=evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate,
            mutation=mutation, mutation_positive=chi2_percent_positive, mutation_negative=chi2_percent_negative)
    # for random

    '''
        每 300 s 打印一次
    '''
    count = 60
    for i in range(iteration):
        ga.evolution()
        end = time.time()
        use_time = end - start
        if use_time >= count:
            f = open(file_name, "a")

            f.write("Percentage discriminatory inputs - " + str(
                float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                / float(len(tot_inputs)) * 100) + "\n")
            f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)) + "\n")
            f.write("Total Inputs are " + str(len(tot_inputs)) + "\n")
            f.write('use time:' + str(end - start) + "\n" + "\n")
            f.close()

            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))

            print('use time:' + str(end - start))
            count += 60
        if i % 60 == 0:
            print("Epochs {}".format(i))
    np.save(
        '../results/' + dataset + '/' + str(sensitive_param) + '/local_samples_{}_chiga_{}_{}.npy'.format(model_name,
                                                                                                          int(max_global / 100),
                                                                                                          int(max_local / 100)),
        np.array(local_disc_inputs_list))

    print("Total Inputs are " + str(len(tot_inputs)))
    print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
    print("Percentage discriminatory inputs - " + str(
        float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
    print(
        "saved as " + '../results/' + dataset + '/' + str(sensitive_param) + '/local_samples_{}_chiga_{}_{}.npy'.format(
            model_name,
            int(max_global / 100),
            int(max_local / 100)))


def main(argv=None):
    xai_fair_testing(max_global=1000, max_local=1000)


if __name__ == '__main__':
    main()
