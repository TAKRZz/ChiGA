# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/14 19:57
# @FileName : GA

import numpy as np


# classifier_name = 'Random_Forest_standard_unfair.pkl'
# model = joblib.load(classifier_name)

class GA:
    # input:
    #     nums: m * n  n is nums_of x, y, z, ...,and m is population's quantity
    #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
    #     DNA_SIZE is binary bit size, None is auto
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003, mutation_positive=None,
                 mutation_negative=None):
        nums = np.array(nums)
        bound = np.array(bound)
        # print(bound)
        self.bound = bound
        min_nums, max_nums = np.array(list(zip(*bound)))
        # print(bound)
        # print(*bound)
        # print(zip(*bound))
        # print(list(zip(*bound)))
        # print(min_nums, max_nums)
        self.var_len = var_len = max_nums - min_nums
        bits = np.ceil(np.log2(var_len + 1))
        if DNA_SIZE is None:
            DNA_SIZE = int(np.max(bits))
        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)
        # POP = np.zeros((*nums.shape, DNA_SIZE))
        # for i in range(nums.shape[0]):
        #     for j in range(nums.shape[1]):
        #         num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
        #         POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        # self.POP = POP
        self.POP = nums
        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        self.mutation_positive = mutation_positive
        self.mutation_negative = mutation_negative
        self.IsDisc = np.array([])
        # self.importance = imp

    # def translateDNA(self):
    #     W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
    #     binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
    #     for i in range(binary_vector.shape[0]):
    #         for j in range(binary_vector.shape[1]):
    #             binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
    #             binary_vector[i, j] += self.bound[j][0]
    #     return binary_vector
    def get_fitness(self, non_negative=False):
        # result = self.func(*np.array(list(zip(*self.translateDNA()))))
        # result = [self.func(self.POP[i]) for i in range(len(self.POP))]
        result = []
        proba = []
        for i in range(len(self.POP)):
            f, p = self.func(self.POP[i])
            result.append(f)
            proba.append(p)
        self.IsDisc = np.array(proba)
        # result 是 [2 * abs(out0-out1) + 1] 组成的 list
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        return result

    def select(self):
        fitness = self.get_fitness()
        fit = [item for item in fitness]
        # print(fit)
        # quit()
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fit / np.sum(fit))]
        self.get_fitness()


    def mutate(self):
        for index in range(len(self.POP)):
            if self.IsDisc[index] == 0:
                # 非歧视样本
                mutation_index = np.random.choice(np.arange(self.DNA_SIZE),
                                                  size=np.random.randint(0, 3, 1),
                                                  replace=False, p=self.mutation_negative)
                # mutation_index = np.random.choice(np.arange(self.DNA_SIZE),
                #                                   size=int(1 + np.random.random(1) / 2 * (self.DNA_SIZE - 1)),
                #                                   replace=False, p=self.mutation_positive)
                # print("NotDisc", mutation_index, self.mutation_positive)

                for a in mutation_index:
                    self.POP[index][a] = np.random.randint(self.bound[a][0], self.bound[a][1])
            else:
                # 歧视样本
                mutation_index = np.random.choice(np.arange(self.DNA_SIZE),
                                                  size=np.random.randint(0, 3, 1),
                                                  replace=False, p=self.mutation_positive)
                # mutation_index = np.random.choice(np.arange(self.DNA_SIZE),
                #                                   size=int(1 + np.random.random(1) / 2 * (self.DNA_SIZE - 1)),
                #                                   replace=False, p=self.mutation_negative)
                # print("IsDisc", mutation_index, self.mutation_negative)
                for a in mutation_index:
                    self.POP[index][a] = np.random.randint(self.bound[a][0], self.bound[a][1])
        # quit()
    # def mutate(self):
    #     for people in self.POP:
    #         for point in range(self.DNA_SIZE):
    #             if np.random.rand() < self.mutation:
    #                 # var[point] = 1 if var[point] == 0 else 1
    #                 people[point] = np.random.randint(self.bound[point][0], self.bound[point][1])

    def crossover(self):
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, len(self.bound))
                end_points = np.random.randint(0, len(self.bound) - cross_points)
                # print(people[cross_points:end_points], self.POP[i_, cross_points:end_points])
                people[cross_points:end_points], self.POP[i_, cross_points:end_points] = self.POP[i_,
                                                                                         cross_points:end_points], \
                                                                                         people[cross_points:end_points]
                # print(people[cross_points:end_points], self.POP[i_, cross_points:end_points])
                # quit()

    def evolution(self):
        self.select()
        self.mutate()
        self.crossover()
