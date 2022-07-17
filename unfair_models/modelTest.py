# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/15 21:32
# @FileName : modelTest
import joblib

model = joblib.load("./census/MLP_unfair1.pkl")

arr = [[1, 1, 1, 10, 1, 1, 1, 50, 20, 10, 20, 10, 1]]
print(model.predict_proba(arr))
print(model.predict(arr))
