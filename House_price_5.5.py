# -*- coding:utf8 -*-
# @TIME : 2018/5/16 下午8:03
# @Author : Allen
# @File : House_price_5.5.py

#导入常用packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/Users/a1/Desktop/learning/house_price_5.15')

#读取数据 & 合并数据
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)

y_train = np.log1p(train_df.pop("SalePrice"))
# print(y_train)
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.head())
print(all_df.shape)




#处理特征值

#处理数字型类别：
all_df["MSSubClass"] = all_df["MSSubClass"].astype(str)
pd.get_dummies(["MSSubClass"], prefix='MSSubClass')
#处理一般类别变量：
all_dummy_df = pd.get_dummies(all_df)
#处理缺失的值
all_dummy_df.isnull().sum().sort_values(ascending=False)
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
print(all_dummy_df.shape)
#处理数字型变量的列：
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_cols_mean = all_dummy_df.loc[:, numeric_cols].mean()
numeric_cols_std = all_dummy_df.loc[:, numeric_cols].std()

all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_cols_mean)/numeric_cols_std
print(all_dummy_df.shape)


#构造模型
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

X_train = dummy_train_df.values
X_test = dummy_test_df.values

#用ridge模型处理数据
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
params = np.logspace(-3, 2, 50)
test_scores = []
for param in params:
    clf = Ridge(param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
print(min(test_scores))
plt.show(block=False)

ridge = Ridge(15)

#用xgboost模型处理数据
from xgboost import XGBRegressor
params = [1, 2, 3, 4, 5, 6, 7]
test_scores = []
for param in params:
    clf = XGBRegressor(max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
print(min(test_scores))
plt.show(block=False)

#利用bagging模型集成
from sklearn.ensemble import BaggingRegressor
params = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
test_scores =[]
for param in params:
    clf = BaggingRegressor(n_estimators=200, base_estimator=ridge)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
print(min(test_scores))
plt.show()