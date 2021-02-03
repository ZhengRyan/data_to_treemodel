#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: data_aes.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-07-01
'''

# user_log_acct,device_id,apply_time,merge


import warnings

from model_code.aes import prpcrypt
from model_code.utils import *

warnings.filterwarnings('ignore')

cust_id = 'PassengerId'
label = 'Survived'
apply_time = 'apply_time'

# =========================step 2 读取数据集=========================
data = pd.read_pickle('titanic_train.pkl')

remove = ['Name', 'Ticket', 'Cabin']
data = data.drop(remove, axis=1)  # 去掉无关信息

# =========================读取数据集=========================


# =========================step 4 划分训练集和测试集=========================
###id,apply_time,target
df_id_tmp = data[[cust_id, label]]
df_id_tmp[apply_time] = '2019-08-15'
df_id_tmp = df_id_tmp[[cust_id, apply_time, label]]
print(df_id_tmp.head())

df_id_tmp.to_csv('df_id_tmp.csv', index=False)

df_id_tmp[label] = df_id_tmp[label].map(str)
df_id_tmp['merge'] = df_id_tmp[cust_id].map(str).str.cat([df_id_tmp[apply_time], df_id_tmp[label]], sep=',')
pc = prpcrypt('ecloud')
df_id_tmp['merge'] = df_id_tmp['merge'].map(lambda x: pc.encrypt(x))
df_id_tmp[[cust_id, apply_time, 'merge']].to_csv('titanic_sample.csv', index=False)

if label in data.columns:
    y = data.pop(label)  # 标签

# data = data.set_index(cust_id)


data[apply_time] = '2019-08-15'
print(data.head())

data.to_csv('titanic_train.csv', index=False)

import sys

sys.exit(0)

selected_features = data.columns.format()

df_id = split_data_type(df_id_tmp, key_col=cust_id, target=label)
print('df_id')
print(df_id.head())
print(data.head())
# X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=42)
