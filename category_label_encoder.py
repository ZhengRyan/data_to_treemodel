#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: category_label_encoder.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-09-21
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def category_to_labelencoder(data, labelencoder=[]):
    label_encoder_dict = {}
    le = LabelEncoder()
    for col in labelencoder:
        print('{} in process!!!'.format(col))
        data[col] = le.fit_transform(data[col].values)
        number = [i for i in range(0, len(le.classes_))]
        key = list(le.inverse_transform(number))
        label_encoder_dict[col] = dict(zip(key, number))
    return label_encoder_dict


def category_to_labelencoder_apply(data, labelencoder_dict={}):
    for col, mapping in labelencoder_dict.items():
        print('{} in process!!!'.format(col))
        data[col] = data[col].map(mapping).fillna(-1)
        data[col] = data[col].astype(int)


#####训练
fruit_data = pd.DataFrame({
    'fruit': ['apple', 'orange', 'pear', 'orange', 'red'],
    'color': ['red', 'orange', 'green', 'green', 'red'],
    'weight': [5, 6, 3, 4, 2]
})
print(fruit_data)

labelencoder_cols = ['fruit', 'color']

label_encoder_dict = category_to_labelencoder(fruit_data, labelencoder_cols)
print(fruit_data)

#####应用
test_data = pd.DataFrame({
    'fruit': ['apple', 'orange', 'pear', 'orange', 'red'],
    'color': ['aaa', 'orange', 'green', 'green', 'red'],
    'weight': [5, 6, 3, 4, 2]
})
print(test_data)

category_to_labelencoder_apply(test_data, label_encoder_dict)
print(test_data)
