#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: woe_transformer.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-22
'''

import operator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin


class WoeTransformer(TransformerMixin):

    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs

    #   def get_feature_bin_woe(self, df_var_bin_woe):
    #       '''
    #       返回每个特征对应的分箱值对应的woe值
    #       :param df_var_bin_woe:  三列的dataframe。['var_name','range','woe']
    #       :return: {'D157': {Interval(-999.0, 1.0, closed='left'): -1.2835846793756913,
    # Interval(1.0, 2.0, closed='left'): -0.9424139219729241,
    # Interval(2.0, 3.0, closed='left'): -0.34860570638043403,
    # Interval(3.0, 4.5, closed='left'): 0.36714548649924816,
    # Interval(4.5, inf, closed='left'): 1.7735952385975287}
    #       '''
    #       return df_var_bin_woe.groupby('var_name').apply(
    #           lambda df: df.set_index('range')['woe'].to_dict()).to_dict()

    # def bin_to_woe(self, df, var_bin_woe_dict):
    #     '''
    #     占比前10的类别会单独转成woe值。其余类别归为一个大类对应一个woe值
    #     :param df:
    #     :param var_bin_woe_dict:    形如{"Sex": {"female": -1.5298770033401874, "male": 0.9838327092415774}, "Embarked": {"C": -0.694264203516269, "S": 0.1977338357888416, "other": -0.030202603851420356}}
    #     :return:
    #     '''
    #     df_ = df.copy()
    #     for feature, bin_woe in var_bin_woe_dict.items():
    #         df_[feature].fillna('miss', inplace=True)
    #         # df_[feature] = df_[feature].map(lambda x: x if (x in bin_woe.keys() or x == 'miss') else 'other')  #缺失再少也归为一类
    #         df_[feature] = df_[feature].map(lambda x: x if x in bin_woe.keys() else 'other')
    #         df_[feature] = df_[feature].map(bin_woe)
    #
    #     return df_

    def bin_to_woe(self, df, var_bin_woe_dict):
        '''
        根据传进来的var_bin_woe_dict对原始值进行映射。
        如在var_bin_woe_dict没有的类别（数据集中新出现的类别，归为到other这类）同时var_bin_woe_dict中得有other该类别对应的woe值
        如果var_bin_woe_dict中没有other该类别对应的woe值，即数据集中新出现的类别归为缺失值，即新出现的类别没有woe值
        :param df:
        :param var_bin_woe_dict:    形如{"Sex": {"female": -1.5298770033401874, "male": 0.9838327092415774}, "Embarked": {"C": -0.694264203516269, "S": 0.1977338357888416, "other": -0.030202603851420356}}
        :return:
        '''

        for feature, bin_woe in var_bin_woe_dict.items():
            df[feature] = df[feature].map(
                lambda x: x if (x in bin_woe.keys() or x is np.nan or pd.isna(x)) else 'other')
            df[feature] = df[feature].map(bin_woe)

        return df

    # def bin_to_woe(self, df, var_bin_woe_dict):
    #     '''
    #     根据传进来的var_bin_woe_dict对原始值进行映射。如在var_bin_woe_dict没有的类别（新出现的类别），归为缺失值，即新出现的类别没有woe值
    #     :param df:
    #     :param var_bin_woe_dict:   形如{"Sex": {"female": -1.5298770033401874, "male": 0.9838327092415774}, "Embarked": {"C": -0.694264203516269, "S": 0.1977338357888416, "other": -0.030202603851420356}}
    #     :return:
    #     '''
    #     df_ = df.copy()
    #     for feature, bin_woe in var_bin_woe_dict.items():
    #         df_[feature] = df_[feature].map(bin_woe)
    #
    #     return df_

    def transform(self, df, var_bin_woe_dict, bins_dict={}):
        '''
        输入三列的dataframe，['var_name','range','woe'] 返回转换woe后的数据集
        :param var_bin_woe_dict:    形如{"Sex": {"female": -1.5298770033401874, "male": 0.9838327092415774}, "Embarked": {"C": -0.694264203516269, "S": 0.1977338357888416, "other": -0.030202603851420356}}
        :return:转换woe后的数据集
        '''

        df_ = df.copy()
        if bins_dict:
            print('需要将原始数据转bin')
            df_ = self.data_to_bin(df, bins_dict=bins_dict)
        return self.bin_to_woe(df_, var_bin_woe_dict)

    def data_to_bin(self, df, bins_dict={}):
        '''
        原始数据根据bins_dict进行分箱
        :param df:含有目标变量的数据集；不需要返回var_summary可以不需要目标变量，将函数中target部分注释
        :param target:目标值变量名称
        :param bins_dict:分箱字典, 形如{'D157': [-999, 1.0, 2.0, 3.0, 5.0, inf]}
        :return:
        '''

        if not isinstance(bins_dict, dict):
            assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

        data_with_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(pd.cut)(df[col], bins=bins, right=False, retbins=True) for col, bins in bins_dict.items())
        data_bin = pd.DataFrame([i[0].astype(str) for i in data_with_bins]).T
        b_dict = dict([(i[0].name, i[1].tolist()) for i in data_with_bins])
        if not operator.eq(bins_dict, b_dict):
            assert '传入的分箱和应用后的分箱不对等，请联系开发者'

        return data_bin
