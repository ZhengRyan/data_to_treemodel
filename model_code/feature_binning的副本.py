#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: feature_binning.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.cluster import KMeans
import json
import sys
import operator
import copy
from model_code.logger_utils import Logger

log = Logger(level="info", name=__name__).logger


class FeatureBinning(object):
    def __init__(self, default_bins=10, n_jobs=2):
        self.default_bins = default_bins
        self.n_jobs = n_jobs
        self.__bin_summary = None
        self.cut_off = dict()
        self.var_bin_woe_dict = dict()

    @property
    def default_bins(self):
        return self._default_bins

    @default_bins.setter
    def default_bins(self, value):
        self._default_bins = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    def get_var_summary(self):
        if self.__bin_summary is not None:
            return self.__bin_summary.copy()  #

    def get_iv(self):
        if self.__bin_summary is None:
            raise ValueError('ERROR: 没有调用分箱函数，请调用category_bin函数')

        return self.__bin_summary.set_index('var_name').IV.drop_duplicates()

    def check_monotonic_single(self, df, col_name, target='target', min_samples=None, empty_separate=False):
        '''
        检查变量单调性
        :param df:含有目标变量及分箱后的数据集
        :param col_name:单变量名称
        :param target:目标变量列名
        :param min_samples:最小样本占比
        :param empty_separate:是否将空值单独最为一箱
        :return:
        '''

        group = df.groupby(col_name)
        table = group[target].agg(['sum', 'count'])

        # 移除缺失值加入最小分箱占比的计算
        if empty_separate:
            table.drop(index=min(table.index), inplace=True)

        if min_samples:  # 是否需要检查最小分箱占比
            is_check_monotonic = min(table['count'] / table['count'].sum()) > min_samples  # 检查最小分箱占比
            if not is_check_monotonic:  # 不满足最小分箱占比
                log.warning('{} 变量不满足最小分箱占比 {} ，返回重新分箱！！！'.format(col_name, min_samples))
                return is_check_monotonic, len(table)

        table['badrate'] = table['sum'] / table['count']
        is_check_monotonic = table['badrate'].is_monotonic_increasing or table[
            'badrate'].is_monotonic_decreasing  # 检查分箱是否单调
        log.warning('{} 变量满足最小分箱占比，检查是否单调 {}'.format(col_name, is_check_monotonic))

        return is_check_monotonic, len(table)

    def __calc_group(self, data_, var_name):
        count = len(data_)
        bad_num = data_.Y.sum()
        good_num = count - bad_num

        return pd.Series({'var_name': var_name, '总人数': count, 'bad_人数': bad_num, 'good_人数': good_num})

    def calc_woe_iv(self, df, col_name, target='target'):
        '''
        计算单变量详情
        :param df:含有目标变量及分箱后的数据集
        :param col_name:单变量名称
        :param target:目标值变量名称
        :return:
        '''

        X, Y = df[col_name], df[target]

        data = pd.DataFrame({'X': X, 'Y': Y})

        bin_g = data.groupby(data['X']).apply(self.__calc_group, var_name=col_name)
        # 排下序
        bin_g = bin_g.loc[
            pd.Index(
                sorted(bin_g.index, key=lambda x: float(x.split()[0].replace('[', '').replace(',', '')), reverse=False),
                name='X')]
        # 排下序
        total = data.Y.count()
        bad_count = (data.Y == 1).sum()
        good_count = (data.Y == 0).sum()
        bin_g['总人数占比'] = bin_g['总人数'] / total  # 总人数占比
        bin_g['bad_人数占比'] = bin_g['bad_人数'] / bad_count  # bad_人数占比
        bin_g['good_人数占比'] = bin_g['good_人数'] / good_count  # good_人数占比
        bin_g['bad_rate'] = bin_g['bad_人数'] / bin_g['总人数']  # bad_rate
        bin_g['累计bad人数'] = bin_g['bad_人数'].cumsum()
        bin_g['累计good人数'] = bin_g['good_人数'].cumsum()
        bin_g['累计bad人数占比'] = bin_g['bad_人数'].cumsum() / bad_count
        bin_g['累计good人数占比'] = bin_g['good_人数'].cumsum() / good_count
        bin_g['woe'] = bin_g.apply(lambda x: 0.0 if x['bad_人数占比'] == 0 else np.log(x['good_人数占比'] / x['bad_人数占比']),
                                   axis=1)
        bin_g['ks'] = abs(bin_g['累计bad人数占比'] - bin_g['累计good人数占比'])
        bin_g['iv'] = (bin_g['good_人数占比'] - bin_g['bad_人数占比']) * bin_g.woe
        bin_g['IV'] = bin_g.iv.replace({np.inf: 0, -np.inf: 0}).sum()
        bin_g.index.name = 'range'
        return bin_g.reset_index()

    def calc_var_summary(self, df, target='target'):
        '''
        计算所有变量的详情
        :param df:含有目标变量及分箱后的数据集
        :param target:目标值变量名称
        :param exclude_cols:排除不参与计算的变量
        :return:
        '''

        summary = Parallel(n_jobs=self.n_jobs)(
            delayed(self.calc_woe_iv)(df[[col, target]], col, target) for col in df.columns if col not in [target])
        var_summary = pd.concat(summary, axis=0)
        return var_summary

    def equal_freq_bin(self, df_, x_list=[], target='target', special_value=[-999999], min_sample_rate=0.05,
                       n_bins=None,
                       q_cut_list=None, is_need_monotonic=True):
        '''
        等频分箱
        :param df:含有目标变量及参与分箱的变量列表的数据集
        :param x_list:参与分箱的变量列表
        :param target:目标值变量名称
        :param special_value:不参与分箱的特殊值
        :param min_sample_rate:每个箱子的最小占比
        :param n_bins:需要分成几箱
        :param q_cut_list:百分比分割点列表
        :param is_need_monotonic:是否强制单调
        :return:
        '''

        if n_bins is None and q_cut_list is None:
            n_bins = self.default_bins

        if q_cut_list is None:
            q_cut_list = np.arange(0, 1, 1 / n_bins)

        Y = df_[target]
        empty_separate = True if special_value else False

        def equal_freq_bin_single(X, q=q_cut_list):
            col_name = X.name
            count = X.count()
            assert count > 0, 'ERROR: 变量 :-) {} :-) 的取值全部是空，请检查变量'.format(col_name)
            # 排除特殊值
            f_series = X[~X.isin(special_value)]

            is_monotonic = False
            step = None
            while not is_monotonic:
                splits = np.quantile(f_series, q)  # [1. 1. 1. 1. 1. 2. 2. 2. 3. 5.]
                if special_value:
                    cut_off = [-np.inf] + [special_value[0] + 0.001] + list(np.unique(splits)[1:]) + [
                        np.inf]  # [-999.0, 1.0, 2.0, 3.0, 5.0, inf]
                else:
                    cut_off = [-np.inf] + list(np.unique(splits)[1:]) + [np.inf]

                X_bin = pd.cut(X, bins=cut_off,
                               right=False)  # [[-999.0, 1.0) < [1.0, 2.0) < [2.0, 3.0) < [3.0, 5.0) < [5.0, inf)]

                # 检查单调性
                to_check_data = pd.DataFrame({col_name: X_bin, target: Y})
                is_monotonic, n_b = self.check_monotonic_single(to_check_data, col_name, target=target,
                                                                min_samples=min_sample_rate,
                                                                empty_separate=empty_separate)
                if step is None and empty_separate:
                    n_b += 2
                elif step is None:
                    n_b += 1
                if n_b <= 2 or not is_need_monotonic:
                    break
                    n_b -= 1
                step = 1 / n_b
                q = np.arange(0, 1, step)

            return X_bin, cut_off

        if not x_list:
            x_list = df_.columns.tolist()

        data = Parallel(n_jobs=self.n_jobs)(
            delayed(equal_freq_bin_single)(df_[col], q_cut_list) for col in x_list)  # 批量处理
        data_bin = pd.concat([i[0].astype(str) for i in data] + [Y], axis=1)  # 分箱后的数据集
        bins_dict = dict([(i[0].name, i[1]) for i in data])  # 分箱后的字典
        self.cut_off = copy.deepcopy(bins_dict)
        var_summary = self.calc_var_summary(data_bin, target=target)
        self.__bin_summary = var_summary.copy()

        return data_bin, bins_dict, var_summary

    def dt_bin(self, df_, x_list=[], target='target', special_value=[-999999], min_sample_rate=0.05, n_bins=None,
               is_need_monotonic=True):
        '''
        决策树分箱
        :param df: 含有目标变量及参与分箱的变量列表的数据集
        :param x_list: 参与分箱的变量列表
        :param target:目标值变量名称
        :param special_value:不参与分箱的特殊值
        :param min_num_rate:每个箱子的最小占比
        :param n_bins:需要分成几箱
        :param is_need_monotonic:是否强制单调
        :return:
        '''

        if n_bins is None:
            n_bins = self.default_bins

        Y = df_[target]
        empty_separate = True if special_value else False

        def dt_bin_single(X, n_b=n_bins):
            col_name = X.name
            count = X.count()
            assert count > 0, 'ERROR: 变量 :-) {} :-) 的取值全部是空，请检查变量'.format(col_name)
            # 排除特殊值
            f_series = X[~X.isin(special_value)]
            y_ = Y[f_series.index]

            is_monotonic = False
            is_first = True
            while not is_monotonic:

                # 决策树分箱逻辑
                tree = DecisionTreeClassifier(
                    min_samples_leaf=min_sample_rate,
                    max_leaf_nodes=n_b
                )
                tree.fit(np.array(f_series).reshape((-1, 1)), y_)
                thresholds = tree.tree_.threshold
                thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
                splits = np.sort(thresholds)
                # 决策树分箱逻辑
                if special_value:
                    cut_off = [-np.inf] + [special_value[0] + 0.001] + list(splits) + [
                        np.inf]  # [-999.0, 1.0, 2.0, 3.0, 5.0, inf]
                else:
                    cut_off = [-np.inf] + list(splits) + [np.inf]

                X_bin = pd.cut(X, bins=cut_off,
                               right=False)  # [[-999.0, 1.0) < [1.0, 2.0) < [2.0, 3.0) < [3.0, 5.0) < [5.0, inf)]

                # 检查单调性
                to_check_data = pd.DataFrame({col_name: X_bin, target: Y})
                is_monotonic, n_b = self.check_monotonic_single(to_check_data, col_name, target=target,
                                                                min_samples=min_sample_rate,
                                                                empty_separate=empty_separate)
                if is_first and empty_separate:
                    n_b += 1
                    is_first = False
                elif is_first:
                    is_first = False
                if n_b <= 2 or not is_need_monotonic:
                    break
                n_b -= 1

            return X_bin, cut_off

        if not x_list:
            x_list = df_.columns.tolist()

        data = Parallel(n_jobs=self.n_jobs)(delayed(dt_bin_single)(df_[col], n_bins) for col in x_list)  # 批量处理
        data_bin = pd.concat([i[0].astype(str) for i in data] + [Y], axis=1)  # 分箱后的数据集
        bins_dict = dict([(i[0].name, i[1]) for i in data])  # 分箱后的字典
        self.cut_off = copy.deepcopy(bins_dict)
        var_summary = self.calc_var_summary(data_bin, target=target)
        self.__bin_summary = var_summary.copy()

        return data_bin, bins_dict, var_summary

    def KMeans_bin(self, df_, x_list=[], target='target', special_value=[-999999], min_sample_rate=0.05, n_bins=None,
                   random_state=1, is_need_monotonic=True):
        '''
        KMeans分箱
        :param df: 含有目标变量及参与分箱的变量列表的数据集
        :param x_list: 参与分箱的变量列表
        :param target: 目标值
        :param special_value: 不参与分箱的特殊值
        :param min_sample_rate: 每个箱子的最小占比
        :param n_bins: 需要分成几箱
        :param random_state: KMeans 模型的随机数
        :param is_need_monotonic:是否强制单调
        :return:
        '''

        if n_bins is None:
            n_bins = self.default_bins

        Y = df_[target]
        empty_separate = True if special_value else False

        def KMeans_bin_single(X, n_b=n_bins):
            col_name = X.name
            count = X.count()
            assert count > 0, 'ERROR: 变量 :-) {} :-) 的取值全部是空，请检查变量'.format(col_name)
            # 排除特殊值
            f_series = X[~X.isin(special_value)]
            y_ = Y[f_series.index]

            is_monotonic = False
            is_first = True

            while not is_monotonic:

                # kmeans 逻辑
                kmeans = KMeans(
                    n_clusters=n_b,
                    random_state=random_state
                )
                kmeans.fit(np.array(f_series).reshape((-1, 1)), y_)

                centers = np.sort(kmeans.cluster_centers_.reshape(-1))

                l = len(centers) - 1
                splits = np.zeros(l)
                for i in range(l):
                    splits[i] = (centers[i] + centers[i + 1]) / 2
                splits = np.unique(splits)
                # kmeans 逻辑
                if special_value:
                    cut_off = [-np.inf] + [special_value[0] + 0.001] + list(splits) + [
                        np.inf]  # [-999.0, 1.0, 2.0, 3.0, 5.0, inf]
                else:
                    cut_off = [-np.inf] + list(splits) + [np.inf]

                X_bin = pd.cut(X, bins=cut_off,
                               right=False)  # [[-999.0, 1.0) < [1.0, 2.0) < [2.0, 3.0) < [3.0, 5.0) < [5.0, inf)]

                # 检查单调性
                to_check_data = pd.DataFrame({col_name: X_bin, target: Y})
                is_monotonic, n_b = self.check_monotonic_single(to_check_data, col_name, target=target,
                                                                min_samples=min_sample_rate,
                                                                empty_separate=empty_separate)

                if is_first and empty_separate:
                    n_b += 1
                    is_first = False
                elif is_first:
                    is_first = False
                if n_b <= 2 or not is_need_monotonic:
                    break
                n_b -= 1

            return X_bin, cut_off

        if not x_list:
            x_list = df_.columns.tolist()

        data = Parallel(n_jobs=self.n_jobs)(delayed(KMeans_bin_single)(df_[col], n_bins) for col in x_list)  # 批量处理
        data_bin = pd.concat([i[0].astype(str) for i in data] + [Y], axis=1)  # 分箱后的数据集
        bins_dict = dict([(i[0].name, i[1]) for i in data])  # 分箱后的字典
        self.cut_off = copy.deepcopy(bins_dict)
        var_summary = self.calc_var_summary(data_bin, target=target)
        self.__bin_summary = var_summary.copy()

        return data_bin, bins_dict, var_summary

    def calc_bin_gb(self, df, col_name, target_name, calc_badrate=False):
        '''
        计算每个箱子的好坏人
        :param df: 含有目标变量参与计算的变量名称的数据集
        :param col_name: 参与计算的变量名称
        :param target_name: 目标变量名称
        :param calc_badrate: 是否计算badrate
        :return:
        '''
        df_tmp = df.copy()
        df_grp = df_tmp.groupby([col_name])[target_name].agg(['count', 'sum'])
        df_grp.columns = ['bin_total', 'bin_bad']
        df_grp['bin_good'] = df_grp['bin_total'] - df_grp['bin_bad']

        df_grp.reset_index(inplace=True)

        if calc_badrate:
            df_grp['bad_rate'] = df_grp['bin_bad'] / df_grp['bin_total']

        return df_grp

    def chi_bin(self, df_, x_list=[], target='target', special_value=[-999999], min_sample_rate=0.05, n_bins=None,
                is_need_monotonic=True):
        '''
        卡方分箱
        :param df: 含有目标变量及参与分箱的变量列表的数据集
        :param x_list:参与分箱的变量列表
        :param target:目标值变量名称
        :param special_value:不参与分箱的特殊值
        :param min_sample_rate:每个箱子的最小占比
        :param n_bins:需要分成几箱
        :param is_need_monotonic:是否强制单调
        :return:
        '''

        if n_bins is None:
            n_bins = self.default_bins

        Y = df_[target]
        empty_separate = True if special_value else False

        def chi_bin_single(X, n_b=n_bins):
            col_name = X.name
            count = X.count()
            assert count > 0, 'ERROR: 变量 :-) {} :-) 的取值全部是空，请检查变量'.format(col_name)
            # 排除特殊值
            f_series = X[~X.isin(special_value)]
            y_ = Y[f_series.index]

            is_monotonic = False
            is_first = True

            split_point_ = []

            while not is_monotonic:

                # 卡方 逻辑
                to_bin = pd.DataFrame({col_name: f_series, target: y_})
                if split_point_:
                    to_bin[col_name] = to_bin[col_name].map(lambda x: bin_split(x, split_point_))
                elif len(np.unique(np.array(to_bin[col_name]))) >= 100:
                    split_point_ = split_col(to_bin[col_name], 100)
                    to_bin[col_name] = to_bin[col_name].map(lambda x: bin_split(x, split_point_))

                df_grp_gb = self.calc_bin_gb(to_bin, col_name, target)

                while True:

                    min_rate = min(df_grp_gb['bin_total'] / df_grp_gb['bin_total'].sum())
                    if split_point_:
                        if len(df_grp_gb) < n_b + 1 and min_rate > min_sample_rate:
                            break
                    else:
                        if len(df_grp_gb) <= n_b and min_rate > min_sample_rate:
                            break

                    chi_li = []
                    for i in range(0, len(df_grp_gb) - 1):
                        tmp = df_grp_gb.loc[i: i + 1, :]
                        chi = calc_chi(tmp, 'bin_total', 'bin_bad', 'bin_good')
                        chi_li.append(chi)

                    merge_ix = chi_li.index(min(chi_li))

                    merge_df_grp_gb = df_grp_gb.loc[merge_ix: merge_ix + 1, :]

                    df_grp_gb['bin_total'][merge_ix] = sum(merge_df_grp_gb['bin_total'])
                    df_grp_gb['bin_bad'][merge_ix] = sum(merge_df_grp_gb['bin_bad'])
                    df_grp_gb['bin_good'][merge_ix] = sum(merge_df_grp_gb['bin_good'])

                    # delete
                    df_grp_gb.drop(index=[merge_ix + 1], inplace=True)

                    # reset index
                    df_grp_gb.reset_index(drop=True, inplace=True)

                if split_point_:
                    split_point_ = df_grp_gb[col_name].tolist()[:-1]
                else:
                    split_point_ = df_grp_gb[col_name].tolist()[1:]

                # 卡方 逻辑

                if special_value:
                    cut_off = [-np.inf] + [special_value[0] + 0.001] + list(split_point_) + [
                        np.inf]  # [-999.0, 1.0, 2.0, 3.0, 5.0, inf]
                else:
                    cut_off = [-np.inf] + list(split_point_) + [np.inf]

                X_bin = pd.cut(X, bins=cut_off,
                               right=False)  # [[-999.0, 1.0) < [1.0, 2.0) < [2.0, 3.0) < [3.0, 5.0) < [5.0, inf)]

                # 检查单调性
                to_check_data = pd.DataFrame({col_name: X_bin, target: Y})
                is_monotonic, n_b = self.check_monotonic_single(to_check_data, col_name, target=target,
                                                                min_samples=min_sample_rate,
                                                                empty_separate=empty_separate)
                if is_first and empty_separate:
                    n_b += 1
                    is_first = False
                elif is_first:
                    is_first = False
                if n_b <= 2 or not is_need_monotonic:
                    break
                n_b -= 1

            return X_bin, cut_off

        def split_col(col, split_num, exclude_attri=[]):

            col = list(col)
            col = list(set(col).difference(set(exclude_attri)))
            size = (max(col) - min(col)) / split_num
            split_point = [min(col) + i * size for i in range(1, split_num + 1)]
            return split_point

        def bin_split(x, split_point):

            if x < split_point[0]:
                return split_point[0]
            elif x >= max(split_point):
                return np.inf
            else:
                for i in range(0, len(split_point) - 1):
                    if split_point[i] <= x < split_point[i + 1]:
                        return split_point[i + 1]

        def calc_chi(df, total_col, bad_col, good_col):
            df_ = df.copy()
            bad_rate = sum(df_[bad_col]) * 1.0 / sum(df_[total_col])
            good_rate = sum(df_[good_col]) * 1.0 / sum(df_[total_col])

            if bad_rate in [0, 1]:
                return 0

            df_['bad_exp'] = df_[total_col].map(lambda x: x * bad_rate)
            df_['good_exp'] = df_[total_col].map(lambda x: x * good_rate)

            bad_zip = zip(df_['bad_exp'], df_[bad_col])
            good_zip = zip(df_['good_exp'], df_[good_col])
            bad_chiv = [(elem[1] - elem[0]) ** 2 / elem[0] for elem in bad_zip]
            good_chiv = [(elem[1] - elem[0]) ** 2 / elem[0] for elem in good_zip]
            chi = sum(bad_chiv) + sum(good_chiv)

            return chi

        if not x_list:
            x_list = df_.columns.tolist()

        data = Parallel(n_jobs=self.n_jobs)(delayed(chi_bin_single)(df_[col], n_bins) for col in x_list)  # 批量处理
        data_bin = pd.concat([i[0].astype(str) for i in data] + [Y], axis=1)  # 分箱后的数据集
        bins_dict = dict([(i[0].name, i[1]) for i in data])  # 分箱后的字典
        self.cut_off = copy.deepcopy(bins_dict)
        var_summary = self.calc_var_summary(data_bin, target=target)
        self.__bin_summary = var_summary.copy()

        return data_bin, bins_dict, var_summary

    def manual_bin(self, df, target='target', bins_dict=dict()):
        '''
        手动分箱
        :param df: 含有目标变量的数据集；不需要返回var_summary可以不需要目标变量，将函数中target部分注释
        :param target:目标值变量名称
        :param bins_dict:分箱字典, 形如{'D157': [-999, 1.0, 2.0, 3.0, 5.0, inf]}
        :return:
        '''
        if not isinstance(bins_dict, dict):
            assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

        df_ = df.copy()
        Y = df_[target]
        data_with_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(pd.cut)(df_[col], bins=bins, right=False, retbins=True) for col, bins in bins_dict.items())
        data_bin = pd.concat([i[0].astype(str) for i in data_with_bins] + [Y], axis=1)  # 分箱后的数据集
        b_dict = dict([(i[0].name, i[1].tolist()) for i in data_with_bins])
        if not operator.eq(bins_dict, b_dict):
            assert '传入的分箱和应用后的分箱不对等，请联系开发者'
        self.cut_off.update(bins_dict)

        var_summary = self.calc_var_summary(data_bin, target=target)

        return data_bin, b_dict, var_summary

    def unpack_tuple(self, x):
        if len(x) == 1:
            return x[0]
        else:
            return x

    def psi(self, no_base, base, return_frame=False):
        '''
        psi计算
        :param no_base: 非基准数据集
        :param base: 基准数据集
        :param return_frame: 是否返回详细的psi数据集
        :return:
        '''
        psi = list()
        frame = list()

        if isinstance(no_base, pd.DataFrame):
            for col in no_base:
                p, f = self.calc_psi(no_base[col], base[col])
                psi.append(p)
                frame.append(f)

            psi = pd.Series(psi, index=no_base.columns)

            frame = pd.concat(
                frame,
                keys=no_base.columns,
                names=['columns', 'id'],
            ).reset_index()
            frame = frame.drop(columns='id')
        else:
            psi, frame = self.calc_psi(no_base, base)

        res = (psi,)

        if return_frame:
            res += (frame,)

        return self.unpack_tuple(res)

    def calc_psi(self, no_base, base):
        '''
        psi计算的具体逻辑
        :param no_base: 非基准数据集
        :param base: 基准数据集
        :return:
        '''
        no_base_prop = pd.Series(no_base).value_counts(normalize=True, dropna=False)
        base_prop = pd.Series(base).value_counts(normalize=True, dropna=False)

        psi = np.sum((no_base_prop - base_prop) * np.log(no_base_prop / base_prop))

        frame = pd.DataFrame({
            'no_base': no_base_prop,
            'base': base_prop,
        })
        frame.index.name = 'value'

        return psi, frame.reset_index()

    def get_var_bin_woe(self):
        '''
        返回每个特征对应的分箱值对应的woe值
        :param df_var_bin_woe:  三列的dataframe。['var_name','range','woe']
        :return: {'D157': {Interval(-999.0, 1.0, closed='left'): -1.2835846793756913,
  Interval(1.0, 2.0, closed='left'): -0.9424139219729241,
  Interval(2.0, 3.0, closed='left'): -0.34860570638043403,
  Interval(3.0, 4.5, closed='left'): 0.36714548649924816,
  Interval(4.5, inf, closed='left'): 1.7735952385975287}
        '''
        self.var_bin_woe_dict = self.get_var_summary()[['var_name', 'range', 'woe']].groupby('var_name').apply(
            lambda df: df.set_index('range')['woe'].to_dict()).to_dict()
        return self.var_bin_woe_dict

    def transform(self, df, target='target', bins_dict={}):
        '''
        原始数据根据bins_dict进行分箱
        :param df:含有目标变量的数据集；不需要返回var_summary可以不需要目标变量，将函数中target部分注释
        :param target:目标值变量名称
        :param bins_dict:分箱字典, 形如{'D157': [-999, 1.0, 2.0, 3.0, 5.0, inf]}
        :return:
        '''

        if not isinstance(bins_dict, dict):
            assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

        if not bins_dict:
            bins_dict = self.cut_off
        elif isinstance(bins_dict, dict):
            assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

        df_ = df.copy()
        Y = df_[target]
        data_with_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(pd.cut)(df_[col], bins=bins, right=False, retbins=True) for col, bins in bins_dict.items())
        data_bin = pd.concat([i[0].astype(str) for i in data_with_bins] + [Y], axis=1)
        b_dict = dict([(i[0].name, i[1].tolist()) for i in data_with_bins])
        if not operator.eq(bins_dict, b_dict):
            assert '传入的分箱和应用后的分箱不对等，请联系开发者'

        var_summary = self.calc_var_summary(data_bin, target=target)
        self.__bin_summary = var_summary.copy()  # add

        return data_bin, b_dict, var_summary

    def save(self, path=None):
        '''
        保存分箱的字典；保存分箱后箱子对应的woe
        :param path: 保存路径，不填默认在运行程序的文件同等目录下
        :return:
        '''
        self.get_var_bin_woe()
        if path is None:
            path = sys.path[0]
        with open(path + '/continuous_var_bin_woe_dict.json', 'w') as f:
            json.dump(self.var_bin_woe_dict, f)

        with open(path + '/continuous_bins_dict.json', 'w') as f:
            json.dump(self.cut_off, f)

    def bin_method_select(self, df, x_list=[], target='target', method='dt', **kwargs):
        '''
        分箱方法选择
        :param df:含有目标变量及参与分箱的变量列表的数据集
        :param x_list: 参与分箱的变量列表
        :param target:目标值变量名称
        :param method（string类型）:使用哪种分箱方法；'dt'、'chi'、'equal_freq'、'kmeans'四种供选择
        :param kwargs:其它参数；special_value=[-999], min_sample_rate=0.05, n_bins=None, is_need_monotonic=True
        :return:
        '''

        df_ = df.copy()
        assert df_[target].isin([0, 1]).all(), 'ERROR: :-) {} :-) 目标变量不是0/1值，请检查！！！'.format(target)

        if method == 'dt':
            return self.dt_bin(df_, x_list=x_list, target=target, **kwargs)
        elif method == 'chi':
            return self.chi_bin(df_, x_list=x_list, target=target, **kwargs)
        elif method == 'equal_freq':
            return self.equal_freq_bin(df_, x_list=x_list, target=target, **kwargs)
        elif method == 'kmeans':
            return self.KMeans_bin(df_, x_list=x_list, target=target, **kwargs)
