#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: utils.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# auc
def get_roc_auc_score(target, y_pred):
    if target.nunique() != 2:
        raise ValueError('the target is not 2 classier target')
    else:
        return roc_auc_score(target, y_pred)


# ks
def get_ks(target, y_pred):
    df = pd.DataFrame({
        'y_pred': y_pred,
        'target': target,
    })
    df = df.sort_values(by='y_pred', ascending=False)
    df['good'] = 1 - df['target']
    df['bad_rate'] = df['target'].cumsum() / df['target'].sum()
    df['good_rate'] = df['good'].cumsum() / df['good'].sum()
    df['ks'] = df['bad_rate'] - df['good_rate']
    return max(abs(df['ks']))


def get_splitted_data(df_selected, target, selected_features):
    X = {}
    y = {}

    X['all'] = df_selected[selected_features]
    y['all'] = df_selected[target]

    for name, df in df_selected.groupby('type'):
        X[name] = df[selected_features]
        y[name] = df[target]

    if not X.__contains__('oot'):
        X['oot'] = None
        y['oot'] = None

    return X['all'], y['all'], X['train'], y['train'], X['test'], y['test'], X['oot'], y['oot']


def to_score(x):
    import math
    if x <= 0.001:
        x = 0.001
    elif x >= 0.999:
        x = 0.999

    A = 404.65547022
    B = 72.1347520444
    result = int(round(A - B * math.log(x / (1 - x))))

    if result < 0:
        result = 0
    if result > 1200:
        result = 1200
    result = 1200 - result
    return result


def psi_statis(df_src, splitted_types, scores):
    def bin_psi(x, y):

        if pd.isnull(y) or y == 0 or pd.isnull(x) or x == 0:
            return None
        else:
            return (x - y) * math.log(x / y)

    if 'train' not in splitted_types:
        print('Error: failt to get psi, for train is not in splitted_types')
        return

    bins = list(range(300, 951, 50))
    l = []
    for (client_batch, splitted_type), df_type in df_src.groupby(['client_batch', 'type']):
        for score in scores:
            df_score = df_type[df_type[score].notnull()]
            df = pd.cut(df_score[score].map(to_score), bins=bins, right=False).value_counts().map(
                lambda v: v / df_score.shape[0] if df_score.shape[0] > 0 else np.nan).to_frame('pct')
            df.index.name = 'bin'
            df.index = df.index.astype(str)
            df = df.reset_index()

            df['client_batch'] = client_batch
            df['type'] = splitted_type
            df['feature'] = score

            l.append(df)

    df_psi_detail = pd.concat(l, ignore_index=True).pivot_table(index=['client_batch', 'feature', 'bin'],
                                                                columns='type', values='pct')
    df_psi_detail.columns = [s + '_pct' for s in df_psi_detail.columns.format()]
    df_psi_detail = df_psi_detail.reset_index()

    for splitted_type in filter(lambda s: s != 'train', splitted_types):
        df_psi_detail['train_{}_psi'.format(splitted_type)] = df_psi_detail.apply(
            lambda r: bin_psi(r['train_pct'], r[splitted_type + '_pct']), axis=1)

    psi_col = list(filter(lambda col: '_psi' in col, df_psi_detail.columns.format()))
    df_psi = df_psi_detail.groupby(['client_batch', 'feature']).sum()[psi_col].reset_index()

    df_psi_detail_sum = df_psi_detail.drop(labels='bin', axis=1).groupby(
        ['client_batch', 'feature']).sum().reset_index()
    df_psi_detail_sum['bin'] = '[sum]'

    df_psi_detail = pd.concat([df_psi_detail, df_psi_detail_sum], ignore_index=True).sort_values(
        ['client_batch', 'feature'])
    df_psi_detail = pd.DataFrame(df_psi_detail, columns=['client_batch', 'feature', 'bin',
                                                         'train_pct', 'test_pct', 'oot_pct', 'train_test_psi',
                                                         'train_oot_psi'])

    return df_psi, df_psi_detail


def train_test_split_(df_src, target='y_target', test_size=0.3):
    """
    样本切分函数.先按target分类，每类单独切成train/test，再按train/test合并，
    使得train/test的badrate能高度一致
    :param df_src:
    :param target:
    :param test_size:
    :return:
    """

    l = [[], [], [], []]
    for target_value, X in df_src.groupby(target):

        X[target] = target_value

        row = train_test_split(X.drop(labels=target, axis=1), X[target], test_size=test_size, random_state=1234)

        for i in range(0, 4):
            l[i].append(row[i])

    list_df = []
    for i in range(0, 4):
        list_df.append(pd.concat(l[i]))

    return tuple(list_df)


def split_data_type(df, key_col='tdid', target='target', apply_time='apply_time', test_size=0.3):
    df_id = df.copy()
    if df_id[target].isin([0, 1]).all():
        print('样本y值在0，1')
    else:
        print('\033[0;31m样本y值不在0，1之间，请检查！！！\033[0m')

    print('样本情况：', df_id.shape)
    df_id.drop_duplicates(subset=key_col, inplace=True)
    print('分布情况：', df_id.groupby(target)[key_col].count().sort_index())
    # df_id.groupby(target)['tdid'].count().sort_index().to_excel(
    #     '{}{}_id_distributed.xlsx'.format(data_dir, client_batch))
    print('样本drop_duplicates情况：', df_id.shape)

    df_id = df_id.loc[df_id[target].isin([0, 1])]
    print('样本y值在0，1的样本情况：', df_id.shape)

    # ---------查看各月badrate---------------------
    df_id['apply_month'] = df_id[apply_time].map(lambda s: s[:7])
    print(df_id.groupby('apply_month').describe()[target])

    # ---------样本划分----------------------------
    ##需要oot
    # df_selected = df_id #can filter records here
    # # df_oot = df_selected[df_selected['apply_time']>= '2019-04-01']
    # # X_train = df_selected[df_selected['apply_time']<= '2019-01-31']
    # # X_test = df_selected[(df_selected['apply_time']> '2019-01-31') & (df_selected['apply_time']< '2019-04-01')]

    # df_oot = df_selected[df_selected['apply_time']>= '2019-03-01']
    # X_train = df_selected[df_selected['apply_time']<= '2018-12-31']
    # X_test = df_selected[(df_selected['apply_time']> '2018-12-31') & (df_selected['apply_time']< '2019-03-01')]

    # #X_train, X_test, y_train, y_test = geo_train_test_split(df_not_oot,label=label)

    # df_id.loc[df_oot.index,'type'] = 'oot'
    ##需要oot

    # 不需要oot的时候运行下面这一行代码
    X_train, X_test, y_train, y_test = train_test_split_(df_id, target=target, test_size=test_size)
    # X_train, X_test, y_train, y_test = train_test_split(df_id.drop(columns=target), df_id[target], test_size=test_size,
    #                                                     random_state=123)
    # 不需要oot的时候运行下面这一行代码

    df_id.loc[X_train.index, 'type'] = 'train'
    df_id.loc[X_test.index, 'type'] = 'test'

    print(df_id.groupby('type').describe()[target])

    # ----------输出---------------------------------
    # df_id.to_csv(data_dir + '{}_split.csv'.format(client_batch), index=False)
    return df_id


def select_features_dtypes(df, exclude=None):
    '''
    根据数据集，筛选出数据类型
    :param df: 数据集
    :param exclude: 排除不需要参与筛选的列
    :return:三个list
    '''
    if exclude is not None:
        df = df.drop(columns=exclude)
    # 筛选出数值类型列
    numeric_df = df.select_dtypes([np.number])

    no_numeric_df = df.select_dtypes(include=['object'])
    # 将object类型的列尝试转成时间类型
    dates_objs_df = no_numeric_df.apply(pd.to_datetime, errors='ignore')
    # 筛选出字符类型列
    objs_df = dates_objs_df.select_dtypes(include=['object'])
    # 筛选出时间类型列
    dates_df = list(set(dates_objs_df.columns) - set(objs_df.columns))

    assert len(numeric_df.columns) + len(objs_df.columns) + len(dates_df) == df.shape[1]

    return numeric_df.columns.tolist(), objs_df.columns.tolist(), dates_df


# def category_2_woe(df, category_cols=[], target='target', topN=10):
#     '''
#     方法说明。占比前10的类别会单独转成woe值。其余类别归为一个大类对应一个woe值
#     :param df:
#     :param category_cols:
#     :param target:
#     :param topN:
#     :return:
#     '''
#     var_value_woe = {}
#
#     df_ = df.copy()
#     for i in category_cols:
#         df_[i].fillna('miss', inplace=True) #先补缺
#         vc = df_[i].value_counts(dropna=False)
#         top_value = vc.sort_values(ascending=False).index[:topN]
#         #print(df_[i].map(lambda x: print(not isinstance(x, str))))
#         #df_[i] = df_[i].map(lambda x: x if (x is np.nan or pd.isna(x) or pd.isnull(x) or x is None or x in top_value or str(x).strip() == '' or str(x).strip() == 'nan' or (not isinstance(x, str))) else 'else')
#         #df_[i] = df_[i].map(lambda x: x if (x in top_value or x == 'miss') else 'other')   #缺失再少也归为一类
#         df_[i] = df_[i].map(lambda x: x if x in top_value else 'other')
#         bin_g = df_.groupby(by=i)[target].agg({'total_cnt': 'count', 'bad_cnt': 'sum'})
#         bin_g['good_cnt'] = bin_g['total_cnt'] - bin_g['bad_cnt']
#         bin_g['bad_rate'] = bin_g['bad_cnt'] / sum(bin_g['bad_cnt'])
#         bin_g['good_rate'] = bin_g['good_cnt'] / sum(bin_g['good_cnt'])
#         bin_g['good_rate'].replace({0: 0.0000000001}, inplace=True)  # good_rate为0的情况下，woe算出来是-inf。即将0使用一个极小数替换
#         bin_g['woe'] = bin_g.apply(lambda x: 0.0 if x['bad_rate'] == 0 else np.log(x['good_rate'] / x['bad_rate']),
#                                    axis=1)
#
#         value_woe = bin_g['woe'].to_dict()
#         var_value_woe[i] = value_woe
#
#     return var_value_woe


def category_2_woe(df, category_cols=[], target='target'):
    '''
    方法说明。每个类别都会转成woe值。缺失值不转，即还是为缺失值。在考虑到未来如果有新类别，给予other对应woe为0
    :param df:
    :param category_cols:
    :param target:
    :return:
    '''
    var_value_woe = {}
    for i in category_cols:
        # bin_g = df.groupby(by=i)[target].agg({'total_cnt': 'count', 'bad_cnt': 'sum'})
        # https://stackoverflow.com/questions/60229375/solution-for-specificationerror-nested-renamer-is-not-supported-while-agg-alo
        bin_g = df.groupby(by=i)[target].agg([('total_cnt', 'count'), ('bad_cnt', 'sum')])
        bin_g['good_cnt'] = bin_g['total_cnt'] - bin_g['bad_cnt']
        bin_g['bad_rate'] = bin_g['bad_cnt'] / sum(bin_g['bad_cnt'])
        bin_g['good_rate'] = bin_g['good_cnt'] / sum(bin_g['good_cnt'])
        bin_g['good_rate'].replace({0: 0.0000000001}, inplace=True)  # good_rate为0的情况下，woe算出来是-inf。即将0使用一个极小数替换
        bin_g['woe'] = bin_g.apply(lambda x: 0.0 if x['bad_rate'] == 0 else np.log(x['good_rate'] / x['bad_rate']),
                                   axis=1)

        value_woe = bin_g['woe'].to_dict()
        value_woe['other'] = 0  # 未来有新类别的情况下，woe值给予0
        var_value_woe[i] = value_woe

    return var_value_woe


# def category_2_woe(df, category_cols=[], target='target'):
#     '''
#     方法说明。每个类别都会转成woe值。缺失值不转，即还是为缺失值
#     :param df:
#     :param category_cols:
#     :param target:
#     :return:
#     '''
#     var_value_woe = {}
#     for i in category_cols:
#         bin_g = df.groupby(by=i)[target].agg({'total_cnt': 'count', 'bad_cnt': 'sum'})
#         bin_g['good_cnt'] = bin_g['total_cnt'] - bin_g['bad_cnt']
#         bin_g['bad_rate'] = bin_g['bad_cnt'] / sum(bin_g['bad_cnt'])
#         bin_g['good_rate'] = bin_g['good_cnt'] / sum(bin_g['good_cnt'])
#         bin_g['good_rate'].replace({0: 0.0000000001}, inplace=True)  # good_rate为0的情况下，woe算出来是-inf。即将0使用一个极小数替换
#         bin_g['woe'] = bin_g.apply(lambda x: 0.0 if x['bad_rate'] == 0 else np.log(x['good_rate'] / x['bad_rate']),
#                                    axis=1)
#
#         value_woe = bin_g['woe'].to_dict()
#         var_value_woe[i] = value_woe
#
#     return var_value_woe


def category_2_woe_save(var_value_woe, path=None):
    if path is None:
        path = sys.path[0]

    with open(path + 'category_var_value_woe.json', 'w') as f:
        json.dump(var_value_woe, f)


def category_2_woe_load(path=None):
    with open(path + 'category_var_value_woe.json', 'r') as f:
        var_value_woe = json.load(f)
    return var_value_woe


def filter_miss(df, miss_threshold=0.9):
    '''

    :param df: 数据集
    :param miss_threshold: 缺失率大于等于该阈值的变量剔除
    :return:
    '''
    names_list = []
    for name, series in df.items():
        n = series.isnull().sum()
        miss_q = n / series.size
        if miss_q < miss_threshold:
            names_list.append(name)
    return names_list
