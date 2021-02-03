#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: tree_selection.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020/5/7
'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model_code.utils import get_ks, get_roc_auc_score


def sigle_feature_fit_model(train_data=(), test_data=(), is_noise=False, is_only_return_auc=True):
    X_train, y_train = train_data
    X_test, y_test = test_data

    l = []
    for i in X_train:
        # xclf = xgb.XGBClassifier(colsample_bytree=0.3, seed=123, random_state=1234)
        xclf = xgb.XGBClassifier(**{
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        })
        xclf.fit(X_train[[i]], y_train)
        pred_y_train = xclf.predict_proba(X_train[[i]])[:, 1]
        auc = get_roc_auc_score(y_train, pred_y_train)
        ks = get_ks(y_train, pred_y_train)
        l.append((i, auc, ks))

    var_auc_ks_df = pd.DataFrame(l, columns=['features', 'auc', 'ks'])

    var_auc_ks_df = var_auc_ks_df[var_auc_ks_df['auc'] > 0.51]
    # var_auc_ks_df = var_auc_ks_df[var_auc_ks_df['auc'] > 0.1]
    # var_auc_ks_df.sort_values(by=['auc', 'ks'], ascending=False, inplace=True)
    # print(var_auc_ks_df)
    if is_noise:
        var_auc_ks_df.to_excel('xgb_not_del_corr_var_auc_ks_df.xlsx')
    else:
        var_auc_ks_df.to_excel('xgb_var_auc_ks_df.xlsx')
    if is_only_return_auc:
        return var_auc_ks_df.drop(columns='ks')
    else:
        return var_auc_ks_df


def sigle_feature_auc_ks(train_data=(), test_data=(), is_noise=False, is_only_return_auc=True):
    X_train, y_train = train_data
    X_test, y_test = test_data

    l = []
    for i in X_train:
        auc = get_roc_auc_score(y_train, X_train[i])
        ks = get_ks(y_train, X_train[i])
        l.append((i, auc, ks))

    var_auc_ks_df = pd.DataFrame(l, columns=['features', 'auc', 'ks'])
    # var_auc_ks_df.sort_values(by=['auc', 'ks'], ascending=False, inplace=True)
    # print(var_auc_ks_df)
    # if is_noise:
    #     var_auc_ks_df.to_excel('process_after_data/xgb_not_del_corr_var_auc_ks_df_zhijie.xlsx')
    # else:
    #     var_auc_ks_df.to_excel('process_after_data/xgb_var_auc_ks_df_zhijie.xlsx')
    if is_only_return_auc:
        return var_auc_ks_df.drop(columns='ks')
    else:
        return var_auc_ks_df


def change_col_subsample_fit_model(train_data=(), test_data=()):
    X_train, y_train = train_data
    X_test, y_test = test_data

    colsample_bytree = [i / 10 for i in range(3, 11)]
    subsample = [i / 10 for i in range(3, 11)]

    imp_l = []
    for i in range(8):
        params = {
            'min_child_weight': 10,
            'subsample': subsample[i],
            'colsample_bytree': colsample_bytree[i],
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        imp = pd.DataFrame(list(model.get_booster().get_score().items()),
                           columns=['features', 'feature_importances']).set_index('features')
        imp_l.append(imp)

        pred_y_train = model.predict_proba(X_train)[:, 1]
        pred_y_test = model.predict_proba(X_test)[:, 1]

        print('subsample is {} and colsample_bytree is {} model result is : '.format(subsample[i], colsample_bytree[i]))
        print('xgb train auc is ：', get_roc_auc_score(y_train, pred_y_train))
        print('xgb train ks is ：', get_ks(y_train, pred_y_train))
        print('xgb test auc is ：', get_roc_auc_score(y_test, pred_y_test))
        print('xgb test ks is ：', get_ks(y_test, pred_y_test))

    imp_df = pd.concat(imp_l, axis=1)
    imp_df['mean_imp'] = imp_df.mean(axis=1)
    # imp_df.to_excel('imp_df_all_mean.xlsx')
    imp_df.drop(columns=['feature_importances'], inplace=True)
    # imp_df.to_excel('imp_df_mean.xlsx')
    imp_df.reset_index(inplace=True)
    return imp_df


def kfold_xgb_model(train_data=(), is_noise=False, cv=StratifiedKFold(10, shuffle=True)):
    X, y = train_data
    cv_data = cv.split(X, y)

    train_auc_l = []
    valid_auc_l = []
    feature_imp = []
    for fold_num, (train_i, valid_i) in enumerate(cv_data):
        X_train, y_train = X.iloc[train_i], y.iloc[train_i]
        X_valid, y_valid = X.iloc[valid_i], y.iloc[valid_i]

        # print(X.shape)
        # print(X_train.shape)
        # print(X_valid.shape)

        model = xgb.XGBClassifier(**{
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        })
        model.fit(X_train, y_train)

        pred_y_train = model.predict_proba(X_train)[:, 1]
        pred_y_valid = model.predict_proba(X_valid)[:, 1]

        train_auc = get_roc_auc_score(y_train, pred_y_train)
        valid_auc = get_roc_auc_score(y_valid, pred_y_valid)

        feature_importance = pd.DataFrame(list(model.get_booster().get_score().items()),
                                          columns=['features', 'feature_importances'])
        # feature_importance.sort_values(by='feature_importances', ascending=False,
        #                                inplace=True)
        feature_importance.set_index('features', inplace=True)
        # feature_importance.reset_index(inplace=True)
        # print(feature_importance)
        feature_imp.append(feature_importance)

        print('Fold {} , train auc is {}, valid_auc is {}'.format(fold_num, train_auc, valid_auc))

        train_auc_l.append(train_auc)
        valid_auc_l.append(valid_auc)

    train_mean_auc = np.array(train_auc_l).mean()
    valid_mean_auc = np.array(valid_auc_l).mean()
    feature_imp_all = pd.concat(feature_imp, axis=1)
    print('train_mean_auc is : {}'.format(train_mean_auc))
    print('valid_mean_auc is : {}'.format(valid_mean_auc))
    # print('feature_imp_all is : {}'.format(feature_imp_all))
    # feature_imp_all.to_excel('process_after_data/feature_imp_all.xlsx')

    feature_imp_all['mean_imp'] = feature_imp_all.mean(axis=1)
    # feature_imp_all.to_excel('process_after_data/feature_imp_all_mean.xlsx')
    feature_imp_all.drop(columns='feature_importances', inplace=True)
    # feature_imp_all.index.name = 'features'
    feature_imp_all.reset_index(inplace=True)
    return feature_imp_all


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def drop_corr(frame, by='auc', threshold=0.95, return_drop=False):
    if not isinstance(by, (str, pd.Series)):

        if isinstance(by, pd.DataFrame):
            by = pd.Series(by.iloc[:, 1].values, index=by.iloc[:, 0].values)
            # by = pd.Series(by.iloc[:, 1].values, index=frame.columns)
        else:
            by = pd.Series(by, index=frame.columns)

    # 给重要性排下序
    by.sort_values(ascending=False, inplace=True)

    # df = frame.copy()

    by.index = by.index.astype(type(frame.columns.to_list()[0]))
    df_corr = frame[by.index.to_list()].fillna(-999).corr().abs()

    ix, cn = np.where(np.triu(df_corr.values, 1) > threshold)

    del_all = []

    if len(ix):

        for i in df_corr:

            if i not in del_all:
                # 找出与当前特征的相关性大于域值的特征
                del_tmp = df_corr[i][(df_corr[i] > threshold) & (df_corr[i] != 1)].index.to_list()

                # 比较当前特征与需要删除的特征的特征重要性
                if del_tmp:
                    by_tmp = by.loc[del_tmp]
                    del_l = by_tmp[by_tmp <= by.loc[i]].index.to_list()
                    del_all.extend(del_l)

    del_f = list(set(del_all))

    r = frame.drop(columns=del_f)

    res = (r,)
    if return_drop:
        res += (del_f,)

    return unpack_tuple(res)


def forward_corr_delete(df, col_list):
    corr_list = []
    corr_list.append(col_list[0])
    delete_col = []
    # 根据特征重要性的大小进行遍历
    for col in col_list[1:]:
        corr_list.append(col)
        corr = df.loc[:, corr_list].corr()
        corr_tup = [(x, y) for x, y in zip(corr[col].index, corr[col].values)]
        corr_value = [y for x, y in corr_tup if x != col]
        # 若出现相关系数大于0。65，则将该特征剔除
        if len([x for x in corr_value if abs(x) >= 0.5]) > 0:
            delete_col.append(col)

    select_corr_col = [x for x in col_list if x not in delete_col]
    return select_corr_col


def xgb_model(train_data=(), test_data=(), is_noise=False):
    X_train, y_train = train_data
    X_test, y_test = test_data

    params = {
        'min_child_weight': 10,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'nthread': -1
    }
    xclf = xgb.XGBClassifier(**params)
    # xclf = xgb.XGBClassifier()
    xclf.fit(X_train, y_train)

    print('===============xgb 不同的重要性===============')
    l = []
    print('weight')
    importance_type = 'weight'
    feature_importance = pd.DataFrame(list(xclf.get_booster().get_score(importance_type=importance_type).items()),
                                      columns=['features', 'feature_importances_{}'.format(importance_type)])
    feature_importance.sort_values(by='feature_importances_{}'.format(importance_type), ascending=False, inplace=True)
    # feature_importance.set_index('features', inplace=True)
    feature_importance.reset_index(inplace=True)
    print(feature_importance)
    l.append(feature_importance)

    print('gain')
    importance_type = 'gain'
    feature_importance = pd.DataFrame(list(xclf.get_booster().get_score(importance_type=importance_type).items()),
                                      columns=['features', 'feature_importances_{}'.format(importance_type)])
    feature_importance.sort_values(by='feature_importances_{}'.format(importance_type), ascending=False, inplace=True)
    # feature_importance.set_index('features', inplace=True)
    feature_importance.reset_index(inplace=True)
    # print(feature_importance)
    l.append(feature_importance)

    print('cover')
    importance_type = 'cover'
    feature_importance = pd.DataFrame(list(xclf.get_booster().get_score(importance_type=importance_type).items()),
                                      columns=['features', 'feature_importances_{}'.format(importance_type)])
    feature_importance.sort_values(by='feature_importances_{}'.format(importance_type), ascending=False, inplace=True)
    # feature_importance.set_index('features', inplace=True)
    feature_importance.reset_index(inplace=True)
    # print(feature_importance)
    l.append(feature_importance)

    # print('total_gain')
    # importance_type = 'total_gain'
    # feature_importance = pd.DataFrame(list(xclf.get_booster().get_score(importance_type=importance_type).items()),
    #                                   columns=['features', 'feature_importances_{}'.format(importance_type)])
    # feature_importance.sort_values(by='feature_importances_{}'.format(importance_type), ascending=False, inplace=True)
    # # feature_importance.set_index('features', inplace=True)
    # feature_importance.reset_index(inplace=True)
    # print(feature_importance)
    # l.append(feature_importance)
    #
    # print('total_cover')
    # importance_type = 'total_cover'
    # feature_importance = pd.DataFrame(list(xclf.get_booster().get_score(importance_type=importance_type).items()),
    #                                   columns=['features', 'feature_importances_{}'.format(importance_type)])
    # feature_importance.sort_values(by='feature_importances_{}'.format(importance_type), ascending=False, inplace=True)
    # # feature_importance.set_index('features', inplace=True)
    # feature_importance.reset_index(inplace=True)
    # print(feature_importance)
    # l.append(feature_importance)

    five_importance = pd.concat(l, axis=1)
    if is_noise:
        five_importance.to_excel('xgb_not_del_corr_five_importance.xlsx')
    else:
        five_importance.to_excel('xgb_five_importance.xlsx')

    print(np.mean(
        cross_val_score(estimator=xclf, X=X_train, y=y_train, scoring='accuracy',
                        cv=StratifiedKFold(5, random_state=123))))

    pred_y_train = xclf.predict_proba(X_train)[:, 1]
    pred_y_test = xclf.predict_proba(X_test)[:, 1]

    print('xgb train auc is ：', get_roc_auc_score(y_train, pred_y_train))
    print('xgb train ks is ：', get_ks(y_train, pred_y_train))
    print('xgb test auc is ：', get_roc_auc_score(y_test, pred_y_test))
    print('xgb test ks is ：', get_ks(y_test, pred_y_test))


if __name__ == '__main__':
    pass
