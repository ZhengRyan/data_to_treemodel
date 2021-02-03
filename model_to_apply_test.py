#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: model_to_apply.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-07-23
'''

from model_code.utils import *
from model_code.woe_transformer import *
from sklearn.externals import joblib
import xgboost as xgb
from model_code import tree_selection
from model_code.feature_binning import *

if __name__ == '__main__':
    # =========================注意配置的修改=========================
    feature_type = 'lhpdat'  # 什么数据
    cust_id = 'apply_no'  # 主键
    target = 'target'  # 目标变量
    data_type = 'type'  # 区分数据集变量
    apply_time = 'apply_time'  # 时间

    client = 'lhp09'
    batch = 'p23'

    is_to_woe = False
    fillna_value = -999999  # 缺失值填充的值

    # =========================注意配置的修改=========================

    project_name = '{}{}'.format(client, batch)
    client_batch = '{}{}'.format(client, batch)
    project_dir = 'model_result_data/{}/{}/'.format(client, batch)
    output_dir = '{}model/{}/'.format(project_dir, feature_type)

    # 读取需要预测的数据集
    data = pd.read_csv('to_model_data/lhp_amount_rule.csv')  # 需要修改

    X_test_index = pd.read_csv('{}{}_{}_X_test_key_{}.csv'.format(
        output_dir, project_name, feature_type, cust_id))
    data_to_pred = data[data[cust_id].isin(X_test_index.iloc[:, 0])]

    var_value_woe = category_2_woe_load('{}'.format(output_dir))
    df_features = WoeTransformer().transform(data_to_pred, var_value_woe)
    df_features.set_index(cust_id, inplace=True)

    if is_to_woe:
        with open('{}/continuous_bins_dict.json'.format(output_dir), 'r') as f:
            cut_off = json.load(f)
        with open('{}/continuous_var_bin_woe_dict.json'.format(output_dir), 'r') as f:
            var_bin_woe_dict = json.load(f)
        df_features.loc[:, list(cut_off.keys())] = df_features.loc[:, list(cut_off.keys())].fillna(fillna_value)

        df_tmp = WoeTransformer(n_jobs=5).transform(df_features, var_bin_woe_dict, bins_dict=cut_off)
        df_features = pd.concat([df_features[set(df_features.columns) - set(cut_off.keys())], df_tmp], axis=1)

    print(df_features.head())

    # load model
    model = joblib.load('{}{}_{}_xgb.ml'.format(output_dir, project_name, feature_type))
    # to predict
    df_features['p'] = model.predict(xgb.DMatrix(df_features[model.feature_names]))
    df_features['score'] = df_features['p'].map(to_score)

    print(df_features['score'].describe())
    # df_features.to_csv('{}{}_{}_model_pred_result.csv'.format(
    #     output_dir, project_name, feature_type, cust_id))

    if target in df_features.columns:
        print('auc is : ')
        print(tree_selection.get_roc_auc_score(df_features[target], df_features['score']))
        print('ks is : ')
        print(tree_selection.get_ks(df_features[target], df_features['score']))


# In[ ]:




