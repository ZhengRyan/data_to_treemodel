#!/usr/bin/env python
# coding: utf-8

# In[37]:


# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: model_fit_v3.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-07-26
'''

import os
import warnings
from datetime import datetime

import joblib

from model_code import tree_selection
from model_code.woe_transformer import *
from model_code.feature_binning import *
from model_code.bayes_opt_tuner import classifiers_model
from model_code.utils import *
from model_code.detector import detect

warnings.filterwarnings('ignore')

from model_code.logger_utils import Logger

log = Logger(level='info', name=__name__).logger

# =========================step 1 相关配置=========================
log.info('step 1 相关配置')
feature_type = 'lhpdat'  # 什么数据
cust_id = 'apply_no'  # 主键
target = 'target'  # 目标变量
data_type = 'type'  # 区分数据集变量
apply_time = 'apply_time'  # 时间

client = 'lhp09'
batch = 'p23'

to_model_var_num = 30  # 不限制的话修改为None
is_model_data_to_woe = False  # 喂入模型的数据是否需要转化为woe值，False不需要，即原始数据入模型
fillna_value = -999999  # 缺失值填充的值

# 阈值配置
exclude_cols = [apply_time, cust_id, target, data_type, 'apply_month']
feature_missing_threshould = 0.95  # 缺失率大于等于该阈值的变量剔除

# 用于训练模型的数据
label_encoder_dict = {}
to_model_data_path = '/Users/ryanzheng/PycharmProjects/data_to_treemodel_v1/to_model_data/lhp_amount_rule.csv'

# =========================后续代码基本可以不用动=========================


# 基本不用动
project_name = '{}{}'.format(client, batch)
client_batch = '{}{}'.format(client, batch)
project_dir = 'model_result_data/{}/{}/'.format(client, batch)
output_dir = '{}model/{}/'.format(project_dir, feature_type)

os.makedirs(project_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(project_dir + 'data/score/', exist_ok=True)
os.makedirs(project_dir + 'data/xgb_score/', exist_ok=True)
# 基本不用动
# =========================相关配置=========================


# In[38]:


# =========================step 2 读取数据集=========================
log.info('step 2 开始读取数据集')
# 读取宽表数据
log.info('读取样本&特征数据集：{}|{}|{}为样本数据，其他为特征数据'.format(cust_id, apply_time, target))
all_data = pd.read_csv(to_model_data_path)

# drop_cols = ['xy_black_version', 'tzre_version']
# all_data.drop(columns=drop_cols, axis=1, inplace=True)
all_data.drop(['applthst_loan_amount', 'tzre_report_info_report_no', 'xy_black_trade_no', 'tzre_id', 'xy_black_version',
               'tzre_version', 'tzre_bi_phone_number'], axis=1, inplace=True)

all_data.set_index(cust_id, inplace=True)
selected_features = all_data.columns.format()
selected_features = list(set(selected_features) - set(exclude_cols))
log.info('特征的个数：{}'.format(len(selected_features)))

# =========================读取字典进行重命名=========================
#     ##读取字典进行重命名
#     fea_dict_df = pd.read_excel('/home/marketingscore/ryanzheng/fit_model_project/新特征数据字典v3.xlsx')
#     fea_dict = fea_dict_df[['feature_code','feature_id']].set_index('feature_code')['feature_id'].to_dict()
#     all_data.rename(columns=fea_dict, inplace=True)

#     selected_features = all_data.columns.format()
#     selected_features = list(set(selected_features) - set(exclude_cols))
# #     if exclude_vars:
# #         selected_features = list(set(selected_features) - set(exclude_vars))

#     ##仅使用数据字典中有的变量
#     fea_dict_df_list = fea_dict_df['feature_id'].tolist()
#     selected_features = list(set(selected_features).intersection(set(fea_dict_df_list)))
#     print(len(selected_features))
#     ##仅使用数据字典中有的变量

# =========================读取字典进行重命名=========================


# 删除特征全为空的样本量
log.info('删除特征全为空的样本量')
print('删除特征全为空的样本之前的数据集行列：', all_data.shape)
all_data.dropna(subset=selected_features, how='all', inplace=True)
print('删除特征全为空的样本之后的数据集行列：', all_data.shape)

log.info('样本数据集情况：')
log.info(all_data[target].value_counts())
# =========================读取数据集=========================

log.info('EDA，整体数据探索性数据分析')
all_data_eda = detect(all_data)
all_data_eda.to_excel('{}{}_{}_all_data_eda.xlsx'.format(
    output_dir, project_name, feature_type))

# =========================step 3 划分训练集和测试集=========================
log.info('step 3 划分训练集和测试集')
if data_type not in all_data.columns:
    df_sample = all_data[[target, apply_time]]
    df_sample.reset_index(inplace=True)

    # 随机切分train、test
    df_sample = split_data_type(df_sample, key_col=cust_id, target=target, apply_time=apply_time, test_size=0.25)
    df_sample.to_csv(project_dir + 'data/{}_split.csv'.format(client_batch), index=False)

    #         #按时间切分
    #         df_oot = df_sample[df_sample['apply_time']>= '2020-04-01']
    #         X_train = df_sample[df_sample['apply_time']<= '2020-02-01']
    #         X_test = df_sample[(df_sample['apply_time']> '2020-02-01') & (df_sample['apply_time']< '2020-04-01')]

    #         df_sample.loc[df_oot.index,'type'] = 'oot'
    #         df_sample.loc[X_train.index,'type'] = 'train'
    #         df_sample.loc[X_test.index,'type'] = 'test'

    df_sample.to_csv(project_dir + 'data/{}_split.csv'.format(client_batch), index=False)
    df_sample.set_index(cust_id, inplace=True)
    print(df_sample['type'].value_counts())

# In[39]:


# 将数据集类别和数据集合并
# df_sample = all_data[[target, apply_time, data_type]]
all_data = pd.merge(df_sample[['type']], all_data, left_index=True, right_index=True, how='inner')

log.info('分开训练集和测试集为两个df')
train_data = all_data[all_data['type'] == 'train']
# test_data = all_data[all_data['type'] == 'test']

log.info('EDA，训练集探索性数据分析')
detect(train_data).to_excel('{}{}_{}_train_data_eda.xlsx'.format(
    output_dir, project_name, feature_type))
#     detect(test_data).to_excel('{}{}_{}_test_data_eda.xlsx'.format(
#         output_dir, project_name, feature_type))

# =========================step 4 初筛=========================
log.info('step 4 变量初筛')
# selected_features = train_data_eda[train_data_eda['missing_q'] <= 0.95].index.to_list()
print('删除缺失率前变量数量：', len(selected_features))
selected_features = filter_miss(train_data[selected_features], miss_threshold=feature_missing_threshould)
print('删除缺失率后变量数量：', len(selected_features))
train_data = train_data[selected_features + [target]]
# test_data = test_data[selected_features + [target]]
# =========================初筛=========================


# =========================step 5 数据处理=========================
log.info('step 5 数据woe处理')

# 离散变量数据处理
# selected_features = list(set(selected_features) - set(exclude_cols))
continuous_cols, category_cols, date_cols = select_features_dtypes(train_data[selected_features])

train_data.loc[:, continuous_cols] = train_data.loc[:, continuous_cols].fillna(fillna_value)
# test_data.loc[:, continuous_cols] = test_data.loc[:, continuous_cols].fillna(fillna_value)
all_data.loc[:, continuous_cols] = all_data.loc[:, continuous_cols].fillna(fillna_value)
# data.loc[:, continuous_cols] = data.loc[:, continuous_cols].fillna(-999)

# =========================labelencode=========================
#     def category_to_labelencoder(data, labelencoder=[]):
#         label_encoder_dict = {}
#         le = LabelEncoder()
#         for col in labelencoder:
#             print('{} in process!!!'.format(col))
#             data[col] = le.fit_transform(data[col].values)
#             number = [i for i in range(0, len(le.classes_))]
#             key = list(le.inverse_transform(number))
#             label_encoder_dict[col] = dict(zip(key, number))
#         return label_encoder_dict


#     def category_to_labelencoder_apply(data, labelencoder_dict={}):
#         for col, mapping in labelencoder_dict.items():
#             print('{} in process!!!'.format(col))
#             data[col] = data[col].map(mapping).fillna(-1)
#             data[col] = data[col].astype(int)


#     if category_cols:
#         train_data.loc[:, category_cols] = train_data.loc[:, category_cols].fillna('-1007')
#         all_data.loc[:, category_cols] = all_data.loc[:, category_cols].fillna('-1007')
#         label_encoder_dict = category_to_labelencoder(train_data, category_cols)
#         category_to_labelencoder_apply(all_data, label_encoder_dict)

# =========================labelencode=========================


if category_cols and not label_encoder_dict:
    log.info('step 5.1 类别变量数据处理')
    # train_data.loc[:, category_cols] = train_data.loc[:, category_cols].fillna('miss')
    # test_data.loc[:, category_cols] = test_data.loc[:, category_cols].fillna('miss')

    var_value_woe = category_2_woe(train_data, category_cols, target=target)
    category_2_woe_save(var_value_woe, '{}'.format(output_dir))
    # var_value_woe = category_2_woe_load('{}'.format(output_dir))
    train_data = WoeTransformer().transform(train_data, var_value_woe)
    # test_data = WoeTransformer().transform(test_data, var_value_woe)
    all_data = WoeTransformer().transform(all_data, var_value_woe)

# 离散变量数据处理


# In[40]:


if is_model_data_to_woe:
    log.info('将箱子转woe')
    log.info('============入模数据需要转化为woe值===========')
    #         train_data_to_model = WoeTransformer().transform(train_data_bin, fb.get_var_bin_woe())
    #         test_data_to_model = WoeTransformer().transform(test_data_bin, fb.get_var_bin_woe())
    #all_data_to_model = WoeTransformer().transform(all_data_bin, fb.get_var_bin_woe())
else:
    log.info('============入模数据不需要转化为woe值===========')
    #         train_data_to_model = train_data.copy()
    #         test_data_to_model = test_data.copy()
    all_data_to_model = all_data.copy()


# In[41]:


def statistics_model_result(all_data=pd.DataFrame()):
    # ===========================step 6 统计=================================
    all_data['score'] = all_data[feature_type].map(lambda v: to_score(v))
    log.info('模型相关结果统计！！！')
    df_splitted_type_auc_ks = all_data.groupby(data_type).apply(
        lambda df: pd.Series({'auc': get_roc_auc_score(df[target], df['score']),
                              'ks': get_ks(df[target], df['score'])}))
    df_splitted_type_auc_ks = df_splitted_type_auc_ks.reindex(['train', 'test', 'oot', 'cv'])

    log.info('模型效果：')
    print(df_splitted_type_auc_ks)

    all_data['month'] = all_data[apply_time].map(lambda s: s[:7])
    df_monthly_auc_ks = all_data.groupby('month').apply(
        lambda df: pd.Series({'auc': get_roc_auc_score(df[target], df['score']),
                              'ks': get_ks(df[target], df['score'])}))
    del all_data['month']
    log.info('不同月份的模型效果：')
    print(df_monthly_auc_ks)

    df_desc = all_data[[feature_type, 'score']].describe()
    df_desc.loc['coverage'] = df_desc.loc['count'] / all_data.shape[0]
    log.info('分数describe')
    print(df_desc)

    all_data[data_type] = all_data[data_type].map(lambda s: s.lower())
    all_data['client_batch'] = client_batch
    # df_psi,df_psi_details = psi_statis(all_data, splitted_types=['train','test','oot'], scores=[feature_type])
    df_psi, df_psi_details = psi_statis(all_data, splitted_types=['train', 'test'], scores=[feature_type])
    del all_data['client_batch']
    log.info('模型psi：')
    print(df_psi[['train_test_psi']])
    # log.info(df_psi[['train_test_psi','train_oot_psi']])

    df_output_statis = df_splitted_type_auc_ks.reset_index()
    df_output_statis['feature'] = feature_type
    df_output_statis['project_name'] = project_name
    df_output_statis['client_batch'] = client_batch
    df_output_statis = df_output_statis.pivot_table(
        index=['project_name', 'client_batch', 'feature'],
        columns=data_type,
        values=['auc', 'ks'])
    df_output_statis.columns = ['_'.join(reversed(x)) for x in df_output_statis.columns]
    df_output_statis['feature_cnt'] = len(selected_features)
    df_output_statis['n_estimators'] = model.get_params()['n_estimators']

    log.info('统计结束')
    return df_output_statis
    # ===========================统计=================================


# In[42]:


# =========================step 6 训练模型=========================
X_all, y_all, X_train, y_train, X_test, y_test, X_oot, y_oot = get_splitted_data(
    all_data_to_model, target=target, selected_features=selected_features)

print('整体数据集大小：', X_all.shape)
print('训练集大小：', X_train.shape)
print('测试集大小：', X_test.shape)
if X_oot is None:
    print('无oot数据集')
else:
    print('oot集大小：', X_oot.shape)

pd.Series(X_test.index).to_csv('{}{}_{}_X_test_key_{}.csv'.format(
    output_dir, project_name, feature_type, cust_id), header=cust_id, index=False)

log.info('step 6 开始训练模型')
start = datetime.now()

log.info('step 6.1 ===筛选变量===')

# ===========================================

log.info('step 6.1 ===筛选变量===10折交叉后，计算变量的平均重要性')
# feature_imp = tree_selection.kfold_xgb_model(train_data=(del_corr_df, y_train))
log.info('筛选前数据集大小：{}'.format(X_train.shape))
feature_imp = tree_selection.change_col_subsample_fit_model(train_data=(X_train, y_train),
                                                            test_data=(X_test, y_test))

log.info('将特征重要性持久化')
feature_imp.to_csv('{}{}_{}_xgb_allfeature_mean_imp_df.csv'.format(
    output_dir, project_name, feature_type))

log.info('根据10折拟合模型处理后的变量重要性进行变量相关性筛选')
del_corr_df = tree_selection.drop_corr(X_train, by=feature_imp, threshold=0.9)
# del_corr_df = tree_selection.drop_corr(del_corr_df, by=feature_imp, threshold=0.8)
log.info('筛选后数据集大小：{}'.format(del_corr_df.shape))

# ===========================================

selected_features = list(del_corr_df.columns)
log.info('最终入模变量的数量：{}'.format(len(selected_features)))
log.info('最终入模变量：{}'.format(selected_features))

feature_imp = tree_selection.change_col_subsample_fit_model(train_data=(del_corr_df, y_train),
                                                            test_data=(X_test[del_corr_df.columns], y_test))

log.info('将待入模特征重要性持久化')
feature_imp.to_csv('{}{}_{}_xgb_tomodel_feature_mean_imp_df.csv'.format(
    output_dir, project_name, feature_type))

log.info('贝叶斯进行模型调参')
model = classifiers_model(train_data=(X_train[selected_features], y_train),
                          test_data=(X_test[selected_features], y_test),
                          init_points=5, iterations=8, verbose=1)
log.info('模型调参完成！！！')
log.info('模型参数：{}'.format(model.get_xgb_params()))
log.info('模型参数：{}'.format(model.get_params()))

df_featurescore = pd.DataFrame(list(model._Booster.get_fscore().items()), columns=['特征名称', '特征权重值']
                               ).sort_values('特征权重值', ascending=False)
df_featurescore.to_csv('{}{}_{}_xgb_featurescore_first.csv'.format(
    output_dir, project_name, feature_type), index=False)

end = datetime.now()
log.info('模型训练完成, 使用 {} 秒'.format((end - start).seconds))

# X_all = pd.concat([X_train, X_test])
X_all[feature_type] = model.predict_proba(X_all[selected_features])[:, 1]
all_data = pd.concat([all_data_to_model, X_all[feature_type]], axis=1)

statistics_model_result(all_data=all_data)

# X_all.to_csv('{}{}_{}_X_all.csv'.format(output_dir, project_name, feature_type))
# all_data.to_csv('{}{}_{}_all_data.csv'.format(output_dir, project_name, feature_type))


if to_model_var_num:
    start = datetime.now()

    print('过滤前{}个特征出来，再次训练'.format(to_model_var_num))
    #         importance = model._Booster.get_fscore()
    #         importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    #         features_importance = pd.DataFrame()
    #         features_importance = features_importance.append(importance, ignore_index=True)
    #         features_importance.columns = ['特征名称', '特征权重值']
    #         # features_importance.to_csv(
    #         #     '{}{}_{}_xgb_features_importance.csv'.format(output_dir, project_name, feature_type))
    #         selected_features = features_importance.iloc[:to_model_var_num]['特征名称'].tolist()

    selected_features = df_featurescore.iloc[:to_model_var_num]['特征名称'].tolist()

    print('过滤后的特征：', selected_features)

    X_all, y_all, X_train, y_train, X_test, y_test, X_oot, y_oot = get_splitted_data(
        all_data_to_model, target=target, selected_features=selected_features)
    print('整体数据集大小：', X_all.shape)
    print('训练集大小：', X_train.shape)
    print('测试集大小：', X_test.shape)
    if X_oot is None:
        print('无oot数据集')
    else:
        print('oot集大小：', X_oot.shape)

    # 手动指定调参
    # model = xgb.XGBClassifier(**ini_params)
    # model.fit(X_train, y_train)

    # 贝叶斯调参
    log.info('贝叶斯进行模型调参')
    model = classifiers_model(train_data=(X_train[selected_features], y_train),
                              test_data=(X_test[selected_features], y_test),
                              init_points=5, iterations=8, verbose=1)
    log.info('模型调参完成！！！')
    log.info('模型参数：{}'.format(model.get_xgb_params()))
    log.info('模型参数：{}'.format(model.get_params()))

    end = datetime.now()
    log.info('模型训练完成, 使用 {} 秒'.format((end - start).seconds))

# X_all = pd.concat([X_train, X_test])
X_all[feature_type] = model.predict_proba(X_all[selected_features])[:, 1]
all_data = pd.concat([all_data_to_model, X_all[feature_type]], axis=1)

df_output_statis = statistics_model_result(all_data=all_data)

# X_all.to_csv('{}{}_{}_X_all.csv'.format(output_dir, project_name, feature_type))
# all_data.to_csv('{}{}_{}_all_data.csv'.format(output_dir, project_name, feature_type))


# ==========================训练模型=========================


# In[43]:


# ===========================step 7 模型持久化=================================

log.info('模型相关结果持久化')
all_data[feature_type].to_frame().to_csv(
    '{}/data/score/{}_{}_score.csv'.format(project_dir, project_name, feature_type))
all_data[feature_type].to_frame().to_csv(
    '{}/data/xgb_score/{}_{}_score.csv'.format(project_dir, project_name, feature_type))
all_data[feature_type].to_frame().to_csv('{}{}_{}_score.csv'.format(
    output_dir, project_name, feature_type))

joblib.dump(model._Booster, '{}{}_{}_xgb.ml'.format(
    output_dir, project_name, feature_type))
json.dump(model.get_params(), open('{}{}_{}_xgb.params'.format(
    output_dir, project_name, feature_type), 'w'))

model._Booster.dump_model('{}{}_{}_xgb.txt'.format(output_dir, project_name, feature_type))

df_featurescore = pd.DataFrame(list(model._Booster.get_fscore().items()), columns=['特征名称', '特征权重值']
                               ).sort_values('特征权重值', ascending=False)
df_featurescore.to_csv('{}{}_{}_xgb_featurescore.csv'.format(
    output_dir, project_name, feature_type), index=False)

df_corr = X_all.corr()
df_corr.to_csv('{}{}_{}_xgb_corr.csv'.format(
    output_dir, project_name, feature_type), index_label='feature')

df_rawdata = all_data[selected_features]
df_rawdata.reset_index(inplace=True)
df_rawdata_col_name = df_rawdata.columns.tolist()
df_rawdata_col_name.insert(len(df_rawdata_col_name) - 1,
                           df_rawdata_col_name.pop(df_rawdata_col_name.index(cust_id)))
df_rawdata = df_rawdata[df_rawdata_col_name]
df_rawdata.head(100).to_csv('{}{}_{}_xgb_rawdata.csv'.format(
    output_dir, project_name, feature_type), index=False)

df_output_statis.to_csv('{}{}_{}_xgb_output_statis.csv'.format(
    output_dir, project_name, feature_type))

os.makedirs(project_dir + 'data/statis/auc_ks', exist_ok=True)
df_output_statis.to_csv('{}data/statis/auc_ks/{}.csv'.format(
    project_dir, feature_type))

log.info('模型相关结果持久化完成')
# ===========================模型持久化=================================


# In[ ]:


# In[ ]:
