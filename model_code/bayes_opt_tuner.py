#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: bayes_opt_tuner.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020/4/26
'''

import time

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from model_code.utils import get_ks, get_roc_auc_score


class ModelTune():
    def __init__(self):
        self.base_model = None
        self.best_model = None
        self.model_params = None
        self.loss = np.inf
        self.metrics = None
        self.default_params = None
        self.int_params = None

    def get_model(self):
        return self.best_model

    def fit(self, train_data=(), test_data=()
            , init_points=10, iterations=15):

        X_train, y_train = train_data
        X_test, y_test = test_data

        # def loss_fun(train_result, test_result):
        #     train_result = train_result * 100
        #     test_result = test_result * 100
        #     if train_result == test_result:
        #         return test_result
        #
        #     import math
        #     return test_result - math.log(abs(test_result - train_result))

        def loss_fun(train_result, test_result):
            train_result = train_result * 100
            test_result = test_result * 100

            return test_result - 2 ** abs(test_result - train_result)

        # def loss_fun(train_result, test_result):
        #     train_result = train_result * 100
        #     test_result = test_result * 100
        #
        #     return train_result - 2 ** abs(train_result - test_result)

        def obj_fun(**params):
            for param in self.int_params:
                params[param] = int(round(params[param]))

            model = self.base_model(**params, **self.default_params)
            model.fit(X_train, y_train)

            pred_test = model.predict_proba(X_test)[:, 1]
            pred_train = model.predict_proba(X_train)[:, 1]

            test_auc = get_roc_auc_score(y_test, pred_test)
            train_auc = get_roc_auc_score(y_train, pred_train)
            print('test_auc is : ', test_auc)
            print('train_auc is : ', train_auc)

            test_ks = get_ks(y_test, pred_test)
            train_ks = get_ks(y_train, pred_train)

            max_result = loss_fun(train_auc, test_auc)
            # max_result = loss_fun(train_ks, test_ks) * 2 + loss_fun(train_auc, test_auc)

            loss = 1 - max_result
            if loss < self.loss:
                self.loss = loss
                self.best_model = model
                print('best model result is {}'.format(1 - loss))
                print('best model result is : ')
                print(self.best_model.get_params())
            print('current obj_fun result is : ', max_result)

            return max_result

        params_optimizer = BayesianOptimization(obj_fun, self.model_params, random_state=1)
        print('params_optimizer is : ', params_optimizer.space.keys)

        print('begain optimizer params!!!')
        start = time.time()
        params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ei', xi=0.0)
        # params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ucb', xi=0.0, alpha=1e-6)
        end = time.time()
        print('optimizer params over!!! 共耗时{} 分钟'.format((end - start) / 60))
        print('the best params is : {}'.format(params_optimizer.max['params']))
        print('Maximum xgb value is : {}'.format(params_optimizer.max['target']))


class ClassifierModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['auc', 'ks']


class RegressorModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['r2', 'rmse']


class XGBClassifierTuner(ClassifierModel):
    def __init__(self):
        super().__init__()  # 先执行父类

        self.base_model = XGBClassifier
        self.model_params = {
            'min_child_weight': (1, 300),
            'max_depth': (2, 10),
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.2),
            'subsample': (0.4, 1.0),
            'colsample_bytree': (0.3, 1.0),
            'gamma': (0, 2.0),
            'reg_alpha': (0, 2.0),
            'reg_lambda': (0, 2.0),
            # 'max_delta_step': (0, 10)
        }

        self.default_params = {
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        }

        self.int_params = ['max_depth', 'n_estimators']


class LGBClassifierTuner(ClassifierModel):
    def __init__(self):
        super().__init__()  # 先执行父类

        self.base_model = LGBMClassifier
        self.model_params = {
            'max_depth': (500, 1500),
            'num_leaves': (200, 800),
            'min_data_in_leaf': (50, 250),
            'n_estimators': (750, 1800),
            'min_child_weight': (0.01, 0.05),
            'bagging_fraction': (0.2, 1.0),
            'feature_fraction': (0.15, 1.0),
            'learning_rate': (0.005, 0.01),
            'reg_alpha': (0.2, 0.6),
            'reg_lambda': (0.25, 1.0)
        }

        self.default_params = {
            'objective': 'binary',
            # 'max_depth': -1,
            'boosting_type': 'gbdt',
            'bagging_seed': 11,
            'metric': 'auc',
            'verbosity': -1,
            'random_state': 47,
            'num_threads': -1
        }

        self.int_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 'n_estimators']


classifiers_dic = {
    # 'logistic_regression': LogisticRegressionTuner,
    # 'random_forest': RandomForestClassifierTuner,
    'xgboost': XGBClassifierTuner,
    # 'lgb': LGBClassifierTuner
}


def classifiers_model(models=[], metrics=[], train_data=(), test_data=()
                      , init_points=10, iterations=25, verbose=1):
    if type(models) != list:
        raise AttributeError('Argument `models` must be a list, ',
                             'but given {}'.format(type(models)))
    if len(models) == 0:
        models = list(classifiers_dic.keys())
    classifiers = []
    for model in models:
        if model in classifiers_dic:
            classifiers.append(classifiers_dic[model])
    loss = float('inf')
    _model = None
    for classifier in classifiers:
        if verbose:
            print("Optimizing {}...".format(classifier()))
        _model = classifier()
        _model.fit(train_data=train_data,
                   test_data=test_data
                   , init_points=init_points, iterations=iterations)

    return _model.get_model()


if __name__ == '__main__':
    X = pd.read_pickle('X_train.pkl')
    X = pd.DataFrame(X)
    y = pd.read_pickle('y_train.pkl')
    y = pd.Series(y)
    X_test = pd.read_pickle('X_test.pkl')
    X_test = pd.DataFrame(X_test)
    y_test = pd.read_pickle('y_test.pkl')
    y_test = pd.Series(y_test)

    best_model = classifiers_model(train_data=(X, y), test_data=(X_test, y_test), verbose=1)
    print('classifiers_model run over!!!')
    print(best_model.get_xgb_params())

    train_pred_y = best_model.predict_proba(X)[:, 1]

    test_pred_y = best_model.predict_proba(X_test)[:, 1]

    train_auc = get_roc_auc_score(y, train_pred_y)
    test_auc = get_roc_auc_score(y_test, test_pred_y)
    train_ks = get_ks(y, train_pred_y)
    test_ks = get_ks(y_test, test_pred_y)
    print('train_auc is : ', train_auc, 'test_auc is : ', test_auc)
    print('train_ks is : ', train_ks, 'test_ks is : ', test_ks)

#####构建数据
# X, y = make_classification(n_samples=20000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# with open('X_train.pkl', 'wb') as f:
#     f.write(pickle.dumps(X_train))
# with open('y_train.pkl', 'wb') as f:
#     f.write(pickle.dumps(y_train))
# with open('X_test.pkl', 'wb') as f:
#     f.write(pickle.dumps(X_test))
# with open('y_test.pkl', 'wb') as f:
#     f.write(pickle.dumps(y_test))
