# -*-coding:utf-8-*-
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
import time
import requests
import numpy as np
import math
import json
from collections import defaultdict
import datetime
import warnings
import os
import matplotlib.pyplot as plt

# from db_load import read_mysql

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 不使用GPU来训练 可能造成GPU内存不够


def normalize(parking_data):  # 异常值处理
    n = len(parking_data) // 1440         # 每天1440分钟
    for i in range(1440):
        temp = []
        for j in range(n):
            temp.append((parking_data.iloc[i+j*1440]['number'], i+j*1440))
        flag = [0]*len(temp)
        for x in range(1, len(temp)):
            if abs(temp[x][0] - temp[0][0]) > 10:
                flag[x] = 1
        if 0 not in flag[1:]:
            parking_data.iloc[temp[0][1], 1] = int(np.mean(list(temp[m][0] for m in range(1, len(temp)))))
        else:
            for i in range(1, len(temp)):
                if flag[i] == 1:
                    parking_data.iloc[temp[i][1], 1] = int(np.mean(list(temp[m][0] for m in range(len(temp)) if flag[m] == 0)))
    return parking_data

def get_train_data(lot_id, lot_name):
    # lot_id 表示停车点的ID，lot_name 表示停车点名字
    # parking_data = defaultdict(list)
    temp = pd.read_csv('cars.csv')           #直接读取表格中的历史数据
    parking_data = temp
    return parking_data


def choose_para(train_x, train_y, lot_id, lot_name): #选择xgboost的参数为了降低使用使用 用了记录参数的方法，
    # 3周的数据需要40S的训练时间，7周的不确定，因此为了降低时间使用用了记录参数的方法
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    try:
        f = open('xgboost{}{}.txt'.format(today, lot_id)).read().splitlines()   #读取已有的训练模型
        xgb1 = xgb.XGBRegressor(
            n_estimators=int(f[0]),
            learning_rate=float(f[1]),
            max_depth=int(f[2]),
            min_child_weight=int(f[3]),
            gamma=float(f[4]),
            subsample=float(f[5]),
            colsample_bytree=float(f[6]),
            scale_pos_weight=1,
            seed=27
        )
    except FileNotFoundError:
        #未找到已经过训练的模型，则训练新的模型，配置参数
        cv_params = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
        other_params = {
            'learning_rate': 0.1,
            'max_depth': 5,              #树的最大深度，这个值也是用来避免过拟合
            'min_child_weight': 1,       #树的最大深度，这个值也是用来避免过拟合
            'gamma': 0,
            'subsample': 0.8,            #这个参数控制对于每棵树，随机采样的比例。
            'colsample_bytree': 0.8,     #用来控制每颗树随机采样的列数的占比每一列是一个特征0.5-1
            'scale_pos_weight': 1,       #
            'seed': 27
        }

        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', verbose=1, n_jobs=-1)
        optimized_GBM.fit(train_x, train_y)
        best_n = optimized_GBM.best_params_

        cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
        other_params = {
            'n_estimators': best_n['n_estimators'],
            'learning_rate': 0.1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'seed': 27
        }
        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', verbose=1, n_jobs=-1)
        optimized_GBM.fit(train_x, train_y)
        best_n2 = optimized_GBM.best_params_

        cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        other_params = {
            'n_estimators': best_n['n_estimators'],
            'learning_rate': 0.1,
            'max_depth': best_n2['max_depth'],
            'min_child_weight': best_n2['min_child_weight'],
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'seed': 27
        }
        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', verbose=1, n_jobs=-1)
        optimized_GBM.fit(train_x, train_y)
        best_n3 = optimized_GBM.best_params_

        cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}

        other_params = {
            'n_estimators': best_n['n_estimators'],
            'learning_rate': 0.1,
            'max_depth': best_n2['max_depth'],
            'min_child_weight': best_n2['min_child_weight'],
            'gamma': best_n3['gamma'],
            'scale_pos_weight': 1,
            'seed': 27
        }
        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', verbose=1, n_jobs=-1)
        optimized_GBM.fit(train_x, train_y)
        best_n4 = optimized_GBM.best_params_

        cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}

        other_params = {
            'n_estimators': best_n['n_estimators'],
            'max_depth': best_n2['max_depth'],
            'min_child_weight': best_n2['min_child_weight'],
            'gamma': best_n3['gamma'],
            'subsample': best_n4['subsample'],
            'colsample_bytree': best_n4['colsample_bytree'],
            'scale_pos_weight': 1,
            'seed': 27
        }
        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', verbose=1, n_jobs=-1)
        optimized_GBM.fit(train_x, train_y)
        best_n5 = optimized_GBM.best_params_

        xgb1 = xgb.XGBRegressor(
            n_estimators=best_n['n_estimators'],
            learning_rate=best_n5['learning_rate'],
            max_depth=best_n2['max_depth'],
            min_child_weight=best_n2['min_child_weight'],
            gamma=best_n3['gamma'],
            subsample=best_n4['subsample'],
            colsample_bytree=best_n4['colsample_bytree'],
            scale_pos_weight=1,
            seed=27

        )
        va = [best_n['n_estimators'],best_n5['learning_rate'],best_n2['max_depth'],best_n2['min_child_weight'],
              best_n3['gamma'],best_n4['subsample'],best_n4['colsample_bytree']]
        with open('xgboost{}{}.txt'.format(today,lot_id),'w') as f:
            for i in range(len(va)):
                f.write('{}\n'.format(va[i]))
    return xgb1


def main(lot_id,lot_name):
    # tic = time.time()        #获取当前时间
    train_data = get_train_data(lot_id,lot_name)  #获取训练数据集
    train_x = train_data[['time', 'start_num', 'time2']]   #数据的横坐标，time表示某一天，start_num为该天开始时的数据
    train_x = train_x.values
    train_y = train_data['train']            #数据纵坐标，表示停车位数
    train_y = train_y.values

    real_data = train_data['real']              #验证集
    real_data = real_data.values
    # val_in = train_data['val_in']                #用来预测下一天的输入数据
    # val_in = val_in.values
    xgb1 = choose_para(train_x, train_y, lot_id, lot_name)  #通过训练集获取合适的参数
    xgb1.fit(train_x, train_y)                            #进行训练

    time_index = list(range(1, 1441))
    num = [41 for i in range(1440)]
    time_2 = []
    for i in range(1, 781):
        time_2.append(i)
    for i in range(661, 1321):
        time_2.append(1440 - i)

    test = pd.DataFrame({'time': time_index, 'start_num': num, 'time2': time_2})

    test_x = test[['time', 'start_num', 'time2']].values
    y_pred = xgb1.predict(test_x)                    #对未来的时间进行预测

    validate = train_data['real'][:1440]
    error_value = validate - y_pred                   #计算误差
    error_value = list(map(abs, error_value))          #取绝对值
    error_rate = (error_value/validate) * 100          #计算误差百分比

    data_out = pd.DataFrame({'time': test['time'], 'predict': y_pred,
                             'true': real_data[:1440], 'error': error_value,
                             'error_rate': error_rate})
    data_out.to_csv('out.csv')
    ##############################
    # 结果输出
    # 这部分因为在本地尝试因此没有进行数据库的输出只是单纯的输出一个文件


if __name__ == '__main__':
    main('test_ID', 'test_name')
    # 实际使用中，需要读取数据库信息，获取每个站点的ID和name，再做一个循环，每个站点单独预测
    print('###################################################################')
    #time.sleep(60)
