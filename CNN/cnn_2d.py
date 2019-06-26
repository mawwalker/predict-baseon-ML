#!/usr/bin/python
# coding: utf-8

from numpy import array
from numpy import hstack
import pandas as pd
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# 划分时间序列
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # 时间窗口的终止位置
        end_ix = i + n_steps
        # 时间窗口超出范围，则停止循环
        if end_ix > len(sequences):
            break
        # 按n_steps取相应大小的列表
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def split_sequences2(sequences, n_steps, forward):
    X, y = list(), list()
    for i in range(len(sequences)):
        # 时间窗口的终止位置
        end_ix = i + n_steps + forward
        # 时间窗口超出范围，则停止循环
        if end_ix > len(sequences):
            break
        # 按n_steps取相应大小的列表
        seq_x, seq_y = sequences[i:(i + n_steps), :-1], sequences[(i + n_steps - 1):(end_ix-1), -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def load_data(filename, colum):
    df = pd.read_csv(filename)
    data = df[colum].values
    # 返回的是np.array
    return data

def split_predict(sequences, n_steps, forward):
    X = list()
    for i in range(0, len(sequences), 10):
        # 时间窗口的终止位置
        end_ix = i + n_steps
        # 时间窗口超出范围，则停止循环
        if end_ix > len(sequences):
            break
        # 按n_steps取相应大小的列表
        seq_x = sequences[i:end_ix, :-1]
        X.append(seq_x)
    return array(X)


class Model():
    def __init__(self):
        self.model = Sequential()

    def model_build(self, X, y, n_steps, n_features):
        # 建立模型，编译模型，训练模型
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(10))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=1000, shuffle=True, verbose=1)

    def model_predict(self, x_input):
        yhat = self.model.predict(x_input, verbose=0)
        return yhat

    def model_save(self, filename):
        self.model.save(filename)

    def model_load(self, filename):
        self.model = load_model(filename)

    def long_predict(self, n_steps, n_features, forward):
        # 长时预测，思路是利用预测结果作为下一步的输入，进行滚动预测
        # 效果不好，已被弃用
        predict = []
        xg_out = load_data('xg_out.csv', 'pred').tolist()
        for i in range(n_steps):
            predict.append(42)
        for i in range(n_steps + 1, 1441, 10):
            time = array([j for j in range(i, i + n_steps)])
            time = time.reshape((len(time), 1))
            input_data = predict[(i - n_steps - 1):(i - 1)]
            input_data = array(input_data).reshape((len(input_data), 1))
            x_input = hstack((time, input_data, input_data))
            x_input = split_predict(x_input, n_steps, forward)
            pred = self.model_predict(x_input)
            pred_n = sum(pred.tolist(), [])
            # average = (pred_n + xg_out[i - 1])/2
            # average = round(average)
            # if abs(pred_n - xg_out[i - 1]) > 1:
            #     average = (pred_n + xg_out[i - 1])//2
            #     # average = round(average)
            # else:
            #     average = pred_n
            for i in pred_n:
                predict.append(i)
            # predict = sum(predict, [])
        return predict


def cnn_main():
    # 读取输入数据，二维
    in_seq1 = load_data('cars.csv', 'time')
    in_seq2 = load_data('cars.csv', 'train')
    # 输出数据
    out_seq = in_seq2
    # 升维后将三者合并，构成最终数据集
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # 合并数组，升维
    dataset = hstack((in_seq1, in_seq2, out_seq))
    # 选择时间窗口长度
    n_steps = 40
    forward = 10
    # 时间序列划分为模型的输入和输出
    X, y = split_sequences2(dataset, n_steps, forward)
    # X, y = split_sequences(dataset, n_steps)
    n_features = X.shape[2]
    # 测试集构建
    time_index = array([i for i in range(1, 1441)])
    test_data = load_data('cars.csv', 'real')[:1440]
    time_index = time_index.reshape((len(time_index), 1))
    test_data = test_data.reshape((len(test_data), 1))
    x_input = hstack((time_index, test_data, test_data))
    x_input = split_predict(x_input, n_steps, forward)
    # 创建模型对象
    cnn = Model()
    try:
        cnn.model_load('models/cnn_cars_2d_40_10.h5')
    except Exception:
        cnn.model_build(X, y, n_steps, n_features)
        cnn.model_save('models/cnn_cars_2d_40_10.h5')
    # 结果预测
    predict = cnn.model_predict(x_input)
    # 结果为列表中含有列表，这一步将列表降维
    predict = sum(predict.tolist(), [])
    #　predict = cnn.long_predict(n_steps, n_features, forward)
    print(predict)
    data_out = pd.DataFrame({'predict': predict})
    data_out.to_csv('out.csv')


if __name__ == '__main__':
    cnn_main()
