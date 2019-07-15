#!/usr/bin/python
# coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class lstm_model():

    def __init__(self, scl):
        # 模型初始化
        self.model = Sequential()
        self.scl = scl

    def load_lstm_model(self, filename):
        #加载模型
        self.model = load_model(filename)

    def train_model(self, X, y, look_back, forward_days):
        #训练模型
        NUM_NEURONS_FirstLayer = 50
        NUM_NEURONS_SecondLayer = 30
        EPOCHS = 50
        # 添加输入层，隐藏层，输出层
        self.model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(look_back, 1), return_sequences=True))
        self.model.add(LSTM(NUM_NEURONS_SecondLayer, input_shape=(NUM_NEURONS_FirstLayer, 1)))
        self.model.add(Dense(forward_days))
        #编译模型
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
        self.model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_validate, y_validate), shuffle=True, batch_size=2, verbose=2)

    def save__model(self, filename):
        # 保存模型
        self.model.save(filename)
        print("Saved model `{}` to disk".format(filename))

    def model_predict(self, X):
        X_predict = self.model.predict(X)
        X_predict = X_predict.ravel()
        #将标准化的结果还原
        X_predict = self.scl.inverse_transform(X_predict.reshape(-1, 1))
        return X_predict


class Data():

    def __init__(self, filename, colum, scl):
        df = pd.read_csv(filename)
        self.df = df[colum]
        self.scl = scl

    def normal_data(self):
        #数据标准化处理
        array = self.df.values.reshape(self.df.shape[0], 1)
        array = self.scl.fit_transform(array)
        return array

    def split_input(self, data, look_back, forward_days, jump=1):
        X = []
        for i in range(0, len(data) - look_back - forward_days +  1, jump):
            X.append(data[i:(i + look_back)])
        return X

    def split_out(self, data, look_back, forward_days, jump=1):
        Y = []
        for i in range(look_back, len(data) - forward_days + 1, jump):
            Y.append(data[i:(i+forward_days)])
        return Y

    def lstm_reshape(self, data):
        return data.reshape((1, 10, 1))


def lstm_main():
    look_back = 40
    forward_days = 10
    #num_periods = 20
    data_length = 1440    #用来划分数据集，表示一天的数据长度
    day_train = 2  #训练集天数
    # day_test = 1   #测试集天数
    scl = MinMaxScaler()   #数据标准化用到的对象
    #读取表格数据
    train_data = Data('cars.csv', 'train', scl)
    array = train_data.normal_data()
    #划分训练集和测试集
    division = data_length * day_train
    array_test = array[division:]
    array_train = array[:division]

    #划分测试集时间窗口
    X_test = train_data.split_input(array_test, look_back, forward_days, forward_days)
    X_test = np.array(X_test)
    #y_test = np.array([list(a.ravel()) for a in y_test])

    #划分训练集时间窗口
    X = train_data.split_input(array_train, look_back, forward_days)
    X = np.array(X)
    y = train_data.split_out(array_train, look_back, forward_days)
    y = np.array(y)
    y = np.array([list(a.ravel()) for a in y])
    model = lstm_model(scl)
    try:
        model.load_lstm_model('saved_model/LSTM_google_drive.h5')
    except Exception:
        model.train_model(X, y, look_back, forward_days)
        model.save__model('saved_model/LSTM_google_drive.h5')
    # 对训练集进行预测
    predict = model.model_predict(X_test)
    predict = sum(predict.tolist(), [])
    # predict = cnn.long_predict(n_steps, n_features)
    print(predict)
    # 将结果输出到文件，由于懒惰，直接输出到文件比较了，也可以用matplotlib画出结果
    data_out = pd.DataFrame({'predict': predict})
    data_out.to_csv('out.csv')


if __name__ == '__main__':
    lstm_main()
