# _*_coding utf-8_*_

import time
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense

'''
本模块为lstm模型
'''

from keras.layers import LSTM
from matplotlib import pyplot as plt

def create_dataset(dataset):
    batch_size = 4  # 一批
    epochs = 20  # 时代
    time_step = 7  # 用多少组天数进行预测
    input_size = 7  # 每组天数，亦即预测天数
    look_back = time_step * input_size
    showdays = 120  # 最后画图观察的天数（测试天数）
    # 忽略掉最近的forget_days天数据（回退天数，用于预测的复盘）
    forget_days = 0  # 原为0

    dataX, dataY = [], []
    print('len of dataset: {}'.format(len(dataset)))
    for i in range(0, len(dataset) - look_back, input_size):
        x = dataset[i: i + look_back]
        dataX.append(x)
        y = dataset[i + look_back: i + look_back + input_size]
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

def build_model():
    batch_size = 4  # 一批
    epochs = 20  # 时代
    time_step = 7  # 用多少组天数进行预测
    input_size = 7  # 每组天数，亦即预测天数
    look_back = time_step * input_size
    showdays = 120  # 最后画图观察的天数（测试天数）
    # 忽略掉最近的forget_days天数据（回退天数，用于预测的复盘）
    forget_days = 0  # 原为0

    model = Sequential()
    model.add(LSTM(units=128, input_shape=(time_step, input_size)))
    model.add(Dense(units=input_size))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main():
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    batch_size = 4  # 一批
    epochs = 20  # 时代
    time_step = 7  # 用多少组天数进行预测
    input_size = 7  # 每组天数，亦即预测天数
    look_back = time_step * input_size
    showdays = 120  # 最后画图观察的天数（测试天数）
    # 忽略掉最近的forget_days天数据（回退天数，用于预测的复盘）
    forget_days = 0  # 原为0

    X_train = []
    y_train = []
    X_validation = []  # 验证
    y_validation = []
    testset = []  # 用来保存测试基金的近期净值

    fileTrain = 'data/funds_data_train.xlsx'
    fileTest = 'data/funds_data_test.xlsx'

    # 设定随机数种子
    seed = 7
    np.random.seed(seed)

    # 导入数据（训练集）
    # 常见的读取文件操作
    rows = pd.read_excel(fileTrain,header=None)
    #print(rows)
    row = [ list(rows.loc[0,:]), list(rows.loc[1,:]), list(rows.loc[2,:]), list(rows.loc[3,:]) ]
    for r in row:
        dataset = []
        r = [x for x in r if x != 'None']
        #print('2row里面包含：', len(r), r)
        #涨跌幅是2天之间比较，数据会减少1个
        days = len(r) - 1
        #有效天数太少，忽略
        if days <= look_back + input_size:     # 49 + 7
            continue
        for i in range(days):
            f1 = float(r[i])
            f2 = float(r[i+1])
            if f1 == 0 or f2 == 0:
                dataset = []
                break
            #把数据放大100倍，相当于以百分比为单位
            f2 = (f2 - f1) / f1 * 100

            #f2 = (r[i]-1)*100
            #如果涨跌幅绝对值超过15%，基金数据恐有问题，忽略该组数据
            if f2 > 15 or f2 < -15:
                dataset = []
                break
            #print(f2)
            dataset.append(f2)
        n = len(dataset)
        #print('原始数据长度', len(r), '涨跌幅度数组长度：', n , '去掉一个FORGET_DAYS', n-forget_days)
        #进行预测的复盘，忽略掉最近forget_days的训练数据
        n -= forget_days

        if n >= look_back + input_size:             # 49 + 7
            #如果数据不是input_size的整数倍，忽略掉最前面多出来的
            m = n % input_size
            print( '去掉多出来后最终用于统计的涨跌幅数据个数：', n-m )
            #print(dataset[m:n][:56])   # 去掉后前50个
            X_1, y_1 = create_dataset(dataset[m:n])
            '''
            print(X_1)
            print(y_1)
            pd.DataFrame(X_1).to_excel('X_1.xlsx')
            pd.DataFrame(y_1).to_excel('y_1.xlsx')
            '''
            X_train = np.append(X_train, X_1)
            y_train = np.append(y_train, y_1)
    #print(type(X_train), len(X_train), X_train[0])


    # 导入数据（测试集）
    rows = pd.read_excel(fileTest,header=None)
    #print(rows)
    row = [ list(rows.loc[0,:]) ]
    #print(row)

    #写成了循环，但实际只有1条测试数据
    for r in row:
        dataset = []
        #去掉记录为None的数据（当天数据缺失）
        r = [x for x in r if x != 'None']
        #涨跌幅是2天之间比较，数据会减少1个
        days = len(r) - 1
        #有效天数太少，忽略，注意：测试集最后会虚构一个input_size
        if days <= look_back:
            print('only {} days data. exit.'.format(days))
            continue
        #只需要最后画图观察天数的数据
        if days > showdays:
            r = r[days-showdays:]
            days = len(r) - 1
        for i in range(days):
            f1 = float(r[i])
            f2 = float(r[i+1])
            if f1 == 0 or f2 == 0:
                print('zero value found. exit.')
                dataset = []
                break
            #把数据放大100倍，相当于以百分比为单位
            f2 = (f2 - f1) / f1 * 100

            #f2 = (r[i] - 1) * 100
            #如果涨跌幅绝对值超过15%，基金数据恐有问题，忽略该组数据
            if f2 > 15 or f2 < -15:
                print('{} greater then 15 percent. exit.'.format(f2))
                dataset = []
                break
            testset.append(f1)
            dataset.append(f2)
        #保存最近一天基金净值
        f1=float(r[days])
        testset.append(f1)
        #测试集虚构一个input_size的数据（若有forget_days的数据，则保留）
        if forget_days < input_size:
            for i in range(forget_days,input_size):
                dataset.append(0)
                testset.append(np.nan)
        else:
            dataset = dataset[:len(dataset) - forget_days + input_size]
            testset = testset[:len(testset) - forget_days + input_size]
        if len(dataset) >= look_back + input_size:
            #将testset修正为input_size整数倍加1
            m = (len(testset) - 1) % input_size
            testset = testset[m:]
            m = len(dataset) % input_size
            #将dataset修正为input_size整数倍
            X_validation, y_validation = create_dataset(dataset[m:])

    #将输入转化成[样本数，时间步长，特征数]
    X_train = X_train.reshape(-1, time_step, input_size)
    X_validation = X_validation.reshape(-1, time_step, input_size)

    #将输出转化成[样本数，特征数]
    y_train = y_train.reshape(-1, input_size)
    y_validation = y_validation.reshape(-1, input_size)

    print('num of X_train: {}\tnum of y_train: {}'.format(len(X_train), len(y_train)))
    print('num of X_validation: {}\tnum of y_validation: {}'.format(len(X_validation), len(y_validation)))

    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.25, shuffle=True)

    # 评估模型
    train_score = model.evaluate(X_train, y_train, verbose=0)
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)

    # 预测
    predict_validation = model.predict(X_validation)

    #将之前虚构的最后一组input_size里面的0涨跌改为NAN（不显示虚假的0）
    if forget_days < input_size:
        for i in range(forget_days,input_size):
            y_validation[-1, i] = np.nan

    print('Train Set Score: {:.3f}'.format(train_score))
    print('Test Set Score: {:.3f}'.format(validation_score))
    print('未来{}天实际百分比涨幅为：{}'.format(input_size, y_validation[-1]))
    print('未来{}天预测百分比涨幅为：{}'.format(input_size, predict_validation[-1]))


    #进行reshape(-1, 1)是为了plt显示
    y_validation = y_validation.reshape(-1, 1)
    predict_validation = predict_validation.reshape(-1, 1)
    testset = np.array(testset).reshape(-1, 1)

    # 把基金实际走势及预测走势分别存入excel文档
    #print(y_validation)
    #print(predict_validation)
    #pd.DataFrame(y_validation).to_excel('data/test_y_rut.xlsx')
    #pd.DataFrame(predict_validation).to_excel('data/test_predict_rut.xlsx')

    # 图表显示
    fig=plt.figure(figsize=(15,6))
    plt.plot(y_validation, color='blue', label='基金每日涨幅')
    plt.plot(predict_validation, color='red', label='预测每日涨幅')
    plt.legend(loc='upper left')
    plt.title('关联组数：{}组，预测天数：{}天，回退天数：{}天'.format(time_step, input_size, forget_days))
    #plt.show()

    return y_validation, predict_validation

if __name__ == '__main__':
    main()