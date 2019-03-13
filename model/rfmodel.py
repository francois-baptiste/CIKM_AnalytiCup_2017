#encoding=utf-8
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import pairwise
import dataprocess.data_process as dp
import dataprocess.generate_percentile as gp

# gets some time slice


def slice_t(train_df, time_sum, time_slice, m, n, h):

    train_np = train_df.reshape((time_sum, h, m, n))
    slice = train_np[time_sum-time_slice:]
    train = slice.reshape((time_slice*h*m*n))
    return train

# gets some height slice
def slice_h(arr, time, m, n, h, asd):

    train = np.zeros((time, h, m*n))

    for i in range(time):
        for j in range(h):
            train[i, j] = arr[i, j + asd]

    train = train.reshape((time, h, m, n))
    return train

# cleans trainning data,return data index
def pre_train(train_df, test_df, train_add, test_add):

    train = train_df.values[:,1:-1]
    t = train_add.values[:,1:-1]
    train = np.hstack((train, t))

    dtest = test_df.values[:,1:]
    tA = test_add.values[:,1:]
    dtest = np.hstack((dtest, tA))

    cor_distance = pairwise.pairwise_distances(dtest, train)

    resultset = set()
    for tmp in cor_distance:
        index = np.argsort(tmp)
        for i in range(10):
            resultset.add(index[i])

    index = []
    for i in resultset:
        index.append(i)

    return index



# produces random forest model
def rf_model_train(train_df, test_df, train_add,test_add, ne, index=0):

    train = train_df.values[:,1:-1]
    train_target = train_df.values[:,-1]

    t = train_add.values[:,1:-1]
    train = np.hstack((train, t))

    rf = RandomForestRegressor(n_estimators=ne, verbose=2, n_jobs=-1)
    rf.fit(train, train_target)

    trainHat = rf.predict(train)
    valid = np.argsort(np.abs((trainHat - train_target)))

    return valid

# produces random forest model
def rf_model_notrain(train_df, test_df, train_add,test_add, ne, index=0):

    train_df = train_df.values[index]
    train = train_df[:,1:-1]
    train_target = train_df[:,-1]

    train_add = train_add.values[index]
    t = train_add[:,1:-1]

    #print(t.shape)
    train = np.hstack((train, t))

    kf = KFold(n_splits=2, shuffle=True)

    dtest = test_df.values[:,1:]
    tA = test_add.values[:,1:]
    dtest = np.hstack((dtest, tA))

    result = np.zeros(2000)

    for train_index, valid_index in kf.split(train):
        x_train, x_valid = train[train_index], train[valid_index]
        y_train, y_valid = train_target[train_index], train_target[valid_index]
        rf = RandomForestRegressor(n_estimators=ne, verbose=2, n_jobs=-1)
        rf.fit(x_train, y_train)
        result += rf.predict(dtest)

    result_rf = result/2.0 + 2

    return result_rf

if __name__ == "__main__":

    trainfile = '../data/train.txt'
    testBfile = '../data/testB.txt'
    train_add = dp.dataprocess(trainfile, data_type='train', windversion='old')
    testA_add = dp.dataprocess(testBfile, data_type='testB', windversion='old')
    train_1ave8extend = dp.dataprocess(trainfile, data_type='train', windversion='new')
    test_1ave = dp.dataprocess(testBfile, data_type='testB', windversion='new')
    train_df = gp.data_process(trainfile, data_type='train')
    test_df = gp.data_process(testBfile, data_type='testB')



    train = train_df.values[:, 1:-1]
    t = train_add.values[:, 1:-1]
    train = np.hstack((train, t))

    dtest = test_df.values[:, 1:]
    tA = testA_add.values[:, 1:]
    dtest = np.hstack((dtest, tA))

    cor_distance = pairwise.pairwise_distances(dtest, train)

    resultset = set()
    for tmp in cor_distance:
        index = np.argsort(tmp)
        for i in range(10):
            resultset.add(index[i])

    index = []
    for i in resultset:
        index.append(i)

    ne = 1100

    train_df = train_df.values[index]
    train = train_df[:,1:-1]
    train_target = train_df[:,-1]

    train_add = train_add.values[index]
    t = train_add[:,1:-1]

    #print(t.shape)
    train = np.hstack((train, t))

    kf = KFold(n_splits=2, shuffle=True)

    dtest = test_df.values[:,1:]
    tA = testA_add.values[:,1:]
    dtest = np.hstack((dtest, tA))

    result = np.zeros(test_df.shape[0])

    for train_index, valid_index in kf.split(train):
        x_train, x_valid = train[train_index], train[valid_index]
        y_train, y_valid = train_target[train_index], train_target[valid_index]
        rf = RandomForestRegressor(n_estimators=ne, verbose=2, n_jobs=-1)
        rf.fit(x_train, y_train)
        result += rf.predict(dtest)

    result_rf = result/2.0 + 2