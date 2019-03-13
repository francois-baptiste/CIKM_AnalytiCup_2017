#encoding=utf-8
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import dataprocess.data_process as dp

# produces xgboost model
def xgb_train(train_df, test_df, params,num_boost_round,early_stopping):

    train = train_df.values[:,1:-1]
    train_target = train_df.values[:,-1]

    kf = KFold(n_splits=5, shuffle=True)

    result = np.zeros(test_df.shape[0])

    dtest = test_df.values[:,1:]
    dtest = xgb.DMatrix(dtest)

    for train_index, valid_index in kf.split(train):
        x_train, x_valid = train[train_index], train[valid_index]
        y_train, y_valid = train_target[train_index], train_target[valid_index]

        dtrain = xgb.DMatrix(x_train, y_train)
        watchlist = [(dtrain, 'train')]
        gbm = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                        evals=watchlist, early_stopping_rounds=early_stopping, )

        result += gbm.predict(dtest)

    result = result/5.0
    return result



if __name__ == "__main__":

    trainfile = '../data/train.txt'
    testBfile = '../data/testB.txt'
    train_1ave8extend = dp.dataprocess(trainfile, data_type='train', windversion='new')
    test_1ave = dp.dataprocess(testBfile, data_type='testB', windversion='new')



    # trains single xgboost model

    eta,max_depth,min_child_weight,subsample,colsample_bytree=(0.1,10,5,1,0.8)
    # eta,max_depth,min_child_weight,subsample,colsample_bytree=(0.02, 43, 5, 0.8, 0.8)
    print(eta,max_depth,min_child_weight,subsample,colsample_bytree)

    params = {"objective":"reg:linear",
            "booster":"gbtree",
            "eta":eta,
            "max_depth":max_depth,
            "min_child_weight":min_child_weight,
            "subsample":subsample,
            "colsample_bytree":colsample_bytree,
            "silent":0,
            "seed":200
          }
    num_boost_round = 1000
    early_stopping_rounds = 100

    result =  xgb_train(train_1ave8extend,test_1ave, params, num_boost_round, early_stopping_rounds)
