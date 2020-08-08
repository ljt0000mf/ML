import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time

## load data
root = 'D:\\AI\\ball\\'
# train_data = pd.read_table(root+'train3.txt') 2 value
train_data = pd.read_table(root + 'train.txt')  # 3 value  3train20200531_noall.txt ，3train20200531_noall_5league
# val_data = pd.read_csv(root+'val.csv')


trainlabel = np.array(train_data.label)
traindata = np.array(train_data.drop("label", axis=1))


# vallabel = np.array(val_data.label)
# valdata = np.array(val_data.drop("label", axis=1))

#print(traindata.shape)
#print(trainlabel.shape)

x_train, x_val, y_train, y_val = train_test_split(traindata, trainlabel, test_size=0.1, shuffle=True)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)

folds = KFold(n_splits=5, shuffle=True, random_state=2020)
prob_oof = np.zeros((traindata.shape[0], 3))


def val_accuracy_limit(y_preds, vallabel, limit):
    i = 0
    right = 0.0
    wrong = 0.0
    other = 0.0
    myput = 0
    rewards = 0.0

    result = np.array(vallabel.result)
    win = np.array(vallabel.win)
    draw = np.array(vallabel.draw)
    loss = np.array(vallabel.loss)

    for x in y_preds:
        x = list(x)
        x_max = max(x)
        if x_max >= limit:
            myput = myput + 1
            if x.index(x_max) == result[i]:
                right = right + 1
                if result[i] == 0:
                    rewards = rewards + loss[i]
                    print(loss[i])
                elif result[i] == 1:
                    rewards = rewards + draw[i]
                    print(draw[i])
                else:
                    rewards = rewards + win[i]
                    print(win[i])
            else:
                wrong = wrong + 1
                #rewards = rewards - 1
        else:
            other = other + 1
        i = i + 1

    print("tatal", len(y_preds))
    print("right", right, ":wrong", wrong, ":other", other, ":myput", myput)
    print("accuracy", right / myput)
    print("wrong ", wrong / len(y_preds))
    print("other ", other / len(y_preds))
    print("rewards", myput, rewards, (rewards - myput), (rewards/myput))


def pred_limit(y_preds, limit, ptype):
    i = 0
    result = []
    for x in y_preds:
        x = list(x)
        x_max = max(x)
        if x_max >= limit:
            result.append(x.index(x_max))
        else:
            result.append('X')

    filename = "result" + ptype + time.strftime("%Y-%m-%d", time.localtime())
    dt = pd.DataFrame(result)
    dt.to_excel(root + filename + ".xlsx", encoding='utf_8_sig')


def get_accuracy_limit(y_preds, testlabel, limit):
    i = 0
    right = 0.0
    wrong = 0.0
    other = 0.0
    myput = 0
    for x in y_preds:
        x = list(x)
        x_max = max(x)
        if x_max >= limit:
            myput = myput + 1
            if x.index(x_max) == testlabel[i]:
                right = right + 1
            else:
                wrong = wrong + 1
        else:
            other = other + 1
        i = i + 1

    print("tatal", len(y_preds))
    print("right", right, ":wrong", wrong, ":other", other, ":myput", myput)
    print("accuracy", right / myput)
    print("wrong ", wrong / len(y_preds))
    print("other ", other / len(y_preds))


def get_accuracy(y_preds, testlabel):
    right = 0.0
    wrong = 0.0
    for a, b in zip(y_preds, testlabel):
        if a == b:
            right = right + 1
        else:
            wrong = wrong + 1

    print("tatal", len(y_preds))
    print("right", right, ":wrong", wrong)
    print("accuracy", right / len(y_preds))
    print("wrong ", wrong / len(y_preds))

    return round(right / len(y_preds), 3)


"""
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2'},
    # metric': {'l2', 'l1'},
    'num_leaves': 60,
    # 'metric': 'binary_logloss',
    'boost_from_average': 'true',
    'learning_rate': 0.01,
    # 'early_stopping_rounds': 500
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}
"""
params = {
    'boosting_type': 'gbdt',  # dart goss  gbdt
    'objective': 'multiclass',  # multiclass    multiclassova
    #'max_depth': 255,  # 9   11   255
    'num_leaves': 333,  # 58  450  777
    #'#min_data_in_leaf': 600,
    'metric': {'multi_logloss'},
    # 'metric': {'multi_logloss', 'multi_error'},
    'learning_rate': 0.01,
    'num_class': 3,
    "random_state": 2020,
    "nthread": 6,
    # "lambda_l1": 0.1,
    # "lambda_l2": 0.1,
    # 'feature_fraction': 0.9,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 10,
    # 'verbose_eval': 0

}


def train():
    print("开始训练")
    gbm = lgb.train(params,  # 参数字典
                    lgb_train,  # 训练集
                    num_boost_round=4000,  # 迭代次数
                    # valid_sets=lgb_val,  # 验证集
                    valid_sets=lgb_train,
                    #early_stopping_rounds=60  # 早停系数
                    )

    # feature names
    # print('Feature names:', gbm.feature_name())
    # feature importances
    # print('Feature importances:', list(gbm.feature_importance()))

    train_predt = gbm.predict(traindata, num_iteration=gbm.best_iteration)
    train_preds = [list(x).index(max(x)) for x in train_predt]  # 3 value

    acc = get_accuracy(train_preds, trainlabel)

    get_accuracy_limit(train_predt, trainlabel, 0.7)

    test_data2 = pd.read_table(root + 'test202005.txt')
    testlabel2 = np.array(test_data2.label)
    test_data2 = np.array(test_data2.drop("label", axis=1))
    y_predt = gbm.predict(test_data2, num_iteration=gbm.best_iteration)
    get_accuracy_limit(y_predt, testlabel2, 0.7)

    test_data = pd.read_table(root + 'test202006.txt')
    testlabel = np.array(test_data.label)
    testdata = np.array(test_data.drop("label", axis=1))
    y_predt = gbm.predict(testdata, num_iteration=gbm.best_iteration)
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value
    # get_accuracy(y_preds, testlabel)
    get_accuracy_limit(y_predt, testlabel, 0.7)

    test_data1 = pd.read_table(root + 'test202007.txt')
    testlabel1 = np.array(test_data1.label)
    test_data1 = np.array(test_data1.drop("label", axis=1))
    y_predt1 = gbm.predict(test_data1, num_iteration=gbm.best_iteration)
    y_preds1 = [list(x).index(max(x)) for x in y_predt1]  # 3 value
    # get_accuracy(y_preds1s, testlabel1)
    get_accuracy_limit(y_predt1, testlabel1, 0.7)

    gbm.save_model(root+'model_gbdt_74000_'+str(acc)+'.txt')


def predict():
    model_file = root + 'usemodel\\model_dart_62000_0.926.txt'  # model_goss_0.889
    gbm = lgb.Booster(model_file=model_file)
    print('开始预测...')
    predict_data = pd.read_table(root + 'predict2020-08-07.txt')  # predict0619
    predictdata = np.array(predict_data)
    y_predt = gbm.predict(predictdata, num_iteration=gbm.best_iteration)
    # y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value
    pred_limit(y_predt, 0.6, 'spf')


def predict_score():
    model_file = root + 'usemodel\\model_score_gbdt_72000_1.0.txt'  # model_goss_0.889
    gbm = lgb.Booster(model_file=model_file)
    print('开始预测...')
    predict_data = pd.read_table(root + 'predict2020-08-07.txt')  # predict0619
    predictdata = np.array(predict_data)
    y_predt = gbm.predict(predictdata, num_iteration=gbm.best_iteration)
    # y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value
    pred_limit(y_predt, 0.6, 'score')



def test():
    # model_dart_0.921:0.67  , model_gbdt_0.947:0.672
    # model_dart_5league0.953   ,   model_gbdt_5league0.953
    # model_dart_not_5league0.913   0.6    , model_gbdt_not_5league0.945
    # model_dart_alldata0.948     ,    model_gbdt_alldata0.946
    #  model_dart_no5league50000.944   model_gbdt_no5league50000.944
    model_file = root + 'usemodel\\model_dart_not_5league0.913.txt'
    gbm = lgb.Booster(model_file=model_file)

    print('开始预测...')
    test_data = pd.read_table(root + 'test2020-07-09_not_5league.txt')  # test0601_0709.txt   test2020-07-09_5league   test2020-07-09_not_5league
    val_data = pd.read_table(root + 'val2020-07-09_not_5league.txt')   # val2020-07-09.txt   val2020-07-09_5league    val2020-07-09_not_5league
    # testlabel = np.array(test_data.label)
    # testdata = np.array(test_data.drop("label", axis=1))
    testdata = np.array(test_data)

    y_predt = gbm.predict(testdata, num_iteration=gbm.best_iteration)
    # print(y_predt)
    # get_accuracy_limit(y_predt, testlabel, 0.45) # 0.5 74.4%,0.55 75.6%,0.6 78.9%,0.65 82.4%,0.7 85.6%,
    val_accuracy_limit(y_predt, val_data, 0.55)
    # y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value
    # get_accuracy(y_preds, testlabel)

    # print(y_preds)


def grid_search_cv():
    # estimator = lgb.LGBMRegressor(num_leaves=777)
    # estimator = lgb.LGBMRegressor(num_leaves=777, objective='multiclass', n_estimators=1000, num_class=3)
    estimator = lgb.LGBMClassifier(objective='multiclass', num_class=3)
    """
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
         #'n_estimators': [20, 40, 100],
         'num_leaves': [31, 58, 88, 158, 213, 777],
         'max_depth': [6, 8, 13, 133,  255]
    }
    """
    param_grid = {
        # 'learning_rate': [0.001, 0.005, 0.01],    #  3 30,3 10,5,5
        'num_leaves': range(3, 9, 3),
        'max_depth': range(1, 7, 2)
    }
    gbm = GridSearchCV(estimator, param_grid, cv=3)
    gbm.fit(traindata, trainlabel)

    print('Best parameters found by grid search are:', gbm.best_params_)



def test_5model():
    # model_goss_noall_0.889: 0.658 , model_gbdt_0.947:0.612   ,model_goss_0.93: 0.642,  model_dart_0.921:0.637
    # model_dart_5league0.953 0.676  ,   model_gbdt_5league0.953  .651
    model_file = root + 'usemodel\\model_gbdt_5league0.953.txt'
    gbm = lgb.Booster(model_file=model_file)

    print('开始预测...')
    test_data = pd.read_table(root + 'predict_0531_5league.txt')
    testlabel = np.array(test_data.label)
    testdata = np.array(test_data.drop("label", axis=1))
    y_predt = gbm.predict(testdata, num_iteration=gbm.best_iteration)
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    get_accuracy(y_preds, testlabel)

def main():
    #train()
    predict()
    predict_score()
    #test()
    # grid_search_cv()
    #test_5model()

if __name__ == '__main__':
    main()
