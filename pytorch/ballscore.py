import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

## load data
root = 'D:\\AI\\ball\\'
# train_data = pd.read_table(root+'train3.txt') 2 value
train_data = pd.read_table(root+'train_score0531.txt')  # 3 value
#val_data = pd.read_csv(root+'val.csv')
predict_data = pd.read_table(root+'test_score0601-0629.txt')  # predict0619

test_data = pd.read_table(root+'test_score0601-0629.txt')

trainlabel = np.array(train_data.label)
traindata = np.array(train_data.drop("label", axis=1))

predictdata = np.array(predict_data)

testlabel = np.array(test_data.label)
testdata = np.array(test_data.drop("label", axis=1))

# vallabel = np.array(val_data.label)
# valdata = np.array(val_data.drop("label", axis=1))

# print(traindata.shape)
# print(trainlabel.shape)

#x_train,x_val,y_train,y_val = train_test_split(traindata,trainlabel, test_size=0.1,shuffle=True)
lgb_train = lgb.Dataset(traindata, trainlabel)
#lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)


def get_accuracy(y_preds, testlabel):

    right = 0.0
    wrong = 0.0
    for a, b in zip(y_preds, testlabel):
        if a == b:
            right = right+1
        else:
            wrong = wrong+1

    print("tatal", len(y_preds))
    print("right", right, ":wrong", wrong)
    print("accuracy", right/len(y_preds))
    print("wrong ", wrong/len(y_preds))

    return round(right/len(y_preds), 3)
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
    'boosting_type': 'gbdt',  #  dart goss  gbdt
    'objective': 'multiclass',  #  multiclass    multiclassova
    #'max_depth': 12, # 9   11
    'num_leaves': 777, # 450  777
    #'min_data_in_leaf': 60,
    'metric': {'multi_logloss'},
    #'metric': {'multi_logloss', 'multi_error'},
    'learning_rate': 0.005,
    'num_class': 6,
    "random_state": 2020,
    "nthread": 6,
    #"lambda_l1": 0.1,
    #"lambda_l2": 0.1,
    #'feature_fraction': 0.9,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 10,
     #'verbose_eval': 0

}

def train():
    print("开始训练")
    gbm = lgb.train(params,  # 参数字典
                    lgb_train,  # 训练集
                    num_boost_round=2000,  # 迭代次数
                    #valid_sets=lgb_val,  # 验证集
                    valid_sets=lgb_train,
                    early_stopping_rounds=30  # 早停系数
                    )

    # feature names
    #print('Feature names:', gbm.feature_name())
    # feature importances
    #print('Feature importances:', list(gbm.feature_importance()))

    train_predt = gbm.predict(traindata, num_iteration=gbm.best_iteration)
    train_preds = [list(x).index(max(x)) for x in train_predt]  # 3 value

    acc = get_accuracy(train_preds, trainlabel)


    y_predt = gbm.predict(testdata, num_iteration=gbm.best_iteration)
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    get_accuracy(y_preds, testlabel)


    gbm.save_model(root+'model_gbdt_score_'+str(acc)+'.txt')


def predict():
    model_file = root + 'usemodel\\model_gbdt_score_1.0.txt'    #   model_goss_0.889
    gbm = lgb.Booster(model_file=model_file)

    print('开始预测...')
    y_predt = gbm.predict(predictdata, num_iteration=gbm.best_iteration)
    #print(y_predt)
    # y_preds = [ 1 if i >=0.5 else 0 for i in y_predt]  # 2 value
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    print(y_preds)

    dt = pd.DataFrame(y_preds)
    dt.to_csv(root + "result10624.csv", encoding='utf_8_sig')


def test():
    model_file = root + 'usemodel\\model_gbdt_score_1.0.txt'  # model_goss_noall_0.889
    gbm = lgb.Booster(model_file=model_file)

    print('开始预测...')
    y_predt = gbm.predict(testdata, num_iteration=gbm.best_iteration)
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    get_accuracy(y_preds, testlabel)

    print(y_preds)

def main():
    #train()
    # predict()
    test()


if __name__ == '__main__':
    main()
