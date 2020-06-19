import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

## load data
root = 'D:\\AI\\ball\\'
# train_data = pd.read_table(root+'train3.txt') 2 value
train_data = pd.read_table(root+'3train.txt')  # 3 value
#val_data = pd.read_csv(root+'val.csv')
predict_data = pd.read_table(root+'predict.txt')



trainlabel = np.array(train_data.label)
traindata = np.array(train_data.drop("label", axis=1))

predictdata = np.array(predict_data)

# vallabel = np.array(val_data.label)
# valdata = np.array(val_data.drop("label", axis=1))

# print(traindata.shape)
# print(trainlabel.shape)

x_train,x_val,y_train,y_val = train_test_split(traindata,trainlabel, test_size=0.2,shuffle=True)

lgb_train = lgb.Dataset(x_train, y_train)
lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)



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
    'boosting_type': 'dart',
    'objective': 'multiclass',
    'num_leaves': 60,
    'metric': 'multi_logloss',
    'boost_from_average': 'true',
    'learning_rate': 0.01,
    'num_class': 3,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}

def train():
    print("开始训练")
    """
    gbm = lgb.train(params,  # 参数字典
                    lgb_train,  # 训练集
                    num_boost_round=1000,  # 迭代次数
                    valid_sets=lgb_val,  # 验证集
                    early_stopping_rounds=50  # 早停系数
                    )
    """
    lgb.cv(params,  # 参数字典
            lgb_train,  # 训练集
            num_boost_round=1000,  # 迭代次数
            nfold=5
            )
    # gbm.save_model(root+'model.txt')


def predict():
    model_file = root + 'model.txt'
    gbm = lgb.Booster(model_file=model_file)

    print('开始预测...')
    y_predt = gbm.predict(predictdata, num_iteration=gbm.best_iteration)
    # print(y_predt)
    # y_preds = [ 1 if i >=0.5 else 0 for i in y_predt]  # 2 value
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    print(y_preds)

    dt = pd.DataFrame(y_preds)
    dt.to_csv(root + "0619_2result.csv", encoding='utf_8_sig')

def main():
    train()
    #predict()


if __name__ == '__main__':
    main()
