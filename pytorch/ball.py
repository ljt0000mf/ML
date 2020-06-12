import numpy as np
import lightgbm as lgb
import pandas as pd

## load data
root = 'D:\\AI\\ball\\'
train_data = pd.read_table(root+'train.txt')
val_data = pd.read_table(root+'val.txt')
#val_data = pd.read_csv(root+'val.csv')
num_round = 10

# traindata = train_data[['s','p','f']]
# label = train_data[['label']]

# valdata = val_data[['s','p','f']]
# vallabel = val_data[['label']]

trainlabel = np.array(train_data.label)
traindata = np.array(train_data.drop("label", axis=1))

vallabel = np.array(val_data.label)
valdata = np.array(val_data.drop("label", axis=1))

print(traindata)
print(trainlabel)


lgb_train = lgb.Dataset(traindata, trainlabel, free_raw_data=False)
lgb_val = lgb.Dataset(valdata, vallabel, free_raw_data=False)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boost_from_average': 'true',
    'learning_rate': 0.01,
    'num_class': 1
}

print("开始训练")

gbm = lgb.train(params,                     # 参数字典
                lgb_train,                  # 训练集
                num_boost_round=20,       # 迭代次数
                valid_sets=lgb_val,        # 验证集
                #early_stopping_rounds=30    # 早停系数
                )

gbm.save_model('model.txt')


ypred = gbm.predict(valdata)

print(ypred)
""""""

