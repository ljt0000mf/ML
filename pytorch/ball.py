import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

## load data
root = 'D:\\AI\\ball\\'
# train_data = pd.read_table(root+'train3.txt') 2 value
train_data = pd.read_table(root+'3train20200531_noall.txt')  # 3 value
#val_data = pd.read_csv(root+'val.csv')
test_data = pd.read_table(root+'test0601-0629.txt')

"""
trainlabel = np.array(train_data.label)
traindata = np.array(train_data.drop("label", axis=1))

testlabel = np.array(test_data.label)
testdata = np.array(test_data.drop("label", axis=1))

x_train,x_val,y_train,y_val = train_test_split(traindata,trainlabel, test_size=0.2,shuffle=True)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
"""

## get train feature
del_feature = ['label']
features = [i for i in train_data.columns if i not in del_feature]

train_x = train_data[features].values
train_y = train_data['label'].values

test_x = test_data[features].values
test_y = test_data['label'].values

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], 3))

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
    'boosting_type': 'dart',  #  dart goss  gbdt
    'objective': 'multiclass',  #  multiclass    multiclassova
    #'max_depth': 11, # 9   11
    'num_leaves': 58,  # 450  1333
    #'min_data_in_leaf': 60,
    'metric': {'multi_logloss'},  #{'multi_logloss', 'multi_error'},
    'learning_rate': 0.01,
    'num_class': 3,
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
    ## train and predict
    feature_importance_df = pd.DataFrame()
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
        print("fold {}".format(fold_ + 1))
        #trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
        #val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])
        trn_data = lgb.Dataset(train_x[trn_idx], label=train_y[trn_idx])
        val_data = lgb.Dataset(train_x[val_idx], label=train_y[val_idx])

        gbm = lgb.train(params,
                        trn_data,
                        num_boost_round=100,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=20,
                        #early_stopping_rounds=60
                        )

    train_predt = gbm.predict(train_x, num_iteration=gbm.best_iteration)
    train_preds = [list(x).index(max(x)) for x in train_predt]  # 3 value

    acc = get_accuracy(train_preds, train_y)

    y_predt = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    get_accuracy(y_preds, test_y)

    """     
        prob_oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print(feature_importance_df)
        """


    """
    gbm = lgb.train(params,  # 参数字典
                    lgb_train,  # 训练集
                    num_boost_round=2000,  # 迭代次数
                    valid_sets=lgb_val,  # 验证集
                    #early_stopping_rounds=50  # 早停系数
                    )
    """
    """
    cv_results = lgb.cv(
        params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=2020)
    print(cv_results)
    print('best n_estimators:', len(cv_results['multi_logloss-mean']))
    print('best cv score:', cv_results['multi_logloss-mean'][-1])
    """
    """
    y_predt = gbm.predict(testdata, num_iteration=gbm.best_iteration)
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    get_accuracy(y_preds, testlabel)
    """

    #gbm.save_model(root+'model0621.txt')


def predict():
    model_file = root + 'usemodel\\goss_73_model0621.txt'
    gbm = lgb.Booster(model_file=model_file)

    print('开始预测...')
    predict_data = pd.read_table(root + 'predict.txt')  # predict0619
    predictdata = np.array(predict_data)
    y_predt = gbm.predict(predictdata, num_iteration=gbm.best_iteration)
    # print(y_predt)
    # y_preds = [ 1 if i >=0.5 else 0 for i in y_predt]  # 2 value
    y_preds = [list(x).index(max(x)) for x in y_predt]  # 3 value

    print(y_preds)

    dt = pd.DataFrame(y_preds)
    dt.to_csv(root + "result1.csv", encoding='utf_8_sig')

def main():
    train()
    #predict()


if __name__ == '__main__':
    main()
