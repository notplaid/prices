import numpy as np
import pandas as pd
from sklearn import metrics
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import plot_importance
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import itertools
from pylab import mpl
from sklearn.model_selection import KFold
mpl.rcParams['font.sans-serif']=['SimHei']        #  解决中文问题
pd.set_option('display.max_columns',1000)          #解决显示问题
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
def read_data():
    train = pd.read_csv('train.csv')  # .fillna(-1)
    test = pd.read_csv('test.csv')  # .fillna(-1)
    return train,test
def rmse(y_true, y_pred):                          #rmse评价指标
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)
def bianma_add(a,b, name):                         #编码函数处理
    t= pd.concat([a,b],sort=False)
    type = t.ix[:, name]
    t[name].fillna('0', inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(type)
    t[name] = (le.transform(type.astype(str))).T
    a[name]=t[name][:len(a)]
    b[name]=t[name][len(a):]
def data_process(train,test):                 #数据预处理函数
    bianma_add(train, test, '房屋朝向')
    #train['有无地铁'] = train['地铁线路'].apply(chuli_ditie)
    #test['有无地铁'] = test['地铁线路'].apply(chuli_ditie)
    #train['有无地铁'] = train['地铁线路'].apply(chuli_ditie)
    #test['有无地铁'] = test['地铁线路'].apply(chuli_ditie)
    #train = train.drop(['时间'], axis=1)
    #test = test.drop(['时间'], axis=1)
    #train = train[train.房屋面积 != 1]
    #train = train[train.房屋面积 < 0.05]
    #train = train[train.厅的数量 !=8 ]
    return train,test
def get_the_vaid_data(train,test):
    traind, val = train_test_split(train, test_size=0.000000001, random_state=1)
    #traind=train[train['时间'] <= 2]
    #val=train[train['时间'] > 2]
    train_y = traind['月租金']
    val_y = val['月租金']
    train_x = traind.drop(['月租金'], axis=1)
    val_x = val.drop(['月租金'], axis=1)
    test_id = test.id
    test_x = test.drop(['id'], axis=1)
    return train_x, train_y, val_x, val_y, test_x,test_id
def run_lgb(train_X, train_y, val_X, val_y, test_X):  #lgb模型


    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'max_depth': 14,
        #'objective': 'regression',
        'learning_rate': 0.1,
        'seed': 2018,
        'num_threads': -1,
       # 'max_bin': 425,
        "metric": "rmse",
       # "lambda_l1": 0.1,
        "lambda_l2": 0.2,
    }
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    #model = lgb.train(params, lgtrain, 20000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    model=lgb.train(params, lgtrain, num_boost_round=13000)
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    print(f"XGB : RMSE val: {rmse(val_y, pred_val_y)}")
    return pred_test_y, model, rmse(val_y, pred_val_y)
def run_xgb(X_train, y_train, X_val, y_val, X_test):      #xgb模型
    '''
    params = {'objective': 'reg:linear',           #  xgb参数设置
              'eval_metric': 'rmse',
              'eta': 0.001,
              'max_depth': 10,
              'subsample': 0.6,
              'colsample_bytree': 0.6,
              'alpha':0.001,
              'random_state': 42,
              'silent': True}
    '''
    params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'subsample': 0.8,
        'colsample_bytree': 0.85,
        'eta': 0.05,
        'max_depth': 13,
        'seed': 2016,
        'silent': 1,
        'eval_metric': 'rmse'
    }
    xgb_train_data = xgb.DMatrix(X_train, y_train)     #数据处理成xgb形式
    xgb_val_data = xgb.DMatrix(X_val, y_val)
    xgb_submit_data = xgb.DMatrix(X_test)
    # 获取最好迭代次数 训练方式
    #model = xgb.train(params, xgb_train_data,num_boost_round=10000,evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],early_stopping_rounds=100,verbose_eval=500)  #获取最好迭代次数
    # 直接指定迭代次数
    model = xgb.train(list(params.items()), xgb_train_data,2000)
    y_pred_train = model.predict(xgb_train_data, ntree_limit=model.best_ntree_limit)
    y_pred_val = model.predict(xgb_val_data, ntree_limit=model.best_ntree_limit)
    y_pred_submit = model.predict(xgb_submit_data, ntree_limit=model.best_ntree_limit)
    plot_importance(model)
    plt.show()
    print(f"XGB : RMSE val: {rmse(y_val, y_pred_val)}  - RMSE train: {rmse(y_train, y_pred_train)}")
    return y_pred_submit, model,rmse(y_val, y_pred_val)
def tezhengzuhe(data):              #构造特征函数
    #UID=data['UID']
    #Tag=data['Tag']
    #data=data.drop(['UID','Tag'],axis=1)
    col_name = data.columns
    #col_name=col_name[2:6]
    print (col_name)
    for i in list(itertools.combinations(col_name,2)):
        data[i[0]+'+'+i[1]]=  data[i[0]] + data[i[1]]
        #data[i[0]+'-'+i[1]] = data[i[0]] - data[i[1]]
        #data[i[0]+'*'+i[1]] = data[i[0]] * data[i[1]]
        #data[i[0]+'/'+i[1]] = data[i[0]] / data[i[1]]
    return data#pd.concat([data, UID, Tag], axis=1)
def remove_the_null(data):  #第二个参数：当缺失率达到多少时，直接删除   根据经验 删除缺失值往往效果不好
    t=[]
    for col_name in data.columns:
        changdu = len(data[col_name])
        cnt = list(data[col_name].isna()).count(True)
        print (col_name,cnt / changdu)
        #if (cnt / changdu > a):
         #   del data[col_name]
         #   del data1[col_name]
         #   t.append(col_name)
    #return data,data1,t
def genrate_result(test_id,pred_test,evaluation):
    ct = pd.DataFrame({'id': test_id, 'price': pred_test})
    #ct.to_csv('sub/result%d.csv' % evaluation, index=False)
    ct.to_csv("result" + str(evaluation) + ".csv", index=False, sep=',')
def kzhejiaocha(train,test,k):
    PREDS = []
    X = train
    kf = KFold(n_splits=k)
    test_id = test.id
    test_x = test.drop(['id'], axis=1)
    for train_index, test_index in kf.split(X):
        train_y = X.reindex(train_index).reset_index(drop=True)['月租金']
        val_y = X.reindex(test_index).reset_index(drop=True)['月租金']
        train_x = X.reindex(train_index).reset_index(drop=True).drop(['月租金'], axis=1)
        val_x = X.reindex(test_index).reset_index(drop=True).drop(['月租金'], axis=1)
        #train_t(X.reindex(train_index).reset_index(drop=True), X.reindex(test_index).reset_index(drop=True))
        y_pred_submit, model, evaluation=run_xgb(train_x, train_y, val_x, val_y, test_x)    #pred_test_y, model, pred_val_y
        PREDS.append(y_pred_submit)
    preds1 = np.mean(PREDS, axis=0)   #每一折取平均
    preds1 = pd.Series(preds1)
    ct = pd.DataFrame({'id': test_id, 'price': preds1})
    ct.to_csv("cv" + str(k) + ".csv", index=False, sep=',')
    bijiao(preds1)

def huatufenxi_data(train,test,name):
     print (train[name].describe())
     print (test[name].describe())
     plt.plot(train[name], 'r-', lw=1)
     #plt.plot(test[name], 'r-', lw=1)
     #plt.title("站点" + str(a) + " 5-7月份还车流量")
     plt.show()
def bijiao(tijiao):
    best=pd.read_csv('lgb2.0.csv')
    jieguo214=pd.read_csv('lgb2.07.csv')
    print (rmse(best['price'],tijiao))
    print(rmse(jieguo214['price'], tijiao))
def chuli_ditie(t):  #非0 表示为1
    if (np.isnan(t)):
        #print (123)
        return 1
    else :
        return 0

def chulimianji(t):
    if(t<0.005):
        return 0
    if (t>=0.05 and t<0.01):
        return 1
    if (t>=0.01and t<0.015):
        return 2
    if (t>=0.015and t<0.02):
        return 3
    if (t>=0.02and t<0.025):
        return 4
    if (t>=0.025):
       return 5
def moxingronghe_tezheng(result1,result2,a,b):  #模型融合
    test_id = result1.id
    ct = pd.DataFrame({'id': test_id, 'price': a*result1['price']+b*result2['price']})
    ct.to_csv("ronghe" + str(a)+str(b) + ".csv", index=False, sep=',')
    return a*result1['price']+b*result2['price']

def tezhengzuhe(data):
    #UID=data['UID']
    #Tag=data['Tag']
    #data=data.drop(['UID','Tag'],axis=1)
    col_name = ['小区房屋出租数量', '楼层', '总楼层',
                '房屋面积', '房屋朝向', '居住状态', '卧室数量',
                '厅的数量', '卫的数量', '出租方式', '区', '位置',
                '地铁线路', '地铁站点', '距离', '装修情况']#data.columns
    lieming=list(itertools.combinations(col_name,2))
    lieming=lieming[0:10]
    #col_name=col_name[10:20]
    print (lieming)
    print (len(lieming))
    for i in lieming:#list(itertools.combinations(col_name,2)):
        #data[i[0]+'+'+i[1]]=  data[i[0]] + data[i[1]]
        #data[i[0]+'-'+i[1]] = data[i[0]] - data[i[1]]
        #data[i[0]+'*'+i[1]] = data[i[0]] * data[i[1]]
        data[i[0]+'/'+i[1]] = data[i[0]] / data[i[1]]

    return data

# 处理房屋朝向
#all_data['东'], all_data['南'], all_data['西'], all_data['北'], all_data['西北'], all_data['西南'], all_data['东北'], all_data['东南'] = ['东','南','西','北','西北','西南','东北','东南']


def processDirection(directions, direction):
    if direction in directions.split():
        return 1
    else:
        return 0

if __name__ == '__main__':
    lgb_result=pd.read_csv('result0.10882.csv')
    best=pd.read_csv('result0.06924.csv')
    ronghe=moxingronghe_tezheng(best,lgb_result,0.6,0.4)
    train,test=read_data()
    train, test=data_process(train,test)

    '''
    #train = pd.concat([train, test], axis=0)

    train['东'] = train.apply(lambda x: processDirection(x.房屋朝向, x.东), axis=1)
    train['南'] = train.apply(lambda x: processDirection(x.房屋朝向, x.南), axis=1)
    train['西'] = train.apply(lambda x: processDirection(x.房屋朝向, x.西), axis=1)
    train['北'] = train.apply(lambda x: processDirection(x.房屋朝向, x.北), axis=1)
    train['西北'] = train.apply(lambda x: processDirection(x.房屋朝向, x.西北), axis=1)
    train['西南'] = train.apply(lambda x: processDirection(x.房屋朝向, x.西南), axis=1)
    train['东北'] = train.apply(lambda x: processDirection(x.房屋朝向, x.东北), axis=1)
    train['东南'] = train.apply(lambda x: processDirection(x.房屋朝向, x.东南), axis=1)
    '''
    train['房型'] = train['卧室数量'].map(str) + '_' + train['厅的数量'].map(str) + '_' + train['卫的数量'].map(str)
    train['楼层状况'] = train['总楼层'].map(str) + '_' + train['楼层'].map(str)

    #feature3 = ['楼层', '总楼层', '居住状态', '出租方式', '位置', '区', '地铁线路', '地铁站点', '装修情况', '房型', '楼层状况']

    #for i in feature3:
    #    train[i] = train[i].astype(str)

    #train = pd.get_dummies(train)


    #print (train)

    #test['东'] = test.apply(lambda x: processDirection(x.房屋朝向, x.东), axis=1)
    #test['南'] = test.apply(lambda x: processDirection(x.房屋朝向, x.南), axis=1)
    #test['西'] = test.apply(lambda x: processDirection(x.房屋朝向, x.西), axis=1)
    #test['北'] = test.apply(lambda x: processDirection(x.房屋朝向, x.北), axis=1)
    #test['西北'] = test.apply(lambda x: processDirection(x.房屋朝向, x.西北), axis=1)
    #test['西南'] = test.apply(lambda x: processDirection(x.房屋朝向, x.西南), axis=1)
    #test['东北'] = test.apply(lambda x: processDirection(x.房屋朝向, x.东北), axis=1)
    #test['东南'] = test.apply(lambda x: processDirection(x.房屋朝向, x.东南), axis=1)

    test['房型'] = test['卧室数量'].map(str) + '_' + test['厅的数量'].map(str) + '_' + test['卫的数量'].map(str)
    test['楼层状况'] = test['总楼层'].map(str) + '_' + test['楼层'].map(str)
    #print (test.columns)

    bianma_add(train, test, '房型')

    bianma_add(train, test, '楼层状况')

    #train['房屋面积1'] = train['房屋面积'].apply(chulimianji)
    #test['房屋面积1'] = test['房屋面积'].apply(chulimianji)
    train['房屋面积6'] = np.round(train['房屋面积'], 4)
    test['房屋面积6'] = np.round(test['房屋面积'], 4)


    train['房屋面积'] = 1/(train['房屋面积']+1)
    test['房屋面积'] =  1/(test['房屋面积']+1)

    train['房屋面积1'] = train['房屋面积'] / (train['厅的数量'] + 1)
    test['房屋面积1'] = test['房屋面积'] / (test['厅的数量'] + 1)

    train['房屋面积2'] = train['房屋面积'] / (train['卫的数量'] + 1)
    test['房屋面积2'] = test['房屋面积'] / (test['卫的数量'] + 1)

    train['房屋面积3'] = train['房屋面积'] / (train['卧室数量'] + 1)
    test['房屋面积3'] = test['房屋面积'] / (test['卧室数量'] + 1)

    train['房屋面积4'] = train['房屋面积'] / (train['卧室数量'] + train['卫的数量']+train['厅的数量'])
    test['房屋面积4'] = test['房屋面积'] / (test['卧室数量'] + test['卫的数量']+test['厅的数量'])

    train['房屋面积5'] = (train['卧室数量'] + train['卫的数量'] + train['厅的数量'])/train['房屋面积']
    test['房屋面积5'] = (test['卧室数量'] + test['卫的数量'] + test['厅的数量'])/test['房屋面积']





    train['房屋面积7'] = train['楼层']/train['厅的数量']
    test['房屋面积7'] = test['楼层'] / test['厅的数量']

    #train['房屋面积8'] = train['小区名'] / train['房屋朝向']
    #test['房屋面积8'] = test['小区名'] / test['房屋朝向']

    #地铁线路+地铁站点
    #train['房屋面积7'] = train['卧室数量']/train['地铁线路']
    #test['房屋面积7'] = test['卧室数量'] / test['地铁线路']

    #train['房屋面积1'] = train['小区名'] + train['卧室数量']
    #test['房屋面积1'] = test['小区名'] + test['卧室数量']

    #train['房屋面积2'] = train['小区名'] - train['地铁站点']
    #test['房屋面积2'] = test['小区名'] - test['地铁站点']
    #train=train.drop(['时间','小区房屋出租数量'],axis=1)
    #test = test.drop(['时间','小区房屋出租数量'], axis=1)

    #train = train.drop(['时间', '小区房屋出租数量'], axis=1)
    #test = test.drop(['时间', '小区房屋出租数量'], axis=1)

    #train['房屋面积3'] = train['房屋面积']/(train['卧室数量'] + train['厅的数量']+train['卫的数量'])
    #test['房屋面积3'] = test['房屋面积']/(test['卧室数量'] + test['厅的数量'] + test['卫的数量'])

    #train['月租金'] = np.round(train['月租金'], 4)
    #test['月租金'] = np.round(test['月租金'], 4)


    #train=tezhengzuhe(train)
    #test = tezhengzuhe(test)
    #kzhejiaocha(train, test, 15)
    #train['小区房屋出租数量'] = np.round(train['小区房屋出租数量'], 4)
    #test['小区房屋出租数量'] = np.round(test['小区房屋出租数量'], 4)

    #train['总楼层'] = np.round(train['总楼层'], 4)
    #test['总楼层'] = np.round(test['总楼层'], 4)

    #train['距离'] = np.round(train['距离'], 4)
    #test['距离'] = np.round(test['距离'], 4)


    #train['房屋面积3'] = train['房屋面积']/train['卧室数量']
    #test['房屋面积3'] = test['房屋面积']/test['卧室数量']

    #train['房屋面积']=train['房屋面积']*1000000
    #train=train.sort_values(axis=0, ascending=False, by='小区房屋出租数量')
    #train=train.groupby(['小区名','总楼层'], as_index=False).mean()
    #test = test.groupby(['小区名', '总楼层'], as_index=False).mean()
    #print (train)
    #train = train[train.小区名 ==1]
    #test = test[test.小区名== 1]
    #train=train[train.房屋面积 ==0.5 ]
    #print (train)
    #huatufenxi_data(train, test, '房屋面积')
    #d = train['房屋面积'].hist().get_figure()
    #plt.show()
    #d.savefig('2.jpg')
    '''
    
    col_name=[ '时间', '小区名', '小区房屋出租数量', '楼层', '总楼层',
             '房屋面积', '房屋朝向', '居住状态', '卧室数量', '厅的数量',
             '卫的数量', '出租方式', '区', '位置', '地铁线路', '地铁站点',
             '距离', '装修情况']

    lieming = list(itertools.combinations(col_name, 2))

    train_x, train_y, val_x, val_y, test_x, test_id = get_the_vaid_data(train, test)
    y_pred_submit, model, evaluation = run_xgb(train_x, train_y, val_x, val_y, test_x)
    bijiao(y_pred_submit)

    for i in lieming:
        train[i[0] + '+' + i[1]] = train[i[0]] + train[i[1]]
        test[i[0] + '+' + i[1]] = test[i[0]] + test[i[1]]
        print(i[0] + '+' + i[1])
        train_x, train_y, val_x, val_y, test_x, test_id = get_the_vaid_data(train, test)
        y_pred_submit, model, evaluation = run_xgb(train_x, train_y, val_x, val_y, test_x)
        bijiao(y_pred_submit)
        del train[i[0] + '+' + i[1]]
        del test[i[0] + '+' + i[1]]

        train[i[0] + '-' + i[1]] = train[i[0]] - train[i[1]]
        test[i[0] + '-' + i[1]] = test[i[0]] - test[i[1]]
        print(i[0] + '-' + i[1])
        train_x, train_y, val_x, val_y, test_x, test_id = get_the_vaid_data(train, test)
        y_pred_submit, model, evaluation = run_xgb(train_x, train_y, val_x, val_y, test_x)
        bijiao(y_pred_submit)
        del train[i[0] + '-' + i[1]]
        del test[i[0] + '-' + i[1]]

        train[i[0] + '*' + i[1]] = train[i[0]] * train[i[1]]
        test[i[0] + '*' + i[1]] = test[i[0]] * test[i[1]]
        print(i[0] + '*' + i[1])
        train_x, train_y, val_x, val_y, test_x, test_id = get_the_vaid_data(train, test)
        y_pred_submit, model, evaluation = run_xgb(train_x, train_y, val_x, val_y, test_x)
        bijiao(y_pred_submit)
        del train[i[0] + '*' + i[1]]
        del test[i[0] + '*' + i[1]]

        train[i[0] + '/' + i[1]] = train[i[0]] / train[i[1]]
        test[i[0] + '/' + i[1]] = test[i[0]] / test[i[1]]
        print(i[0] + '/' + i[1])
        train_x, train_y, val_x, val_y, test_x, test_id = get_the_vaid_data(train, test)
        y_pred_submit, model, evaluation = run_xgb(train_x, train_y, val_x, val_y, test_x)
        bijiao(y_pred_submit)
        del train[i[0] + '/' + i[1]]
        del test[i[0] + '/' + i[1]]

   '''



    #kzhejiaocha(train, test,10)
    train_x, train_y, val_x, val_y, test_x, test_id=get_the_vaid_data(train,test)
    y_pred_submit, model, evaluation=run_xgb(train_x, train_y, val_x, val_y, test_x)
    #y_pred_submit, model, evaluation=run_lgb(train_x, train_y, val_x, val_y, test_x)
    #evaluation=1
    genrate_result(test_id, y_pred_submit, evaluation)
    bijiao(y_pred_submit)

    #best = pd.read_csv('best.csv')
    #data=pd.concat([test, best['price']], axis=1)
    #data = data.sort_values(axis=0, ascending=False, by='小区名')
    #print (data)



