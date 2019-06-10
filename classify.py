import preprocess
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import validation_curve

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from random import randint
import xgboost as xgb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import math
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import matplotlib.pyplot as plt                                              

kernel = C(1.0, (1e-4, 1e4)) *RationalQuadratic(length_scale=1.0, alpha=0.1)
clfs={
    # linear_model.Lasso(alpha=0.01, max_iter=1000),
    # linear_model.Ridge(alpha=0.05, max_iter=1000),
    # linear_model.SGDRegressor(alpha=0.1,penalty='l1'),
    # linear_model.SGDRegressor(alpha=0.1,penalty='l2'),
    'svr':svm.SVR(kernel='rbf', C=2),
    'ada':AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=1000,learning_rate=0.1),
    'rf':RandomForestRegressor(n_estimators =400,max_depth=20),
    'gpr':GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8),
    #GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=64, loss='ls'),
    'xgb':xgb.XGBRegressor(n_estimators =500, max_depth=10, learning_rate=0.05),
    # 'nn':MLPRegressor(learning_rate='adaptive')
}

def compare_process(X,y,res):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)
    # min_max_scaler = preprocessing.StandardScaler()#StandardScaler
    # X_train = min_max_scaler.fit_transform(X_train)
    # X_test = min_max_scaler.fit_transform(X_test)
    for c in clfs:
        mt = MultiOutputRegressor(clfs[c])
        mt.fit(X_train,y_train)
        y_pred = mt.predict(X_test)
        # res.append(mean_absolute_error(y_test, y_pred))
        res[c].append(np.average(np.apply_along_axis(np.linalg.norm,1,y_test-y_pred)))
    print res
    # print r2i

size = [400,500,600,700,800,900,999]
res={'svr':[],'ada':[],'rf':[],'xgb':[],'gpr':[]}
preprocess.get_data()
X = preprocess.X
Y = preprocess.y
for s in size:
    idx=np.random.randint(X.shape[0], size=s)
    x=X[idx,:]
    y=Y[idx,:]
    # print X
    # print y
    compare_process(x,y,res)
df=pd.DataFrame(res)
df.to_csv('size_2.csv')
lines=df.plot.line()
plt.show()

# size = [1,2,3,4,5,6,7]
# res={'svr':[],'ada':[],'rf':[],'xgb':[],'gpr':[]}
# preprocess.get_data()
# X = preprocess.X
# y = preprocess.y
# for s in size:
#     x=X[:,0:s]
#     # print X
#     # print y
#     compare_process(x,y,res)
# df=pd.DataFrame(res)
# df.to_csv('feature_2.csv')
# lines=df.plot.line()
# plt.show()