import preprocess
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

kernel = C(1.0, (1e-4, 1e4)) * RBF(10, (1e-2, 1e2))
clfs=[
    # linear_model.Lasso(alpha=0.01, max_iter=1000),
    # linear_model.Ridge(alpha=0.05, max_iter=1000),
    # linear_model.SGDRegressor(alpha=0.1,penalty='l1'),
    # linear_model.SGDRegressor(alpha=0.1,penalty='l2'),
    # svm.SVR(kernel='linear', C=0.8,epsilon=0.7),
    # AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=1000,learning_rate=0.1),
    RandomForestRegressor(n_estimators =300,max_depth=20),
    #GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8),
    # GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=64, loss='ls'),
    # xgb.XGBRegressor(),
    # MLPRegressor(learning_rate='adaptive')
    ]

def compare_process(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)
    min_max_scaler = preprocessing.StandardScaler()#StandardScaler
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    res=[]
    r2i=[]
    for c in clfs:
        c.fit(X_train,y_train)
        y_pred = c.predict(X_test)
        # res.append(mean_absolute_error(y_test, y_pred))
        print y_pred
        err=0
        for i, p in enumerate(y_pred):
            err+= math.sqrt((p[0]-y_test[i][0])**2+(p[1]-y_test[i][1])**2)
        res.append(err/len(y_pred))
    print res
    # print r2i

preprocess.get_data()
X = preprocess.X
y = preprocess.y
compare_process(X,y)
