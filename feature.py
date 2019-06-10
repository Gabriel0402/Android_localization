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
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFE
def distance(y_true, y_pred):
    return np.average(np.apply_along_axis(np.linalg.norm,1,y_true-y_pred))


score = make_scorer(distance, greater_is_better=False)
gp=MultiOutputRegressor(xgb.XGBRegressor(n_estimators =500, max_depth=10, learning_rate=0.05))
size=[1,2,3,4,5,6,7]
preprocess.get_data()
X = preprocess.X
y = preprocess.y
res=[]
for s in size:
    x=X[:,0:s]
    print x
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)
    gp.fit(X_train,y_train)
    y_pred = gp.predict(X_test)
    res.append(np.average(np.apply_along_axis(np.linalg.norm,1,y_test-y_pred)))
print res

