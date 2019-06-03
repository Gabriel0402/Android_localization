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
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import make_scorer
import numpy as np

def distance(y_true, y_pred):
    return np.average(np.apply_along_axis(np.linalg.norm,1,y_true-y_pred))

score = make_scorer(distance, greater_is_better=False)

preprocess.get_data()
X = preprocess.X
y=preprocess.y

ker_rbf = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e2))

ker_rq = ConstantKernel(5.0, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e2))

ker_expsine = ConstantKernel(32.9, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e2))

# ker_expsine = ConstantKernel(1.0, (1e-4, 1e4)) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))

kernel_list = [
                # ConstantKernel(1.0, (1e-4, 1e4)) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)),
                ConstantKernel(1.0, (1e-4, 1e4))  * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                ConstantKernel(1.0, (1e-4, 1e4))  * RationalQuadratic(length_scale=1.0, alpha=0.1),
                ConstantKernel(1.0, (1e-4, 1e4))  * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)]

param_grid = {"kernel": kernel_list,
              "optimizer": ["fmin_l_bfgs_b"],
              "alpha":[1e-9,1e-8]}


gp = GaussianProcessRegressor()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)

for k in kernel_list:
    gpk = GaussianProcessRegressor(kernel=k,alpha=1e-8)
    gpk.fit(X_train,y_train)
    y_pred=gpk.predict(X_test)
    print np.average(np.apply_along_axis(np.linalg.norm,1,y_test-y_pred))

