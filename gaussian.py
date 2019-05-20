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
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import GridSearchCV

preprocess.get_data()
X = preprocess.X
y_x = preprocess.y_x
y_y = preprocess.y_y

ker_rbf = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e2))

ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)

ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))

kernel_list = [ker_rbf, ker_rq, ker_expsine]

param_grid = {"kernel": kernel_list,
              "optimizer": ["fmin_l_bfgs_b"],
              "n_restarts_optimizer": [8,9,10],
              "normalize_y": [False],
              "copy_X_train": [True], 
              "random_state": [0]}

gp = GaussianProcessRegressor()
grid_search = GridSearchCV(gp, param_grid=param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(X,y_x,test_size=0.2,random_state=38)
grid_search.fit(X_train,y_train)
y_pred = grid_search.predict(X_test)
print mean_absolute_error(y_test, y_pred)
print grid_search.best_estimator_