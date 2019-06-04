from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import preprocess
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import validation_curve
import xgboost as xgb

def distance(y_true, y_pred):
    return np.average(np.apply_along_axis(np.linalg.norm,1,y_true-y_pred))

preprocess.get_data()
X = preprocess.X
y = preprocess.y
# min_max_scaler = preprocessing.StandardScaler()#StandardScaler
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.fit_transform(X_test)

boost = MultiOutputRegressor(xgb.XGBRegressor())
score = make_scorer(distance, greater_is_better=False)
#max iteration
param_range = [300,400,500,600,700]
train_scores, test_scores = validation_curve(
    boost, X, y, param_name="estimator__n_estimators", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

#n_estimators
plt.figure()
gs = gridspec.GridSpec(2,2, height_ratios=[1,1])
plt.subplot(gs[0])
plt.xlabel("number of estimators")
plt.ylabel("negative distance")
plt.ylim(-10.0, 0.0)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
#max depth
param_range = [5,10,15,20,25]
train_scores, test_scores = validation_curve(
    boost, X, y, param_name="estimator__max_depth", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplot(gs[1])
plt.xlabel("max depth")
plt.ylabel("negative distance")
plt.ylim(-10.0, 0.0)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
#subsample  
param_range = [0.5,0.6,0.7,0.8,0.9,1]
train_scores, test_scores = validation_curve(
    boost, X, y, param_name="estimator__subsample", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplot(gs[2])
plt.xlabel("subsample")
plt.ylabel("negative distance")
plt.ylim(-10.0, 0.0)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

#learning rate
param_range = [0.05,0.1,0.15,0.2,0.25]
train_scores, test_scores = validation_curve(
    boost, X, y, param_name="estimator__learning_rate", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplot(gs[3])
plt.xlabel("learning rate")
plt.ylabel("negative distance")
plt.ylim(-10.0, 0.0)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

plt.show()