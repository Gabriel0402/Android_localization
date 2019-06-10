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

def distance(y_true, y_pred):
    return np.average(np.apply_along_axis(np.linalg.norm,1,y_true-y_pred))

preprocess.get_data()
X = preprocess.X
y = preprocess.y
# min_max_scaler = preprocessing.StandardScaler()#StandardScaler
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.fit_transform(X_test)

C = np.linspace(1,10, 100)
tuned_parameters = [{'estimator__C': C}]
svr_rbf = MultiOutputRegressor(SVR(kernel='rbf'))
score = make_scorer(distance, greater_is_better=False)
#gamma
param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    svr_rbf, X, y, param_name="estimator__gamma", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)



plt.figure()
gs = gridspec.GridSpec(2,2, height_ratios=[1,1])
plt.subplot(gs[0])
plt.xlabel(r"$\gamma$ of linear kernel")
plt.ylabel("negative distance")
plt.ylim(-10.0, -2.0)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
#C
param_range = np.linspace(1,10, 100)
train_scores, test_scores = validation_curve(
    svr_rbf, X, y, param_name="estimator__C", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplot(gs[1])
plt.xlabel(r"$C$ of linear kernel")
plt.ylabel("negative distance")
plt.ylim(-10.0, 0.0)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

svr_linear = MultiOutputRegressor(SVR(kernel='linear'))
score = make_scorer(distance, greater_is_better=False)
#gamma
param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    svr_linear, X, y, param_name="estimator__gamma", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplot(gs[2])
plt.xlabel(r"$\gamma$ of linear kernel")
plt.ylabel("negative distance")
plt.ylim(-10.0, -2.0)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
#C
param_range = np.linspace(1,10, 100)
train_scores, test_scores = validation_curve(
    svr_rbf, X, y, param_name="estimator__C", param_range=param_range,
    cv=5, scoring=score, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplot(gs[3])
plt.xlabel(r"$C$ of linear kernel")
plt.ylabel("negative distance")
plt.ylim(-10.0, 0.0)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")


plt.show()