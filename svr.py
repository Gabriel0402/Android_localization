from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

import preprocess
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer

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
n_folds = 5
svr_rbf = MultiOutputRegressor(SVR(kernel='rbf'))

score = make_scorer(distance, greater_is_better=False)
clf = GridSearchCV(svr_rbf, tuned_parameters, cv=n_folds, refit=False, scoring=score)
clf.fit(X, y)
print clf.cv_results_
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

plt.figure()
plt.subplot(221)
plt.plot(C, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.plot(C, scores + std_error, 'b--')
plt.plot(C, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(C, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('Penalty C with RBF kernel')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([C[0], C[-1]])


gamma = np.linspace(0.01,0.2, 10)
tuned_parameters = [{'estimator__gamma': gamma}]
n_folds = 5

clf = GridSearchCV(svr_rbf, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

plt.subplot(222)
plt.plot(gamma, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.plot(gamma, scores + std_error, 'b--')
plt.plot(gamma, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(gamma, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('Kernel coefficient with RBF kernel')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([gamma[0], gamma[-1]])

plt.show()