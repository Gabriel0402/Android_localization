from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

import preprocess
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def distance(y_true, y_pred):
    return np.average(np.apply_along_axis(np.linalg.norm,1,y_pred-y_true))

preprocess.get_data()
X = preprocess.X
y = preprocess.y
# min_max_scaler = preprocessing.StandardScaler()#StandardScaler
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.fit_transform(X_test)
score = make_scorer(distance, greater_is_better=False)
n_folds = 5

def plotFig(clf,p,index,xlabel,plt):
    clf.fit(X, y)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    plt.subplot(index)
    
    plt.plot(p, scores)

    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(n_folds)

    plt.plot(p, scores + std_error, 'b--')
    plt.plot(p, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(p, scores + std_error, scores - std_error, alpha=0.2)

    plt.ylabel('CV score +/- std error')
    plt.xlabel(xlabel)
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([p[0], p[-1]])

plt.figure()
n_estimators =np.arange(50,100,5)
tuned_parameters = [{'estimator__n_estimators': n_estimators}]
ada= MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4)))
clf = GridSearchCV(ada, tuned_parameters, cv=n_folds, refit=False, scoring=score)
plotFig(clf,n_estimators,221,'Number of Iterations of AdaBoost',plt)
learning_rate  =np.linspace(0.1,4,100)
tuned_parameters = [{'estimator__learning_rate': learning_rate}]
ada= MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4)))
clf = GridSearchCV(ada, tuned_parameters, cv=n_folds, refit=False, scoring=score)
plotFig(clf,learning_rate,222,'Learning Rate of AdaBoost',plt)
n_estimators =np.arange(50,100,5)
tuned_parameters = [{'estimator__n_estimators': n_estimators}]
rf= MultiOutputRegressor(RandomForestRegressor())
clf = GridSearchCV(rf, tuned_parameters, cv=n_folds, refit=False, scoring=score)
plotFig(clf,n_estimators,223,'Number of Trees of Random Forest',plt)
learning_rate  =np.linspace(0.1,1,100)
tuned_parameters = [{'estimator__learning_rate': learning_rate}]
gb= MultiOutputRegressor(GradientBoostingRegressor(loss='ls'))
clf = GridSearchCV(gb, tuned_parameters, cv=n_folds, refit=False, scoring=score)
plotFig(clf,learning_rate,224,'Learning Rate of Gradient Tree Boosting',plt)
plt.show()