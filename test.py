import numpy as np
import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from random import randint
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

preprocess.get_data()
X = preprocess.X
y = preprocess.y
classes = np.unique(y)

def plot_matrix_report(clf,xtrain,ytrain,xtest,ytest):
    print("# Tuning hyper-parameters for accuracy")
    print()
    clf.fit(xtrain,ytrain)
    y_true, y_pred = ytest, clf.predict(xtest)
    print(classification_report(y_true, y_pred))
    print()
    return y_pred

# class ada:
#     def __init__(self):
#         self.pipeline=Pipeline([('ada', AdaBoostClassifier(DecisionTreeClassifier(max_depth=50,min_samples_split=25,min_samples_leaf=20),algorithm="SAMME.R"))])
#         self.name='ada'
#         self.parameters = {
#             'ada__learning_rate': (0.4,0.5,0.6),
#             'ada__n_estimators':(50,100)
#         }

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=randint(1,86439))

adapipeline = Pipeline([('ada', AdaBoostClassifier(DecisionTreeClassifier(max_depth=50,min_samples_split=25,min_samples_leaf=20),algorithm="SAMME"))])
adaparameters = {
    'ada__learning_rate': (0.4,0.5,0.6),
    'ada__n_estimators':(50,100)
}
grid_search = GridSearchCV(adapipeline, adaparameters,scoring='accuracy',cv=5)


# confusion matrix
predictions=plot_matrix_report(OneVsRestClassifier(grid_search),X_train,y_train,X_test,y_test)
cnf_matrix = confusion_matrix(y_test,predictions,labels=classes)
np.set_printoptions(precision=2)
