#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Nov 07, 2016

@author: timekeeper
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


class DecisionModel(object):
    def __init__(self, X, Y, seed=7, test_size=0.33):
        """
        Constructor
        """
        self.X = X
        self.Y = Y
        self.normalized_X = preprocessing.normalize(self.X)
        seed = 7  # 7 - xgboost, 2 - dectree
        test_size = 0.33  # размер указывается в долях
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.Y, test_size=test_size, random_state=seed)

    def __del__(self):
        """
        Destructor
        """

    def __execute(self, model):
        model.fit(self.X_train, self.y_train)
        # make predictions for test data
        y_pred = model.predict_proba(self.X_test)[:,1]
        predictions = [round(value) for value in y_pred]
        accuracy = model.score(self.X_test,self.y_test)#accuracy_score(self.y_test, predictions)
        class_report = metrics.classification_report(self.y_test, predictions)
        conf_matrix = metrics.confusion_matrix(self.y_test, predictions)
        # tn, fp, fn, tp = metrics.confusion_matrix(self.y_test, predictions)
        return accuracy, class_report, conf_matrix

    def xgb(self):
        model = xgb.XGBClassifier(base_score=0.9, max_depth=10, n_estimators=100, subsample=0.4,
                                  reg_lambda=2)
        accuracy, class_report, conf_matrix = self.__execute(model)
        bst = model.booster()
        imps = bst.get_fscore()
        return model, accuracy, class_report, conf_matrix, imps

    def logistic_regression(self):
        model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
        accuracy, class_report, conf_matrix = self.__execute(model)
        return model, accuracy, class_report, conf_matrix

    def dectree(self):
        model = DecisionTreeClassifier(criterion='gini',
                                       max_depth=8, max_features=None, max_leaf_nodes=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       random_state=None, splitter='best')
        accuracy, class_report, conf_matrix = self.__execute(model)
        return model, accuracy, class_report, conf_matrix

    def svc(self):
        model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                         coef0=0.0, shrinking=True, probability=True,
                         tol=1e-3, cache_size=500, class_weight=None,
                         verbose=False, max_iter=-1, decision_function_shape=None,
                         random_state=None)
        accuracy, class_report, conf_matrix = self.__execute(model)
        return model, accuracy, class_report, conf_matrix
