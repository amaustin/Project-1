# Load libraries

# Recall/PrecisionReference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
import sklearn.externals
from IPython.display import Image
import pydotplus
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os


import time

# Code reference: https://www.datacamp.com/tutorial/decision-tree-classification-python

def BoostTree(X, Y, dataset_name):
    n_cpu = os.cpu_count()

    min_max_scaler = preprocessing.MinMaxScaler()

    # Split dataset into training set and test set: 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35,random_state=1)
    scale = StandardScaler()
    # Scale Data
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    index = [0, 1]



    for i in index:

        if i == 1:
            if dataset_name == "Dataset_Spotify":
                learning_rate = .5
                n_estimators = 25

            if dataset_name == "Dataset_Raisins":
                learning_rate = .5
                n_estimators = 25
        else:
                # default function values
                learning_rate = 1
                n_estimators = 50
        AdaBoost_Tree_Classifier = AdaBoostClassifier(random_state=1)
        train_scores, test_scores = validation_curve(AdaBoost_Tree_Classifier, X_train, y_train, param_name="n_estimators",
                                                     param_range=np.arange(1, 60), cv=5, n_jobs=n_cpu-3)

        plt.figure()
        plt.plot(np.arange(1, 60, 1), np.mean(train_scores, axis=1), label='Train Score')
        plt.plot(np.arange(1, 60, 1), np.mean(test_scores, axis=1), label='CV Score')
        plt.legend()
        plt.title("Validation Curve for Weak Learners (AdaBoost) \n LR = " + str(learning_rate) + ": " + dataset_name)
        plt.xlabel("Number of Weak Learners")
        plt.ylabel("Score")
        plt.grid()
        plt.show()

        train_scores, test_scores = validation_curve(AdaBoost_Tree_Classifier, X_train, y_train, param_name="learning_rate",
                                                     param_range=np.arange(.1, 1, .05), cv=5, n_jobs=n_cpu-3)

        plt.figure()
        plt.plot(np.arange(.1, 1, .05), np.mean(train_scores, axis=1), label='Train Score')
        plt.plot(np.arange(.1, 1, .05), np.mean(test_scores, axis=1), label='CV Score')
        plt.legend()
        plt.title("Validation Curve for Learning Rate (AdaBoost) \n Weak Learners = " + str(n_estimators) + ": " + dataset_name)
        plt.xlabel("Learning Rate")
        plt.ylabel("Score")
        plt.grid()
        plt.show()

        ########Optimized version
        # param_grid = {'learning_rate': np.arange(.1, 2, .05), 'n_estimators': np.arange(1, 60)}
        # AdaBoost_Tree_Classifier_Opt = GridSearchCV(AdaBoost_Tree_Classifier, param_grid = param_grid, cv=10, n_jobs=n_cpu-2)
        # AdaBoost_Tree_Classifier_Opt.fit(X_train, y_train)
        # print("Best params for BoostTree:", AdaBoost_Tree_Classifier_Opt.best_params_)

        ########Timing and Accuracy

        time_train_start = time.time()
        AdaBoostTree_Class = AdaBoostClassifier(learning_rate=learning_rate,
                                                       n_estimators=n_estimators)

        print("Optimized Model Created")
        AdaBoostTree_timing = AdaBoostTree_Class
        AdaBoostTree_timing.fit(X_train, y_train)
        time_train_stop = time.time()
        time_train = time_train_stop - time_train_start

        time_test_start = time.time()
        predictions = AdaBoostTree_timing.predict(X_test)
        print("Optimized Model Classification Report: \n", classification_report(y_test, predictions))
        time_test_stop = time.time()
        time_test = time_test_stop - time_test_start

        accuracy_optim = accuracy_score(y_test, AdaBoostTree_timing.predict(X_test))

        ######### Learning Curves

        train_size_abs, train_scores, test_scores = learning_curve(AdaBoostTree_Class, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=5)

        plt.figure()
        plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
        plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
        plt.legend()
        plt.title("Learning Curve (Boosted Tree), LR = " + str(learning_rate) + "\n Weak Learners = " + str(n_estimators) + ": " + dataset_name)
        plt.xlabel("Percentage of Training Examples")
        plt.ylabel("Score")
        plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
        plt.grid()

        plt.show()

        with open('model_results.txt', 'a') as f:
            f.writelines('Boosted Tree: ' + dataset_name + '\n')
            f.write('Accuracy of optimized model: ' + str(accuracy_optim) + '\n')
            f.write('Time to Train: ' + str(time_train) + '\n')
            f.write('Time to Test: ' + str(time_test) + '\n \n')
            f.write(str(classification_report(y_test, predictions)) + '\n \n')

    return X, Y
