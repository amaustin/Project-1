# Load libraries

# Recall/PrecisionReference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn import tree
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix

from ML_tools import plot_learning_curve, plot_validation_curve, confusion

from sklearn.tree import export_graphviz
import sklearn.externals
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

import plot_classifiers as plot_c


import time

# Code reference: https://www.datacamp.com/tutorial/decision-tree-classification-python

def SVM(X, Y, dataset_name):

    start_time = time.time()
    scale = StandardScaler()
    n_cpu = os.cpu_count()

    print("CPUs: ", n_cpu)
    # Split dataset into training set and test set: 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)

    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)

    ### Validation Curves ###
    #  param_range=np.logspace(-4,1,10)
    # varying hyperparameter C and kernel
    SVM_Classifier_rbf = svm.SVC(kernel = 'rbf', random_state = 1 )
    train_scores, test_scores = validation_curve(SVM_Classifier_rbf, X_train, y_train, param_range=np.arange(.05, 2, .05)
                                                , param_name='C', cv=10)
    # classifier_name = "(SVM - RBF kernel)"
    # data_set_name = dataset_name
    # plot_c.plot_classifiers(train_scores, test_scores, range, classifier_name, data_set_name)

    plt.figure()
    plt.plot(np.arange(.05, 2, .05), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.arange(.05, 2, .05), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve (SVM - RBF kernel)" + ": " + dataset_name)
    plt.xlabel("C")
    plt.ylabel("Score")
    #plt.xticks(np.arange(1, 100, 2))
    plt.grid()
    plt.savefig('SVM_ValidationScore.png')
    plt.show()

    # varying hyperparameter kernel
    SVM_Classifier_linear = svm.SVC(kernel='linear', random_state=1)
    train_scores, test_scores = validation_curve(SVM_Classifier_linear, X_train, y_train, param_name="C",
                                                 param_range=np.arange(.05, 2, .05), cv=10)

    # classifier_name = "(SVM - Linear kernel)"
    # data_set_name = dataset_name
    # plot_c.plot_classifiers(train_scores, test_scores, range, classifier_name, data_set_name)
    plt.figure()
    plt.plot(np.arange(.05, 2, .05), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.arange(.05, 2, .05), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve (SVM - Linear kernel)" + ": " + dataset_name)
    plt.xlabel("C")
    plt.ylabel("Score")
    #plt.xticks(np.arange(1, 100, 2))
    plt.grid()
    plt.savefig('SVM_ValidationScore.png')
    plt.show()

    # varying hyperparameter kernel -
    SVM_Classifier_poly = svm.SVC(kernel='poly', random_state=1)
    train_scores, test_scores = validation_curve(SVM_Classifier_poly, X_train, y_train, param_name="C",
                                                 param_range=np.arange(.05, 2, .05), cv=10)
    # classifier_name = "(SVM - Poly kernel)"
    # data_set_name = dataset_name
    # plot_c.plot_classifiers(train_scores, test_scores, range, classifier_name, data_set_name)

    plt.figure()
    plt.plot(np.arange(.05, 2, .05), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.arange(.05, 2, .05), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve (SVM - Poly kernel)" + ": " + dataset_name)
    plt.xlabel("C")
    plt.ylabel("Score")
    #plt.xticks(np.arange(1, 100, 2))
    plt.grid()
    plt.savefig('SVM_ValidationScore.png')
    plt.show()

    ########Basic
    basicSVMmodel = svm.SVC()
    basicSVMmodel.fit(X_train, y_train)

    # print prediction results
    predictions = basicSVMmodel.predict(X_test)
    print("Basic Model Classification Report: \n", classification_report(y_test, predictions))


    # ########OPTIMIZED
    #
    # tuned_parameters =[
    #     {"kernel": ["rbf"], "gamma": np.linspace(0.001, 2, 10), "C": np.linspace(0, 10, 10)},
    #     {"kernel": ["linear"], "C": np.linspace(0.1, 10, 10)}, {"kernel": ["poly"], "gamma": np.linspace(0.001, 10, 10), "C": np.linspace(0, 10, 10)}]
    # bestSVMmodel = GridSearchCV(basicSVMmodel, param_grid = tuned_parameters, cv=5, n_jobs=n_cpu-2)
    #
    #
    # start_time = time.time()
    # bestSVMmodel.fit(X_train, y_train)
    # end_time = time.time()
    # time_train[0] = end_time - start_time
    # print("Best_params for SVM:", bestSVMmodel.best_params_)
    #
    # start_time = time.time()
    # classifier_accuracy[0] = accuracy_score(y_test, bestSVMmodel.predict(X_test))
    # end_time = time.time()
    # time_test[0] = end_time - start_time
    # print("Accuracy for best SVM:", classifier_accuracy[0])
    # print("Time to train: ", time_train)
    # print("Time to test: ", time_test)
    #
    #

    ########Timing and Accuracy

    C_poly = .25
    if dataset_name == "Dataset_Spotify":
        C_poly = .2

    # TODO: set C value for raisins
    if dataset_name == "Dataset_Raisins":
        C_poly = .5

    time_train_start = time.time()
    SVM_Classifier_timing = svm.SVC(kernel='poly', random_state=1, C = C_poly)
    print("Optimized Model Created")
    SVM_Classifier_timing.fit(X_train, y_train)
    time_train_stop = time.time()
    time_train = time_train_stop - time_train_start

    time_test_start = time.time()
    predictions = SVM_Classifier_timing.predict(X_test)
    print("Optimized Model Classification Report: \n", classification_report(y_test, predictions))
    time_test_stop = time.time()
    time_test = time_test_stop - time_test_start

    accuracy_optim = accuracy_score(y_test, SVM_Classifier_timing.predict(X_test))




    ### Learning Curves ###

    ################################# Linear
    C_poly = .25
    if dataset_name == "Dataset_Spotify":
        C_poly = .2

    # TODO: set C value for raisins
    if dataset_name == "Dataset_Raisins":
        C_poly = .5

    # Build optimized
    train_size_abs, train_scores, test_scores = learning_curve(SVM_Classifier_linear, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve (SVM - Linear kernel)" + ": " + dataset_name + " C = " + str(C_poly))
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    plt.show()

    ################################# RBF
    C_poly = .25
    if dataset_name == "Dataset_Spotify":
        C_poly = .6

    # TODO: set C value for raisins
    if dataset_name == "Dataset_Raisins":
        C_poly = .5

    train_size_abs, train_scores, test_scores = learning_curve(SVM_Classifier_rbf, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve (SVM - RBF kernel)" + ": " + dataset_name + " C = " + str(C_poly))
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    plt.show()

    ################################# POLY
    C_poly = .25
    if dataset_name == "Dataset_Spotify":
        C_poly = 1.6

    #TODO: set C value for raisins
    if dataset_name == "Dataset_Raisins":
        C_poly = .5

    SVM_Classifier_poly_1= svm.SVC(kernel='poly', C=C_poly ,random_state=1)
    train_size_abs, train_scores, test_scores = learning_curve(SVM_Classifier_poly_1, X_train, y_train,
                                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve (SVM - Poly Kernel)"  + ": " + dataset_name + " C = " + str(C_poly))
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    plt.show()

    end_time = time.time()

    total_time = end_time - start_time

    with open('model_results.txt', 'a') as f:
        f.writelines('SVM: ' + dataset_name + '\n')
        f.write('Accuracy of optimized model: ' + str(accuracy_optim) + '\n')
        f.write('Time to Train: ' + str(time_train) + '\n')
        f.write('Time to Test: ' + str(time_test) + '\n \n')
        f.write(str(classification_report(y_test, predictions)) + '\n \n')


    return X,Y