# Load libraries

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import numpy as np
import pandas as pd
import os

import time

# Code reference: https://www.datacamp.com/tutorial/decision-tree-classification-python

def Decision_Tree(X, Y, dataset_name):

    scale = StandardScaler()
    n_cpu = os.cpu_count()

    # Split dataset into training set and test set: 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)

    # Scale Data
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)

    DT_Classifier = tree.DecisionTreeClassifier(random_state=0)

    ##### Validation Curve
        ##### MAX DEPTH
    train_scores, test_scores = validation_curve(DT_Classifier, X_train, y_train,
                                                 param_range=np.arange(1, 15), param_name='max_depth', cv=10)

    plt.figure()
    plt.plot(np.arange(1, 15), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.arange(1, 15), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve for Max Depth (Decision Tree): " + dataset_name)
    plt.xlabel("Max Depth of Tree")
    plt.ylabel("Score")
    plt.xticks(np.arange(1, 15, 2))
    plt.grid()
    plt.show()

        ##### CCP ALPHA

    train_scores, test_scores = validation_curve(DT_Classifier, X_train, y_train,
                                                 param_range=np.linspace(0, 0.035, 10), param_name='ccp_alpha', cv=10)

    plt.figure()
    plt.plot(np.linspace(0, 0.035, 10), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0, 0.035, 10), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve for CCP Alpha (Decision Tree): " + dataset_name)
    plt.xlabel("CCP Alpha")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0, 0.035, 8))
    plt.grid()
    plt.savefig('Validation Curve for CCP Alpha.png')
    plt.show()

    param_grid = {'max_depth': np.arange(1, 40), 'ccp_alpha': np.linspace(0, 0.05, 10)}

    ######Optimized
    DT_opt = GridSearchCV(DT_Classifier, param_grid=param_grid, cv=10, n_jobs=n_cpu-2)

    DT_opt.fit(X_train, y_train)
    print("Best params for k-NN:", DT_opt.best_params_)


    ########Timing and Accuracy

    time_train_start = time.time()
    DT_Classifier_timing = tree.DecisionTreeClassifier(ccp_alpha = DT_opt.best_params_['ccp_alpha'], max_depth = DT_opt.best_params_['max_depth'])
    print("Optimized Model Created")
    DT_Classifier_timing.fit(X_train, y_train)
    time_train_stop = time.time()
    time_train = time_train_stop - time_train_start

    time_test_start = time.time()
    predictions = DT_Classifier_timing.predict(X_test)
    print("Optimized Model Classification Report: \n", classification_report(y_test, predictions))
    time_test_stop = time.time()
    time_test = time_test_stop - time_test_start

    accuracy_optim = accuracy_score(y_test, DT_Classifier_timing.predict(X_test))



    if dataset_name == "Dataset_Spotify":
        max_depth  = 5
        ccp_alpha = .005


    if dataset_name == "Dataset_Raisins":
        max_depth  = 4
        ccp_alpha = .025




    # Learning Curve without any Pruning
    train_size_abs, train_scores, test_scores = learning_curve(DT_Classifier, X_train, y_train,
                                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve DT - Unpruned: " + dataset_name)
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    plt.show()

    # Learning Curve with Max Depth Pruning
    DT_Classifier_max_depth = tree.DecisionTreeClassifier(random_state=0, max_depth = max_depth)
    train_size_abs, train_scores, test_scores = learning_curve(DT_Classifier_max_depth, X_train, y_train,
                                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve DT - Pruned \n Max Depth = " + str(max_depth) + ": " + dataset_name )
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    # Learning Curve with ccp_alpha for pruning
    DT_Classifier_ccp = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    train_size_abs, train_scores, test_scores = learning_curve(DT_Classifier_ccp, X_train, y_train,
                                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve DT - Pruned \n ccp_alpha = " + str(ccp_alpha) + ": " + dataset_name)
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()
    plt.show()





    with open('model_results.txt', 'a') as f:
        f.writelines('SVM: ' + dataset_name + '\n')
        f.write('Accuracy of optimized model: ' + str(accuracy_optim) + '\n')
        f.write('Time to Train: ' + str(time_train) + '\n')
        f.write('Time to Test: ' + str(time_test) + '\n \n')
        f.write(str(classification_report(y_test, predictions)) + '\n \n')

    return X, Y



