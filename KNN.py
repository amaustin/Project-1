# Load libraries

# Recall/PrecisionReference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py


from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
import os


import time

# Code reference: https://www.datacamp.com/tutorial/decision-tree-classification-python

def KNN(X, Y, dataset_name):
    n_cpu = os.cpu_count()

    min_max_scaler = preprocessing.MinMaxScaler()

    # Split dataset into training set and test set: 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35,random_state=0)
    scale = StandardScaler()
    # Scale Data
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)


    KNN_Classifier_nn = KNeighborsClassifier()
    train_scores, test_scores = validation_curve(KNN_Classifier_nn, X_train, y_train,
                                                 param_range=np.arange(1, 50), param_name='n_neighbors', cv=10, n_jobs=n_cpu-2)

    plt.figure()
    plt.plot(np.arange(1, 50), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.arange(1, 50), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve for k-NN, Number of Leaves = 30 \n Number of Neighbors': " + dataset_name)
    plt.xlabel("Num Neighbors")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

    KNN_Classifier_leaves = KNeighborsClassifier(n_neighbors = 45)
    train_scores, test_scores = validation_curve(KNN_Classifier_leaves, X_train, y_train,
                                                 param_range=np.arange(1, 100), param_name='leaf_size', cv=10, n_jobs= n_cpu-2)

    plt.figure()
    plt.plot(np.arange(1, 100), np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.arange(1, 100), np.mean(test_scores, axis=1), label='Cross Val Score')
    plt.legend()
    plt.title("Validation Curve for k-NN, Neighbors = 45, Number of Leaves': " + dataset_name)
    plt.xlabel("Num Leaves")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

    param_grid = {'n_neighbors': np.arange(1, 50), 'leaf_size': np.arange(1, 100)}
    classifier_knn_best = GridSearchCV(KNN_Classifier_leaves, param_grid=param_grid, cv=10, n_jobs=n_cpu-2)

    classifier_knn_best.fit(X_train, y_train)
    print("Best params for k-NN:", classifier_knn_best.best_params_)

    ########Timing and Accuracy

    time_train_start = time.time()
    classifier_knn_learning = KNeighborsClassifier(n_neighbors=classifier_knn_best.best_params_['n_neighbors'])
    print("Optimized Model Created")
    KNN_timing = classifier_knn_learning
    KNN_timing.fit(X_train, y_train)
    time_train_stop= time.time()
    time_train = time_train_stop - time_train_start


    time_test_start = time.time()
    predictions = KNN_timing.predict(X_test)
    print("Optimized Model Classification Report: \n", classification_report(y_test, predictions))
    time_test_stop = time.time()
    time_test = time_test_stop - time_test_start

    accuracy_optim = accuracy_score(y_test, KNN_timing.predict(X_test))

    ######################### Learning Curves

    train_size_abs, train_scores, test_scores = learning_curve(classifier_knn_learning, X_train, y_train,
                                                  train_sizes=np.linspace(0.1, 1.0, 10), cv=10)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve (k-NN): " + dataset_name)
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    plt.show()

    with open('model_results.txt', 'a') as f:
        f.writelines('KNN: ' + dataset_name + '\n')
        f.write('Accuracy of optimized model: ' + str(accuracy_optim) + '\n')
        f.write('Time to Train: ' + str(time_train) + '\n')
        f.write('Time to Test: ' + str(time_test) + '\n \n')
        f.write(str(classification_report(y_test, predictions)) + '\n \n')

    return X,Y