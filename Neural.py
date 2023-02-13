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

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import os
import warnings




import time

# Code reference: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

def Neural(X, Y, dataset_name):
    n_cpu = os.cpu_count()
    warnings.warn("once")

    min_max_scaler = preprocessing.MinMaxScaler()

    # Split dataset into training set and test set: 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35,random_state=0)
    scale = StandardScaler()
    # Scale Data
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)


    # param_grid = [
    #     {
    #         'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #         'solver': ['sgd', 'adam'],
    #         'hidden_layer_sizes': [
    #             (10, ),(45,), (50,), (55, ),(60,),(100,), (200,)
    #         ],
    #         'learning_rate_init': [.01, .05, .1, .25, .5, .75, 1]
    #     }
    # ]
    #
    NN_Basic = MLPClassifier(hidden_layer_sizes= (100,), learning_rate_init=.25, random_state=1)
    print("Basic Model Created")
    NN_Basic.fit(X_train, y_train)
    predictions = NN_Basic.predict(X_test)
    print("Basic Model Classification Report: \n", classification_report(y_test, predictions))



    if dataset_name == 'Dataset_Spotify':
        learning_rate = .1
        activation = 'logistic'
        h_layers = (50,)
        solver = 'sgd'

    if dataset_name == 'Dataset_Raisins':
        learning_rate = .15
        activation = 'tanh'
        h_layers = (45,)
        solver = 'sgd'

    ###### TIMING AND ACCURACY

    time_train_start = time.time()
    NN_Optimized = MLPClassifier(learning_rate_init = learning_rate, activation = activation, hidden_layer_sizes= h_layers, solver = solver, random_state = 1)
    print("Optimized Model Created")
    NN_Optimized.fit(X_train, y_train)
    time_train_stop = time.time()
    time_train = time_train_stop - time_train_start

    time_test_start = time.time()
    predictions = NN_Optimized.predict(X_test)
    print("Optimized Model Classification Report: \n", classification_report(y_test, predictions))
    time_test_stop = time.time()
    time_test = time_test_stop - time_test_start

    accuracy_optim = accuracy_score(y_test, NN_Optimized.predict(X_test))

    train_scores, test_scores = validation_curve(NN_Optimized, X_train, y_train, param_name="hidden_layer_sizes",
                                                 param_range=np.arange(1, 100, 5), cv=8, n_jobs=n_cpu-4)

    plt.figure()
    plt.semilogx(np.arange(1, 100, 5), np.mean(train_scores, axis=1), label='Train Score')
    plt.semilogx(np.arange(1, 100, 5), np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Validation Curve for Hidden Layers (Neural Network): " + dataset_name)
    plt.xlabel("Hidden Layers")
    plt.ylabel("Score")
    plt.grid()
    plt.show()



    # Loss Curve: Basic Model vs Tuned Model
    plt.figure()
    plt.plot(NN_Basic.loss_curve_, label = 'Hidden Layers: 100, Learning Rate: .25')
    plt.legend()
    plt.title("Loss Function: Hidden Layer and Learning Rate Comparison: \n" + dataset_name  )
    plt.xlabel("Number of Steps")
    plt.ylabel("Loss")
    #plt.xticks(np.arange(1, 200, 2))
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(NN_Optimized.loss_curve_,
             label='Hidden Layers: ' + str(h_layers) + ', Learning Rate: ' + str(learning_rate))
    plt.legend()
    plt.title("Loss Function: Hidden Layer and Learning Rate Comparison: \n" + dataset_name  )
    plt.xlabel("Number of Steps")
    plt.ylabel("Loss")
    #plt.xticks(np.arange(1, 200, 2))
    plt.grid()
    plt.show()

    # Learning Curve
    train_size_abs, train_scores, test_scores = learning_curve(NN_Optimized, X_train, y_train,
                                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=8)

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1, 1.0, 10) * 100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve: Neural Network" + ": " + dataset_name )
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()

    plt.show()

    with open('model_results.txt', 'a') as f:
        f.writelines('Neural Network: ' + dataset_name + '\n')
        f.write('Accuracy of optimized model: '+ str(accuracy_optim) + '\n')
        f.write('Time to Train: ' + str(time_train) + '\n')
        f.write('Time to Test: ' + str(time_test) + '\n \n')
        f.write(str(classification_report(y_test, predictions)) + '\n \n')


    return X,Y