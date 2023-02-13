# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import DecisionTree as DTree
import KNN as KN
import Neural as NN
import SVM as SV
import BoostTree as BT
import DataProcessing
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # clear last run of model_results.txt
    file = open('model_results.txt', 'w')
    file.close()

    # for scaling data
    scale = StandardScaler()

    ### First Data Set Loading ###
    # Multi-class dataset (not balanced with noise)
    data_set_1 = pd.read_csv('SpotifyAudioFeatures2019_subset5500_popularity_test_noise.csv')
    target_1 = ['Popularity_Bins']
    features_1 = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
    X1 = data_set_1[features_1]
    Y1 = data_set_1.Popularity_Bins



    ### Second Data Set Loading ###
    # Binary class dataset (perfectly balanced)
    data_set_2 = pd.read_csv('Raisin_Dataset.csv')
    target_2 = ['Class']
    features_2 = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter']
    X2 = data_set_2[features_2]
    Y2 = data_set_2.Class

    ### Machine Learning Algorithms ###
    #### Decision Tree ####
    DTree.Decision_Tree(X1, Y1, 'Dataset_Spotify')
    DTree.Decision_Tree(X2, Y2, 'Dataset_Raisins')

    # #### Boosted ####
    BT.BoostTree(X1, Y1, 'Dataset_Spotify')
    BT.BoostTree(X2, Y2, 'Dataset_Raisins')

    # #### KNN ####
    KN.KNN(X1, Y1, 'Dataset_Spotify')
    KN.KNN(X2, Y2, 'Dataset_Raisins')
    #
    # #### Neural Network ####
    #
    NN.Neural(X1, Y1, 'Dataset_Spotify')
    NN.Neural(X2, Y2, 'Dataset_Raisins')


    # #### SVM ####
    #SV.SVM(X1, Y1, 'Dataset_Spotify')
    #SV.SVM(X2, Y2, 'Dataset_Raisins')

    print('DUN')