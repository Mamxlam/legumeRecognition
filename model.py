import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 3 models for testing
def neuralNetwork(Xtrain, Ytrain, Xtest):
    # Neural network model
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10),max_iter=3500)
    clf.fit(Xtrain, Ytrain)
    # calculate predictions
    predictions = clf.predict(Xtest)

    return predictions


def randomForest(Xtrain, Ytrain, Xtest):
    clf = RandomForestClassifier(n_estimators=14)
    clf.fit(Xtrain, Ytrain)
    prediction = clf.predict(Xtest)

    return prediction


def knn(Xtrain, Ytrain, Xtest):
    neigh = KNeighborsClassifier(n_neighbors=15)
    neigh.fit(Xtrain, Ytrain)
    prediction = neigh.predict(Xtest)

    return prediction


# Metrics
def accuracy(Ytest, prediction):
    accuracy = accuracy_score(Ytest, prediction)

    return accuracy


def precision(Ytest, prediction):
    precision = precision_score(Ytest, prediction,average='weighted')

    return precision


def recall(Ytest, prediction):
    recall = recall_score(Ytest, prediction,average='weighted')

    return recall


def Fmeasure(Ytest, prediction):
    Fmeasure = f1_score(Ytest, prediction,average='weighted')

    return Fmeasure

def print_metrics(X_train, Y_train, X_test, Y_test, prediction):
    accuracy_score = accuracy(Y_test, prediction)
    precision_score = precision(Y_test, prediction)
    recall_score = recall(Y_test, prediction)
    Fmeasure_score = Fmeasure(Y_test, prediction)

    print('The accuracy of the classifier is:', accuracy_score * 100, '%')
    print('The precision of the classifier is:', precision_score * 100, '%')
    print('The recall of the classifier is:', recall_score * 100, '%')
    print('The F-measure of the classifier is:', Fmeasure_score * 100, '%')









#Reading datasets saved in h5 format
print("Reading features and targets...")

h5f_features = h5py.File('data.h5','r')
h5f_targets = h5py.File('labels.h5','r')

gl_feat_dataset = h5f_features['dataset_1']
gl_targ_dataset = h5f_targets['dataset_1']

global_features = np.array(gl_feat_dataset)
global_targets = np.array(gl_targ_dataset)

h5f_features.close()
h5f_targets.close()


#Train/Test/Split
print("Splitting data into training and testing...")

X_train, X_test, y_train, y_test = train_test_split(global_features, global_targets, test_size=0.2)


# Random Forest
print('')
print('********** Random Forest classifier **********')
randomF_pred = randomForest(X_train, y_train, X_test)
print_metrics(X_train, y_train, X_test, y_test, randomF_pred)

# Neural Network
print('')
print('********** Neural Network classifier **********')
neural_pred = neuralNetwork(X_train, y_train, X_test)
print_metrics(X_train, y_train, X_test, y_test, neural_pred)

# kNN
print('')
print('********** kNN classifier **********')
knn_pred = knn(X_train, y_train, X_test)
print_metrics(X_train, y_train, X_test, y_test, knn_pred)
