import cv2
import numpy as np
import h5py
from skimage import feature
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def lbp_histograms(image,numPoints,radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="default")
    x, bin_edges = np.histogram(lbp.ravel(), bins=256)
    hist = x / sum(x)
    return hist

def hu_moments(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def resize(image):
    height = 256
    width = 256
    dim = (width,height)
    res_img = cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
    return res_img

def neuralNetwork(Xtrain, Ytrain, Xtest):
    # Neural network model
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10 ,10),max_iter=3500)
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

def color_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()






#Start of script


cap = cv2.VideoCapture("http://192.168.1.4:8080/video/mjpeg")
single_feature = []

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

while(True):
    ret, frame = cap.read()
    frame = resize(frame)
    current_hist = lbp_histograms(frame, 8, 2)
    current_moment = hu_moments(frame)
    current_color = color_histogram(frame)
    single_feature = np.hstack([current_hist, current_moment, current_color])
    single_feature = np.array(single_feature)
    single_feature = single_feature.reshape(1, -1)
    #rf_prediction = randomForest(global_features,global_targets,single_feature)
    nn_prediction = neuralNetwork(global_features,global_targets,single_feature)
    # knn_prediction = knn(global_features,global_targets,single_feature)

    #print("Random Forest Prediction : ", rf_prediction)
    print("Neural Network Prediction : ", nn_prediction)
    # print("K-Nearest Neighbor Prediction : ",  knn_prediction)

    #cv2.putText(frame,rf_prediction,org=(0,0),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1, color=(255, 255, 255))
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break