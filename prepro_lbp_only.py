import numpy as np
import cv2
import glob
from skimage import feature
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from matplotlib import pyplot as plt
import csv
import pandas as pd
import h5py


def importing_images():
    images = []
    beans_files = sorted(glob.glob ("beans\\*.jpg"))
    chickpeas_files = sorted(glob.glob ("chickpea\\*.jpg"))
    hazelnuts_files = sorted(glob.glob ("hazelnut\\*.jpg"))
    lentil_files = sorted(glob.glob ("lentil\\*.jpg"))

    for myFile in beans_files:
        print(myFile)
        image = cv2.imread (myFile)
        images.append(image)

    for myFile in chickpeas_files:
        print(myFile)
        image = cv2.imread (myFile)
        images.append(image)

    for myFile in hazelnuts_files:
        print(myFile)
        image = cv2.imread (myFile)
        images.append(image)

    for myFile in lentil_files:
        print(myFile)
        image = cv2.imread (myFile)
        images.append(image)

    return images

def resize(image):
    height = 1080
    width = 1920
    dim = (width,height)
    res_img = cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
    return res_img

def lbp_histograms(image,numPoints,radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="default")
    x, bin_edges = np.histogram(lbp.ravel(), bins=256)
    hist = x / sum(x)
    return hist




#Starting program

#Initializing feature list
global_features = []

#making target list
y = []
for x in range(50):
    y.append("bean")
for x in range(50):
    y.append("chickpea")
for x in range(50):
    y.append("hazelnut")
for x in range(50):
    y.append("lentil")
#y_t = np.array(y).T


images = importing_images()

# for x in range(200):
#     images[x] = resize(images[x])
#     print("Image ", x , " resized...")


images_for_desc = images
for x in range(200):
    current_hist = lbp_histograms(images_for_desc[x],8,2)
    global_feature = np.hstack([current_hist])
    global_features.append(global_feature)
    print("Iteration ", x, " / Current image features extracted...")

#Normalizing features by scaling them
print("Normalizing feature vector")
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(global_features)

labelEncoder = LabelEncoder()
target = labelEncoder.fit_transform(y)

#y = np.array(y).astype(int)



#Saving in h5 file
print("Saving features and targets...")
h5f_features = h5py.File('data.h5', 'w')
h5f_features.create_dataset('dataset_1', data = np.array(scaled_features))

h5f_targets = h5py.File('labels.h5', 'w')
h5f_targets.create_dataset('dataset_1', data = np.array(target))

h5f_features.close()
h5f_targets.close()







#print(len(images))
#print(len(y_t),y_t)
