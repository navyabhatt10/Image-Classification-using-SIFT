"""
IMAGE CLASSIFICATION USING SIFT FEATURES
"""
#IMPORTING THE STANDARD LIBRARIES 
import cv2
import os
import numpy as np
import joblib

train_path='C:/Python/Python38/train' #path of the folder train(contains two classes aeroplane and car)
training_names = os.listdir(train_path)# returns a list containing the names of the entries in the directory given by path(train_path)

#get path to all images and save them in a list
image_paths = []#empty list for the image path 
image_classes = []#empty list for the image class
class_id=0

#function to list all the file names in a directory
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]
  
#filling the empty lists with image path, image class and adding class id number
for training_name in training_names:
    dir=os.path.join(train_path,training_name)
    class_path=imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
des_list=[]#this is the empty list where all the descriptors will be stored

#SIFT locate the ‘keypoints‘ of the image
for image_path in image_paths:
    im=cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kpts, des = sift.detectAndCompute(im,None)
    des_list.append((image_path,des))#storing the keypoints into des_list
    
#stack all the descriptors vertically in a numpy array(convert into a vector for SVM)
descriptors=des_list[0][1]
for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors,descriptor))
    
#kmeans works only on float, so convert integers to float
descriptors_float=descriptors.astype(float)

#perform kmeans clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k=200#200 clusters give high accuracy
voc,variance=kmeans(descriptors_float,k,1)

#calculate histogram of the features and represent them as vectors
#vq assign a codes from a code book to observations
im_features=np.zeros((len(image_paths),k),"float32")
for i in range(len(image_paths)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w]+=1
        
#perform vectorization
nbr_occurences=np.sum((im_features>0)*1,axis=0)
idf=np.array(np.log((1.0*len(image_paths)+1)/(1.0*nbr_occurences+1)),'float32')

#perform normalization(standardize features by removing the mean and scaling to unit variance)
from sklearn.preprocessing import StandardScaler
stdSlr=StandardScaler().fit(im_features)
im_features=stdSlr.transform(im_features)

#Train a machine learning algorithm to descriminate vectors corresponding to positive and negative training using Support Vector Machine(SVM)
from sklearn.svm import LinearSVC
clf=LinearSVC(max_iter=50000)#default is 100 but with 100 accuracy is not good
clf.fit(im_features,np.array(image_classes))

#Now, joblib is used to dump the python objects into one single file 'nav.pkl'
joblib.dump((clf,training_names,stdSlr,k,voc),"nav.pkl",compress=3)
