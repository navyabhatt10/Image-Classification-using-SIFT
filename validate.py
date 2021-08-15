#IMPORTING THE STANDARD LIBRARIES 
import cv2
import os
import numpy as np
import pylab as pl
import joblib
from sklearn.metrics import confusion_matrix,accuracy_score

#load the classifier, class names, scalar, number of clusters and vocabulary from stored pickle file(generated during training)
clf,classes_names,stdSlr,k,voc=joblib.load("nav.pkl")
test_path='C:/Python/Python38/validate'#path of the folder validate(contains two classes aeroplane and car)
testing_names = os.listdir(test_path)# returns a list containing the names of the entries in the directory given by path(train_path)

#get path to all images and save them in a list
image_paths = []#empty list for the image path 
image_classes = []#empty list for the image class
class_id=0

#function to list all the file names in a directory
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]
  
#filling the empty lists with image path, image class and adding class id number
for testing_name in testing_names:
    dir=os.path.join(test_path,testing_name)
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
for image_path,descriptor in des_list[0:]:
    descriptors=np.vstack((descriptors,descriptor))
#calculate histogram of the features and represent them as vectors
#vq assign a codes from a code book to observations
from scipy.cluster.vq import vq
test_features=np.zeros((len(image_paths),k),"float32")
for i in range(len(image_paths)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w]+=1
        
#perform vectorization
nbr_occurences=np.sum((test_features>0)*1,axis=0)
idf=np.array(np.log((1.0*len(image_paths)+1)/(1.0*nbr_occurences+1)),'float32')

#Standardize features by removing the mean and scaling to unit variance
#stdSlr comes from the imported pickle file
test_features=stdSlr.transform(test_features)

#report true class names so that they can be compared with predicted classes
true_class=[classes_names[i] for i in image_classes]
#perform predictions and report the predicted class names
predictions=[classes_names[i] for i in clf.predict(test_features)]

#printing the true class and predictions
print("true_class=" + str(true_class))
print("prediction=" + str(predictions))
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()
accuracy=accuracy_score(true_class,predictions)
print("accuracy=",accuracy)

cm=confusion_matrix(true_class,predictions)
print(cm)
showconfusionmatrix(cm)
