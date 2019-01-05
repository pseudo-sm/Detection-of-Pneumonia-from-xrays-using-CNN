import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

# path where your data directory is stored. Subdirectories in the data_directory
# must have the name of the classes of the image inside them.
DATADIR = '/path/to/data_directory'
# Names of all classes in a list
CATEGORIES = ['Class A','Class B','Class C']

IMG_SIZE = 200
training_data  =[]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #pick each image from each directory and convert it into a 200 X 200 image in gray scale.
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            
create_training_data()

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
# Generate Features and Labels vector
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

# Pickle both vectors so you dont have to do it every time.
import pickle
pickle_out = open("X.pickle",'wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle",'wb')
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
x = pickle.load(pickle_in)

