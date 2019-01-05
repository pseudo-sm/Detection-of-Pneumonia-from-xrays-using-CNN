import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

DATADIR = '/home/box/Desktop/MyPy/shit-happening/CNN/workspace/data_dir'
CATEGORIES = ['NORMAL','PNEUMONIA']

IMG_SIZE = 200
training_data  =[]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            print(img)
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            
create_training_data()
print(len(training_data))

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

import pickle
pickle_out = open("X.pickle",'wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle",'wb')
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
x = pickle.load(pickle_in)

print(x)

