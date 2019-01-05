from keras.models import load_model
model = load_model("/home/box/Desktop/MyPy/shit-happening/CNN/Pneumonia-cnn/model.hdf5")
import cv2
IMG_SIZE = 200
import numpy as np
img = "/home/box/Desktop/MyPy/shit-happening/CNN/chest_xray/test/PNEUMONIA/person21_virus_52.jpeg"
img_array = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
x = np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE,1)
x = x/255.0
image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
disease = model.predict(x)
print(disease)
if disease[0][0] > 0.6:
    cv2.putText(image,"Safe", (350,350), 2,  cv2.FONT_HERSHEY_SIMPLEX, (255,255,255))
else:
    cv2.putText(image,"Pneumonia", (350,350), 2, cv2.FONT_HERSHEY_SIMPLEX, (255,0,0))
    print("new-monea")
    
cv2.imshow('dst_rt', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
