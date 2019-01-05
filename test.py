from keras.models import load_model
model = load_model("path/to/model.hdf5")
import cv2
IMG_SIZE = 200
import numpy as np
img = "/path/to/test/image"
# Convert to required format of image that is 200 X 200 and gray scale.
img_array = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
x = np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#normalize
x = x/255.0
image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
class_name = model.predict(x)
print(class_name)
