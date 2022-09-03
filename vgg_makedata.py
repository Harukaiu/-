from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["Flower","Food","Others"]
num_classes = len(classes)
image_size = 224

X = [] 
Y = [] 

for index, classlabel in enumerate(classes):
    
    photos_dir = "images1124/traindata/" + classlabel
    files = glob.glob(photos_dir + "/*")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image) 
        X.append(data)
        Y.append(index)
    print("ok,", len(Y))

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("chop_data/chop" + str(image_size) +".npy", xy) 
print("ok,", len(Y))
