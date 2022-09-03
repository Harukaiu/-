import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from focal_loss import BinaryFocalLoss
from PIL import Image
import os, glob
import openpyxl
from sklearn import model_selection
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import csv
import datetime

dir_name="images0107/"
datanumber=6
dt_now=datetime.datetime.now()
dtm=dt_now.month
dtd=dt_now.day
classes = ["0", "2","3", "others"]
num_classes = len(classes)
epochs = 20
image_size = 224
npy_name = str(dtm) + str(dtd) + "_" + str(image_size)

excel_file=dir_name + '/result_' + str(datanumber) + str(dtm)+str(dtd)+'.xlsx'
model_weights=dir_name+ str(dtm) + str(dtd) +"_"+ str(datanumber)+"_vgg.h5"
# output_dir=dir_name +"dataA_" + str(datanumber) +"/"
# os.makedirs(output_dir,exist_ok=True)
X = [] 
Y = [] 


datagen = keras.preprocessing.image.ImageDataGenerator(#normalization・上下に移動・左右に移動・せん断・明るさ・回転・左右反転・縮小
    # featurewise_std_normalization=True,
    # height_shift_range=0.1,
    # width_shift_range=0.1,
    # shear_range = 5,
    vertical_flip=True,#上下反転
    horizontal_flip=True,
    zoom_range=[0.8,1.6],
    rotation_range=60)
for index, classlabel in enumerate(classes):
    
    photos_dir = dir_name + "traindata_aug/" + classlabel
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
# np.save("chop_data/chop" + str(image_size) +".npy", xy) 
print("ok,", len(Y))

#(2)
# X_train, X_test, y_train, y_test = np.load("chop_data/chop" + str(image_size) +".npy",allow_pickle=True)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") /255.0

#(3)
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))

print("VGG16の全結合層抜きモデル")
model.summary()

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

print("Sequentialモデル")
top_model.summary()

model = Model(inputs=model.input, outputs=top_model(model.output))

print("統合モデル")
model.summary()

#(4)#13でやってた
for layer in model.layers[:13]:
    layer.trainable = False
for i in model.layers:
    print(i.name, i.trainable)

opt = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
# model.compile(loss=BinaryFocalLoss(gamma=1), optimizer=opt,metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1)

# 評価に用いるモデル重みデータの保存
checkpointer = ModelCheckpoint(model_weights, monitor='val_loss', verbose=1, save_best_only=True)
# checkpointer = ModelCheckpoint(model_weights, monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=True)

# history=model.fit(X_train, y_train, batch_size=32, nb_epoch=epochs)
# history=model.fit(X_train, y_train, batch_size=32, nb_epoch=epochs, validation_split=0.2)
history=model.fit_generator(datagen.flow(X_train, y_train, batch_size=32, save_format='png'),
                    steps_per_epoch=int(len(X_train) / 32), validation_data=(X_test, y_test),epochs=epochs,callbacks=[checkpointer])

# history=model.fit_generator(datagen.flow(X_train, y_train, batch_size=32,save_to_dir=output_dir, save_format='png'),
#                     steps_per_epoch=int(len(X_train) / 32), validation_data=(X_test, y_test),epochs=epochs,callbacks=[early_stopping, checkpointer])
# print(len(X_train))
###
print(history.history)
plt.plot(history.history['accuracy'], label="accuracy")
plt.plot(history.history['val_accuracy'], label="val_accuracy")
plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
plt.legend()
plt.savefig(dir_name + npy_name + "_" + str(epochs) +"_" + str(dtm) + str(dtd) + "_" + str(datanumber)+ "_acc-epoch")
plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close()

plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")

# plt.plot(range(1, epochs+1), history.history['loss'], label="loss")
# plt.plot(range(1, epochs+1), history.history['val_loss'], label="val_loss")
plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
plt.legend()
plt.savefig(dir_name + npy_name + "_" + str(epochs) +"_" + str(dtm) + str(dtd) + "_"+ str(datanumber) + "_loss-epoch")

plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close()
###

score = model.evaluate(X_test, y_test, batch_size=32)


wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = 'sheet1'
wb.save(excel_file)
book=openpyxl.load_workbook(excel_file)
sheet = book['sheet1']
sheet['A1'] = 'loss'
sheet['B1'] = 'acc'
sheet['A2'] = score[0]
sheet['B2'] = score[1]
book.save(excel_file)
# model.save(dir_name+ str(dtm) + str(dtd) +"_"+ str(datanumber)+"_vgg.h5")

#https://qiita.com/ykoji/items/e9c2c5d7288c6290d21b
