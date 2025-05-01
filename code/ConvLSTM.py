# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:21:48 2020

@author: Samanvi
"""

## Importing necessary libraries
#from PIL import Image
from c_path_utils import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import time as time
import cv2

## loading the data set 
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset(650,0.1, 101)

### Slicing parameters
im_w = 256
im_h = 128
slice_x = 3
no_slices = im_w//slice_x


## intiating variables
X_train = np.zeros((train_set_x_orig.shape[0],no_slices,im_h,slice_x))
y_train = np.zeros((train_set_y_orig.shape[0],no_slices,im_h,slice_x))
X_test = np.zeros((test_set_x_orig.shape[0],no_slices,im_h,slice_x))
y_test = np.zeros((test_set_y_orig.shape[0],no_slices,im_h,slice_x))

X_temp = np.zeros((im_h,slice_x))

## processing the variables
X_train_temp = train_set_x_orig.reshape(train_set_x_orig.shape[0],im_h,im_w,4)
X_test_temp  = test_set_x_orig.reshape(test_set_x_orig.shape[0],im_h,im_w,4)
y_train_temp = train_set_y_orig.reshape(train_set_y_orig.shape[0],im_h,im_w)
y_test_temp  = test_set_y_orig.reshape(test_set_y_orig.shape[0],im_h,im_w)

##increasing crack width on training data
for im in range(train_set_x_orig.shape[0]):
    for i in range(im_w):
        k1 = 0
        k2 = 1
        for i1 in range(im_h):
            if y_train_temp[im,i1,i]!=0 and k1==0:
                k1 =i1
            elif y_train_temp[im,i1,i]!=0 and k1!=0:
                k2 += 1
        y_train_temp[im,k1+k2//2-3:k1+k2//2+3,i] = 100
    y_train_temp[im,0:10,:]   = 0
    y_train_temp[im,-10:-1,:] = 0  

##increasing crack width on training data
for im in range(test_set_x_orig.shape[0]):
    for i in range(im_w):
        k1 = 0
        k2 = 1
        for i1 in range(im_h):
            if y_test_temp[im,i1,i]!=0 and k1==0:
                k1 =i1
            elif y_test_temp[im,i1,i]!=0 and k1!=0:
                k2 += 1
        y_test_temp[im,k1+k2//2-3:k1+k2//2+3,i] = 100
    y_test_temp[im,0:10,:]   = 0
    y_test_temp[im,-10:-1,:] = 0 
###Normalize the data set
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#train_set_x_orig = scaler.fit_transform(train_set_x_orig)
#test_set_x_orig  = scaler.fit_transform(test_set_x_orig)

###Converting the training data to required format
for im in range(train_set_x_orig.shape[0]):
    image_x = (Image.fromarray(X_train_temp[im,:,:].astype("uint8"))).convert("L")
    for i in range(no_slices):
        temp_x = image_x.crop((i*slice_x,0,(i+1)*slice_x,im_h))
        temp_a = np.array(temp_x)
        #for t1 in range(temp_a.shape[0]):
        #    for t2 in range(temp_a.shape[1]):
        #        if temp_a[t1,t2] !=0:
        #            temp_a[t1,t2]=temp_a[t1,t2]+255
        #temp_a = scaler.fit_transform(temp_a)
        #temp_a = scaler.fit_transform(temp_a)
        X_train[im,i,:,:] = temp_a[:,:]
    image_y = Image.fromarray(y_train_temp[im,:,:].astype("uint8"),"L")
    for i in range(no_slices):
        temp_y = image_y.crop((i*slice_x,0,(i+1)*slice_x,im_h))
        temp_b = np.asarray(temp_y)
        #temp_b = scaler.fit_transform(temp_b)
        y_train[im,i,:,:] = temp_b[:,:]
#for im in range(train_set_x_orig.shape[0]):
#    for i in range(no_slices):
#        half_slice = slice_x//2
#        for h in range(im_h):
#            for k in range(0,half_slice):
#                if y_train[im,i,h,k] != 0:
#                    X_train[im,i,h,k] = 0

y_train  = y_train.reshape((train_set_y_orig.shape[0],no_slices,im_h*slice_x))
###Converting the testing data to required format
for im in range(test_set_x_orig.shape[0]):
    image_x = (Image.fromarray(X_test_temp[im,:,:].astype("uint8"))).convert("L")
    for i in range(no_slices):
        temp_x = image_x.crop((i*slice_x,0,(i+1)*slice_x,im_h))
        temp_a = np.array(temp_x)
        #for t1 in range(temp_a.shape[0]):
        #    for t2 in range(temp_a.shape[1]):
        #        if temp_a[t1,t2] !=0:
        #            temp_a[t1,t2]=temp_a[t1,t2]+255
        #temp_a = scaler.fit_transform(temp_a)
        #temp_a = scaler.fit_transform(temp_a)
        X_test[im,i,:,:] = temp_a[:,:]
    image_y = Image.fromarray(y_test_temp[im,:,:].astype("uint8"),"L")
    for i in range(no_slices):
        temp_y = image_y.crop((i*slice_x,0,(i+1)*slice_x,im_h))
        temp_b = np.asarray(temp_y)
        #temp_b = scaler.fit_transform(temp_b)
        y_test[im,i,:,:] = temp_b[:,:]
#for im in range(test_set_x_orig.shape[0]):
#    for i in range(no_slices):
#        half_slice = slice_x//2
#        for h in range(im_h):
#            for k in range(0,half_slice):
#                if y_test[im,i,h,k] != 0:
#                    X_test[im,i,h,k] = 0
y_test  = y_test.reshape((test_set_y_orig.shape[0],no_slices,im_h*slice_x))
#Reshaping them to the required form
train_set_x = X_train.reshape(X_train.shape+(1,))/255
print("X train shape: {}".format(train_set_x.shape))
train_set_y = y_train/100
print("y train shape: {}".format(train_set_y.shape))
test_set_x  = X_test.reshape(X_test.shape+(1,))/255
print("X test shape: {}".format(test_set_x.shape))
test_set_y  = y_test/100
print("y test shape: {}".format(test_set_y.shape))
input_shape1 = (None,train_set_x.shape[2], train_set_x.shape[3],train_set_x.shape[4])

del X_train
del y_train
del X_test
del y_test
start_time = time.time()
##############Building Conv LSTM model ######################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ConvLSTM2D, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Bidirectional, LSTM
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, concatenate, Conv2DTranspose, UpSampling2D, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import tensorflow as tf

#Contraction 1
model = Sequential()

model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation ="relu", padding = "same", input_shape=input_shape1)))
#model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation ="relu", padding = "same", input_shape=input_shape1)))
#model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation ="relu", padding = "same", input_shape=input_shape1)))
#model.add(TimeDistributed(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), activation ="relu", padding = "same")))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dropout(0.6)))
model.add(TimeDistributed(Dense(im_h*slice_x, activation ="relu")))
model.add(TimeDistributed(Dropout(0.6)))
model.add(Bidirectional(SimpleRNN(im_h*slice_x, activation = "relu", return_sequences=True, recurrent_dropout=0.2, dropout=0.6)))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(im_h*slice_x, activation ="relu")))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(im_h*slice_x, activation ="sigmoid")))

model.compile(loss = "binary_crossentropy", optimizer = "adam")

#define condition for early stoping
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", verbose =1, patience = 1000)
#print(model.summary())

#fit the model
model.fit(x=train_set_x, y=train_set_y, batch_size=32, epochs=2000, validation_data=(test_set_x, test_set_y), callbacks = [early_stop])
#model.fit(x=train_set_x, y=train_set_y, batch_size=32, epochs=1000, validation_data=(test_set_x, test_set_y))

##################save the model################
##Importing libraries
##from joblib import dump
model.save("ConvLSTM.h5")


#loading losses 
losses = pd.DataFrame(model.history.history)
with open("losses_with_norm_128.txt","w+") as df:
    losses.to_csv(df, sep=" ", float_format = "%2.4f")

end_time = time.time()
print("Total time taken: {}".format(end_time-start_time))