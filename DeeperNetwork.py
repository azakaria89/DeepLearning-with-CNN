# -*- coding: utf-8 -*-
"""
Created on Wed May 03 12:43:22 2017

@author: AZakaria
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten  #Core layers
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D #CNN layers
from keras.utils import np_utils #for data transformation
from keras.datasets import mnist #loading the dataset
import matplotlib.pyplot as plt
from keras import backend as K
import keras
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print K.image_data_format() #channles last
#plt.imshow(X_train[1])
#=======================Preprocessing Steps============================================
# When using the Theano backend, you must explicitly declare a dimension for the depth of the input image.
# For example, a full-color image with all 3 RGB channels will have a depth of 3.
# 
# Our MNIST images only have a depth of 1, but we must explicitly declare that.
#==============================================================================
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

#==============================================================================
# The final preprocessing step for the input data is to convert our data type to float32 
#and normalize our data values to the range [0, 1].
#==============================================================================

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


#==============================================================================
# Convert 1-dimensional class arrays to 10-dimensional class matrices
# This is done in order to have 1 vector of class probabilities(10 classes) for each sample out of the 60,000 training examples
#==============================================================================
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#==========Defining model architecture Steps============================
model= Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3),strides=1,activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3),strides=1, padding='valid',activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, kernel_size=(3, 3),strides=1,activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3),strides=1, padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
#128 output neurons with activation of relu
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))




#The padding argument specifies one of two enumerated values (case-insensitive): valid (default value) or same.
#To specify that the output tensor should have the same width and height values as the input tensor, we set padding=same here,
# which instructs TensorFlow to add 0 values to the edges of the output tensor to preserve width and height

#==============================================================================
# Compile the architecture by declaring the optimization methond and the loss function to be used (objectvie function)
# configure its learning process 
#==============================================================================
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
 
history=model.fit(X_train, Y_train,
          batch_size=128,
          epochs=8,
          verbose=2,
          validation_data=(X_test, Y_test))

print model.summary()
#==============================================================================
# Confusion Matrix for the validation data
#==============================================================================
y_predict=model.predict_classes(X_test)


classes_names=np.array([0,1,2,3,4,5,6,7,8,9])
conf_mat=confusion_matrix(np.argmax(Y_test,axis=1), y_predict)
plt.figure()
plot_confusion_matrix(conf_mat, classes=classes_names,
                      title='Confusion matrix')


#=========================== Commented out Plotting code===================================================
# 
# print(history.history.keys())
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy during validation')
# plt.ylabel('val_accuracy')
# plt.xlabel('epoch number')
# plt.show()
# plt.plot(history.history['val_loss'])
# plt.title('model loss during validation')
# plt.ylabel('val_loss')
# plt.xlabel('epoch number')
# plt.show()
# 
# 
# 
# 
# 
# val_acc_deep=np.array([ 0.9846,0.9878, 0.9904 ,0.9918,0.9916 ,0.9921,0.9928,0.9928])
# epoch=np.array([1,2,3,4,5,6,7,8])
# val_acc_LeNet=np.array([ 0.9754,0.9822, 0.9860 ,0.9865,0.9880 ,0.9890,0.9901,0.9888])
# 
# plt.xticks(np.arange(min(epoch), max(epoch)+1))
# plt.plot(epoch,val_acc_deep)
# plt.plot(epoch,val_acc_LeNet)
# plt.ylabel('Val_accuracy')
# plt.xlabel('epoch')
# plt.legend(['Deep', 'LeNet'], loc='upper left')
# plt.show()
# 
# 
# 
# 
# val_loss_deep=np.array([ 0.0443,0.0336, 0.0273 ,0.0231,0.0273 ,0.0219,0.0236,0.0252])
# epoch=np.array([1,2,3,4,5,6,7,8])
# val_loss_LeNet=np.array([ 0.0763,0.0550, 0.0452 ,0.0391,0.0384 ,0.0336,0.0319,0.0332])
# 
# plt.xticks(np.arange(min(epoch), max(epoch)+1))
# plt.plot(epoch,val_loss_deep)
# plt.plot(epoch,val_loss_LeNet)
# plt.ylabel('Val_Loss')
# plt.xlabel('epoch')
# plt.legend(['Deep', 'LeNet'], loc='upper left')
# plt.show()
# 
# 
# acc_deep=np.array([ 0.9395,0.9794, 0.9842 ,0.9871,0.9889 ,0.9898,0.9916,0.9914])
# epoch=np.array([1,2,3,4,5,6,7,8])
# acc_LeNet=np.array([ 0.8995,0.9659, 0.9740 ,0.9783,0.9813 ,0.9826,0.9847,0.9859])
# 
# plt.xticks(np.arange(min(epoch), max(epoch)+1))
# plt.plot(epoch,acc_deep)
# plt.plot(epoch,acc_LeNet)
# plt.ylabel('train_Acc')
# plt.xlabel('epoch')
# plt.legend(['Deep', 'LeNet'], loc='upper left')
# plt.show()
# 
# loss_deep=np.array([ 0.1993,0.0671, 0.0536 ,0.0432,0.0364 ,0.0332,0.0282,0.0269])
# epoch=np.array([1,2,3,4,5,6,7,8])
# loss_LeNet=np.array([ 0.3310,0.1130, 0.0874 ,0.0734,0.0632,0.0580,0.0515,0.0467])
# 
# plt.xticks(np.arange(min(epoch), max(epoch)+1))
# plt.plot(epoch,loss_deep)
# plt.plot(epoch,loss_LeNet)
# plt.ylabel('train_Loss')
# plt.xlabel('epoch')
# plt.legend(['Deep', 'LeNet'], loc='upper left')
# plt.show()
# val_loss_deep=np.array([ 0.0443,0.0336, 0.0273 ,0.0231,0.0273 ,0.0219,0.0236,0.0252])
# epoch=np.array([1,2,3,4,5,6,7,8])
# loss_deep=np.array([0.1993,0.0671, 0.0536 ,0.0432,0.0364 ,0.0332,0.0282,0.0269])
# 
# plt.xticks(np.arange(min(epoch), max(epoch)+1))
# plt.plot(epoch,val_loss_deep)
# plt.plot(epoch,loss_deep)
# plt.ylabel('validLoss vs trainLoss')
# plt.xlabel('epoch')
# plt.legend(['ValidationLossDeep', 'TrainLossDeep'], loc='upper left')
# plt.show()
#==============================================================================
