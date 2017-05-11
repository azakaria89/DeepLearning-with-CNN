# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:47:28 2017

@author: AZakaria
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  #Core layers
from keras.layers import Conv2D, MaxPooling2D #CNN layers
from keras.utils import np_utils #for data transformation
from keras.datasets import mnist #loading the dataset
import matplotlib.pyplot as plt
from keras import backend as K
import keras
from weight32_visualize import weights_32_visualize





(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print K.image_data_format() #channles last
#print X_train.shape
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

#print Y_train[1]

#==========Defining model architecture Steps=======================================
model= Sequential() #linear stack of layers.
#conv layer with 32 filters, each of size 3*3, relu will then be applied to each feature map, the input to the conv layer is image
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#print model.output_shape
model.add(Dropout(0.25))
#Before desigining the FC layer the output of the previous conv layer/input of the FC layer should be flattened 12*12*64=9216
model.add(Flatten())
#128 output neurons with activation of relu
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



#W=model.layers[0].get_weights()
##B= np.array(W[0])
##print B.shape
#W=np.squeeze(W[0])
#Weights_reshaped= W.reshape(32,3,3)
#weights_32_visualize(Weights_reshaped)
#plt.title('conv1 weights')
#Weights_reshaped= W.reshape(32,3,3)
#plt.imshow(Weights_reshaped[0],cmap=plt.cm.binary)



W=model.layers[0].get_weights()
B= np.array(W[0])
print B.shape
W=np.squeeze(W[0])
Weights_reshaped= W.reshape(32,3,3)
weights_32_visualize(Weights_reshaped)









#==============================================================================
# Compile the architecture by declaring the optimization methond and the loss function to be used (objectvie function)
# configure its learning process 
#==============================================================================
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


#===============visualizing the val accuracy with diff batch sizes==========================================================
# val_acc=np.array([ 0.9756,0.9789, 0.9800 ,0.9801,0.9789 ,0.9767])
# batch_sizes=np.array([1,8,16,32,64,128])
# plt.xticks(np.arange(min(batch_sizes), max(batch_sizes)+1, 7.0))
# plt.plot(batch_sizes,val_acc)
# plt.title('validation accuracy vs Batch Size (include batch of 1 sample)')
# plt.ylabel('val_accuracy')
# plt.xlabel('Batch Size- number of epochs set to 1')
# plt.show()
# 
# 
#==============================================================================
#You can now iterate on your training data in batches:
history=model.fit(X_train, Y_train,
          batch_size=128,
          epochs=2,
          verbose=2,
          validation_data=(X_test, Y_test))
#=====================Task 4=================================================
# Showing the keys "metrics" collected during fitting the model
#plotting accuracy of the model on the test data vs the epoch number 
#==============================================================================
#==============================================================================
print(history.history.keys())
plt.plot(history.history['val_acc'])
plt.title('model accuracy during validation')
plt.ylabel('val_accuracy')
plt.xlabel('epoch number')
plt.show()

plt.plot(history.history['val_loss'])
plt.title('model loss during validation')
plt.ylabel('val_loss')
plt.xlabel('epoch number')
plt.show()




#evaluate our model on the test data:
score = model.evaluate(X_test, Y_test, verbose=0)   
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#===========Task 6 visualize filters=================================================================
# first layer has 32 kernels each of size 3*3
#==============================================================================
B= np.array(W[0])
print B.shape
W=np.squeeze(W[0])
Weights_reshaped= W.reshape(32,3,3)
weights_32_visualize(Weights_reshaped)


