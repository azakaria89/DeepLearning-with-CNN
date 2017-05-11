# A Tutorial on DeepLearning-with-CNN
Code examples on how to train a convolutional neural network on the MNIST dataset Theano over Keras.
## Getting Started Theano over Keras Installation tips on Windows 64bit
* Install TDM GCC x64.
* Install Anaconda x64.
* Open the Anaconda prompt
* Run conda update conda
* Run conda update --all
* Run conda install mingw libpython
* conda install -c msys2 m2w64-toolchain=5.3.0
* pip install git+git://github.com/Theano/Theano.git
* pip install git+git://github.com/fchollet/keras.git
* Change the backend config to Theano as the default is tensorflow
* note Tensorflow doesnt work with Python 2.7 on Windows.

## Useful Plots from the simple CNN architecture
![capture](https://cloud.githubusercontent.com/assets/6147747/25950545/9ace132c-365b-11e7-92cb-0cb89cc1e091.JPG)
![capture1](https://cloud.githubusercontent.com/assets/6147747/25950710/1162d702-365c-11e7-9242-5c52f54f65ff.JPG)

Apparently theano at the backend during training the model utilizes the multithreading in calculating the gradient of loss function by loading the whole batch in memory and act upon it.64 samples was The best batchsize w.r.t the accuracy of prediction while training the network in the SimpleCNN source file.
## Useful Plots from the Deeper CNN architecture
![capture2](https://cloud.githubusercontent.com/assets/6147747/25951300/829bf826-365d-11e7-83bf-24b207fa84b8.JPG)
![capture3](https://cloud.githubusercontent.com/assets/6147747/25951355/afab20f8-365d-11e7-8b15-e03f13ecae7b.JPG)

 The LeNet is basically the simple network and it is obvious that training a deeper NW leads to better accuracy over both the train and the validation sets


## The deeper CNN confusion matrix
![capture4](https://cloud.githubusercontent.com/assets/6147747/25951491/3363bbf8-365e-11e7-854a-69546a654cc2.JPG)

You can check my answer on what a confusion matrix means here:
* [Quora-Ahmed M. Zakaria ](https://www.quora.com/Classification-machine-learning-What-is-confusion-matrix/answer/Ahmed-M-Zakaria?timeline=1) -What is meant by a confusion Matrix?
## Calculating the learnable parameters and Feature maps [in progress]
* The source code visualize the kernels of the first layer using Bokeh visualization library, you will find the visualization layer among the source code as html. In order to give you a quick glimpse of what the html page would look like i attached the below screen shot. This is a bokeh tab layout where each 4 filters out of the 32 filters are located in one out of 8 panels

![capture](https://cloud.githubusercontent.com/assets/6147747/25951953/998ccfc2-365f-11e7-9452-b9493f6d91a4.JPG)

* I found calling the summary function from keras very useful in order to show important information on the model 

```
model.summary()
```
