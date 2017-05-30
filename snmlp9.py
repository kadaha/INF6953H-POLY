'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense , Activation
from keras.models import Model
from keras.optimizers import RMSprop


batch_size = 10000
num_classes = 10
epochs = 1

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




inputs = Input( shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(256, activation='relu')(inputs)
y = Dense(256, activation='relu')(x)
zz= keras.layers.concatenate([x,y])
z = Dense(256, activation='relu')(zz)

a = Dense(128, activation='relu')(z)
b = Dense(128, activation='relu')(a)
cc= keras.layers.concatenate([a,b])
c = Dense(128, activation='relu')(cc)

d=  Dense(64, activation='relu')(c)
e = Dense(64, activation='relu')(d)
ff= keras.layers.concatenate([c,d])
f = Dense(64, activation='relu')(ff)
#z= keras.layers.concatenate([x,y])
predictions = Dense(10, activation='softmax')(f)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.summary()



model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plot_model(model, to_file='modelskipnmlp9.png')
