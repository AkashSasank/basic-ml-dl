import tensorflow as tf
from keras.layers import Layer,Input,Conv2D,Flatten,Dense,MaxPool2D
from keras import models

import numpy as np
import random
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

model = models.Sequential()
model.add(Layer(input_shape=(28,28,1)))
model.add(Conv2D(8,(3,3),activation='relu'))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(Flatten(data_format='channels_last'))
model.add(Dense(50,activation = 'relu',input_shape = (28*28,)))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None,None))
model.summary()

#
#
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = np.expand_dims(x_train,axis=-1).astype('float32')/255
x_test = np.expand_dims(x_test,axis=-1).astype('float32')/255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


tb = tf.keras.callbacks.TensorBoard(log_dir='./log/mnist0',write_images=True)

model.fit(x_train,y_train,epochs=100,callbacks=[tb],batch_size=128)
model.save('mnist.json')
model.save_weights('./mnist.hd5')

