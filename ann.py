import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from sklearn.preprocessing import PolynomialFeatures, scale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('./bmi.csv', delimiter=',')
x1 = data['Height']
x2 = scale(data['Weight'])
y = data['BMI']
X = np.transpose(np.array([x1, x2]))

Y = np.expand_dims(y, axis=1)
print(X.shape)
print(Y.shape)

x_in = Input((2,))
x = Dense(100, activation='relu')(x_in)
x = Dense(100, activation='relu')(x)
x = Dense(1, activation='relu')(x)

model = Model(inputs=x_in, outputs=x)

model.summary()
tb = TensorBoard()

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model.fit(X, Y, batch_size=None, epochs=1000, verbose=2, callbacks=[tb])
model.save('./model.json')
model.save_weights('./weights.h5')

pred = model.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], pred, cmap='hsv', c=X[:, 0])

plt.show()
