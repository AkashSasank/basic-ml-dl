import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

data = pd.read_csv(filepath_or_buffer='./housing.csv', delimiter=',')
x = data['median_income'][0:1000]
y = data['median_house_value'][0:1000]
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
plt.figure()
plt.scatter(x, y)

regr = linear_model.LinearRegression()
regr.fit(x, y)
coeff = regr.coef_
inter = regr.intercept_
print(coeff)
print(inter)

pred = regr.predict(x)
plt.scatter(x, pred)
plt.show()
