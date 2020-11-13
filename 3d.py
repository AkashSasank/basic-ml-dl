import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv(filepath_or_buffer='./housing.csv', delimiter=',')
x = data['median_income'][0:1000]
y = data['median_house_value'][0:1000]
z = data['population'][0:1000]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=x, cmap='hsv')

plt.show()
