import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt', header=None)
x_values = dataframe[[0]]
y_values = dataframe[[1]]

#train model on data
data_reg = linear_model.LinearRegression()
data_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, data_reg.predict(x_values))
plt.show()
