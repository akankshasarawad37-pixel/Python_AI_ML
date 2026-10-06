#pip install pandas numpy matplotlib 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

csv_path = r"/home/sbs-lab13a-07/Akankshafiles/Salary_dataset.csv" 

salary_csv = pd.read_csv(csv_path)
columnnames = salary_csv.columns.tolist()

#print(columnnames)
print(columnnames)
print(np.shape(salary_csv[columnnames[1]]))
print(np.shape(np.array(salary_csv[columnnames[1]]).reshape(-1, 1)))


# #Initialize 
x = np.array(salary_csv[columnnames[1]]).reshape(-1, 1)
y = np.array(salary_csv[columnnames[2]]).reshape(-1, 1)

#plot the data
# plt.scatter(x, y, color='blue') 
# plt.plot(x, y, color='blue', label='Data points')

# #creating the model
#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# plt.plot(x_test, y_pred, color='red', label='Data points')
# plt.show()

# # creating the nonlinear regression model
poly = PolynomialFeatures(degree=5)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# # #Creating the model
model_poly = LinearRegression()
model_poly.fit(x_train_poly, y_train)
y_pred_poly = model_poly.predict(x_test_poly)
