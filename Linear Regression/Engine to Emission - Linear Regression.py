import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
%matplotlib inline

data = pd.read_csv("FuelConsumption.csv")
data.head() #look at the data set you read.

data.describe() #summarizes the data

subdata= data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
subdata.head(9) #looks at a subsection of the data made

#visualize = subdata[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#visualize.hist()
#plt.show()

#plt.scatter(subdata.FUELCONSUMPTION_COMB, subdata.CO2EMISSIONS, color='blue')
#plt.xlabel('FUELCONSUMPTION_COMB')
#plt.ylabel('Emission')
#plt.show()

plt.scatter(subdata.ENGINESIZE, subdata.CO2EMISSIONS, color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

#separate the data into training data and testing data
mask = np.random.rand(len(data)) < 0.8
train = subdata[mask]
test = subdata[~mask]

#model the data with sklearn
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
#print the coefficients and intercepts (theta values)
print('Coefficients: ', regr.coef_)
print('intercept: ', regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_= regr.predict(test_x)

print('Mean absolute error: %.2f' %np.mean(np.absolute(test_y_ - test_y)))
print('Residual sum of squares (MSE): %.2f' %np.mean((test_y_ - test_y)**2))
print('R2-score: %.2f' %r2_score(test_y_ , test_y))


