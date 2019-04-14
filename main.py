print("Task 2")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#load data - skip first row 
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)-1

#Columns 1-6 are features
#We delete X0 here
#X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X2=the house age (unit: year) 
#X3=the distance to the nearest MRT station (unit: meter) 
#X4=the number of convenience stores in the living circle on foot (integer) 
#X5=the geographic coordinate, latitude. (unit: degree) 
#X6=the geographic coordinate, longitude. (unit: degree)

trainingData = reData[:,1:numCols-1]

#Last column are the labels
labels = reData[:,[numCols-1]]

"""
OLS Derivation
# Ax = b 
# A'Ax = A'b
# x = inverse(A'A)*A'b
# Here x is a feature vector, A is trainingData, and b is labels.
"""
def ordinaryLeastSquares(trainingData, labels):

    XTX = np.dot(trainingData.T,trainingData)
    XTY = np.dot(trainingData.T, labels)
    parameters = np.dot(np.linalg.inv(XTX),XTY)
    return parameters

"""
Calculate the hypothesis = X * parameters
Calculate the cost = (h - y)^2 
Calculate the gradient = sum(X' * loss )/ m  #this is the part that makes it batch
Update the parameters parameters = parameters - alpha * gradient of cost 
Check if cost is less than epsilon
"""
def bgd(trainingData, labels, alpha, epsilon, epochs):
    k = 0
    col = np.size(trainingData,1)
    row = np.size(trainingData,0)
    parameters = np.array([[0 for x in range(col)]], ndmin=2).T
    Cost = epsilon + 1
    while k < epochs and Cost > epsilon:
        Hypo = np.dot(trainingData, parameters)
        Diff = Hypo - labels
        Cost = (1/2*numRows) * np.sum(np.square(Diff) ) 
        parameters = parameters - alpha * (1.0/row) * np.dot(np.transpose(trainingData), Diff)
        k += 1
    return parameters
    
    
"""
Report the affects of trying a variety of learning rates and number of
epochs. 
Plot the cost of the bgd function after each epoch for
a variety of number of epochs and learning rate. 
"""
test = bgd(trainingData, labels, .00000001, .000001, 100)
print(test)
parameters = ordinaryLeastSquares(trainingData, labels)
print('test2')
Hypo = np.dot(trainingData, parameters)
sse = np.sum(np.square(np.subtract(labels,Hypo)))
mse = np.mean(np.square(np.subtract(labels,Hypo)))
print('OLS SSE and MME:' + str(sse) + str(', ') + str(mse) + str(', '))
 

#Here is where a variety of alpha, epochs are tested

for i in range(5):
 alpha = 10**(-i-6)
 CostHistory = []
 EpochHistory = []
 for w in range(5):
  epochs =10**(w)
  parameters = bgd(trainingData, labels, alpha, .0000001, epochs)
  Hypo = np.dot(trainingData, parameters)
  sse = np.sum(np.square(np.subtract(labels,Hypo)))
  mse = np.mean(np.square(np.subtract(labels,Hypo)))
  print('DataSet 1 SSE and MME: alpha and epochs ' + str(sse) + str(', ') + str(mse) + str(', ') + str(alpha) + str(', ') + str(epochs))
  Diff = Hypo - labels
  Cost = (1/2*numRows) * np.sum(np.square(Diff) )
  CostHistory.append(Cost)
  EpochHistory.append(epochs)
 fig = plt.figure()
 plt.plot(EpochHistory, CostHistory, color = 'r')
 fig.suptitle("alpha = " + str(alpha))
 plt.xlabel("Epoch #")
 plt.ylabel("Cost")
 plt.show()

 
 
"""
5. Repeat tasks 1.3, 2.3 and 2.4 for another dataset of your choosing. You
may use any dataset you wish from any public repository (UCI, Kaggle,
etc.). Give a brief description of the dataset (features, labels).

"""
#this is using the new dataset
reData = np.loadtxt('Admission_Predict.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)-1


trainingData = reData[:,1:numCols-1]
labels = reData[:,[numCols-1]]

newtest = bgd(trainingData, labels, .00001, .0000001, 100000)


#Here is where a variety of alpha, epochs are tested on the 2nd dataset

for i in range(5):
 alpha = 10**(-i-6)
 CostHistory = []
 EpochHistory = []
 for w in range(5):
  epochs =10**(w)
  parameters = bgd(trainingData, labels, alpha, .0000001, epochs)
  Hypo = np.dot(trainingData, parameters)
  sse = np.sum(np.square(np.subtract(labels,Hypo)))
  mse = np.mean(np.square(np.subtract(labels,Hypo)))
  print('Dataset 2 SSE and MME: alpha and epochs ' + str(sse) + str(', ') + str(mse) + str(', ') + str(alpha) + str(', ') + str(epochs))
  Diff = Hypo - labels
  Cost = (1/2*numRows) * np.sum(np.square(Diff) )
  CostHistory.append(Cost)
  EpochHistory.append(epochs)
 fig = plt.figure()
 plt.plot(EpochHistory, CostHistory, color = 'r')
 fig.suptitle("alpha = " + str(alpha))
 plt.xlabel("Epoch #")
 plt.ylabel("Cost")
 plt.show()

x1 = trainingData[:,[1]]
reg = LinearRegression().fit(x1 , labels)
predictedY = reg.predict(x1)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 1:" + str(reg._residues))
plt.figure()
plt.scatter(x1, labels, color = 'g')
plt.plot(x1, predictedY, color = 'r')


x2 = trainingData[:,[2]]
reg = LinearRegression().fit(x2 , labels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 2:" + str(reg._residues))
plt.figure()
plt.scatter(x2, labels, color = 'g')
plt.plot(x2, predictedY, color = 'r')


x3 = trainingData[:,[3]]
reg = LinearRegression().fit(x3 , labels)
predictedY = reg.predict(x3)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3:" + str(reg._residues))
plt.figure()
plt.scatter(x3, labels, color = 'g')
plt.plot(x3, predictedY, color = 'r')


x4 = trainingData[:,[4]]
reg = LinearRegression().fit(x4 , labels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 4:" + str(reg._residues))
plt.figure()
plt.scatter(x4, labels, color = 'g')
plt.plot(x4, predictedY, color = 'r')


x5 = trainingData[:,[5]]
reg = LinearRegression().fit(x5 , labels)
predictedY = reg.predict(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5:" + str(reg._residues))
plt.figure()
plt.scatter(x5, labels, color = 'g')
plt.plot(x5, predictedY, color = 'r')








newtest2 = ordinaryLeastSquares(trainingData, labels)

Hypo = np.dot(trainingData, newtest2)
sse = np.sum(np.square(np.subtract(labels,Hypo)))
mse = np.mean(np.square(np.subtract(labels,Hypo)))
print('OLS2 SSE and MME: alpha and epochs ' + str(sse) + str(', ') + str(mse) + str(', ') + str(alpha) + str(', ') + str(epochs))


reg = LinearRegression()

# Find the best fit linear regression model
reg = reg.fit(trainingData, labels)

# Predict new values based on some given samples.
yPredicted  = reg.predict(trainingData)

# The sum of square error: (yPredicted - labels)^2
# This should be the same as reg.residues_
sse = np.sum(np.square(np.subtract(labels,yPredicted)))

print('All features coefficients: ' + str(reg.coef_))
print('Intercept: ' + str(reg.intercept_))
print('reg.residue all features: '+ str(reg._residues))
print('SSE all features: '+ str(sse))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(labels,yPredicted)))))



"""


#References

#https://the-tarzan.com/2012/10/27/calculate-ols-regression-manually-in-python-using-numpy/
#https://machinelearningmastery.com/gradient-descent-for-machine-learning/
#https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1
#https://www.kaggle.com/mohansacharya/graduate-admissions/version/2    ---Dataset
"""
