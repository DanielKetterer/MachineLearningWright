import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
## Written by Jacob Ross for CS4840 - Spring 2019
## This is an exmaple of doing some basic things with Sci-kit learn.
## This code is not meant to be a shining example of good programming practices, rather
## is structured to demonstrate basic usage in the classroom.

## This code can be extended to do other forms of regression, combine features, etc.

## I demonstrate an improved linear regression model in terms of SSE by incorporating
## basic feature transforms. 


# Demonstrate:
#  - loading data from a file (CSV) (numpy)
#  - Manipulating, preprocessing data (numpy and sklearn)
#  - Linear Regression Assumption checking
#  - Fitting a model to loaded data (sklearn.linear_model)
#  - Showing fitted model attributes (SSE, coefficients, etc)
#  - Transforming features
#  - Showing model performance after 
 

#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)

#Columns 0-6 are features
#X0= id number
#X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X2=the house age (unit: year) 
#X3=the distance to the nearest MRT station (unit: meter) 
#X4=the number of convenience stores in the living circle on foot (integer) 
#X5=the geographic coordinate, latitude. (unit: degree) 
#X6=the geographic coordinate, longitude. (unit: degree)

xFeatures = reData[:,0:numCols-1]

#Last column are the labels
yLabels = reData[:,7]

# #ID number
# plt.figure()
# plt.title('ID Number vs. Y')
# plt.ylabel('Y label')
# plt.xlabel('ID Number')
# plt.scatter(xFeatures[:,0],yLabels)

# # transaction date
# plt.figure()
# plt.title('Transaction Date vs. Y')
# plt.ylabel('Y label')
# plt.xlabel('Transaction Date')
# plt.scatter(xFeatures[:,1],yLabels)

# # house age
# plt.figure()
# plt.title('House Age vs. Y')
# plt.ylabel('Y label')
# plt.xlabel('House Age (years)')
# plt.scatter(xFeatures[:,2],yLabels)

# distance to nearest MRT station
plt.figure()
plt.title('Dist to MRT vs. Y')
plt.ylabel('Y label')
plt.xlabel('Dist to MRT (meters)')
plt.scatter(xFeatures[:,3],yLabels)

# number of convenience stores
plt.figure()
plt.title('Number of convenience stores vs. Y')
plt.ylabel('Y label')
plt.xlabel('Num Stores (int)')
plt.scatter(xFeatures[:,4],yLabels)


# lat coord
plt.figure()
plt.title('Lat Coord vs. Y')
plt.ylabel('Y label')
plt.xlabel('Num Stores (int)')
plt.scatter(xFeatures[:,5],yLabels)

#long coord
plt.figure()
plt.title('Long coord vs. Y')
plt.ylabel('Y label')
plt.xlabel('Long Coord (double)')
plt.scatter(xFeatures[:,6], yLabels)

plt.show()






reg = LinearRegression()

# Find the best fit linear regression model
reg = reg.fit(xFeatures, yLabels)

# Predict new values based on some given samples.
# In this case, this fits the instances I used to create the model
# so I can do residual analysis
yPredicted  = reg.predict(xFeatures)

# The sum of square error: (yPredicted - yLabels)^2
# This should be the same as reg.residues_
sse = np.sum(np.square(np.subtract(yLabels,yPredicted)))

print('All features coefficients: ' + str(reg.coef_))
print('Intercept: ' + str(reg.intercept_))
print('reg.residue all features: '+ str(reg._residues))
print('SSE all features: '+ str(sse))
print('MSE all feautre: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))

# Univariate Linear Regression with each feature individually
# The double bracket notation because it expects a "PANDAS" data frame as input



#X0 = id number
reg = LinearRegression().fit(xFeatures[:,[0]], yLabels)
predictedY = reg.predict(xFeatures[:,[0]])

print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 0:" + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,0], yLabels, color = 'g')
#plt.plot(xFeatures[:,0], predictedY, color = 'r')


#X1 = the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
reg = LinearRegression().fit(xFeatures[:,[1]], yLabels)
predictedY = reg.predict(xFeatures[:,[1]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 1: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,1], yLabels, color = 'g')
#plt.plot(xFeatures[:,1], predictedY, color = 'r')

#X2 = the house age (unit: year) 
reg = LinearRegression().fit(xFeatures[:,[2]], yLabels)
predictedY = reg.predict(xFeatures[:,[2]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 2: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,2], yLabels, color = 'g')
#plt.plot(xFeatures[:,2], predictedY, color = 'r')

#X3 = the distance to the nearest MRT station (unit: meter) 
reg = LinearRegression().fit(xFeatures[:,[3]], yLabels)
predictedY = reg.predict(xFeatures[:,[3]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,3], yLabels, color = 'g')
#plt.plot(xFeatures[:,3], predictedY, color = 'r')

#X4 = the number of convenience stores in the living circle on foot (integer) 
reg = LinearRegression().fit(xFeatures[:,[4]], yLabels)
predictedY = reg.predict(xFeatures[:,[4]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 4: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,4], yLabels, color = 'g')
#plt.plot(xFeatures[:,4], predictedY, color = 'r')

#X5=the geographic coordinate, latitude. (unit: degree) 
reg = LinearRegression().fit(xFeatures[:,[5]], yLabels)
predictedY = reg.predict(xFeatures[:,[5]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,5], yLabels, color = 'g')
#plt.plot(xFeatures[:,5], predictedY, color = 'r')

#X6=the geographic coordinate, longitude. (unit: degree)
reg = LinearRegression().fit(xFeatures[:,[6]], yLabels)
predictedY = reg.predict(xFeatures[:,[6]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,6], yLabels, color = 'g')
#plt.plot(xFeatures[:,6], predictedY, color = 'r')

#Check correlation coefficients
r = np.corrcoef(xFeatures, rowvar = False)
plt.imshow(r, cmap ='hot', interpolation='nearest')
plt.colorbar()
plt.show()



#############################################################################################################
print("iteration 1")

# cut the "bad" variables and see what happens
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]])       )

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))





# Playing with some feature transforms:
#X1 = the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
# take the mod 1 of each date to hack off the year
x1 = xFeatures[:,[1]]
x1 = x1 % 1
x1 = np.sqrt(x1)

reg = LinearRegression().fit(x1 , yLabels)
predictedY = reg.predict(x1)


# take the log of feature 1
x1 = xFeatures[:,[1]]
x1 = np.log(x1)
reg = LinearRegression().fit(x1 , yLabels)
predictedY = reg.predict(x1)



#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))
# plt.figure()
# plt.scatter(x1, yLabels, color = 'g')
# plt.plot(x1, predictedY, color = 'r')

# log scale 3rd feature
x3 = xFeatures[:,[3]]
x3 = np.log10(x3)

reg = LinearRegression().fit(x3 , yLabels)
predictedY = reg.predict(x3)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))
plt.figure()
plt.scatter(x3, yLabels, color = 'g')
plt.plot(x3, predictedY, color = 'r')


# sqrt 3rd feature
x3 = xFeatures[:,[3]]
x3 = np.sqrt(x3)

reg = LinearRegression().fit(x3 , yLabels)
predictedY = reg.predict(x3)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (sqrt):" + str(reg._residues))
plt.figure()
plt.scatter(x3, yLabels, color = 'g')
plt.plot(x3, predictedY, color = 'r')




# log 5th feature
x5 = xFeatures[:,[5]]
x5 = np.log10(x5)

reg = LinearRegression().fit(x5 , yLabels)
predictedY = reg.predict(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5 (log):" + str(reg._residues))
plt.figure()
plt.scatter(x5, yLabels, color = 'g')
plt.plot(x5, predictedY, color = 'r')



# log 6th feature
x6 = xFeatures[:,[6]]
x6 = np.log10(x6)

reg = LinearRegression().fit(x6 , yLabels)
predictedY = reg.predict(x6)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (log):" + str(reg._residues))
plt.figure()
plt.scatter(x6, yLabels, color = 'g')
plt.plot(x6, predictedY, color = 'r')



# sqrt 6th feature
x6 = xFeatures[:,[6]]
x6 = np.sqrt(x6)

reg = LinearRegression().fit(x6 , yLabels)
predictedY = reg.predict(x6)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (sqrt):" + str(reg._residues))
plt.figure()
plt.scatter(x6, yLabels, color = 'g')
plt.plot(x6, predictedY, color = 'r')



# square 6th feature
x6 = xFeatures[:,[6]]
x6 = np.square(x6)

reg = LinearRegression().fit(x6 , yLabels)
predictedY = reg.predict(x6)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (square):" + str(reg._residues))
plt.figure()
plt.scatter(x6, yLabels, color = 'g')
plt.plot(x6, predictedY, color = 'r')


#Fit the model again with the updated features:
xFeatures[:,1] = np.log(xFeatures[:,1])
xFeatures[:,3] = np.log(xFeatures[:,3])
xFeatures[:,5] = np.log(xFeatures[:,5])
xFeatures[:,6] = np.log(xFeatures[:,6])

reg = LinearRegression().fit(xFeatures, yLabels)
yPredicted  = reg.predict(xFeatures)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 2")
print("square the lat and long xFeatures")
#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]]))

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X3=the sum of the latitudes and longitudes

# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)

reg = LinearRegression().fit(x0, yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale x2 
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# sqr 5th feature
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.exp2(x4)

reg = LinearRegression().fit(x4 , yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5 (square):" + str(reg._residues))


# sqr 6th feature
x5 = xFeaturesTrimmed[:,[5]]
x5 = np.exp2(x5)

reg = LinearRegression().fit(x5 , yLabels)
predictedY = reg.predict(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (square):" + str(reg._residues))

#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.log(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.square(xFeatures[:,4])
xFeaturesTrimmed[:,5] = np.square(xFeaturesTrimmed[:,5])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 3")
#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]+xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by adding")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, SUMMED together")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X3=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1, transaction date
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable 0 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature, distance to MRT
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 2 (log):" + str(reg._residues))


# # 4th feature (lat + long)
x4 = xFeaturesTrimmed[:,[4]]
reg = LinearRegression().fit(x3, yLabels)
predictedY = reg.predict(x3)

# #print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 4 (no transform):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0]) #xFeatures[:,1]
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2]) #xFeatures[:,3]


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 4")
#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by multiplying")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, combined together")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X3=the sum of the latitudes and longitudes


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable 0 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)
print("SSE for variable 2 (log):" + str(reg._residues))

# # 4th feature (lat * long)
x4 = xFeaturesTrimmed[:,[4]]
reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)


#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined multiplied variables lat & long (no transform):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels) #xFeatures
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 5")
print("multiplying the lat and long features results in a lower SSE than adding the features")
print("now we will square the combined multiplied lat and longs")


#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by multiplying")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, combined together")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 
reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# # square 5th feature
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.square(x4)
reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined multiplied variables lat & long  x4(square):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.square(xFeaturesTrimmed[:,4])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 6")
print("squaring the multiplied lat and long features increases SSE")
print("now we will take the log of the combined multiplied lat and longs")


#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by multiplying")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, combined together")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

x4 = xFeaturesTrimmed[:,[4]]
x4 = np.log(x4)
reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for x4 (log):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.log(xFeaturesTrimmed[:,4])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 7")
print("taking the log of the multiplied lat and long features also increases the SSE")
print("now we will cube the combined multiplied lat and longs")


#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by multiplying")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, combined together")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# # log 5th feature
# x5 = xFeatures[:,[5]]
# x5 = np.exp2(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5 (exp2):" + str(reg._residues))
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.power(x4, 3)
reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for x4 (cubed):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.power((xFeaturesTrimmed[:,4]), 3)

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 8")
print("cubing the multiplied lat and long features slightly decreases the SSE more than squaring")
print("now we will take the square root of the combined multiplied lat and longs")


#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by multiplying")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, combined together")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# sqrt 5th feature
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.sqrt(x4)
reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for x4(sqrt):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.sqrt(xFeaturesTrimmed[:,4])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 9")
print("taking the sq rt of the multiplied lat and long features has a similiar effect to the log")
print("the greatest reduction in SSE for the combined lat and long features is the polynomial (cube) transform of the features")



#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (multiplying features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)
print("combine lats and longs by multiplying")
# trimming: xFeatures[:,[5]],xFeatures[:,[6]]
print("trimmed x5 and x6, combined together")

xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0, yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))


x4 = xFeaturesTrimmed[:,[4]]
reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for x4 (no transform):" + str(reg._residues))

#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.power((xFeaturesTrimmed[:,2]), 3)


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 10")
print("the greatest reduction in SSE for the combined lat and long features is by performing a polynomial (cube) transform of the that feature")
print("we will manually remove the most the outlier from the graph (id no. 271), and use the original lat and log features")
print("we will not apply any transformation to x5 and x6")
#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]


print("reinserting: xFeatures[:,[5]],xFeatures[:,[6]]")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]])) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# no transformation of "outlier-less" 5th feature
x4 = xFeaturesTrimmed[:,[4]]

reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5 (no transform):" + str(reg._residues))

# no transformation of 6th feature
x5 = xFeaturesTrimmed[:,[5]]
reg = LinearRegression().fit(x5, yLabels)
predictedY = reg.predict(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (no transform):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,1] = np.log(xFeaturesTrimmed[:,2])


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 11")
print("removing an outlier decreased the SSE and the MSE by 2807 and 6.64 reprectively")
print("the dataset may not respond well to a regression model")
print("we will now square the lat and long features")

#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]


print("reinserting: xFeatures[:,[5]],xFeatures[:,[6]]")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]])) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# square 5th feature
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.square(x4)


reg = LinearRegression().fit(x4 , yLabels)
predictedY = reg.predict(x4)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5 (square):" + str(reg._residues))

# square 6th feature
x5 = xFeaturesTrimmed[:,[5]]
x5 = np.square(x5)


reg = LinearRegression().fit(x5 , yLabels)
predictedY = reg.predict(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (square):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.square(xFeaturesTrimmed[:,4])
xFeaturesTrimmed[:,5] = np.square(xFeaturesTrimmed[:,5])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 12")
print("squaring the modified lat and long data decreased the SSE by 2358 and MSE by 5.709")
print("we will now recombine the lat and long features by multiplication and without any transform")


#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)

print("trimming: xFeatures[:,[5]],xFeatures[:,[6]]")
print("inserting: combined x5 and x6 by multiplication")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))


# the product of x5 and x6 features
x4 = xFeaturesTrimmed[:,[4]]

reg = LinearRegression().fit(x4 , yLabels)
predictedY = reg.predict(x4)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for product of variables 5 & 6 (no transform):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))



#################################################################################
print("iteration 13")
print("multiplying the x5 and x6 features by decreased the variable SSE by an average of 5502.238")
print("retrying the polynomial transform using the square")


#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)

print("trimming: xFeatures[:,[5]],xFeatures[:,[6]]")
print("inserting: combined x5 and x6 by multiplication")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0, yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))

# the combined x5 and x6 features
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.square(x4)
reg = LinearRegression().fit(x4 , yLabels)
predictedY = reg.predict(x4)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined variables 5 & 6 (sqrt):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.square(xFeaturesTrimmed[:,4])
reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 14")
print("squaring the location feature did not lower the residuals below iteration 11")
print("we assume that a cubing the feature will be lower than squaring, but will not beat iteration 11")

#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)

print("trimming: xFeatures[:,[5]],xFeatures[:,[6]]")
print("inserting: combined x5 and x6 by multiplication")
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], location)) 

# New Column indexing 0-4 are features
#X0=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X1=the house age (unit: year) 
#X2=the distance to the nearest MRT station (unit: meter) 
#X3=number of convenience stores in foot travel
#X4=the sum of the latitudes and longitudes

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# take the mod 1 of each date to hack off the year
x0 = xFeaturesTrimmed[:,[0]]
x0 = x0 % 1
x0 = np.sqrt(x0)


# take the log of feature 1
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log(x0)
reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)


reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))

# log scale 3rd feature
x2 = xFeaturesTrimmed[:,[2]]
x2 = np.log10(x2)

reg = LinearRegression().fit(x2 , yLabels)
predictedY = reg.predict(x2)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log2 and sqrt):" + str(reg._residues))

# the combined x5 and x6 features
x4 = xFeaturesTrimmed[:,[4]]
x4 = np.power((x4), 3)
reg = LinearRegression().fit(x4 , yLabels)
predictedY = reg.predict(x4)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined variables 5 & 6 (sqrt):" + str(reg._residues))


#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.sqrt(xFeaturesTrimmed[:,0])
xFeaturesTrimmed[:,2] = np.log(xFeaturesTrimmed[:,2])
xFeaturesTrimmed[:,4] = np.power((xFeaturesTrimmed[:,4]), 3)
reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


print("Of these features, Iteration 11 had the lowest MSE at 51.999, and the lowest SSE at 21475.76")
print("this iteration trimmed the ID feature, took the mod of x1, the log of x3, and squared x5 and x6 features")
print("the manual removal of outlier data gave the regression algorithm the largest decrease in SSE and MSE")



#################################################################################
print("iteration 15")
print("we will now trim the x1 feature (transaction date) to explore additional changes to the feature set")

#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

# combine features 5 & 6 (adding features)
location = xFeatures[:,5]*xFeatures[:,6]
#reshape location array
location = location.reshape(-1,1)

print("trimming: xFeature[:,1]")
print("reinserting: x5 and x6 seperately")
xFeaturesTrimmed = np.hstack((xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]])) 

# New Column indexing 0-4 are features
#X0=the house age (unit: year) 
#X1=the distance to the nearest MRT station (unit: meter) 
#X2=number of convenience stores in foot travel
#X3=the geographic coordinate, latitude. (unit: degree) 
#x4=the geographic coordinate, longitude. (unit: degree) 

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# log scale 2nd feature (distance to MRT)
x1 = xFeaturesTrimmed[:,[1]]
x1 = np.log10(x1)

reg = LinearRegression().fit(x1 , yLabels)
predictedY = reg.predict(x1)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log2 and sqrt):" + str(reg._residues))

# the x3 feature (lat)
x3 = xFeaturesTrimmed[:,[3]]

reg = LinearRegression().fit(x3, yLabels)
predictedY = reg.predict(x3)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined variable 4 (lat) (no transform):" + str(reg._residues))

# the x4 feature (long)
x4 = xFeaturesTrimmed[:,[4]]

reg = LinearRegression().fit(x4, yLabels)
predictedY = reg.predict(x4)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined variable 4 (lat) (no transform):" + str(reg._residues))

#Fit the model again with the updated features:
xFeaturesTrimmed[:,1] = np.log(xFeaturesTrimmed[:,1])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


#################################################################################
print("iteration 16")
print("trimming the x1 feature (transaction date) increased bothed MSE and SSE")
print("we will trim a final feature, x2, house age")

#load data - skip first row which only contains metadata
reData = np.loadtxt('reDataUCI-modified.csv', delimiter = ",", skiprows = 1)
numRows = np.size(reData,0)
numCols = np.size(reData,1)
xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,7]

print("trimming: xFeature[:,1], xFeature[:,2]")
print("reinserting: x5 and x6 seperately")
xFeaturesTrimmed = np.hstack((xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]])) 

# New Column indexing 0-4 are features
#X0=the distance to the nearest MRT station (unit: meter) 
#X1=number of convenience stores in foot travel
#X2=the geographic coordinate, latitude. (unit: degree) 
#X3=the geographic coordinate, longitude. (unit: degree)


reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))


# log scale 1st feature (distance to MRT)
x0 = xFeaturesTrimmed[:,[0]]
x0 = np.log10(x0)

reg = LinearRegression().fit(x0 , yLabels)
predictedY = reg.predict(x0)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log2 and sqrt):" + str(reg._residues))

# the x3 feature (lat)
x2 = xFeaturesTrimmed[:,[2]]

reg = LinearRegression().fit(x2, yLabels)
predictedY = reg.predict(x2)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for combined variable 3 (lat) (no transform):" + str(reg._residues))

# the x4 feature (long)
x3 = xFeaturesTrimmed[:,[3]]

reg = LinearRegression().fit(x3, yLabels)
predictedY = reg.predict(x3)
#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for variable 4 (lat) (no transform):" + str(reg._residues))

#Fit the model again with the updated features:
xFeaturesTrimmed[:,0] = np.log(xFeaturesTrimmed[:,0])

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
yPredicted  = reg.predict(xFeaturesTrimmed)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))


print("removing additional features increased both SSE and MSE")
print("iteration 11 produced the lowest errors for any of the previous iterations")

exit

























