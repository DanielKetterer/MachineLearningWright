print("Homework 2")
import numpy as np
import matplotlib.pyplot as plt

reData = np.loadtxt('C:/Users/w181dxk/Desktop/MachineLearningWright-master/Absenteeism_at_work.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)-1



trainingData = reData[:,1:numCols-1]

#Last column are the labels
labels = reData[:,[numCols-1]]


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
def tester(trainingData,labels,arange,erange,astart,estart):
  MetaCostHistory = []
  MetaEpochHistory = []
  for i in range(arange):
   alpha = 10**(-i-astart)
   CostHistory = []
   EpochHistory = []
   for w in range(erange):
    epochs =10**(w+estart)
    parameters = bgd(trainingData, labels, alpha, .0000001, epochs)
    Hypo = np.dot(trainingData, parameters)
    sse = np.sum(np.square(np.subtract(labels,Hypo)))
    mse = np.mean(np.square(np.subtract(labels,Hypo)))
    print('DataSet SSE and MME: alpha and epochs ' + str(sse) + str(', ') + str(mse) + str(', ') + str(alpha) + str(', ') + str(epochs))
    Diff = Hypo - labels
    Cost = (1/2*numRows) * np.sum(np.square(Diff) )
    CostHistory.append(Cost)
    EpochHistory.append(epochs)
   MetaCostHistory.append(CostHistory)
   MetaEpochHistory.append(EpochHistory)
  return [MetaCostHistory, MetaEpochHistory]


arange = 6
erange = 4
astart = 9
estart = 1
[a,b]=tester(trainingData,labels,arange,erange,astart,estart)

i = 1
while i < arange:
 fig = plt.figure()
 plt.plot(a[i], b[i], color = 'r')
 fig.suptitle("alpha = " + str(i))
 plt.xlabel("Epoch #")
 plt.ylabel("Cost")
 plt.show()
 i=i+1
"""
5. Repeat tasks 1.3, 2.3 and 2.4 for another dataset of your choosing. You
may use any dataset you wish from any public repository (UCI, Kaggle,
etc.). Give a brief description of the dataset (features, labels).

"""
#this is using the new dataset
#reData = np.loadtxt('Admission_Predict.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)-1


trainingData = reData[:,1:numCols-1]
labels = reData[:,[numCols-1]]

newtest = bgd(trainingData, labels, .00001, .0000001, 100000)





"""


#References
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass
https://archive.ics.uci.edu/ml/datasets/Tarvel+Review+Ratings
https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""
