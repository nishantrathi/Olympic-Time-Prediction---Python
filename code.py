# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:39:25 2017

@author: nishantrathi
"""


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

mat = sio.loadmat('F:\\study\\Fall2017\\483\\Project1\\olympics.mat')

keyList = list(mat.keys())
keyList.pop(0)
keyList.pop(0)
keyList.pop(0)

#Question number 1
#Data Cleaning and putting values in a dictionary
olympicDict = {}
for key in keyList:
    olympicDict[key]={}
    for v in mat.get(key):
       olympicDict[key][int(v[0])]=float("{0:.2f}".format(v[1]))


male100_year = list(olympicDict.get('male100').keys())
male100_time = list(olympicDict.get('male100').values())
listSize = len(male100_year)
percent = int((listSize*20)/100)

#Question number 2
plt.figure(1)
plt.plot(male100_year,male100_time,"o")
plt.xlabel('Year')
plt.ylabel('Time (seconds)')


#Linear Regression
year_x_train = male100_year[:listSize-percent]
year_x_test = male100_year[listSize-percent:listSize]
time_y_train = male100_time[0:listSize-percent]
time_y_test = male100_time[listSize-percent:listSize]

regr = linear_model.LinearRegression()

#for 80% data traing
regr.fit(np.array(year_x_train).reshape(-1,1), np.array(time_y_train).reshape(-1,1))

#Prediction for 20%
#time_y_pred = regr.predict((np.array(year_x_test).reshape(-1,1)))
#print(time_y_pred)

#Prediction for 100%
time_y_pred = regr.predict((np.array(male100_year).reshape(-1,1)))


#Prediction for 2012 and 2016 
newTimePred = regr.predict(np.array([2012,2016]).reshape(-1,1))
print("Question 3")
print("Prediction for x=2012 and x=2016")
print("2012= ",newTimePred[0]," 2016=",newTimePred[1])
print("  ")
print('Coefficients: w1 for Male 100m', regr.coef_[0])
print('Coefficients: w0 for Male 100m', regr.intercept_)
error_Male100 = mean_squared_error(male100_time, time_y_pred);
print("Mean squared error for Male 100m: %.2f" % mean_squared_error(male100_time, time_y_pred))

#Question number 4
plt.figure(2)
plt.scatter(male100_year, male100_time,  color='black')
plt.xlabel('Year')
plt.ylabel('Time (seconds)')
plt.plot(male100_year, time_y_pred, color='blue', linewidth=1)

#Question number 5
female400_year = list(olympicDict.get('female400').keys())
female400_time = list(olympicDict.get('female400').values())

listSize = len(female400_year)
percent = int((listSize*20)/100)
#Spliting data set in training set and test set
year_x_train = female400_year[0:listSize-percent]
year_x_test = female400_year[listSize-percent:listSize]
time_y_train = female400_time[0:listSize-percent]
time_y_test = female400_time[listSize-percent:listSize]

regr.fit(np.array(year_x_train).reshape(-1,1), np.array(time_y_train).reshape(-1,1))
    
#Prediction for 100% for Female 400m
time_y_pred = regr.predict((np.array(female400_year).reshape(-1,1)))

print("  ")
print('Coefficients: w1 for Female 400m', regr.coef_[0])
print('Coefficients: w0 for Female 400m', regr.intercept_)
error_female400= mean_squared_error(female400_time, time_y_pred)
print("Mean squared error for Female 400m: %.2f" % error_female400)
print("Comparision between Female 400m and Male 100m ->", error_female400-error_Male100)

plt.figure(3)
plt.scatter(female400_year, female400_time,  color='black')
plt.xlabel('Year')
plt.ylabel('Time (seconds)')
plt.plot(female400_year, time_y_pred, color='blue', linewidth=1, label="Linear")

#Function to find polynomial equation
def findPolyEquation(year,time,degree):
    coef = np.polyfit(year,time,degree)
    polyEqu=np.poly1d(coef)
    return polyEqu;

#Question number 6
plt.figure(3)
p= findPolyEquation(female400_year,female400_time,3)
plt.plot( female400_year, p(female400_year), 'k--', label="Degree 3")
print("  ")
print("Question 6----->>")
print("Mean squared error for Female 400m 3 poly: %.2f" % mean_squared_error(female400_time, p(female400_year)))

#Question number 7
plt.figure(3)
p= p= findPolyEquation(female400_year,female400_time,5)
plt.plot(female400_year, p(female400_year), 'r--', label="Degree 5")
plt.legend();
print("  ")
print("Question 7----->>")
print("Mean squared error for Female 400m 5 poly: %.2f" % mean_squared_error(female400_time, p(female400_year)))

#Question number 8 using Leave One Out Cross Validation
year = np.array(female400_year)
time = np.array(female400_time)
loo = LeaveOneOut()
loo.get_n_splits(year)
error_3degree =0
error_5degree =0

for train_index, test_index in loo.split(year):
    year_train, year_test = year[train_index], year[test_index]
    time_train, time_test = time[train_index], time[test_index]
    
    #Calculating error for degree 3
    p3 = findPolyEquation(year_train,time_train,3)
    error_3degree += mean_squared_error(time_train,p3(year_train))
    
    #Calculating error for degree 5
    p5 = findPolyEquation(year_train,time_train,5)
    error_5degree += mean_squared_error(time_train,p5(year_train))
    
print(" ")
print("Question 8----->>")
print("Mean squared error for Female 400m (3 Degree) using LOOCV: ",error_3degree/len(year))
print("Mean squared error for Female 400m (5 Degree) using LOOCV: ",error_5degree/len(year))

#Question number 9
p5 = findPolyEquation(female400_year,female400_time,5)
time_5poly = p5(female400_year)
regr = linear_model.Ridge(alpha=0.1)
regr.fit(np.array(female400_year).reshape(-1,1),np.array(time_5poly).reshape(-1,1))
time_y_pred = regr.predict(np.array(female400_year).reshape(-1,1))
print("  ")
print("Question 9----->>")
print("Mean squared error for Female 400m :Ridge Model: %.2f" % mean_squared_error(female400_time, time_y_pred))
print("Coefficients: w1 for Female 400m: Ridge Model",regr.coef_[0])
print("Coefficients: w0 for Female 400m: Ridge Model",regr.intercept_)

#Question number 10
regCV = linear_model.RidgeCV(alphas=[0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0],fit_intercept=True,store_cv_values=True)
regCV.fit(np.array(female400_year).reshape(-1,1),np.array(female400_time).reshape(-1,1))
print("  ")
print("Question 10----->>")
#print(regCV.cv_values_)
print("Best alpha value= ",regCV.alpha_)
