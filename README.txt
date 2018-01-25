Perfom below experiments on olympic dataset:

1) Load the data in olympics.mat into Python.  This file contains several datasets: male100, male200, male400, female100, female200, and female400.  For each dataset, the first column is the year, and second column is the time in seconds.  You may ignore the other columns.
2) Use matplotlib.pyplot to plot male100, reproducing Figure 1.1 in the textbook.
3) Use sklearn.linear_model.LinearRegression to fit male100.  List the coefficients, then predict the values for x = 2012 and x = 2016.
4) Plot male100 and the line you fit in experiment.
5) Use linear regression to fit a line to female400.  How does the error for this model compare to the error when fitting a line to male100?
6) Fit a 3rd order polynomial to female400.  Does the error improve?
7) Fit a 5th order polynomial to female400.  Does the error improve?
8) Use LOOCV for both the 3rd and 5th order polynomials. (Hint: use sklearn.model_selection.LeaveOneOut.) Which polynomial is a better choice?
9) Use sklearn.linear_model.Ridge with α = 0.1 to fit a 5th order polynomial to female400.  How do the coefficients compare to those found with linear regression?
10) Use sklearn.linear_model.RidgeCV to find the best value for α across the range 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0.


To execute the code please change the file path on line number 16 with your directory location. (Download the attached dataset on our system)