"""
Objectives
After completing this lab you will be able to:

Implement, evaluate, and compare the performance of three regularization techniques for linear regression
Analyze the effect of simple linear regularization when modelling on noisy data with and without outliers
Use Lasso regularization to reduce the number of features for subsequent multiple linear regression modelling
"""
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

"""
Define a function to display evaluation metrics
We'll be evaluating several models, so its a good idea to make an evaluation function that includes the common metrics and explaned variances etc.
"""
def regression_results(y_true, y_pred, regression_type ):

  ev = explained_variance_score(y_true, y_pred)
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  r2_score = r2_score(y_true, y_pred)

  print('Evaluation metrics for ' + regression_type + ' Linear Regression')
  print("explained_variance_score", round(ev, 4))
  print("mean_absolute_error", round(mae, 4))
  print("mean_squared_error", round(mse, 4))
  print("RMSE", round(np.sqrt(mse), 4))
  print("r2_score", round(r2_score, 4))

"""
Generate a simple dataset with one feature¶
We'll create a simple data set with a linear relationship between the target and a single feature. We'll add some noise to the target to simulate real data. We'll also create a data set from this one that has outliers added.
Then you'll compare the three linear regression model performances on the two datasets, with and without sigiifcant outliers added.
"""

# Generate synthetic data
noise = 1
np.random.seed(42) # np.random.seed(42) makes NumPy's random-number sequence reproducible, which is useful when testing and debugging machine learning code.

X = 2 * np.random.rand(1000, 1) # generates 1,000 random numbers arranged in a column with 1,000 rows and 1 column.
y = 4 + 3 * X + noise * np.random.randn(1000, 1)  # Linear relationship with some noise
y_ideal =  4 + 3 * X

# Specify the portion of the dataset to add outliers (e.g., the last 20%)
y_outlier = pd.Series(y.reshape(-1).copy())

# Identify indices where the feature variable X is greater than a certain threshold
threshold = 1.5  # Example threshold to add outliers for larger feature values
outlier_indices = np.where(X.flatten() > threshold)[0]

# Add outliers at random locations within the specified portion
num_outliers = 5  # Number of outliers to add
selected_indices = np.random.choice(outlier_indices, num_outliers, replace=False)

# Modify the target values at these indices to create outliers (add significant noise)
y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)

# Plot the data with outliers and the ideal fit line

plt.figure( figsize = (12, 6))

plt.scatter( X, y_outlier , ec= 'k' , alpha = 0.3 , label = "Original Data with Outliers")
plt.plot(X, y_ideal , lw = 3, color='g', label = "ideal fit line")

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()

# Exercise 1. Plot the data without the outliers and the ideal fit line

plt.figure( figsize = (12, 6))

plt.scatter(X , y , ec= 'k', alpha = 0.3, label = "Original Data without Outliers")
plt.plot(X, y_ideal, lw = 3, color='g', label = "Ideal, noise free data" )

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()

# Fit Ordinary, Ridge, and Lasso regression models and use them to make predictions on the outlier data

lin_reg = LinearRegression()
lin_reg.fit(X, y_outlier)
y_outlier_pred_lin = lin_reg.predict(X)

# How it works: Ridge encourages coefficients to become smaller, but generally does not reduce them exactly to zero.
ridge_reg = Ridge( alpha = 1) # alpha=1: Controls the strength of regularization, which penalizes large model coefficients.
ridge_reg.fit(X, y_outlier)
y_outlier_pred_ridge = ridge_reg.predict(X)

# How it works: Lasso can shrink some coefficients to exactly zero, effectively selecting a smaller set of features.
lasso_reg = Lasso( alpha = 0.2 ) # alpha=0.2: Controls the strength of regularization.
lasso_reg.fit(X, y_outlier)
y_outlier_pred_lasso = lasso_reg.predict(X)

# Print the regression results
regression_results (y_outlier , y_outlier_pred_lin, "Regular")
regression_results (y_outlier , y_outlier_pred_lin, "Ridge")
regression_results (y_outlier , y_outlier_pred_lin, "Lasso")

"""
Evaluation metrics for Ordinary Linear Regression
explained_variance:  0.6748
r2:  0.6357
MAE:  0.9468
MSE:  1.4164
RMSE:  1.1901

Evaluation metrics for Ridge Linear Regression
explained_variance:  0.6766
r2:  0.6375
MAE:  0.9443
MSE:  1.4094
RMSE:  1.1872

Evaluation metrics for Lasso Linear Regression
explained_variance:  0.7394
r2:  0.7003
MAE:  0.8597
MSE:  1.1651
RMSE:  1.0794
"""


  

