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

"""
Ridge and Lasso are regularized linear regression models; Ridge shrinks coefficients, while Lasso can shrink some coefficients to zero and perform feature selection.
"""

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

Judging from the low R^2 values, these are poor predictions
"""

"""
Plot the data and the predictions for comparison
Let's see how well the predictions match expected ideal values.
"""

plt.figure( figsize = (12, 6))

# Scatter plot of the original data with outliers
plt.scatter(X, y , alpha = 0.4, ec = 'k' , label="Original data")

# Plot the ideal regression line (noise free data)
plt.plot(X, y_ideal, lw = 2, color='k' label = "Ideal, noise free data")

# Plot predictions from the simple linear regression model
plt.plot(X, y_outlier_pred_lin, lw = 5 , label = "Linear regression")

# Plot predictions from the ridge regression model
plt.plot(X, y_outlier_pred_ridge, lw= 2, linestyle = '--', label = "Ridge regression")

# Plot predictions from the lasso regression model
plt.plot(X, y_outlier_pred_lasso, lw= 2, color = 'g', label = "Ridge regression")

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with Outliers')
plt.legend()
plt.show()

# Exercise 2. Build the models and the prediction plots from the same data, excluding the outliers

# Fit a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Fit a ridge regression model (regularization to control large coefficients)
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y)
y_pred_ridge = ridge_reg.predict(X)

# Fit a lasso regression model (regularization to control large coefficients)
lasso_reg = Lasso(alpha=0.2)
lasso_reg.fit(X, y)
y_pred_lasso = lasso_reg.predict(X)

# Print the regression results
regression_results(y, y_pred_lin, 'Ordinary')
regression_results(y, y_pred_ridge, 'Ridge')
regression_results(y, y_pred_lasso, 'Lasso')

"""
Evaluation metrics for Ordinary Linear Regression
explained_variance:  0.7492
r2:  0.7492
MAE:  0.7873
MSE:  0.975
RMSE:  0.9874

Evaluation metrics for Ridge Linear Regression
explained_variance:  0.7492
r2:  0.7492
MAE:  0.7874
MSE:  0.975
RMSE:  0.9874

Evaluation metrics for Lasso Linear Regression
explained_variance:  0.7191
r2:  0.7191
MAE:  0.8408
MSE:  1.0923
RMSE:  1.0451
"""


# Plot the data and the predictions
plt.figure(figsize=(12, 8))

# # Scatter plot of the original data
plt.scatter(X, y, alpha=0.4, ec='k', label='Original Data')

# Plot the ideal regression line (noise free data)
plt.plot(X, y_ideal,  linewidth=2, color='k',label='Ideal, noise free data')

# Plot predictions from the simple linear regression model
plt.plot(X, y_pred_lin,  linewidth=5, label='Linear Regression')

# Plot predictions from the ridge regression model
plt.plot(X, y_pred_ridge, linestyle='--',linewidth=2, label='Ridge Regression')

# Plot predictions from the lasso regression model
plt.plot(X, y_pred_lasso,  linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')

plt.title('Comparison of predictions with no outliers')
plt.legend()
plt.show()

  

