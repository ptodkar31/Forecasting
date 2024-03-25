# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:55:12 2024

@author:priyanka
"""

'''
1.	Officeworks is a leading retail store in Australia, with numerous 
outlets around the country. The manager would like to improve the customer
experience by providing them online predictive prices for their laptops 
if they want to sell them. To improve this experience the manager would 
like us to build a model which is sustainable and accurate enough. Apply Lasso and Ridge Regression model on the dataset and predict the price,
 given other attributes. Tabulate R squared, RMSE, and correlation values.
 '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
# Replace 'dataset.csv' with the actual path to your dataset
data = pd.read_csv("C:\Data Set\Forecasting\50_Startups.csv")

# Assuming 'Price' is the target variable and other columns are features
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust alpha parameter for regularization strength
lasso_model.fit(X_train_scaled, y_train)

# Define and train Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust alpha parameter for regularization strength
ridge_model.fit(X_train_scaled, y_train)

# Predictions
lasso_preds = lasso_model.predict(X_test_scaled)
ridge_preds = ridge_model.predict(X_test_scaled)

# Evaluate models
lasso_r2 = r2_score(y_test, lasso_preds)
ridge_r2 = r2_score(y_test, ridge_preds)

lasso_rmse = mean_squared_error(y_test, lasso_preds, squared=False)
ridge_rmse = mean_squared_error(y_test, ridge_preds, squared=False)

# Calculate correlation between actual and predicted values
lasso_corr = pd.Series(lasso_preds).corr(pd.Series(y_test))
ridge_corr = pd.Series(ridge_preds).corr(pd.Series(y_test))

# Tabulate results
results = pd.DataFrame({
    'Model': ['Lasso Regression', 'Ridge Regression'],
    'R-squared': [lasso_r2, ridge_r2],
    'RMSE': [lasso_rmse, ridge_rmse],
    'Correlation': [lasso_corr, ridge_corr]
})

print(results)








