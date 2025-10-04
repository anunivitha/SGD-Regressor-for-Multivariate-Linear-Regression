# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.

2. Load the California housing dataset.

3. Select input features (X) and output targets (y).

4. Split dataset into training and testing sets.

5. Standardize both input and output data using StandardScaler.

6. Initialize SGDRegressor and wrap it with MultiOutputRegressor for multivariate prediction.

7. Train the model with training data.

8. Predict results on the test data.

9. Inverse transform predictions to original scale.

10. Compute and display Mean Squared Error (MSE) and sample predictions.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Anu Nivitha U
RegisterNumber: 212223040016
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
x = data.data[:, :3]
y = np.column_stack((data.target, data.data[:, 6]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train)

y_pred = multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("\nPredictions:\n", y_pred[:5])

```

## Output:
<img width="572" height="354" alt="Screenshot 2025-10-04 at 10 29 01 AM" src="https://github.com/user-attachments/assets/7b97e756-71cd-49db-ba7e-5b618d06cff0" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
