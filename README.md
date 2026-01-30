# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Define the input features (house size and number of rooms).

3. Define the output variables (house price and number of occupants).

4. Perform feature scaling.

5. Create SGD Regressor models.

6. Train the models using the training data.

7. Predict the house price and number of occupants for new input data.

8. Plot the regression graph.


## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X = np.array([
    [500, 2],
    [800, 3],
    [1000, 3],
    [1200, 4],
    [1500, 5]
])


price = np.array([20, 35, 45, 55, 70])      
occupants = np.array([2, 3, 4, 5, 6])     


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


price_model = SGDRegressor(max_iter=5000, tol=1e-4)
occupant_model = SGDRegressor(max_iter=5000, tol=1e-4)

price_model.fit(X_scaled, price)
occupant_model.fit(X_scaled, occupants)

new_house = np.array([[1100, 4]])
new_house_scaled = scaler.transform(new_house)

predicted_price = price_model.predict(new_house_scaled)
predicted_occupants = occupant_model.predict(new_house_scaled)

print("Predicted House Price:", round(predicted_price[0], 2))
print("Predicted Number of Occupants:", round(predicted_occupants[0]))

house_size = X[:, 0].reshape(-1, 1)

plt.figure()
plt.scatter(house_size, price)
plt.plot(house_size, price_model.predict(X_scaled))
plt.scatter(new_house[0][0], predicted_price)
plt.xlabel("House Size (sq.ft)")
plt.ylabel("House Price")
plt.title("Multivariate Linear Regression using SGD Regressor")
plt.show()


```

## Output:


<img width="1198" height="692" alt="Screenshot 2026-01-30 143208" src="https://github.com/user-attachments/assets/f53bdc4f-3af5-44ed-94e0-218c81ea16df" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
