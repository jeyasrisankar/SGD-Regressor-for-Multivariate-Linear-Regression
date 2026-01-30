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
from sklearn.linear_model import LinearRegression
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

price_model = LinearRegression()
occupant_model = LinearRegression()

price_model.fit(X_scaled, price)
occupant_model.fit(X_scaled, occupants)

new_house = np.array([[1100, 4]])
new_house_scaled = scaler.transform(new_house)

predicted_price = price_model.predict(new_house_scaled)
predicted_occupants = occupant_model.predict(new_house_scaled)

print("Predicted House Price:", round(predicted_price[0], 2))
print("Predicted Number of Occupants:", round(predicted_occupants[0]))

house_size = X[:, 0].reshape(-1, 1)


rooms_mean = np.mean(X[:, 1])
X_plot = np.column_stack((house_size, np.full(house_size.shape, rooms_mean)))
X_plot_scaled = scaler.transform(X_plot)

plt.figure()
plt.scatter(X[:, 0], price, color='blue', label='Actual Price')
plt.plot(house_size, price_model.predict(X_plot_scaled), color='red', label='Predicted Line')
plt.scatter(new_house[0][0], predicted_price, color='green', s=100, label='Predicted Point')
plt.xlabel("House Size (sq.ft)")
plt.ylabel("House Price")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()



```

## Output:




<img width="1186" height="695" alt="Screenshot 2026-01-30 144553" src="https://github.com/user-attachments/assets/477634ed-d8dc-4e01-b521-af87f6a6e112" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
