# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Preparation: Collect a dataset of study hours (g) and corresponding marks (k), then split them into training and testing sets.

2.Model Training: Calculate the slope (o) and intercept (p) using the Ordinary Least Squares method to minimize the sum of squared differences between predicted and actual values.

3.Prediction: Apply the learned parameters to new input data using the linear equation: " K = oG + p".

4.Evaluation: Assess the model's performance by calculating metrics such as Mean Squared Error (MSE) or R^2 score on the test set.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Abdul Rahim M
RegisterNumber:  212225040007
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
G = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
K = np.array([40, 50, 60, 80, 100])
model = LinearRegression()
model.fit(G, K)
o = model.coef_[0]
p = model.intercept_
print("Slope (o):", o)
print("Intercept (p):", p)
g_input = float(input("Enter hours studied: "))
predicted_marks = model.predict([[g_input]])
print("Predicted Marks:", predicted_marks[0])
K_pred = model.predict(G)
plt.scatter(G, K, label="Actual Data")
plt.plot(G, K_pred, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression (Using sklearn)")
plt.legend()
plt.show()
```

## Output:

<img width="732" height="664" alt="image" src="https://github.com/user-attachments/assets/89e20550-b84a-42f5-84f9-8c63e8b8cc06" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
