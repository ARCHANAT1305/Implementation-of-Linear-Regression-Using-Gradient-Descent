# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset from a CSV file.
2. Separate the data into features (X) and target variable (y).
3. Add a bias term (column of ones) to the feature matrix
4. Calculate the cost J(θ) for linear regression, which measures the difference between predicted values and actual value.
5. This is done by computing the squared errors and taking the average.
6. Initialize the parameters (theta) to zero.Iterate to update the parameters using the gradient descent algorithm, which minimizes the cost function by moving in the direction of steepest descent.
7. Update the parameters until convergence or until a specified number of iterations is reached.Track the history of the cost function values to monitor convergence.
8. Train the linear regression model by minimizing the cost function using gradient descent.
9. Use the trained model (theta) to make predictions for given input values (e.g., population sizes).Example predictions are made for populations of 35,000 and 70,000.
10. Plot the original data points and the fitted regression line.Plot the cost function values over iterations to visualize the convergence.
## Program:

Program to implement the linear regression using gradient descent.   
Developed by: ARCHANA T    
RegisterNumber:  212223240013   
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header = None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def computeCost(X,y,theta):
  m=len(y) 
  h=X.dot(theta) 
  square_err=(h-y)**2
  return 1/(2*m) * np.sum(square_err) #returning J
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

from matplotlib.container import ErrorbarContainer
from IPython.core.interactiveshell import error
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
      predictions=X.dot(theta)
      error=np.dot(X.transpose(),(predictions -y))
      descent=alpha *1/m*error
      theta-=descent
      J_history.append(computeCost(X,y,theta))
    return theta,J_history
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population =35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```
## Output:
Profit Prediction Graph :

![image](https://github.com/user-attachments/assets/d3c2e3ea-eb4c-4796-8e02-a1400663700b).

![image](https://github.com/user-attachments/assets/6b77689e-e320-428f-992f-d8e3509ca237)


Cost function using Gradient Descent Graph :
![image](https://github.com/user-attachments/assets/77fb2702-149c-4c61-bded-86c4e02e2f77)

![image](https://github.com/user-attachments/assets/7d37b265-b807-440f-8962-2d9fafecb628).
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
