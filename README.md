# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1:
Import the needed packages.

Step2:
Read the txt file using read_csv.

Step3:
Use numpy to find theta,x,y values.

Step4:
To visualize the data use plt.plot.

## Program:
```
Program to implement the linear regression using gradient descent.
NAME:PRASANNA G R 
REG NO:212221040129
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) *"+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.ylabel("Profit predictions")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:


Profit Prediction:




![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/63895cc7-a4c8-46fd-bbd2-612047a8bcec)


Computecost:



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/058812b6-dd72-436f-852d-958f3dfa6e4e)




![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/adfd6b58-743e-431f-9d37-75f6074dc4b5)



Cost Function using Gradient Descent:



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/576898f3-dedb-44ac-bb9a-79f860df8dc0)


Profit Prediction:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/2e453d55-25dd-4731-a391-184ec19328b3)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/86e17f1c-cf23-4224-a4f8-7940545eaafb)


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128134963/79ae251a-5ed1-4414-8998-553440d20fe8)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
