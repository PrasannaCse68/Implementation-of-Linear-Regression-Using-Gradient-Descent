# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step1: Import the needed packages.

Step2: Read the txt file using read_csv.

Step3: Use numpy to find theta,x,y values.

Step4: To visualize the data use plt.plot.
```

## Program:
```

/*
Program to implement the linear regression using gradient descent.
NAME:PRASANNA GR
REG NO:212221040129
*/
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

![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/404921d7-0217-4165-9e4f-440a3f82369c)


Computecost:


![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/91c15d19-e35b-49e2-9752-16962a9dfb39)

![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/41b9b345-83d4-4b75-b842-d02fa6069aa1)



Cost Function using Gradient Descent:

![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/5049b6cd-0881-447f-9ee9-3f82cba9b6d1)


Profit Prediction:

![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/3feeaa13-f175-4004-9881-71823ce11899)


![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/ad5d6932-f585-407f-8626-e64c5ee73361)


![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/4a10dc15-3095-4329-80bd-1d550e0bd9ef)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
