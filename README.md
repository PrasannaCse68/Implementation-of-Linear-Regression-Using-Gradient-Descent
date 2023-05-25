# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python.
2.Set variables for assigning dataset values.
3.Import LinearRegression from the sklearn.
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of graph.
6.Compare the graphs and hence we obtain the LinearRegression for the given data
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

import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:


![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/64a6ce86-9d5b-4739-ada0-8fca70cfdb56)


![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/34004a6c-f39c-430d-982d-6b8389cbc987)



![image](https://github.com/PrasannaCse68/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127935950/41bbfa4e-0af1-4b64-a42b-ef03b010d087)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
