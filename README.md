# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sri hari R
RegisterNumber: 212223040202
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:
# HEAD(), INFO() & NULL():
![WhatsApp Image 2024-04-08 at 17 02 31_0e3a0ea9](https://github.com/srrihaari/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145550674/76e3a0d3-08d1-4834-bd9b-4e941a108595)

# Converting string literals to numerical values using label encoder:
![WhatsApp Image 2024-04-08 at 17 02 36_8d1d3e52](https://github.com/srrihaari/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145550674/8fe61671-fd9e-47ec-86f3-400f7bbc0e98)


# MEAN SQUARED ERROR:
![WhatsApp Image 2024-04-08 at 17 02 41_5042b2aa](https://github.com/srrihaari/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145550674/50de743e-7ee3-4f4b-bcb1-27cd800640ce)


# R2 (Variance):
![WhatsApp Image 2024-04-08 at 17 02 45_cf181592](https://github.com/srrihaari/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145550674/9b58030e-923a-4e8f-85e8-4a4618086e0d)


# DATA PREDICTION & DECISION TREE REGRESSOR FOR PREDICTING THE SALARY OF THE EMPLOYEE:
![WhatsApp Image 2024-04-08 at 17 02 48_2f5565f7](https://github.com/srrihaari/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145550674/cc80a373-461f-46b9-8e4c-95bc9b76445e)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
