import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pizza = pd.read_csv("pizza.csv")

lr = LinearRegression()

X = pizza[['Promote']]
y = pizza['Sales']

lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)

# yi^
y_pred = lr.predict(X)

print(r2_score(y, y_pred))
# numerator = np.sum((y - y_pred)**2)
# denominator = np.sum((y - y.mean())**2)
# 1 - (numerator/denominator)

############## insure auto #####################
insure = pd.read_csv("Insure_auto.csv")

X = insure[['Home']]
y = insure['Operating_Cost']
lr.fit(X,y)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

X = insure[['Automobile']]
y = insure['Operating_Cost']
lr.fit(X,y)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

X = insure[['Home','Automobile']]
y = insure['Operating_Cost']
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

############ Boston #######################
boston = pd.read_csv("Boston.csv")
X = boston.iloc[:,:-1]
# or
X = boston.drop('medv', axis=1)
y = boston['medv']
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

############### Concrete ###################
concrete = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']
lr = LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

############ Exp Salaries ################
exp_sals = pd.read_csv("Exp_Salaries.csv")
dum_sals = pd.get_dummies(exp_sals, drop_first=True)
X = dum_sals.drop('Salary', axis=1)
y = dum_sals['Salary']
lr = LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

# Generating the predictions
salsToPred = pd.read_csv("SalsToPredict.csv")
dum_pred = pd.get_dummies(salsToPred, drop_first=True)
predictions = lr.predict(dum_pred)

############ Wedding
import os
os.chdir(r"C:\Training\Academy\Business Analytics 3e Resources\Excel Datasets\eba3e_datasets_xls")
wedding = pd.read_excel("Weddings.xlsx",usecols="A:F",skiprows=2)

# 1.
X = wedding[['Wedding cost']]
y = wedding['Attendance']
lr = LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

# 2.
X = wedding[['Wedding cost']]
y = wedding['Value Rating']
lr = LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))

# 3.
X = wedding[["Couple's Income","Payor"]]
X = pd.get_dummies(X, drop_first=True)
y = wedding['Value Rating']
lr = LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X)
print(r2_score(y, y_pred))




