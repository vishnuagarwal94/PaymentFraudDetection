# Import Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Selecting Machine Learning Algorithm to Fit Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Uploading Dataset 
FraudData = pd.read_csv("FraudBook3.csv")
print(FraudData.head())
print(FraudData.shape)

# Data Analysis
print(FraudData.isnull().sum())
print(FraudData.info())
print(FraudData.describe())
print(FraudData.corr())

print(FraudData.type.value_counts())
print(FraudData.isFraud.value_counts())
print(FraudData.isFlaggedFraud.value_counts())

# Ploting of Data to understand Relation
plt.hist(FraudData.type)
plt.show()

plt.pie(FraudData.type.value_counts() , labels=["CASH_OUT","PAYMENT","CASH_IN","TRANSFER","DEBIT"])
plt.show()

plt.hist(FraudData.oldbalanceOrg)
plt.show()

plt.hist(FraudData.newbalanceOrig)
plt.show()

plt.hist(FraudData.oldbalanceDest)
plt.show()

plt.hist(FraudData.newbalanceDest)
plt.show()

plt.hist(FraudData.isFraud)
plt.show()

# Formating Data from String to Numerical
FraudData["type"] = FraudData["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
X = FraudData[["type","amount","oldbalanceOrg", "newbalanceOrig","oldbalanceDest","newbalanceDest"]]
FraudData["isFraud"] = FraudData["isFraud"].map({0: "No Fraud", 1: "Fraud"})
y = FraudData["isFraud"]

# Spliting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train)
print(y_train)

# Fitting Model to Logistic Regression Algorithm
Log = LogisticRegression()

model = Log.fit(X_train,y_train)
prediction = model.predict(X_test)

# Checking Accurecy of Model
print(model.score(X_test,y_test))

# Making Prediction on Model
feature = np.array([[2,181,181,0,0,0]])
print(model.predict(feature))