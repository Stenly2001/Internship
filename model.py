
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("credit.csv")

le= LabelEncoder()

data["Name"] = le.fit_transform(data["Name"])

data["Occupation"] = le.fit_transform(data["Occupation"])

data["Type_of_Loan"] = le.fit_transform(data["Type_of_Loan"])

data["Credit_Mix"] = le.fit_transform(data["Credit_Mix"])

data["Payment_of_Min_Amount"] = le.fit_transform(data["Payment_of_Min_Amount"])

data["Payment_Behaviour"] = le.fit_transform(data["Payment_Behaviour"])

data["Credit_Score"] = le.fit_transform(data["Credit_Score"])

data = pd.get_dummies(data)

X = data.drop("Credit_Score",axis=1)
Y = pd.DataFrame(data["Credit_Score"])

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2, random_state=42)

rf_cls=RandomForestClassifier()

model_rf=rf_cls.fit(x_train,y_train)

y_pred_rf=model_rf.predict(x_test)

import pickle
filename="savemodel.pickle"
with open(filename,"wb") as file:
  pickle.dump(model_rf,file)