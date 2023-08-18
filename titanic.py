import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

class titanic:
    def entry(self,data):
        df=pd.read_csv("tested.csv")
        df["Age"].fillna(df["Age"].mean(),inplace=True)
        df["Embarked"].replace({"S":0,"C":1,"Q":2},inplace=True)
        df["Sex"].replace({"male":0,"female":1},inplace=True)
        survive_data=df["Survived"]
# Since Ticket, Name and id doesnt have any good purpose in our data set we can remove them
        new_data=df.drop(["Ticket","Name","PassengerId","Cabin","Survived"],axis=1)
        new_data.fillna(df["Fare"].mean(),inplace=True)
        x_train,x_test,y_train,y_test=train_test_split(new_data,survive_data,test_size=0.2,random_state=42)
        fr= RandomForestClassifier()

        # Train the model
        fr.fit(x_train, y_train)

        #Make predictions on the test set
        predictions =int(fr.predict(data))
        
        return predictions