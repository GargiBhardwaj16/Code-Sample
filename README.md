# Code-Sample
As a B.tech student , we all dream to be in big companys and solve a real world problem at our own level , but in current senario all the companys including FAANG is doing mass layoffs. Mostly freshers are lossing the jobs , if company could predict that which employee wanted to work dedicatedly , it would we easy for them to decide which employee should be fired , so to solve this issue I have coded in python , "Employee Atrittion Prediction".
# Importing all Library
*import pandas as pd
*import numpy as np
*import matplotlib.pyplot as plt
*import seaborn as sns
*from sklearn.model_selection import train_test_split
*from sklearn.metrics import confusion_matrix
*from sklearn import datasets
*from sklearn.metrics import accuracy_score
*from sklearn.metrics import plot_confusion_matrix 
*from sklearn.linear_model import LogisticRegression
*from sklearn.tree import DecisionTreeClassifier
*from sklearn.ensemble import RandomForestClassifier
*from sklearn.naive_bayes import GaussianNB
*from sklearn.neighbors import KNeighborsClassifier
*from sklearn import svm
*from sklearn.metrics import classification_report
*df=pd.read_csv(r"C:\Users\dushy\Downloads\WA_Fn-UseC_-HR-Employee-Attrition (1).csv") //dataset is of IBM , taken from kaggle.
*df.head()
*df.drop(0,inplace=True)
*df.isnull().sum()
*df.dropna(axis=0,inplace=True)
# One Hot Encoding
def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
df['OverTime'] = df['OverTime'].replace({'No': 0, 'Yes': 1})
    
    # Ordinal-encode the BusinessTravel column
df['BusinessTravel'] = df['BusinessTravel'].replace({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    
 for column in ['Department', 'EducationField', 'JobRole', 'MaritalStatus']:
        df = onehot_encode(df, column=column)
    
attrition_dict = df["Attrition"].value_counts()
attrition_dict
No     1233
Yes     236
Name: Attrition, dtype: int64
sns.set_style('darkgrid')
sns.countplot(x ='Attrition', data = df)
# Model Used
lr=LogisticRegression(C = 0.1, random_state = 42, solver = 'liblinear')
dt=DecisionTreeClassifier()
rm=RandomForestClassifier()
gnb=GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
svm = svm.SVC(kernel='linear')
y = df['Attrition']
X = df.drop('Attrition', axis=1)
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
X1=X_train
X_train.head()
for a,b in zip([lr,dt,knn,svm,rm,gnb],["Logistic Regression","Decision Tree","KNN","SVM","Random Forest","Naive Bayes"]):
    a.fit(X_train,y_train)
    prediction=a.predict(X_train)
    y_pred=a.predict(X_test)
    score1=accuracy_score(y_train,prediction)
    score=accuracy_score(y_test,y_pred)
    msg1="[%s] training data accuracy is : %f" % (b,score1)
    msg2="[%s] test data accuracy is : %f" % (b,score)
    print(msg1)
    print(msg2)
model_scores={'Logistic Regression':lr.score(X_test,y_test),
             'KNN classifier':knn.score(X_test,y_test),
             'Support Vector Machine':svm.score(X_test,y_test),
             'Random forest':rm.score(X_test,y_test),
              'Decision tree':dt.score(X_test,y_test),
              'Naive Bayes':gnb.score(X_test,y_test)
             }
model_scores
# Classification Report
LOGISTIC REGRESSION: classification_report
from sklearn.metrics import classification_report

lr_y_preds = lr.predict(X_test)

print(classification_report(y_test,lr_y_preds))
# confusion_matrix
lr=LogisticRegression()
lr.fit(X_train,y_train)
disp=plot_confusion_matrix(lr,X_train,y_train,cmap="Blues",values_format='3g')
