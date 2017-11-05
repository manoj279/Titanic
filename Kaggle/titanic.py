import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv('train.csv')
target=pd.read_csv('gender_submission.csv')

X_test=pd.read_csv('test.csv')
Y_test=target['Survived']

Y_train=train['Survived']

y=np.array(X_test['PassengerId'])

for column_name in ['Name','Ticket','Cabin','PassengerId']:
        train.drop(column_name, axis=1, inplace=True)
        X_test.drop(column_name, axis=1, inplace=True)

train.drop('Survived', axis=1, inplace=True)

#Fare
train['Fare'].fillna(train['Fare'].median())

#Gender
gender=np.array([])
for x in train['Sex']:
    if x=='male':
        gender=np.append(gender,0)
    else:
        gender=np.append(gender,1)

train['Sex']=gender

#Embarked
embarked=np.array([])
for x in train['Embarked']:
    if x=='S':
        embarked=np.append(embarked,0)
    elif x=='C':
        embarked=np.append(embarked,1)
    else:
        embarked=np.append(embarked,2)

train['Embarked']=embarked


X_train=train

#Age
count=0
age_mean=0
for x in X_train['Age']:
    if math.isnan(x)==False:
        age_mean=age_mean+x
        count+=1

age_mean=age_mean/count

age=np.array([])
for x in X_train['Age']:
    if math.isnan(x):
        age=np.append(age,age_mean)
    else:
        age=np.append(age,x)

X_train['Age']=age

#x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=0)
#for test data

#Fare
fare=np.array([])
for x in X_test['Fare']:
    if math.isnan(x):
        fare=np.append(fare,X_test['Fare'].median())
    else:
        fare=np.append(fare,x)

X_test['Fare']=fare

#Gender
gender=np.array([])
for x in X_test['Sex']:
    if x=='male':
        gender=np.append(gender,0)
    else:
        gender=np.append(gender,1)

X_test['Sex']=gender

#Embarked
embarked=np.array([])
for x in X_test['Embarked']:
    if x=='S':
        embarked=np.append(embarked,0)
    elif x=='C':
        embarked=np.append(embarked,1)
    else:
        embarked=np.append(embarked,2)

X_test['Embarked']=embarked



#Age
count=0
age_mean=0
for x in X_test['Age']:
    if math.isnan(x)==False:
        age_mean=age_mean+x
        count+=1

age_mean=age_mean/count

age=np.array([])
for x in X_test['Age']:
    if math.isnan(x):
        age=np.append(age,age_mean)
    else:
        age=np.append(age,x)

X_test['Age']=age

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


#dc=DecisionTreeClassifier(max_depth=9).fit(X_train,Y_train)
dc=RandomForestClassifier(n_estimators=10,max_depth=9).fit(X_train,Y_train)

x=np.array(dc.predict(X_test))
print(dc.score(X_test,Y_test))
#plt.scatter(Y_test,x)
#pd.DataFrame({'PassengerId':y,'Survived':x}).to_csv('output.csv',index=False)