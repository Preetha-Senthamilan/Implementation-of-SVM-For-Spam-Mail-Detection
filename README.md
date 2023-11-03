# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: PREETHA.S
RegisterNumber: 212222230110 
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
     print('Result output')
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```

## Output:

data.head()

![image](https://github.com/Preetha-Senthamilan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390282/7ae8fa1e-70a6-4f56-8c51-6dd5146162c3)

data.info()

![image](https://github.com/Preetha-Senthamilan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390282/8701a9c9-8dc7-41df-99f7-98091927fb14)


data.isnull().sum()

![image](https://github.com/Preetha-Senthamilan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390282/6253d9ae-98c7-4b78-a457-356fbcbbdec8)


y_pred

![image](https://github.com/Preetha-Senthamilan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390282/870fea94-99cf-4bf8-82d3-b1c295d89e87)

Accuracy

![image](https://github.com/Preetha-Senthamilan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390282/018f6ff1-43e1-43a8-847e-155399b470f1)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
