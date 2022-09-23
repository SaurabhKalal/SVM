import pandas as pd
data=pd.read_csv(r"C:\Users\Admin\Desktop\PYTHON\PROJECT\Social_Network_Ads.csv")
print(data.isna().sum())
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop="first"),["Gender"])],remainder="passthrough")
data=CT.fit_transform(data)
X=data[:,1:4]
Y=data[:,-1]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
print(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=23)

from sklearn.svm import SVC
model=SVC(kernel="linear",random_state=0)
model.fit(X_train,Y_train)
model1=SVC(kernel="rbf",random_state=0)
model1.fit(X_train,Y_train)
model2=SVC(kernel="poly",random_state=0)
model2.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
Y_pred1=model1.predict(X_test)
Y_pred2=model2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_pred,Y_test)
cm1=confusion_matrix(Y_pred1,Y_test)
cm2=confusion_matrix(Y_pred2,Y_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(Y_pred,Y_test)
print(score)
score1=accuracy_score(Y_pred1,Y_test)
print(score1)
score2=accuracy_score(Y_pred2,Y_test)
print(score2)
import matplotlib.pyplot as plt
plt.title("Accurary plot")
plt.xlabel("Kernals")
plt.ylabel("Accuracy Score")
plt.bar("linear",score)
plt.bar("rbf",score1)        
plt.bar("poly",score2)
plt.legend()
plt.show()

