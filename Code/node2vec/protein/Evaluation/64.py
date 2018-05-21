from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
import pandas as pd
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import f1_score
import numpy as np

test = pd.read_csv("64vv.csv", header=0);
X = test.iloc[1:6000 , 1:65];
Y = test.iloc[1: 6000, 65:67];
Y_1 = test.iloc[1: 6000, 65:66]
Y_2 = test.iloc[1: 6000, 66:67]
X_test = test.iloc[6001: 9826, 1:65];
y_true =  test.iloc[6001 : 9826, 65:67];
y_svm_1 = test.iloc[6001 : 9826, 65:66];
y_svm_2 = test.iloc[6001 : 9826, 66:67];


print("rf")
rf = RandomForestClassifier(random_state=0).fit(X, Y)
print(rf.predict_proba(X))
print("$$$$$")
y_pred = rf.predict(X_test)
print(y_pred)
print("mean_squared_log_error")
print(mean_squared_log_error(y_true, y_pred))
print("hamming loss")
y_pred1=np.array(y_pred)
y_true1=np.array(y_true)
print(np.sum(np.not_equal(y_true1, y_pred1))/float(y_true1.size))

print("KNN")
knn = KNeighborsClassifier(n_neighbors = 7).fit(X, Y)
print(knn.predict_proba(X))
print("$$$$$")
y_pred = knn.predict(X_test)
#print(y_pred)
print("mean_squared_log_error")
print(mean_squared_log_error(y_true, y_pred))
print("hamming loss")
y_pred1=np.array(y_pred)
y_true1=np.array(y_true)
print(np.sum(np.not_equal(y_true1, y_pred1))/float(y_true1.size))

print("DTC")
dtc = DecisionTreeClassifier(random_state=0).fit(X, Y)
print(dtc.predict_proba(X))
print("$$$$$")
y_pred = dtc.predict(X_test)
#print(y_pred)
print("mean_squared_log_error")
print(mean_squared_log_error(y_true, y_pred))
print("hamming loss")
y_pred1=np.array(y_pred)
y_true1=np.array(y_true)
print(np.sum(np.not_equal(y_true1, y_pred1))/float(y_true1.size))

print("SupportVectorMachine")
model = svm.SVC(kernel='rbf', C=1000,gamma=100);
model.fit(X, Y_1);
#print(model.score(X, Y_1))
y_pred_svm_1 = model.predict(X_test);
mod1 = model.score(X_test, y_svm_1)

#print("----------evaluation-------------")
#print("macrof1 score")
macrof1 = f1_score(y_svm_1, y_pred_svm_1 , average='macro')
#print(macrof1)
#print("microf1 score")
microf1 = f1_score(y_svm_1, y_pred_svm_1, average='micro')
#print(microf1)
#print("weightedf1 score")
weightedf1 = f1_score(y_svm_1, y_pred_svm_1, average='weighted') 
#print(weightedf1)
#print("f1 score")
f1 = f1_score(y_svm_1, y_pred_svm_1, average=None)
print(f1)



model = svm.SVC(kernel='rbf', C=1000,gamma=100);
model.fit(X, Y_2);
#print(model.score(X, Y_2))
y_pred_svm_2 = model.predict(X_test);
mod2 = model.score(X_test, y_svm_2)


print("----------evaluation-------------")
#print("macrof1 score")
macrof12 = f1_score(y_svm_2, y_pred_svm_2 , average='macro')
#print(macrof12)
#print("microf1 score")
microf12 = f1_score(y_svm_2, y_pred_svm_2, average='micro')
#print(microf12)
#print("weightedf1 score")
weightedf12 = f1_score(y_svm_2, y_pred_svm_2, average='weighted') 
#print(weightedf12)
#print("f1 score")
f12 = f1_score(y_svm_2, y_pred_svm_2, average=None)
#print(f12)

print("average model score")
m = float(mod1) + float(mod2)
print(m/2) 

print("average macrof1 score")
one = float(macrof1) + float(macrof12)
print(one/2)

print("average microf1 score")
t = float(microf1) + float(microf12)
print(t/2)

print("average weightedf1 score")
k = float(weightedf1) + float(weightedf12)
print(k/2)