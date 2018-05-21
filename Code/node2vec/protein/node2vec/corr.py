from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
import pandas as pd
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression



#X, Y = make_multilabel_classification(random_state=0, n_samples=5,return_indicator=True, n_classes=3)

test = pd.read_csv("mult.csv", header=0);
X = test.iloc[0:2000 , 0:128];
Y = test.iloc[0: 2000, 128:131];
#Y_1 = test.iloc[0:2000, 128:129]
#Y_2 = test.iloc[0: 2000, 129:130]
X_test = test.iloc[2001: 2361 , 0:128];

print("rf")
rf = RandomForestClassifier(random_state=0).fit(X, Y)
print(rf.predict_proba(X))
print("$$$$$")
print(rf.predict(X_test))

print("KNN")
knn = KNeighborsClassifier(n_neighbors = 7).fit(X, Y)
print(knn.predict_proba(X))
print("$$$$$")
print(knn.predict(X_test))

print("LR")
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, Y)
print(lr.predict_proba(X))
print("$$$$$")
print(lr.predict(X_test))
