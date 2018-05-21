import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import f1_score


test = pd.read_csv("128.csv", header=0);

X = test.iloc[0:256 , 0:128];
#print(X)
Y = test.iloc[0: 256, -1];
#print(Y) 
X_test = test.iloc[301: 516 , 0:128];
Y_test = test.iloc[301:516, -1];


print("--------------SupportVectorMachine----------------")
model = svm.SVC(kernel='rbf', C=1000,gamma=100);
model.fit(X, Y);
print(model.score(X, Y))
Y_pred = model.predict(X_test);

print("---------------model score--------------")
print(model.score(X_test, Y_test))
#print(model.predict(test.iloc[514: 516, 0:256 ]))

print("------------evaluation-------------")
print("macrof1 score")
print(f1_score(Y_test, Y_pred , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_pred, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_pred, average='weighted')  )

print("mean_squared_log_error")
print(mean_squared_log_error(Y_test, Y_pred))
print("hamming loss")
Y_pred1=np.array(Y_pred)
Y_test1=np.array(Y_test)
print(np.sum(np.not_equal(Y_test1, Y_pred1))/float(Y_test1.size))


print("-----------------RandomForestClassifier---------------")
modelrf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None,min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None);
modelrf.fit(X, Y);
print(modelrf.score(X, Y))
Y_predrf = modelrf.predict(X_test);

print("-------------model score-------------")
print(modelrf.score(X_test, Y_test))
#print(modelrf.predict(test.iloc[514: 516, 0:256 ]))

print("--------------evaluation--------------")
print("macrof1 score")
print(f1_score(Y_test, Y_predrf , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_predrf, average='micro'))
print("weightedf1 score")
print(f1_score(Y_test, Y_predrf, average='weighted'))

print("mean_squared_log_error")
print(mean_squared_log_error(Y_test, Y_pred))
print("hamming loss")
Y_pred1=np.array(Y_pred)
Y_test1=np.array(Y_test)
print(np.sum(np.not_equal(Y_test1, Y_pred1))/float(Y_test1.size))


print("--------------KNeighborsClassifier------------")
modelk = KNeighborsClassifier(n_neighbors = 7);
modelk.fit(X, Y);
print(modelk.score(X, Y))
Y_predkn = modelk.predict(X_test);

print("------------model score---------")
print(modelk.score(X_test, Y_test))
#print(modelk.predict(test.iloc[514: 516, 0:256]))

print("------------evaluation---------------")
print("macrof1 score")
print(f1_score(Y_test, Y_predkn , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_predkn, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_predkn, average='weighted')  )

print("mean_squared_log_error")
print(mean_squared_log_error(Y_test, Y_pred))
print("hamming loss")
Y_pred1=np.array(Y_pred)
Y_test1=np.array(Y_test)
print(np.sum(np.not_equal(Y_test1, Y_pred1))/float(Y_test1.size))


print("---------------GuassienNaiveBayes---------------")
modelgn = GaussianNB();
modelgn.fit(X, Y);
print(modelgn.score(X, Y))
Y_prednb = modelgn.predict(X_test);

print("---------model score--------")
print(modelgn.score(X_test, Y_test))
#print(modelgn.predict(test.iloc[514: 516, 0:256]))

print("--------------evaluation----------")
print("macrof1 score")
print(f1_score(Y_test, Y_prednb , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_prednb, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_prednb, average='weighted')  )

print("mean_squared_log_error")
print(mean_squared_log_error(Y_test, Y_pred))
print("hamming loss")
Y_pred1=np.array(Y_pred)
Y_test1=np.array(Y_test)
print(np.sum(np.not_equal(Y_test1, Y_pred1))/float(Y_test1.size))

