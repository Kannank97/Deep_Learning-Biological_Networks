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



test = pd.read_csv("output.csv", header=0);

X = test.iloc[0:5500 , 1:256];
#print(X)
Y = test.iloc[0: 5500, -1];
#print(Y) 
X_test = test.iloc[5501: 8507 , 1:256];
Y_test = test.iloc[5501:8507, -1];


print("-----------SupportVectorMachine-----------------")
model = svm.SVC(kernel='rbf', C=1000,gamma=100);
model.fit(X, Y);
print(model.score(X, Y))
Y_pred = model.predict(X_test);

print("--------model score-----------")
print(model.score(X_test, Y_test))
print(model.predict(test.iloc[8505: 8507, 1:256 ]))

print("----------evaluation-------------")
print("macrof1 score")
print(f1_score(Y_test, Y_pred , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_pred, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_pred, average='weighted')  )
print("f1 score")
print(f1_score(Y_test, Y_pred, average=None))



print("-----------------RandomForestClassifier---------------")
modelrf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None,min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None);
modelrf.fit(X, Y);
print(modelrf.score(X, Y))
Y_predrf = modelrf.predict(X_test);

print("--------model score----------")
print(modelrf.score(X_test, Y_test))
print(modelrf.predict(test.iloc[8505: 8507, 1:256 ]))

print("--------------evaluation--------------")
print("macrof1 score")
print(f1_score(Y_test, Y_predrf , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_predrf, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_predrf, average='weighted')  )
print("f1 score")
print(f1_score(Y_test, Y_predrf, average=None))


print("--------------KNeighborsClassifier---------------")
modelk = KNeighborsClassifier(n_neighbors = 7);
modelk.fit(X, Y);
print(modelk.score(X, Y))
Y_predkn= modelk.predict(X_test);

print("------------model score---------------")
print(modelk.score(X_test, Y_test))
print(modelk.predict(test.iloc[8505: 8507, 1:256]))

print("------------evaluation---------------")
print("macrof1 score")
print(f1_score(Y_test, Y_predkn , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_predkn, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_predkn, average='weighted')  )
print("f1 score")
print(f1_score(Y_test, Y_predkn, average=None))


print("--------------------GuassienNaiveBayes-----------------------")
modelgn = GaussianNB();
modelgn.fit(X, Y);
print(modelgn.score(X, Y))
Y_prednb= modelgn.predict(X_test);

print("-----------model score-----------------")
print(modelgn.score(X_test, Y_test))
print(modelgn.predict(test.iloc[8505: 8507, 1:256]))

print("--------------evaluation----------")
print("macrof1 score")
print(f1_score(Y_test, Y_prednb , average='macro'))
print("microf1 score")
print(f1_score(Y_test, Y_prednb, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test, Y_prednb, average='weighted')  )
print("f1 score")
print(f1_score(Y_test, Y_prednb, average=None))


testa = pd.read_csv("output_arch.csv", header=0);

X_a = testa.iloc[0:5500 , 1:128];
#print(X)
Y_a = testa.iloc[0: 5500, -1];
#print(Y) 
X_test_a = testa.iloc[5501: 9441 , 1:128];
Y_test_a = testa.iloc[5501:9441, -1];


print("-----------SupportVectorMachine-----------------")
modela = svm.SVC(kernel='rbf', C=1000,gamma=100);
modela.fit(X_a, Y_a);
print(modela.score(X_a, Y_a))
Y_pred_a = modela.predict(X_test_a);

print("--------model score-----------")
print(modela.score(X_test_a, Y_test_a))
print(modela.predict(test.iloc[8505: 8507, 1:128 ]))

print("----------evaluation-------------")
print("macrof1 score")
print(f1_score(Y_test_a, Y_pred_a , average='macro'))
print("microf1 score")
print(f1_score(Y_test_a, Y_pred_a, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test_a, Y_pred_a, average='weighted')  )
print("f1 score")
print(f1_score(Y_test_a, Y_pred_a, average=None))



print("-----------------RandomForestClassifier---------------")
modelrfa = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None,min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None);
modelrfa.fit(X_a, Y_a);
print(modelrf.score(X_a, Y_a))
Y_predrfa = modelrfa.predict(X_test_a);

print("--------model score----------")
print(modelrfa.score(X_test_a, Y_test_a))
print(modelrfa.predict(testa.iloc[8505: 8507, 1:128 ]))

print("--------------evaluation--------------")
print("macrof1 score")
print(f1_score(Y_test_a, Y_predrfa , average='macro'))
print("microf1 score")
print(f1_score(Y_test_a, Y_predrfa, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test_a, Y_predrfa, average='weighted')  )
print("f1 score")
print(f1_score(Y_test_a, Y_predrfa, average=None))


print("--------------KNeighborsClassifier---------------")
modelka = KNeighborsClassifier(n_neighbors = 7);
modelka.fit(X_a, Y_a);
print(modelka.score(X_a, Y_a))
Y_predkna= modelka.predict(X_test_a);

print("------------model score---------------")
print(modelka.score(X_test_a, Y_test_a))
print(modelka.predict(testa.iloc[8505: 8507, 1:128]))

print("------------evaluation---------------")
print("macrof1 score")
print(f1_score(Y_test_a, Y_predkna , average='macro'))
print("microf1 score")
print(f1_score(Y_test_a, Y_predkna, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test_a, Y_predkna, average='weighted')  )
print("f1 score")
print(f1_score(Y_test_a, Y_predkna, average=None))


print("--------------------GuassienNaiveBayes-----------------------")
modelgna = GaussianNB();
modelgna.fit(X_a, Y_a);
print(modelgna.score(X_a, Y_a))
Y_prednba= modelgna.predict(X_test_a);

print("-----------model score-----------------")
print(modelgna.score(X_test_a, Y_test_a))
print(modelgna.predict(testa.iloc[8505: 8507, 1:128]))

print("--------------evaluation----------")
print("macrof1 score")
print(f1_score(Y_test_a, Y_prednba , average='macro'))
print("microf1 score")
print(f1_score(Y_test_a, Y_prednba, average='micro') )
print("weightedf1 score")
print(f1_score(Y_test_a, Y_prednba, average='weighted')  )
print("f1 score")
print(f1_score(Y_test_a, Y_prednba, average=None))

