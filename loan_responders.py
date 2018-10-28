import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir("F:\Rohan\data science\python")
bank=pd.read_csv("PL_XSELL.csv")
bank.head()
bank.info()
bank.TARGET.value_counts()
bank['FLG_HAS_CC']=bank['FLG_HAS_CC'].astype(str)
bank['FLG_HAS_ANY_CHGS']=bank['FLG_HAS_ANY_CHGS'].astype(str)
bank['FLG_HAS_OLD_LOAN']=bank['FLG_HAS_OLD_LOAN'].astype(str)

bank1=list(bank.columns)
bank1.remove('CUST_ID')
bank1.remove('random')
bank1.remove('NO_OF_L_CR_TXNS')
bank1.remove('NO_OF_L_DR_TXNS')
bank1.remove('AMT_ATM_DR')
bank1.remove('AMT_BR_CSH_WDL_DR')
bank1.remove('AMT_CHQ_DR')
bank1.remove('AMT_NET_DR')
bank1.remove('AMT_MOB_DR')
bank1.remove('TARGET')
bank1.remove('ACC_OP_DATE')
bank1.remove('FLG_HAS_NOMINEE')
bank1.remove('NO_OF_BR_CSH_WDL_DR_TXNS')
bank1.remove('NO_OF_ATM_DR_TXNS')
bank1.remove('NO_OF_NET_DR_TXNS')
bank1.remove('NO_OF_MOB_DR_TXNS')
bank1.remove('NO_OF_CHQ_DR_TXNS')
bank2 = pd.get_dummies(bank[bank1],drop_first=True )
len(bank2.columns)
bank2.head().T
from sklearn.cross_validation import train_test_split
y=bank.TARGET
x=bank2
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "gini" , 
                             min_samples_split = 100,
                             min_samples_leaf = 10,
                             max_depth = 50)
clf.fit(x_train,y_train)
import pydot
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file = None)
graph=graphviz.Source(dot_data)
graph.render("Bank")
Nodes = pd.DataFrame(clf.tree_.__getstate__()["nodes"])
Nodes
feature_importance = pd.DataFrame([x_train.columns,
                               clf.tree_.compute_feature_importances()])
feature_importance.T
pred_y_train = clf.predict(x_train )
pred_y_train
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
score = accuracy_score(y_train, pred_y_train)
score
y_train_prob = clf.predict_proba(x_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc(fpr, tpr)
pred_y_test = clf.predict(x_test)
score_h = accuracy_score(y_test, pred_y_test)
score_h
y_test_prob = clf.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc(fpr, tpr)
y_freq = np.bincount(y_train)
y_val = np.nonzero(y_freq)[0]
np.vstack((y_val,y_freq[y_val])).T
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(clf, x_train , y_train, cv = 10, scoring='roc_auc')
scores.mean()
scores.std()
from sklearn.grid_search import GridSearchCV
param_dist = {"criterion": ["gini","entropy"],
              "max_depth": np.arange(3,10),
              }

tree = DecisionTreeClassifier(min_samples_split = 100,
                             min_samples_leaf = 10)

tree_cv  = GridSearchCV(tree, param_dist, cv = 10, 
                        scoring = 'roc_auc', verbose = 100)

tree_cv.fit(x_train,y_train)
print("Tuned Decision Tree parameter : {}".format(tree_cv.best_params_))
classifier = tree_cv.best_estimator_

classifier.fit(x_train,y_train)
y_train_prob = classifier.predict_proba(x_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc_d = auc(fpr, tpr)
auc_d
y_test_prob = classifier.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc_h = auc(fpr, tpr)
auc_h
train=pd.concat([x_train,y_train],axis=1)

Prediction = classifier.predict_proba(x_train)
train["prob_score"] = Prediction[:,1]
def deciles(x):
    decile = pd.Series(index=[0,1,2,3,4,5,6,7,8,9])
    for i in np.arange(0.1,1.1,0.1):
        decile[int(i*10)]=x.quantile(i)
    def z(x):
        if x<decile[1]: return(1)
        elif x<decile[2]: return(2)
        elif x<decile[3]: return(3)
        elif x<decile[4]: return(4)
        elif x<decile[5]: return(5)
        elif x<decile[6]: return(6)
        elif x<decile[7]: return(7)
        elif x<decile[8]: return(8)
        elif x<decile[9]: return(9)
        elif x<=decile[10]: return(10)
        else:return(np.NaN)
    s=x.map(z)
    return(s)
    
def Rank_Ordering(X,y,TARGET):
   X['decile']=deciles(X[y])
   Rank=X.groupby('decile').apply(lambda x: pd.Series([
        np.min(x[y]),
        np.max(x[y]),
        np.mean(x[y]),
        np.size(x[y]),
        np.sum(x[TARGET]),
        np.size(x[TARGET][x[TARGET]==0]),
        ],
        index=(["min_resp","max_resp","avg_resp",
                "cnt","cnt_resp","cnt_non_resp"])
        )).reset_index()
   Rank = Rank.sort_values(by='decile',ascending=False)
   Rank["rrate"] = Rank["cnt_resp"]*100/Rank["cnt"]
   Rank["cum_resp"] = np.cumsum(Rank["cnt_resp"])
   Rank["cum_non_resp"] = np.cumsum(Rank["cnt_non_resp"])
   Rank["cum_resp_pct"] = Rank["cum_resp"]/np.sum(Rank["cnt_resp"])
   Rank["cum_non_resp_pct"]=Rank["cum_non_resp"]/np.sum(Rank["cnt_non_resp"])
   Rank["KS"] = Rank["cum_resp_pct"] - Rank["cum_non_resp_pct"]
   Rank
   return(Rank)
   
   Rank = Rank_Ordering(train,"prob_score","TARGET")
Rank
test=pd.concat([x_test,y_test],axis=1)
Prediction_h = classifier.predict_proba(x_test)
test["prob_score"] = Prediction_h[:,1]
Rank_h = Rank_Ordering(test,"prob_score","TARGET")
Rank_h