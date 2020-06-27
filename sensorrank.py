import numpy as np
import pandas as pd
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc

def data_prep():
    return 1

def run_LogReg(features_fin,Y,kf):
    lr_qual =[]
    
    for i in range(-5,5):
        C=10**i
        print(C)
        lr = LogisticRegression(C=C, random_state=241)
        start_time = datetime.datetime.now()
        lr_scores = cross_val_score(lr, features_fin, Y, cv=kf, scoring='roc_auc')
        run_time=datetime.datetime.now() - start_time
        print ('Time elapsed:', run_time)
        print (lr_scores)
        lr_qual.append((i,C,run_time.total_seconds(),np.mean(lr_scores)))
    return lr_qual

def bestC(lr_qual):
    lr_qual.sort(key=lambda x:x[3])
    return max(lr_qual, key=lambda x: x[3])[3], max(lr_qual, key=lambda x: x[3])[1]

def main():
    data_path = './'
    filename = 'task_data.csv'
    data = pd.read_csv(data_path+filename, index_col='sample index')
    y_tot = data.pop('class_label')
    X_train, X_test, y_train, y_test = train_test_split(data, y_tot, test_size = 0.2, random_state=241)
    kf = KFold(n_splits=5, shuffle=True, random_state=241)

    #lr_qual = run_LogReg(X_train,y_train,kf)
    #all_values = bestC(lr_qual)[0]
    #all_C = bestC(lr_qual)[1]
    all_C = 1
    #print ("best val=%.4f, with C=%.2f" % (all_values,all_C))
    print(y_train.values)
    model = LogisticRegression(C=all_C, random_state=241)
    model.fit(X_train,y_train.values)

    score = model.score(X_test,y_test)
    print( 'Scikit score: ', score)
    #print(y_test.values)
    pred = model.predict_proba(X_test)[:, 1]
    print(pred)
    #print( classification_report(y_test.values, pred.round()))# .round()  , target_names=["1", "-1"]
    #auc = roc_auc_score(y_test.values, pred)
    #print( "Area under ROC curve: %.4f"%(auc))
    
    # score_tr = model.evaluate(X_train, y_train)
    # score = model.evaluate(X_test, y_test)
    # print("train ",score_tr)
    # print("test ",score)
if __name__ == "__main__":
    main()
