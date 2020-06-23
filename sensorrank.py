import numpy as np
import pandas as pd
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score


def data_prep():
    return 1

def run_LogReg(features_fin,Y,kf):
    lr_qual =[]
    for i in range(-5,5):
        C=10**i
        print(C)
        lr = LogisticRegression(C=C, random_state=241)
        start_time = datetime.datetime.now()
        lr_scores = cross_val_score(lr, features_fin, Y.ravel(), cv=kf, scoring='roc_auc')
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
    kf = KFold(n_splits=5, shuffle=True, random_state=241)
    #print(data.head())
    
    lr_qual = run_LogReg(data, y_tot,kf)
    all_values = bestC(lr_qual)[0]
    all_C = bestC(lr_qual)[1]
    print ("best val=%.4f, with C=%.2f" % (all_values,all_C))
    
if __name__ == "__main__":
    main()
