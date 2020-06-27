import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.inspection import permutation_importance
import xgboost

def data_prep():
    return 1

def feature_rank(model,df,name,top_vars=10):
    tree_importance_sorted_idx = np.flip(np.argsort(model.feature_importances_))
    tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5

    f, ax = plt.subplots(figsize=(12, 8))
    ax.barh(tree_indices,
            model.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax.set_yticklabels(df.columns[tree_importance_sorted_idx])
    ax.set_yticks(tree_indices)
    ax.set_ylim((0,top_vars))
    f.tight_layout()
    f.savefig("Plots/rank20"+name+".png", transparent=True)

    rank = pd.DataFrame(list(zip(df.columns[tree_importance_sorted_idx][:top_vars],
                                     model.feature_importances_[tree_importance_sorted_idx][:top_vars])),
                            columns =['var', 'score'])
    rank.to_csv("Plots/rank20"+name+".csv")


    
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
    y_tot1 = data.pop('class_label')
    y_tot = y_tot1.replace(-1, 0)
    X_train, X_test, y_train, y_test = train_test_split(data, y_tot, test_size = 0.33, random_state=241)
    kf = KFold(n_splits=5, shuffle=True, random_state=241)

    #lr_qual = run_LogReg(X_train,y_train,kf)
    #all_values = bestC(lr_qual)[0]
    #all_C = bestC(lr_qual)[1]
    all_C = 1
    #print ("best val=%.4f, with C=%.2f" % (all_values,all_C))
    #print(y_train.values)
    #model = LogisticRegression(C=all_C, random_state=241)
    #model = AdaBoostClassifier( DecisionTreeClassifier(max_depth=4, max_features='auto', min_samples_split=10, min_samples_leaf=10), n_estimators=100, learning_rate=1)
    model = xgboost.XGBClassifier(tree_method="hist", thread_count=-1)

    model.fit(X_train,y_train) #.values

    score = model.score(X_test,y_test)
    print( 'Scikit score: ', score)
    #print(y_test.values)
    pred = model.predict_proba(X_test)[:, 1]
    print('y_test = ',y_test[:5])
    print('pred = ',pred[:5].round())
    print( classification_report(y_test.values, pred.round()))# .round()  , target_names=["1", "-1"]
    auc = roc_auc_score(y_test.values, pred)
    print( "Area under ROC curve: %.4f"%(auc))

    #print(model.coef_)
    print(model.feature_importances_)
    #print(model.feature_importances_)

    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    print(result.importances_mean)
    for i in result.importances_mean.argsort()[::-1]:
        #if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
            print(f"{X_test.columns[i]:<8}"
                  f"{result.importances_mean[i]:.3f}"
                  f" +/- {result.importances_std[i]:.3f}")
            
    #fig, ax = plt.subplots()
    #ax.boxplot(result.importances[sorted_idx].T,
   #            vert=False, labels=X_test.columns[sorted_idx])
    #ax.set_title("Permutation Importances (test set)")
    #fig.tight_layout()
    #plt.show()
    
    #score_tr = model.evaluate(X_train, y_train)
    # score = model.evaluate(X_test, y_test)
    #print("train ",score_tr)
    # print("test ",score)
    #feature_rank(model,X_test,"AdaBoost")
    
if __name__ == "__main__":
    main()
