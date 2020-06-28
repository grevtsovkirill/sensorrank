import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, roc_curve, auc

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


def main():
    data_path = './'
    filename = 'task_data.csv'
    data = pd.read_csv(data_path+filename, index_col='sample index')
    y_tot1 = data.pop('class_label')
    y_tot = y_tot1.replace(-1, 0)
    X_train, X_test, y_train, y_test = train_test_split(data, y_tot, test_size = 0.33, random_state=242)
    kf = KFold(n_splits=5, shuffle=True, random_state=241)

    model = xgboost.XGBClassifier()
    model.fit(X_train,y_train)

    score = model.score(X_test,y_test)
    print( 'Scikit score: ', score)
    #print(y_test.values)
    pred = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    print('y_test = ',y_test[:5])
    print('pred = ',pred[:5].round())
    print( classification_report(y_test.values, pred.round()))# .round()  , target_names=["1", "-1"]
    auc = roc_auc_score(y_test.values, pred)
    print( "Area under ROC curve: %.4f"%(auc))

    print(confusion_matrix(y_test, predictions))

    # result = permutation_importance(model, X_test, y_test, n_repeats=10,
    #                                 random_state=42, n_jobs=2)
    # sorted_idx = result.importances_mean.argsort()
    # print(result.importances_mean)
    # for i in result.importances_mean.argsort()[::-1]:
    #     #if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
    #         print(f"{X_test.columns[i]:<8}"
    #               f"{result.importances_mean[i]:.3f}"
    #               f" +/- {result.importances_std[i]:.3f}")
            
    #fig, ax = plt.subplots()
    #ax.boxplot(result.importances[sorted_idx].T,
   #            vert=False, labels=X_test.columns[sorted_idx])
    #ax.set_title("Permutation Importances (test set)")
    #fig.tight_layout()
    #plt.show()
    
    
if __name__ == "__main__":
    main()
