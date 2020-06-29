import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score,  auc, plot_confusion_matrix
from sklearn.inspection import permutation_importance
import xgboost
from xgboost import plot_importance

#import eli5
#from eli5.sklearn import PermutationImportance

def data_prep():
    return 1

def feature_rank(model,df,name,top_vars=10):
    result = model.feature_importances_
    #tree_importance_sorted_idx = np.flip(np.argsort(result))
    tree_importance_sorted_idx = np.argsort(result)
    tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5
    for i in result.argsort()[::-1]:
        #if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
        print(f"{df.columns[i]:<8}"
              f"{result[i]:.6f}")

              #f" +/- {result.importances_std[i]:.6f}")

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

def dropcol_importances(model, X_train, y_train, X_test, y_test):
    model_ = model
    model_.random_state = 999
    model_.fit(X_train, y_train)
    baseline = model_.score(X_test, y_test)
    #print("baseline = ",baseline)
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        Xt = X_test.drop(col, axis=1)
        model_ = model
        model_.random_state = 999
        model_.fit(X, y_train)
        o = model_.score(Xt, y_test)
        #print("col: ",col," score = ",o)
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
        data={'Feature':X_train.columns,
            'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I

def main():
    data_path = './'
    filename = 'task_data.csv'
    data = pd.read_csv(data_path+filename, index_col='sample index')
    y_tot1 = data.pop('class_label')
    y_tot = y_tot1.replace(-1, 0)

    X_train, X_test, y_train, y_test = train_test_split(data, y_tot, test_size = 0.33, random_state=42)
    model = xgboost.XGBClassifier(objective='binary:logistic')
    model.fit(X_train,y_train)

    pred = model.predict_proba(X_test)[:, 1]
    predictionsT = model.predict(X_train)
    predictions = model.predict(X_test)
    print( classification_report(y_test.values, pred.round()))
    auc = roc_auc_score(y_test.values, pred)
    print( "Area under ROC curve: %.4f"%(auc))
    print('test')
    print(confusion_matrix(y_test, predictions))

    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig("Plots/conf_m.png", transparent=True)


    m1 = xgboost.XGBClassifier(objective='binary:logistic') 
    impdrop = dropcol_importances(m1 ,X_train,y_train,X_test,y_test)
    print(impdrop)
    #result = permutation_importance(model, X_test, y_test, n_repeats=100, random_state=41)
    result = permutation_importance(model, data, y_tot, n_repeats=100, random_state=41)
    #result = permutation_importance(model, X_train, y_train, n_repeats=1000, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    #print(result.importances_mean)
    for i in result.importances_mean.argsort()[::-1]:
        #if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
        print(f"{X_test.columns[i]:<8}"
              f"{result.importances_mean[i]:.6f}"
              f" +/- {result.importances_std[i]:.6f}")
        

    plot_importance(model)
    #plt.show()
    plt.savefig("Plots/xgb_importance.png", transparent=True)   


    
    # perm = PermutationImportance(model, random_state=1).fit(X_test,y_test)
    # print(perm.results_)
    # print(perm.feature_importances_)
    # eli5.show_weights(perm, feature_names = X_test.columns.tolist())
    
    print("get_fscore = ", model.get_booster().get_fscore() ) 
    imp_types = ["weight","gain","cover","total_gain","total_cover"]
    print("score ")
    for i in range(len(imp_types)):
        imp_vals = model.get_booster().get_score(importance_type=imp_types[i])
        print(imp_types[i], " ", sorted(imp_vals.items(), key=lambda x: x[1], reverse=True))
    
    #fig, ax = plt.subplots()
    #ax.boxplot(result.importances[sorted_idx].T,
   #            vert=False, labels=X_test.columns[sorted_idx])
    #ax.set_title("Permutation Importances (test set)")
    #fig.tight_layout()
    #plt.show()
    #feature_rank(model,X_test,"feature_importances")
    
    
if __name__ == "__main__":
    main()
