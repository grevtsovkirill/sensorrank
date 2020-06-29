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

from eli5.sklearn import PermutationImportance

debug = False
def data_prep():
    return 1


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
    cols = X_train.columns
    return imp,cols

def perf_out(imp, cols,name):
    print(name)
    I = pd.DataFrame(
        data={'Feature':cols,
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
    feat_cols = data.columns
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


    # drop column 
    m1 = xgboost.XGBClassifier(objective='binary:logistic') 
    im,co = dropcol_importances(m1 ,X_train,y_train,X_test,y_test)
    impdrop = perf_out(im,co,"Drop Column")
    print(impdrop)


    # Permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=100, random_state=41)
    impper = perf_out(result.importances_mean,X_test.columns,"Permutation Importance")
    print(impper)

    # Permutation importance v2
    perm = PermutationImportance(model, random_state=41).fit(X_test,y_test)
    imppereli5 = perf_out(perm.feature_importances_,X_test.columns,"Permutation Importance ELI5")
    print(imppereli5)
    
        
    if debug==True:
        plot_importance(model)
        #plt.show()
        plt.savefig("Plots/xgb_importance.png", transparent=True)   

    imp_types = ["weight","gain","cover","total_gain","total_cover"]
    for i in range(len(imp_types)):
        imp_vals = model.get_booster().get_score(importance_type=imp_types[i])
        imp_vals = sorted(imp_vals.items(), key=lambda x: x[1], reverse=True)
        dftype = pd.DataFrame(imp_vals,columns=['Feature','Importance'])
        cur_feats = dftype['Feature'].values
        diff = set(feat_cols)-set(cur_feats)
        if len(diff)!=0:
            null_imp = [0]*len(diff)
            miss_features = zip(diff, null_imp)
            dftype = dftype.append(pd.DataFrame(miss_features,columns=['Feature','Importance']), ignore_index=True)

        dftype = dftype.set_index('Feature') 
        #print(imp_types[i], " ", imp_vals)
        print("Model-based ranking: ",imp_types[i])
        print(dftype)
        if imp_types[i]=="total_gain": dftype.to_csv("rank.csv")
    
    
if __name__ == "__main__":
    main()
