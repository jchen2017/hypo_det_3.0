# coding=utf-8
#
# Released under MIT License
#
# Copyright (c) 2019, Jinying Chen
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import pickle
import numpy as np
import re
import random
import scipy
import time

import utils
from hypoglycemia_classifier import HypoglycemiaClassifier
from sklearn.base import is_classifier, clone
from sklearn import datasets, neighbors
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, EditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

from scipy import sparse

new_split=0 # 1: create new data split  0: using existing data split

def better_inv_dist(dist):
    c = 1.
    return 1. / (c + dist)

def topn_PRF(labels, preds, rank_cutoff):
    predL=zip(preds,labels)
    sorted_predL=sorted(predL, key=lambda x: x[0], reverse=True)
    total=len(labels)
    TP=0
    pos=sum(labels)
    neg=total-pos
    
    print ("total %d examples (%d pos and %d neg)"%(total,pos,neg))
    for (pred,label) in sorted_predL[0:rank_cutoff]:
        if label == 1:
            TP+=1
            
    Prec=1.0*TP/rank_cutoff
    Recall=1.0*TP/pos
    if Prec+Recall == 0:
        F=0.0
    else:
        F=2*Prec*Recall/(Prec+Recall)
        
    return(Prec,Recall,F)

def upsampling (skf_split, X, Y, Y_rules, use_rule, meth, rpl):
    print (meth, rpl)
    us_tr_dict={}
    
    if rpl == "r":
        rpl=True
    else:
        rpl=False

    foldnum=0
    for train_idx, test_idx in skf_split:
        us_tr_dict[foldnum]={}
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        Y_train_rules=Y_rules[train_idx]

        if use_rule:
            X_train=X_train_cv[Y_train_rules != 0]
            Y_train=Y_train_cv[Y_train_rules != 0]
        else:
            X_train=X_train_cv
            Y_train=Y_train_cv

        if meth == "ros":
            ros = RandomOverSampler()
            X_resampled, y_resampled = ros.fit_sample(X_train, Y_train)
        elif meth == "smote":
            X_resampled, y_resampled = SMOTE().fit_sample(X_train, Y_train)
        elif meth == "adasyn":
            X_resampled, y_resampled = ADASYN().fit_sample(X_train, Y_train)
        elif meth == "smoteb1":
            X_resampled, y_resampled = SMOTE(kind="borderline1").fit_sample(X_train, Y_train)
        elif meth == "smoteb2":
            X_resampled, y_resampled = SMOTE(kind="borderline2").fit_sample(X_train, Y_train)
        elif meth == "smotesvm":
            X_resampled, y_resampled = SMOTE(kind="svm").fit_sample(X_train, Y_train)
        elif meth == "smoteenn":
            X_resampled, y_resampled = SMOTEENN().fit_sample(X_train, Y_train)
        elif meth == "smotetomek":
            X_resampled, y_resampled = SMOTETomek().fit_sample(X_train, Y_train)
            
        us_tr_dict[foldnum]=(sparse.csr_matrix(X_resampled),y_resampled)
        foldnum+=1
            
    return us_tr_dict

def downsampling (skf_split, X, Y, Y_rules, use_rule, meth, rpl):
    ds_tr_dict={}
    
    if rpl == "r" :
        rpl=True
    else:
        rpl=False

    foldnum=0
    
    for train_idx, test_idx in skf_split:
        ds_tr_dict[foldnum]={}
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        Y_train_rules=Y_rules[train_idx]
    
        if use_rule:
            X_train=X_train_cv[Y_train_rules != 0]
            Y_train=Y_train_cv[Y_train_rules != 0]
            
        else:
            X_train=X_train_cv
            Y_train=Y_train_cv

        if meth == "rule":
            X_resampled=X_train_cv[Y_train_rules != 0]
            y_resampled=Y_train_cv[Y_train_rules != 0]
                        
        elif meth == "rus":
            rus = RandomUnderSampler(replacement=rpl)
            X_resampled, y_resampled = rus.fit_sample(X_train, Y_train)

        elif meth == "tomek":
            X_resampled, y_resampled = TomekLinks().fit_sample(X_train, Y_train)

        elif meth == "cc":
            X_resampled, y_resampled = ClusterCentroids().fit_sample(X_train, Y_train)

        elif meth == "enn":
            X_resampled, y_resampled = EditedNearestNeighbours().fit_sample(X_train, Y_train)
        elif meth == "allknn":
            X_resampled, y_resampled = AllKNN().fit_sample(X_train, Y_train)
        elif meth == "iht":
            X_resampled, y_resampled = InstanceHardnessThreshold().fit_sample(X_train, Y_train)
        elif meth == "nm3":
            X_resampled, y_resampled = NearMiss().fit_sample(X_train, Y_train)
             
         
        
        ds_tr_dict[foldnum]=(sparse.csr_matrix(X_resampled), y_resampled)

        foldnum+=1
        
    return ds_tr_dict


def downsampling_for_af (skf_split, Y, repeat_num):
    ds_dict={}
    ds_dict["train"]={}
    ds_dict["test"]={}
    ds_dict["train"]["cv"]={}
    ds_dict["test"]["cv"]={}
    
    fold_num=0
    Y_test=[]
    Y_train=[]
    for train_idx, test_idx in skf_split:
        Y_test_cv = Y[test_idx]
        Y_train_cv = Y[train_idx]
        total=len(Y_test_cv)
        pos=sum(Y_test_cv)
        neg=total-pos
        Y_test += list(Y_test_cv)
        Y_train += list(Y_train_cv)
                
        if neg > pos:  # down sampling neg examples
            neg_idx=[]
            pos_idx=[]
        
            for i in range(0,len(Y_test_cv)):
                if Y_test_cv[i] == 0:
                    neg_idx.append(i)
                else:
                    pos_idx.append(i)
        
            ds_dict["test"]["cv"][fold_num]={}
            for i in range(0,repeat_num):
                sel_neg_idx=np.random.choice(neg_idx,len(pos_idx),replace=False)
                total_idx=pos_idx+list(sel_neg_idx)
                ds_dict["test"]["cv"][fold_num][i]=total_idx

        
        total=len(Y_train_cv)
        pos=sum(Y_train_cv)
        neg=total-pos

        if neg > pos:
            neg_idx=[]
            pos_idx=[]
                
            for i in range(0,len(Y_train_cv)):
                if Y_train_cv[i] == 0:
                    neg_idx.append(i)
                else:
                    pos_idx.append(i)

            ds_dict["train"]["cv"][fold_num]={}
            for i in range(0,repeat_num):
                sel_neg_idx=np.random.choice(neg_idx,len(pos_idx),replace=False)
                total_idx=pos_idx+list(sel_neg_idx)
                ds_dict["train"]["cv"][fold_num][i]=total_idx
        
        fold_num+=1
        
    total=len(Y_test)
    pos=sum(Y_test)
    neg=total-pos

    if neg > pos:
        neg_idx=[]
        pos_idx=[]
            
        for i in range(0,len(Y_test)):
            if Y_test[i] == 0:
                neg_idx.append(i)
            else:
                pos_idx.append(i)

        ds_dict["test"]["full"]={}
        for i in range(0,repeat_num):
            sel_neg_idx=np.random.choice(neg_idx,pos,replace=False)
            total_idx=pos_idx+list(sel_neg_idx)
            ds_dict["test"]["full"][i]=total_idx

    total=len(Y_train)
    pos=sum(Y_train)
    neg=total-pos

    if neg > pos:
        neg_idx=[]
        pos_idx=[]

        for i in range(0,len(Y_train)):
            if Y_train[i] == 0:
                neg_idx.append(i)
            else:
                pos_idx.append(i)
            
        ds_dict["train"]["full"]={}
        for i in range(0,repeat_num):
            sel_neg_idx=np.random.choice(neg_idx,pos,replace=False)
            total_idx=pos_idx+list(sel_neg_idx)
            ds_dict["train"]["full"][i]=total_idx

    return ds_dict

        
def adjusted_F_score (y_test, y_hat):
    repeat_num=100
    adjusted_f_scores=[]
    total=len(y_test)
    pos=sum(y_test)
    neg=total-pos
    if neg > pos:  # down sampling neg examples
        neg_idx=[]
        pos_idx=[]
        for i in range(0,len(y_test)):
            if y_test[i] == 0:
                neg_idx.append(i)
            else:
                pos_idx.append(i)

        for i in range(0,repeat_num):
            sel_neg_idx=np.random.choice(neg_idx,pos,replace=False)
            total_idx=pos_idx+list(sel_neg_idx)
            f1=f1_score(np.array(y_test)[total_idx], np.array(y_hat)[total_idx])
            adjusted_f_scores.append(f1)
        f1=np.mean(adjusted_f_scores)
    else:
        f1=f1_score(y_test,y_hat)
        
    return f1


def adjusted_F_score_v2 (y_test, y_hat, ds):
    adjusted_f_scores=[]
    for i in ds:
        f1=f1_score(np.array(y_test)[ds[i]], np.array(y_hat)[ds[i]])
        adjusted_f_scores.append(f1)
    #print ("adjusted f scores: ", adjusted_f_scores)
    f1=np.mean(adjusted_f_scores)
    return f1

    
def evaluate_ranking (clf,model,modeltype,X_test,y_test,n_train,use_rule_test,y_test_rules):
    if modeltype == "linearsvc" or modeltype == "svm":
        pred_test=clf.decision_function(X_test)
    elif modeltype in ["logit", "rf", "dt", "gnb"] or "knn" in modeltype:
        pred_test_prob=clf.predict_proba(X_test)
        pred_test=pred_test_prob[:,1]
    elif modeltype in ["nc"]:
        pred_test=clf.predict_dist(X_test)

    if use_rule_test:
        if modeltype in ["linearsvc", "svm", "nc"]:
            pred_test[y_test_rules == 0] -= 1000000
        elif modeltype in ["logit", "rf", "dt", "gnb"] or "knn" in modeltype:
            pred_test[y_test_rules == 0] /= 1000000
        
    auc_test = roc_auc_score(y_test, pred_test)
    print ("AUC-ROC (%d training insts): test=%.3f"%(n_train, auc_test))
    total_rank=len(y_test)
    for rank in [5, 10, 20]:
        (Pn,Rn,Fn)=topn_PRF(y_test, pred_test, rank)
        print (("P%d=%.3f R%d=%.3f F%d=%.3f (%d training insts)"%(rank,Pn,rank,Rn,rank,Fn,n_train)))
        
    AP=average_precision_score(y_test,pred_test)
    print ("AveP (%d training insts): test=%.3f"%(n_train, AP))

    '''
    print ("=== system output ===")
    print ("=== inst id,pred,label ===")
    i=0
    for p in pred_test:
    print (i,p,y_test[i])
    i+=1
    '''
    sys.stdout.flush()

    return(auc_test, pred_test)


def auc_roc_scoring_func(clf, X, y):
    if re.search("(SVC)|(SGDClassifier)", type(clf).__name__):
        y_pred=clf.decision_function(X)
        auc_test = roc_auc_score(y, y_pred)
    else:
        y_pred=clf.predict_proba(X)
        auc_test = roc_auc_score(y, y_pred[:,1])
    return auc_test


def F_scoring_func(clf, X, y):
    y_hat=clf.predict(X)
    f1=f1_score(y, y_hat)
    return f1


if '__main__' == __name__:
    args = utils.parse_args()
    print (args.annotations)
    print (args.train)
    print (args.preprocess)
    
    cv_split_file="./split/cv_split.pkl"
    down_sampling_file="./sampling/ds_tr.pkl"
    up_sampling_file="./sampling/us_tr.pkl"
    down_sampling_af_file="./sampling/ds.pkl"
    output_file="./output/out.pkl"
    
    X, Y, Y_rules, ids, X_info = utils.load_data(args.annotations, args.train, args.preprocess)
    print ("total %d features"%(np.shape(X)[1]))
       
    Y = np.array(Y)
    try:
        assert X.shape[0] == Y.shape[0]
    except:
        print (X.shape[0], Y.shape[0])
        exit (0)
        
    Y_rules=np.array(Y_rules)
    ids=np.array(ids)
    X_info=np.array(X_info)

    use_rule=args.rules  # use rules to process training data
    use_rule_test=0
    if args.postprocessing == "r":  # use rules to process test data
        use_rule_test=1
        
    model=args.model
    params=args.param.split(":")

    clf=None
    
    if args.test == None:
        mode="cv"
    else:
        mode=""
    
    if model == "svc":
        kernel=params[0]
        if kernel == "":
            kernel = "rbf"
        
        gamma=params[1]
        if gamma == "":
            gamma = "auto"
        else:
            try:
                gamma = float(gamma)
            except:
                pass

        c=params[2]
        if c == "":
            c = 1.0
        else:
            c=float(c)

        prob=True
        clf = SVC(gamma=gamma,kernel=kernel,probability=prob,class_weight="balanced")

    if model == "linearsvc":
        c=params[0]
        if c == "" or c == "None":
            c = 1.0
        else:
            c=float(c)
            
            
        if mode != "cv":
            clf = LinearSVC(C=c, class_weight="balanced")
        else:
            clf_base=LinearSVC(class_weight="balanced")

            para_grid={'C':[100, 10,1,0.1,0.01]}
            
    
    if model == "rf":
        n_est=params[0]
        if n_est == "":
            n_est= 100
        else:
            n_est= int(n_est)

        if mode != "cv":
            clf = RandomForestClassifier(n_estimators=n_est, class_weight="balanced_subsample")
        else:
            clf_base = RandomForestClassifier(random_state=0)

            para_grid={'n_estimators':[100, 200, 300, 500], 'max_features': ['sqrt'], 'class_weight': ["balanced_subsample"], 'max_depth': [3,5,7]}

                        
    if model == "gnb":
        if mode != "cv":
            clf = GaussianNB()
        else:
            clf_base=GaussianNB()

    if re.search(r'knn', model):
        if mode != "cv":
            clf = neighbors.KNeighborsClassifier()
        else:
            clf_base = neighbors.KNeighborsClassifier()
            #para_grid={'n_neighbors':[5,51,101], 'weights':['uniform', better_inv_dist]}
            para_grid={'n_neighbors':[31,51], 'weights':[better_inv_dist]}
                        
    if model == 'nc':
        if mode != "cv":
            clf = neighbors.NearestCentroid()
        else:
            clf_base=neighbors.NearestCentroid()                                 
            
    if model == "logit":
        if mode != "cv":
            clf = LogisticRegression(class_weight="balanced", C=0.1)
        else:
            clf_base=LogisticRegression(class_weight="balanced")

            para_grid={'C':[100,10,1,0.1,0.01]}
            
    if model == "dt":
        if mode != "cv":
            clf = DecisionTreeClassifier(class_weight="balanced")
        else:
            clf_base=DecisionTreeClassifier(class_weight="balanced")
            para_grid={'max_features': ['sqrt', 'log2']}
        
    if mode == "cv" and not (model in ["logit", "linearsvc", "rf", "dt", "gnb", "nc"] or "knn" in model):
        print ("gridsearch not implemented for %s"%(model))
        exit (0)
        
    start_time = time.time()
    
    if args.test == None:  # cross-validation
        skf_split=[]
        if new_split == 1:
            skf = StratifiedKFold(n_splits=args.split, shuffle=True)
            for train_idx, test_idx in skf.split(X, Y):
                skf_split.append((train_idx,test_idx))

            f=open(cv_split_file, 'wb')
            pickle.dump(skf_split,f)
            f.close()
                                                                                        
        else:
            f=open(cv_split_file, 'rb')
            if sys.version_info[0] < 3:
                skf_split=pickle.load(f)
            else:
                skf_split=pickle.load(f, encoding='latin1')
            f.close()

        ds_dict={}
        us_tr_dict={}
        ds_tr_dict={}
            
        try:
            f=open(down_sampling_af_file, 'rb')
            if sys.version_info[0] < 3:
                ds_dict=pickle.load(f)
            else:
                ds_dict=pickle.load(f, encoding='latin1')
            f.close()
        except:
            print ("create downsampling file!")
            ds_dict=downsampling_for_af(skf_split, Y, 100)
            f=open(down_sampling_af_file, 'wb')
            pickle.dump(ds_dict,f)
            f.close()

            
        if re.search(r"down\.", args.sampling):
            (meth, rpl)=re.search(r"down\.([^ \.]+)\.([^ \.]+)", args.sampling).groups()
            down_sampling_file=re.sub(r"\.pkl", "."+args.preprocess+"."+meth+"."+rpl+"_clean.pkl", down_sampling_file)
            if use_rule:
                down_sampling_file=re.sub(r"\.pkl", "."+args.preprocess+"."+meth+"."+rpl+"_rule_clean.pkl", down_sampling_file)
                
            try:
                f=open(down_sampling_file, 'rb')
                if sys.version_info[0] < 3:
                    ds_tr_dict=pickle.load(f)
                else:
                    ds_tr_dict=pickle.load(f, encoding='latin1')
                f.close()
            except:
                print ("create downsampling file!")
                ds_tr_dict=downsampling(skf_split, X, Y, Y_rules, use_rule, meth, rpl)
                f=open(down_sampling_file, 'wb')
                pickle.dump(ds_tr_dict,f)
                f.close()

                            
        if re.search(r"up\.", args.sampling):
            (meth, rpl)=re.search(r"up\.([^ \.]+)\.([^ \.]+)", args.sampling).groups()
            up_sampling_file=re.sub(r"\.pkl", "."+args.preprocess+"."+meth+"."+rpl+"_clean.pkl", up_sampling_file)
            if use_rule:
                up_sampling_file=re.sub(r"\.pkl", "."+args.preprocess+"."+meth+"."+rpl+"_rule_clean.pkl", up_sampling_file)
                
                        
            try:
                f=open(up_sampling_file, 'rb')
                if sys.version_info[0] < 3:
                    us_tr_dict=pickle.load(f)
                else:
                    us_tr_dict=pickle.load(f, encoding='latin1')
                f.close()
            except:
                print ("create upsampling file!")
                us_tr_dict=upsampling(skf_split, X, Y, Y_rules, use_rule, meth, rpl)
                f=open(up_sampling_file, 'wb')
                pickle.dump(us_tr_dict,f)
                f.close()
            

        aucs= []
        f1s = []
        af1s = []
        y_true_ls =[]
        y_hat_ls =[]
        y_prob_ls = []
        id_ls=[]
        foldnum = 0
        result_dict={}
        result_dict["pred_l"]={}
        result_dict["pred_prob"]={}
        result_dict["gold_l"]={}

        for train_idx, test_idx in skf_split:
            print ("train_idx: ", train_idx)
            print ("test_idx: ", test_idx)
            X_train_cv = X[train_idx]
            Y_train_cv = Y[train_idx]
            Y_train_cv_rules = Y_rules[train_idx]

            X_test_cv = X[test_idx]
            Y_test_cv = Y[test_idx]
            Y_test_cv_rules = Y_rules[test_idx]
            id_test_cv=ids[test_idx]
            X_test_info_cv = X_info[test_idx]
            id_ls+=list(id_test_cv)

            X_train=X_train_cv
            y_train=Y_train_cv
            y_train_rules=Y_train_cv_rules

            X_test=X_test_cv
            y_test=Y_test_cv
            y_test_rules=Y_test_cv_rules

            result_dict["gold_l"][foldnum]=y_test
            result_dict["pred_l"][foldnum]=None
            result_dict["pred_prob"][foldnum]=None
                                                
            if re.search(r"down\.", args.sampling):
                if foldnum in ds_tr_dict:
                    try:
                        (X_train,y_train)=ds_tr_dict[foldnum]
                    except:
                        print ("fold %d does not exist in ds_tr_dict"%(foldnum))
                        exit (0)
                    

            elif re.search(r"up\.", args.sampling):
                if foldnum in us_tr_dict:
                    try:
                        (X_train, y_train)=us_tr_dict[foldnum]
                    except:
                        print ("fold %d does not exist in us_tr_dict"%(foldnum))
                        exit (0)

            else:
                if use_rule:
                    X_train=X_train_cv[y_train_rules != 0]
                    y_train=Y_train_cv[y_train_rules != 0]

                    
            cv=10  # using cv to find optimal params for each fold 
            if model in ['gnb', 'nc']: # or "knn" in model:
                cv=1
            
            y_train_pos=sum(y_train)
            y_train_neg=len(y_train) - y_train_pos
            min_class=min(y_train_pos,y_train_neg)
            if min_class < cv:
                cv_new = min(5,min_class)
                print ("warning-1: %d pos and %d neg trg instances, reduce cv fold number from %d to %d"%(y_train_pos,y_train_neg,cv,cv_new))
                cv = cv_new

            if cv > 1:
                #clf = GridSearchCV(clf_base, para_grid, auc_roc_scoring_func, cv=cv)
                clf = GridSearchCV(clf_base, para_grid, F_scoring_func, cv=cv)
            else:
                clf = clone(clf_base) 
                
            clf.fit(X_train, y_train)
            
            n=len(y_train)
            print ("fit model on %d training insts (%d pos, %d neg)"%(n, y_train_pos, y_train_neg))

            pred_l_test = clf.predict(X_test)
            if use_rule_test:
                pred_l_test[y_test_rules == 0] = 0
            
            true_l_test = y_test

            print ("=== report for test (training: %d insts)==="%(n))
            print(classification_report(true_l_test, pred_l_test))
            f1_test=f1_score(true_l_test, pred_l_test)

            af1_test=-1
            if foldnum in ds_dict["test"]["cv"]:
                af1_test=adjusted_F_score_v2(true_l_test, pred_l_test, ds_dict["test"]["cv"][foldnum])

                
            (auc_test, pred_prob_test)=evaluate_ranking (clf,"sl",model,X_test,y_test,n,use_rule_test,y_test_rules)
            aucs.append(auc_test)
            af1s.append(af1_test)
            f1s.append(f1_test)

            y_true_ls += list(y_test)
            y_hat_ls += list(pred_l_test) 
            y_prob_ls += list(pred_prob_test)

            result_dict["pred_l"][foldnum]=pred_l_test
            result_dict["pred_prob"][foldnum]=pred_prob_test
            
            if  cv > 1:
                print ("best paras: ", clf.best_params_)

            foldnum+=1

        # save results
        if use_rule:
            output_file=re.sub(r"\.pkl", "_%s_%s_%s_%s_rule.pkl"%(model,args.preprocess, args.sampling, args.postprocessing), output_file)
        else:
            output_file=re.sub(r"\.pkl", "_%s_%s_%s_%s.pkl"%(model,args.preprocess, args.sampling, args.postprocessing), output_file)

        f=open(output_file, 'wb')
        pickle.dump(y_true_ls,f)
        pickle.dump(y_hat_ls,f)
        pickle.dump(y_prob_ls,f)
        pickle.dump(id_ls,f)
        pickle.dump(result_dict,f)
        f.close()

        print ('average auc (micro): %.6f'%(np.mean(aucs)))
        print ('average F1 (micro): %.6f'%(np.mean(f1s)))
        print ('average adjusted F1 (micro): %.6f'%(np.mean(af1s)))
        
        f1_test=f1_score(y_true_ls, y_hat_ls)

        af1_test=-1
        if "full" in ds_dict["test"]:
            af1_test=adjusted_F_score_v2(y_true_ls, y_hat_ls, ds_dict["test"]["full"])


        prec_test=precision_score(y_true_ls, y_hat_ls)
        recall_test=recall_score(y_true_ls, y_hat_ls)
        auc_test=roc_auc_score(y_true_ls, y_prob_ls)
        print ('average auc (macro): %.6f'%(auc_test))
        print ('average F1 (macro): %.6f'%(f1_test))
        print ('average Precision (macro): %.6f'%(prec_test))
        print ('average Recall (macro): %.6f'%(recall_test))
        print ('average adjusted F1 (macro): %.6f'%(af1_test))
                        
        print("--- %s seconds ---" % (time.time() - start_time))
        

        
