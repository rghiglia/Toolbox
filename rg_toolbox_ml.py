# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:06:13 2016

This is a start of a toolbox


@author: rghiglia

"""

#import numpy as np
#from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score

from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import svm # SVM

import sys; sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND')
from rg_toolbox_data import data_types, create_buckets, cat_series2num


def train_clf(clf, X_train, y_train, verbose=False):
    if verbose: print "Training {} ...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    if verbose: print "Training time (secs): {:.3f}".format(end - start)
    return end - start

def pred_clf(clf, X, y = [], score='f1', verbose=False):
    if verbose: print "Predicting using {} ...".format(clf.__class__.__name__)
    t_start = time.time()
    y_pred = clf.predict(X)
    t_end = time.time()
    dt = t_end - t_start
    if verbose: print "Prediction time (secs): {:.3f}".format(dt)
    if len(y)==0:
        score = np.NaN
    else:
        if score=='f1':
            score = f1_score(y, y_pred)
    out = {'score': score, 'y_pred': y_pred, 'dt': dt} 
    return out


def run_clf(name, X_trn, y_trn, X_tst, y_tst=[]):
    if name=='NB':
        clf = GaussianNB()
    elif name=='LR':
        clf = LogisticRegression()
    elif name=='DT':
        clf = DecisionTreeClassifier(max_depth=3)
    elif name=='kNN':
        clf = KNeighborsClassifier()
    elif name=='SVM':
        clf = svm.SVC(C=1.0)
    else:
        print "Unrecognized classifier"
    
    out_trn = train_clf(clf, X_trn, y_trn);
    out_prd_ins = pred_clf(clf, X_trn, y_trn)
    out_prd_ots = pred_clf(clf, X_tst, y_tst)
    print "Prediction with %s F1 = %1.4f (in-sample)" % (clf.__class__.__name__ ,out_prd_ins['score'])
    print "Prediction with %s F1 = %1.4f (out-of-sample)" % (clf.__class__.__name__ ,out_prd_ots['score'])
    out = {'out_trn': out_trn, 'out_prd_ins': out_prd_ins, 'out_prd_ots': out_prd_ots}
    return out


# Data Augmentation
def data_augment(data, nB=5):
    df_types = data_types(data)
    data_aug = data.copy()
    for nm, sp in df_types['support'].iteritems():
        print
        print nm, sp
        if sp=='continuous':
            print df_types.loc[nm]
            # Create buckets
            col_num = create_buckets(data_aug[nm], nB)
            data_aug = pd.concat([data_aug, col_num], axis=1)
            out_tmp = cat_series2num(data_aug[col_num.name]) # convert string to numeric level
            col_num = out_tmp['x_num']
            col_num.name = nm + '_num'
            data_aug = pd.concat([data_aug, col_num], axis=1)
        else:
            # Create the couple of (bucket txt value, bucket numerical value)
            # Also create a dictionary of that
            if df_types.ix[nm, 'type'] in (np.int64, np.float64):
                col_txt = nm + '_' + data_aug[nm].astype(str) # convert numeric level to string
                col_txt.name = nm + '_lev'
                data_aug = pd.concat([data_aug, col_txt], axis=1)
            elif df_types.ix[nm, 'type']==np.object:
                out_tmp = cat_series2num(data_aug[nm]) # convert string to numeric level
                col_num = out_tmp['x_num']
                col_num.name = nm + '_num'
                data_aug = pd.concat([data_aug, col_num], axis=1)
    return data_aug

def summarize_feat_supervised(data_trn, y_trn, feat, col_tgt, isPct=1):
    # Summarizes output target (col_tgt) conditional on values of feature (feat)
    df_tmp = pd.concat([data_trn, y_trn], axis=1)
    df_grp = df_tmp.groupby([feat, col_tgt]).count()
    df_grp = df_grp.ix[:,0].unstack()
    df_grp.ix['Tot'] = df_grp.sum(axis=0)
    df_grp['Tot'] = df_grp.sum(axis=1)
    if isPct:
        df_tmp = df_grp.copy()
        for i in df_grp.index:
            df_tmp.ix[i,:] = df_grp.loc[i,:]/df_grp.loc[i, 'Tot']
        df_grp = df_tmp.ix[:-1, :-1]
    return df_grp
            
