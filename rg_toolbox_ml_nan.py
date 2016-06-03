# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:06:13 2016

This is a start of a toolbox for plots


@author: rghiglia

"""

# This module deals with removing NaN's for pre-processing. This is a very
# problem specific step
# It is VERY important that you:
# -) Set it up to work for both training and testing set, i.e. they should call
# the same function
# -) Even more important is that any adjustment is done SOLELY based on training
# set data

import numpy as np

# This is a very problem-dependent step
def nan2avg(data_trn, data_tst=[]):
    if (type(data_tst)==list) & (len(data_tst)==0):
        is_tst = 0
    else:
        is_tst = 1
    if not(is_tst): # processing training data
        print "Training data"
        data_trn_sub = data_trn.copy()
        for col, x in data_trn_sub.iteritems():
            if x.dtype==np.float64:
                data_trn_sub.ix[x.isnull(), col] = data_trn_sub.ix[x.notnull(), col].mean()
            elif x.dtype==np.int64:
                data_trn_sub.ix[x.isnull(), col] = np.rint(data_trn_sub.ix[x.notnull(), col].mean()).astype(int) # rounding will convert to the closest integer, so ok for categoricals, for integert and continuous it will be the integer closest to the mean, also ok
        return data_trn_sub
    else:  # processing testing data BUT with stats from training data!
        print "Test data"
        data_tst_sub = data_tst.copy()
        for col, x in data_tst_sub.iteritems():
            if x.dtype==np.float64:
                data_tst_sub.ix[x.isnull(), col] = data_trn.ix[data_trn[col].notnull(), col].mean()
            elif x.dtype==np.int64:
                data_tst_sub.ix[x.isnull(), col] = np.rint(data_trn.ix[data_trn[col].notnull(), col].mean()).astype(int) # rounding will convert to the closest integer, so ok for categoricals, for integert and continuous it will be the integer closest to the mean, also ok
        return data_tst_sub


