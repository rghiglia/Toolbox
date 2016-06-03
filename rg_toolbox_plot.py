# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:06:13 2016

This is a start of a toolbox for plots


@author: rghiglia

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skealrn.metrics import f1_score

def heat_feat(df_grp, col_tgt, feat, data_trn, y_trn):
    # F1 score
    y_feat = y_trn.copy()
    for i in set(data_trn[feat]):
        y_feat[data_trn[feat]==i] = df_grp.columns[(df_grp.loc[i]).argmax()]
    f1_feat = f1_score(y_trn, y_feat)
    
    # For now burden of calculating F1 score on user
    plt.subplots(figsize=(6, 4))
    h = sns.heatmap(df_grp.transpose(), annot=True, cmap='YlGnBu', vmin=0, vmax=1);
    plt.subplots_adjust(top=0.9)
    tit = col_tgt + ' by ' + feat + ' F1 = {0:1.2f}'.format(f1_feat)
    h.figure.suptitle(tit);


def barplot_feature(data, feat):
    df = data.groupby(feat)
    
    # Set the width of each bar
    wd = 0.4

    # Display each category's survival rates
    for i in np.arange(len(frame)):
        nonsurv_bar = plt.bar(i-wd, frame.loc[i]['NSurvived'], width=wd, color='r')
        surv_bar = plt.bar(i, frame.loc[i]['Survived'], width=wd, color='g')

        plt.xticks(np.arange(len(frame)), values)
        plt.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

    # Common attributes for plot formatting
    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature'%(key))
    plt.show()
