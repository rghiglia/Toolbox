# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:06:13 2016

This is a start of a toolbox


@author: rghiglia

"""

import numpy as np
from pandas import DataFrame, Series


# -----------------------------------------------------------------------------
# Data Analytics
# -----------------------------------------------------------------------------

def hist_discr(z):
    # Assumes 'z' is a discrete variable
    # Example
    # z = Series([0, 0, 1, 1, 0, 2, 1])
    # z1, Ix = hist_discr(z)
    z = Series(z)
    c_lev = set(z)
    n = Series(index=c_lev)
    Ix = []
    for i in c_lev:
        z_tmp = z[z==i];
        n[i] = len(z_tmp)
        Ix.append(z_tmp.index)
    return (n, Ix)


def prb_discr(z):
    # Example
    # z = Series([0, 0, 1, 1, 0, 2, 1])
    # z2 = prb_discr(z)
    n, _ = hist_discr(z)
    return n/sum(n)


# -----------------------------------------------------------------------------
# Entropy
# -----------------------------------------------------------------------------

def entropy(z):
    # Example
    # z = Series([0, 0, 1, 1, 0, 2, 1])
    # e = entropy(z) # can be larger than 1 because you have 3 possoble values
    # z = Series([0, 0, 1, 1, 0, 0, 1])
    # e = entropy(z)
    p = prb_discr(z)
    return sum(-p * np.log2(p))


def entropy_gain(x,y):
    _, IxA = hist_discr(x)
    ew = []
    w = []
    for ix in IxA:
        w.append(float(len(ix))/len(y))
        ew.append(w[-1] * entropy(y[ix]))
    return entropy(y) - sum(ew)
    


# -----------------------------------------------------------------------------
# Hamtropy
# -----------------------------------------------------------------------------

# Hamtropy is a mixture between Hamming distance for Boolean variables and Entropy
# The 'problem' with Hamming distance is that a and not(a) are maximally separated
# (distance = len(a)), however from an information viewpoint they are just the same
# they are 'collinear'
# Combining Hamming and Entropy will obviate that issue

def hamtropy(z1,z2):

    """
    Inputs
    z1, z2: lists or Series

    Outputs
    d: Hamtropy (scalar)   
    
    # Example
    
    z1 = X_all_aug.ix[:,0]
    z2 = X_all_aug.ix[:,1]
    h = abs(z2 - z1)
    p = h / len(z1)
    hamtropy(z1,z2)
    hamtropy(X_all_aug.ix[:,0],X_all_aug.ix[:,3])
    """
    
    h = abs(z2 - z1)
    p = sum(h / len(z1))
    if p==0:
        d = 0
    else:
        d = max(-p*np.log2(p),0)
    return d



def Hamtropy(X):
    
    """
    This calculates the Hamtropy matrix akin a correlation matrix between all
    columns in X
    
    Inputs
    X: nd array or DataFrame

    Outputs
    H: Hamtropy (n x n matrix)   
    
    # Example
    H = Hamm_entropy(X_all_aug)
    
    """
    
    if type(X)==DataFrame:
        X = X.values
    nC = X.shape[1]
    H = np.zeros([nC, nC])
    for i in range(nC):
        for j in range(i+1,nC):
            H[i,j] = hamtropy(X[:,i],X[:,j])
            H[j,i] = H[i,j]
    return H



def hamtropy_Xy(X,y):
    
    """
    This calculates the Hamtropy of a vector y vs a matrix X
    
    Inputs
    X: nd array or DataFrame
    y: nd vectoe commensurate with rows in X

    Outputs
    d1: Hamtropy (vector)
    
    # Example
    d1 = hamtropy_Xy(X_all_aug,y)
    d1 = hamtropy_Xy(X_all_aug,X_all_aug.ix[:,0])
    
    """
    
    if type(X)==DataFrame: X = X.values
    nO = X.shape[0]
    d1 = np.zeros([nO,1])
    for i in range(nO):
        d1[i] = hamtropy(X[i,:],y)
    return d1[:,0]


def hamm_entropy_kNN(X,y,kNN=3):
    
    """
    This calculates the Hamtropy of a vector y vs a matrix X but only for the closest kNN neighbors of y
    
    Inputs
    X: nd array or DataFrame
    y: nd vectoe commensurate with rows in X
    kNN: (optional) # of neighbors

    Outputs
    d1: Hamtropy (vector)
    ixs: index corresponding to the locations of the neighbors in X
    
    # Example
    d1, ixs = hamtropy_Xy(X_all_aug,y,3)
    d1, ixs = hamtropy_Xy(X_all_aug,X_all_aug.ix[:,0],3)
    
    """
    
    d1 = hamtropy_Xy(X,y)
    d1 = Series(d1)
    d1.sort(ascending=True)
    ixs = d1.index
    d1_kNN = d1[0:kNN]
    ixs_kNN = ixs[0:kNN]
    return (np.array(d1_kNN), ixs_kNN)


