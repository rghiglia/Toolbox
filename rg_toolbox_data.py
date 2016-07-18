# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:06:13 2016

This is a start of a toolbox for plots


@author: rghiglia

"""

# Use following to import
#import sys
#sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND')
#from rg_toolbox_data import preproc_data
#data_num, data_bol, data_bin, data_mlnd = preproc_data(data)

from __future__ import division


# Import libraries
import numpy as np
import pandas as pd

# -RG 7/15/2016
#def data_types(data, n_thr=5):
#    # You need to determine both type and support
#    d_typ = DataFrame(data.dtypes, columns = ['type']) # data.dtypes a Series with Index = column name
#    d_typ['support'] = 'continuous'
#    for col, x in data.iteritems():
#        if x.dtypes!=np.object:
#            z_tmp = x.value_counts()  # Series
#            if len(z_tmp)<= n_thr: d_typ.ix[col, 'support'] = 'categorical'
#        else:
#            z_tmp = x.unique()        # array
#            if len(z_tmp)<= n_thr: d_typ.ix[col, 'support'] = 'categorical'
#    return d_typ

#from datetime import datetime
import dateutil.parser

def data_types(data, n_thr=5):
    # You need to determine both type and support
    # n_thr determines how many different values a column may assume before it is considered as continuous support
    d_typ = pd.DataFrame(data.dtypes, columns = ['type']) # data.dtypes a Series with Index = column name
    d_typ['subtype'] = '' # add a column for sub-type
    d_typ['support'] = '' # add a column for support
    for col, x in data.iteritems():
        print "Processing column '{}'".format(col)
        if x.dtypes!=np.object:
            z_tmp = x.value_counts()  # Series
            if len(z_tmp)<= n_thr:
                d_typ.ix[col, 'support'] = 'discrete'
            else:
                d_typ.ix[col, 'support'] = 'continuous'
        else:
            z_tmp = x.unique()        # array
            if len(z_tmp)<=n_thr:
                d_typ.ix[col, 'support'] = 'discrete'
            else:
                d_typ.ix[col, 'support'] = 'continuous'
            # Try to determine the type further
            s_tmp = x.dropna()
            if len(s_tmp)>0:
                if type(s_tmp.iloc[0])==type('string'):
                    # Try to see if it is a numerical variable
                    try:
                        int(s_tmp.iloc[0])
                        d_typ.ix[col, 'type'] = 'num'
                        d_typ.ix[col, 'subtype'] = 'int'
                    except:
                        try:
                            float(s_tmp.iloc[0])
                            d_typ.ix[col, 'type'] = 'num'
                            d_typ.ix[col, 'subtype'] = 'float'
                        except:
                            # Try to see if it is a time variable
                            try:
                                dateutil.parser.parse(s_tmp.iloc[0])
                                d_typ.ix[col, 'type'] = 'time'
                                d_typ.ix[col, 'subtype'] = 'time'
                            except ValueError:
                                d_typ.ix[col, 'type'] = 'string'
                                d_typ.ix[col, 'subtype'] = 'string'
                            except:
                                print 'Column {}: unrecognized type'.format(col)
#                                print d_typ.ix[col, 'type'], d_typ.ix[col, 'subtype']
                                d_typ.ix[col, 'type'] = 'string'
                                d_typ.ix[col, 'subtype'] = 'string'
#                                print d_typ.ix[col, 'type'], d_typ.ix[col, 'subtype']
#                elif type(s_tmp.iloc[0])==type(1):
#                    d_typ.ix[col, 'type'] = 'num'
#                    d_typ.ix[col, 'subtype'] = 'int'
                
    return d_typ


from datetime import datetime

def totimestamp(dt, epoch=datetime(1970,1,1)):
    # Example
#    now = datetime.utcnow()
#    print now
#    print totimestamp(now)

    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 

def str2timestamp(dt_str, epoch=datetime(1970,1,1)):
#    if dt_str!='nan' | np.isnan(dt_str):
    if dt_str!='nan':
        td = epoch
    else:
        dt = dateutil.parser.parse(dt_str)
        td = dt - epoch
        
    # return td.total_seconds()
    return (td.microsecond + (td.second + td.day * 86400) * 10**6) / 10**6 


def create_buckets(x, nB=5):
    # Create buckets
    rng = x.max() - x.min()
    b = np.linspace(x.min()-0.01*rng, x.max()+0.01*rng, nB+1) # creates the edges
    x_num = cut_to_series(x, b)
    return x_num
    
def cut_to_series(x, b):
    x_new = x.copy()
    x_new.name = x_new.name + '_bucket'
    cats = pd.cut(x_new, b) # this creates some serious problem; don't try to see it in the Variable Explorer!
    
    df_tmp = cats.to_frame()        # don't try to see it in the Variable Explorer!
    return df_tmp.ix[:,0]
    

def preproc_data(data, nL_max = 4):
    # First convert into numerical values
    # Then convert those into boolean (use dummy functionality)
    # Then convert to binary representation
    data_num = pd.DataFrame(index=data.index) # a numerical version of the data
    
    for col, col_data in data.iteritems():  # text and Series
        print "Processing column: ", col, ", data type = ", col_data.dtype
        
        # Extract levels
        u = sorted(set(col_data))   # unique values in the column (levels)
        levels = col_data.replace(u,range(len(u))) # create an integer representation for each level 
                
        # If non-numeric
        if col_data.dtype == object:
            data_num = data_num.join(levels)
        elif len(u)<=nL_max:
            # Do nothing for now, add as-is
            data_num = data_num.join(col_data)
        else:
            # Create buckets
            data_num = data_num.join(col_data)
            rng = col_data.max() - col_data.min()
            b = np.linspace(col_data.min()-0.01*rng, col_data.max()+0.01*rng, nL_max+1)
            data_num = cut_to_df(data_num, col, b)
                
    return data_num

def cat2num(data):
    data_num = pd.DataFrame(index=data.index) # a numerical version of the data
    
    for col, col_data in data.iteritems():  # text and Series
        print "Processing column: ", col, ", data type = ", col_data.dtype
        
        # Extract levels
        u = sorted(set(col_data))   # unique values in the column (levels)
        levels = col_data.replace(u,range(len(u))) # create an integer representation for each level 
                
        # If non-numeric
        if col_data.dtype == object:
            data_num = data_num.join(levels)
        else:
            data_num = data_num.join(col_data)
            
    return data_num
    
def cat_series2num(x):
    
    # x_num = Series(index=x.index) # a numerical version of the data
    # No: this will convert to numeric, i.e. all NaN's
    
    # Correct NaN's
    x_tmp = x.copy()
    x_tmp[pd.isnull(x_tmp)] = 'NA'

    # Extract levels
    u = sorted(set(x_tmp))   # unique values in the column (levels)
    x_num = x_tmp.replace(u,range(len(u))) # create an integer representation for each level
    x_num = x_num.astype(int)
    x_dic = {ui: i for i, ui in enumerate(u)} 
    x_Rdic = {i: ui for i, ui in enumerate(u)} 
    out = {'x_num': x_num, 'x_dic': x_dic, 'x_Rdic': x_Rdic}
    return out
    

def preproc_data_mlnd(data):

    # MLND
    data_mlnd= pd.DataFrame(index=data.index) # a MLND version of the data
    
    for col, col_data in data.iteritems():  # text and Series
        print "Processing column: ", col, ", data type = ", col_data.dtype
        
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int
    
        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'
    
        data_mlnd = data_mlnd.join(col_data)  # collect column(s) in output dataframe
        
    return data_mlnd
    

def df_num2bol(data):
    # Converts a numerical DataFrame into Boolean
    data_bol = pd.DataFrame(index=data.index) # a boolean version of the data
    return data_bol


def df_num2bin(data):
    # Converts a numerical DataFrame into binary
    data_bin = pd.DataFrame(index=data.index) # a binary version of the data
    return data_bin


def cut_to_df(data, col, b):
    new_col = col + '_bucket'
    data_new = data.copy()
    data_new[new_col] = data_new[col]
    cats = pd.cut(data_new[col], b) # this creates some serious problem; don't try to see it in the Variable Explorer!
    df_tmp = cats.to_frame()        # don't try to see it in the Variable Explorer!
    df_tmp = df_tmp.rename(columns = {col:new_col})
    for (i, x) in enumerate(df_tmp[new_col]):
        data_new.ix[data_new.index[i], new_col] = x
    del cats, df_tmp
    return data_new
    

# Old version
def preproc_data_20160504(data):
    data_num = pd.DataFrame(index=data.index)
    data_bol = pd.DataFrame(index=data.index)
    data_bin = pd.DataFrame(index=data.index)
    data_mlnd= pd.DataFrame(index=data.index)
    
    for col, col_data in data.iteritems():  # text and Series
        print "Processing column: ", col, ", data type = ", col_data.dtype
        
        u = sorted(set(col_data))
        col_data_num = col_data.replace(u,range(len(u)))
        data_num = data_num.join(col_data_num)
        if len(u)<=2:
            data_bol = data_bol.join(col_data_num)
        elif col_data.dtype == object:
            # Create Boolean representation
            for nm in u:
                col_nm = col + "_" + str(nm)
                data_bol[col_nm] = Series((col_data==nm).astype(int), index=data.index)
                
            # Create binary representation
                
        elif len(u)<=4:
            # Create Boolean representation
            for nm in u:
                col_nm = col + "_" + str(nm)
                data_bol[col_nm] = Series((col_data==nm).astype(int), index=data.index)
                
            # Create binary representation
        else:
            # Create Boolean representation by max of 4 buckets ... getting complicated
            # You will also need to make sure it's not really a string, so use col_data_num for this part
            # Actually no, only if input is a numerical value
            # Create a cdf
            # Split in 4
            # Assign to relative bucket
            print "To be coded"
            
        
        # MLND
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int
    
        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'
    
        data_mlnd = data_mlnd.join(col_data)  # collect column(s) in output dataframe
        
    return(data_num, data_bol, data_bin, data_mlnd)


# 6/20/2016
def df_summarize(df):
    clmns = df.columns
    nn = pd.DataFrame(index=range(len(clmns)), columns=['col_nm', 'not_null', 'uniq'])
    for i, (col, x) in enumerate(df.iteritems()):
        nn.ix[i, 'col_nm'] = col
        x_tmp = x[x.notnull()]
        nn.ix[i, 'not_null'] = len(x_tmp) # is it the same as x.count()?
        nn.ix[i, 'uniq'] = len(set(x_tmp))
    nn.sort_values(['not_null', 'uniq', 'col_nm'], ascending=[False, False, True], inplace=True)
    return nn
    