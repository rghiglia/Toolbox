f# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 15:08:16 2016

@author: rghiglia
"""

# Snippets


# ------------------------------------------------------
# References
# ------------------------------------------------------

## Comprehension
#http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/


# ------------------------------------------------------
# Path
# ------------------------------------------------------

import sys
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\bokeh-master')

# List files
import glob
print glob.glob("/home/adam/*.txt")


# Careful!
#In [12]: np.nan == np.nan
#Out[12]: False


# ------------------------------------------------------
# Finding first
# ------------------------------------------------------
# http://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array

# For list
l = list(1,2,3)
l.index(2)

# For numpy.array
itemindex = numpy.where(array==item)

array[itemindex[0][0]][itemindex[1][0]]
#would be equal to your item and so would
array[itemindex[0][1]][itemindex[1][1]]



# http://stackoverflow.com/questions/9868653/find-first-list-item-that-matches-criteria
#If you don't have any other indexes or sorted information for your objects, then you will have to iterate until such an object is found:
next(obj for obj in objs if obj.val==5)


# ------------------------------------------------------
# Extract files (unzip)
# ------------------------------------------------------

import gzip

# Extract files
dnm = r'C:\Users\rghiglia\Documents\ML_ND\Kaggle\Expedia'
fnz = 'sample_submission.csv.gz'
fnzL = dnm + '\\' + fnz
fnmL = fnzL[:-3]  # remove the '.gz' from the filename

# Read from .gz
with gzip.open(fnzL, 'rb') as in_file:
    s = in_file.read()

# Store uncompressed file data from 's' variable
with open(fnmL, 'w') as f:
    f.write(s)


# ------------------------------------------------------
# Saving Figure
# ------------------------------------------------------

if sav: plt.savefig(dnm + '\\' + 'Succ.png', bbox_inches='tight')


# ------------------------------------------------------
# Assign multiple variabels to multiple names
# ------------------------------------------------------

#for x, y in zip( varlist, data ):
#    l[x] = y
#
#or, more tersely:
#
#[ locals()[x] = y for x, y in zip( varlist, data ) ]



# ------------------------------------------------------
# Timing
# ------------------------------------------------------

import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))


# ------------------------------------------------------
# Saving data
# ------------------------------------------------------

fnmO = (dnm + '\\' + 'yelp_data.pkl')
import dill                            #pip install dill --user
dill.dump_session(fnmO)

# Load session
dill.load_session(filename)



# ------------------------------------------------------
# Loops
# ------------------------------------------------------

for i, clf in enumerate(('svc', 'lin_svc', 'rbf_svc', 'poly_svc')):
    print i, clf
 
for n, (name, estimator) in enumerate(estimators):
    print n, (name, estimator)


# ------------------------------------------------------
# with Statement
# ------------------------------------------------------

with open('output.txt', 'w') as f:
    f.write('Hi there!')

#The above with statement will automatically close the file after the nested block of code. (Continue reading to see exactly how the close occurs.) The advantage of using a with statement is that it is guaranteed to close the file no matter how the nested block exits. If an exception occurs before the end of the block, it will close the file before the exception is caught by an outer exception handler. If the nested block were to contain a return statement, or a continue or break statement, the with statement would automatically close the file in those cases, too.


# ------------------------------------------------------
# format Statement
# ------------------------------------------------------
# see https://pyformat.info/
'F1 = {0:1.2f}'.format(f1_feat, 2.0) # '0' is for the index in the list withing format(), here refers to f1_feat
'F1 = {1:1.2f}'.format(f1_feat, 2.0) # '0' is for the index in the list withing format(), here refers to 2.0
'F1 = {0:1.2f}, {1:.2%}'.format(f1_feat, 0.2) # '0' is for the index in the list withing format(), here refers to 2.0
# :1.2f is the format itself





# ------------------------------------------------------
# Matrices in numpy
# ------------------------------------------------------

#dot(a, b[, out])	Dot product of two arrays.
#inner(a, b)	Inner product of two arrays.
#outer(a, b[, out])	Compute the outer product of two vectors.
#matmul(a, b[, out])	Matrix product of two arrays.
#tensordot(a, b[, axes])	Compute tensor dot product along specified axes for arrays >= 1-D.



# ------------------------------------------------------
# Data
# ------------------------------------------------------

op = np.where(df['a'].str.contains('Windows'), 'Windows', 'Not Windows')
# Wow

# Split data set
X_train_df = X_all_aug[:num_train,:]
# Error: unhashable type, so I cannot select a subset (slice) of a dataframe?
X_train_df = X_all_aug.iloc[:num_train]
feat_nm_aug = list(my_dataframe.columns.values)     # column names
df1.loc[:, df1.loc['a']>0]   # slicing by column


# For a pd.Series
s[s>0]  # returns the subset
s.where(s>0) # returns an entire series with True and False, same size as s

x_tmp = np.arange(5)
print np.tile(x_tmp, 2)
print x_tmp.repeat(2) # interesting!
print np.tile(x_tmp, [2, 1])


# Concatenate np arrays
S = np.c_[s1, s2, s3]



# ------------------------------------------------------
# DataFrame
# ------------------------------------------------------

# Delete column
df.drop('column_name', axis=1, inplace=True)


def _col_seq_set(df, col_list, seq_list):
    ''' set dataframe 'df' col_list's sequence by seq_list '''
    col_not_in_col_list = [x for x in list(df.columns) if x not in col_list]
    for i in range(len(col_list)):
        col_not_in_col_list.insert(seq_list[i], col_list[i])

    return df[col_not_in_col_list]
DataFrame.col_seq_set = _col_seq_set

x = X_all_aug[0]
X_all_aug[0] # gives me an error. This is weird
X_all_aug.ix[0] # first row
X_all_aug.ix[:,0] # first column

ix_tmp = X_all_aug[X_all_aug['school_GP']==1].index
y_num = y_all.replace(['yes', 'no'],[1, 0])
y_GP = y_num[ix_tmp]

# Changing one column at a time conditional on some value
mu = df.Age.mean()  # improvements: use age conditional on sex
df.loc[np.isnan(df.Age), 'Age'] = mu

# Differentiating by dtype
for y in agg.columns:
    if(agg[y].dtype == np.float64 or agg[y].dtype == np.int64):
          treat_numeric(agg[y])
    else:
          treat_str(agg[y])


# Renaming column
df=df.rename(columns = {'two':'new_name'})

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


# Cleaning
# Across the board
import numpy as np
for col in df.columns:
    if df[col].dtype != object:
        u = df[col].unique()
        if len(u)>2:
            df.loc[np.isnan(df[col]), col] = df[col].mean()
        elif len(u)==2:
            if len(df[col]==u[0]):
                df.loc[np.isnan(df[col]), col] = u[0]
            else:
                df.loc[np.isnan(df[col]), col] = u[1]


# Need to check this out:
# crosstab
#temp1 = pd.crosstab([df3.personno],[df3.Activitycode],rownames=['person'],colnames=['Activity'])
##Cross-tabulation of person no and activity code
#
#temp1.plot(kind='bar',stacked = True,color=['red','green','blue','black','yellow','brown'],grid = False)
#plt.show() # plotting stacked histogram

foo = 'foo'
bar = 'bar'
a = np.array([foo, foo, foo, foo, bar, bar,
       bar, bar, foo, foo, foo])
one = 'one'
two = 'two'
b = np.array([one, one, one, two, one, one,
       one, two, two, two, one])
dull = 'dull'
shiny = 'shiny'
c = np.array([dull, dull, shiny, dull, dull, shiny,
       shiny, dull, shiny, shiny, shiny])

df = pd.DataFrame([a, b, c], index=['a', 'b', 'c']).T

pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
pd.crosstab(df['a'], [df['b'], df['c']], rownames=['a'], colnames=['b', 'c'])


# Adding rows
#df = pd.DataFrame({'c1': 0.0, 'c2': 2}) # error: ValueError: If using all scalar values, you must pass an index
df = pd.DataFrame({'c1': 0.0, 'c2': 2}, index=[0])
print df
pd.concat([df, pd.DataFrame({'c1': 2.0, 'c2': 3.0}, index=[0])])
print df
# not really
df = pd.concat([df, pd.DataFrame({'c1': 2.0, 'c2': 3.0}, index=[0])])
print df
# better
df = pd.concat([df, pd.DataFrame({'c1': 2.0, 'c2': 3.0}, index=[0])], ignore_index=True)
print df
# Ok

# ------------------------------------------------------
# Strings
# ------------------------------------------------------

myList = ['element1\t0238.94', 'element2\t2.3904', 'element3\t0139847']
myList2 = [i.split('\t')[0] for i in myList] 



# ------------------------------------------------------
# Graphic Libraries
# ------------------------------------------------------
Pandas
Seaborn
ggplot
Bokeh
pygal
Plotly

# Histograms
# via pyplot
# http://chrisalbon.com/python/matplotlib_stacked_bar_plot.html: here it creates it from the ground up
# https://plot.ly/python/histograms/

## Don't have this librabry ...
#
#import plotly.plotly as py
#import plotly.graph_objs as go
#
#import numpy as np
#x0 = np.random.randn(500)
#x1 = np.random.randn(500)+1
#
#trace1 = go.Histogram(
#    x=x0,
#    histnorm='count',
#    name='control',
#    autobinx=False,
#    xbins=dict(
#        start=-3.2,
#        end=2.8,
#        size=0.2
#    ),
#    marker=dict(
#        color='fuchsia',
#        line=dict(
#            color='grey',
#            width=0
#        )
#    ),
#    opacity=0.75
#)
#trace2 = Histogram(
#    x=x1,
#    name='experimental',
#    autobinx=False,
#    xbins=dict(
#        start=-1.8,
#        end=4.2,
#        size=0.2
#    ),
#    marker=dict(
#        color='rgb(255, 217, 102)'
#    ),
#    opacity=0.75
#)
#data = [trace1, trace2]
#layout = go.Layout(
#    title='Sampled Results',
#    xaxis=dict(
#        title='Value'
#    ),
#    yaxis=dict(
#        title='Count'
#    ),
#    barmode='overlay',
#    bargap=0.25,
#    bargroupgap=0.3
#)
#fig = go.Figure(data=data, layout=layout)
#plot_url = py.plot(fig, filename='style-histogram')



#!/usr/bin/env python
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt


N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='r', yerr=menStd)
p2 = plt.bar(ind, womenMeans, width, color='y',
             bottom=menMeans, yerr=womenStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()


# ------------------------------------------------------
# Plots
# ------------------------------------------------------

# See: http://matplotlib.org/users/pyplot_tutorial.html

# For graphic options:
#http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot


#kwargs control the Line2D properties:
#
#    Property 	Description
#    agg_filter 	unknown
#    alpha 	float (0.0 transparent through 1.0 opaque)
#    animated 	[True | False]
#    antialiased or aa 	[True | False]
#    axes 	an Axes instance
#    clip_box 	a matplotlib.transforms.Bbox instance
#    clip_on 	[True | False]
#    clip_path 	[ (Path, Transform) | Patch | None ]
#    color or c 	any matplotlib color
#    contains 	a callable function
#    dash_capstyle 	[‘butt’ | ‘round’ | ‘projecting’]
#    dash_joinstyle 	[‘miter’ | ‘round’ | ‘bevel’]
#    dashes 	sequence of on/off ink in points
#    drawstyle 	[‘default’ | ‘steps’ | ‘steps-pre’ | ‘steps-mid’ | ‘steps-post’]
#    figure 	a matplotlib.figure.Figure instance
#    fillstyle 	[‘full’ | ‘left’ | ‘right’ | ‘bottom’ | ‘top’ | ‘none’]
#    gid 	an id string
#    label 	string or anything printable with ‘%s’ conversion.
#    linestyle or ls 	[‘solid’ | ‘dashed’, ‘dashdot’, ‘dotted’ | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']
#    linewidth or lw 	float value in points
#    marker 	A valid marker style
#    markeredgecolor or mec 	any matplotlib color
#    markeredgewidth or mew 	float value in points
#    markerfacecolor or mfc 	any matplotlib color
#    markerfacecoloralt or mfcalt 	any matplotlib color
#    markersize or ms 	float
#    markevery 	[None | int | length-2 tuple of int | slice | list/array of int | float | length-2 tuple of float]
#    path_effects 	unknown
#    picker 	float distance in points or callable pick function fn(artist, event)
#    pickradius 	float distance in points
#    rasterized 	[True | False | None]
#    sketch_params 	unknown
#    snap 	unknown
#    solid_capstyle 	[‘butt’ | ‘round’ | ‘projecting’]
#    solid_joinstyle 	[‘miter’ | ‘round’ | ‘bevel’]
#    transform 	a matplotlib.transforms.Transform instance
#    url 	a url string
#    visible 	[True | False]
#    xdata 	1D array
#    ydata 	1D array
#    zorder 	any number

# Legend:
#http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend


# Create the figure window
import numpy as np
import matplotlib.pyplot as pl
#from matplotlib import pyplot as pl


ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
ax.legend(pols, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0) # legend outside

fig = pl.figure()
fig.add_subplot(111)
fig.add_subplot(1,1,1)  # equivalent but more general
fig.add_subplot(212, axisbg='r') # add subplot with red background
fig.add_subplot(111, projection='polar') # add a polar subplot

# add Subplot instance sub
fig.add_subplot(sub)

# Scatter plot
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

# Plot the model complexity graph
pl.figure(figsize=(7, 5))
pl.title('Decision Tree Regressor Complexity Performance')
pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
pl.legend()
pl.xlabel('Maximum Depth')
pl.ylabel('Total Error')
pl.show()



# Subplot the learning curve graph
fig = pl.figure(figsize=(10,8))
ax = fig.add_subplot(2, 2, k+1)
ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
ax.legend()
ax.set_title('max_depth = %s'%(depth))
ax.set_xlabel('Number of Data Points in Training Set')
ax.set_ylabel('Total Error')
ax.set_xlim([0, len(X_train)])

# Histogram
pl.hist(prc)
pl.title("Boston Prices")
pl.xlabel("000$")
pl.ylabel("Occurrances")


# Legend
## See
#http://matplotlib.org/users/legend_guide.html
ax.legend('1', '2')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Ticks
plt.xticks(x, labels, rotation='vertical')

# Pie chart
# make a square figure and axes
from matplotlib import pyplot as pl
fig = pl.figure(1, figsize=(6,6))
ax = pl.axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15, 30, 45, 10]
explode=(0, 0.05, 0, 0)

pl.pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

pl.title('Raining Hogs and Dogs', bbox={'facecolor':'0.8', 'pad':5})

## Check also:
#https://plot.ly/python/pie-charts/

#https://plot.ly/python/
## Nope:
#fig = {
#    'data': [{'labels': ['Residential', 'Non-Residential', 'Utility'],
#              'values': [19, 26, 55],
#              'type': 'pie'}],
#    'layout': {'title': 'Forcasted 2014 U.S. PV Installations by Market Segment'}
#}
#
#url = pl.plot(fig)

# Wow: you need to check this out:

#https://plot.ly/python/offline/


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(x, y, bins=4)

elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

#ticksx = np.arange(0.5, 5, 1)
#plt.xticks(ticksx, column_names)
#
#ticksy = np.arange(0.6, 7, 1)
#plt.yticks(ticksy, row_names)


# Flat 3D bar plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
    xs = np.arange(20)
    ys = np.random.rand(20)

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# Showing a matrix
plt.matshow(prescriber_dist)


import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

slices = [1,2,3] * 4 + [20, 25, 30] * 2
shuffle(slices)

fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(111)

cmap = plt.cm.prism
colors = cmap(np.linspace(0., 1., len(slices)))

labels = ["Some text"] * len(slices)

ax.pie(slices, colors=colors, labels=labels, labeldistance=1.05)
ax.set_title("Figure 1");



# Vertical line
# The vertical line for average silhoutte score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")


"""
.. versionadded:: 1.1.0
   This demo depends on new features added to contourf3d.
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()



# Seaborn!!
import seaborn as sb
# Yes! Installed with conda :)


# Look at categorical plots!
#https://stanford.edu/~mwaskom/software/seaborn/tutorial/categorical.html#categorical-tutorial

# Combined with categories
g = sns.PairGrid(iris, hue="species")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();


# Even more sophisticated
g = sns.PairGrid(iris)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(sns.kdeplot, lw=3, legend=False);

g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", size=2.5)





# Great pair-plot
iris = sns.load_dataset("iris")
sns.pairplot(iris);

# Contours of joint distributions!
g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);


# look at percentile ranks
pcts = 100. * data.rank(axis=0, pct=True).iloc[indices].round(decimals=3)
print pcts

# visualize percentiles with heatmap
_ = sns.heatmap(pcts.reset_index(drop=True), annot=True, cmap='YlGnBu')
# Amazing graphics

import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")


from string import letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(letters[:26]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)



# -----------------------------------------------------------------------------
# Seaborn
# -----------------------------------------------------------------------------
#https://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html

import numpy as np
import pandas as pd
import seaborn as sns

sns.set() # resets parameters

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)
#df_tmp = sns.load_dataset("car_crashes")
#df_tmp.info()

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)

# Horizontal bar plot

# Joint distribution
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="white")

# Generate a random correlated bivariate dataset
rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .5), (.5, 1)]
x1, x2 = rs.multivariate_normal(mean, cov, 500).T
x1 = pd.Series(x1, name="$X_1$")
x2 = pd.Series(x2, name="$X_2$")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(x1, x2, kind="kde", size=7, space=0)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# Grouped violin
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True,
               inner="quart", palette={"Male": "b", "Female": "y"})
sns.despine(left=True)



# Dot plot on several variables
import seaborn as sns
sns.set(style="whitegrid")

# Load the dataset
crashes = sns.load_dataset("car_crashes")

# Make the PairGrid
g = sns.PairGrid(crashes.sort_values("total", ascending=False),
                 x_vars=crashes.columns[:-3], y_vars=["abbrev"],
                 size=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="Reds_r", edgecolor="gray")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Total crashes", "Speeding crashes", "Alcohol crashes",
          "Not distracted crashes", "No previous crashes"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)


# Time series
import seaborn as sns
sns.set(style="darkgrid")

# Load the long-form example gammas dataset
gammas = sns.load_dataset("gammas")

# Plot the response with standard error
sns.tsplot(data=gammas, time="timepoint", unit="subject",
           condition="ROI", value="BOLD signal")

import seaborn as sns
sns.set()

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")


# Bar plot time series
import numpy as np
import seaborn as sns
sns.set(style="white")

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Make a range of years to show categories with no observations
years = np.arange(2000, 2015)

# Draw a count plot to show the number of planets discovered each year
g = sns.factorplot(x="year", data=planets, kind="count",
                   palette="BuPu", size=6, aspect=1.5, order=years)
g.set_xticklabels(step=2)


# Multiple Linear Regressions
import seaborn as sns
sns.set(style="ticks", context="talk")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Make a custom sequential palette using the cubehelix system
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)

# Plot tip as a function of toal bill across days
g = sns.lmplot(x="total_bill", y="tip", hue="day", data=tips,
               palette=pal, size=7)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Total bill ($)", "Tip ($)")


# Logistic regression
import seaborn as sns
sns.set(style="darkgrid")

# Load the example titanic dataset
df = sns.load_dataset("titanic")

# Make a custom palette with gendered colors
pal = dict(male="#6495ED", female="#F08080")

# Show the survival proability as a function of age and sex
g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=df,
               palette=pal, y_jitter=.02, logistic=True)
g.set(xlim=(0, 80), ylim=(-.05, 1.05))


# Distributions
sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = rs.normal(size=100)

# Plot a simple histogram with binsize determined automatically
sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(d, color="m", ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()

# Heatmap
import seaborn as sns
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5);

# Some amazing stuff
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="dark")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()


# Simple bar chart
sb.countplot(x)

# More complex

sns.set(font="monospace")

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Create a custom palette to identify the networks
network_pal = sns.cubehelix_palette(len(used_networks),
                                    light=.9, dark=.1, reverse=True,
                                    start=1, rot=-2)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks).map(network_lut)

# Create a custom colormap for the heatmap values
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

# Draw the full plot
sns.clustermap(df.corr(), row_colors=network_colors, linewidths=.5,
               col_colors=network_colors, figsize=(13, 13), cmap=cmap)








# Bokeh

## To import data
#import bokeh.sampledata
#bokeh.sampledata.download()
## something went wrong but  but maybe it went through ... or at least some of the data
## not the market data

## Might have worked from shell
#python -c "import bokeh.sampledata; bokeh.sampledata.download()"

from math import pi
import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.stocks import MSFT
# This line didn't work


df = pd.DataFrame(MSFT)[:50]
df["date"] = pd.to_datetime(df["date"])

mids = (df.open + df.close)/2
spans = abs(df.close-df.open)

inc = df.close > df.open
dec = df.open > df.close
w = 12*60*60*1000 # half day in ms

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, toolbar_location="left")

p.title = "MSFT Candlestick"
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.3

p.segment(df.date, df.high, df.date, df.low, color="black")
p.rect(df.date[inc], mids[inc], w, spans[inc], fill_color="#D5E1DD", line_color="black")
p.rect(df.date[dec], mids[dec], w, spans[dec], fill_color="#F2583E", line_color="black")

output_file("candlestick.html", title="candlestick.py example")

show(p)  # open a browser
# frriggin awesome!



from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.us_counties import data as counties # ok
from bokeh.sampledata.us_states import data as states # ok
from bokeh.sampledata.unemployment import data as unemployment # not ok :(

del states["HI"]
del states["AK"]

EXCLUDED = ("ak", "hi", "pr", "gu", "vi", "mp", "as")

state_xs = [states[code]["lons"] for code in states]
state_ys = [states[code]["lats"] for code in states]

county_xs=[counties[code]["lons"] for code in counties if counties[code]["state"] not in EXCLUDED]
county_ys=[counties[code]["lats"] for code in counties if counties[code]["state"] not in EXCLUDED]

colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]

county_colors = []
for county_id in counties:
    if counties[county_id]["state"] in EXCLUDED:
        continue
    try:
        rate = unemployment[county_id]
        idx = int(rate/6)
        county_colors.append(colors[idx])
    except KeyError:
        county_colors.append("black")

p = figure(title="US Unemployment 2009", toolbar_location="left",
           plot_width=1100, plot_height=700)

p.patches(county_xs, county_ys,
          fill_color=county_colors, fill_alpha=0.7,
          line_color="white", line_width=0.5)

p.patches(state_xs, state_ys, fill_alpha=0.0,
          line_color="#884444", line_width=2, line_alpha=0.3)


output_file("choropleth.html", title="choropleth.py example")

show(p)





# ------------------------------------------------------
# ML
# ------------------------------------------------------
# Shuffling
from sklearn.utils import shuffle
X_all_aug_d, y_all, y_all_d = shuffle(X_all_aug, y_all, y_all_d, random_state=0)


# Stratified sampling
for train_index, test_index in sss:
    xtrain, xtest = data.iloc[train_index], data.iloc[test_index]
    ytrain, ytest = target[train_index], target[test_index]


# ------------------------------------------------------
# Extras
# ------------------------------------------------------
sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)

# Histogram
a = np.arange(5)
hist, bin_edges = np.histogram(a, density=True)
