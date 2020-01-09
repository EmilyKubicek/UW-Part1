#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:09:36 2019

@author: e.kubicekmcafee

###Short narrative on the data preparation for your chosen data set for 
Milestone 3

The goal of this script is to perform k-means clustering on the cleaned 
cleaniing performed in other script) "adult" dataset from 
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data. 

This clean dataset contains 32,561 observations and 19 attributes. Eight attributes 
(income, workclass, fnlwgt, education, occupation, relationship,
race, nativecountry) were removed prior to k-means clustering, as the attributes
that would be used in kmeans clustering are normalized versions of age, marital 
status, and hoursperweek.

Does the data report an equal amount of female and male data? No.
female observations = 10771, male observations = 21790

What is the minimums and maximum # of hours reported having been worked?
min = 1, max = 99

K-mean clustering was performed on the following variables in normalized 
(minmax) form:
        age - age of person represented by the observation (17 - 90)
        marital status - separated (0), married (1), notmarried(2)
        hoursperweek - reported hours worked weekly (1 - 99)
    
The script will generate two plots. One with categorical/numerical 
(maritalstatus, age) data and the other with numeric/numeric (age, 
hoursperweek) data.

Added cluster labels in dataframe correspond to the age/hoursperweek clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


####################################################

#load in data from http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

#ensure display is large enough
pd.options.display.max_columns = 1000

#put csv into pandas df
adult = pd.read_csv(url, header = None)

#check data
adult.head()
adult.shape
adult.dtypes

#give columns names
adult.columns = ["age", "workclass", "fnlwgt", "education", "educationnum", 
                 "maritalstatus", "occupation", "relationship", "race", "sex", 
                 "capitalgain", "capitalloss", "hoursperweek", "nativecountry", "income"]

#################################################################
### Account for abberant data (missing and outlier values)
#check entire df for NaNs
adult.isna().sum()

#hm - that's weird. we know from the source that there are missing values
#let's check question marks
sum(adult.loc[:,'nativecountry'] == "?")

#hm - still weird. When looking at the data head(20) we can see there are
#question marks
adult.head(20)

# data often isn't formatted the way we expect it to be -
# let's add a space in front of the ?
sum(adult.loc[:,'nativecountry'] == " ?")

#there we go - now we need to replace the "  ?" within the df
adult = adult.replace(to_replace= " ?", value=float("NaN"))

#check head again
adult.head(20)

#now see is NaNs come up
adult.isna().sum()

#NaNs found are all in non-numeric categories - we will replace the NaNs
#with Unknown since there is no way for us to currently find out this information
adult = adult.replace(np.nan, 'Unknown', regex=True)

#check head/various columns again to make sure ? is replaced with string Unknown
adult.head(20)
adult.nativecountry.unique()
adult.loc[:,'occupation'].unique()

### Normalize numeric values (at least 1 column)
sum(adult.loc[:,'nativecountry'] == " ?")

adult.isnull().values.any()

#################################################################
### Normalize numeric values (at least 1 column)
#check which columns are numeric
adult.dtypes

#normalize relevant columns using numpy (capital gain, capital loss)
#capital gain
zscaled_gain = (adult.loc[:,'capitalgain'] - np.mean(adult.loc[:,'capitalgain']))/np.std(adult.loc[:,'capitalgain'])

#capital loss
zscaled_loss = (adult.loc[:,'capitalloss'] - np.mean(adult.loc[:,'capitalloss']))/np.std(adult.loc[:,'capitalloss'])

#add zscaled data as column next to original attribute
#capital gain
adult.insert(11, 'z_gain',zscaled_gain)

#capital loss
adult.insert(13, 'z_loss',zscaled_loss)

#see if the new columns have appeared
adult.head()

#################################################################
### Bin categorical variables (at least 1 column)
#marital status
#check value counts for variable
adult.loc[:,"maritalstatus"].value_counts()

#change categories to married, separated, and not married
#married
adult.loc[adult.loc[:, "maritalstatus"] == " Married-civ-spouse", "maritalstatus"] = "married"
adult.loc[adult.loc[:, "maritalstatus"] == " Married-spouse-absent", "maritalstatus"] = "married"
adult.loc[adult.loc[:, "maritalstatus"] == " Married-AF-spouse", "maritalstatus"] = "married"

#notmarried
adult.loc[adult.loc[:, "maritalstatus"] == " Never-married", "maritalstatus"] = "notmarried"
adult.loc[adult.loc[:, "maritalstatus"] == " Divorced", "maritalstatus"] = "notmarried"
adult.loc[adult.loc[:, "maritalstatus"] == " Widowed", "maritalstatus"] = "notmarried"

#separated
adult.loc[adult.loc[:, "maritalstatus"] == " Separated", "maritalstatus"] = "separated"

#check value counts for binned categories
adult.loc[:,"maritalstatus"].value_counts()

#################################################################
### Construct new categorical variables (one-hot encoding; at least one column)
#sex
#check data
adult.loc[:,"sex"].value_counts()

#create dummy variables (one-hot encoding)
onehot_sex = pd.get_dummies(adult.loc[:, "sex"]).astype(int)

#add dummy variable columns to main data frame
adult[['female','male']] = onehot_sex.loc[:,[' Female',' Male']]

#check to see if new columns have been added
adult.head()

#################################################################
### Remove obsolete columns
adult.drop('sex', axis = 1, inplace = True)

#check to make sure it is gone
adult.head()



#drop unecessary columns for kmeans clustering
adult.drop('income', axis = 1, inplace= True)
adult.drop(['workclass', 'fnlwgt',
            'education',
            'occupation', 'relationship',
            'race','nativecountry'], axis = 1, 
            inplace= True)


#make sure it dropped
adult.dtypes

#check out data
adult.head()
#####################################################
# Perform a k-means with sklearn using at least one categorical
# attribute and one numeric attribute (normalized)

#normalize age using numpy
#minmaxscaled =(x - min(x))/(max(x) - min(x))
minmax_age = (adult.iloc[:,0] - min(adult.iloc[:,0])) / (max(adult.iloc[:,0]) - min(adult.iloc[:,0]))

#add normalized data as columm next to the original attribute
adult.insert(1, 'minmax_age',minmax_age)

#normalize hoursperweek using numpy
minmax_hrs = (adult.loc[:,'hoursperweek'] - min(adult.loc[:,'hoursperweek'])) / (max(adult.loc[:,'hoursperweek']) - min(adult.loc[:,'hoursperweek']))

#add normalized data as column next to the original attribute
adult.insert(9, 'minmax_hrs',minmax_hrs)

# prepare categorical attribute (marital status)
#look at data
adult.loc[:,'maritalstatus'].value_counts().sort_values(ascending = True)

#change each label to numeric
adult.loc[adult.loc[:, "maritalstatus"] == "separated", "maritalstatus"] = 0
adult.loc[adult.loc[:, "maritalstatus"] == "married", "maritalstatus"] = 1
adult.loc[adult.loc[:, "maritalstatus"] == "notmarried", "maritalstatus"] = 2

#normaize marital status using numpy
minmax_ms = (adult.loc[:,'maritalstatus'] - min(adult.loc[:,'maritalstatus'])) / (max(adult.loc[:,'maritalstatus']) - min(adult.loc[:,'maritalstatus']))

#add normalized data as column next too the original attribute
adult.insert(4, 'minmax_ms', minmax_ms)


# Perform k-means on one categorical and one numerical attribute 
# (age, maritalstatus)
AM = np.array(adult.loc[:,['minmax_ms', 'minmax_age']])

#visualize the data before clustering
plt.scatter(AM[:,0], AM[:,1], label = 'True Position')

#create clusters
kmeansAM = KMeans(n_clusters = 5)
kmeansAM.fit(AM)

#see what centroid values were generated
print(kmeansAM.cluster_centers_)
print(kmeansAM.labels_)

#create plot
print(plt.scatter(AM[:,0], AM[:,1], c=kmeansAM.labels_, cmap='rainbow',alpha = .1),
      plt.scatter(kmeansAM.cluster_centers_[:,0],kmeansAM.cluster_centers_[:,1], color='black'),
      plt.title('Age and Marital Status (minmax normalized)'),
      plt.xlabel('Marital Status'),
      plt.ylabel('Age'),
      plt.show())

# Perform k-means on numerical attributes
# (age, hours worked per week)
# create array X of data to be put into k-means algorithm
X = np.array(adult.loc[:,['minmax_age', 'minmax_hrs']])

#visualize the data before clustering
plt.scatter(X[:,0], X[:,1], label = 'True Position')
plt.show()

#create clusters
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)

#see what centroid values were generated
print(kmeans.cluster_centers_)

print(kmeans.labels_)


#create plot
print(plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow', alpha = .1), 
      plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='black'),
      plt.title('Age and Hours Worked per Week'), 
      plt.title('Age and Hours Worked per Week'), 
      plt.xlabel('Age (normalized)'), 
      plt.ylabel('Hours per week worked (normalized)'),
      plt.show())
##################################################

# Add cluster labels to dataset
adult.insert(13, 'cluster_labels', kmeans.labels_)

##################################################
# Add a summary comment block that describes the cluster labels.
print('\n', 'cluster label breakdown: ', '\n', np.array(np.unique(kmeans.labels_, return_counts=True)).T)

print(
'''
The clusters for k-means clustering of age and hours worked per week  (immediately above) are labeled as 0, 1, and 2.

The centroid of the first cluster indicates a group that is young in age and primarily works part-time.

The second cluster indicates a group that is middle age and primarily works more than 40 hours a week.

The third cluster, which is the largest, contains observations of those older in age and who primarily work around 40 hour weeks. However, the spread of 
this groups' work hours is larger than the others, and it should be taken into consderation when comparing all three clusters.

'''
)