#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:47:13 2019

@author: e.kubicek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:24:52 2019

@author: e.kubicekmcafee
### Short narrative

This script reads in and cleans the "adult" dataset from 
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data. 


After k-means clustering was completed, the cluster labels (age/hoursperweek)
were added as a feature to the original dataset.

Primarily, this script was made to create a classification model to predict
if a single person has an income of <=50K (0) or >50K (1) based on 10 features.
A logistic regression was used and the model maintained an 83% accuracy.

The script ends by writing the test data (including actual outcomes and
probabilities of the classifications) to csv.


"""

### Import statements for necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mt



### Read in dataset from http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

### Data preparation (address abberant values, normalization, one-hot encoding, )
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

#account for abberant data (missing and outlier values)
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

# normalize numeric values (at least 1 column)
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

#bin categorical variables (at least 1 column)
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

# construct new categorical variables (one-hot encoding; at least one column)
#sex
#check data
adult.loc[:,"sex"].value_counts()

#create dummy variables (one-hot encoding)
onehot_sex = pd.get_dummies(adult.loc[:, "sex"]).astype(int)

#add dummy variable columns to main data frame
adult[['female','male']] = onehot_sex.loc[:,[' Female',' Male']]

#check to see if new columns have been added
adult.head()


#remove obsolete columns
adult.drop('sex', axis = 1, inplace = True)

#check to make sure it is gone
adult.head()

### Ask binary-choice question
# Does the person have an income above or below 50K
# expert label = income
# Decision comments
# Asking a person's income based on other SES factors could benefit a business
# in many ways, including but not limited to selling products or services.

### Apply k-means
#check data
adult.dtypes
adult.head()

# perform a k-means with sklearn using at least one categorical
# attribute and one numeric attribute (normalized)
#normalize age using numpy
#minmaxscaled =(x - min(x))/(max(x) - min(x))
minmax_age = (adult.loc[:,'age'] - min(adult.loc[:,'age'])) / (max(adult.loc[:,'age']) - min(adult.loc[:,'age']))

#add normalized data as columm next to the original attribute
adult.insert(1, 'minmax_age',minmax_age)

#normalize hoursperweek using numpy
minmax_hrs = (adult.loc[:,'hoursperweek'] - min(adult.loc[:,'hoursperweek'])) / (max(adult.loc[:,'hoursperweek']) - min(adult.loc[:,'hoursperweek']))

#add normalized data as column next to the original attribute
adult.insert(15, 'minmax_hrs',minmax_hrs)

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
adult.insert(7, 'minmax_ms', minmax_ms)

# Perform k-means on one categorical and one numerical attribute 
# (age, maritalstatus)
AM = np.array(adult.loc[:,['minmax_age', 'minmax_ms']])

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
X = np.array(adult.loc[:,['age', 'hoursperweek']])

#visualize the data before clustering
plt.scatter(X[:,0], X[:,1], label = 'True Position')

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

# Add cluster labels to dataset
adult.insert(21, 'cluster_labels', kmeans.labels_)

### Split data into training/testing sets
### using the proper function in sklearn
# get rid of unecessary features
# these features were chosen either bcause there was no description
# of them provided in the original data set or they the normalized
# versions have been added as a column
adult2 = adult.drop(['age','workclass', 'education',
                     'occupation', 'relationship',
                     'race', 'nativecountry','fnlwgt', 'maritalstatus', 'capitalgain',
                     'capitalloss', 'hoursperweek'], axis = 1)

#change expert label into numeric
adult2.loc[adult2.loc[:, "income"] == " <=50K", "income"] = "0"
adult2.loc[adult2.loc[:, "income"] == " >50K", "income"] = "1"

adult2.loc[:,"income"] = pd.to_numeric(adult2.loc[:, "income"])

#check data
adult2.head()
adult2.dtypes

# split into labels and features
# everything but the 'income' column will become features
# for this model
y = adult2.income
x = adult2.drop('income', axis = 1)

# split into train and test data
# should be around a 80/20 split - check shape
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
print('x_train:')
print(x_train.head())
print(x_train.shape)
print('x_test:')
print(x_test.head())
print(x_test.shape)

# Create a classification model (logistic regression)
# for the expert label based on the training data
#start the model (default parameters)
logres = LogisticRegression()

#fit the model with data
logres.fit(x_train,y_train)

# apply your (trained) classifiers to the test data to predict probabilities
# predict from test
log_pred = logres.predict(x_test)

# classification probability
class_pred = logres.predict_proba(x_test)

#add outcomes to test set
x_test.insert(9, 'log_pred', log_pred)

#add classification probabilities
x_test.insert(10, 'class_pred_0', class_pred[:,0])
x_test.insert(11, 'class_pred_1', class_pred[:,1])

#add actual outcomes
x_test.insert(9, 'actual_outcomes', y_test)


#write to csv: test data, actual outcomes, and probabilities
#of your classification
x_test.to_csv(r'/Users/e.kubicek/Documents/UW/ProccessandTools/Unit3/Lesson8/adult_test_classification.csv')



# Determine accuracy rate (w/commentary)
print('The accuracy rate of this logistic regression is:', round(mt.accuracy_score(log_pred, y_test)*100), '%')

'''
The accuracy rate of the logistic regression model created in this script is
83%. After scouring the internet for guidance, I have come to see that the
threshold for a "good" accuracy rate varies dependent on the problem/question.
Given the dimension reduction I completed prior to running the model, I am
satisfied with an accuracy rate this high. The model would undoubtedly
benefit (i.e. increased accuracy) if working class and occupation variables (in 
numerical forms) were added to the features.
'''


# Summary comment on classification
'''
The goal of the current classification was to predict the categorical variable
of income, either <=50K or >50K, for each observation. A logistic regression
model was created with an accuracy of 84%. Future iterations of this work 
should consider making other categorical variables numerical, so they can be 
added to the model and see if that improves accuracy.

'''








