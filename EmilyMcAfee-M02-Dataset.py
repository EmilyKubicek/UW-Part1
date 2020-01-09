#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:24:52 2019

@author: e.kubicekmcafee
"""
#import packages
import pandas as pd
import numpy as np

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

#################################################################
###Export to CSV file
adult.to_csv(r'/Users/e.kubicek/Documents/UW/Unit2/Lesson6/EmilyMcAfee-M02-Dataset.csv')



