#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:47:41 2019

@author: e.kubicekmcafee
###Summary comment block
"
Data cleaning and analysis in this script was completed on FiveThirtyEight's
Comic Characters Dataset retrieved from https://www.kaggle.com/fivethirtyeight/fivethirtyeight-comic-characters-dataset.

The original dataset contained a variety of numerical and categorical
variables but only the following were kept: 
    page_id
    name
    ID
    sex
    alive
    appearances
    year

In-depth descriptions of the variables can be found at the url above. 

Numeric variables (APPEARANCES, Year) were normalized using z-scaling after 
imputating medians for missing values. A column was added that indicates three
different bins the variable Year can be organized into.

Categorical variables (SEX, ALIVE) were decoded, imputed and consolidated. SEX
and ALIVE labels were replaced with less redundant naming (i.e. 'Living Character'
replaced with 'living'). SEX groups were consolidated into three categories
instead of five for clearer visualization in later plotting.

Attribute SEX were chosen to be plotted to clearly demonstrate the male
dominant theme of Marvel superhero comics. Also, by consolidating agender and
genderfluid labels, the group is able to be represented in a way that is actually
seen by visual plotting (i.e. the group was so small it did not show on visual
representations of the data).
    
"""

###import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###read in the dataset from source (https://www.kaggle.com/fivethirtyeight/fivethirtyeight-comic-characters-dataset#marvel-wikia-data.csv)
url = "/Users/e.kubicek/Documents/UW/Unit2/Lesson 5/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv"
cbc = pd.read_csv(url)

#ensure display is large enough
pd.options.display.max_columns = 1000

#check data
cbc.head()
cbc.shape
cbc.dtypes

#create data frame
cbc_df = pd.DataFrame(cbc)

#drop columns we are not working with in this script
cbc_df.drop(['urlslug','ALIGN','EYE','HAIR','GSM','FIRST APPEARANCE'],
             axis = 1, inplace = True)

#check to make sure what we expect is left
cbc_df.dtypes

###normalize numeric values (APPEARANCES, Year)
#APPEARANCES
#check to see if there are any missing values in APPEARANCES
cbc_df.loc[:,"APPEARANCES"].unique()

#locate nans
nans_appearances = np.isnan(cbc_df.loc[:,'APPEARANCES'])

#replace nans with median of attribute
cbc_df.loc[nans_appearances,"APPEARANCES"] = np.nanmedian(cbc_df.loc[:,"APPEARANCES"])

#check unique values (should be no NaNs)
cbc_df.loc[:,"APPEARANCES"].unique()

#normalize using numpy
zscaled_appearances = (cbc_df.loc[:,"APPEARANCES"] - np.mean(cbc_df.loc[:,"APPEARANCES"]))/np.std(cbc_df.loc[:,"APPEARANCES"])

#add zscaled data as column next to original attribute
cbc_df.insert(6, 'app_zscaled',zscaled_appearances)

#see if the new column has appeared
cbc_df.head()

#Year
#check to see if there are any missing values in Year
cbc_df.loc[:,"Year"].unique()

#locate nans
nans_year = np.isnan(cbc_df.loc[:,"Year"])

#replace nans with median of attribute
cbc_df.loc[nans_year,"Year"] = np.nanmedian(cbc_df.loc[:,"Year"])

#check unique values (should be no NaNs)
cbc_df.loc[:,"Year"].unique()

#normalize using numpy
zscaled_year = (cbc_df.loc[:,"Year"] - np.mean(cbc_df.loc[:,"Year"]))/np.std(cbc_df.loc[:,"Year"])

#add zscaled data as column next to original attribute
cbc_df['year_zscaled'] = zscaled_year

#see if the new column has appeared
cbc_df.head()

###Bin numeric variables (Year)
#by histogram - np.histogram(cbc_df.loc[:,"Year"], 3)
#more straight-forward way for obtaining the boundaries of the bins
#number of bins
nb= 3
bounds = np.linspace(np.min(cbc_df.loc[:,"Year"]), np.max(cbc_df.loc[:,"Year"]), nb + 1) 

#define function to bin Year into the proper bounds
# x = data array, b = boundaries array
def bin(x, b):
    nb = len(b)
    N = len(x)
    
    #empty integer array to store the bin numbers (output)
    y = np.empty(N, int)
    
    #repeat for each pair of bin boundaries
    for i in range(1, nb):
        y[(x >= b[i-1])&(x < b[i])] = i
    
    #ensure that the borderline cases are also binned appropriately
    y[x == b[-1]] = nb - 1
    return y

#call function on year attribute
bin((cbc_df.loc[:,"Year"]), bounds)

#add column of bins
cbc_df['year_bins'] = bin((cbc_df.loc[:,"Year"]), bounds)

#check to see if new column has appeared
cbc_df.head()

###decode categorical variables (SEX, ALIVE)
###impute missing categories
#SEX
#check for unique labels
cbc_df.loc[:,"SEX"].unique()

#decode labels
replace_male = cbc_df.loc[:,"SEX"] == "Male Characters"
cbc_df.loc[replace_male, "SEX"] = "male"

replace_female = cbc_df.loc[:,"SEX"] == "Female Characters"
cbc_df.loc[replace_female, "SEX"]= "female"

replace_agender = cbc_df.loc[:,"SEX"] == "Agender Characters"
cbc_df.loc[replace_agender, "SEX"]= "agender"

replace_genderfluid = cbc_df.loc[:,"SEX"] == "Genderfluid Characters"
cbc_df.loc[replace_genderfluid, "SEX"]= "genderfluid"

#replace nulls with string
cbc_df.loc[:,'SEX'] = cbc_df.loc[:,'SEX'].replace(np.nan, 'unknown', regex=True)

#get counts for each value to make sure our new string label is accounted for
cbc_df.loc[:,"SEX"].value_counts()

#make sure all nulls are gone
cbc_df.loc[:,'SEX'].isnull().sum()

#ALIVE
#check for unique labels
cbc_df.loc[:,"ALIVE"].unique()

#decode labels
replace_living = cbc_df.loc[:,"ALIVE"] == "Living Characters"
cbc_df.loc[replace_living, "ALIVE"] = "living"

replace_deceased = cbc_df.loc[:, "ALIVE"] == "Deceased Characters"
cbc_df.loc[replace_deceased, "ALIVE"] = "deceased"

#replace nulls with string
cbc_df.loc[:,"ALIVE"] = cbc_df.loc[:,"ALIVE"].replace(np.nan, "unknown", regex = True)

#get counts for each value to make sure our new string label is accounted for
cbc_df.loc[:,"ALIVE"].value_counts()

#make sure all nulls are gone
cbc_df.loc[:,'ALIVE'].isnull().sum()

###Consolidate categorical data (SEX)
#check SEX labels
cbc_df.loc[:,"SEX"].value_counts()

#consolidate agender and genderfluid to nonbinary
cbc_df.loc[cbc_df.loc[:, "SEX"] == "agender", "SEX"] = "nonbinary"
cbc_df.loc[cbc_df.loc[:, "SEX"] == "genderfluid", "SEX"] = "nonbinary"

#check to see if our new category is there and correct (should be 47)
cbc_df.loc[:,"SEX"].value_counts()

###One-hot encode categorical data with at least 3 categories (ALIVE)
#check data
cbc_df.loc[:,"ALIVE"].value_counts()

#Creating dummy variables
onehot_alive = pd.get_dummies(cbc_df.loc[:, "ALIVE"]).astype(int)

#add dummy variable columns to main data frame
cbc_df[['deceased','living','unknown']] = onehot_alive.loc[:,['deceased','living','unknown']]

###remove obsolete columns
cbc_df.drop('ALIVE', axis = 1, inplace = True)

#check to make sure it is gone
cbc_df.head()

###present plot for categorical column (SEX)
print(cbc_df.loc[:,"SEX"].value_counts().plot(kind = "pie", figsize = (6,6), 
          title = "Gender Breakdown of Marvel Superheroes (%)", autopct='%.2f'))





