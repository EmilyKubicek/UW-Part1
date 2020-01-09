#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 09:56:00 2019

@author: e.kubicekmcafee
"""

###Lesson 3 Assignment
###Prepare data by identifying and dealing with aberrant data

#Import statements from necessary packages
import numpy as np

#Create a numeric 30 element numpy array w/outliers
arr1 = np.array([2, 6, 3, 77, 3, 7, 2, 2, 4, 5, 6, 8, 9, 9, 10, 14, 22,
              35, 35, 28, 4, 6, 4, 64, 5, 92, 20, 19, 18, 26])

#Create an array that has improper non-numeric missing values
arr2 = np.array([5, 32, 6, 23, 7, 9, 99, "?", 3, 2, 22, " ", "seven",
                " ", "NA", 55, 51, 30 , 29, 32, 40, 20, 41, 13, 12, 3,
                3, 7, '??', 1])

#Define a fx that removes outliers in arr1
def remove_outlier(arg1):
    
    #establish outlier limits
    lh = np.mean(arg1) + 2*np.std(arg1)
    ll = np.mean(arg1) - 2*np.std(arg1)
    
    #index elements within limits
    fg = (arg1 < lh) & (arg1 > ll)
    
    #keep elements within limits
    arg1 = arg1[fg]
    
    #return new array w/o outliers
    return(arg1)

#Define a function that replaces outliers w/mean of non-outliers
def replace_outlier(arg1):
    
    #establish outlier limits
    lh2 = np.mean(arg1) + 2*np.std(arg1)
    ll2 = np.mean(arg1) - 2*np.std(arg1)
    
    #index elements outside of limits (outliers)
    fb = (arg1 > lh2) | (arg1 < ll2)
    
    #index elements within limits
    fg2 = ~fb
    
    #change elements outside of limits to mean of those within limits
    arg1[fb] = np.mean(arg1[fg2].astype(float))
    
    #return array with imputations
    return(arg1)

#Define a function that fills missing values in arr2 with arr2 median
def fill_median(arg1):
    
    #identify numeric values
    numval = (arg1 != " ") & (arg1 != "?") & (arg1 != "??") & (arg1 != "NA") & (arg1 != "seven")
    
    #identify non-numeric values
    nonnumval = ~numval
    
    #replace non-numeric values
    arg1[nonnumval] = np.median(arg1[numval].astype(float))
    arg1 = arg1.astype(float)
    
    return(arg1)
    

#Call remove_outlier(arr1)
remove_outlier(arr1)

#Call replace_outlier(arr1)
replace_outlier(arr1)

#Call fill_median(arr2)
fill_median(arr2)
    
