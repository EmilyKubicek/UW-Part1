#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA)in Python
Created on Sat Feb  2 07:35:26 2019

The following analysis is compelted on a data set from
http://archive.ics.uci.edu/ml/datasets/Automobile. it contains information on
cars made in 1985. This EDA is focused on properly importing, cleaning, 
organizing, and imputating data for messy numerical attributes.

See histograms and scatterplots of relevant data as well as a summary
of EDA decisions at end of script.

Attribute information: 
1. symboling: -3, -2, -1, 0, 1, 2, 3. 
2. normalized-losses: continuous from 65 to 256. 
3. make:  alfa-romero, audi, bmw, chevrolet, dodge, honda, 
isuzu, jaguar, mazda, mercedes-benz, mercury, 
mitsubishi, nissan, peugot, plymouth, porsche, 
renault, saab, subaru, toyota, volkswagen, volvo 
4. fuel-type: diesel, gas. 
5. aspiration: std, turbo. 
6. num-of-doors: four, two. 
7. body-style: hardtop, wagon, sedan, hatchback, convertible. 
8. drive-wheels: 4wd, fwd, rwd. 
9. engine-location: front, rear. 
10. wheel-base: continuous from 86.6 120.9. 
11. length: continuous from 141.1 to 208.1. 
12. width: continuous from 60.3 to 72.3. 
13. height: continuous from 47.8 to 59.8. 
14. curb-weight: continuous from 1488 to 4066. 
15. engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor. 
16. num-of-cylinders: eight, five, four, six, three, twelve, two. 
17. engine-size: continuous from 61 to 326. 
18. fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi. 
19. bore: continuous from 2.54 to 3.94. 
20. stroke: continuous from 2.07 to 4.17. 
21. compression-ratio: continuous from 7 to 23. 
22. horsepower: continuous from 48 to 288. 
23. peak-rpm: continuous from 4150 to 6600. 
24. city-mpg: continuous from 13 to 49. 
25. highway-mpg: continuous from 16 to 54. 
26. price: continuous from 5118 to 45400.
(retrieved from http://archive.ics.uci.edu/ml/datasets/Automobile)

@author: e.kubicekmcafee
"""
###import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###read in dataset
source = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

#change display sz
pd.set_option('display.max_colwidth',25)

#make data frame, keep header names
auto = pd.read_csv(source, header = None)

#check first rows/shape of data
auto.head()
auto.shape
auto.dtypes

###assign reasonable column names
auto.columns =["symbol", 'normloss', 'make','fuel-type','aspiration',
'numdoors','body','drive-wheels','engineloc', 'wheel-base', 'length',
'width','height', 'curb-weight', 'engine-type','numcyl','enginesize',
'fuelsys', 'bore', 'stroke', 'compratio', 'horsepower', 'peakrpm',
'citympg', 'hwmpg','price']

#replace qms with nans
auto = auto.replace(to_replace="?", value=float("NaN"))

#check for missing/aberrant numeric data
auto.isnull().sum()

###impute and assign median values for missing numeric values as indicated from auto.isnull().sum()
#normloss
#convert normloss to numeric data including nans
auto.loc[:,'normloss'] = pd.to_numeric(auto.loc[:,"normloss"], errors = 'coerce')

#determine location of nans
hasnan = np.isnan(auto.loc[:,'normloss'])

#determine median
normlossmedian = np.nanmedian(auto.loc[:, 'normloss'])

#impute median in place of NaNs
auto.loc[hasnan,'normloss'] = normlossmedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'normloss']))

#numdoors
#convert number strings to num data type
auto.loc[:,'numdoors'] = auto.loc[:,'numdoors'].replace("two",2)
auto.loc[:,'numdoors'] = auto.loc[:,'numdoors'].replace("four",4)

#convert numdoors to numeric data including nans
auto.loc[:,'numdoors'] = pd.to_numeric(auto.loc[:,"numdoors"], errors = 'coerce')

#determine location of nans
hasnan2 = np.isnan(auto.loc[:,'numdoors'])

#determine median
numdoorsmedian = np.nanmedian(auto.loc[:,'numdoors'])

#impute median in place of NaNs
auto.loc[hasnan2,'numdoors'] = numdoorsmedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'numdoors']))

#bore
#convert to num including nans
auto.loc[:,'bore']= pd.to_numeric(auto.loc[:,'bore'], errors = 'coerce')

#determine location of nans
hasnan3 = np.isnan(auto.loc[:,'bore'])

#determine median
boremedian = np.nanmedian(auto.loc[:,'bore'])

#impute median in place of NaNs
auto.loc[hasnan3, 'bore'] = boremedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'numdoors']))

#peakrpm
#convert to num including nans
auto.loc[:,'peakrpm'] = pd.to_numeric(auto.loc[:,'peakrpm'], errors = 'coerce')

#determine location of nans
hasnan4 = np.isnan(auto.loc[:,'peakrpm'])

#determine median
peakrpmmedian = np.nanmedian(auto.loc[:,'peakrpm'])

#impute median in place of NaNs
auto.loc[hasnan4, 'peakrpm'] = peakrpmmedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'peakrpm']))

#price
#convert to num including nans
auto.loc[:,'price'] = pd.to_numeric(auto.loc[:,'price'], errors = 'coerce')

#determine location of NaNs
hasnan5 = np.isnan(auto.loc[:,'price'])

#determine median
pricemedian= np.nanmedian(auto.loc[:,'price'])

#impute median in place of NaNs
auto.loc[hasnan5,'price'] = pricemedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'peakrpm']))

#stroke
#convert to num including nans
auto.loc[:,'stroke'] = pd.to_numeric(auto.loc[:,'stroke'], errors = 'coerce')

#determine location of nans
hasnan6 = np.isnan(auto.loc[:,'stroke'])

#determine median
strokemedian= np.nanmedian(auto.loc[:,'stroke'])

#impute median in place of NaNs
auto.loc[hasnan6,'stroke'] = strokemedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'stroke']))

#horsepower
#convert to num including nans
auto.loc[:,'horsepower'] = pd.to_numeric(auto.loc[:,'horsepower'], errors = 'coerce')

#determine location of nans
hasnan7 = np.isnan(auto.loc[:,'horsepower'])

#determine median
horsepowermedian= np.nanmedian(auto.loc[:,'horsepower'])

#impute median in place of NaNs
auto.loc[hasnan7,'horsepower'] = strokemedian

#check to see if any nans are left
sum(np.isnan(auto.loc[:,'horsepower']))

###replace outliers of num attributes
###create histogram
###determine std of each num variable
#determine num attributes
auto.dtypes

#horsepower
lh = np.mean(auto.loc[:,'horsepower']) + 2*np.std(auto.loc[:,'horsepower'])
ll = np.mean(auto.loc[:,'horsepower']) - 2*np.std(auto.loc[:,'horsepower'])
#flag values within limits
fg = (auto.loc[:,'horsepower'] >= ll) & (auto.loc[:,'horsepower'] <= lh)
sum(~fg)
#replace outliers with median
auto.loc[~fg,'horsepower'] = np.median(auto.loc[:,'horsepower'])
#create histogram
plt.hist(auto.loc[:,'horsepower'])
plt.show(plt.hist(auto.loc[:,'horsepower']))
#determine std
print("std =",np.std(auto.loc[:,'horsepower']))

#peakrpm
lh2 = np.mean(auto.loc[:,'peakrpm']) + 2*np.std(auto.loc[:,'peakrpm'])
ll2 = np.mean(auto.loc[:,'peakrpm']) - 2*np.std(auto.loc[:,'peakrpm'])
#flag values within limits
fg2 = (auto.loc[:,'peakrpm'] >= ll2) & (auto.loc[:,'peakrpm'] <= lh2)
sum(~fg2)
#replace outliers with median
auto.loc[~fg2,'peakrpm'] = np.median(auto.loc[:,'peakrpm'])
#create histogram
plt.hist(auto.loc[:,'peakrpm'])
plt.show(plt.hist(auto.loc[:,'peakrpm']))
#determine std
print("std =",np.std(auto.loc[:,'peakrpm']))

#citympg
lh3 = np.mean(auto.loc[:,'citympg']) + 2*np.std(auto.loc[:,'citympg'])
ll3 = np.mean(auto.loc[:,'citympg']) - 2*np.std(auto.loc[:,'citympg'])
#flag values within limits
fg3 = (auto.loc[:,'citympg'] >= ll3) & (auto.loc[:,'citympg'] <= lh3)
sum(~fg3)
#replace outliers with median
auto.loc[~fg3,'citympg'] = np.median(auto.loc[:,'citympg'])
#create histogram
plt.hist(auto.loc[:,'citympg'])
plt.show(plt.hist(auto.loc[:,'citympg']))
#determine std
print("std =",np.std(auto.loc[:,'citympg']))

#hwmpg
lh4 = np.mean(auto.loc[:,'hwmpg']) + 2*np.std(auto.loc[:,'hwmpg'])
ll4 = np.mean(auto.loc[:,'hwmpg']) - 2*np.std(auto.loc[:,'hwmpg'])
#flag values within limits
fg4 = (auto.loc[:,'hwmpg'] >= ll4) & (auto.loc[:,'hwmpg'] <= lh4)
sum(~fg4)
#replace outliers with median
auto.loc[~fg4,'hwmpg'] = np.median(auto.loc[:,'hwmpg'])
#create histogram
plt.hist(auto.loc[:,'hwmpg'])
plt.show(plt.hist(auto.loc[:,'hwmpg']))
#determine std
print("std =",np.std(auto.loc[:,'hwmpg']))

#price
lh5 = np.mean(auto.loc[:,'price']) + 2*np.std(auto.loc[:,'price'])
ll5 = np.mean(auto.loc[:,'price']) - 2*np.std(auto.loc[:,'price'])
#flag values within limits
fg5 = (auto.loc[:,'price'] >= ll5) & (auto.loc[:,'price'] <= lh5)
sum(~fg5)
#replace outliers with median
auto.loc[~fg5,'price'] = np.median(auto.loc[:,'price'])
#create histogram
plt.hist(auto.loc[:,'price'])
plt.show(plt.hist(auto.loc[:,'price']))
#determine std
print("std =",np.std(auto.loc[:,'price']))

###create a scatterplot
plt.scatter(auto.loc[:,'price'],auto.loc[:,'citympg'], c = 'r', alpha =.5, label = 'City')
plt.scatter(auto.loc[:,'price'],auto.loc[:,'hwmpg'], c = 'b', alpha =.5, label = 'Highway')
plt.title('Pricing & Mileage for 1985 Automobiles')
plt.xlabel('Price ($)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.show()

###summary on how numeric variables have been treated
'''
Any numeric attributes that were determined to have missing data were imputed 
with the respective attribute's median. Since this data analysis was exploratory 
(EDA) I wanted to clean all numeric variables and keep all attributes within the 
dataset before determining which to include in later analyses. No rows were 
removed for this same reason. 
 
Attributes numdoors, bore, peakrpm, price, stroke, 
and horsepower were should all be numeric data types but were originally object 
datatypes, as they contained '?'s. Thus, these attributes required coercion 
into num data types (including NaNs) and imputation of their respective medians.
 
Horsepower, peakrpm, citympg, hwmpg, and price were histogrammed in this EDA. 
These attributes were chosen due to the typical interest general population 
have in these characteristics of automobiles (i.e. RPM, MPG, and Price) when 
purchasing and selling.
'''











