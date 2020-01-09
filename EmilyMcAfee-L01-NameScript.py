#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:22:22 2019

@author: e.kubicekmcafee
"""
### Workplace scenario where the IT department needs to test my
### computer/software - the following script will produce my name
### and the current date/time in which it is ran

##function that does not require arguments/returns author's name as string
def my_name():
    return("Emily")

#call to my_name using print statement
print(my_name())


##import statement for dt that uses 'as'
import datetime as dt

#make fx using dt package/returns string of current date+time
def date_and_time():
    return (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

#call new fx with print statement
print(date_and_time())

