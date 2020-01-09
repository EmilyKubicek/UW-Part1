#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:06:31 2019

@author: e.kubicekmcafee
"""

### import necessary packages
import requests
from bs4 import BeautifulSoup
import re

### read in an html page from a freely and easily
### available source on the internet - html page
### must contain at least 3 links
#source url
url = 'https://en.wikipedia.org/wiki/Gallaudet_University'

#use urllib to store html as a requesst object
response = requests.get(url)

#view page headers
print(response.headers)

#get page content
content = response.content
print(content)

#make into readable format (using lxml)
soup = BeautifulSoup(content,'lxml')

#now make it readable
print(soup.prettify())


### write code to tally the number of links in the html page
#find links on the page
all_a = soup.find_all("a")

# find only web links on the page (https)
all_a_https = soup.findAll('a', attrs={'href': re.compile("^http://")})

for x in all_a_https:
    print(x)

#pull out links and compile into dict
data = {}
for a in all_a_https:
    title = a.string
    data[title] = a.attrs['href']

### use print to present the tally
print('There are', len(data), 'total links in this html page.')
