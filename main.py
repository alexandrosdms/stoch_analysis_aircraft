# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:15:50 2020

@author: Alexandros Dimas
University of Patras
Stochastic Signals & Systems 
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

df = pd.read_csv('data.csv', names = ['In', 'Out'])		#data gets saved as dataframe fprmat
print(df)

n = len(df['Out'])
print('The length of the data is: ' + str(n))

#INPUT - OUTPUT PLOT
plt.figure(num = 1, figsize = (12,6))
plt.subplot(2,1,1)		#INPUT PLOT
plt.plot(df['In'])
plt.xlim(1,5000)
plt.title('Input - Output')
plt.ylabel('Input')

plt.subplot(2,1,2)		#OUTPUT PLOT
plt.plot(df['Out'])
plt.xlim(1,5000)
plt.ylabel('Output')
plt.xlabel('Time Interaval')	##Check "Interval", Should time be used from frequency (256hz)
#plt.savefig('in_out_show.png', dpi = 100)

out = pd.Series.tolist(df['Out'])			#make list out of column
mean_est = sum(out)							#mean estimate
out_center = []								#initialize empty list
for index in out:							#fill list
    out_center.append(index - mean_est)		#center data

plt.figure(num = 3, figsize = (9,3), dpi = 100.0)		#CENTERED FIGURE
plt.plot(out_center)
plt.xlim(1,5000)
plt.title('Output Centered')
plt.xlabel('Time Interval')		##"Interval"??
#plt.savefig('out_center.png', dpi = 100)

##MAKE WITH fig,ax = plt.figure() COMMAND (TITLES, AXIS LABELS)#
plt.figure(num = 4, figsize = (9,3))    #plotting whole signal & portions
plt.subplot(3,1,1)                      #whole
plt.plot(df['Out'])
plt.xlim(1,5000)
#plt.title('Output')
#plt.ylabel('Out')

plt.subplot(3,1,2)                      #porsion 1
plt.plot(out[3500:4000])                ##FIX XTICKS
#plt.title('A porsion of the signal')
#plt.xlabel('Time Interval')
#plt.ylabel('Out')

plt.subplot(3,1,3)                      #posion 2
plt.plot(out[1500:2000])                ##FIX XTICKS
#plt.title('A porsion of the signal')
#plt.xlabel('Time Interval')
#plt.ylabel('Out')

#plt.savefig('porsion.png', dpi = 100)
#pdf
mu = mean_est

var = 0								#initialize
for i in out:
    var = var + (i - mu)**2/(n-1)	#estimate of variance

sigma = math.sqrt(var)				#standard deviation
x = np.linspace(-1, 1, 100)			
plt.figure(num = 5, figsize = (18,6))	#histogram
plt.hist(out, bins = int(np.sqrt(n)), 
	edgecolor = 'black', density = 'True')
plt.plot(x, (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-1/2 * ((x-mu)/sigma)**2),
	color = 'red', linewidth = 3)		#normal destribution pdf
plt.title('Output Histogram and Normal Destribution PDF')
plt.xlabel('Output Value')

#_ = plt.savefig('hist.png', dpi = 100)

##FIX THESE
inp = pd.Series.tolist(df['In']) #Scater Diagrams
plt.figure(figsize = (12,10))
plt.scatter(inp[:],out[:])
plt.title('Scatter Diagram 1')
plt.xlabel('Input')
plt.ylabel('Output')

plt.figure(figsize = (12,10))
plt.scatter(inp[0:200],out[0:200])
plt.title('Scatter Diagram 2 - first 200 mesurements')
plt.xlabel('Input')
plt.ylabel('Output')