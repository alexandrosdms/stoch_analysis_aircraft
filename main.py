# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:15:50 2020

@author: Alexandros Dimas
1054531
University of Patras
Stochastic Signals & Systems
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

df = pd.read_csv('data.csv', names=['In', 'Out'])  # data gets saved as dataframe format
print(df)

n = len(df['Out'])
print('The length of the data is: ' + str(n))

# --------------------------------------INPUT - OUTPUT PLOT----------------------------------------------------#
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True,figsize = (9,6))
# INPUT PLOT
ax1.plot(df['In'], label = 'In')
#ax1.set_xlim(1, 5000)
ax1.set_ylabel('Accelaration')
ax1.set_title('Aircraft skeleton data 256Hz')
ax1.legend()

ax2.plot(df['Out'], color  = 'coral', label = 'Output')
#ax2.set_xlim(1, 5000)
ax2.set_ylabel('Accelaration')
ax2.set_xlabel('Time Interval')  # Check "Interval", Should time be used from frequency (256hz)
ax2.legend()
plt.tight_layout()
# plt.savefig('in_out_show.png', dpi = 100)
# -------------------------------------------------------------------------------------------------------------#

out = pd.Series.tolist(df['Out'])  # make list out of column
inp = pd.Series.tolist(df['In'])

m_est_out = sum(out)/n  # mean estimate
m_est_in = sum(inp)/n  # mean estimate

out_center = []  # initialize empty list
in_center = []
for i in range(0,5000):  # fill list
    out_center.append(out[i] - m_est_out)  # center data
    in_center.append(inp[i]-m_est_in)

# ------------------------------------------CENTERED GRAPHS----------------------------------------------------#
# CENTERED FIGURE
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True, figsize = (9,6))
# INPUT PLOT
ax1.plot(in_center, label = 'Input')
#ax1.set_xlim(1, 5000)
ax1.set_ylabel('Accelaration')
ax1.set_title('Aircraft skeleton data 256Hz - Centered Data')
ax1.legend()

ax2.plot(out_center, color  = 'coral', label = 'Output')
#ax2.set_xlim(1, 5000)
ax2.set_ylabel('Accelaration')
ax2.set_xlabel('Time Interval')  # Check "Interval", Should time be used from frequency (256hz)
ax2.legend()
plt.tight_layout()
# plt.savefig('in_out_show.png', dpi = 100)

# ---------------------------PORSIONS OF THE SIGNAL AND WHOLE SIGNAL--------------------------------------------#
def portions(x):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize=(9, 8))
    ax1.plot(x, label = 'Whole Data')

    if x == out:
        ax1.set_title('Output Data')
    else:
        ax1.set_title('Input Data')
    
    ax1.set_ylabel('Accelaration')
    ax1.legend()
    ax2.plot(x[3500:3521], 'o-',color = 'coral', label = 'Portion 1')  # FIX XTICKS
    ax2.set_ylabel('Accelaration')
    ax2.set_xticks(np.arange(0,21,2))
    ax2.set_xticklabels(np.arange(3500,3521,2))
    ax2.legend()

    ax3.plot(x[1500:1541], 'o-', color = 'tomato', label = 'Portion 2')  # FIX XTICKS
    ax3.set_xlabel('Time Interval')
    ax3.set_ylabel('Accelaration')
    ax3.set_xticks(np.arange(0,41,5))
    ax3.set_xticklabels(np.arange(1500,1541,5))
    ax3.legend()
    # plt.savefig('porsion.png', dpi = 100)

portions(out)
portions(inp)
print(type(ax1))
# -------------------------------------------------------------------------------------------------------------#

# -----------------------------------------------------PDF & HIST----------------------------------------------#
mu1 = m_est_in
mu2 = m_est_out

# use numpy built in function
var = (pd.DataFrame.var(df))
print(var)
sigma1 = math.sqrt(var[0])  # standard deviation
sigma2 = math.sqrt(var[1])

x = np.linspace(-1, 1, 100)
plt.figure(num=5, figsize=(18, 6))  # histogram
plt.hist(inp, bins=int(np.sqrt(n)),edgecolor='black', density='True', color = '#006B62')
plt.plot(x, (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-1 / 2 * ((x - mu1) / sigma1) ** 2),'--', linewidth=5,
	color = 'black')  # normal destribution pdf
plt.title('Input Histogram and Normal Destribution PDF')
plt.xlabel('Input Value')
# _ = plt.savefig('hist_IN.png', dpi = 100)
        
plt.figure(num=6, figsize=(18, 6))  # histogram
plt.hist(out, bins=int(np.sqrt(n)),edgecolor='black', density='True', color = '#6B0047')
plt.plot(x, (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-1 / 2 * ((x - mu2) / sigma2) ** 2),'--' ,linewidth=5,
	color = 'black')  # normal destribution pdf         
plt.title('Output Histogram and Normal Destribution PDF')        
plt.xlabel('Output Value')
# _ = plt.savefig('hist_OUT.png', dpi = 100)

print('Skewness for Input is estimated as: ' + str(df['In'].skew().round(4)))
print('Skewness for Output is estimated as: ' + str(df['Out'].skew().round(4)))
# -------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------SCATTER------------------------------------------------------#
import random

dummy = True
def scat_k(x,col): 
    r = 1
    plt.figure(figsize = (10,10))
    for k in [10,100,3000,4500]:
        plt.subplot(2,2,r)
        plt.xlabel('x(t-' + str(k) + ')')
        plt.ylabel('x(t)')
        plt.title('Scatter ' + str(r))
        r+=1
        
        l=0
        m=k
        while m<len(out)-1:
            plt.scatter(x[l],x[m], color = col)
            l+=1
            m+=1
        plt.tight_layout()
        
scat_k(inp, '#B23B7F')
scat_k(out, '#3B85B2')
# -------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------AUTOCORRELATION---------------------------------------------#
plt.figure(figsize=(12, 10))
plt.acorr(out)
plt.show()
# -------------------------------------------------------------------------------------------------------------#
