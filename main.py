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
from scipy import signal
import math
import random

df = pd.read_csv('data.csv', names=['In', 'Out'])  # data gets saved as dataframe format
print(df)

n = len(df['Out'])
print('The length of the data is: ' + str(n))

out = pd.Series.tolist(df['Out'])  # make list out of column
inp = pd.Series.tolist(df['In'])

m_est_out = sum(out)/n  # mean estimate
m_est_in = sum(inp)/n  # mean estimate

# --------------------------------------INPUT - OUTPUT PLOT----------------------------------------------------#
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True,figsize = (9,6))
# INPUT PLOT
ax1.plot(df['In'], label = 'In')
#ax1.set_xlim(1, 5000)
ax1.set_ylabel('Accelaration')
ax1.set_xlim(index[0], index[-1])
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
ax1.set_xlim(index[0], index[-1])
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
    plt.show()

portions(out)
portions(inp)
# -------------------------------------------------------------------------------------------------------------#

# -----------------------------------------------------PDF & HIST----------------------------------------------#
mu1 = m_est_in
mu2 = m_est_out

# use numpy built in function
var = (pd.DataFrame.var(df))
print(var)
sigma1 = math.sqrt(var[1])  # standard deviation
sigma2 = math.sqrt(var[2])

x = np.linspace(-1, 1, 100)

x = np.linspace(-1, 1, 100)
plt.figure(num=5, figsize=(18, 6))  # histogram
plt.hist(inp, bins=int(np.sqrt(n)),edgecolor='black', density='True', color = '#006B62')
plt.plot(x, (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-1 / 2 * ((x - mu1) / sigma1) ** 2),'--', linewidth=5,
	color = 'black')  # normal destribution pdf
plt.title('Input Histogram and Normal Destribution PDF')
plt.xlabel('Input Value')
# _ = plt.savefig('hist_IN.png', dpi = 100)
plt.show()
        
plt.figure(num=6, figsize=(18, 6))  # histogram
plt.hist(out, bins=int(np.sqrt(n)),edgecolor='black', density='True', color = '#6B0047')
plt.plot(x, (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-1 / 2 * ((x - mu2) / sigma2) ** 2),'--' ,linewidth=5,
	color = 'black')  # normal destribution pdf         
plt.title('Output Histogram and Normal Destribution PDF')        
plt.xlabel('Output Value')
# _ = plt.savefig('hist_OUT.png', dpi = 100)
plt.show()


print('Skewness for Input is estimated as: ' + str(df['In'].skew().round(4)))
print('Skewness for Output is estimated as: ' + str(df['Out'].skew().round(4)))
# -------------------------------------------------------------------------------------------------------------#

""" # ------------------------------------------------SCATTER------------------------------------------------------#
# TOO SLOW
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
        plt.show()
scat_k(inp, '#B23B7F')
scat_k(out, '#3B85B2')
# -------------------------------------------------------------------------------------------------------------#
"""
# -------------------------------------------------AUTOCORRELATION---------------------------------------------#
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,8))
_ = ax1.acorr(inp, maxlags = 100)

x = np.linspace(0,101)
a = 1.96/np.sqrt(n)
_ = ax1.plot(x, a*x**0, x, -a*x**0, color = 'red')
ax1.set_xlim(0,100)
ax1.set_ylim(-1,1)
ax1.set_xlim(0, 100)
ax1.set_ylabel('ρ_κ') ### use latex
ax1.set_title('Autocorrelation of Input Data')

plt.figure(figsize = (12,8))
_ = ax2.acorr(out, maxlags = 100)
_ = ax2.plot(x, a*x**0, x, -a*x**0, color = 'red')
ax2.set_xlim(0,100)
ax2.set_ylim(-1,1)
ax2.set_ylabel('ρ_κ') ### use latex
ax2.set_title('Autocorrelation of Output Data')
ax2.set_xlabel('Lag κ') ### use latex
plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------DFT---------------------------------------------------------#
samp_freq = 256

#DFT
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,8))
fft1 = np.fft.fft(inp,n)
fft1_m = np.abs(fft1)

freq = samp_freq/n * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype = 'int')
ax1.plot(freq[L],fft1_m[L], LineWidth = 2)
#ax1.set_xlim(freq[L[0]], freq[L[-1]])
ax1.set_yscale('log')
ax1.set_title('DFT')
ax1.set_ylabel("Metro (dB)")

fft2 = np.fft.fft(out,n)
fft2_m = np.abs(fft2)
ax2.plot(freq[L],fft2_m[L])
ax2.set_yscale('log')
ax2.set_ylabel("Metro (dB)")
ax2.set_xlabel("Frequency (Hz)")

df = (samp_freq/n)
print('Sample frequency Is: ' + str(df) + 'Hz')
# ------------------------------------------PERIODOGRAM--------------------------------------------------------#

#Periogram
fig,(ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,8))

f, PSD1 = signal.periodogram(inp, fs = samp_freq, detrend=False)
f, PSD2 = signal.periodogram(out, fs = samp_freq, detrend=False)

PSD1[0] = (PSD1[0]+PSD1[1])/2
PSD2[0] = (PSD2[0]+PSD2[1])/2

#PSD1 = fft1_m**2/n
#PSD2 = fft2_m**2/n

ax1.plot(f,PSD1, LineWidth = 2, color = 'coral')
#ax1.set_xlim(freq[L[0]], freq[L[-1]])
ax1.set_yscale('log')
ax1.set_title('PSD Estimation - Input')
ax1.set_ylabel("(dB)")
ax2.plot(f,PSD2, LineWidth = 2, color = 'coral')
#ax2.set_xlim(freq[L[0]], freq[L[-1]])
ax2.set_yscale('log')
ax2.set_title('PSD Estimation - Output')
ax2.set_ylabel("(dB)")

# -------------------------------------------------------------------------------------------------------------#

# ------------------------------------WELCH/ COMPARE SEGMENT LENGTHS-------------------------------------------#

# -------------------------------------------------------------------------------------------------------------#

# ---------------------------------------WELCH/ COMPARE OVERLAPS-----------------------------------------------#

# -------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------#
