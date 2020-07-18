#!/usr/bin/env python
# coding: utf-8

# # ΣΤΟΧΑΣΤΙΚΑ ΣΗΜΑΤΑ ΚΑΙ ΣΥΣΤΗΜΑΤΑ
# ## ΘΕΜΑ ΕΞΑΜΗΝΟΥ
# ### ΑΛΕΞΑΝΔΡΟΣ ΔΗΜΑΣ
# ### Α.Μ.: 1054531

# ## Τμήμα Α - προκαταρκτικά

#---------------------------------------------------------------------------------------------------------#
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import math
import random
import statistics as st

plt.close('all')
df = pd.read_csv('data.csv', names=['In', 'Out'])  # data gets saved as dataframe format
print(df)

n = len(df['Out'])
out = pd.Series.tolist(df['Out'])  # make list out of column
inp = pd.Series.tolist(df['In'])
index = np.arange(0, n)
m_est_out = sum(out)/n  # mean estimate
m_est_in = sum(inp)/n  # mean estimate
samp_freq = 256

#---------------------------------------------------------------------------------------------------------#

# ### Α1. Προκαταρκτικά

print('The length of the data is: ' + str(n))

fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True,figsize = (9,6))
# INPUT PLOT
ax2.plot(df['In'], label = 'In', color  = 'tab:red')
#ax1.set_xlim(1, 5000)
ax2.set_ylabel('Accelaration')
ax2.set_xlim(index[0], index[-1])
ax2.set_title('Aircraft skeleton data 256Hz')
ax2.legend()

ax1.plot(df['Out'], label = 'Output')
#ax2.set_xlim(1, 5000)
ax1.set_ylabel('Accelaration')
ax1.set_xlabel('Time Interval')  # Check "Interval", Should time be used from frequency (256hz)
ax1.legend()
plt.tight_layout()
# plt.savefig('in_out_show.png', dpi = 100)
plt.show()

#---------------------------------------------------------------------------------------------------------#

# ### Α2. Κανονικοποίηση

out_center = []  # initialize empty list
in_center = []
for i in range(0,5000):  # fill list
    out_center.append(out[i] - m_est_out)  # center data
    in_center.append(inp[i]-m_est_in)

# CENTERED FIGURE
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True, figsize = (9,6))
# INPUT PLOT
ax2.plot(in_center, label = 'Input', color  = 'tab:red')
#ax1.set_xlim(1, 5000)
ax2.set_ylabel('Accelaration')
ax2.set_title('Aircraft skeleton data 256Hz - Centered Data')
ax2.set_xlim(index[0], index[-1])
ax2.legend()

ax1.plot(out_center, label = 'Output')
#ax2.set_xlim(1, 5000)
ax1.set_ylabel('Accelaration')
ax1.set_xlabel('Time Interval')  # Check "Interval", Should time be used from frequency (256hz)
ax1.legend()
plt.tight_layout()
plt.show()
# plt.savefig('in_out_show.png', dpi = 100)
#---------------------------------------------------------------------------------------------------------#

# ### Α3. Προκαταρκτική ανάλυση στο πεδίο χρόνου

fig, (ax2, ax3) = plt.subplots(nrows = 2, ncols = 1, figsize=(9, 8))
fig.suptitle('Portions of Output Signal', fontsize=16)


ax2.plot(out[3500:3601], 'o-',color = 'tab:red', label = 'Portion 1')  # FIX XTICKS
ax2.set_ylabel('Output', fontsize=14)
ax2.set_xticks(np.arange(0,101,20))
ax2.set_xticklabels(np.arange(3500,3601,20))
ax2.legend()

ax3.plot(out[1500:1551], 'o-', color = 'tab:green', label = 'Portion 2')  # FIX XTICKS
ax3.set_xlabel('Time Interval', fontsize=14)
ax3.set_ylabel('Output', fontsize=14)
ax3.set_xticks(np.arange(0,51,10))
ax3.set_xticklabels(np.arange(1500,1551,10))
ax3.legend()

# plt.savefig('portions.png')
plt.show()

#---------------------------------------------------------------------------------------------------------#
import scipy

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (12,8))
_,bins,_ = ax1.hist(out, bins=int(np.sqrt(n)), density='True', facecolor = 'tab:blue')
ax1.set_title('Histogram', fontsize = 16)
# ax1.set_xlabel('Bins', fontsize = 14)
ax1.set_ylabel('Frequency' ,fontsize = 14)

mu, sigma = scipy.stats.norm.fit(out)
y = scipy.stats.norm.pdf(bins, mu, sigma)
ax1.plot(bins, y, linewidth = 4, color = 'tab:red')

_ = scipy.stats.probplot(out, plot = ax2)
ax2.set_ylabel('Ordered Values', fontsize = 14)
ax2.set_xlabel('Theoritical Quantiles', fontsize = 14)

# _ = plt.savefig('hist.png')
plt.show()

print('Skewness for Output is estimated as: ' + str(df['Out'].skew().round(4)))
print('Escess kurtosis of Output is estimated as: ' + str(3 - df['Out'].kurt().round(4)))
#---------------------------------------------------------------------------------------------------------#
def scat_k(x):
     fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8,8), sharex = True, sharey = True)
     fig.suptitle('Scatter diagrams')
     k = [1,2,5,10]
     r = 0
     for i in range(0,2):
         for j in range(0,2):
             ax[i][j].set_xlabel('x(t-' + str(k[r]) + ')')
             ax[i][j].set_ylabel('x(t)')
             l=0
             m=k[r]
             while m<len(out)-1:
                 ax[i][j].scatter(x[l],x[m], color = 'tab:blue')
                 l+=1
                 m+=1
             # plt.tight_layout()
             r+=1
     plt.show()
     # plt.savefig('Scatter.png')

scat_k(out)
# plt.savefig('Scatter.png')
#---------------------------------------------------------------------------------------------------------#
# ## Τμήμα Β-μη παραμετρική ανάλυση

# ### Β1. Μη παραμετρική ανάλυση στο πεδίο χρόνου
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,6))
c = ax1.acorr(inp, maxlags = 100)

x = np.linspace(0,101)  # confidence interval lines
a = 1.96/np.sqrt(n)
ax1.plot(x, a*x**0, x, -a*x**0, color = 'tab:red')
ax1.set_xlim(0,100)
ax1.set_ylim(-0.6,0.6)
ax1.set_xlim(0, 100)
ax1.set_ylabel('ρ_κ') ### use latex
ax1.set_title('Autocorrelation of Input Data')

_ = ax2.acorr(out, maxlags = 100)
_ = ax2.plot(x, a*x**0, x, -a*x**0, color = 'tab:red')
ax2.set_xlim(0,100)
ax2.set_ylim(-.6,.6)
ax2.set_ylabel('ρ_κ') ### use latex
ax2.set_title('Autocorrelation of Output Data')
ax2.set_xlabel('Lag κ') ### use latex
plt.tight_layout()
# plt.savefig('ACF_signal.png')

plt
# Testing of white noise hypothesis for input
c = list(c[1])
c = [abs(num) for num in c]
indices = c>a
perob = sum(indices)/100
print(str(perob*100) + ' %' + 'of Input autocorrelation values are out of bound.')
print('We have similar outcome for more lags')
print('This indicates that the Input Is not white noise exacly')
# plt.savefig('ACF_signal.png')

#---------------------------------------------------------------------------------------------------------#

# ### Β2. Μη παραμετρική ανάλυση στο πεδίο συχνοτήτων

#-----------------------------------------------DFT------------------------------------------------------#

fig, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
fig.suptitle('Discrete Fourier Transform', fontsize = 16)

freq = samp_freq/n * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype = 'int')
fft2 = np.fft.fft(out,n)
fft2_m = np.abs(fft2)
ax2.plot(freq[L],fft2_m[L], label = 'Output')
ax2.set_yscale('log')
ax2.set_ylabel("Metro (dB)", fontsize = 14)
df = (samp_freq/(2*n)) # Sample frequency
print('Sample frequency for DFT Is: ' + str(df) + 'Hz')

#------------------------------------------Periogram---------------------------------------------------------#
fig,ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
fig.suptitle('Peridogram', fontsize = 16)
f, PSD1 = signal.periodogram(inp, fs = samp_freq, detrend=False)
f, PSD2 = signal.periodogram(out, fs = samp_freq, detrend=False)

PSD1[0] = (PSD1[0]+PSD1[1])/2
PSD2[0] = (PSD2[0]+PSD2[1])/2

#PSD1 = fft1_m**2/n
#PSD2 = fft2_m**2/n

ax2.plot(f,PSD2, LineWidth = 2, color = 'tab:red', label = 'Output')
#ax2.set_xlim(freq[L[0]], freq[L[-1]])
ax2.set_yscale('log')
ax2.set_ylabel("PSD (dB)", fontsize = 14)
ax2.set_xlabel("Frequency (Hz)", fontsize = 14)


df = samp_freq/(2*len(f))   # Sample frequency
print('frequency resolution for periodogram: %.3f' % df + ' Hz')
# plt.savefig('Periodogram.png')

#------------------------------------------Welch (2,2)---------------------------------------------------------#
fig,ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12,6), sharex = True, sharey=True)
fig.suptitle('PSD Estimation - Welch', fontsize = 16)
lengths = [128,256,512,1024]
overl = [0.5,0.75,0.95]

k = 0
l=0
dfl = []
for i in range (0,2):
    for j in range(0,2):
        f, PSD = signal.periodogram(out, fs = samp_freq, detrend=False)
        PSD[0] = (PSD[0]+PSD[1])/2
        ax[i][j].plot(f,PSD)

        nperseg = lengths[k]    # Segment length changes
        noverlap = overl[l] * nperseg   # Overlap remains the same
        f, PSD_w = signal.welch(out, fs = samp_freq, nperseg = nperseg, noverlap = noverlap, detrend=False)
        label = "Overlap = " + str(int(overl[l]*100)) + '%, ' + 'Segment Length = ' + str(lengths[k])
        ax[i][j].plot(f, PSD_w, color = 'tab:red', label = label)
        ax[i][j].set_yscale('log')
        ax[i][j].legend()

        df = samp_freq/(2*len(f))
        dfl.append(df)
        k = k+1
ax[0][0].set_ylabel("PSD (dB) Welch")
ax[1][0].set_ylabel("PSD (dB) Welch")
ax[1][0].set_xlabel("Frequency (Hz)")
ax[1][1].set_xlabel("Frequency (Hz)")

plt.show()
dfl = [round(df,3) for df in dfl]
print('Frequence resolutions: ' + str(dfl))

#--------------------------------Welch comparing segment lengths in a single plot------------------------------------#
fig,ax = plt.subplots(1,1, figsize = (12,6))
for i in range(1, len(lengths)+1):
    nperseg = lengths[-i]
    noverlap = 0.5 * nperseg
    f, PSD_w = signal.welch(out, fs = samp_freq, nperseg = nperseg, noverlap = noverlap, detrend=False)
    label = 'Segment Length = ' + str(lengths[-i]) + ', Overlap = 50%'
    ax.plot(f, PSD_w, label = label, lw = 2)
    ax.set_yscale('log')
    ax.set_title('Comparison between Different Segment Lengths L', fontsize = 16)
    ax.set_ylabel("PSD (dB) Welch", fontsize = 14)
    ax.set_xlabel("Frequency (Hz)", fontsize = 14)
    ax.legend()
plt.show()

fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
overl = [0.5,0.75,0.95]
f, PSD = signal.periodogram(out, fs = samp_freq, detrend=False)
PSD[0] = (PSD[0]+PSD[1])/2

#--------------------------------Welch comparing Overlaps in a single plot---------------------------------------------#
for i in range (0,3):
    nperseg = lengths[2]
    noverlap = overl[i] * nperseg
    f, PSD_w = signal.welch(out, fs = samp_freq, nperseg = nperseg, noverlap = noverlap, detrend=False)
    label = "Overlap = " + str(int(overl[-i]*100)) + '%, ' + 'Segment Length = ' + str(nperseg)
    ax.plot(f, PSD_w, label = label)
    ax.set_yscale('log')
    ax.set_title('Comparison between Different Overlaps k', fontsize = 16)
    ax.set_ylabel("PSD (dB) Welch", fontsize = 14)
    ax.set_xlabel("Frequency (Hz)", fontsize = 14)
    ax.legend()

plt.show()

noverlap = 0.5
#-----------------------------------------------------------------------------------------------------------------------#
fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
ax.plot(f, PSD_w, linewidth = 2, label = 'Estimation')
ax.set_yscale('log')
ax.set_title('Welch Estimation and CI', fontsize = 16)
ax.set_ylabel("PSD (dB) Welch", fontsize = 14)
ax.set_xlabel("Frequency (Hz)", fontsize = 14)
ax.legend()

k = (n-nperseg)/(nperseg*noverlap) + 1
# =============================================================================
x1 = 57.15 - 0.3*(57.15-48.76)
x2 = 106.63 - 0.3*(106.63-95.02)
# =============================================================================

llimit = 2*k*PSD_w/x1
ulimit = 2*k*PSD_w/x2

ax.plot(f, llimit, color = 'tab:red', linewidth = 1, label = 'Confidence Interval')
ax.plot(f, ulimit, color = 'tab:red', linewidth = 1)
ax.legend()
plt.show()
#---------------------------------------blackman - tuckey---------------------------------------------------------------#
# w_ = scipy.signal.get_window(window = 'hann', Nx = 1000, fftbins = True)
# w_ = list(w_)
# a = len(w_)
# w = []
# for i in range(1,6):
#     for j in np.arange(0,a):
#         w.append(w_[j])

# plt.plot(w)
# plt.show()

# fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
# # plt.plot(w)
# fbt, PSDbt = scipy.signal.periodogram(out, 256, window = w, nfft = 5000, detrend = False)
# ax.plot(fbt,PSDbt)
# ax.set_yscale('log')
# plt.show()

# ### Β3. Έλεγχος στασιμότητας
#--------------------------------Stationarity test----------------------------------------------------------------------#
# ACF Plots
q1 = out[0:2500]
q2 = out[2500:5000]

# import statsmodels
# tstat, pvalue = statsmodels.stats.weightstats.CompareMeans.ztest_ind(q1, q2, alternative = 'two-sided', usevar = 'unequal', value = 0)

mu1 = st.mean(q1)
mu2 = st.mean(q2)

fig,(ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,6))
fig.suptitle('Autocorrelation of two Segments', fontsize = 16)
c1 = ax1.acorr(q1, maxlags = 500)
c2 = ax2.acorr(q2, maxlags = 500)
ax1.set_xlim(0,100)
ax2.set_xlabel('lags κ', fontsize = 14)
ax1.set_ylabel('ACF of segment 1', fontsize = 14)
ax2.set_ylabel('ACF of segment 2', fontsize = 14)
plt.show()

# Test of hypothesis
# See B.2.5 eclass
# to so: Use statsmodels functions see line 346
# Mean

lags = c1[0]
n0 = len(lags)
acf1 = c1[1][:]
acf2 = c2[1][:]
l = 0
var1_ = 0
for lag in lags:
    s = (1-abs(lag)/n0)*acf1[l]
    var1_ = var1_ + s
    l+=1
var1 = var1_/n0

l = 0
var2_ = 0
for lag in lags:
    s = (1-abs(lag)/n0)*acf2[l]
    var2_ = var2_ + s
    l+=1
var2 = var2_/n0

def twoSampZ(X1, X2, mudiff, var1, var2):
	from numpy import sqrt, abs, round
	from scipy.stats import norm
	pooledSE = sqrt(var1 + var2)
	z = ((X1 - X2) - mudiff)/(pooledSE/sqrt(2500))
	pval = 2*(1 - norm.cdf(abs(z)))
	return round(z,2), round(pval,3)
# z1, p1 = twoSampZ(mu1, mu2, 0, var1, var2)

import scipy.stats as st
z1,p1 = st.ttest_ind(q1, q2, equal_var = False)

a = 0.05
if p1>a:
    print("Null Hypothesis can't be rejected,\nSample means could be the same (with 5% risk).")
else:
    print("Null Hypothesis can be rejected,\nSample means are not the same (with 5% risk).")

# Autocorrelation
vargk1 = []
vargk2 = []

#def vargk_est(acf):
for lag in lags:
    gamak = 0
    if lag<=0:
        sumrange = np.arange(-500-lag,500+lag)
    else:
        sumrange = np.arange(-500+lag,500-lag)
    for i in sumrange:
        gamak = gamak + acf1[i+500]**2 + acf1[i+lag+500]*acf1[i-lag+500]
    gamak = gamak/2500
    vargk1.append(gamak)

for lag in lags:
    gamak = 0
    if lag<=0:
        sumrange = np.arange(-500-lag,500+lag)
    else:
        sumrange = np.arange(-500+lag,500-lag)
    for i in sumrange:
        gamak = gamak + acf2[i+500]**2 + acf2[i+lag+500]*acf2[i-lag+500]
    gamak = gamak/2500
    vargk2.append(gamak)
#        return vargk

# vargk1 = vargk_est(acf1)
# vargk2 = vargk_est(acf1)

pvalues = []
for l in range(0,len(vargk1)):
    z, p = twoSampZ(acf1[l], acf2[l], 0, vargk1[l], vargk2[l])
    pvalues.append(p)
t = np.arange(0,len(pvalues))
ptest = pvalues<0.05*t**0
if sum(ptest)/len(ptest) >  0.05:
    print('Null hypothesis rejected, ACFs are not equal')
else:
    print('ACFs might be equal with risk 5%')
#-----------------------------------------------------------------------------------------------------------------------#
fr1, PSD1 = signal.welch(q1, fs = samp_freq, nperseg = 256, noverlap = 0.5)
fr2, PSD2 = signal.welch(q2, fs = samp_freq, nperseg = 256, noverlap = 0.5)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), sharex = True)
fig.suptitle('PSDs Comparison', fontsize = 16)
ax1.plot(fr1,PSD1)
ax1.set_yscale('log')
ax1.set_ylabel('PSD_1', fontsize = 14)
ax1.set_yticks([10**-6,10**-4,10**-2])
ax2.plot(fr2,PSD2)
ax2.set_yscale('log')
ax2.set_ylabel('PSD_2', fontsize = 14)
ax2.set_xlabel('Frequency (Hz)', fontsize = 14)
plt.show()

from scipy import stats as st
f1 = 1/1.92
f2 = 1.92

test = []
for l in np.arange(0, len(fr1)):
    f = PSD1[l]/PSD2[l]
    if f<=f1 or f>=f2:
        test.append(False)  #test rejected
    else:
        test.append(True)   #test accepted

if sum(test)/len(test) >= 0.95:
    print('PSDs are equal with risk 5%')
else:
    print('PSDs are unequal with risk 5%')

# ## Τμήμα Γ-παραμετρική ανάλυση AR

# ### Γ1. Προκαταρκτικά

# ### Γ2. Εκτίμηση μοντέλων AR

# ### Γ3. Έλεγχος εγκυρότητας του επιλεγέντος μοντέλου AR

# ### Γ4. Ανάλυση του επιλεγέντος μοντέλου AR

# ### Γ5. Πρόβλεψη βάσει του επιλεγέντος μοντέλου AR
