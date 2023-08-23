import pywt
import numpy as np
from Artef_sig import doublesin
import matplotlib.pyplot as plt



fs = 1000
sig, t = doublesin(20,50,fs,2)
#-----tmp----#
fs = 500
from Artef_sig import sigTotest
sig = sigTotest(fs=fs)
t = np.arange(0, len(sig)/fs, 1/fs)
#------------#
# Define wavelets
wavelet = 'morl'
num_scales = 80

# Perform CWT using PyWavelets
cwtmatr, freqs = pywt.cwt(sig, np.arange(1, num_scales),wavelet, method='conv')

# Plot the results
plt.figure(figsize=(10,5))
plt.subplot(211)
plt.plot(t, sig)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.contourf(t, np.arange(1, num_scales), np.abs(cwtmatr))
plt.colorbar()
plt.title('CWT with Morlet Wavelet')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.show()
