import numpy as np
from Artef_sig import doublesin
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
sig, t = doublesin(20,50,fs,2)
# Define wavelets
wavelet = signal.morlet2
num_scales = 80

scales = np.arange(1,num_scales)

cwtmatr = signal.cwt(sig, wavelet, scales)
#cwtmatr_yflip = np.flipud(cwtmatr)

# Plot the results
plt.figure(figsize=(10,5))
plt.subplot(211)
plt.plot(t, sig)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.contourf(t, scales, np.abs(cwtmatr))
plt.colorbar()
plt.title('CWT with Morlet Wavelet')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.show()


