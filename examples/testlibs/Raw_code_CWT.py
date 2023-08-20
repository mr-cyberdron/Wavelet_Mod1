import numpy as np
from Artef_sig import doublesin
import matplotlib.pyplot as plt

def morlet_wavelet(t, scale, omega0 = 5, norm = 2):
    t = np.arange(0, t) - (t - 1.0) / 2
    t = t / scale
    if norm == 1:
        wavelet =  np.pi ** (-0.25) *((np.exp(1j*omega0*t)-np.exp(-(omega0**2/2)))*np.exp(-(t**2/2)))
        scaled = np.sqrt(1/scale) * wavelet
    if norm == 2:
        scaled = np.pi ** (-0.25) * ((np.exp(1j * omega0 * t) - np.exp(-(omega0 ** 2 / 2))) * np.exp(-(t ** 2 / 2)))
    return scaled

def cwt(signal, scales, complex = False):
    n = len(signal)
    cwt_matrix = np.zeros((len(scales), n),dtype='complex_')
    for ind, scale in enumerate(scales):
        wavelet_data = np.conj(morlet_wavelet(n, scale))
        cwt_matrix[ind, :] = np.convolve(signal, wavelet_data, mode='same')
        if complex:
            cwt_matrix = cwt_matrix
        else:
            cwt_matrix = np.abs(cwt_matrix)
    return cwt_matrix

fs = 1000
num_scales = 80
sig, t = doublesin(20,50,fs,2)
scales = np.arange(1, num_scales)
cwtmatr = cwt(sig, scales)


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

