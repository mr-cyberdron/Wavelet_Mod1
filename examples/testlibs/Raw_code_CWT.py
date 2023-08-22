import numpy as np
from Artef_sig import doublesin
import matplotlib.pyplot as plt

def convolve(signal1, signal2, mode = None):
    print('conv_calc')
    assert len(signal1) == len(signal2)
    # Get the dimensions of the signals
    signal1_length = len(signal1)
    signal2_length = len(signal2)

    # Create an empty output signal
    output_signal = [0] * (signal1_length + signal2_length - 1)

    # Iterate over the output signal and apply the convolution
    for i in range(signal1_length + signal2_length - 1):
        for j in range(max(0, i - signal2_length + 1), min(i + 1, signal1_length)):
            output_signal[i] += signal1[j] * signal2[i - j]

    if mode == 'same':
        fromm = int(np.round(signal1_length/2)-1)
        to = int(signal1_length+fromm)
        output_signal = np.array(output_signal)[fromm:to]

    return output_signal


def morlet_wavelet(t, scale, omega0 = 5, norm = 2):
    scaled = None
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
        #cwt_matrix[ind, :] = np.convolve(signal, wavelet_data, mode='same')
        cwt_matrix[ind, :] = convolve(signal, wavelet_data, mode='same')
        if complex:
            cwt_matrix = cwt_matrix
        else:
            cwt_matrix = np.abs(cwt_matrix)
    return cwt_matrix

fs = 500
num_scales = 80
sig, t = doublesin(20,50,fs,1)
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

