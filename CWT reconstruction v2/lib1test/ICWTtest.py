from wavelets import WaveletAnalysis
import numpy as np
import matplotlib.pyplot as plt

#pip install git+https://github.com/endolith/wavelets

time = np.linspace(0, 1, 400)
signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 10 * time)+ np.sin(2 * np.pi * 8 * time)*1

dt = 0.0025

wa = WaveletAnalysis(signal, dt=dt)

# wavelet power spectrum
power = wa.wavelet_power

# scales
scales = wa.scales


# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()

plt.figure()
plt.subplot(3,1,1)
plt.plot(signal)
plt.subplot(3,1,2)
plt.plot(rx)
plt.subplot(3,1,3)
T, S = np.meshgrid(t, scales,)
plt.contourf(T, S, power, 100)
plt.colorbar()
plt.show()