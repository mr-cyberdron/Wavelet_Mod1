import numpy as np
import matplotlib.pyplot as plt

def morlet_wavelet(t, scale, omega0 = 5):
    t = np.arange(0, t) - (t - 1.0) / 2
    t = t/scale
    wavelet =  np.pi ** (-0.25) *((np.exp(1j*omega0*t)-np.exp(-(omega0**2/2)))*np.exp(-(t**2/2)))
    scaled = np.sqrt(1/scale) * wavelet
    # scaled =wavelet
    return scaled

def morlet_wavelet2(t, scale):
    return np.pi**(-0.25) * np.exp(1j * 6 * t / scale) * np.exp(-t**2 / (2 * scale**2))

t = 500

for i in [1,10,100,245]:
    wavelet = morlet_wavelet(t, i+1)

    plt.plot(np.arange(0, t), wavelet.real, label='Real part')
    #plt.plot(t, wavelet.imag, label='Imaginary part')
    plt.legend()
    plt.show()
