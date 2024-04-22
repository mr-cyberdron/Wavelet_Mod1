from Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
from Artifitial_signal_creation import sigTotest
import matplotlib.pyplot as plt
# from CWT_basis import cwt,icwt,morlet_wavelet,scale_to_frequency
from CWT_mod1 import cwt,icwt,morlet_wavelet,scale_to_frequency
import numpy as np
from AnalogFilters import AnalogFilterDesign


if __name__ == '__main__':
    w0 = 6
    fs = 1000
    sig = simulate_ecg_with_VLP_ALP(duration = 4, #sec
                                  fs = fs, #hz
                                  noise_level =40, #db
                                  hr = 80,#bpm
                                  Std = 0, #bpm
                                  unregular_comp = False,
                                    random_state = 11,
                                  lap_amp = 25,
                                  lvp_amp = 25)



    # sig = sig[120:600]
    # sig = sig[240:1200]
    # sig = sig[280:700]
    sig = sig[560:1400]







    # sig = sigTotest(fs=fs)
    # print(len(sig))
    # sig = sig[200:480]
    t_vector = np.array(list(range(len(sig))))/fs


    # scales = np.linspace(2, 250, 100)

    scales = np.linspace(3, 40, 40)

    # def overlap_50_scales(from_scale, num_scales):
    #     result_scales = []
    #     n_mass = np.array(list(range(num_scales)))
    #     for n in n_mass:
    #         result_scales.append(from_scale * ((2 ** 0.5) ** n))
    #     return np.array(result_scales)
    #
    #
    # scales = overlap_50_scales(3, 12)


    cwt_coefficients = cwt(sig, scales, morlet_wavelet, dt=1, fs=fs,w0=6, plot_wavelets_spectrum=False)
    reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet, dt=1, ds=1)

    plt.figure()
    X, Y = np.meshgrid(t_vector, scales)
    plt.pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    plt.show(block=False)

    freqs = scale_to_frequency(scales, w0, fs)
    plt.figure()
    X, Y = np.meshgrid(t_vector, freqs)
    plt.pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
    # plt.ylim([0, 50])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('frequency')
    plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    plt.show(block=False)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_vector, sig, label='Original Signal')
    plt.legend()
    plt.title('Original Signal')

    plt.subplot(2, 1, 2)
    plt.plot(t_vector, reconstructed_signal, label='Reconstructed Signal', color='orange')
    plt.legend()
    plt.title('Reconstructed Signal')

    plt.tight_layout()
    plt.show()

