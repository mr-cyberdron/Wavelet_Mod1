import numpy as np
import matplotlib.pyplot as plt

def doublesin(sin1_hz,sin2_hz,fs,dur, plotflag = False):
    t = np.arange(0, dur, 1/fs)  # Time vector

    signal1 = np.sin(2 * np.pi * sin1_hz * t)
    signal2 = np.sin(2 * np.pi * sin2_hz * t)
    # Combine the two sinusoids
    combined_signal = signal1 + signal2
    if plotflag:
        # Plotting the signal
        plt.figure(figsize=(10, 6))
        plt.plot(t, combined_signal)
        plt.title('Combined Sinusoidal Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    return combined_signal, t

