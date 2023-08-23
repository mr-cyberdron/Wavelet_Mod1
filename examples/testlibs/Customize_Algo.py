import numpy as np
import matplotlib.pyplot as plt

from Artef_sig import sigTotest
from wavelets_func import morlet_wavelet

def plot(sig, block = True):
    plt.figure()
    plt.plot(sig)
    plt.show(block = block)

def waveletBody(wavelet, amp = 1.0):
    Wavelet_start_count = 0
    Wavelet_end_count = len(wavelet) - 1
    wavelet_real = np.real(wavelet)
    wavelet_max = max(wavelet_real)
    correction_value = amp/wavelet_max
    treshold = wavelet_max*0.001
    #find_startpos
    for count in range(len(wavelet)):
        if wavelet_real[count] > treshold:
            Wavelet_start_count = count
            break
    #find_endpos
    for count in range(len(wavelet)-1,0,-1):
        if wavelet_real[count] > treshold:
            Wavelet_end_count = count+1
            break
    return np.array(wavelet)[Wavelet_start_count:Wavelet_end_count]*correction_value

def Similarity_func(sig1, sig2):
    if len(sig1)>1 and len(sig2)>1:
        return np.corrcoef(sig1,sig2)[0][1]
    else:
        return 0

def Similarity_func2(sig1, sig2):
    if len(sig1)>1 and len(sig2)>1:
        from scipy.spatial.distance import cosine
        return 1 - cosine(sig1, sig2)
    else:
        return 0



def Similarity_func3(sig1, sig2):
    if len(sig1)>1 and len(sig2)>1:
        return np.linalg.norm(np.array(sig1) - np.array(sig2))
    else:
        return 0



def custom_conv_with_metric(sig, wavelet_body):
    print('custom conv')
    sig = np.array(sig)
    wavelet_body = np.array(wavelet_body)
    # Create an empty output mass
    output_mass = [0] * len(sig)
    wavelet_len = len(wavelet_body)
    half_wavelet_offset = ((wavelet_len - 1) / 2).__floor__()
    for sig_point_num in range(len(sig)):
        start_sig_point = max(sig_point_num-half_wavelet_offset,0)
        stop_sig_point = min(sig_point_num+half_wavelet_offset+1,len(sig)-1)
        start_wavelet_point = max(half_wavelet_offset-sig_point_num, 0)
        stop_wavelet_point =  min(half_wavelet_offset + (len(sig)-1 - sig_point_num),wavelet_len-1)

        signal_part_mass = sig[start_sig_point:stop_sig_point]
        wavelet_part_mass = wavelet_body[start_wavelet_point:stop_wavelet_point]
        if len(signal_part_mass) == len(wavelet_part_mass):
            pass
        else:
            print(len(signal_part_mass))
            print(len(wavelet_part_mass))
            print(start_sig_point,stop_sig_point,start_wavelet_point,stop_wavelet_point)
            print(wavelet_len)
            print(half_wavelet_offset)
            raise ValueError ("Signal part and wavelet parts should be equal")

        similarity_coef = Similarity_func3(signal_part_mass,wavelet_part_mass)

        output_mass[int(sig_point_num)] = similarity_coef


    return output_mass








def cwt_customized(signal, scales):
    n = len(signal)
    custom_cwt_matrix = np.zeros((len(scales), n), dtype='complex_')
    for ind, scale in enumerate(scales):
        wavelet_data = np.conj(morlet_wavelet(n, scale))
        wavelet_body = waveletBody(wavelet_data, amp=1)
        custom_cwt_matrix[ind, :] = custom_conv_with_metric(signal,np.real(wavelet_body))
    return custom_cwt_matrix




fs = 500
sig = sigTotest(fs=fs)
t = np.arange(0, len(sig)/fs, 1/fs)
num_scales = 80
scales = np.arange(1, num_scales)

cwtmatr = cwt_customized(sig, scales)

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