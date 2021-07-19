import numpy as np
import matplotlib.pyplot as plt

def collect_data(n_samples=int(1e7), T = 300.0, delta_t=100e-9):
    mean = 2.**11
    sigma_in = (T  * 200e-9 / delta_t ) ** 0.5 + 15
    sigma_out = (T**(1/2)  * 200e-9 / delta_t) ** 0.5 + 15
    data_in = np.round(np.random.normal(mean, sigma_in, n_samples))
    data_out = np.round(np.random.normal(mean, sigma_out, n_samples))
    data_in_mean = data_in.mean()
    data_out_mean = data_out.mean()
    data_in -= data_in_mean
    data_out -= data_out_mean
    t = delta_t * np.arange(data_in.size)

    data_in_ft = np.fft.fft(data_in)
    data_out_ft = np.fft.fft(data_out)
    freqs_ft = np.fft.fftfreq(data_in.size,delta_t)
    
    bandpass_center = np.max(freqs_ft)/2
    bandpass_width = np.max(freqs_ft) * (3/4)
    window_sharpness = 10
    window_function = (np.exp( -((2 * (np.abs(freqs_ft) - bandpass_center)/bandpass_width) ** ( 2 * window_sharpness) ) ))  
    sharp_window_function_scale = 1
    sharp_window_function = sharp_window_function_scale * np.ones(data_in_ft.shape)
    window_function = sharp_window_function
    for i in range(sharp_window_function.size):
        if np.abs(freqs_ft[i])<bandpass_center - bandpass_width/2 or np.abs(freqs_ft[i])>bandpass_center + bandpass_width/2:
            sharp_window_function[i] = 0

    data_in_ft_windowed = window_function * data_in_ft
    data_out_ft_windowed = (1 - window_function) * data_out_ft


    data_in_ft_ft = np.real(np.fft.ifft(data_in_ft_windowed)) + data_in_mean
    data_out_ft_ft = np.real(np.fft.ifft(data_out_ft_windowed)) + data_out_mean
    

    
    data = np.round(data_in_ft_ft + data_out_ft_ft)
    
    return data