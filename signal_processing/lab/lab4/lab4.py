import numpy as np
import time
import matplotlib.pyplot as plt
import os
import scipy

curr_dir = os.path.dirname(__file__)
plot_dir = os.path.join(curr_dir, "graphs")

def get_Fourier_trans(signal: np.ndarray):
   return np.array([sum(signal * np.exp(-2 * np.pi * 1j * np.array(range(signal.size)) * (omega / signal.size))) for omega in range(signal.size)])

def get_fast_Fourier_trans(signal: np.ndarray):
    if signal.size <= 1:
        return signal
    
    even = get_fast_Fourier_trans(signal[::2])
    odd = get_fast_Fourier_trans(signal[1::2])

    factor = np.exp(-2 * np.pi * 1j * np.array(range(signal.size)) / signal.size)

    return np.concatenate([
        even + factor[:signal.size//2] * odd,
        even - factor[:signal.size//2] * odd 
    ])

def my_sin(t, freq=100, ampl=1, fase=0):
    return ampl * np.sin(2 * np.pi * freq * t + fase)

def ex1():
    N = [128, 256, 512, 1024, 2048, 4096, 8192]
    V = [np.random.randint(0, 100000, n) for n in N]

    my_dft_durations = []
    my_fft_durations = []
    numpy_fft_durations = []
    for i, v in enumerate(V):
        start = time.perf_counter()
        print(f"TESTING FOR N={N[i]}")
        my_dft = get_Fourier_trans(v)
        my_dft_durations.append(time.perf_counter() - start)
        print(f"    MY DFT TIME: {my_dft_durations[-1]}")
        
        start = time.perf_counter()
        my_fft = get_fast_Fourier_trans(v)
        my_fft_durations.append(time.perf_counter() - start)
        print(f"    MY FFT TIME: {my_fft_durations[-1]}")
        
        start = time.perf_counter()
        numpy_fft = np.fft.fft(v)
        numpy_fft_durations.append(time.perf_counter() - start)
        print(f"    NUMPY FFT TIME: {numpy_fft_durations[-1]}")

        print("----------------------")

        assert np.allclose(my_dft, my_fft)
        assert np.allclose(my_dft, numpy_fft)
        assert np.allclose(my_fft, numpy_fft)


    plt.plot(N, my_dft_durations, marker='o', label='My DFT (naive O(N^2))')
    plt.plot(N, my_fft_durations, marker='o', label='My FFT (Cooley-Tukey)')
    plt.plot(N, numpy_fft_durations, marker='o', label='NumPy FFT (optimized)')
    plt.title('DFT vs FFT vs numpy execution time')
    plt.xlabel('Signal length N')
    plt.ylabel('Elapsed time (seconds)')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show()
    plt.savefig(f"{plot_dir}/ex1.pdf")


# EX2 AND EX3
def ex2_3(ex2: bool):
    signal_count = 4
    fig, axs = plt.subplots(signal_count)
    low = 0
    high = 1
    freq = 5
    sample_count_cont = 1000
    time_axis = np.linspace(low, high, sample_count_cont)

    # < Nyquist freq
    sampling_freq_bad = 3 * freq // 2
    time_samples_bad = np.linspace(low, high, sampling_freq_bad + 1)

    # > Nyquist freq
    sampling_freq_good = freq * 2 + 1
    time_samples_good = np.linspace(low, high, sampling_freq_good + 1)

    # Choose samples based on exercise
    if ex2:
        chosen_time_samples = time_samples_bad
        file_name = "ex2.pdf"
    else:
        chosen_time_samples = time_samples_good
        file_name = "ex3.pdf"

    signals = [[my_sin(t, freq + k * sampling_freq_bad) for t in time_axis] for k in range(signal_count + 1)]
    signals_sampled = [[my_sin(t, freq + k * sampling_freq_bad) for t in chosen_time_samples] for k in range(signal_count + 1)]

    for k in range(0, signal_count):
        axs[k].plot(time_axis, signals[k])
        axs[k].stem(chosen_time_samples, signals_sampled[k])

    fig.show()
    fig.savefig(f"{plot_dir}/{file_name}")

# ex2_3(True)
# ex2_3(False)

# EX4: 2 * 200HZ

# EX5, EX6
plt.figure()
sample_rate, aeiou_signal = scipy.io.wavfile.read(f"{curr_dir}/aeiou.wav")
group_size = int(0.1 * aeiou_signal.size)
overlap_percentage = 0.5
aeiou_signal_grouped = np.array([aeiou_signal[i:i + group_size] for i in range(0, aeiou_signal.size - group_size + 1, int(group_size * overlap_percentage))])
ffts = np.array([np.abs(np.fft.fft(signal)[:group_size // 2]) for signal in aeiou_signal_grouped])
plt.imshow(20 * np.log10(ffts.T + 1e-6), aspect='auto', origin='lower')
plt.xlabel("Window index (time)")
plt.ylabel("Frequency bin")
plt.colorbar(label="Amplitude (dB)")
plt.show()

# EX7
SNR_db = 80
P_s = 90
SNR = (10 ** (SNR_db / 10))
P_n = P_s / SNR
print(P_n)
