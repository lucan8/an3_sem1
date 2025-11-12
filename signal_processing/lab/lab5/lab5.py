import numpy as np
import os
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(__file__)
plot_dir = os.path.join(curr_dir, "graphs")
# a): (once an hour) : 1 / 3600 hz

# b): (0 - 23 hours) : 0 - 23 * 3600 ?

# c) : 27433.5
signal = np.array([x[2] for x in np.genfromtxt(f'{curr_dir}/Train.csv', delimiter=',')])

# d)
sampling_freq = 1 / 3600
signal_fft = np.fft.fft(signal) / signal.size
signal_fft_abs = np.abs(np.fft.fft(signal) / signal.size)
signal_fft_half = signal_fft_abs[:signal.size // 2]
freq_axis = sampling_freq*np.linspace(0, signal.size // 2, signal.size // 2)

def d():
    fig, axs = plt.subplots(2)
    axs[0].plot(signal)
    axs[1].plot(freq_axis, signal_fft_half)
    fig.savefig(f"{plot_dir}/d.pdf")

# e)
def e():
    fig, axs = plt.subplots(2)
    axs[0].plot(signal)
    axs[1].plot(signal - np.mean(signal))
    fig.savefig(f"{plot_dir}/e.pdf")

# f)
def f():
    print(signal_fft_half)
    
    signal_fft_sorted_indexes = np.argsort(signal_fft_half)
    needed_indexes = [signal_fft_sorted_indexes[-1], signal_fft_sorted_indexes[-2], signal_fft_sorted_indexes[-3], signal_fft_sorted_indexes[-4]]
    needed_freq = freq_axis[needed_indexes]
    
    print(needed_freq)

    return np.array(needed_indexes)

# g)
def g():
    fig, axs = plt.subplots(1)
    first_monday_index = 48
    sample_skip_count = 1000
    sampling_start = first_monday_index + sample_skip_count
    sample_count = 24 * 30

    signal_samples = signal[sampling_start:sampling_start + sample_count]
    axs.plot(signal_samples)
    fig.savefig(f"{plot_dir}/g.pdf")

# h)
# From the singal's spikes we could say that a special event occured like 
# Christmas (or any other fixed date event that creates an expected deviation from the norm)
# Let's say the chosen spike is sample k
# Let's say the sampling freq is once every hour
# Now the start date would be k.date - k.nr / 24

# i)
def i():
    high_freq_indexes = f()

    signal_fft[high_freq_indexes] = 0
    signal_fft[signal.size - high_freq_indexes - 1] = 0


    signal_filtered = np.fft.ifft(signal_fft)

    fig, axs = plt.subplots(2)
    axs[0].plot(signal)
    axs[1].plot(signal_filtered)
    fig.savefig(f"{plot_dir}/i.pdf")

d()
e()
f()
g()
i()