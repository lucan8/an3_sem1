import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

curr_dir = os.path.dirname(__file__)
lab_dir = os.path.dirname(curr_dir)
plot_dir = os.path.join(curr_dir, "graphs")

def ex1():
    def from_samples(time_axis_cont, time_axis_sampled, sampled_sinc, Ts):
        return np.array([sum(sampled_sinc * np.sinc((t - time_axis_sampled) / Ts)) for t in time_axis_cont])
        
    sampling_freq = [1, 1.5, 2, 4]
    Ts = [1 / freq for freq in sampling_freq]
    
    fig, axs = plt.subplots(len(sampling_freq))
    low = -3
    high = 3
    sample_count_cont = 10000
    B = 1

    # Continuous signal
    time_axis_cont = np.linspace(low, high, sample_count_cont)
    cont_sinc = np.sinc(time_axis_cont * B) ** 2

    # Sampled sinc at different freq, make sure to include 0
    time_axis_sampled = [np.concatenate([
                                        np.linspace(low, 0, int(abs(low) * freq), False),
                                        [0],
                                        np.linspace(0, high, int(high * freq + 1))[1:]])
                        for freq in sampling_freq
                        ]

    sampled_sinc = [np.sinc(time_axis * B) ** 2 for time_axis in time_axis_sampled]

    for i in range(len(sampling_freq)):
        axs[i].plot(time_axis_cont, cont_sinc)
        axs[i].stem(time_axis_sampled[i], sampled_sinc[i])
        axs[i].plot(time_axis_cont, from_samples(time_axis_cont, time_axis_sampled[i],sampled_sinc[i], Ts[i]))
    
    fig.savefig(f"{plot_dir}/ex1.pdf")
    # Observation: As B grows we go towards dirac's function


def ex2():
    low = 40
    high = 60
    vec_size = 100
    random_vec = np.random.random(vec_size)
    rect_signal = [1 if i >= low and i <= high else 0 for i in range(vec_size)]

    signals = [random_vec, rect_signal]
    signals_types = ["random", "rectangular"]
    iter_count = 3

    for j, signal in enumerate(signals):
        fig, axs = plt.subplots(iter_count + 1)
        axs[0].plot(signal)

        for i in range(1, iter_count + 1):
            signal = np.convolve(signal, signal)
            axs[i].plot(signal)

        fig.savefig(f"{plot_dir}/ex2/{signals_types[j]}.pdf")

def ex3():
    polynomial_grade = 3
    pol1 = np.random.random(polynomial_grade)
    pol2 = np.random.random(polynomial_grade)

    #Normal convolution
    res1 = np.convolve(pol1, pol2)

    # FFT convolution
    pol1 = np.concatenate([pol1, np.zeros(polynomial_grade - 1)])
    pol2 = np.concatenate([pol2, np.zeros(polynomial_grade - 1)])
    res2 = np.fft.ifft(np.fft.fft(pol1) * np.fft.fft(pol2)).real

    assert(np.allclose(res1, res2))

def ex4():
    vec_size = 20
    d = np.random.randint(1, 19, 1)[0] # Must be smaller than vec_size to work
    vec = np.random.random(vec_size)

    # Shift
    last_d = vec[vec_size - d:]
    dep_vec = scipy.ndimage.shift(vec, d)
    dep_vec[:d] = last_d

    # Correlation
    correlations = np.abs(np.fft.ifft(np.fft.fft(vec).conjugate() * np.fft.fft(dep_vec)))
    result = correlations.argmax()
    assert(result == d)

    # Division
    division = np.abs(np.fft.ifft(np.fft.fft(dep_vec) / np.fft.fft(vec)))
    result = division.argmax()
    assert(result == d)

def ex5():
    def my_sin1(t, freq=100, ampl=1, fase=0):
        return ampl * np.sin(2 * np.pi * freq * t + fase)

    def create_rect_window(size):
        return np.ones(size)

    def create_hann_window(size):
        return 0.5 * (1 - np.cos(2 * np.arange(size) * np.pi // size))


    windows_func = [create_rect_window, create_hann_window]
    windows_names = ["Rectangular Window", "Hann Window"]

    low = 0
    high = 1
    sample_count = 1000
    window_size = 200
    time_axis_cont = np.linspace(low, high, sample_count)
    my_signal = np.array([my_sin1(t) for t in time_axis_cont])

    for i, w_func in enumerate(windows_func):
        fig, axs = plt.subplots(3)
        window = w_func(window_size)
        axs[0].plot(my_signal)
        axs[1].plot(window)
        axs[2].plot(np.convolve(my_signal, window))
        fig.savefig(f"{plot_dir}/ex5/{windows_names[i]}.pdf")

# Ex6

def ex6():
    # a)
    start = 1000
    hours_in_day = 24
    desired_days = 3
    signal = np.array([x[2] for x in np.genfromtxt(f'{lab_dir}/Train.csv', delimiter=',')])[start : start + desired_days * hours_in_day]

    # b)
    window_sizes = [5, 9, 13, 17]
    fig, axs = plt.subplots(1)
    axs.plot(signal, label="Initial signal")
    for i, w in enumerate(window_sizes):
        axs.plot(np.convolve(signal, np.ones(w), 'valid'), label=f"window_size={w}")
    
    axs.legend()
    fig.savefig(f"{plot_dir}/ex6/b.pdf")

    # c)
    sampling_freq = 1 / 3600

    # Cut point at 6 hours
    cut_point = 6
    cut_freq = sampling_freq / cut_point
    nyquist_freq = sampling_freq / 2
    norm_cut_freq = cut_freq / nyquist_freq

    # d)
    def plot_butter(axs, filter_order):
        butter_b, butter_a = scipy.signal.butter(filter_order, norm_cut_freq, btype="low")
        butter_w, butter_h = scipy.signal.freqz(butter_b, butter_a, fs=sampling_freq)

        axs.plot(butter_w, 20 * np.log10(abs(butter_h)), 
                label=f'Butterworth (Order {filter_order})', linewidth=2)

        axs.legend()
        return butter_b, butter_a, axs

    def plot_cheby(axs, filter_order, rp):
        cheby_b, cheby_a = scipy.signal.cheby1(filter_order, rp, norm_cut_freq, btype="low")
        cheby_w, cheby_h = scipy.signal.freqz(cheby_b, cheby_a, fs=sampling_freq)

        axs.plot(cheby_w, 20 * np.log10(abs(cheby_h)), 
                label=f'Chebyshev Type I (Order {filter_order}, rp={rp}dB)', 
                linewidth=2, linestyle='--')

       
         
        axs.legend()
        return cheby_b, cheby_a, axs

    # d)
    filter_order = 5
    rp = 5
    fig, axs = plt.subplots(1)
    butter_b, butter_a, axs = plot_butter(axs, filter_order)
    cheby_b, cheby_a, axs = plot_cheby(axs, filter_order, rp)
    fig.savefig(f"{plot_dir}/ex6/d.pdf")

    # e)
    fig, axs = plt.subplots(1)

    # Plot signal after filters
    axs.plot(signal, label=f"Initial signal")
    axs.plot(scipy.signal.filtfilt(butter_b, butter_a, signal), label=f'Butterworth (Order {filter_order})', linewidth=2)
    axs.plot(scipy.signal.filtfilt(cheby_b, cheby_a, signal), label=f'Chebyshev Type I (Order {filter_order}, rp={rp}dB)', 
                        linewidth=2, linestyle='--')
    axs.legend()
    fig.savefig(f"{plot_dir}/ex6/e.pdf")

    # f)
    filter_orders = [2, 5, 10, 20]
    rp = [1, 3, 5, 10]
    fig_filters, axs_filters = plt.subplots(1)
    fig_signals, axs_signals = plt.subplots(1)

    axs_signals.plot(signal, label=f"Initial signal")
    for i, order in enumerate(filter_orders):
        # Plot filters
        butter_b, butter_a, axs_filters = plot_butter(axs_filters, order)
        cheby_b, cheby_a, axs_filters = plot_cheby(axs_filters, order, rp[i])
        
        # Plot signal after filters
        axs_signals.plot(scipy.signal.filtfilt(butter_b, butter_a, signal), label=f'Butterworth (Order {order})', linewidth=2)
        axs_signals.plot(scipy.signal.filtfilt(cheby_b, cheby_a, signal), label=f'Chebyshev Type I (Order {order}, rp={rp[i]}dB)', 
                         linewidth=2, linestyle='--')

    axs_signals.legend()
    fig_filters.savefig(f"{plot_dir}/ex6/f_filters.pdf")
    fig_signals.savefig(f"{plot_dir}/ex6/f_signals.pdf")
        

ex1()
ex2()
ex3()
ex4()
ex5()
ex6()