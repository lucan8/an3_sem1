import numpy as np
import matplotlib.pyplot as plt
import scipy

sound_rate = 44100
def ex1():
    # Continous
    lower_b = 0
    upper_b = 0.03
    step = 0.00005
    sample_count = int((upper_b - lower_b) / step)
    real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]

    def signal1(t):
        return np.cos(np.pi * (520 * t + 1 / 3))

    def signal2(t):
        return np.cos(np.pi * (280 * t - 1 / 3))

    def signal3(t):
        return np.cos(np.pi * (120 * t + 1 / 3))

    fig, axs = plt.subplots(3)

    signal1_plots = [signal1(elem) for elem in real_axis_cont]
    signal2_plots = [signal2(elem) for elem in real_axis_cont]
    signal3_plots = [signal3(elem) for elem in real_axis_cont]

    axs[0].plot(real_axis_cont, signal1_plots)
    axs[1].plot(real_axis_cont, signal2_plots)
    axs[2].plot(real_axis_cont, signal3_plots)

    # Discrete
    sample_freq = 200
    sample_count = int(sample_freq * (upper_b - lower_b))
    real_axis_stem = [x for x in np.linspace(lower_b, upper_b, sample_count)]

    signal1_plots = [signal1(elem) for elem in real_axis_stem]
    signal2_plots = [signal2(elem) for elem in real_axis_stem]
    signal3_plots = [signal3(elem) for elem in real_axis_stem]

    axs[0].stem(real_axis_stem, signal1_plots)
    axs[1].stem(real_axis_stem, signal2_plots)
    axs[2].stem(real_axis_stem, signal3_plots)

    plt.savefig("signal_processing/lab/lab1/graphs/ex1.pdf", format="pdf")
    plt.close()

def ex2():
    def a():
        plt.figure()
        lower_b = 0
        upper_b = 20
        sample_count = 40000
        real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]
        def signal_2_1(t):
            freq = 20000
            return np.sin(np.pi * (2 * freq * t + 1 / 3))

        signal21_plots = np.array([signal_2_1(elem) for elem in real_axis_cont])
        plt.stem(real_axis_cont, signal21_plots)
        plt.savefig("signal_processing/lab/lab1/graphs/ex2/a.pdf", format="pdf")
        
        scipy.io.wavfile.write('signal_processing/lab/lab1/sounds/ex2/a.wav', sound_rate, signal21_plots)
        plt.close()

    def b():
        plt.figure()
        lower_b = 0
        upper_b = 20
        sample_count = sound_rate * (upper_b - lower_b)
        real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]
        def signal2_2(t):
            freq = 440
            return np.sin(np.pi * (2 * freq * t + 1 / 3))

        signal22_plots = np.array([signal2_2(elem) for elem in real_axis_cont])
        plt.plot(real_axis_cont, signal22_plots)
        plt.savefig("signal_processing/lab/lab1/graphs/ex2/b.pdf", format="pdf")

        scaled = np.int16(signal22_plots / np.max(np.abs(signal22_plots)) * 32767)
        scipy.io.wavfile.write('signal_processing/lab/lab1/sounds/ex2/b.wav', sound_rate, scaled)
        plt.close()

    def c():
        plt.figure()
        lower_b = 0
        upper_b = 20
        sample_count_per_sec = 240
        sample_count = (upper_b - lower_b) * sample_count_per_sec
        real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]
        
        def signal_sawtooth(t):
            freq = 5
            return np.mod(t, freq)

        signal_sawtooth_plot = np.array([signal_sawtooth(elem) for elem in real_axis_cont])
        plt.plot(real_axis_cont, signal_sawtooth_plot)
        plt.savefig("signal_processing/lab/lab1/graphs/ex2/c.pdf", format="pdf")

        scipy.io.wavfile.write('signal_processing/lab/lab1/sounds/ex2/c.wav', sound_rate, signal_sawtooth_plot)
        plt.close()

    def d():
        plt.figure()
        lower_b = 0
        upper_b = 20
        sample_count_per_sec = 300
        sample_count = (upper_b - lower_b) * sample_count_per_sec
        real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]
        def signal_square(t):
            return np.sign(np.sin(t))

        signal_square_plot = np.array([signal_square(elem) for elem in real_axis_cont])
        plt.plot(real_axis_cont, signal_square_plot)
        plt.savefig("signal_processing/lab/lab1/graphs/ex2/d.pdf", format="pdf")

        scipy.io.wavfile.write('signal_processing/lab/lab1/sounds/ex2/d.wav', sound_rate, signal_square_plot)
        plt.close()

    def e():
        plt.figure()
        random_2d_signal = np.random.rand(128, 128)
        plt.imshow(random_2d_signal)
        plt.savefig("signal_processing/lab/lab1/graphs/ex2/e.pdf", format="pdf")
        plt.close()

    def f():
        plt.figure()
        def my_2d_signal_func(x, y):
            return x % 5 + y % 7

        my_2d_signal = np.zeros((128, 128))
        for i in range(128):
            for j in range(128):
                my_2d_signal[i][j] = my_2d_signal_func(i, j)
        
        plt.imshow(my_2d_signal)
        plt.savefig("signal_processing/lab/lab1/graphs/ex2/f.pdf", format="pdf")
        plt.close()
    a()
    b()
    c()
    d()
    e()
    f()

ex1()
ex2()