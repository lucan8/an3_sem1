import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice

graph_dir = "signal_processing/lab/lab2/graphs/"
lower_b = 0
upper_b = 20
sample_count = 44100
real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]
# 1
def my_sin1(t, freq=100):
    ampl = 1
    fase = 0
    return ampl * np.sin(2 * np.pi * freq * t + fase)

def my_cos1(t):
    ampl = 1
    freq = 1
    fase = 0
    return ampl * np.cos(2 * np.pi * freq * t + fase - np.pi / 2)

def ex1():
    fig, axs = plt.subplots(2)

    sin_plots = [my_sin1(elem) for elem in real_axis_cont]
    cos_plots = [my_cos1(elem) for elem in real_axis_cont]
    axs[0].plot(real_axis_cont, sin_plots)
    axs[1].plot(real_axis_cont, cos_plots)
    axs[0].set_title('Exercise 1: Sine and Cosine')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('amplitude')
    axs[0].legend(['sin(2πft)'])
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('amplitude')
    axs[1].legend(['cos(2πft - π/2)'])
    # plt.show()
    plt.savefig(f"{graph_dir}/ex1.pdf", format="pdf")

def ex2():
    def my_sin2(t, fase):
        ampl = 1
        freq = 1
        return ampl * np.sin(2 * np.pi * freq * t + fase)

    def get_umsilon(x, z, SNR):
        return np.sqrt(np.linalg.norm(x) ** 2 / (np.linalg.norm(z) ** 2 * SNR))
    
    sin_plots1 = [my_sin2(elem, 0) for elem in real_axis_cont]
    sin_plots2 = [my_sin2(elem, 1) for elem in real_axis_cont]
    sin_plots3 = [my_sin2(elem, 10) for elem in real_axis_cont]
    sin_plots4 = [my_sin2(elem, 100) for elem in real_axis_cont]

    z = np.random.normal(0, 1, sample_count)
    SNRs = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    umsilons = [get_umsilon(sin_plots1, z, snr) for snr in SNRs]

    fig, axs1 = plt.subplots(4)

    axs1[0].plot(real_axis_cont, sin_plots1)
    axs1[0].set_title('Exercise 2: Sine with different phases')
    axs1[0].legend(['phase = 0'])
    axs1[1].plot(real_axis_cont, sin_plots2)
    axs1[1].legend(['phase = 1'])
    axs1[2].plot(real_axis_cont, sin_plots3)
    axs1[2].legend(['phase = 10'])
    axs1[3].plot(real_axis_cont, sin_plots4)
    axs1[3].legend(['phase = 100'])
    for ax in axs1:
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude')
    fig.savefig(f"{graph_dir}/ex2.pdf", format="pdf")

    fig, axs2 = plt.subplots(len(umsilons))
    for i, ums in enumerate(umsilons):
        axs2[i].plot(real_axis_cont, np.array(sin_plots1) + ums * z)
        axs2[i].set_xlabel('time (s)')
        axs2[i].set_ylabel('amplitude')
        axs2[i].legend([f'SNR={SNRs[i]}'])

    # plt.show()
    fig.savefig(f"{graph_dir}/ex2_noise.pdf", format="pdf")

# Ex 3
def ex3():
    lab_sound_dir = 'signal_processing/lab/lab1/sounds/ex2/'
    rate, x = scipy.io.wavfile.read(lab_sound_dir + 'b.wav')

    sounddevice.play(x, sample_count)
    sounddevice.wait()

# Ex 4
def ex4():
    def signal_sawtooth(t):
        freq = 5
        return np.mod(t, freq)

    fig, axs = plt.subplots(3)
    sin_plots = np.array([my_sin1(elem) for elem in real_axis_cont])
    sawtooth_plots = np.array([signal_sawtooth(elem) for elem in real_axis_cont])

    axs[0].plot(sin_plots)
    axs[1].plot(sawtooth_plots)
    axs[2].plot(sin_plots + sawtooth_plots)
    axs[0].set_title('Exercise 4: Sine, Sawtooth and Sum')
    axs[0].legend(['sine'])
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('amplitude')
    axs[1].legend(['sawtooth'])
    axs[2].legend(['sine + sawtooth'])
    fig.savefig(f"{graph_dir}/ex4.pdf", format="pdf")

# Ex 5
def ex5():
    freq1 = 400
    freq2 = 800

    sin1_signal = [my_sin1(t, freq1) for t in real_axis_cont]
    sin2_signal = [my_sin1(t, freq2) for t in real_axis_cont]
    concat_sin_signals = sin1_signal + sin2_signal

    fig, axs = plt.subplots(3)
    axs[0].plot(real_axis_cont, sin1_signal)
    axs[0].set_title('Exercise 5: Two sine signals and their concatenation')
    axs[0].legend([f'{freq1} Hz'])
    axs[1].plot(real_axis_cont, sin2_signal)
    axs[1].legend([f'{freq2} Hz'])
    axs[2].plot(concat_sin_signals)
    axs[2].legend(['concatenated signals'])
    for ax in axs:
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude')
    fig.savefig(f"{graph_dir}/ex5.pdf", format="pdf")

    sounddevice.play(concat_sin_signals, sample_count)
    sounddevice.wait()
# Ex 6
def ex6():
    sampling_freq = 100
    freq = [sampling_freq, sampling_freq / 2, sampling_freq / 4, 0]

    upper_b = 1
    sample_count = sampling_freq
    real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sample_count)]
    fig, axs = plt.subplots(len(freq))
    for i, f in enumerate(freq):
        axs[i].stem([my_sin1(t, f) for t in real_axis_cont])
        axs[i].legend([f'freq: {f}'])
        axs[i].set_xlabel('sample index')
        axs[i].set_ylabel('amplitude')

    # plt.show()
    fig.savefig(f"{graph_dir}/ex6.pdf", format="pdf")

# Ex 7 
def ex7():
    fig, axs = plt.subplots(3)
    sampling_freq = 1000
    real_axis_cont = [x for x in np.linspace(lower_b, upper_b, sampling_freq)]
    signal = [my_sin1(t, sampling_freq) for t in real_axis_cont]

    quarter_signal = [s if i % 4 == 0 else 0 for i, s in enumerate(signal)]
    quarter_signal_from_2 = [s if (i + 3) % 4 == 0 else 0 for i, s in enumerate(signal)]

    axs[0].stem(signal)
    axs[0].set_title('Exercise 7: Subsampling examples')
    axs[0].legend(['original'])
    axs[1].stem(quarter_signal)
    axs[1].legend(['every 4th sample'])
    axs[2].stem(quarter_signal_from_2)
    axs[2].legend(['shifted every 4th sample'])
    for ax in axs:
        ax.set_xlabel('sample index')
        ax.set_ylabel('amplitude')
    # plt.show()
    fig.savefig(f"{graph_dir}/ex7.pdf", format="pdf")

# Ex 8
def ex8():
    def Pade_sin_aprox(t):
        return (t - (7 * t ** 3 / 60)) / (1 + t ** 2 / 20)


    def Taylor_sin_aprox(t, n = 10):
        factorial = {-1: 1, 1: 1}
        res = t
        for i in range(1, n):
            # print(res)
            factorial[2 * i + 1] = factorial[2 * i - 1] * (2 * i) * (2 * i + 1)
            if i % 2 == 0:
                res +=  t ** (2 * i + 1) / factorial[2 * i + 1]
            else:
                res -= t ** (2 * i + 1) / factorial[2 * i + 1]
            # print(t ** (2 * i + 1) / factorial[2 * i + 1])
        return res

    lower_b = -np.pi / 2
    upper_b = np.pi / 2
    freq = 10000
    sample_count = int((upper_b - lower_b) * freq)
    real_axis_cont = np.linspace(lower_b, upper_b, sample_count)

    fig, axs = plt.subplots(3)
    sin_signal = np.sin(real_axis_cont)
    pade_sin_aprox_signal = np.array([Pade_sin_aprox(t) for t in real_axis_cont])
    taylor_sin_aprox_signal = np.array([Taylor_sin_aprox(t) for t in real_axis_cont])

    axs[0].plot(real_axis_cont, sin_signal)
    axs[0].set_title('Exercise 8: sin and approximations')
    axs[0].legend(['sin(x)'])
    axs[1].plot(real_axis_cont, real_axis_cont)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('value')
    axs[2].semilogy(abs(sin_signal - real_axis_cont))
    axs[2].set_ylabel('error (log scale)')
    # plt.show()
    fig.savefig(f"{graph_dir}/ex8_def.pdf", format="pdf")

    fig, axs = plt.subplots(3)
    axs[0].plot(real_axis_cont, sin_signal)
    axs[0].legend(['sin(x)'])
    axs[1].plot(real_axis_cont, pade_sin_aprox_signal)
    axs[1].legend(["Pade approx"]) 
    axs[2].semilogy(abs(sin_signal - pade_sin_aprox_signal))
    axs[2].set_ylabel('error (log scale)')
    # plt.show()
    fig.savefig(f"{graph_dir}/ex8_pade.pdf", format="pdf")

    fig, axs = plt.subplots(3)
    axs[0].plot(real_axis_cont, sin_signal)
    axs[0].legend(['sin(x)'])
    axs[1].plot(real_axis_cont, taylor_sin_aprox_signal)
    axs[1].legend(["Taylor approx"]) 
    axs[2].semilogy(abs(sin_signal - taylor_sin_aprox_signal))
    axs[2].set_ylabel('error (log scale)')
    # plt.show()
    fig.savefig(f"{graph_dir}/ex8_taylor.pdf", format="pdf")

ex1()
ex2()
ex3()
ex4()
ex5()
ex6()
ex7()
ex8()