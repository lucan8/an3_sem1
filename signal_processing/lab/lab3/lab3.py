import os
import numpy as np
import matplotlib.pyplot as plt
plot_dir = os.path.join(os.path.dirname(__file__), "graphs")
def get_Fourier_matrix(n: int):
    res = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            res[i][j] = np.exp(-2 * np.pi * 1j * i * (j / n))

    return res

def get_Fourier_trans(signal: np.ndarray, omega: int):
    res = np.abs(sum(signal * np.exp(-2 * np.pi * 1j * np.array(range(signal.size)) * (omega / signal.size))))

    return res

def my_sin1(t, freq=100, ampl=1, fase=0):
    return ampl * np.sin(2 * np.pi * freq * t + fase)


def ex1():
    n = 8
    Fourier_matrix = get_Fourier_matrix(n)

    fig, axs = plt.subplots(n)
    for i in range(n):
        sin_row = []
        cos_row = []
        for j in range(n):
            cos_row.append(Fourier_matrix[i][j].real)
            sin_row.append(Fourier_matrix[i][j].imag)

        axs[i].plot(cos_row)
        axs[i].plot(sin_row)

        axs[i].set_title(f'Fourier matrix row {i}')
   
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ex1.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Does not match
    hermitian = np.transpose(np.conjugate(Fourier_matrix))

    hopefully_id = Fourier_matrix * hermitian
    
    print(np.allclose(hopefully_id, np.identity(n)))

def ex2():
    fig, axs = plt.subplots(2)
    low = 0
    high = 1
    sample_count = 1000
    sin_freq = 5
    real_axis = np.linspace(low, high, sample_count)
    my_signal = np.array([my_sin1(t, sin_freq) for t in real_axis])

    omegas = [1, 2, 5, 7]
    # make the figure twice as wide as Matplotlib's default (default width ~6.4)
    fig, axs = plt.subplots(len(omegas) + 1, figsize=(12.8, 4.8))

    for i, omega in enumerate(omegas):
        my_signal_circle = np.array([
            my_signal[t_idx] * np.exp(-2 * np.pi * 1j * omega * real_axis[t_idx])
            for t_idx in range(len(real_axis))
        ])

        real = np.array([elem.real for elem in my_signal_circle])
        imag = np.array([elem.imag for elem in my_signal_circle])

        # color each point by distance from origin
        distances = np.sqrt(real ** 2 + imag ** 2)
        sc = axs[i].scatter(real, imag, c=distances, cmap='plasma', s=6, edgecolors='none')
        axs[i].set_title(f'Complex-plane samples (omega={omega})')
        axs[i].set_aspect('equal', adjustable='box')

    axs[-1].plot(real_axis, my_signal)
    axs[-1].set_title('Initial signal')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ex2.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def ex3():
    fig, axs = plt.subplots(2)
    low = 0
    high = 1
    sample_count_cont = 1024
    real_axis = np.linspace(low, high, sample_count_cont)

    # Create composed signal
    sin_freq = [10, 35, 87, 185]
    sin_ampl = [1, 0.5, 2, 3]
    fase = [0.5, 2, 5, 7]
    my_signals = np.array([np.array([my_sin1(t, sin_freq[i], sin_ampl[i], fase[i]) for t in real_axis]) for i in range(len(sin_freq))])
    sum_signal = my_signals.sum(axis=0)

    # Get the abs of fourier trans elements for different omegas 
    N = 1024
    sum_signal_samples = np.array([sum_signal[i] for i in range(0, sum_signal.size, sum_signal.size // N)])
    fourier_trans = np.array(get_Fourier_matrix(N))
    transformed =  np.abs(np.matmul(fourier_trans, sum_signal_samples))

    print(sum_signal_samples.shape, fourier_trans.shape, transformed.shape)
    axs[0].plot(real_axis, sum_signal)
    axs[0].set_title('Composed signal')

    print(sum_signal_samples.shape, transformed.shape)
    axs[1].stem([i * (sample_count_cont / N) for i in range(sum_signal_samples.size)], transformed)
    axs[1].set_title('Magnitude of Fourier transform')
    axs[1].set_xlabel('freq')
    axs[1].set_ylabel('Fourier[freq]')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ex3.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig)


ex1()
ex2()
ex3()