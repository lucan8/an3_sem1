import numpy as np
import matplotlib.pyplot as plt
import os

curr_dir = os.path.dirname(__file__)
lab_dir = os.path.dirname(curr_dir)
plot_dir = os.path.join(curr_dir, "graphs")

def trend(x):
    return 5 * x ** 2 + 3 * x - 2

def my_sin(t, freq=100, ampl=1, fase=0):
    return ampl * np.sin(2 * np.pi * freq * t + fase)

def seasonality(time_axis):
    freq1 = np.random.randint(2, 20)
    freq2 = np.random.randint(2, 20)

    w1 = np.random.randint(1, 10)
    w2 = np.random.randint(1, 10)
    return w1 * my_sin(time_axis, freq1) + w2 * my_sin(time_axis, freq2)

# Signal creation
N = 1000
trend_x = np.random.random(N)
trend_of_time_series = trend(trend_x)
seasonality_of_time_series = seasonality(np.linspace(0, N, N))
noise_of_time_series = np.random.normal(0, 1, N)
time_series = trend_of_time_series + seasonality_of_time_series + noise_of_time_series

def ex1():
    fig, axs = plt.subplots(4)
    axs[0].plot(trend_x, trend_of_time_series, label="Trend")
    axs[1].plot(seasonality_of_time_series, label="Seasonality")
    axs[2].plot(noise_of_time_series, label="Noise")
    axs[3].plot(time_series, label="Whole time series")

    plt.legend()
    fig.savefig(f"{plot_dir}/ex1.pdf")
    plt.close()

def ex2():
    def auto_correlation_true(x: np.ndarray):
        corr_vec = []
        for t in range(len(x)):
            old_x = x[:len(x) - t]
            corr = np.dot(old_x, x[t:])

            corr_vec.append(corr)
        return np.array(corr_vec)

    plt.plot(auto_correlation_true(time_series), label="Auto correlation")
    plt.legend()
    plt.savefig(f"{plot_dir}/ex2.pdf")
    plt.close()

def make_pairs(x, y):
    res = []

    for v1 in x:
        for v2 in y:
            res.append((v1, v2))
    
    return res

def AR_model(x: np.ndarray, p:int, m:int):
    Y = np.array([x[x.shape[0] - m - i:x.shape[0] - i] for i in range(1, p + 1)]).transpose()
    x_samples =  x[x.shape[0] - m:]
    Y_mult_with_trans = np.matmul(Y.transpose(), Y)
    Y_mult_with_trans += np.random.random(Y_mult_with_trans.shape)
    params = np.matmul(np.matmul(np.linalg.inv(Y_mult_with_trans), Y.transpose()), x_samples.transpose())

    pred = np.zeros(x.shape)
    err = 0
    for i in range(p - 1, pred.shape[0] - 1):
        pred[i + 1] = np.matmul(params.transpose(), x[i - p + 1 : i + 1])
        err += np.linalg.norm(x[i + 1] - pred[i + 1]) ** 2
    
    return err, pred, params

# Ex 3 and 4 are kinda the same
def ex3_4():
    grid_size = 50
    p = np.arange(2, grid_size)
    m = np.arange(2, grid_size)
    p_m = make_pairs(p, m)
    model_result = [(i, AR_model(time_series, p_s, m_s)) for i, (p_s, m_s) in enumerate(p_m)]
    ind_min_err, (min_err, best_pred, chosen_params) = min(model_result, key=lambda x: x[1][0])

    best_p_m = p_m[ind_min_err]
    print(f"Best params: p:{best_p_m[0]}, m:{best_p_m[1]}, err:{min_err}")
    plt.plot(time_series, label="Initial")
    plt.plot(best_pred, label=f"AR grid_size={grid_size}, p={best_p_m[0]}, m={best_p_m[1]}, err={min_err:.2f}")
    plt.legend()
    plt.savefig(f"{plot_dir}/ex3_4.pdf")

ex1()
ex2()
ex3_4()