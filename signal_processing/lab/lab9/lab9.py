import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model.ARIMA
import os

# TODO: Test the arima models
curr_dir = os.path.dirname(__file__)
lab_dir = os.path.dirname(curr_dir)
plot_dir = os.path.join(curr_dir, "graphs")

def trend(x):
    return 5 * x ** 2 + 3 * x - 2

def my_sin(t, freq=55.5, ampl=1, fase=0):
    return ampl * np.sin(2 * np.pi * freq * t + fase)

def seasonality(time_axis):
    freq1 = np.random.randint(2, 20)
    freq2 = np.random.randint(2, 20)

    w1 = np.random.randint(1, 10)
    w2 = np.random.randint(1, 10)
    return w1 * my_sin(time_axis, freq1) + w2 * my_sin(time_axis, freq2)

def make_pairs(x, y):
    res = []

    for v1 in x:
        for v2 in y:
            res.append((v1, v2))
    
    return res

def make_triplets(x, y, z):
    res = []

    for v1 in x:
        for v2 in y:
            for v3 in z:
                res.append((v1, v2, v3))
    
    return res

# Generate signal
N = 200

trend_x = np.random.random(N)
trend_of_time_series = trend(trend_x)
seasonality_of_time_series = seasonality(np.linspace(0, N, N))
noise_of_time_series = np.random.normal(0, 1, N)
time_series = trend_of_time_series + seasonality_of_time_series + noise_of_time_series

def exp_mediation(time_series: np.ndarray, alfa: float):
    s = np.array([0 for _ in range(len(time_series) - 1)])
    s[0] = time_series[0]
    error = 0
    for i in range(1, len(time_series) - 1):
        s[i] = alfa * time_series[i] + (1 - alfa) * s[i - 1]
        error += (s[i] - time_series[i + 1]) ** 2
    
    return error, s

def double_exp_mediation(time_series: np.ndarray, alfa: float, beta: float, m: int):
    pred = np.array([0 for _ in range(len(time_series) - 1)])
    s = np.array([0 for _ in range(len(time_series) - 1)])
    b = np.array([0 for _ in range(len(time_series) - 1)])
    error = 0
    
    # First m pred are the time series
    pred[:m + 1] = time_series[:m + 1]
    s[0] = time_series[0]
    b[0] = time_series[1] - time_series[0]

    for i in range(1, len(time_series) - m - 1):
        s[i] = alfa * time_series[i] + (1 - alfa) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
        pred[i + m] = s[i] + m * b[i]
        error += (pred[i + m] - time_series[i + m]) ** 2
    
    return error, pred

def triple_exp_mediation(time_series: np.ndarray, alfa: float, beta: float, m: int, gamma: float, L:int):
    pred = np.array([0 for _ in range(len(time_series) - 1)])
    s = np.array([0 for _ in range(len(time_series) - 1)])
    b = np.array([0 for _ in range(len(time_series) - 1)])
    c = np.array([0 for _ in range(len(time_series) - 1)])
    error = 0
    
    # First m pred are the time series
    pred[:m + 1] = time_series[:m + 1]
    s[0] = time_series[0]
    b[0] = time_series[1] - time_series[0]

    for i in range(1, len(time_series) - m - 1):
        s[i] = alfa * (time_series[i] - c[i - L]) + (1 - alfa) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
        c[i] = gamma * (time_series[i] - s[i] - b[i - 1]) + (1 - gamma) * c[i - L]
        pred[i + m] = s[i] + m * b[i] + c[i - L + 1 + (m - 1) % L]
        error += (pred[i + m] - time_series[i + m]) ** 2
    
    return error, pred

def ex2():
    # Simple mediation
    grid_size = 20
    alfa_grid = np.linspace(0, 1, grid_size)
    errors = [(i, exp_mediation(time_series, alfa)) for i, alfa in enumerate(alfa_grid)]

    ind_min_err, (min_err, best_pred) = min(errors, key=lambda x: x[1][0])
    ind_max_err, (max_err, worst_pred) = max(errors, key=lambda x: x[1][0])

    best_alfa = alfa_grid[ind_min_err]
    worst_alfa = alfa_grid[ind_max_err]

    print("Normal exp mediation:")
    print("Normal exp mediation:")
    print(f"Best alfa: {best_alfa:.2f} with error: {min_err:.2f}")
    print(f"Worst alfa: {worst_alfa:.2f} with error: {max_err:.2f}\n")

    plt.plot(time_series, label=f"Initial")
    plt.plot(best_pred, label=f"Simple exp med alfa={best_alfa:.2f}, err={min_err:.2f}")
    plt.legend()
    plt.savefig(f"{plot_dir}/ex2/simple_exp.pdf")
    plt.close()
    # axs[2].plot(worst_pred)

    # Double mediation
    m = 1
    alfa_beta = make_pairs(alfa_grid, alfa_grid)
    errors = [(i, double_exp_mediation(time_series, alfa, beta, m)) for i, (alfa, beta) in enumerate(alfa_beta)]

    ind_min_err, (min_err, best_pred) = min(errors, key=lambda x: x[1][0])
    ind_max_err, (max_err, worst_pred) = max(errors, key=lambda x: x[1][0])

    best_alfa, best_beta = alfa_beta[ind_min_err]
    worst_alfa, worst_beta = alfa_beta[ind_max_err]

    print("Double exp mediation:")
    print("Double exp mediation:")
    print(f"Best alfa: {best_alfa:.2f}, Best beta: {best_beta:.2f} with error: {min_err:.2f}")
    print(f"Worst alfa: {worst_alfa:.2f}, Worst beta: {worst_beta:.2f} with error: {max_err:.2f}\n")

    plt.plot(time_series, label=f"Initial")
    plt.plot(best_pred, label=f"Double exp med alfa={best_alfa:.2f}, beta={best_beta:.2f}, err={min_err:.2f}")
    plt.legend()
    plt.savefig(f"{plot_dir}/ex2/double_exp.pdf")
    plt.close()
    # axs[4].plot(worst_pred)


    # Triple additive mediation
    L = 3
    alfa_beta_gamma = make_triplets(alfa_grid, alfa_grid, alfa_grid)
    errors = [(i, triple_exp_mediation(time_series, alfa, beta, m, gamma, L)) for i, (alfa, beta, gamma) in enumerate(alfa_beta_gamma)]

    ind_min_err, (min_err, best_pred) = min(errors, key=lambda x: x[1][0])
    ind_max_err, (max_err, worst_pred) = max(errors, key=lambda x: x[1][0])

    best_alfa, best_beta, best_gamma = alfa_beta_gamma[ind_min_err]
    worst_alfa, worst_beta, worst_gamma = alfa_beta_gamma[ind_max_err]

    print("Triple exp mediation:")
    print(f"Best alfa: {best_alfa:.2f}, Best beta: {best_beta:.2f}, Best gamma: {best_gamma:.2f}  with error: {min_err:.2f}")
    print(f"Worst alfa: {worst_alfa:.2f}, Worst beta: {worst_beta:.2f}, Worst gamma: {worst_gamma:.2f} with error: {max_err:.2f}")

    plt.plot(time_series, label="Initial")
    plt.plot(best_pred, label=f"Triple exp med alfa={best_alfa:.2f}, beta={best_beta:.2f}, gamma={best_gamma:.2f}, err={min_err:.2f}")
    plt.legend()
    plt.savefig(f"{plot_dir}/ex2/triple_exp.pdf")

# ERROR? FIRST AND LAST ELEM OF PARAM SHOULD BE BUT ONLY THE LAST IS
# PROBABLY SHOULD NOT TAKE THE CURRENT DEVIATION AND THE MEAN WHEN COMPUTING PARAMS
def MA_model_fit(time_series:np.ndarray, horizon:int):
    time_series_samples = time_series[horizon:]
    deviations = np.zeros((time_series_samples.shape[0], horizon + 2))
    
    # Fill deviations matrix
    for i in range(horizon, time_series.shape[0]):
        curr_window = time_series[i - horizon : i + 1][::-1]
        mean = np.mean(curr_window)
        deviations[i - horizon] = np.array([elem - mean for elem in curr_window] + [mean])
    
    deviations_t = deviations.transpose()
    deviations_mult_with_trans = np.matmul(deviations_t, deviations)
    deviations_mult_with_trans += np.random.random(deviations_mult_with_trans.shape)

    # Calculate params
    params = np.matmul(np.matmul(np.linalg.inv(deviations_mult_with_trans), deviations_t), time_series_samples.transpose())

    return deviations, params

def MA_model_pred(time_series:np.ndarray, params:np.ndarray, deviations:np.ndarray):
    pred = np.zeros(time_series.shape)
    horizon = params.shape[0]
    pred[:horizon] = time_series[:horizon]
    err = 0
    for i in range(horizon - 1, pred.shape[0] - 1):
        pred[i + 1] = np.matmul(params.transpose(), deviations[i - horizon + 1])
        err += np.linalg.norm(time_series[i + 1] - pred[i + 1]) ** 2
    
    return err, pred

def AR_model_fit(x: np.ndarray, p:int, m:int):
    Y = np.array([x[x.shape[0] - m - i:x.shape[0] - i]for i in range(1, p + 1)]).transpose()
    x_samples =  x[x.shape[0] - m:]

    Y_mult_with_trans = np.matmul(Y.transpose(), Y)
    Y_mult_with_trans += np.random.random(Y_mult_with_trans.shape)
    params = np.matmul(np.matmul(np.linalg.inv(Y_mult_with_trans), Y.transpose()), x_samples.transpose())

    return params

def AR_model_pred(time_series: np.ndarray, params:np.ndarray):
    pred = np.zeros(time_series.shape)
    p = params.shape[0]
    pred[:p] = time_series[:p]
    err = 0
    for i in range(p - 1, pred.shape[0] - 1):
        pred[i + 1] = np.matmul(params.transpose(), time_series[i - p + 1 : i + 1])
        err += np.linalg.norm(time_series[i + 1] - pred[i + 1]) ** 2
    
    return err, pred

def ex3_4():
    # Train test MA model
    horizon = 10
    deviations, params_MA = MA_model_fit(time_series, horizon)
    err_MA, pred_MA = MA_model_pred(time_series, params_MA, deviations)

    # Train test AR model
    p = horizon
    m = horizon * 2
    params_AR = AR_model_fit(time_series, p, m)
    err_AR, pred_AR = AR_model_pred(time_series, params_AR)

    # Test ARMA: NOT GOOD SURELY
    pred_ARMA = 0.5 * (pred_AR + pred_MA)
    err_ARMA = np.linalg.norm(time_series - pred_ARMA) ** 2

    fig, axs = plt.subplots(3)

    axs[0].plot(time_series, label="Initial")
    axs[0].plot(pred_AR, label=f"AR p={p}, m={m}, err={err_AR:.2f}")
    axs[0].legend()

    axs[1].plot(time_series, label="Initial")
    axs[1].plot(pred_MA, label=f"MA pred horizon={horizon}, err={err_MA:.2f}")
    axs[1].legend()

    axs[2].plot(time_series, label="Initial")
    axs[2].plot(pred_ARMA, label=f"ARMA err={err_ARMA:.2f}")
    axs[2].legend()
    
    fig.savefig(f"{plot_dir}/ex3_4.pdf")

ex2()
ex3_4()