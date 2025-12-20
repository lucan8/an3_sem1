import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

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

# ERROR? FIRST AND LAST ELEM OF PARAM SHOULD BE 1 BUT ONLY THE LAST IS
# PROBABLY SHOULD NOT TAKE THE CURRENT DEVIATION AND THE MEAN WHEN COMPUTING PARAMS
def MA_model_fit(time_series:np.ndarray, q:int):
    time_series_samples = time_series[q:]
    deviations = np.zeros((time_series_samples.shape[0], q + 2))
    
    # Fill deviations matrix
    for i in range(q, time_series.shape[0]):
        curr_window = time_series[i - q : i + 1][::-1]
        mean = np.mean(curr_window)
        deviations[i - q] = np.array([elem - mean for elem in curr_window] + [mean])
    
    deviations_t = deviations.transpose()
    deviations_mult_with_trans = np.matmul(deviations_t, deviations)
    deviations_mult_with_trans += np.random.random(deviations_mult_with_trans.shape)

    # Calculate params
    params = np.matmul(np.matmul(np.linalg.inv(deviations_mult_with_trans), deviations_t), time_series_samples.transpose())

    return deviations, params

def MA_model_pred(time_series:np.ndarray, params:np.ndarray, deviations:np.ndarray):
    pred = np.zeros(time_series.shape)
    q = params.shape[0]
    pred[:q] = time_series[:q]
    err = 0
    for i in range(q - 1, pred.shape[0] - 1):
        pred[i + 1] = np.matmul(params.transpose(), deviations[i - q + 1])
        err += np.linalg.norm(time_series[i + 1] - pred[i + 1]) ** 2
    
    return err, pred

def MA_model_fit_pred(time_series:np.ndarray, q:int):
    deviations, params_MA = MA_model_fit(time_series, q)
    return MA_model_pred(time_series, params_MA, deviations)

# Chooses the best MA model for time series using q_grid
def MA_best_model(time_series:np.ndarray, q_grid: list[int]):
    best_model_pred = None
    best_model_err = np.inf
    chosen_q = None

    for q in q_grid:
        err, pred = MA_model_fit_pred(time_series, q)
        if err < best_model_err:
            best_model_err = err
            best_model_pred = pred
            chosen_q = q
    return best_model_err, best_model_pred, chosen_q

def AR_model_fit(time_series: np.ndarray, p:int, m:int):
    ts_size = time_series.shape[0]
    Y = np.array([time_series[ts_size - m - i:ts_size - i]for i in range(1, p + 1)]).transpose()
    time_series_samples =  time_series[ts_size - m:]

    Y_mult_with_trans = np.matmul(Y.transpose(), Y)
    Y_mult_with_trans += np.random.random(Y_mult_with_trans.shape)
    params = np.matmul(np.matmul(np.linalg.inv(Y_mult_with_trans), Y.transpose()), time_series_samples.transpose())

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

def AR_model_fit_pred(time_series: np.ndarray, p:int, m:int):
    params_AR = AR_model_fit(time_series, p, m)
    return AR_model_pred(time_series, params_AR)

# m will be implicitly double p
def AR_best_model(time_series: np.ndarray, p_grid:list[int]):
    best_model_pred = None
    best_model_err = np.inf
    chosen_p = None

    for p in p_grid:
        m = p * 2
        err, pred = AR_model_fit_pred(time_series, p, m)
        if err < best_model_err:
            best_model_err = err
            best_model_pred = pred
            chosen_p = p
    return best_model_err, best_model_pred, chosen_p


def ARMA_best_model(time_series: np.ndarray, p_q_grid: list[tuple[int, int]]):
    smallest_sse = np.inf
    best_model = None
    chosen_p = None
    chosen_q = None

    # Get the best model based on sse
    for p, q in p_q_grid:
        try:
            model = ARIMA(time_series, order=(p, 0, q)).fit()

            if model.sse < smallest_sse:
                best_model = model
                smallest_sse = model.sse
                chosen_p = p
                chosen_q = q
        except Exception:
            pass
    
    pred, err = best_model.predict(), best_model.sse

    return err, pred, chosen_p, chosen_q

# Compare AR, MA, their average pred and library ARMA
# Note: their average pred was my initial interpretation of ARMA so I let it here for fun
def ex3_4():
    # Grid creation
    grid_size = 22
    p_grid = list(range(2, grid_size))
    q_grid = list(range(2, grid_size))
    p_q_grid = make_pairs(p_grid, q_grid)

    # Train test AR model
    err_AR, pred_AR, p_AR = AR_best_model(time_series, p_grid)
    print(f"AR model p={p_AR}, err={err_AR:.2f}\n")

    # Best MA model
    err_MA, pred_MA, q_MA = MA_best_model(time_series, q_grid)
    print(f"MA model q={q_MA}, err={err_MA:.2f}\n")

    # My interpretation of ARMA from the course: Averaging the predictions of AR and MA
    pred_My_ARMA = 0.5 * (pred_AR + pred_MA)
    err_MY_ARMA = np.linalg.norm(time_series - pred_My_ARMA) ** 2
    print(f"My ARMA model err={err_MY_ARMA:.2f}\n")

    # Actual ARMA
    err_ARMA, pred_ARMA, p_ARMA, q_ARMA = ARMA_best_model(time_series, p_q_grid)
    print(f"Real ARMA model p={p_ARMA}, q={q_ARMA}, err={err_ARMA:.2f}")

    fig, axs = plt.subplots(4)

    axs[0].plot(time_series, label="Initial")
    axs[0].plot(pred_AR, label=f"AR p={p_AR}, err={err_AR:.2f}")
    axs[0].legend()
    
    axs[1].plot(time_series)
    axs[1].plot(pred_MA, label=f"MA pred q={q_MA}, err={err_MA:.2f}")
    axs[1].legend()

    axs[2].plot(time_series)
    axs[2].plot(pred_My_ARMA, label=f"(AR + MA) / 2, err={err_MY_ARMA:.2f}")
    axs[2].legend()

    axs[3].plot(time_series)
    axs[3].plot(pred_ARMA, label=f"ARMA, p={p_ARMA}, q={q_ARMA}, err={err_ARMA:.2f}")
    axs[3].legend()
    
    fig.savefig(f"{plot_dir}/ex3_4.pdf")

ex2()
ex3_4()
