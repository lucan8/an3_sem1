import numpy as np
import matplotlib.pyplot as plt
import os

curr_dir = os.path.dirname(__file__)
lab_dir = os.path.dirname(curr_dir)
plot_dir = os.path.join(curr_dir, "graphs")

def AR_model_fit(time_series: np.ndarray, Y: np.ndarray):
    m = Y.shape[0]
    time_series_samples =  time_series[time_series.shape[0] - m:]

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

def AR_model_fit_pred(time_series: np.ndarray, Y: np.ndarray):
    params = AR_model_fit(time_series, Y)
    return params, AR_model_pred(time_series, params)

def AR_model_fit_pred_greedy(time_series: np.ndarray, Y: np.ndarray, m:int, p:int):
    best_params = np.zeros(p)
    ts_size = time_series.shape[0]

    chosen_cols_i = set()

    smallest_err_global = np.inf
    best_pred = None

    print(Y.shape)
    for _ in range(1, p + 1):
        smallest_err = np.inf
        chosen_ind = -1
        
        # Try each remaining column
        for j in range(p):
            if j in chosen_cols_i:
                continue
            # Create partial Y matrix
            col_ind = list(chosen_cols_i) + [j]
            Y_curr = Y[:, col_ind]
            
            # Determine params and pred err
            params = AR_model_fit(time_series, Y_curr)
            err, pred = AR_model_pred(time_series, params)

            # Choose the column that gives the smallest error
            if err < smallest_err:
                smallest_err = err
                chosen_ind = j
                if err < smallest_err_global: # Keep track of the results gloablly
                    smallest_err_global = err
                    best_pred = pred

                    params_with_zeros = np.zeros(best_params.shape)
                    params_with_zeros[col_ind] = params
                    best_params = params_with_zeros

        chosen_cols_i.add(chosen_ind)
    
    return best_params, smallest_err_global, best_pred


def AR_model_l1_reg(time_series_samples: np.ndarray, Y: np.ndarray, params: np.ndarray, lam: int):
    p = params.shape[0]
    
    res_params = params.copy()
    sq_norms = np.sum(Y**2, axis=0)

    param_diff_target = 1 / (10 ** 4)
    max_iter = 100

    for _ in range(max_iter):
        params_old = res_params.copy()
        for i in range(p):
            curr_Y = np.delete(Y, i, 1)
            curr_params = np.delete(res_params, i)

            ro = np.matmul(Y[:, i].transpose(), time_series_samples - np.matmul(curr_Y, curr_params))
            res_params[i] = soft_thresh(ro, lam) / sq_norms[i]

        param_diff = np.linalg.norm(res_params - params_old)
        if param_diff <= param_diff_target:
            break
    return res_params

def soft_thresh(ro, lam):
    if ro > lam:
        return ro - lam
    elif ro < -lam:
        return ro + lam
    return 0

# coef 0 is for the constant and so on
def calc_pol_roots(pol_coef: np.ndarray):
    # Eliminate all 0 that are to right of the first non-zero value, from right to left
    for i in range(len(pol_coef) - 1, -1, -1):
        if pol_coef[i] != 0:
            pol_coef = pol_coef[:i + 1]
            break

    pol_coef = pol_coef / pol_coef[-1]
    useful_coef = np.delete(-pol_coef, -1)

    # Construct companion matrix
    comp_mat = np.zeros((useful_coef.size, useful_coef.size))
    comp_mat[:, -1] = useful_coef.copy()
    np.fill_diagonal(comp_mat[1:], 1)

    roots, _ = np.linalg.eig(comp_mat)

    return roots

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

# Also returns the params for all the models
def ex_1_2_3():
    m = 30
    p = 10

    fig, axs = plt.subplots(3)

    ts_size = time_series.shape[0]

    # Everyone will use this matrix, construct it once
    Y = np.array([time_series[ts_size - i - m : ts_size - i][::-1]  for i in range(1, p + 1)]).transpose()

    # Normal AR model
    normal_ar_params, (err, pred) = AR_model_fit_pred(time_series, Y)
    axs[0].plot(time_series, label="Initial")
    axs[0].plot(pred, label=f"Default AR model: err:{err:.2f}, p:{p}, m:{m}")
    axs[0].legend()

    # Greedy AR model
    greedy_ar_params, err, pred = AR_model_fit_pred_greedy(time_series, Y, m, p)
    non_zero_param_count = np.nonzero(greedy_ar_params)[0].shape[0]

    axs[1].plot(time_series)
    axs[1].plot(pred, label=f"Greedy AR model: err:{err:.2f}, p:{p}, m:{m}, p_non_zero:{non_zero_param_count}")
    axs[1].legend()

    # L1 AR model
    lam = 20

    time_series_samples =  time_series[ts_size - m:]
    l2_ar_params = AR_model_l1_reg(time_series_samples, Y, normal_ar_params, lam)

    err, pred = AR_model_pred(time_series, l2_ar_params)

    non_zero_param_count = np.nonzero(l2_ar_params)[0].shape[0]

    axs[2].plot(time_series)
    axs[2].plot(pred, label=f"L1 reg AR model: err:{err:.2f}, p:{p}, m:{m}, p_non_zero:{non_zero_param_count}")
    axs[2].legend()

    fig.savefig(f"{plot_dir}/ex3.pdf")

    return normal_ar_params, greedy_ar_params ,l2_ar_params

def ex_4():
    pol = np.array([0, 2, -3, 1])
    print(calc_pol_roots(pol))

def ex_5():
    normal_ar_params, greedy_ar_params, l2_ar_params =  ex_1_2_3()

    print("Are models stationary???\n")
    print(f"Normal AR: {isStationary(normal_ar_params)}")
    print(f"Greedy AR: {isStationary(greedy_ar_params)}")
    print(f"L2 AR: {isStationary(l2_ar_params)}")

def isStationary(params: np.ndarray):
    pol_coef = np.array([1] + list(-params))
    
    roots = calc_pol_roots(pol_coef)
    return sum(np.abs(roots) > 1) == roots.size

ex_5()