import numpy as np

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
N = 15
trend_x = np.random.random(N)
trend_of_time_series = trend(trend_x)
seasonality_of_time_series = seasonality(np.linspace(0, N, N))
noise_of_time_series = np.random.normal(0, 1, N)
time_series = trend_of_time_series + seasonality_of_time_series + noise_of_time_series

# EX 2:

window_size = 5
Hankel_mat_shape = (window_size, time_series.shape[0] - window_size + 1)
Hankel_mat = np.zeros(Hankel_mat_shape)

for i in range(len(Hankel_mat)):
    Hankel_mat[i] = time_series[i:time_series.shape[0] - window_size + i + 1].copy()

# EX 3:

Hankel_squared = np.matmul(Hankel_mat, Hankel_mat.transpose())
Hankel_squared_eigenvalues, _ = np.linalg.eig(Hankel_squared)

Hankel_singular_values = np.linalg.svd(Hankel_mat)

# Observation: The middle matrix(S) of Hankel_singular_values is the sqrt(Hankel_squared_eigenvalues)

# Sqrt + sort desc
sqrt_Hankel_squared_eigenvalues = np.sqrt(sorted(Hankel_squared_eigenvalues, key=lambda x: -x))
assert np.allclose(sqrt_Hankel_squared_eigenvalues, Hankel_singular_values.S)

# EX 4:
Hankel_mat_rank = np.linalg.matrix_rank(Hankel_mat)
S = Hankel_singular_values.S[:Hankel_mat_rank]
U = Hankel_singular_values.U[:Hankel_mat_rank]
V_t = Hankel_singular_values.Vh[:Hankel_mat_rank]

Hankel_mat_comp = np.zeros((Hankel_mat_rank, Hankel_mat_shape[0], Hankel_mat_shape[1]))
for i in range(Hankel_mat_rank):
    Hankel_mat_comp[i] = S[i] * np.outer(U[i], V_t[i])