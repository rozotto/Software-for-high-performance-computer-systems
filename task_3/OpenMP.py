import numpy as np
from numba import njit, prange  # импортируем модуль для работы JIT-компиляцией и распараллеливанием
import time


def f(x, y):
    return np.sin(x) * np.cos(y) + x ** 2


@njit(parallel=True)           # включаем JIT-компиляцию и позволяем prange использовать многопоточность
def compute_df_dx(A, dx, B):   # производная методом центральной разности
    N, M = A.shape
    for i in prange(1, N - 1):
        for j in range(M):
            B[i, j] = (A[i + 1, j] - A[i - 1, j]) / (2 * dx)


sizes = [10, 1000, 2000]

print(f"{'Size':>8} | {'Time':>8}")
for N in sizes:
    M = N
    dx = 1.0

    X, Y = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    A = f(X, Y).astype(np.float64)
    B = np.zeros_like(A)

    start = time.time()
    compute_df_dx(A, dx, B)
    end = time.time()

    print(f"{N:8} | {end - start:10.6f}")
