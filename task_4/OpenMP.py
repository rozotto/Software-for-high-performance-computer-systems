import numpy as np
from numba import njit, prange  # импортируем модуль для работы JIT-компиляцией и распараллеливанием
import time

sizes = [10, 1000, 2000]


@njit(parallel=True)            # включаем JIT-компиляцию и позволяем prange использовать многопоточность
def matmul_openmp(A, B, C):     # умножение матриц с параллельной обработкой строк
    N = A.shape[0]
    for i in prange(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


print(f"{'Size':>8} | {'Time':>8}")
for N in sizes:
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    start = time.time()
    matmul_openmp(A, B, C)
    end = time.time()

    print(f"{N:8} | {end - start:10.6f}")
