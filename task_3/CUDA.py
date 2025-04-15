import numpy as np
from numba import cuda
import math
import time


def f(x, y):
    return np.sin(x) * np.cos(y) + x ** 2


sizes = [10, 1000, 2000]


@cuda.jit                                                 # делаем функцию CUDA-ядром, которое выполняется на GPU
def df_dx_cuda(A, B, dx, N, M):
    i, j = cuda.grid(2)                                   # глобальный индекс потока

    if 1 <= i < N - 1 and j < M:
        B[i, j] = (A[i + 1, j] - A[i - 1, j]) / (2 * dx)  # производная методом центральной разности


print(f"{'Size':>8} | {'Time':>8}")
for N in sizes:
    M = N
    dx = 1.0

    X, Y = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    A_host = f(X, Y).astype(np.float32)
    B_host = np.zeros_like(A_host)

    # копируем массив на GPU
    A_device = cuda.to_device(A_host)
    B_device = cuda.to_device(B_host)

    # считаем размеры блоков и сетки
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(N / threads_per_block[0])
    blocks_per_grid_y = math.ceil(M / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time.time()
    df_dx_cuda[blocks_per_grid, threads_per_block](A_device, B_device, dx, N, M)
    cuda.synchronize()  # ждем окончания всех операций на GPU
    end = time.time()

    B_host = B_device.copy_to_host()

    print(f"{N:8} | {end - start:10.6f}")
