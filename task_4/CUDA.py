import numpy as np
from numba import cuda
import time


@cuda.jit                                 # делаем функцию CUDA-ядром, которое выполняется на GPU
def matmul_cuda(A, B, C, N):
    row, col = cuda.grid(2)               # глобальный индекс потока
    if row < N and col < N:
        tmp = 0.0
        for k in range(N):
            tmp += A[row, k] * B[k, col]  # произведение матриц
        C[row, col] = tmp


sizes = [10, 1000, 2000]
print(f"{'Size':>8} | {'Time':>8}")

for N in sizes:
    A = np.random.rand(N, N).astype(np.float32)  # данные на CPU
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    d_A = cuda.to_device(A)                      # копируем на GPU
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((N, N), dtype=np.float32)

    threads_per_block = (16, 16)                 # считаем размеры блоков и сетки
    blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    cuda.synchronize()
    start = time.time()

    matmul_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C, N)

    cuda.synchronize()
    end = time.time()

    C = d_C.copy_to_host()

    print(f"{N:8} | {end - start:10.6f}")
