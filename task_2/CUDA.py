from numba import cuda
import numpy as np
import time


@cuda.jit                                           # делаем функцию CUDA-ядром, которое выполняется на GPU
def sum_kernel(arr, partial_sums):
    idx = cuda.grid(1)                              # глобальный индекс потока

    if idx < arr.size:                              # получаем размер массива
        cuda.atomic.add(partial_sums, 0, arr[idx])  # атомарная операция


sizes = [10, 1000, 10_000_000]

print(f"{'Size':>12} | {'Sum':>12} | {'Time':>12}")
for N in sizes:
    arr = np.random.randint(0, 1000, size=N, dtype=np.int32)

    d_arr = cuda.to_device(arr)                                         # копируем массив на GPU
    d_sum = cuda.to_device(np.array([0], dtype=np.int32))

    threads_per_block = 128                                             # выбираем кол-во потоков
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block  # вычисляем количество блоков,
                                                                        # чтобы все элементы были покрыты потоками
    start = time.time()
    sum_kernel[blocks_per_grid, threads_per_block](d_arr, d_sum)
    cuda.synchronize()                                                  # ждем окончания всех операций на GPU
    end = time.time()

    total_sum = d_sum.copy_to_host()[0]
    print(f"{N:12} | {total_sum:12} | {end - start:10.10f}")
