from numba import njit, prange  # импортируем модуль для работы JIT-компиляцией и распараллеливанием
import numpy as np
import time


@njit(parallel=True)            # включаем JIT-компиляцию и позволяем prange использовать многопоточность
def parallel_sum(arr):          # функция для подсчета суммы
    partial = np.zeros(len(arr), dtype=np.int32)
    for i in prange(len(arr)):
        partial[i] = arr[i]
    return np.sum(partial)


array_sizes = [10, 1000, 10_000_000]

print(f"{'Size':>12} | {'Sum':>12} | {'Time':>12}")
for N in array_sizes:
    arr = np.random.randint(0, 1000, size=N, dtype=np.int32)
    start = time.time()
    total = parallel_sum(arr)
    end = time.time()
    print(f"{N:12} | {total:12} | {end - start:10.10f}")
