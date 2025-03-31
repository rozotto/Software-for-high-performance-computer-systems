from numba import cuda
import numpy as np


@cuda.jit                                                       # делаем функцию CUDA-ядром, которое выполняется на GPU
def hello_cuda(output):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # считаем глобальный индекс потока по формуле
    if idx < output.size:
        output[idx] = idx                                       # записываем индекс в массив


num_threads = 10                                                # количество потоков
output_array = np.zeros(num_threads, dtype=np.int32)            # массив нулей

d_output = cuda.to_device(output_array)                         # копируем массив на GPU

hello_cuda[1, num_threads](d_output)                            # запускаем CUDA-ядро

output_array = d_output.copy_to_host()                          # копируем результат обратно на CPU

for i in output_array:
    print(f"CUDA thread {i}")
