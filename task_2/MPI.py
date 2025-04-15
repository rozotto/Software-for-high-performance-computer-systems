import numpy as np
import time
from mpi4py import MPI  # импортируем библиотеку для использования MPI

comm = MPI.COMM_WORLD   # коммутатор, объединяющий процессы в одну группу
rank = comm.Get_rank()  # возвращает ранг текущего процесса
size = comm.Get_size()  # возвращает количество процессов в группе comm

array_sizes = [10, 1000, 10_000_000]  # размеры массивов для анализа времени

if rank == 0:
    print(f"{'Size':>12} | {'Sum':>12} | {'Time':>12}")  # выводим заголовок таблицы

for N in array_sizes:
    if rank == 0:                                                   # создаем массив только на процессе 0
        data = np.random.randint(0, 1000, size=N, dtype=np.int32)   # заполняем массив рандомными числами
        chunks = np.array_split(data, size)                         # делим массив между процессами
        start_time = time.time()                                    # начинаем считать время подсчета суммы
    else:
        chunks = None

    local_data = comm.scatter(chunks, root=0)                       # распределяем подмассив каждому процессу
    local_sum = np.sum(local_data)                                  # каждый процесс считает сумму своего куска
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)          # суммируем все локальные суммы на процессе 0

    if rank == 0:  # на нулевом процессе выводим результат
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{N:12} | {total_sum:12} | {elapsed:10.10f}")
