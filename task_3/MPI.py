from mpi4py import MPI  # импортируем библиотеку для использования MPI
import numpy as np


comm = MPI.COMM_WORLD   # коммутатор, объединяющий процессы в одну группу
rank = comm.Get_rank()  # возвращает ранг текущего процесса
size = comm.Get_size()  # возвращает количество процессов в группе


def f(x, y):
    return np.sin(x) * np.cos(y) + x**2


sizes = [10, 1000, 2000]
if rank == 0:
    print(f"{'Size':>8} | {'Time':>8}")

for N in sizes:
    M = N                      # квадратная сетка N x M
    dx = 1.0                   # шаг по x

    rows_per_proc = N // size  # распределение строк по процессам
    remainder = N % size

    if rank == 0:              # на процессе 0 создаём сетку и рассчитываем значения функции
        X, Y = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
        A = f(X, Y).astype(np.float64)
    else:
        A = None

    local_rows = rows_per_proc + (1 if rank < remainder else 0)  # каждый процесс получает свои строки
    sendcounts = [(rows_per_proc + (1 if r < remainder else 0)) * M for r in range(size)]
    displacements = [sum(sendcounts[:r]) for r in range(size)]

    local_A_flat = np.zeros(local_rows * M, dtype=np.float64)
    comm.Scatterv([A.flatten() if rank == 0 else None, sendcounts, displacements, MPI.DOUBLE], local_A_flat, root=0)
    local_A = local_A_flat.reshape((local_rows, M))

    comm.Barrier()
    start_time = MPI.Wtime()

    # добавляем по одной строке сверху и снизу, обрабатывая края, для вычисления производной
    if rank > 0:
        upper = np.empty(M, dtype=np.float64)
        comm.Sendrecv(local_A[0], dest=rank-1, recvbuf=upper, source=rank-1)
        local_A = np.vstack([upper, local_A])
    else:
        local_A = np.vstack([local_A[0], local_A])  # повтор первой строки

    if rank < size - 1:
        lower = np.empty(M, dtype=np.float64)
        comm.Sendrecv(local_A[-1], dest=rank+1, recvbuf=lower, source=rank+1)
        local_A = np.vstack([local_A, lower])
    else:
        local_A = np.vstack([local_A, local_A[-1]])  # повтор последней строки

    # вычисляем производную по x методом центральной разности
    B_local = (local_A[2:, :] - local_A[:-2, :]) / (2 * dx)

    B_flat = B_local.flatten()
    recvcounts = [sendcounts[r] for r in range(size)]
    B_global = None
    if rank == 0:
        B_global = np.empty((N * M), dtype=np.float64)

    comm.Gatherv(B_flat, [B_global, recvcounts, displacements, MPI.DOUBLE], root=0)

    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"{N:8} | {end_time - start_time:10.6f}")
