from mpi4py import MPI  # импортируем библиотеку для использования MPI
import numpy as np

comm = MPI.COMM_WORLD   # коммутатор, объединяющий процессы в одну группу
rank = comm.Get_rank()  # возвращает ранг текущего процесса
size = comm.Get_size()  # возвращает количество процессов в группе

sizes = [10, 1000, 2000]

if rank == 0:
    print(f"{'Size':>8} | {'Time':>8}")

for N in sizes:
    rows_per_proc = N // size    # распределение строк по процессам
    remainder = N % size
    counts = [rows_per_proc + 1 if i < remainder else rows_per_proc for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    if rank == 0:  # на процессе 0 создаём две матрицы
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
    else:
        A = None
        B = None

    local_rows = counts[rank]    # каждый процесс получает свои части матрицы А
    A_local = np.zeros((local_rows, N), dtype=np.float64)
    sendcounts = [c * N for c in counts]
    displs_bytes = [d * N for d in displs]

    comm.Scatterv([A, sendcounts, displs_bytes, MPI.DOUBLE], A_local, root=0)

    if rank != 0:
        B = np.zeros((N, N), dtype=np.float64)
    comm.Bcast(B, root=0)

    comm.Barrier()
    start = MPI.Wtime()

    C_local = A_local @ B        # умножение матриц

    if rank == 0:
        C = np.zeros((N, N), dtype=np.float64)
    else:
        C = None

    recvcounts = [c * N for c in counts]    # собираем результат
    displs_recv = [d * N for d in displs]
    comm.Gatherv(C_local, [C, recvcounts, displs_recv, MPI.DOUBLE], root=0)

    comm.Barrier()
    end = MPI.Wtime()

    if rank == 0:
        print(f"{N:8} | {end - start:10.6f}")
