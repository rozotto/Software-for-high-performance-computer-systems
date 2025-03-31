from mpi4py import MPI  # импортируем библиотеку для использования MPI

comm = MPI.COMM_WORLD   # коммутатор, объединяющий процессы в одну группу
rank = comm.Get_rank()  # возвращает ранг текущего процесса
size = comm.Get_size()  # возвращает количество процессов в группе comm

print(f"MPI process {rank} out of {size}")   # процессы выводят сообщения из каждого потока по отдельности

"""
для запуска в консоль: mpiexec -np 16 python task_1.py 
"""
