# Сравнение производительности

## Задача 2

### Описание задачи

Посчитать сумму элементов массива.

Размеры массива: `10`, `1_000`, `10_000_000`.


## 📊 Сравнительная таблица

| Размер массива |    MPI    |    OpenMP    |    CUDA    |
|----------------|-----------|--------------|------------|
| 10             | 0.0107    | 0.6163       | 0.1424     |
| 1.000          | 0.0012    | 0.0013       | 0.0009     |
| 10.000.000     | 0.0254    | 0.0011       | 0.0001     |

---

## Задача 3

### Описание задачи

Посчитать производную какой-либо фунции от двух переменных.

Размеры массива: `10`, `1_000`, `2_000`

## 📊 Сравнительная таблица

| Размер массива |    MPI    |    OpenMP    |    CUDA    |
|----------------|-----------|--------------|------------|
| 10             | 0.000048  | 0.413938     | 0.154647   |
| 1.000          | 0.007923  | 0.001002     | 0.001000   |
| 2.000          | 0.034111  | 0.002046     | 0.001001   |

---
