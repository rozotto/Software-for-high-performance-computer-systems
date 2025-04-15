# Сравнение производительности

## Задача 2

## Описание задачи

Посчитать сумму элементов массива. Замерить время выполнения на трёх фреймворках:
- **MPI**
- **OpenMP**
- **CUDA**

Размеры массива: `10`, `1_000`, `10_000_000`.

---

## 📊 Сравнительная таблица

| Размер массива | MPI (сек) | OpenMP (сек) | CUDA (сек) |
|----------------|-----------|--------------|------------|
| 10             | 0.0042    | 0.0011       | 0.0063     |
| 1,000          | 0.0056    | 0.0013       | 0.0069     |
| 10,000,000     | 0.0284    | 0.0121       | 0.0275     |
