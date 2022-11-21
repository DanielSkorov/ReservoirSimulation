---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} python
:tags: [hide-input]

import numpy as np
```

<a id='math-lab-matrix'></a>
# Матрицы

```{prf:определение}
:nonumber:
Под ***матрицей*** в общем смысле понимается совокупность строк и столбцов, на пересечении которых находяется элементы. В качестве элементов в разделе линейной алгебры используются действительные числа.
```

Рассмотрим следующую матрицу:

$$ M = \begin{bmatrix} 3 & 5 & -17 \\ -1 & 0 & 10 \end{bmatrix}. $$

С использованием *[numpy](https://numpy.org/)* матрица задается следующим образом:

```{code-cell} python
M = np.array([[3, 5, -17], [-1, 0, 10]])
M
```

Данная матрица состоит из двух строк и трех столбцов.

```{code-cell} python
M.shape
```

Если в матрице количество строк и/или столбцов равно 1, то такая матрица называется ***вектором***. Данная тема подробно освещалась в [разделе, посвященном векторам](./LAB-1-Vectors.md).

+++

Чтобы ***умножить матрицу на число***, необходимо умножить каждый ее элемент на данное число:

$$ \lambda \cdot M = \lambda \cdot \begin{bmatrix} m_{11} & \dots & m_{1k} \\ \vdots & \ddots & \vdots \\ m_{n1} & \dots & m_{nk} \end{bmatrix} = \begin{bmatrix} \lambda \cdot m_{11} & \dots & \lambda \cdot m_{1k} \\ \vdots & \ddots & \vdots \\ \lambda \cdot m_{n1} & \dots & \lambda \cdot m_{nk} \end{bmatrix}. $$

```{code-cell} python
2 * M
```

```{code-cell} python
M * 2
```

Чтобы ***сложить матрицы*** между собой, необходимо сложить их значения поэлементно. Можно складывать матрицы только с одинаковыми размерностями.

$$ M + P = \begin{bmatrix} m_{11} & \dots & m_{1k} \\ \vdots & \ddots & \vdots \\ m_{n1} & \dots & m_{nk} \end{bmatrix} + \begin{bmatrix} p_{11} & \dots & p_{1k} \\ \vdots & \ddots & \vdots \\ p_{n1} & \dots & p_{nk} \end{bmatrix} = \begin{bmatrix} m_{11} + p_{11} & \dots & m_{1k} + p_{1k} \\ \vdots & \ddots & \vdots \\ m_{n1} + p_{n1} & \dots & m_{nk} + p_{nk} \end{bmatrix}. $$

```{code-cell} python
P = np.array([[1, 3, 14], [5, 2, -7]])
P
```

```{code-cell} python
M + P
```

```{code-cell} python
P + M
```

***Произведением двух матриц*** $M$ и $P$ называется матрица $Q$, элемент которой, находящийся на пересечении $i$-ой строки и $j$-го столбца равен сумме произведений элементов $i$-ой строки матрицы $M$ на соответствующие (по порядку) элементы $j$-го столбца матрицы $P$.

$$ q_{ij}=m_{i1} \cdot p_{1j} + m_{i2} \cdot p_{2j} + \ldots + m_{in} \cdot p_{nj}. $$

Исходя из данного определения, количество столбцов матрицы $m$ должно быть равно количеству строк матрицы $p$.

```{code-cell} python
M = np.array([[1, 2], [-1, 3]])
M
```

```{code-cell} python
P = np.array([[3, 5, -2], [-1, 0, 10]])
P
```

```{code-cell} python
M.dot(P)
```

Для того чтобы ***транспонировать матрицу***, нужно ее строки записать в столбцы транспонированной матрицы.

```{code-cell} python
np.transpose(M)
```

Под ***следом матрицы*** понимается сумма компонентов главной диагонали квадратной матрицы.

```{code-cell} python
A = np.array([[1, 3, 14], [5, 2, -7], [4, -2, 7]])
A
```

```{code-cell} python
np.trace(A)
```
