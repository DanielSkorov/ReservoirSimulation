---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 3.0.1
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} python
:tags: ['remove-input']

import numpy as np
```

(math-lab-matrix)=
# Матрицы

(math-lab-matrix-definition)=
## Определение и объявление

```{admonition} Определение
:class: tip
Под ***матрицей*** в общем смысле понимается совокупность строк и столбцов, на пересечении которых находяется элементы. В качестве элементов в разделе линейной алгебры используются действительные числа.
```

````{margin}
```{admonition} Дополнительно
:class: note
Существуют также многомерные матрицы, называемые *тензорами*. Например, трехмерный тензор объявляется следующим образом:

$$ \exists ~ \mathbf{T} \in {\rm I\!R}^{n \times m \times k}. $$

```
````

Для обозначения матрицы здесь и далее будут использоваться заглавные латинские буквы, выделенные жирным, например, $\mathbf{A}$. В отличие от [вектора](LAB-1-Vectors.md), представляющего собой одномерный набор действительных чисел, матрица является двумерной. Например, следующая запись

$$ \exists ~ \mathbf{A} \in {\rm I\!R}^{n \times m} $$

читается следующим образом: *существует* $\left( \exists \right)$ *матрица* $\left( \mathbf{A} \right)$ *, принадлежащая пространству действительных чисел* $\left( {\rm I\!R} \right)$, *размерностью* $n \times m$. Данная запись объявляет матрицу $\mathbf{A}$ следующего вида:

$$ \mathbf{A} = \begin{bmatrix} a_{1,1} & a_{1,2} & \ldots & a_{1,m} \\ a_{2,1} & a_{2,2} & \ldots & a_{2,m} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n,1} & a_{n,2} & \ldots & a_{n,m} \end{bmatrix}. $$

Кроме того, матрица может обозначаться путем указания элементов с индексами, например, для изложенного выше примера:

$$ \mathbf{A} = \left\{ a_{ij}, \, i=1 \ldotp \ldotp n, \, j = 1 \ldotp \ldotp m \right\}. $$

<!-- ```{admonition} Определение
:class: tip
***Рангом матрицы*** с $n$ строками и $m$ столбцами называется максимальное число линейно независимых строк или столбцов (то есть таких строк и столбцов, которые не могут быть выражены линейно через другие) матрицы.
``` -->

```{admonition} Определение
:class: tip
***Квадратной матрицей*** называется матрица, у которой число строк равняется числу столбцов (и это число называется порядком).
```

Рассмотрим следующую *прямоугольную* матрицу:

$$ \mathbf{M} = \begin{bmatrix} 3 & 5 & -17 \\ -1 & 0 & 10 \end{bmatrix}. $$

С использованием *[numpy](https://numpy.org/)* матрица задается следующим образом:

```{code-cell} python
M = np.array([[3, 5, -17], [-1, 0, 10]])
M
```

Данная матрица состоит из двух строк и трех столбцов.

```{code-cell} python
M.shape
```

Если в матрице количество строк (столбцов) равно 1, то такая матрица называется ***вектором-строкой*** (***вектором-столбцом***). Данная тема подробно освещалась в [разделе, посвященном векторам](LAB-1-Vectors.md).

(math-lab-matrix-dotnumber)=
## Умножение матрицы на число

Чтобы ***умножить матрицу на число***, необходимо умножить каждый ее элемент на данное число:

$$ \lambda \cdot \mathbf{M} = \lambda \mathbf{M} = \lambda \cdot \begin{bmatrix} m_{11} & \dots & m_{1k} \\ \vdots & \ddots & \vdots \\ m_{n1} & \dots & m_{nk} \end{bmatrix} = \begin{bmatrix} \lambda \cdot m_{11} & \dots & \lambda \cdot m_{1k} \\ \vdots & \ddots & \vdots \\ \lambda \cdot m_{n1} & \dots & \lambda \cdot m_{nk} \end{bmatrix}. $$

```{code-cell} python
2 * M
```

```{code-cell} python
M * 2
```

(math-lab-matrix-addition)=
## Сложение матриц

Чтобы ***сложить матрицы*** между собой, необходимо сложить их значения поэлементно. Можно складывать матрицы только с одинаковыми размерностями.

$$ \mathbf{M} + \mathbf{P} = \begin{bmatrix} m_{11} & \dots & m_{1k} \\ \vdots & \ddots & \vdots \\ m_{n1} & \dots & m_{nk} \end{bmatrix} + \begin{bmatrix} p_{11} & \dots & p_{1k} \\ \vdots & \ddots & \vdots \\ p_{n1} & \dots & p_{nk} \end{bmatrix} = \begin{bmatrix} m_{11} + p_{11} & \dots & m_{1k} + p_{1k} \\ \vdots & \ddots & \vdots \\ m_{n1} + p_{n1} & \dots & m_{nk} + p_{nk} \end{bmatrix}. $$

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

(math-lab-matrix-multiplication)=
## Произведение матриц

***Произведением двух матриц*** $\mathbf{M}$ и $\mathbf{P}$ называется матрица $\mathbf{Q}$, элемент которой, находящийся на пересечении $i$-ой строки и $j$-го столбца равен сумме произведений элементов $i$-ой строки матрицы $\mathbf{M}$ на соответствующие (по порядку) элементы $j$-го столбца матрицы $\mathbf{P}$.

$$ q_{ij} = m_{i1} \cdot p_{1j} + m_{i2} \cdot p_{2j} + \ldots + m_{in} \cdot p_{nj}. $$

Исходя из данного определения, количество столбцов матрицы $\mathbf{M}$ должно быть равно количеству строк матрицы $\mathbf{P}$. Произведение двух матриц будет обозначаться $\mathbf{M} \mathbf{P}$

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

Результатом произведения некоторой матрицы на *единичную матрицу* является эта же самая матрица:

```{code-cell} python
I = np.identity(P.shape[1], dtype=int)
P.dot(I)
```

```{admonition} Определение
:class: tip
***Единичная матрица*** является квадратной матрицей, на главной диагонали которой расположены единицы, а все остальные элементы равны нулю.
```

```{code-cell} python
I
```

(math-lab-matrix-dotvector)=
## Произведение матрицы и вектора

В результате произведения матрицы и вектора получится вектор:

$$ \mathbf{M} \mathbf{a} = \mathbf{b}. $$

```{code-cell} python
a = np.array([1, 4])
M.dot(a)
```

Кроме того, произведение матрицы $\mathbf{A} = \left\{ a_{ij}, \, i=1 \ldotp \ldotp n, \, j = 1 \ldotp \ldotp m \right\}$ и вектора $\mathbf{b} = \left\{b_j, \, j = 1 \ldotp \ldotp m \right\}$ можно записать следующим образом:

$$ \sum_{j=1}^m a_{ij} \cdot b_j = c_i, \; i = 1 \, \ldots \, n. $$

```{code-cell} python
A = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
b = np.array([4, 5, 6])
np.sum(A * b, axis=1)
```

```{code-cell} python
A.dot(b)
```

Результатом произведения единичной матрицы и вектора является этот же вектор.

```{code-cell} python
I = np.identity(b.shape[0], dtype=int)
I.dot(b)
```

(math-lab-matrix-transpose)=
## Транспонирование матрицы

Для того чтобы ***транспонировать матрицу***, нужно ее строки записать в столбцы транспонированной матрицы. Транспонирование обозначается символом $^\top$ или изменением индексов матрицы:

$$ \begin{align} & \mathbf{A} = \left\{ a_{ij}, \, i=1 \ldotp \ldotp n, \, j = 1 \ldotp \ldotp m \right\}, \\ & \mathbf{A}^{\top} = \left\{ a_{ji}, \, j = 1 \ldotp \ldotp m, \, i=1 \ldotp \ldotp n \right\}. \end{align} $$

```{code-cell} python
A
```

```{code-cell} python
A.T
```

```{admonition} Определение
:class: tip
***Симметричной матрицей*** называется *квадратная матрица*, совпадающая со своей транспонированной матрицей, то есть:

$$ A = A^{\top}. $$

```

Например, единичная матрица является симметричной:

```{code-cell} python
np.allclose(I, I.T)
```

Например, симметричная матрица получается в результате [тензорного произведения вектора](LAB-2-VectorOperations.md#math-lab-vectoroperations-outerproduct) на самого себя:

```{code-cell} python
np.outer(b, b)
```

(math-lab-matrix-trace)=
## След матрицы

Под ***следом матрицы*** понимается сумма компонентов главной диагонали квадратной матрицы.

```{code-cell} python
A = np.array([[1, 3, 14], [5, 2, -7], [4, -2, 7]])
A
```

```{code-cell} python
np.trace(A)
```
