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

(math-lab-detinv)=
# Определитель матрицы. Обратная матрица

## Определитель матрицы

Определитель матрицы первого порядка равен единственному элементу этой матрицы.

```{code-cell} python
A = np.array([[5]])
np.linalg.det(A)
```

Определитель квадратной матрицы размерностью $\left( 2 \times 2 \right)$:

$$ \begin{vmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{vmatrix} = a_{11} \cdot a_{22} - a_{12} \cdot a_{21}. $$

```{code-cell} python
A = np.array([[1, 2], [-3, -4]])
A
```

```{code-cell} python
np.linalg.det(A)
```

При нахождении определителя матрицы размерностью $\left( 3 \times 3 \right)$ необходимо "раскрыть" определитель по любой строке или столбцу с учетом матрицы знаков:

$$ \begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{vmatrix} = a_{11} \cdot \begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} - a_{12} \cdot \begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} + a_{13} \cdot \begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}. $$

```{code-cell} python
A = np.array([[1, 3, 14], [5, 2, -7], [4, -2, 7]])
np.linalg.det(A)
```

Одним из свойств векторов является то, что если определитель матрицы, составленной из координат этих векторов, равен нулю, то данные векторы являются [коллинеарными](LAB-2-VectorOperations.md#math-lab-vectoroperations-collinearity). Рассмотрим пример. Пусть даны два вектора $\mathbf{v_1} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$ и $\mathbf{v_2} = \begin{bmatrix} 2 \\ 6 \end{bmatrix}$. Тогда матрица $\mathbf{M} = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$. Вычислим определитель этой матрицы:

```{code-cell} python
M = np.array([[1, 2], [3, 6]])
np.linalg.det(M)
```

Определитель этой матрицы равен нулю, следовательно, векторы $\mathbf{v_1}$ и $\mathbf{v_2}$ являются коллинеарными.

## Обратная матрица

```{admonition} Определение
:class: tip
Под ***обратной матрицей*** матрице $\mathbf{M}$ называется матрица, удовлетворяющая следующему выражению:

$$ \mathbf{M} \mathbf{M}^{-1} = \mathbf{E}, $$

где $\mathbf{E}$ – единичная матрица.
```

При умножении матрицы на единичную матрицу в результате получается исходная матрица.

```{code-cell} python
A = np.array([[1, 2], [-1, 3]])
E = np.array([[1, 0], [0, 1]])
A.dot(E)
```

При нахождении обратной матрицы можно использовать следующее выражение:

$$ \mathbf{M}^{-1} = \frac{1}{\left| \mathbf{M} \right|} \cdot \mathbf{M}_{*}^\top, $$

где $\mathbf{M}_{*}^\top$ – транспонированная матрица алгебраических дополнений. Следует отметить, что, исходя из указанного выше выражения, необходимым условием для существования обратной матрицы является $\left| \mathbf{M} \right| \neq 0$.

Однако использование библиотеки [*numpy*](https://numpy.org/) может значительно упростить данную операцию.

```{code-cell} python
A_inv = np.linalg.inv(A)
A_inv
```

```{code-cell} python
A.dot(A_inv)
```

Матрица, обратная обратной матрице $\mathbf{A}$, равняется самой матрице $\mathbf{A}$:

```{code-cell} python
np.linalg.inv(np.linalg.inv(A))
```
