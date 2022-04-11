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

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
```

+++

<a id='math-lab-det_inv'></a>
# Определитель матрицы. Обратная матрица

+++

Определитель матрицы первого порядка равен единственному элементу этой матрицы.

+++

```{code-cell} ipython3
A = np.array([[5]])
np.linalg.det(A)
```

+++

Определитель квадратной матрицы размерностью $(2\times2)$:

+++

$$ \begin{vmatrix}a_{11} & a_{12} \\ a_{21} & a_{22}\end{vmatrix}=a_{11} \cdot a_{22} - a_{12} \cdot a_{21}. $$

+++

```{code-cell} ipython3
A = np.array([[1, 2], [-3, -4]])
A
```

+++

```{code-cell} ipython3
np.linalg.det(A)
```

+++

При нахождении определителя матрицы размерностью $(3\times3)$ необходимо "раскрыть" определитель по любой строке или столбцу с учетом матрицы знаков:

+++

$$ \begin{vmatrix}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{vmatrix} = a_{11} \cdot \begin{vmatrix}a_{22} & a_{23} \\ a_{32} & a_{33}\end{vmatrix} - a_{12} \cdot \begin{vmatrix}a_{21} & a_{23} \\ a_{31} & a_{33}\end{vmatrix} + a_{13} \cdot \begin{vmatrix}a_{21} & a_{22} \\ a_{31} & a_{32}\end{vmatrix}. $$

+++

```{code-cell} ipython3
A = np.array([[1, 3, 14], [5, 2, -7], [4, -2, 7]])
np.linalg.det(A)
```

+++

Одним из свойств векторов является то, что если определитель матрицы, составленной из координат этих векторов, равен нулю, то данные векторы являются [коллинеарными](LAB-2-VectorOperations.html#math-lab-collinearity). Рассмотрим пример. Пусть даны два вектора $\vec{v_1}=\begin{bmatrix} 1 \\ 3 \end{bmatrix}$ и $\vec{v_2}=\begin{bmatrix} 2 \\ 6 \end{bmatrix}$. Тогда матрица $M=\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$. Вычислим определитель этой матрицы:

+++

```{code-cell} ipython3
M = np.array([[1, 2], [3, 6]])
np.linalg.det(M)
```

+++

Определитель этой матрицы равен нулю, следовательно, векторы $\vec{v_1}$ и $\vec{v_2}$ являются коллинеарными.

+++

```{prf:определение}
:nonumber:
Под ***обратной матрицей*** матрице $M$ называется матрица, удовлетворяющая следующему выражению:
+++
$$ M \cdot M^{-1} = E, $$
+++
где $E$ – единичная матрица.
```

+++

При умножении матрицы на единичную матрицу в результате получается исходная матрица.

+++

```{code-cell} ipython3
A = np.array([[1, 2], [-1, 3]])
E = np.array([[1, 0], [0, 1]])
A.dot(E)
```

+++

При нахождении обратной матрицы чаще всего используют формулу:

+++

$$ M^{-1}=\frac{1}{|M|} \cdot M_{*}^{T}, $$

+++

где $M_{*}^{T}$ – транспонированная матрица алгебраических дополнений. Следует отметить, что, исходя из указанного выше выражения, необходимым условием для существования обратной матрицы является $|M| \neq 0$.

+++

Однако использование библиотеки [*numpy*](https://numpy.org/) может значительно упростить данную операцию.

```{code-cell} ipython3
A_inv = np.linalg.inv(A)
A_inv
```

+++

```{code-cell} ipython3
A.dot(A_inv)
```

+++

Матрица, обратная обратной матрице $A$, равняется самой матрице $A$:

```{code-cell} ipython3
np.linalg.inv(np.linalg.inv(A))
```
