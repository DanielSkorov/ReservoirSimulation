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
from sympy import Matrix, Symbol
```

<a id='math-lab-matrix_eq'></a>
# Матричные уравнения

+++

Допустим, имеется следующая система линейных уравнений:

$$ \left\{\begin{array}\\3x + 2y + z = 5 \\ 2x + 3y + z = -1 \\ 2x + y + 3z = 3\end{array}\right. $$

Левую часть уравнений в данной системе можно представить в виде произведения двух матриц:

$$ A \cdot X, $$

где:

$$ A = \begin{bmatrix} 3 & 2 & 1 \\ 2 & 3 & 1 \\ 2 & 1 & 3 \end{bmatrix}, \; X = \begin{bmatrix} x \\ y \\ z \end{bmatrix}. $$

```{code-cell} python
from sympy import Matrix, Symbol
A = Matrix([[3, 2, 1], [2, 3, 1], [2, 1, 3]])
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
X = Matrix([[x], [y], [z]])
A * X
```

Правые части уравнений также можно представить в виде матрицы-столбца $B$:

$$ B = \begin{bmatrix} 5 \\ -1 \\ 3 \end{bmatrix}. $$

Тогда исходную систему линейных уравнений можно представить в виде матричного уравнения:

$$ A \cdot X = B. $$

Домножим обе части уравнения на $A^{-1}$:

$$ A^{-1} \cdot A \cdot X = A^{-1} \cdot B. $$

Левую часть уравнения можно упростить, применив определение обратной матрицы:

$$ X = A^{-1} \cdot B. $$

```{code-cell} python
A = np.array([[3, 2, 1], [2, 3, 1], [2, 1, 3]])
B = np.array([[5], [-1], [3]])
X = np.linalg.inv(A).dot(B)
X
```

Проверка:

```{code-cell} python
A.dot(X)
```
