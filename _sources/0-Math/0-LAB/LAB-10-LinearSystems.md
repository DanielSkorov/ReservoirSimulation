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

(math-lab-linearsystems)=
# Решение систем линейных уравнений

Допустим, имеется следующая система линейных уравнений:

$$ \left\{\begin{array} \\ 3x + 2y + z = 5 \\ 2x + 3y + z = -1 \\ 2x + y + 3z = 3 \end{array}\right. $$

Левую часть уравнений в данной системе можно представить в виде произведения матрицы и вектора искомых переменных:

$$ \mathbf{A} \mathbf{x}, $$

где:

$$ \mathbf{A} = \begin{bmatrix} 3 & 2 & 1 \\ 2 & 3 & 1 \\ 2 & 1 & 3 \end{bmatrix}, \; \mathbf{x} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}. $$

Тогда в результате произведения матрицы $\mathbf{A}$ и вектора $\mathbf{x}$ получится вектор:

$$ \mathbf{A} \mathbf{x} = \begin{bmatrix} 3x + 2y + z \\ 2x + 3y + z \\ 2x + y + 3z \end{bmatrix}. $$

Правые части уравнений также можно представить в виде матрицы-столбца $\mathbf{b}$:

$$ \mathbf{b} = \begin{bmatrix} 5 \\ -1 \\ 3 \end{bmatrix}. $$

Тогда исходную систему линейных уравнений можно представить в виде матричного уравнения:

$$ \mathbf{A} \mathbf{x} = \mathbf{b}. $$

Домножим обе части уравнения на $\mathbf{A}^{-1}$:

$$ \mathbf{A}^{-1} \mathbf{A} \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}. $$

Левую часть уравнения можно упростить, применив определение обратной матрицы:

$$ \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}. $$

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
