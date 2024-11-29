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

(math-lab-eigen)=
# Собственные векторы и значения матриц

Пусть дана матрица:

$$ \mathbf{A} = \begin{bmatrix} -1 & -6 \\ 2 & 6 \end{bmatrix}. $$

Умножим матрицу $\mathbf{A}$ на вектор $\mathbf{u} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}$:

$$ \mathbf{A} \mathbf{u} = \begin{bmatrix} -1 & -6 \\ 2 & 6 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix}=\begin{bmatrix} -1 \cdot 2 + (-6) \cdot (-1) \\ 2 \cdot 2 + 6 \cdot (-1) \end{bmatrix} = \begin{bmatrix} 4 \\ -2 \end{bmatrix} = 2 \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \lambda \mathbf{u}. $$

В результате умножения матрицы $\mathbf{A}$ на вектор $\mathbf{u}$ получился тот же самый вектор $\mathbf{u}$ с числовым коэффициентом $\lambda = 2$:

$$ \mathbf{A} \mathbf{u} = \lambda \mathbf{u}. $$

Такой вектор $\mathbf{u}$ называется ***собственным вектором*** (*eigenvector*) матрицы $\mathbf{A}$, а $\lambda$ – ***собственным значением*** матрицы $\mathbf{A}$ (*eigenvalue*).

```{admonition} Определение
:class: tip
Ненулевой вектор $\mathbf{u}$, который при умножении на некоторую квадратную матрицу $\mathbf{A}$ преобразуется в самого же себя с числовым коэффициентом $\lambda$, называется ***собственным вектором*** матрицы $\mathbf{A}$, а число $\lambda$ – ***собственным значением*** матрицы $\mathbf{A}$.
```

Нахождение собственного вектора и собственного значения некоторой матрицы реализуется с использованием следующего алгоритма. Выражение, полученное из определения собственного вектора, можно преобразовать:

$$ \mathbf{A} \mathbf{u} - \lambda \mathbf{u} = 0. $$

$$ \begin{bmatrix} \mathbf{A} - \lambda \mathbf{E} \end{bmatrix} \mathbf{u} = 0. $$

Поскольку тривиальное решение данного уравнения не удовлетворяет условию, указанному в определении собственного вектора $\left( \mathbf{u} \neq 0 \right)$, то необходимо, чтобы:

$$ \det \left(\mathbf{A} - \lambda \mathbf{E} \right) = 0. $$

Данное уравнение называется характеристическим для матрицы $\mathbf{A}$. Отсюда выражается $\lambda$. Последующее определение $\mathbf{u}$ основано на решении уравнения $\begin{bmatrix} \mathbf{A} - \lambda \mathbf{E} \end{bmatrix} \mathbf{u} = 0$ относительно $\mathbf{u}$.

```{admonition} Доказательство
:class: proof
Предположим, что $\det \left( \mathbf{A} - \lambda \mathbf{E} \right) \neq 0$. Следовательно, существует такая обратная матрица, что:

$$ {\begin{bmatrix} \mathbf{A} - \lambda \mathbf{E} \end{bmatrix}}^{-1} \begin{bmatrix} \mathbf{A} - \lambda \mathbf{E} \end{bmatrix} \mathbf{u} = {\begin{bmatrix} \mathbf{A} - \lambda \mathbf{E} \end{bmatrix}}^{-1} \cdot 0, $$

откуда следует:

$$ \mathbf{u} = 0, $$

что противоречит условию $(\mathbf{u} \neq 0)$.
```

Рассмотрим пример. Пусть дана матрица $\mathbf{A}$:

$$ \mathbf{A} = \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix}. $$

Необходимо найти собственные значения и собственные векторы матрицы $\mathbf{A}$. Для начала найдем собственные значения. Запишем уравнение и решим его относительно $\lambda$:

$$ \begin{align}
\begin{vmatrix} \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - \lambda \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\end{vmatrix} &= 0 \\ \begin{vmatrix} 4 - \lambda & -5 \\ 2 & -3 - \lambda \end{vmatrix} &= 0 \\ (4 - \lambda) \cdot (-3 - \lambda) + 10 &= 0 \\ {\lambda}^2 - \lambda - 2 &= 0 \\ {\lambda}_{1,2} &= (-1, 2)
\end{align} $$

При $\lambda = -1$:

$$ \begin{align}
\left( \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - (-1) \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \cdot \mathbf{u} &= 0 \\ \begin{bmatrix} 5 & -5 \\ 2 & -2 \end{bmatrix} \cdot \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &= 0.
\end{align} $$

Данное выражение может быть преобразовано в систему линейных уравнений и решено с использованием [метода Гаусса](https://en.wikipedia.org/wiki/Gaussian_elimination). Его целью является получение "треугольной матрицы нулей" путем простейших математических преобразований – сложения и умножения. В результате, получится следующее выражение:

$$ \begin{matrix}\\ 1 & -1 & 0\\ 0 & 0 & 0 \end{matrix}. $$

Исходя из второй строчки, $u_2$ может принимать любые значения. Поэтому пусть $u_2 = 1$. Тогда $u_1 = 1$. Отнормируем $u_1$ и $u_2$ на величину $\sqrt{{u_1}^2 + {u_2}^2}$ исключительно для сопоставления с результатом, получаемым в *[numpy](https://numpy.org/)* (далее будет показано, что данная операция, по сути, не играет существенной роли). Тогда $u_1 = u_2 = \frac{1}{\sqrt{2}}$.

При $\lambda = 2$:

$$ \begin{align} \left( \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - 2 \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \cdot u &= 0 \\ \begin{bmatrix} 2 & -5 \\ 2 & -5 \end{bmatrix} \cdot \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &= 0 \end{align} $$

После применения метода Гаусса получим:

$$ \begin{matrix}\\ 2 & -5 & 0\\ 0 & 0 & 0 \end{matrix}. $$

Исходя из второй строчки, $u_2$ может принимать любые значения. Поэтому пусть $u_2 = 1$. Тогда $u_1 = \frac{5}{2}$. Аналогично отнормируем $u_1$ и $u_2$ на величину $\sqrt{{u_1}^2 + {u_2}^2}$. Тогда $u_1 = \frac{5}{\sqrt{29}}$ и $u_2 = \frac{2}{\sqrt{29}}$.

```{code-cell} python
Lambda = np.array([2, -1])
u = np.array([[5 / 29**0.5, 1 / 2**0.5], [2 / 29**0.5, 1 / 2**0.5]])
Lambda, u
```

Полученные результаты полностью совпадают с расчетом, выполненным *numpy*:

```{code-cell} python
A = np.array([[4, -5], [2, -3]])
np.linalg.eig(A)
```

Выполним проверку:

```{code-cell} python
Lambda = 2
u = np.array([[5 / 29**0.5], [2 / 29**0.5]])
A.dot(u) - Lambda * u
```

```{code-cell} python
Lambda = -1
u = np.array([[1 / 2**0.5], [1 / 2**0.5]])
A.dot(u) - Lambda * u
```

Также проверим правильность подобранных значений $\mathbf{u}$ без нормирования:

```{code-cell} python
Lambda = 2
u = np.array([[5 / 2], [1]])
A.dot(u) - Lambda * u
```

```{code-cell} python
Lambda = -1
u = np.array([[1], [1]])
A.dot(u) - Lambda * u
```

Кроме того, правильность решения не зависит от выбора значения $u_2$. При $\lambda = 2$ предположим, что $u_2 = 2$. Тогда $u_1 = 5$.

```{code-cell} python
Lambda = 2
u = np.array([[5], [2]])
A.dot(u) - Lambda * u
```

Также необходимо отметить, что найденные собственные векторы матрицы не являются коллинеарными – [определитель](LAB-5-Determinant-InverseMatrix.md#math-lab-detinv) матрицы, составленный из их координат, отличен от нуля:

```{code-cell} python
Lambda, V = np.linalg.eig(A)
np.linalg.det(V)
```

```{admonition} Определение
:class: tip
Два неколлинеарных вектора образуют ***базис***. Под базисом понимается совокупность неколлинеарных векторов в векторном пространстве, взятых в определенном порядке, при этом любой вектор в этом пространстве может быть выражен через линейную комбинацию базисных векторов.
```
