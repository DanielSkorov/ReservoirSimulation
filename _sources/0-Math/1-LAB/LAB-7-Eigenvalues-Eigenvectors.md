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

<a id='math-lab-eigen'></a>
# Собственные векторы и значения матриц

+++

Пусть дана матрица:

+++

$$ A=\begin{bmatrix} -1 & -6 \\ 2 & 6 \end{bmatrix}. $$

+++

Умножим матрицу $A$ на вектор $\vec{u} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}$:

+++

$$ A \cdot \vec{u} = \begin{bmatrix} -1 & -6 \\ 2 & 6 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix}=\begin{bmatrix} -1 \cdot 2 + (-6) \cdot (-1) \\ 2 \cdot 2 + 6 \cdot (-1) \end{bmatrix} = \begin{bmatrix} 4 \\ -2 \end{bmatrix} = 2 \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \lambda \cdot \vec{u}. $$

+++

В результате умножения матрицы $A$ на вектор $\vec{u}$ получился тот же самый вектор $\vec{u}$ с числовым коэффициентом $\lambda=2$:

+++

$$ A \cdot \vec{u} = \lambda \cdot \vec{u}. $$

+++

Такой вектор $\vec{u}$ называется ***собственным вектором*** (*eigenvector*) матрицы $A$, а $\lambda$ – ***собственным значением*** матрицы $A$ (*eigenvalue*).

+++

```{prf:определение}
:nonumber:
Ненулевой вектор $\vec{u}$, который при умножении на некоторую квадратную матрицу $A$ преобразуется в самого же себя с числовым коэффициентом $\lambda$, называется ***собственным вектором*** матрицы $A$, а число $\lambda$ – ***собственным значением*** матрицы $A$.
```

+++

Нахождение собственного вектора и собственного значения некоторой матрицы реализуется с использованием следующего алгоритма. Выражение, полученное из определения собственного вектора, можно преобразовать:

+++

$$ A \cdot \vec{u} - \lambda \cdot \vec{u} = 0. $$

+++

$$ \begin{bmatrix} A - \lambda \cdot E \end{bmatrix} \cdot \vec{u} = 0. $$

+++

Поскольку тривиальное решение данного уравнения не удовлетворяет условию, указанному в определении собственного вектора $(\vec{u} \neq 0)$, то необходимо, чтобы:

+++

$$ det(A - \lambda \cdot E) = 0. $$

+++

Данное уравнение называется характеристическим для матрицы $A$. Отсюда выражается $\lambda$. Последующее определение $\vec{u}$ основано на решении уравнения $\begin{bmatrix} A - \lambda \cdot E \end{bmatrix} \cdot \vec{u} = 0$ относительно $\vec{u}$.

+++

Предположим, что $det(A - \lambda \cdot E) \neq 0$. Следовательно, существует такая обратная матрица, что:

+++

$$ {\begin{bmatrix} A - \lambda \cdot E \end{bmatrix}}^{-1} \cdot \begin{bmatrix} A - \lambda \cdot E \end{bmatrix} \cdot \vec{u} = {\begin{bmatrix} A - \lambda \cdot E \end{bmatrix}}^{-1} \cdot 0, $$

+++

откуда следует:

+++

$$ \vec{u} = 0, $$

+++

что противоречит условию $(\vec{u} \neq 0)$.

+++

Рассмотрим пример. Пусть дана матрица $A$:

+++

$$ A = \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix}. $$

+++

Необходимо найти собственные значения и собственные векторы матрицы $A$. Для начала найдем собственные значения. Запишем уравнение и решим его относительно $\lambda$:

+++

$$ \begin{align}
\begin{vmatrix} \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - \lambda \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\end{vmatrix} &= 0 \\ \begin{vmatrix} 4 - \lambda & -5 \\ 2 & -3 - \lambda \end{vmatrix} &= 0 \\ (4 - \lambda) \cdot (-3 - \lambda) + 10 &= 0 \\ {\lambda}^2 - \lambda - 2 &= 0 \\ {\lambda}_{1,2} &= (-1, 2)
\end{align} $$

+++

При $\lambda = -1$:

+++

$$ \begin{align}
\left( \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - (-1) \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \cdot \vec{u} &= 0 \\ \begin{bmatrix} 5 & -5 \\ 2 & -2 \end{bmatrix} \cdot \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &= 0.
\end{align} $$

+++

Данное выражение может быть преобразовано в систему линейных уравнений и решено с использованием [метода Гаусса](https://en.wikipedia.org/wiki/Gaussian_elimination). Его целью является получение "треугольной матрицы нулей" путем простейших математических преобразований – сложения и умножения. В результате, получится следующее выражение:

+++

$$ \begin{matrix}\\ 1 & -1 & 0\\ 0 & 0 & 0 \end{matrix}. $$

+++

Исходя из второй строчки, $u_2$ может принимать любые значения. Поэтому пусть $u_2 = 1$. Тогда $u_1 = 1$. Отнормируем $u_1$ и $u_2$ на величину $\sqrt{{u_1}^2 + {u_2}^2}$ исключительно для сопоставления с результатом, получаемым в *[numpy](https://numpy.org/)* (далее будет показано, что данная операция, по сути, не играет существенной роли). Тогда $u_1 = u_2 = \frac{1}{\sqrt{2}}$.

+++

При $\lambda = 2$:

+++

$$ \begin{align} \left( \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - 2 \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \cdot u &= 0 \\ \begin{bmatrix} 2 & -5 \\ 2 & -5 \end{bmatrix} \cdot \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &= 0 \end{align} $$

+++

После применения метода Гаусса получим:

+++

$$ \begin{matrix}\\ 2 & -5 & 0\\ 0 & 0 & 0 \end{matrix}. $$

+++

Исходя из второй строчки, $u_2$ может принимать любые значения. Поэтому пусть $u_2 = 1$. Тогда $u_1 = \frac{5}{2}$. Аналогично отнормируем $u_1$ и $u_2$ на величину $\sqrt{{u_1}^2 + {u_2}^2}$. Тогда $u_1 = \frac{5}{\sqrt{29}}$ и $u_2 = \frac{2}{\sqrt{29}}$.

+++

```{code-cell} ipython3
Lambda = np.array([2, -1])
u = np.array([[5 / 29**0.5, 1 / 2**0.5], [2 / 29**0.5, 1 / 2**0.5]])
Lambda, u
```

+++

Полученные результаты полностью совпадают с расчетом, выполненным *numpy*:

+++

```{code-cell} ipython3
A = np.array([[4, -5], [2, -3]])
np.linalg.eig(A)
```

+++

Выполним проверку:

+++

```{code-cell} ipython3
Lambda = 2
u = np.array([[5 / 29**0.5], [2 / 29**0.5]])
A.dot(u) - Lambda * u
```

+++

```{code-cell} ipython3
Lambda = -1
u = np.array([[1 / 2**0.5], [1 / 2**0.5]])
A.dot(u) - Lambda * u
```

+++

Также проверим правильность подобранных значений $\vec{u}$ без нормирования:

+++

```{code-cell} ipython3
Lambda = 2
u = np.array([[5 / 2], [1]])
A.dot(u) - Lambda * u
```

+++

```{code-cell} ipython3
Lambda = -1
u = np.array([[1], [1]])
A.dot(u) - Lambda * u
```

+++

Кроме того, правильность решения не зависит от выбора значения $u_2$. При $\lambda = 2$ предположим, что $u_2 = 2$. Тогда $u_1 = 5$.

+++

```{code-cell} ipython3
Lambda = 2
u = np.array([[5], [2]])
A.dot(u) - Lambda * u
```

+++

Также необходимо отметить, что найденные собственные векторы матрицы не являются коллинеарными – [определитель](./LAB-5-Determinant-InverseMatrix.html#math-lab-det_inv) матрицы, составленный из их координат, отличен от нуля:

+++

```{code-cell} ipython3
Lambda, V = np.linalg.eig(A)
np.linalg.det(V)
```

+++

```{prf:определение}
:nonumber:
Два неколлинеарных вектора образуют ***базис***. Под базисом понимается совокупность неколлинеарных векторов в векторном пространстве, взятых в определенном порядке, при этом любой вектор в этом пространстве может быть выражен через линейную комбинацию базисных векторов.
```
