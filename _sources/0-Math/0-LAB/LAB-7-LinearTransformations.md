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

%matplotlib inline

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': False})

import numpy as np
```

(math-lab-lineartransform)=
# Линейные преобразования

```{admonition} Определение
:class: tip
Пусть имеется множество [векторов](LAB-1-Vectors.md), для каждого из которых определены операции [сложения и умножения на скаляр](LAB-2-VectorOperations.md#math-lab-vectoroperations), при этом, если результаты данных операций также принадлежат данному множеству, и для каждого элемента этого множества выполняются [аксиомы векторного пространства](https://en.wikipedia.org/wiki/Vector_space), то такое множество называется ***векторным (или линейным) пространством***.
```

```{admonition} Определение
:class: tip
Если в некотором линейном пространстве $\mathbf{V}$ каждому вектору $\mathbf{v}$ по некоторому правилу $\mathbf{T}$ поставлен в соответствие вектор $\mathbf{u}$ этого же пространства, то говорят, что в данном пространстве задана ***функция линейного преобразования*** (*linear transformation*).

$$ \mathbf{u} = \mathbf{T} \left(\mathbf{v} \right). $$

```

При этом, для функции линейного преобразования должны выполняться ***свойства линейности***:

$$ \mathbf{T}\left( a + b \right)=\mathbf{T} \left( a \right) + \mathbf{T} \left( b \right) ;\\ \mathbf{T} \left( \lambda a \right) = \lambda \mathbf{T} \left( a \right). $$

Рассмотрим пример:

```{admonition} Пример
:class: exercise
Пусть функция преобразования определена так, что:

$$ \mathbf{T} \left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1 + x_2 \\ 3x_1 \end{bmatrix}. $$

Необходимо доказать, что данное преобразование является линейным.
```

````{dropdown} Решение
Рассмотрим два вектора $\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}$ и $\mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}.$ Их сумма:

$$ \mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}. $$

Тогда:

$$ \mathbf{T} \left( \mathbf{a} + \mathbf{b} \right) = \mathbf{T} \left( \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix} \right) = \begin{bmatrix} a_1 + b_1 + a_2 + b_2 \\ 3a_1 + 3b_2 \end{bmatrix}.$$

Линейное преобразование вектора $\mathbf{a}$:

$$ \mathbf{T} \left( \mathbf{a} \right)= \mathbf{T} \left( \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \right) = \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix}. $$

Линейное преобразование вектора $\mathbf{b}$:

$$ \mathbf{T} \left( \mathbf{b} \right) = \mathbf{T} \left( \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} \right) = \begin{bmatrix} b_1 + b_2 \\ 3b_1 \end{bmatrix}. \\ \mathbf{T} \left( \mathbf{a} \right) + \mathbf{T} \left( \mathbf{b} \right) = \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix} + \begin{bmatrix} b_1 + b_2 \\ 3b_1 \end{bmatrix} = \begin{bmatrix} a_1 + a_2 + b_1 + b_2 \\ 3a_1 + 3b_1 \end{bmatrix}. $$

Из этого следует, что

$$ \mathbf{T} \left(\mathbf{a} + \mathbf{b} \right) = \mathbf{T} \left( \mathbf{a} \right) + \mathbf{T} \left( \mathbf{b} \right). $$

Докажем для данного примера и второе свойство линейности:

$$ \mathbf{T} \left( \lambda \mathbf{a} \right) = \mathbf{T} \left(\lambda \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}\right) = \mathbf{T} \left( \begin{bmatrix} \lambda a_1 \\ \lambda a_2 \end{bmatrix} \right) = \begin{bmatrix} \lambda a_1 + \lambda a_2 \\ 3 \lambda a_1 \end{bmatrix} \\ \lambda \mathbf{T} \left( \mathbf{a} \right) = \lambda \mathbf{T} \left( \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \right) = \lambda \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix} = \begin{bmatrix} \lambda a_1 + \lambda a_2 \\ 3 \lambda a_1 \end{bmatrix}. $$

Из этого следует, что

$$ \mathbf{T} \left( \lambda \mathbf{a} \right) = \lambda \mathbf{T} \left( \mathbf{a} \right). $$

Таким образом, преобразование $\mathbf{T} \left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1 + x_2 \\ 3x_1 \end{bmatrix}$ является линейным.
````

```{admonition} Теорема
:class: danger
Любое линейное преобразование вектора можно представить в виде произведения матрицы на данный вектор.
```

```{admonition} Доказательство
:class: proof
Пусть имеется вектор $\mathbf{x}$. Данный вектор можно разложить на сумму:

$$ \begin{align}
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_n \end{bmatrix}
&= x_1 \cdot \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} + x_2 \cdot \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} + x_3 \cdot \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix} + \ldots + x_n \cdot \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix} \\
&= x_1 \cdot \mathbf{e_1} + x_2 \cdot \mathbf{e_2} + \ldots + x_n \cdot \mathbf{e_n},
\end{align} $$

где $\mathbf{e_1}$, $\mathbf{e_2}$, $\ldots$, $\mathbf{e_n}$ – вертикальные векторы единичной квадратной матрицы $\mathbf{I_n}$ размерностью $n \times n$:

$$ \mathbf{I_n} = \begin{bmatrix} 1 & 0 & 0 & \ldots & 0 \\ 0 & 1 & 0 & \ldots & 0 \\ 0 & 0 & 1 & \ldots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & 1 \end{bmatrix}. $$

Тогда линейное преобразование вектора $\mathbf{x}$:

$$ \begin{align}
\mathbf{T} \left( \mathbf{x} \right)
&= \mathbf{T} \left( x_1 \cdot \mathbf{e_1} + x_2 \cdot \mathbf{e_2} + \ldots + x_n \cdot \mathbf{e_n} \right) \\
&= \mathbf{T} \left( x_1 \cdot \mathbf{e_1} \right) + \mathbf{T} \left( x_2 \cdot \mathbf{e_2} \right) + \ldots + \mathbf{T} \left( x_n \cdot \mathbf{e_n} \right) \\
&= x_1 \cdot \mathbf{T} \left( \mathbf{e_1} \right) + x_2 \cdot \mathbf{T} \left( \mathbf{e_2} \right) + \ldots + x_n \cdot \mathbf{T} \left( \mathbf{e_n} \right).
\end{align}$$

Здесь последовательно были применены свойства линейности линейных преобразований. Последнее выражение может быть преобразовано следующим образом:

$$ \begin{align}
\mathbf{T} \left( \mathbf{x} \right)
&= x_1 \cdot \mathbf{T} \left( \mathbf{e_1} \right) + x_2 \cdot \mathbf{T} \left( \mathbf{e_2} \right) + \ldots + x_n \cdot \mathbf{T} \left( \mathbf{e_n} \right) \\
&= \begin{bmatrix} | & | & \, & | \\ \mathbf{T} \left( \mathbf{e_1} \right) & \mathbf{T} \left( \mathbf{e_2} \right) & \ldots & \mathbf{T} \left( \mathbf{e_n} \right) \\ | & | & \, & | \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}.
\end{align} $$

В свою очередь, $\mathbf{T} \left( \mathbf{e_n} \right)$ – вектор. Таким образом, линейное преобразование любого вектора может быть представлено в виде произведения матрицы на этот же вектор:

$$ \mathbf{T} \left( \mathbf{x} \right) = \mathbf{A} \mathbf{x}. $$

Такая матрица $\mathbf{A}$ называется матрицей преобразования по отношению к рассматриваемому (*стандартному*) базису.
```

Однако один и тот же вектор может быть представлен в различных базисах. Пусть имеется вектор $\mathbf{x}$, координаты которого выражены через базис $\mathbf{B}$. Это может быть записано следующим образом:

$$ \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

Тогда вектор $\mathbf{T} \left( \mathbf{x} \right)$ также выражается через координаты базиса $\mathbf{B}$:

$$ \begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B}. $$

При этом, линейное преобразование $\mathbf{T}$ по-прежнему устанавливает связь между этими двумя векторами, поэтому можно записать:

$$ \begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B} = \mathbf{D} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

Такая матрица $\mathbf{D}$ называется матрицей преобразования по отношению к базису $\mathbf{B}$. Известно, что любой базис определяется через $n$ базисных векторов:

$$ \mathbf{B} = \begin{Bmatrix} \mathbf{v_1}, \mathbf{v_2}, \ldots, \mathbf{v_n} \end{Bmatrix}. $$

При этом, любой вектор [можно представить](LAB-1-Vectors.md) в виде суммы произведений координат данного вектора и базисных векторов. Если имеется вектор $\begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}$, представленный в виде координат $\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}_\mathbf{B}$, отнесенных к базису $\mathbf{B}$, то:

$$ \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = x_1 \cdot \mathbf{v_1} + x_2 \cdot \mathbf{v_2} + \ldots + x_n \cdot \mathbf{v_n}. $$

Рассмотрим пример. Пусть базис $\mathbf{B}$ задан векторами $\mathbf{v_1} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ и $\mathbf{v_2} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$:

$$ \mathbf{B} = \begin{Bmatrix} \mathbf{v_1}, \mathbf{v_2}\end{Bmatrix}. $$

Рассмотрим вектор:

$$ \mathbf{x} = 3 \cdot \mathbf{v_1} + 2 \cdot \mathbf{v_2} = 3 \cdot \begin{bmatrix} 2 \\ 1 \end{bmatrix} + 2 \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 8 \\ 7 \end{bmatrix}. $$

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0., 0., 0.]

U = [[2., 1., 8.]]
V = [[1., 2., 7.]]

ax.plot([0., 8.], [0., 4.], color='k', ls='--', zorder=1)
ax.plot([1., 2., 3.], [2., 4., 6.], lw=0., marker='o', color='k', ms=4.)
ax.plot([0., 4.], [0., 8.], color='k', ls='--', zorder=1)
ax.plot([2., 4., 6.], [1., 2., 3.], lw=0., marker='o', color='k', ms=4.)
ax.plot([6., 8.], [3., 7.], color='r', ls='--', zorder=1)
ax.plot([2., 8.], [4., 7.], color='r', ls='--', zorder=1)
ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'], zorder=2)

ax.text(2., 0.5, r'$\mathbf{v_1}$')
ax.text(4., 1.5, r'$2 \cdot \mathbf{v_1}$')
ax.text(6., 2.5, r'$3 \cdot \mathbf{v_1}$')
ax.text(0.5, 2, r'$\mathbf{v_2}$')
ax.text(1., 4., r'$2 \cdot \mathbf{v_2}$')
ax.text(2., 6., r'$3 \cdot \mathbf{v_2}$')
ax.text(8., 7., r'$\mathbf{x}$', c='r')

ax.set_xlim(-1., 10.)
ax.set_ylim(-1., 10.)
ax.set_axisbelow(True)
ax.grid()
ax.set_xticks(range(0, 10, 1))
ax.set_yticks(range(0, 10, 1))
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

Тогда вектор $\mathbf{x}$ относительно базиса $\mathbf{B}$:

$$ \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}_\mathbf{B}. $$

Таким образом, вектор $\mathbf{x}$ в стандартном базисе может быть выражен в виде произведения матрицы (назовем ее $\mathbf{C}^{-1}$), составленной из базисных векторов другого базиса $\mathbf{B}$, на данный вектор, представленный в базисе $\mathbf{B}$:

$$ \mathbf{x} = \begin{bmatrix} \mathbf{v_1} & \mathbf{v_2} & \ldots & \mathbf{v_n} \end{bmatrix} \cdot \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = \mathbf{C}^{-1} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

```{code-cell} python
x_B = np.array([[3], [2]])
C_inv = np.array([[2, 1], [1, 2]])
x = C_inv.dot(x_B)
x
```

И, наоборот, если у нас есть координаты вектора $\mathbf{x}$ в стандтартном базисе, то определить его координаты относительно базиса $\mathbf{B}$ можно следующим образом:

$$ \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = \mathbf{C} \mathbf{x}. $$

```{code-cell} python
np.linalg.inv(C_inv).dot(x)
```

Из этого следует, что любой переход от одного базиса к другому можно записать в виде матричного произведения.


Итак, линейное преобразование $\mathbf{T}$ в базисе $\mathbf{B}$, устанавливающее связь между вектором $\begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}$ и вектором $\begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B}$:

$$ \begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B} = \mathbf{D} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

Однако линейное преобразование $\mathbf{T} \left( \mathbf{x} \right)$ может быть выражено в виде матричного произведения:

$$ \mathbf{T} \left( \mathbf{x} \right) = \mathbf{A} \mathbf{x}. $$

Тогда:

$$ \begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B} = \mathbf{D} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = \begin{bmatrix} \mathbf{A} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

В свою очередь, переход от базиса $\mathbf{B}$ к стандартному может быть записан в виде матричного произведения. Поэтому:

$$ \begin{bmatrix} \mathbf{A} \mathbf{x} \end{bmatrix}_\mathbf{B} = \mathbf{C} \mathbf{A} \mathbf{x}. $$

Подставляя это равенство в выражение выше, получим:

$$ \begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B} = \mathbf{D} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = \begin{bmatrix} \mathbf{A} \mathbf{x} \end{bmatrix}_\mathbf{B} = \mathbf{C} \mathbf{A} \mathbf{x}. $$

В свою очередь, вектор $\mathbf{x}$ в базисе $\mathbf{B}$ может быть записан следующим образом:

$$ \mathbf{x} = \mathbf{C}^{-1} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

Тогда:

$$ \begin{bmatrix} \mathbf{T} \left( \mathbf{x} \right) \end{bmatrix}_\mathbf{B} = \mathbf{D} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B} = \begin{bmatrix} \mathbf{A} \mathbf{x} \end{bmatrix}_\mathbf{B} = \mathbf{C} \mathbf{A} \mathbf{x} = \mathbf{C} \mathbf{A}  \mathbf{C}^{-1} \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathbf{B}. $$

Итак, матрица $\mathbf{D}$ линейного преобразования $\mathbf{T}$ по отношению к базису $\mathbf{B}$:

$$ \mathbf{D} = \mathbf{C} \mathbf{A} \mathbf{C}^{-1}, $$

где: $\mathbf{C}$ – матрица перехода от стандартного базиса к базису $\mathbf{B}$, $\mathbf{A}$ – матрица линейного преобразования вектора $\mathbf{x}$ по отношению к стандартному базису.


Пусть матрица $\mathbf{C}^{-1}$ составлена из координат собственных векторов $\mathbf{v_1}$, $\mathbf{v_2}$, $\ldots$ матрицы $\mathbf{A}$:

$$ \mathbf{C}^{-1} = \begin{bmatrix} \vert & \vert &  \\ \mathbf{v_1} & \mathbf{v_2} & \ldots \\ \vert & \vert &  \end{bmatrix}. $$

Умножим левую и правую части равенства $\mathbf{D} = \mathbf{C} \mathbf{A} \mathbf{C}^{-1}$ на единичный вектор $\mathbf{e_i}$, сонаправленный с собственным вектором $\mathbf{v_i}$:

$$ \mathbf{D} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{C}^{-1} \mathbf{e_i}. $$

Поскольку матрица $\mathbf{D}$ задана относительно базиса $\mathbf{B}$, определенного собственными векторами матрицы $\mathbf{A}$, то результатом произведения матрицы $\mathbf{D}$ на единичный вектор $\mathbf{e_i}$ будет $i$-ый столбец матрицы. Например:

```{code-cell} python
M = np.array([[2, 3, 1], [4, 5, 2], [8, 1, 3]])
M.dot([[1], [0], [0]])
```

Произведение матрицы $\mathbf{C}^{-1}$ на единичный вектор $\mathbf{e_i}$ даст $i$-ый собственный вектор:

$$ \mathbf{C}^{-1} \mathbf{e_i} = \mathbf{v_i}. $$

Тогда:

$$ \mathbf{D} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{C}^{-1} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{v_i}. $$

По [определению собственного вектора матрицы](LAB-6-Eigenvalues-Eigenvectors.md#math-lab-eigen) данное выражение преобразуется в:

$$ \mathbf{D} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{C}^{-1} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{v_i}= \mathbf{C} \lambda_i \mathbf{v_i}. $$

С учетом того, что $\mathbf{e_i} = \mathbf{C} \mathbf{v_i}$, получим:

$$ \mathbf{D} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{C}^{-1} \mathbf{e_i} = \mathbf{C} \mathbf{A} \mathbf{v_i} = \mathbf{C} \lambda_i \mathbf{v_i} = \lambda_i \mathbf{e_i}. $$

Таким образом, вертикальный столбец матрицы $\mathbf{D}$ представляет собой произведение единичного вектора на скаляр, следовательно, если в качестве базиса выбрать собственные векторы матрицы $\mathbf{A}$, то матрица $D$ будет являться диагонализированной матрицей $\mathbf{A}$ относительно базиса $\mathbf{B}$.


Рассмотрим данное свойство на следующем примере:

$$ \mathbf{A} = \begin{bmatrix} -1 & 3 & -1 \\ -3 & 5 & -1 \\ -3 & 3 & 1 \end{bmatrix}. $$

```{code-cell} python
A = np.array([[-1, 3, -1], [-3, 5, -1], [-3, 3, 1]])
Lambda, C_inv = np.linalg.eig(A)
C_inv
```

```{code-cell} python
np.linalg.det(C_inv)
```

Определитель матрицы, составленной из координат собственных векторов, не равен нулю, следовательно, собственные векторы матрицы $\mathbf{A}$ образуют базис.

```{code-cell} python
np.linalg.inv(C_inv).dot(A).dot(C_inv)
```
