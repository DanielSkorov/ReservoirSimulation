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

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': False})

%matplotlib widget

import numpy as np
```

+++

<a id='math-lab-linear_transform'></a>
# Линейные преобразования

+++

```{prf:определение}
:nonumber:
Пусть имеется множество [векторов](./LAB-1-Vectors.html#math-lab-vector), для каждого из которых определены операции [сложения и умножения на скаляр](./LAB-2-VectorOperations.html#math-lab-vector_operations), при этом, если результаты данных операций также принадлежат данному множеству, и для каждого элемента этого множества выполняются [аксиомы векторного пространства](https://en.wikipedia.org/wiki/Vector_space), то такое множество называется ***векторным (или линейным) пространством***.
```

+++

```{prf:определение}
:nonumber:
Если в некотором линейном пространстве $V$ каждому вектору $\vec{v}$ по некоторому правилу $T$ поставлен в соответствие вектор $\vec{u}$ этого же пространства, то говорят, что в данном пространстве задана ***функция линейного преобразования*** (*linear transformation*).
+++
$$ \vec{u} = T \left(\vec{v} \right). $$
```
+++

При этом, для функции линейного преобразования должны выполняться ***свойства линейности***:

$$ T\left(a+b\right)=T\left(a\right)+T\left(b\right) ;\\ T\left(\lambda a\right) = \lambda T\left(a\right). $$

+++

Рассмотрим пример. Пусть функция преобразования определена так, что:

+++

$$ T\left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1 + x_2 \\ 3x_1 \end{bmatrix}. $$

+++

Рассмотрим два вектора $\vec{a}=\begin{bmatrix} a_1 \\ a_2 \end{bmatrix}$ и $\vec{b}=\begin{bmatrix} b_1 \\ b_2 \end{bmatrix}.$ Их сумма:

+++

$$ \vec{a} + \vec{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}. $$

+++

Тогда:

+++

$$ T\left(\vec{a} + \vec{b}\right)=T\left( \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix} \right) = \begin{bmatrix} a_1 + b_1 + a_2 + b_2 \\ 3a_1 + 3b_2 \end{bmatrix}.$$

+++

Линейное преобразование вектора $\vec{a}$:

+++

$$ T \left(\vec{a} \right)=T\left( \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \right) = \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix}. $$

+++

Линейное преобразование вектора $\vec{b}$:

+++

$$ T\left(\vec{b}\right)=T\left( \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} \right) = \begin{bmatrix} b_1 + b_2 \\ 3b_1 \end{bmatrix}. \\ T(\vec{a}) + T(\vec{b}) = \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix} + \begin{bmatrix} b_1 + b_2 \\ 3b_1 \end{bmatrix} = \begin{bmatrix} a_1 + a_2 + b_1 + b_2 \\ 3a_1 + 3b_1 \end{bmatrix}. $$

+++

Из этого следует, что

+++

$$ T \left(\vec{a} + \vec{b} \right)=T \left(\vec{a} \right) + T \left(\vec{b} \right). $$

+++

Докажем для данного примера и второе свойство линейности:

+++

$$ T \left(\lambda \vec{a} \right) = T \left(\lambda \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}\right) = T \left(\begin{bmatrix} \lambda a_1 \\ \lambda a_2 \end{bmatrix}\right) = \begin{bmatrix} \lambda a_1 + \lambda a_2 \\ 3 \lambda a_1 \end{bmatrix} \\ \lambda T(\vec{a}) = \lambda T \left(\begin{bmatrix} a_1 \\ a_2 \end{bmatrix}\right) = \lambda \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix} = \begin{bmatrix} \lambda a_1 + \lambda a_2 \\ 3 \lambda a_1 \end{bmatrix}. $$

+++

Из этого следует, что

+++

$$ T \left(\lambda \vec{a} \right) = \lambda T \left(\vec{a} \right). $$

+++

Таким образом, преобразование $T\left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1 + x_2 \\ 3x_1 \end{bmatrix}$ является линейным.

+++

Любое линейное преобразование вектора можно представить в виде произведения матрицы на данный вектор. Пусть имеется вектор $\vec{x}$. Данный вектор можно разложить на сумму:

+++

$$ \vec{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_n \end{bmatrix} = x_1 \cdot \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} + x_2 \cdot \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} + x_3 \cdot \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix} + \ldots + x_n \cdot \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix} = x_1 \cdot \vec{e_1} + x_2 \cdot \vec{e_2} + \ldots + x_n \cdot \vec{e_n}, $$

+++

где $\vec{e_1}$, $\vec{e_2}$, $\ldots$, $\vec{e_n}$ – вертикальные векторы единичной квадратной матрицы $n \times n$ $I_n$:

+++

$$ I_n=\begin{bmatrix} 1 & 0 & 0 & \ldots & 0 \\ 0 & 1 & 0 & \ldots & 0 \\ 0 & 0 & 1 & \ldots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & 1 \end{bmatrix}. $$

+++

Тогда линейное преобразование вектора $\vec{x}$:

+++

$$ \begin{align}
T \left( \vec{x} \right)
&= T \left( x_1 \cdot \vec{e_1} + x_2 \cdot \vec{e_2} + \ldots + x_n \cdot \vec{e_n} \right) \\
&= T \left( x_1 \cdot \vec{e_1} \right) + T \left( x_2 \cdot \vec{e_2} \right) + \ldots + T \left( x_n \cdot \vec{e_n} \right) \\
&= x_1 \cdot T \left( \vec{e_1} \right) + x_2 \cdot T \left( \vec{e_2} \right) + \ldots + x_n \cdot T \left( \vec{e_n} \right).
\end{align}$$

+++

Здесь последовательно были применены свойства линейности линейных преобразований. Последнее выражение может быть преобразовано следующим образом:

+++

$$ T \left( \vec{x} \right) = x_1 \cdot T \left( \vec{e_1} \right) + x_2 \cdot T \left( \vec{e_2} \right) + \ldots + x_n \cdot T \left( \vec{e_n} \right) = \begin{bmatrix} T \left( \vec{e_1} \right) & T \left( \vec{e_2} \right) & \ldots & T \left( \vec{e_n} \right) \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}. $$

+++

В свою очередь, $T(\vec{e_n})$ – вектор. Таким образом, линейное преобразование любого вектора может быть представлено в виде произведения матрицы на этот же вектор:

+++

$$ T(\vec{x})= A \cdot \vec{x}. $$

+++

Такая матрица $A$ называется матрицей преобразования по отношению к рассматриваемому (*стандартному*) базису.

+++

Однако один и тот же вектор может быть представлен в различных базисах. Пусть имеется вектор $\vec{x}$, координаты которого выражены через базис $B$. Это может быть записано следующим образом:

+++

$$ \begin{bmatrix} \vec{x} \end{bmatrix}_B. $$

+++

Тогда вектор $T(\vec{x})$ также выражается через координаты базиса $B$:

+++

$$ \begin{bmatrix} T(\vec{x}) \end{bmatrix}_B. $$

+++

При этом, линейное преобразование $T$ по-прежнему устанавливает связь между этими двумя векторами, поэтому можно записать:

+++

$$ \begin{bmatrix} T(\vec{x}) \end{bmatrix}_B = D \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B. $$

+++

Такая матрица $D$ называется матрицей преобразования по отношению к базису $B$. Известно, что любой базис определяется через $n$ базисных векторов:

+++

$$ B=\begin{Bmatrix} \vec{v_1}, \vec{v_2}, \ldots, \vec{v_n} \end{Bmatrix}. $$

+++

При этом, любой вектор [можно представить](./LAB-1-Vectors.html#math-lab-vector) в виде суммы произведений координат данного вектора и базисных векторов. Если имеется вектор $\begin{bmatrix} \vec{x} \end{bmatrix}_B$, представленный в виде координат $\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}_B$, отнесенных к базису $B$, то:

+++

$$ \begin{bmatrix} \vec{x} \end{bmatrix}_B = x_1 \cdot \vec{v_1} + x_2 \cdot \vec{v_2} + \ldots + x_n \cdot \vec{v_n}. $$

+++

Рассмотрим пример. Пусть базис $B$ задан векторами $\vec{v_1}=\begin{bmatrix} 2 \\ 1 \end{bmatrix}$ и $\vec{v_2}=\begin{bmatrix} 1 \\ 2 \end{bmatrix}$:

+++

$$ B=\begin{Bmatrix} \vec{v_1}, \vec{v_2}\end{Bmatrix}. $$

+++

Рассмотрим вектор:

+++

$$ \vec{x}=3 \cdot \vec{v_1} + 2 \cdot \vec{v_2}=3 \cdot \begin{bmatrix} 2 \\ 1 \end{bmatrix} + 2 \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 8 \\ 7 \end{bmatrix}. $$

+++

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0, 0, 0]

U = [[2, 1, 8]]
V = [[1, 2, 7]]

ax.plot([0, 8], [0, 4], color='k', ls='--', zorder=1)
ax.plot([1, 2, 3], [2, 4, 6], lw=0, marker='o', color='k', ms=4)
ax.plot([0, 4], [0, 8], color='k', ls='--', zorder=1)
ax.plot([2, 4, 6], [1, 2, 3], lw=0, marker='o', color='k', ms=4)
ax.plot([6, 8], [3, 7], color='r', ls='--', zorder=1)
ax.plot([2, 8], [4, 7], color='r', ls='--', zorder=1)
ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'], zorder=2)

ax.text(2, 0.5, '$\overrightarrow{v_1}$')
ax.text(4, 1.5, '$\overrightarrow{2 \cdot v_1}$')
ax.text(6, 2.5, '$\overrightarrow{3 \cdot v_1}$')
ax.text(0.5, 2, '$\overrightarrow{v_2}$')
ax.text(1, 4, '$\overrightarrow{2 \cdot v_2}$')
ax.text(2, 6, '$\overrightarrow{3 \cdot v_2}$')
ax.text(8, 7, '$\overrightarrow{x}$')

ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_axisbelow(True)
ax.grid()
ax.set_xticks(range(0, 10, 1))
ax.set_yticks(range(0, 10, 1))
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

+++

Тогда вектор $\vec{x}$ относительно базиса $B$:

+++

$$ \begin{bmatrix} \vec{x} \end{bmatrix}_B = \begin{bmatrix} 3 \\ 2 \end{bmatrix}_B. $$

+++

Таким образом, вектор $\vec{x}$ в стандартном базисе может быть выражен в виде произведения матрицы (назовем ее $C^{-1}$), составленной из базисных векторов другого базиса $B$, на данный вектор, представленный в базисе $B$:

+++

$$ \vec{x} = \begin{bmatrix} \vec{v_1} & \vec{v_2} & \ldots & \vec{v_n} \end{bmatrix} \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B = C^{-1} \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B. $$

+++

```{code-cell} ipython3
x_B = np.array([[3], [2]])
C_inv = np.array([[2, 1], [1, 2]])
x = C_inv.dot(x_B)
x
```

+++

И, наоборот, если у нас есть координаты вектора $\vec{x}$ в стандтартном базисе, то определить его координаты относительно базиса $B$ можно следующим образом:

+++

$$ \begin{bmatrix} \vec{x} \end{bmatrix}_B = C \cdot \vec{x}. $$

+++

```{code-cell} ipython3
np.linalg.inv(C_inv).dot(x)
```

+++

Из этого следует, что любой переход от одного базиса к другому можно записать в виде матричного произведения.

+++

Итак, линейное преобразование $T$ в базисе $B$, устанавливающее связь между вектором $\begin{bmatrix} \vec{x} \end{bmatrix}_B$ и вектором $\begin{bmatrix} T \left( \vec{x} \right) \end{bmatrix}_B$:

+++

$$ \begin{bmatrix} T \left( \vec{x} \right) \end{bmatrix}_B = D \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B. $$

+++

Однако линейное преобразование $T(\vec{x})$ может быть выражено в виде матричного произведения:

+++

$$ T \left( \vec{x} \right) =A \cdot \vec{x}. $$

+++

Тогда:

+++

$$ \begin{bmatrix} T \left( \vec{x} \right) \end{bmatrix}_B = D \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B = \begin{bmatrix} A \cdot \vec{x} \end{bmatrix}_B. $$

+++

В свою очередь, переход от базиса $B$ к стандартному может быть записан в виде матричного произведения. Поэтому:

+++

$$ \begin{bmatrix} A \cdot \vec{x} \end{bmatrix}_B = C \cdot A \cdot \vec{x}. $$

+++

Подставляя это равенство в выражение выше, получим:

+++

$$ \begin{bmatrix} T \left( \vec{x} \right) \end{bmatrix}_B = D \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B = \begin{bmatrix} A \cdot \vec{x} \end{bmatrix}_B = C \cdot A \cdot \vec{x}. $$

+++

В свою очередь, вектор $\vec{x}$ в базисе $B$ может быть записан следующим образом:

+++

$$ \vec{x} = C^{-1} \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B. $$

+++

Тогда:

+++

$$ \begin{bmatrix} T \left( \vec{x} \right) \end{bmatrix}_B = D \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B = \begin{bmatrix} A \cdot \vec{x} \end{bmatrix}_B = C \cdot A \cdot \vec{x} = C \cdot A \cdot C^{-1} \cdot \begin{bmatrix} \vec{x} \end{bmatrix}_B. $$

+++

Итак, матрица $D$ линейного преобразования $T$ по отношению к базису $B$:

+++

$$ D = C \cdot A \cdot C^{-1}, $$

+++

где: $C$ – матрица перехода от стандартного базиса к базису $B$, $A$ – матрица линейного преобразования вектора $\vec{x}$ по отношению к стандартному базису.

+++

Пусть матрица $C^{-1}$ составлена из координат собственных векторов $\vec{v_1}$, $\vec{v_2}$, $\ldots$ матрицы $A$:

+++

$$ C^{-1} = \begin{bmatrix} \vert & \vert &  \\ \vec{v_1} & \vec{v_2} & \ldots \\ \vert & \vert &  \end{bmatrix}. $$

+++

Умножим левую и правую части равенства $D=C \cdot A \cdot C^{-1}$ на единичный вектор $\vec{e_i}$, сонаправленный с собственным вектором $\vec{v_i}$:

+++

$$ D\cdot \vec{e_i} = C \cdot A \cdot C^{-1} \cdot \vec{e_i}. $$

+++

Поскольку матрица $D$ задана относительно базиса $B$, определенного собственными векторами матрицы $A$, то результатом произведения матрицы $D$ на единичный вектор $\vec{e_i}$ будет $i$-ый столбец матрицы. Например:

+++

```{code-cell} ipython3
M = np.array([[2, 3, 1], [4, 5, 2], [8, 1, 3]])
M.dot([[1], [0], [0]])
```

+++

Произведение матрицы $C^{-1}$ на единичный вектор $\vec{e_i}$ даст $i$-ый собственный вектор:

+++

$$ C^{-1} \cdot \vec{e_i} = \vec{v_i}. $$

+++

Тогда:

+++

$$ D\cdot \vec{e_i} = C \cdot A \cdot C^{-1} \cdot \vec{e_i} = C \cdot A \cdot \vec{v_i}. $$

+++

По [определению собственного вектора матрицы](./LAB-7-Eigenvalues-Eigenvectors.html#math-lab-eigen) данное выражение преобразуется в:

+++

$$ D\cdot \vec{e_i} = C \cdot A \cdot C^{-1} \cdot \vec{e_i} = C \cdot A \cdot \vec{v_i}= C \cdot \lambda_i \cdot \vec{v_i}. $$

+++

С учетом того, что $\vec{e_i} = C \cdot \vec{v_i}$, получим:

+++

$$ D\cdot \vec{e_i} = C \cdot A \cdot C^{-1} \cdot \vec{e_i} = C \cdot A \cdot \vec{v_i}= C \cdot \lambda_i \cdot \vec{v_i} = \lambda_i \cdot \vec{e_i}. $$

+++

Таким образом, вертикальный столбец матрицы $D$ представляет собой произведение единичного вектора на скаляр, следовательно, если в качестве базиса выбрать собственные векторы матрицы $A$, то матрица $D$ будет являться диагонализированной матрицей $A$ относительно базиса $B$.

+++

Рассмотрим данное свойство на следующем примере:

+++

$$ A = \begin{bmatrix} -1 & 3 & -1 \\ -3 & 5 & -1 \\ -3 & 3 & 1 \end{bmatrix}. $$

+++

```{code-cell} ipython3
A = np.array([[-1, 3, -1], [-3, 5, -1], [-3, 3, 1]])
Lambda, C_inv = np.linalg.eig(A)
C_inv
```

+++

```{code-cell} ipython3
np.linalg.det(C_inv)
```

+++

Определитель матрицы, составленной из координат собственных векторов, не равен нулю, следовательно, собственные векторы матрицы $A$ образуют базис.

+++

```{code-cell} ipython3
np.linalg.inv(C_inv).dot(A).dot(C_inv)
```
