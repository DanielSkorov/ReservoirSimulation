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

(math-lab-vectoroperations)=
# Операции с векторами
Операции с векторами представляют собой комплекс алгебраических манипуляций, направленных на преобразование векторов.

(math-lab-vectoroperations-addition)=
## Сложение векторов

```{admonition} Определение
:class: tip
Под ***сложением*** векторов понимают вычисление элементов нового вектора $\mathbf{c}$, удовлетворяющих условию:

$$ \mathbf{c} = \mathbf{a} + \mathbf{b}. $$

```

При сложении векторов чаще всего оперируют правилом треугольника.

Рассмотрим пример:

$$ \left\{ \begin{array} \\ \mathbf{a} + \mathbf{b} = \mathbf{c} \\ \mathbf{a} = (2, 1) \\ \mathbf{b} = (1, 3) \end{array} \right. $$

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = [0., 2., 0.]
y0 = [0., 1., 0.]

U = [[2., 1., 3.]]
V = [[1., 3., 4.]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'])

ax.text(1., 0., r'$\mathbf{a}$')
ax.text(2.5, 2., r'$\mathbf{b}$')
ax.text(1., 2., r'$\mathbf{c}$')

ax.set_xlim(-1., 5.)
ax.set_ylim(-1., 5.)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

$$ \mathbf{c} = \left( 3, \, 4 \right). $$

Таким образом, чтобы найти координаты вектора, равного сумме других векторов, необходимо поэлементно сложить составляющие векторов. Это можно сделать с использованием *[numpy](https://numpy.org/)*:

```{code-cell} python
a = np.array([2, 1])
b = np.array([1, 3])
c = a + b
c
```

(math-lab-vectoroperations-length)=
## Нахождение длины вектора

```{admonition} Определение
:class: tip
Под ***длиной вектора*** $\mathbf{a}$ понимается его числовое значение, равное расстоянию между его началом и концом. Длина вектора обозначается следующим образом: $\lVert \mathbf{a} \rVert_2$. Математически длина вектора вычисляется по следующему выражению:

$$ \lVert \mathbf{a} \rVert_2 =\sqrt{\sum_{i=1}^{n} a_i^2}, $$

где $n$ – количество элементов в векторе (размерность векторного пространства).
```

Например, найдем длину вектора $\mathbf{a} = \left( 3, \, 4 \right)$.

```{code-cell} python
a = np.array([3, 4])
np.sqrt(a[0]**2 + a[1]**2)
```

Эту же операцию можно осуществить с использованием специального метода библиотеки *[numpy](https://numpy.org/)*.

```{code-cell} python
np.linalg.norm([3, 4])
```

```{admonition} Определение
:class: tip
***Нулевым вектором*** называется вектор, длина которого равна нулю.
```

```{code-cell} python
a = np.array([0, 0])
np.linalg.norm(a)
```

(math-lab-vectoroperations-collinearity)=
## Коллинеарные векторы

```{admonition} Определение
:class: tip
Два вектора называются ***коллинеарными***, если они лежат на одной прямой или на параллельных прямых.
```

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = [0., 2., 4., 0., 5.]
y0 = [0., 4., 3., 4., 7.]

U = [[1., 2., -1., 1., -1.]]
V = [[2., 4., -2., 3., -3.]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['g', 'g', 'g', 'r', 'r'])

ax.text(0., 1., r'$\mathbf{a}$')
ax.text(3., 2., r'$\mathbf{b}$')
ax.text(3., 7., r'$\mathbf{c}$')
ax.text(0., 6., r'$\mathbf{d}$')
ax.text(4., 5., r'$\mathbf{e}$')

ax.set_xlim(-1., 6.)
ax.set_ylim(-1., 8.)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

В данном случае коллинеарными являются векторы: $\mathbf{a} \; || \; \mathbf{b} \; || \; \mathbf{c}$ и $\mathbf{d} \; || \; \mathbf{e}$. При этом векторы $\mathbf{a} \uparrow \uparrow \mathbf{c}$ – сонаправлены, а $\mathbf{d} \uparrow \downarrow \mathbf{e}$ – противоположно направлены. Если два вектора коллинеарны, то они являются ***линейно зависимыми***.

(math-lab-vectoroperations-dotnumber)=
## Произведение вектора на число

```{admonition} Определение
:class: tip
***Произведением ненулевого вектора $\mathbf{a}$ на число $\lambda$*** является такой вектор $\mathbf{b}$, длина которого равна $\left| \lambda \right| \cdot \lVert \mathbf{a} \rVert_2$, причем векторы $\mathbf{a}$ и $\mathbf{b}$ сонаправлены, если $\lambda>0$, и противоположно направлены, если $\lambda<0$.
```

```{admonition} NB
:class: note
В дальнейшем в рамках данного пособия следующие обозначения произведения вектора на число эквивалентны:

$$ \lambda \cdot \mathbf{a} = \lambda \mathbf{a}. $$

```

Рассмотрим пример. Допустим, имеется вектор $\mathbf{a} = \left( 3, \, 4 \right)$. Необходимо найти векторы $\mathbf{b} = 2 \cdot \mathbf{a}$ и $\mathbf{c} = -1 \cdot \mathbf{a}$.

```{code-cell} python
a = np.array([3, 4])
b = 2 * a
c = -1 * a
b, c
```

Геометрически произведение вектора на скаляр можно отобразить следующим образом.

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0., 0., 0.]

U = [[3., 6., -3.]]
V = [[4., 8., -4.]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'g', 'r'], linewidths=[3, 0.5, 0.5], edgecolors=['k', 'g', 'r'])

ax.text(2., 1., r'$\mathbf{a}$', c='k')
ax.text(4., 6., r'$\mathbf{b}$', c='g')
ax.text(-2., -2., r'$\mathbf{c}$', c='r')

ax.set_xlim(-4., 7.)
ax.set_ylim(-5., 9.)
ax.set_xticks(np.linspace(-4., 7., 12))
ax.set_yticks(np.linspace(-5., 9., 15))
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

(math-lab-vectoroperations-innerproduct)=
## Скалярное произведение векторов

```{admonition} Определение
:class: tip
***Скалярным произведением двух векторов*** $\mathbf{a}$ и $\mathbf{b}$ называется число, равное произведению длин этих векторов на косинус угла между ними:

$$ \mathbf{a}^\top \mathbf{b} = \lVert \mathbf{a} \rVert_2 \cdot \lVert \mathbf{b} \rVert_2 \cdot \cos \alpha. $$

```

Необходимо отметить, что скалярное произведение (*dot product* или *inner product*) может обозначаться несколькими способами:
* с использованием символа $\cdot$, например, следующим образом:

$$ \mathbf{a} \cdot \mathbf{b},$$

* с использованием $\langle$ и $\rangle$:

$$ \langle \mathbf{a} , \, \mathbf{b} \rangle,$$

* с использованием транспонирования первого вектора (преобразования его из вектора-столбца в вектор-строку), если оба вектора являются векторами-столбцами:

$$ \mathbf{a}^\top \mathbf{b},$$

* с использованием поэлементного обозначения:

$$ \sum_{i} a_i b_i, $$

где $a_i$ и $b_i$ – $i$-ые элементы векторов $\mathbf{a}$ и $\mathbf{b}$ соответственно.

```{admonition} NB
:class: note
В дальнейшем в рамках данного пособия для обозначения скалярного произведения векторов будут использоваться третий $\mathbf{a}^\top \mathbf{b}$ и четвертый $\sum_{i} a_i b_i$ способы, а выражение $\mathbf{a} \cdot \mathbf{b}$ или $\mathbf{a} \mathbf{b}$ будет использоваться для *поэлементного произведения двух векторов*, принадлежащих пространству с одинаковой размерностью, в результате которого получается вектор:

$$ \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \mathbf{b} = \mathbf{c} \, : \, c_i = a_i b_i, \, i = 1 \, \ldots \, n .$$

```

Рассмотрим следущий пример: необходимо найти скалярное произведение двух векторов $\mathbf{a} = \left( 1, \, 2 \right)$ и $\mathbf{b} = \left( 4, \, 2 \right)$.

```{code-cell} python
a = np.array([1, 2])
b = np.array([4, 2])
mod_a = np.linalg.norm(a)
mod_b = np.linalg.norm(b)
mod_a, mod_b
```

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = [0., 0., 1., 0., 4.]
y0 = [0., 0., 2., 0., 2.]

U = [[1., 4., 3., 3., -1.]]
V = [[2., 2., 0., 0., -2.]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'g', 'r', 'r', 'k'])

ax.text(0.2, 1., r'$\mathbf{a}$', c='k')
ax.text(2., 0.5, r'$\mathbf{b}$', c='g')
ax.text(2., 2.2, r'$\mathbf{c}$', c='r')
ax.text(1., -0.5, r'$\mathbf{c}$', c='r')
ax.text(3.7, 1., r'-$\mathbf{a}$', c='k')

ax.set_xlim(-1., 5.)
ax.set_ylim(-1., 5.)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

```{code-cell} python
c = b - a
mod_c = np.linalg.norm(c)
c, mod_c
```

Угол $\alpha$ между векторами $\mathbf{a}$ и $\mathbf{b}$ можно определить по теореме косинусов:

$$ \lVert \mathbf{c} \rVert_2^2 = \lVert \mathbf{a} \rVert_2^2 + \lVert \mathbf{b} \rVert_2^2 - 2 \cdot \lVert \mathbf{a} \rVert_2 \cdot \lVert \mathbf{b} \rVert_2 \cdot \cos \alpha. $$

```{code-cell} python
cos_alpha = (mod_a**2 + mod_b**2 - mod_c**2) / (2. * mod_a * mod_b)
cos_alpha
```

Тогда скалярное произведение векторов $\mathbf{a}$ и $\mathbf{b}$:

```{code-cell} python
mod_a * mod_b * cos_alpha
```

Однако, зная координаты векторов, скалярное произведение можно найти путем сложения попарных произведений соответствующих координат:

$$ \mathbf{a}^\top \mathbf{b} = a_1 \cdot b_1 + a_2 \cdot b_2 + \ldots + a_n \cdot b_n. $$

```{code-cell} python
a.dot(b)
```

Пусть вектор $\mathbf{v}$ имеет координаты $\left( 4, \, 2 \right)$. Тогда скалярные произведения вектора $\mathbf{v}$ и векторов базиса $\left( \mathbf{i}, \, \mathbf{j} \right)$:

```{code-cell} python
v = np.array([4, 2])
i = np.array([1, 0])
j = np.array([0, 1])
v.dot(i), v.dot(j)
```

Таким образом, скалярное произведение вектора и единичного вектора, сонаправленного с определенной осью, дает значение координаты вектора на данной оси, иными словами, проекцию вектора на ось.

(math-lab-vectoroperations-outerproduct)=
## Тензорное произведение векторов

Существует также и тензорное произведение (*outer product*) векторов для обозначения которого используется символ $\otimes$. В результате тензорного произведения увеличивается размерность, то есть результатом тензорного произведения двух векторов является [матрица](LAB-4-Matrices.md):

$$ \forall ~ \mathbf{a} \in {\rm I\!R}^{n}, \, \mathbf{b} \in {\rm I\!R}^{m} ~ \exists ~ \mathbf{a} \otimes \mathbf{b} = \mathbf{M} ~ : ~ \mathbf{M} \in {\rm I\!R}^{n \times m}.  $$

Также необходимо отметить, что тензорное произведение двух векторов-стобцов может обозначаться следующим образом:

$$ \mathbf{a} \mathbf{b}^\top. $$

С использованием *[numpy](https://numpy.org/)* тензорное произведение векторов может быть выполнено следующим образом:

```{code-cell} python
a = np.array([0, 1, 2, 3])
b = np.array([1, 2, 3])
np.outer(a, b)
```

Другим способом выполнения тензорного произведения является *[broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)*:

```{code-cell} python
a[:, np.newaxis] * b
```

