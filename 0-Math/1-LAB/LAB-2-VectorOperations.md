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

+++ {"tags": []}

<a id='math-lab-vector_operations'></a>
# Операции с векторами
Операции с векторами представляют собой комплекс алгебраических манипуляций, направленных на преобразование векторов.

+++

```{prf:определение}
:nonumber:
Под ***сложением*** векторов понимают вычисление элементов нового вектора $\vec{c}$, удовлетворяющих условию:
+++
$$ \vec{c}=\vec{a}+\vec{b}. $$
+++
```

+++

При сложении векторов чаще всего оперируют правилом треугольника.

+++

Рассмотрим пример:

+++

$$ \left\{ \begin{array}\\ \vec{a}+\vec{b}=\vec{c}\\ \vec{a}=(2, 1)\\ \vec{b}=(1, 3) \end{array} \right. $$

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = [0, 2, 0]
y0 = [0, 1, 0]

U = [[2, 1, 3]]
V = [[1, 3, 4]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'])

ax.text(1, 0, '$\overrightarrow{a}$')
ax.text(2.5, 2, '$\overrightarrow{b}$')
ax.text(1, 2, '$\overrightarrow{c}$')

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

$$ \vec{c}=(3, 4).$$

+++

Таким образом, чтобы найти координаты вектора, равного сумме других векторов, необходимо поэлементно сложить составляющие векторов. Это можно сделать с использованием *[numpy](https://numpy.org/)*:

+++

```{code-cell} ipython3
:tags: []

a = np.array([2, 1])
b = np.array([1, 3])
c = a + b
c
```

+++

```{prf:определение}
:nonumber:
Под ***длиной вектора*** $\vec{a}$ понимается его числовое значение, равное расстоянию между его началом и концом. Длина вектора обозначается следующим образом: $|\vec{a}|$. Математически длина вектора вычисляется по следующему выражению:
+++
$$ |\vec{a}|=\sqrt{\sum_{i=1}^{n} a_i^2}, $$
+++
где $n$ – количество элементов в векторе (размерность векторного пространства).
+++
```

+++

Например, найдем длину вектора $\vec{a}=(3, 4)$.

+++

```{code-cell} ipython3
a = np.array([3, 4])
np.sqrt(a[0]**2 + a[1]**2)
```

+++

Эту же операцию можно осуществить с использованием специального метода библиотеки *[numpy](https://numpy.org/)*.

+++

```{code-cell} ipython3
np.linalg.norm([3, 4])
```

+++

```{prf:определение}
:nonumber:
***Нулевым вектором*** называется вектор, длина которого равна нулю.
```

+++

```{code-cell} ipython3
a = np.array([0, 0])
np.linalg.norm(a)
```

+++

<a id='math-lab-collinearity'></a>

```{prf:определение}
:nonumber:
Два вектора называются ***коллинеарными***, если они лежат на одной прямой или на параллельных прямых.
```

+++

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = [0, 2, 4, 0, 5]
y0 = [0, 4, 3, 4, 7]

U = [[1, 2, -1, 1, -1]]
V = [[2, 4, -2, 3, -3]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['g', 'g', 'g', 'r', 'r'])

ax.text(0, 1, '$\overrightarrow{a}$')
ax.text(3, 2, '$\overrightarrow{b}$')
ax.text(3, 7, '$\overrightarrow{c}$')
ax.text(0, 6, '$\overrightarrow{d}$')
ax.text(4, 5, '$\overrightarrow{e}$')

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 8)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

+++

В данном случае коллинеарными являются векторы: $\vec{a} \; || \; \vec{b} \; || \; \vec{c}$ и $\vec{d} \; || \; \vec{e}$. При этом, векторы $\vec{a}\uparrow\uparrow\vec{c}$ – сонаправлены, а $\vec{d}\uparrow\downarrow\vec{e}$ – противоположно направлены. Если два вектора коллинеарны, то они являются ***линейно зависимыми***.

+++

```{prf:определение}
:nonumber:
***Произведением ненулевого вектора $\vec{a}$ на число $\lambda$*** является такой вектор $\vec{b}$, длина которого равна $|\lambda|\cdot|\vec{a}|$, причем векторы $\vec{a}$ и $\vec{b}$ сонаправлены, если $\lambda>0$, и противоположно направлены, если $\lambda<0$.
```

+++

Рассмотрим пример. Допустим, имеется вектор $\vec{a}=(3, 4)$. Необходимо найти векторы $\vec{b}=2\cdot\vec{a}$ и $\vec{c}=-1\cdot\vec{a}$.

+++

```{code-cell} ipython3
a = np.array([3, 4])
b = 2 * a
c = -1 * a
b, c
```

+++

Геометрически произведение вектора на скаляр можно отобразить следующим образом.

+++

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0, 0, 0]

U = [[3, 6, -3]]
V = [[4, 8, -4]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'g', 'r'], linewidths=[3, 0.5, 0.5],
          edgecolors=['k', 'g', 'r'])

ax.text(2, 1, '$\overrightarrow{a}$', c='k')
ax.text(4, 6, '$\overrightarrow{b}$', c='g')
ax.text(-2, -2, '$\overrightarrow{c}$', c='r')

ax.set_xlim(-4, 7)
ax.set_ylim(-5, 9)
ax.set_xticks(np.linspace(-4, 7, 12))
ax.set_yticks(np.linspace(-5, 9, 15))
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

+++

```{prf:определение}
:nonumber:
***Скалярным произведением двух векторов*** $\vec{a}$ и $\vec{b}$ называется число, равное произведению длин этих векторов на косинус угла между ними:
+++
$$ \vec{a}\cdot\vec{b}=|\vec{a}|\cdot|\vec{b}|\cdot\cos\alpha. $$
+++

```
Рассмотрим следущий пример: необходимо найти скалярное произведение двух векторов $\vec{a}=(1, 2)$ и $\vec{b}=(4, 2)$.

+++

```{code-cell} ipython3
a = np.array([1, 2])
b = np.array([4, 2])
mod_a = np.linalg.norm(a)
mod_b = np.linalg.norm(b)
mod_a, mod_b
```

+++

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = [0, 0, 1, 0, 4]
y0 = [0, 0, 2, 0, 2]

U = [[1, 4, 3, 3, -1]]
V = [[2, 2, 0, 0, -2]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'g', 'r', 'r', 'k'])

ax.text(0.2, 1, '$\overrightarrow{a}$', c='k')
ax.text(2, 0.5, '$\overrightarrow{b}$', c='g')
ax.text(2, 2.2, '$\overrightarrow{c}$', c='r')
ax.text(1, -0.5, '$\overrightarrow{c}$', c='r')
ax.text(3.7, 1, '-$\overrightarrow{a}$', c='k')

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

+++

```{code-cell} ipython3
c = b - a
mod_c = np.linalg.norm(c)
c, mod_c
```

+++

Угол $\alpha$ между векторами $\vec{a}$ и $\vec{b}$ можно определить по теореме косинусов:

+++

$$ |\vec{c}|^2=|\vec{a}|^2+|\vec{b}|^2-2\cdot|\vec{a}|\cdot|\vec{b}|\cdot\cos\alpha. $$

+++

```{code-cell} ipython3
cos_alpha = (mod_a**2 + mod_b**2 - mod_c**2) / (2 * mod_a * mod_b)
cos_alpha
```

+++

Тогда скалярное произведение векторов $\vec{a}$ и $\vec{b}$:

+++

```{code-cell} ipython3
mod_a * mod_b * cos_alpha
```

+++

Однако, зная координаты векторов, скалярное произведение можно найти путем сложения попарных произведений соответствующих координат:

+++

$$ \vec{a}\cdot\vec{b}=a_1 \cdot b_1 + a_2 \cdot b_2 + \ldots + a_n \cdot b_n. $$

+++

```{code-cell} ipython3
np.inner(a, b)
```

+++

Пусть вектор $\vec{v}$ имеет координаты $(4, 2)$. Тогда скалярные произведения вектора $\vec{v}$ и векторов базиса $(\vec{i},\vec{j})$:

+++

```{code-cell} ipython3
v = np.array([4, 2])
i = np.array([1, 0])
j = np.array([0, 1])
np.inner(v, i), np.inner(v, j)
```

+++

Таким образом, скалярное произведение вектора и единичного вектора, сонаправленного с определенной осью, дает значение координаты вектора на данной оси, иными словами, проекцию вектора на ось.
