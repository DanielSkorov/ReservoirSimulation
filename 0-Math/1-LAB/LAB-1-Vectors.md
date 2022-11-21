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

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': False})

%matplotlib widget
```

+++

<a id='math-lab-vector'></a>
# Вектор. Координаты вектора
В рамках линейной алгебры под ***вектором*** понимается упорядоченный набор чисел:

$$ v_1, v_2, v_3, \ldots, v_n, $$

записанных в виде строки или столбца. Иными словами,

```{prf:определение}
:nonumber:
***Вектор*** – это элемент [векторного простраства](https://en.wikipedia.org/wiki/Vector_space) (со свойственными ему аксиомами), который в частности может представлять собой упорядоченную строку или столбец действительных чисел.
```

Так, например, двумерный вектор:

$$ \vec{v}=(v_1, v_2) $$

понимается, как упорядоченная пара чисел, которую можно интерпретировать, как координаты геометрического вектора:

$$ \vec{v}=v_1 \cdot \vec{i}+v_2 \cdot \vec{j}, $$

где $\vec{i}$ и $\vec{j}$ – единичные векторы базиса $(\vec{i}, \vec{j})$.

+++

Рассмотрим вектор:

$$ \vec{v}=(3, 4). $$

Его можно разложить на единичные векторы базиса $(\vec{i}, \vec{j})$ следующим образом:

$$ \vec{v}=3 \cdot \vec{i}+4 \cdot \vec{j}. $$

```{code-cell} python
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0, 0, 0] # Начальные координаты векторов по X и Y

U = [[1, 0, 3]] # Количество i-векторов
V = [[0, 1, 4]] # Количество j-векторов

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'])

ax.text(1, 0, r'$\overrightarrow{i}$')
ax.text(0, 1, r'$\overrightarrow{j}$')
ax.text(3, 4, r'$\overrightarrow{v}$')

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

