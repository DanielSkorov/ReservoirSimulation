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
```

(math-lab-vector)=
# Вектор. Координаты вектора
В рамках линейной алгебры под ***вектором*** понимается упорядоченный набор чисел:

$$ \mathbf{v} = \left( v_1, \, v_2, \, v_3, \, \ldots, \, v_n \right), $$

записанных в виде строки или столбца. Иными словами,

```{admonition} Определение
:class: tip
***Вектор*** – это элемент [векторного простраства](https://en.wikipedia.org/wiki/Vector_space) (со свойственными ему аксиомами), который в частности может представлять собой упорядоченную строку или столбец действительных (вещественных) чисел.
```

Для обозначения векторов здесь и далее будут использоваться малые латинские буквы, выделенные жирным, например, $\mathbf{v}$. Например, следующая запись

$$ \exists ~ \mathbf{v} \in {\rm I\!R}^n $$

читается следующим образом: *существует* $\left( \exists \right)$ *вектор* $\left( \mathbf{v} \right)$ *, принадлежащий пространству действительных чисел* $\left( {\rm I\!R} \right)$, *размерностью* $n$. Эта запись эквивалентна представленной выше.

Одной из интерпретаций векторов является геометрическая. Согласно ей, вектор представляет собой направленный отрезок, элементами которого являются координаты в некотором заданном базисе. Так, например, вектор, состоящий из двух действительных чисел:

$$ \mathbf{v} = \left( v_1, \, v_2 \right) $$

понимается, как упорядоченная пара чисел, которую можно интерпретировать, как координаты геометрического вектора:

$$ \mathbf{v} = v_1 \cdot \mathbf{i} + v_2 \cdot \mathbf{j}, $$

где $\mathbf{i}$ и $\mathbf{j}$ – единичные векторы базиса $\left( \mathbf{i}, \, \mathbf{j} \right)$. Данная запись соответвует разложению вектора по заданному базису. При этом, такое разложение единственно.

Рассмотрим вектор:

$$ \mathbf{v} = \left( 3, \, 4 \right). $$

Его можно разложить на единичные векторы базиса $\left( \mathbf{i}, \, \mathbf{j} \right)$ следующим образом:

$$ \mathbf{v} = 3 \cdot \mathbf{i} + 4 \cdot \mathbf{j}. $$

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0., 0., 0.]

U = [[1., 0., 3.]]
V = [[0., 1., 4.]]

ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'])

ax.text(1., 0., r'$\mathbf{i}$')
ax.text(0., 1., r'$\mathbf{j}$')
ax.text(3., 4., r'$\mathbf{v}$')

ax.set_xlim(-1., 5.)
ax.set_ylim(-1., 5.)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

