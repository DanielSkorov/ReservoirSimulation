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
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': False})
%matplotlib widget
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../../SupportCode/')
from Graphics import cube, Arrow3D
```

<a id='geomech-rg-stress'></a>
# Напряжение. Тензор напряжений

+++

```{prf:определение}
:nonumber:
Отношение силы, к площади поверхности, к которой она приложена, называется ***вектором напряжения*** (*traction*):
+++
$$\vec{T} = \frac{\vec{F}}{A}$$
+++
```

+++

Пусть имеется некоторое тело, к которому приложены некоторые внешние силы, находящееся в равновесии. Выделим в данном теле элементарный объем.

```{code-cell} ipython3
:tags: [hide-input]

fig = plt.figure(figsize=(6, 4))
fig.canvas.header_visible = False
ax = plt.gca(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

body = cube(20, 40, 5, scale_mult=0.1, equal_scale=True)

ax.add_collection3d(body.collection())

ax.set_xlim(body.xlim)
ax.set_ylim(body.ylim)
ax.set_zlim(body.zlim)

ax.add_artist(Arrow3D([10, 20], [0, 10], [0, 10], color='r', mutation_scale=10, lw=1))
ax.add_artist(Arrow3D([-10, -20], [0, -20], [0, -10], color='r', mutation_scale=10, lw=1))
ax.add_artist(Arrow3D([0, 0], [-10, -20], [2.5, 20], color='r', mutation_scale=10, lw=1))
ax.add_artist(Arrow3D([-5, 0], [10, 20], [-2.5, -20], color='r', mutation_scale=10, lw=1))

ax.text(20, 10, 10, '$\overrightarrow{F_1}$')
ax.text(-20, -20, -10, '$\overrightarrow{F_2}$')
ax.text(0, -20, 20, '$\overrightarrow{F_3}$')
ax.text(0, 20, -20, '$\overrightarrow{F_n}$')

unit_cube = cube(1, 1, 1, facecolors='g', edgecolors='g')
ax.add_collection3d(unit_cube.collection())

fig.tight_layout()
```

+++

Рассмотрим напряженное состояние данного элементарного объема. Для этого поместим его в базис $(\vec{e_1}, \vec{e_2}, \vec{e_3})$. Действующие на тело внешние силы также будут оказывать воздействие на вырезанный элементарный объем, исходя из условия сплошности изучаемого объекта. Следовательно, к граням элементарного объема будут приложены некоторые векторы напряжения.

+++

```{code-cell} ipython3
:tags: [hide-input]

fig = plt.figure(figsize=(6, 4))
fig.canvas.header_visible = False
ax = plt.gca(projection='3d')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

unit_cube = cube(1, 1, 1, facecolors='g', edgecolors='g', equal_scale=True)
ax.add_collection3d(unit_cube.collection())

ax.set_xlim(unit_cube.xlim)
ax.set_ylim(unit_cube.ylim)
ax.set_zlim(unit_cube.zlim)

ax.add_artist(Arrow3D([0.5, 2], [0, -0.5], [0, -0.5], color='r'))
ax.add_artist(Arrow3D([-0.5, -2], [0, 0.5], [0, 0.5], color='b'))
ax.add_artist(Arrow3D([0, -0.5], [0.5, 2], [0, -1.5], color='r'))
ax.add_artist(Arrow3D([0, 0.5], [-0.5, -2], [0, 1.5], color='b'))
ax.add_artist(Arrow3D([0, 0], [0, -1], [0.5, 2.5], color='r'))
ax.add_artist(Arrow3D([0, 0], [0, 1], [-0.5, -2.5], color='b'))

ax.text(2, -0.5, -0.5, '$\overrightarrow{T_1}$')
ax.text(-0.5, 2, -1.5, '$\overrightarrow{T_2}$')
ax.text(0, -1, 2.5, '$\overrightarrow{T_3}$')
ax.text(-2, 0.5, 0.5, '$\overrightarrow{-T_1}$')
ax.text(0.5, -2, 1.5, '$\overrightarrow{-T_2}$')
ax.text(0, 1, -2.5, '$\overrightarrow{-T_3}$')

ax.add_artist(Arrow3D([0.5, 1.5], [0, 0], [0, 0], color='k', lw=1))
ax.add_artist(Arrow3D([0, 0], [0.5, 1.5], [0, 0], color='k', lw=1))
ax.add_artist(Arrow3D([0, 0], [0, 0], [0.5, 1.5], color='k', lw=1))

ax.text(1.5, 0, 0, '$\overrightarrow{e_1}$')
ax.text(0, 1.5, 0, '$\overrightarrow{e_2}$')
ax.text(0, 0, 1.5, '$\overrightarrow{e_3}$')

ax.view_init(20, 55)

fig.tight_layout()
```

+++

Рассечем данный элементарный объем плоскостью проходящей через его вершины. Поскольку оставшаяся треугольная пирамида находится в равновесии, то к рассекающей плоскости также будет приложен вектор напряжения.

+++

```{code-cell} ipython3
:tags: [hide-input]

fig = plt.figure(figsize=(6, 4))
fig.canvas.header_visible = False
ax = plt.gca(projection='3d')

v = np.array([[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5],  [-0.5, -0.5, 0.5]])

verts = [[v[0],v[1],v[2]], [v[0],v[1],v[3]], [v[1],v[2],v[3]]]
verts_s = [[v[0],v[2],v[3]]]

ax.add_collection3d(Poly3DCollection(verts, facecolors='g', linewidths=1, edgecolors='g', alpha=.25))
ax.add_collection3d(Poly3DCollection(verts_s, facecolors='r', linewidths=1, edgecolors='r', alpha=.25))

ax.set_xlim(unit_cube.xlim)
ax.set_ylim(unit_cube.ylim)
ax.set_zlim(unit_cube.zlim)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

ax.add_artist(Arrow3D([-0.5, -2], [0, 0.5], [0, 0.5], color='b'))
ax.add_artist(Arrow3D([0, 0.5], [-0.5, -2], [0, 1.5], color='b'))
ax.add_artist(Arrow3D([0, 0], [0, 1], [-0.5, -2.5], color='b'))
ax.add_artist(Arrow3D([-0.25, 1], [-0.25, 1], [0, 0.5], color='r'))
ax.add_artist(Arrow3D([-0.25, 0.5], [-0.25, 0.5], [0, 0.75], color='k'))

ax.text(-2, 0.5, 0.5, '$\overrightarrow{-T_1}$')
ax.text(0.5, -2, 1.5, '$\overrightarrow{-T_2}$')
ax.text(0, 1, -2.5, '$\overrightarrow{-T_3}$')
ax.text(1, 1, 0.5, '$\overrightarrow{T}$')
ax.text(0.5, 0.5, 0.5, '$\overrightarrow{n}$')

ax.view_init(20, 110)

fig.tight_layout()
```

+++

Поскольку данный элементарный объем находится в равновесии, запишем для него [первый закон Ньютона](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion):

+++

$$\vec{T_{}} \cdot dA -\vec{T_1} \cdot dA_1 - \vec{T_2} \cdot dA_2 - \vec{T_3} \cdot dA_3 = 0,$$

+++

где $dA, dA_1, dA_2, dA_3$ – площади элементарного объема, к которому приложены соотвествующие векторы напряжения. Известно, что площадь $dA_1$ выражается через площадь $dA$ следующим образом:

+++

$$dA_1 = dA \cdot \cos{\alpha},$$

+++

где $\alpha$ – угол между плоскостями $dA$ и $dA_1$. Угол между двумя плоскостями равен углу между нормалями к данным плоскостям: $\vec{n}$ и $\vec{e_1}$. Поскольку вектор $\vec{e_1}$ является базисным, то косинус угла между нормалью $\vec{n}$ (нормаль является единичным вектором) и вектором $\vec{e_1}$ является [направляющим косинусом](../../0-Math/1-LAB/LAB-3-RotationAngles.html#math-lab-rotation_angles) нормали $\vec{n}$ и равен ее первой координате. Аналогично – для площадок $ dA_2$ и $dA_3$. Тогда координаты нормали $\vec{n}$:

+++

$$\vec{n} = \begin{bmatrix} \cos{(\vec{n}, \vec{e_1})} \\ \cos{(\vec{n}, \vec{e_2})} \\ \cos{(\vec{n}, \vec{e_2})} \end{bmatrix}$$

+++

Пусть

+++

$$\cos{(\vec{n}, \vec{e_1})} = n_1, \cos{(\vec{n}, \vec{e_2})} = n_2, \cos{(\vec{n}, \vec{e_2})} = n_3$$

+++

Тогда:

+++

$$\vec{T_{}} \cdot dA -\vec{T_1} \cdot dA \cdot n_1 - \vec{T_2} \cdot dA \cdot n_2 - \vec{T_3} \cdot dA \cdot n_3 = 0 \\ \vec{T_{}} = \vec{T_1} \cdot n_1 + \vec{T_2} \cdot n_2 + \vec{T_3} \cdot n_3$$

+++

Векторы напряжения $\vec{T_{}}, \vec{T_1}, \vec{T_2}, \vec{T_3}$ являются векторами, то есть имеют три координаты (проекции на каждую из трех осей). Согласно [правилу сложения векторов](../../0-Math/1-LAB/LAB-2-VectorOperations.html#math-lab-vector_operations), для этих координат можно записать следующие выражения

+++

$$T_{x_1} = n_1 \cdot T_{1_{x_1}} + n_2 \cdot T_{2_{x_1}} + n_3 \cdot T_{3_{x_1}} \\ T_{x_2} = n_1 \cdot T_{1_{x_2}} + n_2 \cdot T_{2_{x_2}} + n_3 \cdot T_{3_{x_2}} \\ T_{x_3} = n_1 \cdot T_{1_{x_3}} + n_2 \cdot T_{2_{x_3}} + n_3 \cdot T_{3_{x_3}}$$

+++

Данное выражение можно записать в виде матричного произведения:

+++

$$\begin{bmatrix} T_{x_1} \\ T_{x_2} \\ T_{x_3} \end{bmatrix} = \begin{bmatrix} T_{1_{x_1}} & T_{2_{x_1}} & T_{3_{x_1}} \\ T_{1_{x_2}} & T_{2_{x_2}} & T_{3_{x_2}} \\ T_{1_{x_3}} & T_{2_{x_3}} & T_{3_{x_3}} \end{bmatrix} \cdot \begin{bmatrix} n_1 \\ n_2 \\ n_3 \end{bmatrix} $$

+++

$$ \vec{T} = S \cdot \vec{n}$$

+++

```{prf:определение}
:nonumber:
Матрица 
+++
$$S = \begin{bmatrix} T_{1_{x_1}} & T_{2_{x_1}} & T_{3_{x_1}} \\ T_{1_{x_2}} & T_{2_{x_2}} & T_{3_{x_2}} \\ T_{1_{x_3}} & T_{2_{x_3}} & T_{3_{x_3}} \end{bmatrix}$$
+++
называется ***тензором напряжений Коши*** и характеризует напряженное состояние в точке. Нормальная составляющая вектора напряжения, действующая на эту площадку, называется ***нормальным напряжением*** и обозначается $\sigma$. Составляющая вектора напряжения, которая лежит в плоскости рассматриваемой площадки, к которой приложен этот вектор напряжения, называется ***касательным (или тангенциальным) напряжением*** и обозначается $\tau$.
+++
```

+++

С учетом этого, тензор напряжений можно записать следующим образом:

+++

$$S = \begin{bmatrix} \sigma_{1} & \tau_{12} & \tau_{13} \\ \tau_{21} & \sigma_{2} & \tau_{23} \\ \tau_{31} & \tau_{32} & \sigma_{3} \end{bmatrix}$$

+++

Рассматривая элементарный объем, тензор напряжений геометрически интерпретируется следующим образом:

+++

```{code-cell} ipython3
:tags: [hide-input]
fig = plt.figure(figsize=(6, 4))
fig.canvas.header_visible = False
ax = plt.gca(projection='3d')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
unit_cube = cube(1, 1, 1, facecolors='goldenrod', edgecolors='goldenrod', scale_mult=0.5)
ax.add_collection3d(unit_cube.collection())
ax.set_xlim(unit_cube.xlim)
ax.set_ylim(unit_cube.ylim)
ax.set_zlim(unit_cube.zlim)
ax.add_artist(Arrow3D([0.5, 1.5], [0, 0], [0, 0], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0, 0], [0.5, 1.5], [0, 0], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0, 0], [0, 0], [0.5, 1.5], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0.5, 0.5], [0, 0.5], [0, 0], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0.5, 0.5], [0, 0.0], [0, 0.5], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0, 0.5], [0.5, 0.5], [0, 0], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0, 0], [0.5, 0.5], [0, 0.5], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0, 0.5], [0, 0], [0.5, 0.5], color='k', lw=1, mutation_scale=4))
ax.add_artist(Arrow3D([0, 0], [0, 0.5], [0.5, 0.5], color='k', lw=1, mutation_scale=4))
ax.text(1.5, 0, 0, '$\overrightarrow{\sigma_1}$')
ax.text(0, 1.5, 0, '$\overrightarrow{\sigma_2}$')
ax.text(0, 0, 1.5, '$\overrightarrow{\sigma_3}$')
ax.text(0.5, 0.3, 0.05, '$\overrightarrow{\\tau_{12}}$')
ax.text(0.5, 0.0, 0.3, '$\overrightarrow{\\tau_{13}}$')
ax.text(0.45, 0.5, -0.2, '$\overrightarrow{\\tau_{21}}$')
ax.text(0, 0.55, 0.35, '$\overrightarrow{\\tau_{23}}$')
ax.text(0.5, 0.0, 0.6, '$\overrightarrow{\\tau_{31}}$')
ax.text(0, 0.5, 0.6, '$\overrightarrow{\\tau_{32}}$')
ax.grid(None)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
ax.view_init(20, 55)
fig.tight_layout()
```
+++