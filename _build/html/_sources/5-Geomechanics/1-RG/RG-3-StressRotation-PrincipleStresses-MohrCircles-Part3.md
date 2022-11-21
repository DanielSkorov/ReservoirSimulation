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

```{admonition} Внимание!
:class: warning
Данная страница инициализирована в статичном режиме – все графики неинтерактивны. При необходимости Вы можете посмотреть [интерактивную копию](RG-3-StressRotation-PrincipleStresses-MohrCircles-Part3-I.md) данной страницы.
```

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib as mpl
import sympy as smp
from ipywidgets import interact, widgets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Text3D
%matplotlib widget
import sys
sys.path.append('../../SupportCode/')
from Graphics import cube_plane, Arrow3D
```

+++

<a id='geomech-rg-stress_rotation-3'></a>
# Базис тензора напряжений. Главные напряжения. Круги Мора. Часть 3

+++

Другим подходом к изображению напряженного состояния в точке (по аналогии с 2D задачей) является построение кругов Мора. Представим, что имеется элементарный куб, который рассекается некоторой плоскостью. Отсеченная часть отбрасывается, а оставшаяся уравновешивается вектором напряжения, действующим на площадь сечения. Данный вектор напряжения имеет нормальную и касательую составляющие. Поскольку рассечь элементарный объем можно под любыми углами, то совокупность векторов напряжения, действующих на площадь сечения, характеризует напряженное состояние в данной точке. Рассмотрим вывод уравнений, описывающих круги Мора для трехмерного случая.

+++

При рассмотрении тем [собственных векторов матриц](../../0-Math/1-LAB/LAB-7-Eigenvalues-Eigenvectors.md) и их [линейных преобразований](../../0-Math/1-LAB/LAB-8-LinearTransformations.md) было показано, что любая квадратная матрица может быть представлена в диагонализированном виде при ее линейном преобразовании из стандартного базиса в базис, построенный на собственных векторах матрицы. Следовательно, трехмерный тензор напряжений также может быть диагонализирован. Рассмотрим пример. Пусть имеется тензор напряжений:

$$S = \begin{bmatrix}2 & 1 & 0 \\ 1 & 3 & -2 \\ 0 & -2 & 1 \end{bmatrix}$$

Найдем его собственные векторы и преобразуем к диагональному виду:

```{code-cell} python
S = np.array([[2, 1, 0], [1, 3, -2], [0, -2, 1]])
Lambda, C_inv = np.linalg.eig(S)
C = np.linalg.inv(C_inv)
Sp = C.dot(S).dot(C_inv)
Sp
```

Полученные напряжения называются главными и записываются в виде следующей матрицы:

$$S_p = \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{bmatrix}$$

При этом, для главных напряжений выполняется сдедующее неравенство:

$$\sigma_1 > \sigma_2 > \sigma_3$$

```{code-cell} python
s1, s2, s3 = Sp.diagonal()
```

Пусть матрица $C^{-1}$ составлена из координат собственных векторов тензора напряжений, которые называются главными направляющими векторами:

$$C^{-1} = \begin{bmatrix} \vert & \vert &  \\ \vec{v_1} & \vec{v_2} & \ldots \\ \vert & \vert &  \end{bmatrix}$$

```{code-cell} python
v1, v2, v3 = C_inv.T
```

Векторы $\vec{v_1}, \vec{v_2}, \vec{v_3}$ являются единичными взаимоперпендикулярными и образуют новый базис, в котором рассматриваемое напряженное состояние в точке представлено исключительно нормальными напряжениями.

```{code-cell} python
np.inner(v1, v2), np.inner(v2, v3), np.inner(v3, v1)
```

```{code-cell} python
v1[0]**2 + v1[1]**2 + v1[2]**2, v2[0]**2 + v2[1]**2 + v2[2]**2, v3[0]**2 + v3[1]**2 + v3[2]**2
```

Рассмотрим элементарный объем тела относительно базиса, представленного собственными векторами тензора напряжений. На данный элементарный объем действуют только нормальные напряжения. Рассечем данный объем плоскостью, нормаль к которой имеет координаты: 

$$\vec{n} = \begin{bmatrix} n_1 \\ n_2 \\ n_3 \end{bmatrix}.$$

При этом, вектор нормали является единичным вектором, то есть:

$$n_1^2 + n_2^2 + n_3^2 = 1.$$

```{code-cell} python
:tags: [hide-input]

fig = plt.figure(figsize=(8, 4))
fig.canvas.header_visible = False

ax = fig.add_subplot(1, 1, 1, projection='3d')
cube = cube_plane(1, 1, 1, alpha=0.1, linewidths=0.4, equal_scale=True)
n = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
ax.add_collection3d(cube.cube_collection())
ax.add_collection3d(cube.inter_collection(normal=n))
ax.set_xlim(cube.xlim)
ax.set_ylim(cube.ylim)
ax.set_zlim(cube.zlim)
ax.add_artist(Arrow3D(*zip([0, 0, 0], n), color='r', lw=1))
ax.text(*n, '$\\overrightarrow{n}$', c='r')
T = Sp.dot(n.T)
ax.add_artist(Arrow3D(*zip([0, 0, 0], T), color='k', lw=1))
ax.text(*T, '$\\overrightarrow{T}$', c='k')
Tn = T.dot(n.T) * n
ax.add_artist(Arrow3D(*zip([0, 0, 0], Tn), color='g', lw=1))
ax.text(*Tn, '$\\overrightarrow{T_n}$', c='g')
Ts = T - Tn
ax.add_artist(Arrow3D(*zip([0, 0, 0], Ts), color='c', lw=1))
s = Ts / np.linalg.norm(Ts)
ax.text(*Ts, '$\\overrightarrow{T_s}$', c='c')
ax.add_artist(Arrow3D([0.5, 0.5+s1], [0, 0], [0, 0], color='b', lw=1))
ax.add_artist(Arrow3D([0, 0], [0.5, 0.5+s2], [0, 0], color='b', lw=1))
ax.add_artist(Arrow3D([0, 0], [0, 0], [0.5, 0.5+s3], color='b', lw=1))
ax.text(0.5+s1, 0, 0, '$\\sigma_{1}$')
ax.text(0, 0.5+s2, 0, '$\\sigma_{2}$')
ax.text(0, 0, 0.5+s3, '$\\sigma_{3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

fig.tight_layout()
```

Вектор напряжения, действующий на данную площадку:

$$\vec{T} = S_p \cdot \vec{n} = \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{bmatrix} \cdot \begin{bmatrix} n_1 \\ n_2 \\ n_3 \end{bmatrix} = \begin{bmatrix} \sigma_1 \cdot n_1 \\ \sigma_2 \cdot n_2 \\ \sigma_3 \cdot n_3 \end{bmatrix}$$

Нормальная составляющая вектора напряжения (коллинеарная с нормалью):

$$T_n = \vec{T} \cdot \vec{n} = S_p \cdot \vec{n} \cdot \vec{n} = \sigma_1 \cdot n^2_1 + \sigma_2 \cdot n^2_2 + \sigma_3 \cdot n^2_3$$

Скалярный квадрат вектора напряжения:

$$\vec{T}^2 = \vec{T_n}^2 + \vec{T_s}^2 = {\left(\sigma_1 \cdot n_1 \right)}^2 + {\left(\sigma_2 \cdot n_2 \right)}^2 + {\left(\sigma_3 \cdot n_3 \right)}^2$$

Последнее равенство получается следующим образом:

$$\vec{T}^2 = \vec{T} \cdot \vec{T} = \left(\vec{T_n} + \vec{T_s} \right) \cdot \left(\vec{T_n} + \vec{T_s} \right) = \vec{T_n}^2 + 2 \cdot \vec{T_n} \cdot \vec{T_s} + \vec{T_s}^2 = \vec{T_n}^2 + \vec{T_s}^2$$

Таким образом, имеется система уравнений (в матричном виде):

$$\begin{bmatrix} 1 & 1 & 1 \\ \sigma_1 & \sigma_2 & \sigma_3 \\ \sigma_1^2 & \sigma_2^2 & \sigma_3^2 \end{bmatrix} \cdot \begin{bmatrix} n_1^2 \\ n_2^2 \\ n_3^2 \end{bmatrix} = \begin{bmatrix} 1 \\ T_n \\ \vec{T_n}^2 + \vec{T_s}^2 \end{bmatrix}$$

Решением данной системы уравнений относительно $n_1^2, n_2^2, n_3^2$ является:

```{code-cell} python
c1, c2, c3, n1, n2, n3, Tn, Ts = smp.symbols('c1, c2, c3, n1, n2, n3, Tn, Ts')
smp.solve([n1**2 + n2**2 + n3**2 - 1,
           c1 * n1**2 + c2 * n2**2 + c3 * n3**2 - Tn,
           c1**2 * n1**2 + c2**2 * n2**2 + c3**2 * n3**2 - Tn**2 - Ts**2],
          (n1**2, n2**2, n3**2))
```

Перепишем полученное решение следующим образом:

$$n_1^2 = \frac{\vec{T_s}^2 + \left(T_n - \sigma_2 \right) \cdot \left(T_n - \sigma_3 \right)}{\left(\sigma_1 - \sigma_2 \right) \cdot \left(\sigma_1 - \sigma_3 \right)} \geq 0 \\ n_2^2 = \frac{\vec{T_s}^2 + \left(T_n - \sigma_1 \right) \cdot \left(T_n - \sigma_3 \right)}{\left(\sigma_2 - \sigma_1 \right) \cdot \left(\sigma_2 - \sigma_3 \right)} \geq 0 \\ n_3^2 = \frac{\vec{T_s}^2 + \left(T_n - \sigma_1 \right) \cdot \left(T_n - \sigma_2 \right)}{\left(\sigma_3 - \sigma_1 \right) \cdot \left(\sigma_3 - \sigma_2 \right)} \geq 0$$

С учетом соотношения главных напряжений, получим:

$$\vec{T_s}^2 + \left(T_n - \sigma_2 \right) \cdot \left(T_n - \sigma_3 \right) \geq 0 \\ \vec{T_s}^2 + \left(T_n - \sigma_1 \right) \cdot \left(T_n - \sigma_3 \right) \leq 0 \\ \vec{T_s}^2 + \left(T_n - \sigma_1 \right) \cdot \left(T_n - \sigma_2 \right) \geq 0$$

Раскроем скобки в неравенствах и применим следующее преобразование:

$$\begin{alignat}{1}
\left(T_n - \sigma_2 \right) \cdot \left(T_n - \sigma_3 \right)
&= & \; \left(T_n^2 - T_n \cdot \sigma_3 - T_n \cdot \sigma_2 + \sigma_2 \cdot \sigma_3 \right) \\
&= & \; T_n^2 - 2 \cdot \frac{1}{2} \cdot T_n \cdot \left(\sigma_2 + \sigma_3 \right) + \left(\frac{1}{2} \cdot \left(\sigma_2 + \sigma_3 \right) \right)^2 - \left(\frac{1}{2} \cdot \left(\sigma_2 + \sigma_3 \right) \right)^2 \\
&& \; + \sigma_2 \cdot \sigma_3 \\
&= & \; \left(T_n - \frac{1}{2} \cdot \left(\sigma_2 + \sigma_3 \right) \right)^2 - \left(\frac{1}{2} \cdot \left(\sigma_2 + \sigma_3 \right) \right)^2 + \sigma_2 \cdot \sigma_3 \\
&= & \; \left(T_n - \frac{1}{2} \cdot \left(\sigma_2 + \sigma_3 \right) \right)^2 - \left(\frac{1}{2} \cdot \left(\sigma_2 - \sigma_3 \right) \right)^2
\end{alignat} $$

С учетом этого получим:

$$\vec{T_s}^2 + \left(T_n - \frac{1}{2} \cdot \left(\sigma_2 + \sigma_3 \right) \right)^2 \geq \left(\frac{1}{2} \cdot \left(\sigma_2 - \sigma_3 \right) \right)^2 \\ \vec{T_s}^2 + \left(T_n - \frac{1}{2} \cdot \left(\sigma_1 + \sigma_3 \right) \right)^2 \leq \left(\frac{1}{2} \cdot \left(\sigma_1 - \sigma_3 \right) \right)^2 \\ \vec{T_s}^2 + \left(T_n - \frac{1}{2} \cdot \left(\sigma_1 + \sigma_2 \right) \right)^2 \geq \left(\frac{1}{2} \cdot \left(\sigma_1 - \sigma_2 \right) \right)^2$$

Уравнение окружности имеет вид:

$$\left(y - y_0 \right)^2 + \left(x - x_0 \right)^2 = R^2$$

с центром в точке $\left(x_0, y_0 \right)$ и радиусом $R$.

+++

Полученная система неравенств характеризует множество всевозможных точек с координатами $\left(T_n, T_s \right)$. Построим данные окружности для рассматриваемого примера.

```{code-cell} python
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(5, 4.5))
fig.canvas.header_visible = False
ax.add_patch(plt.Circle(((s1 + s2) / 2, 0), (s1 - s2) / 2, fill=False, ec='r', lw=2))
ax.add_patch(plt.Circle(((s2 + s3) / 2, 0), (s2 - s3) / 2, fill=False, ec='b', lw=2))
ax.add_patch(plt.Circle(((s1 + s3) / 2, 0), (s1 - s3) / 2, fill=False, ec='g', lw=2))
ax.set_xlim(s3, s1)
ax.set_ylim(-(s1-s3)/2, (s1-s3)/2)
ax.set_xlabel('$T_n$')
ax.set_ylabel('$T_s$')
ax.grid()
ax.scatter(T.dot(n.T), T.dot(s.T), c='k', alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
```

На рисунке выше точкой обозначены величины нормальной и касательной составляющих вектора напряжения, действующего на площадку, заданную нормалью $\vec{n}$. Ниже представлена интерактивная диаграмма, позволяющая отобразить значения $T_n, T_s$ при любых значениях углов поворота нормали секущей плоскости.

```{code-cell} python
:tags: [hide-input]
f = plt.figure(figsize=(5, 8))
f.canvas.header_visible = False
S = np.array([[2, 1, 0], [1, 3, -2], [0, -2, 1]])
[[s11, s12, s13], [s21, s22, s23], [s31, s32, s33]] = S
ax1 = f.add_subplot(2, 1, 1, projection='3d')
normal = np.array([1, 0, 0])
unit_cube = cube_plane(1, 1, 1, alpha=0.1, linewidths=0.4, equal_scale=True)
ax1.set_xlim(unit_cube.xlim)
ax1.set_ylim(unit_cube.ylim)
ax1.set_zlim(unit_cube.zlim)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.invert_zaxis()
ax1.invert_xaxis()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
s11_arr = Arrow3D([0.5, 0.5+s11], [0, 0], [0, 0], color='k', lw=1, mutation_scale=4)
s22_arr = Arrow3D([0, 0], [0.5, 0.5+s22], [0, 0], color='k', lw=1, mutation_scale=4)
s33_arr = Arrow3D([0, 0], [0, 0], [0.5, 0.5+s33], color='k', lw=1, mutation_scale=4)
s12_arr = Arrow3D([0.5, 0.5], [0, s12], [0, 0], color='k', lw=1, mutation_scale=4)
s23_arr = Arrow3D([0, 0], [0.5, 0.5], [0, s23], color='k', lw=1, mutation_scale=4)
s31_arr = Arrow3D([0, s31], [0, 0], [0.5, 0.5], color='k', lw=1, mutation_scale=4)
ax2 = f.add_subplot(2, 1, 2)
ax2.add_patch(plt.Circle(((s1 + s2) / 2, 0), (s1 - s2) / 2, fill=False, ec='r', lw=2))
ax2.add_patch(plt.Circle(((s2 + s3) / 2, 0), (s2 - s3) / 2, fill=False, ec='b', lw=2))
ax2.add_patch(plt.Circle(((s1 + s3) / 2, 0), (s1 - s3) / 2, fill=False, ec='g', lw=2))
ax2.set_xlim(s3, s1)
ax2.set_ylim(-(s1-s3)/2, (s1-s3)/2)
ax2.set_xlabel('$T_n$')
ax2.set_ylabel('$T_s$')
ax2.grid()
f.tight_layout()
removing2 = [Text3D, Poly3DCollection, Arrow3D, mpl.collections.PathCollection]
@interact(alpha=widgets.IntSlider(min=0, max=360, step=1, value=0), beta=widgets.IntSlider(min=0, max=360, step=1, value=0), gamma=widgets.IntSlider(min=0, max=360, step=1, value=0))
def plane_intersection_3d(alpha, beta, gamma):
    for dax in [ax1, ax2]:
        for child in dax.get_children():
            if type(child) in removing2:
                try:
                    child.remove()
                except:
                    break
    alpha, beta, gamma = alpha * np.pi / 180, beta * np.pi / 180, gamma * np.pi / 180
    for art in [s11_arr, s22_arr, s33_arr, s12_arr, s23_arr, s31_arr]:
        ax1.add_artist(art)
    arrow_n = Arrow3D(*zip([0, 0, 0], normal), color='r', rotation_angles=(alpha, beta, gamma), lw=1)
    ax1.add_artist(arrow_n)
    rot_m = arrow_n.rotation_matrix()
    n = rot_m.dot(normal.T)
    t_vec = S.dot(n)
    ax1.add_artist(Arrow3D(*zip([0, 0, 0], t_vec), color='b', lw=1))
    tn = t_vec.dot(n.T)
    tn_vec = tn * n
    ax1.add_artist(Arrow3D(*zip([0, 0, 0], tn_vec), color='g', lw=1))
    ts_vec = t_vec - tn_vec
    s = ts_vec / np.linalg.norm(ts_vec)
    ts = t_vec.dot(s.T)
    ax1.add_artist(Arrow3D(*zip([0, 0, 0], ts_vec), color='c', lw=1))
    ax1.add_collection3d(unit_cube.cube_collection())
    ax1.add_collection3d(unit_cube.inter_collection(normal=n))
    ax1.text(*n, '$\\overrightarrow{n}$', c='r')
    ax1.text(0.5+s11, 0, 0, '$\\sigma_{11}$')
    ax1.text(0, 0.5+s22, 0, '$\\sigma_{22}$')
    ax1.text(0, 0, 0.5+s33, '$\\sigma_{33}$')
    ax1.text(0.5, s12, 0, '$\\tau_{12}$')
    ax1.text(0, 0.5, s23, '$\\tau_{23}$')
    ax1.text(s31, 0, 0.5, '$\\tau_{31}$')
    ax1.text(*t_vec, '$\\overrightarrow{T}$', c='b')
    ax1.text(*tn_vec, '$\\overrightarrow{T_n}$', c='g')
    ax1.text(*ts_vec, '$\\overrightarrow{T_s}$', c='c')
    ax2.scatter(tn, ts, c='k', alpha=0.3)
    pass
```

Таким образом, круги Мора являются инструментом для отображения напряженного состояния. Они часто используются при проверке теории прочности системы, которые будут рассмотрены следующих разделах.

+++

После изложения понятия напряженного состояния и его изменения в зависимости от выбранного базиса перейдем к рассмотрению [напряженного состояния в земной коре](RG-4-StressesInEarthsCrust.md).

+++