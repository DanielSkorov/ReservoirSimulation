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
from matplotlib import patches
import numpy as np

def plot_angle_arc(axis, x0, y0, theta1, theta2, rad, num=1, inc=0.1, color='k'):
    inc *= rad
    for r in np.linspace(rad, rad + (num - 1) * inc, num):
        axis.add_patch(patches.Arc((x0, y0), width=r, height=r, theta1=theta1, theta2=theta2, color=color))
    pass

```

(math-lab-rotationangles)=
# Направляющие косинусы

Рассмотрим вектор $\mathbf{v} = \left(3, \, 4 \right)$, заданный своими координатами в базисе $\left( \mathbf{i}, \, \mathbf{j} \right)$.

```{code-cell} python
:tags: ['remove-input']

fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.header_visible = False

x0 = y0 = [0., 0., 0., 0.]

U = [[1., 0., 3., 0.6]]
V = [[0., 1., 4., 0.8]]

ax.plot([0., 3.], [0., 0.], lw=1.5, c='r', zorder=1)
ax.plot([3., 3.], [0., 4.], lw=1., c='r', ls='--', zorder=1)
ax.plot([0., 0.], [0., 4.], lw=1.5, c='r', zorder=1)
ax.plot([0., 3.], [4., 4.], lw=1., c='r', ls='--', zorder=1)
ax.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r', 'y'], zorder=2)

plot_angle_arc(ax, 0., 0., 0., np.rad2deg(np.arctan(4./3.)), 1., 1, color='b')
plot_angle_arc(ax, 0., 0., np.rad2deg(np.arctan(4./3.)), 90., 1., 2, color='g')

ax.text(1., -0.35, r'$\mathbf{i}$')
ax.text(-0.2, 1., r'$\mathbf{j}$')
ax.text(3., 4., r'$\mathbf{v}$', c='r')
ax.text(2., -0.35, '$v_x$')
ax.text(-0.35, 2., '$v_y$')
ax.text(0.5, 0.2, r'$\alpha$', c='b')
ax.text(0.15, 0.65, r'$\beta$', c='g')
ax.text(0.6, 0.62, r'$\mathbf{u}$', c='y')

ax.set_xlim(-1., 5.)
ax.set_ylim(-1., 5.)
ax.set_axisbelow(True)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
```

Проекцией вектора на координатную ось является его координата.

```{code-cell} python
v = np.array([3, 4])
v_x = 3
v_y = 4
```

Длина вектора $\mathbf{v}$:

```{code-cell} python
mod_v = np.linalg.norm(v)
mod_v
```

Обозначим через $\alpha$ угол между вектором $\mathbf{v}$ и положительным направлением оси $Ox$ (вектором $\mathbf{i}$). Тогда косинус этого угла:

$$ \cos \alpha = \frac{v_x}{\left| \mathbf{v} \right|}. $$

```{code-cell} python
cos_alpha = v_x / mod_v
cos_alpha
```

Аналогично для угла $\beta$:

$$ \cos \beta = \frac{v_y}{\left| \mathbf{v} \right|}. $$

```{code-cell} python
cos_beta = v_y / mod_v
cos_beta
```

Косинусы $\cos \alpha$ и $\cos \beta$ называются ***направляющими косинусами***. Причем, для любого ненулевого вектора справедливо равенство:

$$ \cos^2 \alpha + \cos^2 \beta = 1. $$

```{code-cell} python
cos_alpha**2 + cos_beta**2
```

Данное свойство характерно и для трехмерного пространства.

Таким образом, вектор $\mathbf{u}$, координатами которого являются направляющие косинусы вектора $\mathbf{v}$, сонаправлен с ним. При этом, длина такого вектора $\mathbf{u}$ равна 1.
