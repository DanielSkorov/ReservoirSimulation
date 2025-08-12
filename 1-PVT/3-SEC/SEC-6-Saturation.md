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

(pvt-esc-saturation)=
# Определение насыщенного состояния системы

````{margin}
```{admonition} Дополнительно
:class: note
Данный раздел посвящен рассмотрению численных алгоритмов построения границы двухфазной области, являющейся одной из наиболее распространенных задач в PVT-моделировании пластовых углеводородных систем. Важно отметить, что для многокомпонентных термодинамических систем могут быть определены границы трехфазной области, подходы к построению которых не будут рассматриваться в рамках данного раздела. Достаточно подробно они рассмотрены авторами работы \[[Lindeloff and Michelsen, 2003](https://doi.org/10.2118/85971-PA)\]. Также в данном разделе не будет уделено внимание алгоритмам построения фазовых диаграмм для бинарных систем, для изучения которых рекомендуется обратиться к работам \[[Cismondi and Michelsen, 2007](https://doi.org/10.1016/j.supflu.2006.03.011); [Cismondi and Michelsen, 2007](https://doi.org/10.1016/j.fluid.2007.07.019); [Cismondi et al, 2008](https://doi.org/10.1021/ie8002914)\].
```
````

В предыдущих разделах были подробно рассмотрены условия [стабильности](SEC-1-Stability.md) и [равновесности](SEC-5-Equilibrium.md) фазового состояния многокомпонентной системы. Данный раздел будет посвящен определению *насыщенного* состояния.

```{admonition} Определение
:class: tip
***Насыщенным*** состоянием системы будем называть такое состояние, в котором начинается или заканчивается процесс изменения ее фазового состояния.
```

Например, *давлением насыщения нефти* принято считать такое давление, которое соответствует переходу нефти из однофазного состояния в двухфазное. Иными словами, добавление небольшого количества энергии нефти, находящейся на *линии насыщения*, будет приводить к переходу небольшого количества ее вещества в газовую фазу. Аналогично формулируется определение для газа, находящегося на *линии конденсации*.

[Ранее](SEC-1-Stability.md) было показано, что однофазное состояние многокомпонентной системы ($N_c$ – количество компонентов) известного компонентного состава $\mathbf{z} \in {\rm I\!R}^{N_c}$ является стабильным при фиксированных и известных давлении и температуре, если касательная, проведенная к функции энергии Гиббса в точке с заданным компонентным составом, не пересекает ее в любых других точках. Данный критерий эквивалентен условию неотрицательности функции TPD (*tangent plane distance*), определяемой расстоянием от гиперповерхности энергии Гиббса до касательной гиперплоскости, проведенной в точке с заданным компонентным составом:

$$ \bar{D} \left( \mathbf{y} \right) = \sum_{i=1}^{N_c} y_i \left( \ln y_i + \ln \varphi_i \left( \mathbf{y} \right) - \ln \varphi_i \left( \mathbf{z} \right) - \ln z_i \right), $$

где $\bar{D} \left( \mathbf{y} \right)$ – приведенная функция TPD, $\mathbf{y} \in {\rm I\!R}^{N_c}$ – мольные доли компонентов в мнимой фазе, $\varphi_i \left( \mathbf{y} \right)$ – коэффициент летучести $i$-го компонента в мнимой фазе, $\varphi_i \left( \mathbf{z} \right)$ – коэффициент летучести $i$-го компонента в исходной рассматриваемой теродинамической системе.

При этом нет необходимости проверять значения функции TPD во всех точках – достаточно проверить ее значение в глобальном минимуме или во всех локальных минимумах, задаваясь различными начальными приближениями. То есть если для рассматриваемой термодинамической системы функция TPD характеризуется отрицательным значением в ее нетривиальном минимуме (то есть не соответствующим решению $\mathbf{y} = \mathbf{z}$), то однофазное состояние не является стабильным. И наоборот: если положительным, то однофазное состояние системы стабильно. Следовательно, граница двухфазной области, характеризующаяся пренебрежимо малым количеством другой, мнимой фазы (*trial phase*), находящейся в термодинамическом равновесии с исходным компонентным составом, определяется условием равенства нулю функции TPD в ее нетривиальном минимуме.

Для иллюстрации данного условия обратимся к примеру, рассмотренному [ранее](SEC-5-Equilibrium.md#pvt-sec-equilibrium-pt-ss-examples).

```{admonition} Пример
:class: exercise
Пусть имеется $1 \; моль$ смеси из метана и диоксида углерода при температуре $10 \; ^{\circ} C$ и давлении $6 \; МПа$ с мольной долей метана $0.1$. Необходимо определить равновесное состояние системы.
```

Для решения данной и последующих задач будем использовать [уравнение состояние Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.md) и его [реализацию](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/eos.py). Кроме того, для проверки [стабильности](SEC-1-Stability.md) системы будет применяться подход, основанный на примениии метода последовательных подстановок и метода Ньютона, алгоритм которого реализован [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/stability.py). Для расчета стационарного состояния системы будем также использовать комбинацию метода последовательных подстановок и метода Ньютона, с реализацией можно ознакомиться [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/flash.py).

Импортируем необходимые классы и функции.

```{code-cell} python
import sys
sys.path.append('../../_src/')
from eos import pr78
from stability import stabilityPT
from flash import flash2pPT
```

Зададим исходные термобарические условия и компонентный состав.

```{code-cell} python
import numpy as np
P = 6e6 # Pressure [Pa]
T = 10. + 273.15 # Temperature [K]
yi = np.array([.9, .1]) # Mole fractions [fr.]
```

Зададим свойства компонентов, необходимые для уравнения состояния Пенга-Робинсона, и выполним инициализацию класса.

```{code-cell} python
Pci = np.array([7.37646, 4.600155]) * 1e6 # Critical pressures [Pa]
Tci = np.array([304.2, 190.6]) # Critical temperatures [K]
wi = np.array([0.225, 0.008]) # Acentric factors
mwi = np.array([0.04401, 0.016043]) # Molar mass [kg/gmole]
vsi = np.array([0., 0.]) # Volume shift parameters
dij = np.array([0.025]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
```

Проиницилизируем класс для проведения теста стабильности и выполним проверку стабильности однофазного состояния.

```{code-cell} python
stab = stabilityPT(pr, method='ss-newton', checktrivial=False)
stabres = stab.run(P, T, yi)
print(stabres)
```

В результате проверки стабильности однофазное состояние системы оказалось нестабильным. Определим компонентные составы фаз для двухфазного состояния.

```{code-cell} python
flash = flash2pPT(pr, method='ss-newton')
flashres = flash.run(P, T, yi)
print(flashres)
```

Проверим, является ли найденное стационарное состояние равновесным:

```{code-cell} python
print(stab.run(P, T, flashres.yji[0]))
```

Рассчитаем значения функции TPD для компонентного состава первой фазы и трех температур: $9 \; ^{\circ} \mathrm{C}$, $10 \; ^{\circ} \mathrm{C}$ и $11 \; ^{\circ} \mathrm{C}$.

```{code-cell} python
yj1 = np.linspace(1e-4, 0.9999, 1000, endpoint=True)
yji = np.vstack([yj1, 1. - yj1]).T

lnfji = pr.getPT_lnphiji_Zj(P, T, yji)[0] + np.log(P * yji)
Gj = np.vecdot(yji, lnfji)
lnfi = pr.getPT_lnfi(P, T, flashres.yji[0])
Lj = yji.dot(lnfi)
Dj_10 = Gj - Lj

lnfji = pr.getPT_lnphiji_Zj(P, T-1., yji)[0] + np.log(P * yji)
Gj = np.vecdot(yji, lnfji)
lnfi = pr.getPT_lnfi(P, T-1., flashres.yji[0])
Lj = yji.dot(lnfi)
Dj_9 = Gj - Lj

lnfji = pr.getPT_lnphiji_Zj(P, T+1., yji)[0] + np.log(P * yji)
Gj = np.vecdot(yji, lnfji)
lnfi = pr.getPT_lnfi(P, T+1., flashres.yji[0])
Lj = yji.dot(lnfi)
Dj_11 = Gj - Lj
```

Построим графики функций.

```{code-cell} python
from matplotlib import pyplot as plt

fig1, ax1 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax1.plot(yj1, Dj_9, lw=2., c='cyan', zorder=2, label='t = 9 °C')
ax1.plot(yj1, Dj_10, lw=2., c='lime', zorder=2, label='t = 10 °C')
ax1.plot(yj1, Dj_11, lw=2., c='orange', zorder=2, label='t = 11 °C')
ax1.grid(zorder=1)
ax1.set_xlim(0., 1.)
# ax1.set_ylim(0., 1.)
ax1.set_xlabel('Количество вещества диоксида углерода в первой фазе, моль')
ax1.set_ylabel('Tangent plane distance (TPD)')
ax1.legend(loc=3)

ax1ins = ax1.inset_axes([.55, .4, .42, .55], xlim=(.7, 1.), ylim=(-0.01, .04))
ax1ins.plot(yj1, Dj_9, lw=2., c='cyan', zorder=3)
ax1ins.plot(yj1, Dj_10, lw=2., c='lime', zorder=3)
ax1ins.plot(yj1, Dj_11, lw=2., c='orange', zorder=3)
ax1ins.text(0.8, 0.02, '$y_{CO_2} = 0.818$', fontsize=8, color='b', rotation='vertical')
ax1ins.plot([0.818, 0.818], [0., .035], lw=1., ls='--', c='b', zorder=4)
ax1ins.plot([0.818], [1e-3], lw=0., marker='v', c='b', zorder=4)
ax1ins.text(0.9, 0.02, '$x_{CO_2} = 0.918$', fontsize=8, color='g', rotation='vertical')
ax1ins.plot([0.918, 0.918], [0., .035], lw=1., ls='--', c='g', zorder=4)
ax1ins.plot([0.918], [1e-3], lw=0., marker='v', c='g', zorder=4)
ax1ins.plot([0., 1.], [0., 0.], lw=1., ls='-', c='k', zorder=2)
ax1ins.set_xlabel('Количество вещества диоксида\nуглерода в первой фазе, моль', fontsize=9)
ax1ins.set_ylabel('Tangent plane distance (TPD)', fontsize=9)
ax1ins.tick_params(axis='both', labelsize=8)
ax1ins.grid(zorder=1)
```

Таким образом, по данным, представленным на графике выше, видно, что состояние фазы многофазной системы, находящейся в равновесном состоянии, является для этой фазы *насыщенным состоянием* при равновесных давлении, температуре и компонентном составе. Действительно, увеличение температуры на $1 \; ^{\circ} \mathrm{C}$ нарушит равновесное состояние системы и приведет к изменению компонентных составов фаз, то есть к протеканию фазового перехода, что подтверждается отрицательностью функции TPD в нетривиальном минимуме. Аналогичный результат может быть получен для изменения давления и компонентного состава фазы, находящейся в насыщенном состоянии.

Данный вывод может быть проиллюстрирован следующим графиком.

```{code-cell} python
:tags: [remove-input]

fig2 = plt.figure(figsize=(6., 4.), tight_layout=True)

def _plot(fig):
  from matplotlib.colors import Normalize
  from matplotlib.gridspec import GridSpec

  from boundary import env2pPT

  yi = np.array([.9, .1])

  Pci = np.array([7.37646, 4.600155]) * 1e6
  Tci = np.array([304.2, 190.6])
  wi = np.array([.225, .008])
  mwi = np.array([0.04401, 0.016043])
  vsi = np.array([0., 0.])
  dij = np.array([.025])
  pr = pr78(Pci, Tci, wi, mwi, vsi, dij)

  stab_ = stabilityPT(pr, method='qnss', tol=1e-8)

  gs = GridSpec(nrows=1, ncols=2, width_ratios=(25, 1))
  ax = fig.add_subplot(gs[0, 0])
  ax_cm = fig.add_subplot(gs[0, 1])

  gradient = np.linspace(0, 1, 256)
  gradient = np.vstack((gradient, gradient))

  ax_cm.imshow(gradient.T[::-1], aspect='auto', cmap='viridis')
  ax_cm.set_xticks([])
  ax_cm.yaxis.tick_right()
  ax_cm.yaxis.set_label_position("right")
  ax_cm.set_yticks(np.linspace(0, 255, 10, endpoint=True))
  ax_cm.set_yticklabels(['$-10^{%s}$'%s for s in range(1, -9, -1)])
  ax_cm.set_ylabel('Глобальный минимум функции TPD')

  class neglognorm(Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
      super().__init__(vmin, vmax, clip)

    def __call__(self, value):
      return (np.log10(-value) - self.vmin) / (self.vmax - self.vmin)

  norm = neglognorm(vmin=np.log10(1e-8), vmax=np.log10(10.))

  ax.pcolormesh([-150., 50], [0., 10.], [[-1e-8, -1e-8], [-1e-8, -1e-8]],
                norm=norm, cmap='viridis')

  lims = [
    [0., 25., 6e6, 8e6],
    [0., 25., 4e6, 6e6],
    [-25., 0., 4e6, 6e6],
    [-25., 0., 2e6, 4e6],
    [-50., -25., 2e6, 4e6],
    [-50., -25., 1e4, 2e6],
    [-75., -50., 1e4, 2e6],
    [-100., -75., 1e4, 2e6],
    [-125., -100., 1e4, 2e6],
    [-150., -125., 1e4, 2e6],
  ]

  for lim in lims:
    Ps = np.linspace(lim[2], lim[3], 50, endpoint=True)
    Ts = np.linspace(lim[0]+273.15, lim[1]+273.15, 50, endpoint=True)
    TPDs = []
    for P in Ps:
      TPDs.append([])
      for T in Ts:
        stabres = stab_.run(P, T, yi)
        TPDs[-1].append(stabres.TPD)
    TPDs = np.array(TPDs)
    ax.pcolormesh(Ts - 273.15, Ps / 1e6, TPDs, norm=norm, cmap='viridis')

  env = env2pPT(pr, Tmin=120., Tmax=300., Pmax=10e6,
                psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                stabkwargs=dict(method='qnss-newton')),
                flashkwargs=dict(method='qnss-newton', runstab=False,
                                 useprev=True, tol=1e-8))

  P0 = 4351324.8
  T0 = -10. + 273.15
  res = env.run(P0, T0, yi, 0., maxpoints=91, sidx0=1)

  ax.plot(res.Tk-273.15, res.Pk/1e6, lw=2., c='teal', zorder=4,
           label='Граница двухфазной области')
  if res.Pc is not None:
    ax.plot(res.Tc-273.15, res.Pc/1e6, 'o', lw=0., c='r', zorder=5,
             label='Критическая точка')
  ax.set_xlim(-150., 50.)
  ax.set_ylim(0., 10.)
  ax.set_xlabel('Температура, °C')
  ax.set_ylabel('Давление, МПа')
  ax.legend(loc=2, fontsize=9)
  ax.grid()
  pass

_plot(fig2)
```

Данный рисунок иллюстрирует значения глобального минимума функции TPD для рассматриваемой системы при различных давлениях и температурах. Внутри двухфазной области значения функции TPD отрицательны, что свидетельствует о нестабильности однофазового состояния. Таким образом, целью данного раздела будет исследование алгоритмов определения границ двухфазной области (линий *насыщения*, *точек росы*) и особых точек, таких как *криконденбара*, *крикондентерма*, *критическая точка*.

(pvt-esc-saturation-psat)=
## Определение давления насыщения


(pvt-esc-saturation-tsat)=
## Определение температуры насыщения


(pvt-esc-saturation-psatmax)=
## Определение криконденбары


(pvt-esc-saturation-tsatmax)=
## Определение крикондентермы


(pvt-esc-saturation-bound)=
## Построение границы двухфазной области


(pvt-esc-saturation-internals)=
## Промежуточные линии (линии постоянной мольной доли фазы)


(pvt-esc-saturation-retrograde)=
## Границы ретроградных областей

