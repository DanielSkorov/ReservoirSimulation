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

(pvt-esc-equilibrium)=
# Определение равновесного состояния системы

В предыдущих разделах данной главы были представлены особенности проверки [стабильности фазового состояния системы](SEC-1-Stability.md), а также получено [уравнение Речфорда-Райса](SEC-2-RR.md) и рассмотрены численные методы его решения для [двухфазной](SEC-3-RR-2P.md) и [многофазной](SEC-4-RR-NP.md) систем. Данный раздел посвящен расчету равновесного фазового состояния для различных формулировок: [PT-термодинамика](#pvt-esc-equilibrium-pt), [VT-термодинамика](#pvt-esc-equilibrium-vt).

(pvt-esc-equilibrium-pt)=
## PT-термодинамика

В результате решения задачи поиска равновесного состояния многокомпонентной ($N_c$ – количество компонентов), многофазной ($N_p$ – количество фаз) системы для фиксированных и известных давления $P$, температуры $T$, а также количеств вещества компонентов в системе $n_i, \, i = 1 \, \ldots \, N_c,$ требуется определить количества вещества компонентов в фазах $n_{ji}, \, j = 1 \, \ldots N_p - 1, \, i = 1 \, \ldots \, N_c,$ то есть всего $N_c \times \left( N_p - 1 \right)$ неизвестных, соответствующих положению [глобального минимума функции энергии Гиббса](../1-TD/TD-14-PhaseEquilibrium.md), с учетом ограничений $0 \leq n_{ji} \leq n_i, \, j = 1 \, \ldots N_p, \, i = 1 \, \ldots \, N_c,$ и $\sum_{j=1}^{N_p} n_{ji} = n_i, \, i = 1 \, \ldots \, N_c,$. В этом случае задача поиска равновесного состояния формулируется как задача минимизации функции энергии Гиббса и, принимая во внимание ее [экстенсивность](../1-TD/TD-9-Observables.md#pvt-td-observables-extensive), может быть записана следующим образом:

````{margin}
```{admonition} Дополнительно
:class: note
При использовании двойного индексирования здесь и далее первый индекс будет обозначать фазу, второй – компонент. То есть под $n_{ji}$ следует понимать количество вещества $i$-го компонента в $j$-й фазе, а под, например, $f_{N_pk}$ – летучесть $k$-го компонента в $N_p$-й фазе. При использовании одинарного индексирования будет явно указано отношение данного элемента к вектору, характеризующему свойства фаз или компонентов.
```
````

$$ \begin{alignat}{1}
\min_{\hat{K}} & \;\;\;\;\;\;\; && G \left( \hat{K} \right) = \sum_{j=1}^{N_p} G_j \left( \mathbf{n}_j \right) \\
\mathrm{subject\,to} &&& 0 \leq n_{ji} \leq n_i, \; j = 1 \, \ldots \, N_p, \; i = 1 \, \ldots \, N_c \\
&&& \sum_{j=1}^{N_p} n_{ji} = n_i, \; i = 1 \, \ldots \, N_c
\end{alignat} $$

В представленном выше выражении используется система нестрогих неравенств: $0 \leq n_{ji} \leq n_i, \, j = 1 \, \ldots \, N_p, \, i = 1 \, \ldots \, N_c,$ позволяющая методам глобальной оптимизации убирать из системы фазы, задавая количествам вещества компонентов в них нулевые значения. Несмотря на возможность применения методов глобальной оптимизации для решения задачи определения равновесного состояния системы \[[Pan and Firoozabadi, 1992](https://doi.org/10.2118/37689-PA); [Nichita et al, 2002](https://doi.org/10.1016/S0098-1354(02)00144-8)\] их целесообразность для задач гидродинамического моделирования остается сомнительной из-за высокой сложности алгоритма. В связи с этим на практике зачастую применяюся методы локальной минимизации для фиксированного количества фаз $N_p$. Причем, поскольку количество фаз $N_p$ заранее неизвестно, то определение равновесного состояния сводится к циклу, состоящему из следующих последовательных действий:

1. Делается предположение $N_p$-фазового состояния. Проводится расчет распределения количеств вещества компонентов для $N_p$-фазового состояния.
2. Выполняется проверка стабильности $N_p$-фазового состояния.
3. Если $N_p$-фазовое состояние оказалось стабильным, то оно является равновесным. Иначе количество фаз увеличивается на единицу, и цикл повторяется.

При отсутствии исходных данных о количестве фаз в системе данный алгоритм начинается с предположения о нахождении системы в однофазном состоянии. Заканчивается данный цикл, пока не будет определено равновесное состояние, или пока не будет достигнуто максимальное количество фаз в системе.

[Известно](https://en.wikipedia.org/wiki/First_derivative_test), что необходимым условием для соответствия некоторой точки (состояния) глобальному минимуму функции (энергии Гиббса) является равенство нулю производной этой функции по независимым переменным в этой точке (этом состоянии). [Ранее](SEC-1-Stability.md) было показано, что состояние, в котором частные производные энергии Гиббса всей системы по количеству вещества $i$-го компонента в $j$-ой фазе для всех $i = 1 \, \ldots \, N_c$ и $j = 1 \, \ldots \, N_p - 1$ при фиксированных давлении и температуре равняются нулю, называется *стационарным* и характеризуется равенством химических потенциалов соответствующих компонентов в фазах:

$$ \mu_{ji} - \mu_{N_pi} = 0, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c. $$

Принимая во внимание [соотношение](../1-TD/TD-15-Fugacity.md) между летучестью и химическим потенциалом компонентов при фиксированных и известных давлении и температуре всей системы, система уравнений, определяющая положение стационарных точек функции энергии Гиббса (локальных минимумов), может быть записана с использованием летучестей компонентов:

$$ f_{ji} - f_{N_pi} = 0, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c. $$

При использовании *констант фазового равновесия*, введенных при рассмотрении [уравнения Речфорда-Райса](SEC-2-RR.md), и коэффициентов летучести компонентов данная система уравнений записывается следующим образом:

$$ g_{ji} = \ln K_{ji} + \ln \varphi_{ji} - \ln \varphi_{N_pi} = 0, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c. $$

(pvt-sec-equilibrium-pt-ss)=
### Метод последовательных подстановок

Одним из наиболее распространенных подходов к решению системы нелинейных уравнений, определяющей положение стационарных точек функии энергии Гиббса, является *метод последовательных подстановок (successive substitution method)*, также называемый [методом простой итерации](https://en.wikipedia.org/wiki/Fixed-point_iteration), суть которого заключается в итерационном обновлении основных переменных с использованием следующего выражения:

$$ \mathbf{t}^{k+1} = f \left( \mathbf{t}^{k} \right), $$

где $\mathbf{t}^{k+1} \in {\rm I\!R}^n$ – вектор основных переменных на $\left( k+1 \right)$-й итерации, $\mathbf{t}^{k} \in {\rm I\!R}^n$ – вектор основных переменных на $\left( k \right)$-й итерации, $f \, : \, {\rm I\!R}^n \rightarrow {\rm I\!R}^n$ – функция, принимающая на вход вектор размерности $n$ и возвращающая вектор такой же размерности. Соответственно, такая система уравнений должна иметь решение, удовлетворяющее следующему соотношению:

$$ \mathbf{t}^{*} = f \left( \mathbf{t}^{*} \right), $$

где $\mathbf{t}^{*} \in {\rm I\!R}^n$ – вектор основных переменных, соответствующий решению системы нелинейных уравнений.

Применительно к рассматриваеммой задаче обновление основных переменных (логарифмов констант фазового равновесия) может быть осуществлено в методе последовательных подстановок с использованием следующего выражения:

$$ \begin{align}
\ln K_{ji}^{k+1} &= \ln \varphi_{N_pi}^k - \ln \varphi_{ji}^k, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \\
\ln K_{ji}^{k+1} &= \ln K_{ji}^{k} - g_{ji}^k, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c,
\end{align} $$

где $k$ – номер итерации.

Важно отметить, что при использовании констант фазового равновесия в качестве основных переменных для решения системы нелинейных уравнений, определяющей положение локальных минимумов функции энергии Гиббса, методом последовательных подстановок необходимо на каждой итерации внешнего цикла (обновления констант фазового равновесия) решать [уравнение Речфорда-Райса](SEC-2-RR.md) во внутреннем цикле для определения мольных долей фаз и их компонентных составов.

Рассмотрим алгоритм метода последовательных подстановок.

```{eval-rst}
.. role:: comment
    :class: comment
```

```{admonition} Алгоритм. Метод последовательных подстановок для определения равновесного состояния
:class: algorithm

**Дано:** Вектор компонентного состава исследуемой системы $\mathbf{z} \in {\rm I\!R}^{N_c}$; термобарические условия $P$ и $T$; количество вещества в системе $n=1 \, моль$; необходимые свойства компонентов для нахождения коэффициентов летучести компонентов с использованием уравнения состояния; количество фаз в системе $N_p$; набор (тензор) начальных приближений констант фазового равновесия $\mathbf{K}_0 \in {\rm I\!R}^{N \times \left( N_p - 1 \right) \times N_c}$; максимальное число итераций $N_{iter}$; точность $\epsilon$.

**Определить:** Компонентные составы фаз $\mathbf{Y} \in {\rm I\!R}^{ \left( N_p - 1 \right) \times N_c}$ и мольные доли фаз $\mathbf{F} \in {\rm I\!R}^{N_p-1}$ в системе, соответствующие равновесному состоянию.

**Псевдокод:**  
**def** $\phi \left( \mathbf{Y} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}, \, \ldots \right) \rightarrow \mathbf{\Phi} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ {comment}`# Функция для расчета матрицы коэф-тов летучести`  
**def** $R \left( \mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}, \, \mathbf{z} \right) \rightarrow \mathbf{F} \in {\rm I\!R}^{N_p-1}$ {comment}`# Функция для решения уравнения Речфорда-Райса`  
**for** $i := 1$ **to** $N$ **do** {comment}`# Цикл перебора начальных приближений`  
&emsp;$\mathbf{K} := \mathbf{K}_0 \left[ i,:,: \right]$ {comment}`# Матрица начальных приближений констант фазового равновесия`  
&emsp;$\mathbf{F} := R \left( \mathbf{K}, \, \mathbf{z} \right)$ {comment}`# Вектор мольных долей фаз для начального приближения`  
&emsp;$\mathbf{x} := \mathbf{z} \, / \left( \mathbf{F}^\top \left(\mathbf{K} - 1 \right) + 1 \right)$ {comment}`# Компонентный состав референсной фазы`  
&emsp;$\mathbf{Y} := \mathbf{K} \cdot \mathbf{x}$ {comment}`# Матрица компонентных составов нереференсных фаз`  
&emsp;$\mathbf{\Phi} := \phi \left( \mathbf{Y} \right)$ {comment}`# Матрица коэффициентов летучести компонентов в нереференсных фазах`  
&emsp;$\boldsymbol{\varphi} := \phi \left( \mathbf{x} \right)$ {comment}`# Вектор коэффициентов летучести компонентов в референсной фазе`  
&emsp;$\mathbf{g} := \ln \mathbf{K} + \ln \mathbf{\Phi} - \ln \boldsymbol{\varphi}$ {comment}`# Матрица невязок`  
&emsp;$k := 1$ {comment}`# Счетчик итераций`  
&emsp;**while** $\lVert \mathbf{g} \rVert_2 > \epsilon$ **and** $k < N_{iter}$ **do** {comment}`# Цикл решения системы нелинейных уравнений`  
&emsp;&emsp;$\mathbf{K} := \mathbf{K} \cdot \exp \left( - \mathbf{g} \right)$  {comment}`# Обновление матрицы основных переменных`  
&emsp;&emsp;$\mathbf{F} := R \left( \mathbf{K}, \, \mathbf{z} \right)$ {comment}`# Вектор мольных долей фаз`  
&emsp;&emsp;$\mathbf{x} := \mathbf{z} \, / \left( \mathbf{F}^\top \left(\mathbf{K} - 1 \right) + 1 \right)$ {comment}`# Компонентный состав референсной фазы`  
&emsp;&emsp;$\mathbf{Y} := \mathbf{K} \cdot \mathbf{x}$ {comment}`# Матрица компонентных составов нереференсных фаз`  
&emsp;&emsp;$\mathbf{\Phi} := \phi \left( \mathbf{Y} \right)$ {comment}`# Матрица коэффициентов летучести компонентов в нереференсных фазах`  
&emsp;&emsp;$\boldsymbol{\varphi} := \phi \left( \mathbf{x} \right)$ {comment}`# Вектор коэффициентов летучести компонентов в референсной фазе`  
&emsp;&emsp;$\mathbf{g} := \ln \mathbf{K} + \ln \mathbf{\Phi} - \ln \boldsymbol{\varphi}$ {comment}`# Матрица невязок`  
&emsp;&emsp;$k := k + 1$ {comment}`# Обновление счетчика итераций`  
&emsp;**end while**  
&emsp;**if** $\lVert \mathbf{g} \rVert_2 < \epsilon$ **then**  
&emsp;&emsp;**exit for**  
&emsp;**end if**  
**end for**  
```

(pvt-sec-equilibrium-pt-ss-init)=
#### Начальные приближения

В качестве начального приближения для алгоритма метода последовательных подстановок при определении двухфазного равновесного состояния можно использовать результаты анализа стабильности, одним из которых является вектор $Y_i, \, i = 1 \, \ldots \, N_c,$ соответствующий наименьшему стационарному значению функции $TPD$ и интерепретируемый как вектор количеств вещества компонентов в мнимой фазе, находящейся в термодинамическом равновесии с известным компонентным составом проверяемой на стабильность системы. Тогда вектор

$$ y_i = \frac{Y_i}{\sum_{j=1}^{N_c} Y_j}, \; i = 1 \, \ldots \, N_c, $$

представляет собой вектор мольных долей компонентов в мнимой фазе. Следовательно, набор начальных приближений для расчета равновесного состояния может быть записан следующим образом \[[Pan and Firoozabadi, 2003](https://doi.org/10.2118/87335-PA)\]:

$$ \mathbf{K}_0 = \left\{ y_i \, / \, z_i, \; z_i  \, / \, y_i, \; i = 1 \, \ldots \, N_c \right\}. $$

Использование такого набора начальных приближений для расчета двухфазного равновесного состояния отлично подходит для состояний систем, находящихся вблизи границы двуфазной области, однако для случаев, когда состояние далеко от границы двухфазной области, рекомендуется использовать набор начальных приближений, что и для [проверки стабильности системы](SEC-1-Stability.md#pvt-sec-stability-pt-ss-init) \[[Pan and Firoozabadi, 2003](https://doi.org/10.2118/87335-PA); [Nichita and Graciaa, 2011](https://doi.org/10.1016/j.fluid.2010.11.007)\]. В качестве критерия, определяющего, находится ли система вблизи границы двухфазной области, можно использовать значение [функции TPD](SEC-1-Stability.md). То есть если $TPD > -c_1$, то начальное приближение лучше генерировать на основе результатов анализа стабильности, в противном случае – на основе эмпирических корреляций. Значения параметра $c_1$ авторами работы \[[Pan and Firoozabadi, 2003](https://doi.org/10.2118/87335-PA)\] рекомендуется выбирать из диапазона $\left[ 10^{-4}, \, 10^{-3} \right]$.

Подход к определению начальных приближений для многофазного расчета равновесного состояния был предложен авторами работы \[[Li and Firoozabadi, 2012](https://doi.org/10.1016/j.fluid.2012.06.021)\]. Результатом двухфазного расчета равновесного состояния является некоторое состояние системы, характеризующееся равенством летучестей соответствующих компонентов, компонентными составами фаз $y_{1i}, \, i = 1 \, \ldots \, N_c,$ и $y_{2i}, \, i = 1 \, \ldots \, N_c,$, мольными долями фаз $F_1$ и $F_2$, а также константами фазого равновесия $K_{1i} = y_{1i} \, / \, y_{2i}, \, i = 1 \, \ldots \, N_c$. Затем, согласно представленному выше алгоритму, должен быть проведен тест стабильности компонентного состава одной из фаз, например, $y_{2i}, \, i = 1 \, \ldots \, N_c,$. В случае нестабильности двухфазного состояния системы результатом теста является некоторая мнимая фаза *(trial phase)* с компонентным составом $x_i, \, i = 1 \, \ldots \, N_c,$ и константами фазового равновесия $K_{ti} = x_i \, / \, y_{2i}, \, i = 1 \, \ldots \, N_c,$ находящаяся в термодинамическом равновесии с компонентным составом тестируемой фазы. Таким образом, набор констант фазового равновесия $\mathbf{K} = \left\{ K_{1i}, \, K_{ti} \right\}$ является хорошим начальным приближением для расчета трехфазного равновесного состояния. При этом начальное приближение необходимо и для [алгоритма](SEC-4-RR-NP.md) решения системы уравнений Речфорда-Райса, в качестве которого может выступить вектор $\mathbf{F} = \left( F_1, \, 0 \right)^\top$, поскольку мнимая фаза характеризуется пренебрежимо малой мольной долей.

(pvt-sec-equilibrium-pt-ss-examples)=
#### Примеры

Рассмотрим применение метода последовательных подстановок для решения системы нелинейных уравнений, определяющей положение локальных минимумов функции энергии Гиббса.

```{admonition} Пример
:class: exercise
Пусть имеется $1 \; моль$ смеси из метана и диоксида углерода при температуре $10 \; ^{\circ} C$ и давлении $6 \; МПа$ с мольной долей метана $0.1$. Необходимо определить равновесное состояние системы.
```

````{dropdown} Решение
Для решения данной и последующих задач будем использовать [уравнение состояние Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.md) и его [реализацию](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/eos.py). Кроме того, для проверки стабильности системы будет применяться метод последовательных подстановок, алгоритм которого был рассмотрен [ранее](SEC-1-Stability.md), реализованный [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/stability.py). Также на каждой итерации решения системы уравнений термодинамического равновесия необходимо решать [уравнение Речфорда-Райса](SEC-2-RR.md). Для решения уравнения в двухфазной постановке будет использоваться [реализация](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/rr.py) [метода FGH](SEC-3-RR-2P.md), для решения системы уравнений Речфорда-Райса – модифицированный метод, основанный на рассмотренном [ранее](SEC-3-RR-NP.md) подходе \[[Okuno et al, 2010](https://doi.org/10.2118/117752-PA)\], реализация которого представлена [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/rr.py).

Импортируем необходимые классы и функции.

``` python
import sys
sys.path.append('../../_src/')
from eos import pr78
from stability import stabilityPT
from rr import solve2p_FGH, solveNp
```

Зададим исходные термобарические условия и компонентный состав.

``` python
import numpy as np
P = 6e6 # Pressure [Pa]
T = 10. + 273.15 # Temperature [K]
yi = np.array([.9, .1]) # Mole fractions [fr.]
```

Также зададим максимальное число итераций $N_{iter}$, точность решения системы нелинейных уравнений $\epsilon$:

``` python
maxiter = 50 # Maximum number of iterations
eps = 1e-6 # Tolerance
```

Зададим свойства компонентов, необходимые для уравнения состояния Пенга-Робинсона, и выполним инициализацию класса.

``` python
Pci = np.array([7.37646, 4.600155]) * 1e6 # Critical pressures [Pa]
Tci = np.array([304.2, 190.6]) # Critical temperatures [K]
wi = np.array([.225, .008]) # Acentric factors
mwi = np.array([0.04401, 0.016043]) # Molar mass [kg/gmole]
vsi = np.array([0., 0.]) # Volume shift parameters
dij = np.array([.025]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
```

Проиницилизируем класс для проведения теста стабильности и выполним проверку стабильности однофазного состояния:

``` python
stab = stabilityPT(pr, method='ss')
stabres = stab.run(P, T, yi)
print(stabres)
```

```{glue:} glued_out1
```

В результате проверки стабильности (путем вызова метода `run`) однофазное состояние системы оказалось нестабильным. Также был получен набор начальных приближений для решения нахождения решения системы нелинейных уравнений, определяющей положение локальных (для двухфазной системы) минимумов функции энергии Гиббса:

``` python
print(f'kvji:\n{stabres.kvji}')
```

```{glue:} glued_out2
```

Создадим функцию, которая будет принимать на вход кортеж из результатов предыдущей итерации, точность и максимальное число итераций, и возвращать необходимость расчета следующей итерации цикла решения системы нелинейных уравнений термодинамического равновесия методом последовательных подстановок. Проиницилизируем данную функцию.

``` python
from functools import partial

def condit_ssi(carry, tol, maxiter):
    k, kvi, _, gi = carry
    return k < maxiter and np.linalg.norm(gi) > tol

pcondit_ssi = partial(condit_ssi, tol=eps, maxiter=maxiter)
```

Также создадим функцию, которая будет принимать на вход результаты предыдущей итерации в виде кортежа и рассчитывать результаты для новой итерации. Для расчета логарифмов летучести компонентов будем использовать метод `getPT_lnphii` инициализированного класса с уравнением состояния, принимающий на вход давление (в Па), температуру (в K) и компонентный состав в виде одномерного массива (размерностью `(Nc,)`) и возвращающий массив логарифмов коэффициентов летучести компонента такой же размерности.

``` python
def update_ssi_2p(carry, yi, plnphi):
    k, kvi_k, _, gi_k = carry
    kvi_kp1 = kvi_k * np.exp(-gi_k)
    F1_kp1 = solve2p_FGH(kvi_kp1, yi)
    y2i = yi / (F1_kp1 * (kvi_kp1 - 1.) + 1.)
    y1i = y2i * kvi_kp1
    lnphi2i = plnphi(yi=y2i)
    lnphi1i = plnphi(yi=y1i)
    gi_kp1 = np.log(kvi_kp1) + lnphi1i - lnphi2i
    return k + 1, kvi_kp1, F1_kp1, gi_kp1

pupdate_ssi_2p = partial(update_ssi_2p, yi=yi, plnphi=partial(pr.getPT_lnphii, P=P, T=T))
```

Найдем решение системы нелинейных уравнений с использованием метода последовательных подстановок.

``` python
for i, kvi in enumerate(stabres.kvji):
    F1 = solve2p_FGH(kvi, yi)
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i = pr.getPT_lnphii(P, T, y2i)
    lnphi1i = pr.getPT_lnphii(P, T, y1i)
    gi = np.log(kvi) + lnphi1i - lnphi2i
    carry = (1, kvi, F1, gi)
    while pcondit_ssi(carry):
        carry = pupdate_ssi_2p(carry)
    k, kvi, F1, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        print(f'For the initial guess #{i}:\n'
              f'\ttolerance of equations: {gnorm}\n'
              f'\tnumber of iterations: {k}\n'
              f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
              f'\tphase mole fractions: {F1}, {1.-F1}')
        break
```

```{glue:} glued_out3
```

Выполним проверку стабильности найденного решения.

``` python
print(stab.run(P, T, y1i))
```

```{glue:} glued_out4
```

Тест стабильности показал, что найденное решение соответствует равновесному состоянию. Проиллюстрируем данный пример графически. Для этого построим зависимость функции приведенной добавочной энергии Гиббса для первой фазы от компонентного состава и проведем касательную в точке с найденным равновесным составом этой фазы. Для расчета коэффициентов летучестей компонентов для различных составов будем использовать метод `getPT_lnphiji_Zj`, принимающий в качестве аргументов давление (в Па), температуру (в K) и набор компонентных составов в виде двумерного массива (размерностью `(Np, Nc)`, где `Np` – количество наборов компонентных составов, `Nc` – количество компонентов в каждом компонентном составе) и возвращающий соответствующие коэффициенты летучести компонентов в виде двумерного массива такой же размерности и коэффициенты сверхсжимаемости в виде одномерного массива для каждого компонентного состава.

``` python
yj1 = np.linspace(1e-4, 0.9999, 1000, endpoint=True)
yji = np.vstack([yj1, 1. - yj1]).T
lnphiji, Zj = pr.getPT_lnphiji_Zj(P, T, yji)
lnfji = lnphiji + np.log(P * yji)
Gj = np.vecdot(yji, lnfji)
```

Получим знаяения касательной, проведенной к функции приведенной добавочной энергии Гиббса. Для расчета логарифмов летучести компонентов будем использовать метод `getPT_lnfi`, принимающий на вход давление (в Па), температуру (в K) и компонентный состав в виде одномерного масива (размерностью `(Nc,)`) и возвращающий логарифмы летучести компонентов в виде одномерного массива такой же размерности.

``` python
lnfi = pr.getPT_lnfi(P, T, y1i)
Lj = yji.dot(lnfi)
```

Построим графики функции приведенной добавочной энергии Гиббса и касательной к ней.

``` python
from matplotlib import pyplot as plt

fig1, ax1 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax1.plot(yj1, Gj, lw=2., c='teal', zorder=2, label='Приведенная добавочная энергия Гиббса')
ax1.plot(yj1, Lj, lw=2., c='orchid', zorder=2, label='Касательная')
ax1.set_xlim(0., 1.)
ax1.set_xlabel('Количество вещества диоксида углерода в первой фазе, моль')
ax1.set_ylabel('Приведенная добавочная энергия Гиббса')
ax1.grid(zorder=1)

ax1ins = ax1.inset_axes([0.04, 0.05, 0.5, 0.89], xlim=(0.65, 0.975), ylim=(14.5, 15.1),
                        xticklabels=[], yticklabels=[])
ax1.indicate_inset_zoom(ax1ins, edgecolor='black')
ax1ins.plot(yj1, Gj, lw=2., c='teal', zorder=2, label='Приведенная добавочная\nэнергия Гиббса')
ax1ins.plot(yj1, Lj, lw=2., c='orchid', zorder=2, label='Касательная')
ax1ins.plot(y1i[0], y1i.dot(lnfi), 'o', lw=0., mfc='blue', mec='blue', ms=5., zorder=3)
ax1ins.plot(y2i[0], y2i.dot(lnfi), 'o', lw=0., mfc='green', mec='green', ms=5., zorder=3)
ax1ins.plot([y1i[0], y1i[0]], [0., y1i.dot(lnfi)], '--', lw=1., c='blue', zorder=2)
ax1ins.plot([y2i[0], y2i[0]], [0., y2i.dot(lnfi)], '--', lw=1., c='green', zorder=2)
ax1ins.text(0.8, 14.55, '$y_{CO_2} = 0.818$', fontsize=8, color='blue', rotation='vertical')
ax1ins.text(0.9, 14.55, '$x_{CO_2} = 0.918$', fontsize=8, color='green', rotation='vertical')
ax1ins.set_xticks([0.8])
ax1ins.set_yticks([14.6, 14.8, 15.0])
ax1ins.legend(loc=2, fontsize=8)
ax1ins.grid(zorder=1)

plt.show()
```

```{glue:} glued_fig1
```

<br>
<br>

Из данного графика следует, что касательная, проведенная к функции энергии Гиббса в точке с компонентным составом первой фазы, не имеет пересечений с самой функцией, что подтверждает сделанный вывод о равновесности найденного состояния. Кроме того, можно отметить, что касательная имеет две точки касания, абсциссы которых соответствуют равновесным компонентным составам фаз, то есть функция энергии Гиббса имеет одинаковую касательную для каждого из компонентных составов, определяющих стационарное состояние. Это следует из равенства летучестей соответствующих компонентов в фазах, а полное доказательство данного утверждения было рассмотрено [ранее](SEC-1-Stability.md). Стоит также отметить, что две точки, показанные на графике, соответствующие двухфазному равновесному состоянию, называются ***бинодальными точками***.

Рассчитаем значения и построим график функции TPD для рассматриваемого компонентного состава.

``` python
Dj = Gj - Lj

fig2, ax2 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax2.plot(yj1, Dj, lw=2., c='lime', zorder=2)
ax2.grid(zorder=1)
ax2.set_xlim(0., 1.)
ax2.set_ylim(0., 1.)
ax2.set_xlabel('Количество вещества диоксида углерода в первой фазе, моль')
ax2.set_ylabel('Tangent plane distance (TPD)')

ax2ins = ax2.inset_axes([.55, .4, .42, .55], xlim=(.7, 1.), ylim=(0., .04))
ax2ins.plot(yj1, Dj, lw=2., c='m', zorder=2)
ax2ins.text(0.8, 0.02, '$y_{CO_2} = 0.818$', fontsize=8, color='b', rotation='vertical')
ax2ins.plot([0.818, 0.818], [0., .035], lw=1., ls='--', c='b', zorder=3)
ax2ins.plot([0.818], [1e-3], lw=0., marker='v', c='b', zorder=3)
ax2ins.text(0.9, 0.02, '$x_{CO_2} = 0.918$', fontsize=8, color='g', rotation='vertical')
ax2ins.plot([0.918, 0.918], [0., .035], lw=1., ls='--', c='g', zorder=3)
ax2ins.plot([0.918], [1e-3], lw=0., marker='v', c='g', zorder=3)
ax2ins.set_xlabel('Количество вещества диоксида\nуглерода в первой фазе, моль', fontsize=9)
ax2ins.set_ylabel('Tangent plane distance (TPD)', fontsize=9)
ax2ins.tick_params(axis='both', labelsize=8)
ax2ins.grid(zorder=1)

plt.show()
```

```{glue:} glued_fig2
```

<br>
<br>

Равенство касательных, проведенных к функции энергии Гиббса в точках с компонентными составами фаз, обуславливает равенство функции TPD нулю в этих точках.
````

```{code-cell} python
:tags: [remove-cell]

import sys
sys.path.append('../../_src/')
from eos import pr78
from stability import stabilityPT
from rr import solve2p_FGH, solveNp

import numpy as np
P = 6e6 # Pressure [Pa]
T = 10. + 273.15 # Temperature [K]
yi = np.array([.9, .1]) # Mole fractions [fr.]

maxiter = 50 # Number of iterations
eps = 1e-6 # Tolerance

Pci = np.array([7.37646, 4.600155]) * 1e6 # Critical pressures [Pa]
Tci = np.array([304.2, 190.6]) # Critical temperatures [K]
wi = np.array([.225, .008]) # Acentric factors
mwi = np.array([0.04401, 0.016043]) # Molar mass [kg/gmole]
vsi = np.array([0., 0.]) # Volume shift parameters
dij = np.array([.025]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)

stab = stabilityPT(pr, method='ss')
stabres = stab.run(P, T, yi)
out1 = str(stabres)

out2 = f'kvji:\n{stabres.kvji}'

from functools import partial

def condit_ssi(carry, tol, maxiter):
    k, kvi, _, gi = carry
    return k < maxiter and np.linalg.norm(gi) > tol

pcondit_ssi = partial(condit_ssi, tol=eps, maxiter=maxiter)

def update_ssi_2p(carry, yi, plnphi):
    k, kvi_k, _, gi_k = carry
    kvi_kp1 = kvi_k * np.exp(-gi_k)
    F1_kp1 = solve2p_FGH(kvi_kp1, yi)
    y2i = yi / (F1_kp1 * (kvi_kp1 - 1.) + 1.)
    y1i = y2i * kvi_kp1
    lnphi2i = plnphi(yi=y2i)
    lnphi1i = plnphi(yi=y1i)
    gi_kp1 = np.log(kvi_kp1) + lnphi1i - lnphi2i
    return k + 1, kvi_kp1, F1_kp1, gi_kp1

pupdate_ssi_2p = partial(update_ssi_2p, yi=yi, plnphi=partial(pr.getPT_lnphii, P=P, T=T))

out3 = ''

for i, kvi in enumerate(stabres.kvji):
    F1 = solve2p_FGH(kvi, yi)
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i = pr.getPT_lnphii(P, T, y2i)
    lnphi1i = pr.getPT_lnphii(P, T, y1i)
    gi = np.log(kvi) + lnphi1i - lnphi2i
    carry = (1, kvi, F1, gi)
    while pcondit_ssi(carry):
        carry = pupdate_ssi_2p(carry)
    k, kvi, F1, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        out3 += (f'For the initial guess #{i}:\n'
                 f'\ttolerance of equations: {gnorm}\n'
                 f'\tnumber of iterations: {k}\n'
                 f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
                 f'\tphase mole fractions: {F1}, {1.-F1}')
        break

out4 = str(stab.run(P, T, y1i))

yj1 = np.linspace(1e-4, 0.9999, 1000, endpoint=True)
yji = np.vstack([yj1, 1. - yj1]).T
lnphiji, Zj = pr.getPT_lnphiji_Zj(P, T, yji)
lnfji = lnphiji + np.log(P * yji)
Gj = np.vecdot(yji, lnfji)

lnfi = pr.getPT_lnfi(P, T, y1i)
Lj = yji.dot(lnfi)

from matplotlib import pyplot as plt

fig1, ax1 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax1.plot(yj1, Gj, lw=2., c='teal', zorder=2, label='Приведенная добавочная энергия Гиббса')
ax1.plot(yj1, Lj, lw=2., c='orchid', zorder=2, label='Касательная')
ax1.set_xlim(0., 1.)
ax1.set_xlabel('Количество вещества диоксида углерода в первой фазе, моль')
ax1.set_ylabel('Приведенная добавочная энергия Гиббса')
ax1.grid(zorder=1)

ax1ins = ax1.inset_axes([0.04, 0.05, 0.5, 0.89], xlim=(0.65, 0.975), ylim=(14.5, 15.1),
                        xticklabels=[], yticklabels=[])
ax1.indicate_inset_zoom(ax1ins, edgecolor='black')
ax1ins.plot(yj1, Gj, lw=2., c='teal', zorder=2, label='Приведенная добавочная\nэнергия Гиббса')
ax1ins.plot(yj1, Lj, lw=2., c='orchid', zorder=2, label='Касательная')
ax1ins.plot(y1i[0], y1i.dot(lnfi), 'o', lw=0., mfc='blue', mec='blue', ms=5., zorder=3)
ax1ins.plot(y2i[0], y2i.dot(lnfi), 'o', lw=0., mfc='green', mec='green', ms=5., zorder=3)
ax1ins.plot([y1i[0], y1i[0]], [0., y1i.dot(lnfi)], '--', lw=1., c='blue', zorder=2)
ax1ins.plot([y2i[0], y2i[0]], [0., y2i.dot(lnfi)], '--', lw=1., c='green', zorder=2)
ax1ins.text(0.8, 14.55, '$y_{CO_2} = 0.818$', fontsize=8, color='blue', rotation='vertical')
ax1ins.text(0.9, 14.55, '$x_{CO_2} = 0.918$', fontsize=8, color='green', rotation='vertical')
ax1ins.set_xticks([0.8])
ax1ins.set_yticks([14.6, 14.8, 15.0])
ax1ins.legend(loc=2, fontsize=8)
ax1ins.grid(zorder=1)

Dj = Gj - Lj

fig2, ax2 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax2.plot(yj1, Dj, lw=2., c='lime', zorder=2)
ax2.grid(zorder=1)
ax2.set_xlim(0., 1.)
ax2.set_ylim(0., 1.)
ax2.set_xlabel('Количество вещества диоксида углерода в первой фазе, моль')
ax2.set_ylabel('Tangent plane distance (TPD)')

ax2ins = ax2.inset_axes([.55, .4, .42, .55], xlim=(.7, 1.), ylim=(0., .04))
ax2ins.plot(yj1, Dj, lw=2., c='m', zorder=2)
ax2ins.text(0.8, 0.02, '$y_{CO_2} = 0.818$', fontsize=8, color='b', rotation='vertical')
ax2ins.plot([0.818, 0.818], [0., .035], lw=1., ls='--', c='b', zorder=3)
ax2ins.plot([0.818], [1e-3], lw=0., marker='v', c='b', zorder=3)
ax2ins.text(0.9, 0.02, '$x_{CO_2} = 0.918$', fontsize=8, color='g', rotation='vertical')
ax2ins.plot([0.918, 0.918], [0., .035], lw=1., ls='--', c='g', zorder=3)
ax2ins.plot([0.918], [1e-3], lw=0., marker='v', c='g', zorder=3)
ax2ins.set_xlabel('Количество вещества диоксида\nуглерода в первой фазе, моль', fontsize=9)
ax2ins.set_ylabel('Tangent plane distance (TPD)', fontsize=9)
ax2ins.tick_params(axis='both', labelsize=8)
ax2ins.grid(zorder=1)

class MultilineText(object):
    def __init__(self, text):
        self.text = text
        self.template = """
<div class="output text_plain highlight-myst-ansi notranslate"><div class="highlight"><pre id="codecell20" tabindex="0"><span></span>{}
</pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell20">
      <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <title>Copy to clipboard</title>
  <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
  <rect x="8" y="8" width="12" height="12" rx="2"></rect>
  <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
</svg>
    </button></div>
</div>
"""

    def _repr_html_(self):
        return self.template.format(self.text.replace('\n', '<br>'))

from myst_nb import glue
glue('glued_out1', MultilineText(out1))
glue('glued_out2', MultilineText(out2))
glue('glued_out3', MultilineText(out3))
glue('glued_out4', MultilineText(out4))
glue('glued_fig1', fig1)
glue('glued_fig2', fig2)
```

Представленным выше примером было проиллюстрировано применение метода последовательных подстановок для расчета двухфазного равновесного состояния. Однако данный метод может применяться и для расчета многофазной системы. Рассмотрим следующий пример.

```{admonition} Пример
:class: exercise
Пусть имеется $1 \; моль$ смеси из метана, гексана и воды при температуре $20 \; ^{\circ} C$ и давлении $1 \; атм$ с мольными долями компонентов $0.1, \, 0.6, \, 0.3$ соответственно. Необходимо определить равновесное состояние системы.
```

````{dropdown} Решение
Зададим исходные термобарические условия и компонентный состав.

``` python
P = 101325. # Pressure [Pa]
T = 20. + 273.15 # Temperature [K]
yi = np.array([.1, .6, .3]) # Mole fractions [fr.]
```

Зададим свойства компонентов, необходимые для уравнения состояния Пенга-Робинсона, и выполним инициализацию класса.

``` python
Pci = np.array([4.600155, 3.2890095, 22.04832]) * 1e6 # Critical pressures [Pa]
Tci = np.array([190.6, 507.5, 647.3]) # Critical temperatures [K]
wi = np.array([.008, .27504, .344]) # Acentric factors
mwi = np.array([0.016043, 0.086, 0.018015]) # Molar mass [kg/gmole]
vsi = np.array([0., 0., 0.]) # Volume shift parameters
dij = np.array([.0253, 0.4907, 0.48]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
```

Проиницилизируем класс для проведения теста стабильности и выполним проверку стабильности однофазного состояния.

``` python
# level=1 indicates an increased number of arrays of initial k-values
stab = stabilityPT(pr, method='ss', level=1)
stabres = stab.run(P, T, yi)
print(stabres)
```

```{glue:} glued_out5
```

Однофазное состояние оказалось нестабильным, выполним расчет двухфазного равновесного состояния. Для этого сначала проинициализируем функцию `update_ssi_2p` для рассматриваемого примера. Поскольку точность расчета и максимальное количество итераций не изменилось, то будем использовать функцию `pcondit_ssi` из предыдущего примера.

``` python
pupdate_ssi_2p = partial(update_ssi_2p, yi=yi, plnphi=partial(pr.getPT_lnphii, P=P, T=T))
```

Перебирая различные начальные приближения констант фазового равновесия, полученные после проверки стабильности однофазного состояния, найдем решение системы нелинейных уравнений, определяющей положение локальных минимумов функции энергии Гиббса в двухфазной постановке.

``` python
for i, kvi in enumerate(stabres.kvji):
    F1 = solve2p_FGH(kvi, yi)
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i = pr.getPT_lnphii(P, T, y2i)
    lnphi1i = pr.getPT_lnphii(P, T, y1i)
    gi = np.log(kvi) + lnphi1i - lnphi2i
    carry = (1, kvi, F1, gi)
    while pcondit_ssi(carry):
        carry = pupdate_ssi_2p(carry)
    k, kvi, F1, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        print(f'For the initial guess #{i}:\n'
              f'\ttolerance of equations: {gnorm}\n'
              f'\tnumber of iterations: {k}\n'
              f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
              f'\tphase mole fractions: {F1}, {1.-F1}')
        break
```

```{glue:} glued_out6
```

Метод последовательных подстановок нашел состояние, характеризующееся равенством летучестей соответствующих компонентов в фазах, за пять итераций. Проверим, является ли двухфазное состояние рассматриваемой системы стабильным. Для этого проведем тест стабильности для одной из фаз.

``` python
stabres = stab.run(P, T, y2i)
print(stabres)
```

```{glue:} glued_out7
```

По результатам проверки стабильности двухфазное состояние оказалось нестабильным, следовательно, необходимо рассмотреть трехфазное состояние.

С учетом изложенного выше подхода к определению начального приближения для расчета многофазного состояния сформируем матрицу констант фазового равновесия для первой итерации расчета трехфазного состояния.

``` python
kvji = np.vstack([kvi, stabres.kvji[0]])
print(f'kvji:\n{kvji}')
```

```{glue:} glued_out8
```

Определим мольные доли фаз, соответствующие этому начальному приближению.

``` python
Fj = solveNp(kvji, yi, np.array([F1, 0.]))
print(f'{Fj = }')
```

```{glue:} glued_out9
```

Рассчитаем компонентные составы фаз.

``` python
xi = yi / (Fj.dot(kvji - 1.) + 1.)
yji = xi * kvji
```

Выполним расчет коэффициентов летучестей для каждого компонента в каждой из фаз. Для этого воспользуемся методом `getPT_lnphiji_Zj` инициализированного класса с уравнением состояния.

``` python
lnphiji, Zj = pr.getPT_lnphiji_Zj(P, T, np.vstack([yji, xi]))
```

Рассчитаем матрицу невязок.

``` python
gji = np.log(kvji) + lnphiji[:-1] - lnphiji[-1]
```

Создадим функцию, которая будет получать на вход результаты предыдущей итерации и возвращать результаты следующей итерации метода последовательных подстановок для задачи определения многофазного равновесного состояния.

``` python
def update_ssi_Np(carry, yi, plnphi):
    k, kvji_k, Fj_k, gji_k = carry
    kvji_kp1 = kvji_k * np.exp(-gji_k)
    Fj_kp1 = solveNp(kvji_kp1, yi, Fj_k)
    xi = yi / (Fj_kp1.dot(kvji_kp1 - 1.) + 1.)
    yji = xi * kvji_kp1
    lnphiji, Zj = plnphi(yji=np.vstack([yji, xi]))
    gji_kp1 = np.log(kvji_kp1) + lnphiji[:-1] - lnphiji[-1]
    return k + 1, kvji_kp1, Fj_kp1, gji_kp1

pupdate_ssi_Np = partial(update_ssi_Np, yi=yi, plnphi=partial(pr.getPT_lnphiji_Zj, P=P, T=T))
```

В цикле `while` найдем решение системы нелинейных уравнений, определяющей положение локальных минимумов функции энергии Гиббса, соответствующих трехфазному состоянию системы.

``` python
carry = (1, kvji, Fj, gji)

while pcondit_ssi(carry):
    carry = pupdate_ssi_Np(carry)

k, kvji, Fj, gji = carry

xi = yi / (Fj.dot(kvji - 1.) + 1.)
yji = np.vstack([xi * kvji, xi])
Fj = np.hstack([Fj, 1. - Fj.sum()])
gnorm = np.linalg.norm(gji)

print(f'Tolerance of equations: {gnorm}\n'
      f'Number of iterations: {k}\n'
      f'Phase compositions:\n{yji}\n'
      f'Phase mole fractions: {Fj}')
```

```{glue:} glued_out10
```

Выполним проверку стабильности компонентного состава одной из фаз.

``` python
print(stab.run(P, T, yji[1]))
```

```{glue:} glued_out11
```

Тест стабильности показал, что определенное трехфазное состояние системы является равновесным.
````

```{code-cell} python
:tags: [remove-cell]

P = 101325. # Pressure [Pa]
T = 20. + 273.15 # Temperature [K]
yi = np.array([.1, .6, .3]) # Mole fractions [fr.]

Pci = np.array([4.600155, 3.2890095, 22.04832]) * 1e6 # Critical pressures [Pa]
Tci = np.array([190.6, 507.5, 647.3]) # Critical temperatures [K]
wi = np.array([.008, .27504, .344]) # Acentric factors
mwi = np.array([0.016043, 0.086, 0.018015]) # Molar mass [kg/gmole]
vsi = np.array([0., 0., 0.]) # Volume shift parameters
dij = np.array([.0253, 0.4907, 0.48]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)

stab = stabilityPT(pr, method='ss', level=1)
stabres = stab.run(P, T, yi)
out5 = str(stabres)

pupdate_ssi_2p = partial(update_ssi_2p, yi=yi, plnphi=partial(pr.getPT_lnphii, P=P, T=T))

out6 = ''
for i, kvi in enumerate(stabres.kvji):
    F1 = solve2p_FGH(kvi, yi)
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i = pr.getPT_lnphii(P, T, y2i)
    lnphi1i = pr.getPT_lnphii(P, T, y1i)
    gi = np.log(kvi) + lnphi1i - lnphi2i
    carry = (1, kvi, F1, gi)
    while pcondit_ssi(carry):
        carry = pupdate_ssi_2p(carry)
    k, kvi, F1, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        out6 += (f'For the initial guess #{i}:\n'
                 f'\ttolerance of equations: {gnorm}\n'
                 f'\tnumber of iterations: {k}\n'
                 f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
                 f'\tphase mole fractions: {F1}, {1.-F1}')
        break

stabres = stab.run(P, T, y2i)
out7 = str(stabres)

kvji = np.vstack([kvi, stabres.kvji[0]])
out8 = f'kvji:\n{kvji}'

Fj = solveNp(kvji, yi, np.array([F1, 0.]))
out9 = f'{Fj = }'

xi = yi / (Fj.dot(kvji - 1.) + 1.)
yji = xi * kvji
lnphiji, Zj = pr.getPT_lnphiji_Zj(P, T, np.vstack([yji, xi]))
gji = np.log(kvji) + lnphiji[:-1] - lnphiji[-1]

def update_ssi_Np(carry, yi, plnphi):
    k, kvji_k, Fj_k, gji_k = carry
    kvji_kp1 = kvji_k * np.exp(-gji_k)
    Fj_kp1 = solveNp(kvji_kp1, yi, Fj_k)
    xi = yi / (Fj_kp1.dot(kvji_kp1 - 1.) + 1.)
    yji = xi * kvji_kp1
    lnphiji, Zj = plnphi(yji=np.vstack([yji, xi]))
    gji_kp1 = np.log(kvji_kp1) + lnphiji[:-1] - lnphiji[-1]
    return k + 1, kvji_kp1, Fj_kp1, gji_kp1

pupdate_ssi_Np = partial(update_ssi_Np, yi=yi, plnphi=partial(pr.getPT_lnphiji_Zj, P=P, T=T))

carry = (1, kvji, Fj, gji)

while pcondit_ssi(carry):
    carry = pupdate_ssi_Np(carry)

k, kvji, Fj, gji = carry

xi = yi / (Fj.dot(kvji - 1.) + 1.)
yji = np.vstack([xi * kvji, xi])
Fj = np.hstack([Fj, 1. - Fj.sum()])
gnorm = np.linalg.norm(gji)

out10 = (f'Tolerance of equations: {gnorm}\n'
         f'Number of iterations: {k}\n'
         f'Phase compositions:\n{yji}\n'
         f'Phase mole fractions: {Fj}')

out11 = str(stab.run(P, T, yji[1]))

glue('glued_out5', MultilineText(out5))
glue('glued_out6', MultilineText(out6))
glue('glued_out7', MultilineText(out7))
glue('glued_out8', MultilineText(out8))
glue('glued_out9', MultilineText(out9))
glue('glued_out10', MultilineText(out10))
glue('glued_out11', MultilineText(out11))
```

Таким образом, данными примерами было проиллюстрировано применение метода последовательных подстановок для нахождения двух- и трехфазного равновесного состояния. Как и в случае с [анализом стабильности](SEC-1-Stability.md), недостатком этого метода является сравнительно большое количество итераций и медленная сходимость для термодинамических систем, находящихся вблизи [линии насыщения](SEC-6-Saturation.md) или вблизи [критической точки](SEC-7-Criticality.md). В качестве примера рассмотрим следующую задачу.

````{admonition} Пример
:class: exercise
Пусть имеется $1 \; моль$ смеси следующего мольного состава:

```{table}
:width: 100%
:widths: "1, 1, 1, 1, 1, 1"
| Компонент | $\mathrm{CH_4}$ | $\mathrm{C_2 H_6}$ | $\mathrm{C_3 H_8}$ | $\mathrm{n \mbox{-} C_4}$ | $\mathrm{C_{5+}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Мольная доля | 0.7167 | 0.0895 | 0.0917 | 0.0448 | 0.0573 |
```

<br>

Необходимо определить равновесное состояние системы при давлении $17 \; МПа$ и температуре $68 \; ^{\circ} C$.
````

````{dropdown} Решение
Зададим исходные термобарические условия и компонентный состав.

``` python
P = 17e6 # Pressure [Pa]
T = 68. + 273.15 # Temperature [K]
yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573]) # Mole fractions [fr.]
```

Зададим свойства компонентов, необходимые для уравнения состояния Пенга-Робинсона, и выполним инициализацию класса.

``` python
Pci = np.array([4.599, 4.872, 4.248, 3.796, 2.398]) * 1e6 # Critical pressures [Pa]
Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02]) # Critical temperatures [K]
wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414]) # Acentric factors
mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120]) # Molar mass [kg/gmole]
vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661]) # Volume shift parameters
dij = np.array([
    0.002689,
    0.008537, 0.001662,
    0.014748, 0.004914, 0.000866,
    0.039265, 0.021924, 0.011676, 0.006228,
]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
```

Проиницилизируем класс для проведения теста стабильности и выполним проверку стабильности однофазного состояния:

``` python
stab = stabilityPT(pr, method='ss', maxiter=200)
stabres = stab.run(P, T, yi)
print(stabres)
```

```{glue:} glued_out12
```

Однофазное состояние оказалось нестабильным. Найдем решение системы нелинейных уравнений, определяющей положение локальных минимумов функции энергии Гиббса для двухфазного состояния. Для этого сначала проинициализируем функцию `update_ssi_2p` для рассматриваемого примера.

``` python
pupdate_ssi_2p = partial(update_ssi_2p, yi=yi, plnphi=partial(pr.getPT_lnphii, P=P, T=T))
```

Увеличим максимальное количество итераций для функции `pcondit_ssi`.

``` python
pcondit_ssi = partial(condit_ssi, tol=eps, maxiter=110)
```

Перебирая различные начальные приближения констант фазового равновесия, полученные после проверки стабильности однофазного состояния, найдем решение системы нелинейных уравнений для двухфазного состояния.

``` python
for i, kvi in enumerate(stabres.kvji):
    F1 = solve2p_FGH(kvi, yi)
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i = pr.getPT_lnphii(P, T, y2i)
    lnphi1i = pr.getPT_lnphii(P, T, y1i)
    gi = np.log(kvi) + lnphi1i - lnphi2i
    carry = (1, kvi, F1, gi)
    while pcondit_ssi(carry):
        carry = pupdate_ssi_2p(carry)
    k, kvi, F1, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        print(f'For the initial guess #{i}:\n'
              f'\ttolerance of equations: {gnorm}\n'
              f'\tnumber of iterations: {k}\n'
              f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
              f'\tphase mole fractions: {F1}, {1.-F1}')
        break
```

```{glue:} glued_out13
```

Метод последовательных подстановок нашел состояние, характеризующееся равенством летучестей соответствующих компонентов в фазах, за 99 итераций. Проверим, является ли двухфазное состояние стабильным. Для этого проведем тест стабильности для одной из фаз:

``` python
stabres = stab.run(P, T, y2i)
print(stabres)
```

```{glue:} glued_out14
```

Тест стабильности показал, что найденное решение соответствует равновесному состоянию.
````

```{code-cell} python
:tags: [remove-cell]

P = 17e6 # Pressure [Pa]
T = 68. + 273.15 # Temperature [K]
yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573]) # Mole fractions [fr.]

Pci = np.array([4.599, 4.872, 4.248, 3.796, 2.398]) * 1e6 # Critical pressures [Pa]
Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02]) # Critical temperatures [K]
wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414]) # Acentric factors
mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120]) # Molar mass [kg/gmole]
vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661]) # Volume shift parameters
dij = np.array([
    0.002689,
    0.008537, 0.001662,
    0.014748, 0.004914, 0.000866,
    0.039265, 0.021924, 0.011676, 0.006228,
]) # Binary interaction parameters
pr = pr78(Pci, Tci, wi, mwi, vsi, dij)


stab = stabilityPT(pr, method='ss', maxiter=200)
stabres = stab.run(P, T, yi)
out12 = str(stabres)

pupdate_ssi_2p = partial(update_ssi_2p, yi=yi, plnphi=partial(pr.getPT_lnphii, P=P, T=T))
pcondit_ssi = partial(condit_ssi, tol=eps, maxiter=110)

out13 = ''
for i, kvi in enumerate(stabres.kvji):
    F1 = solve2p_FGH(kvi, yi)
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i = pr.getPT_lnphii(P, T, y2i)
    lnphi1i = pr.getPT_lnphii(P, T, y1i)
    gi = np.log(kvi) + lnphi1i - lnphi2i
    carry = (1, kvi, F1, gi)
    while pcondit_ssi(carry):
        carry = pupdate_ssi_2p(carry)
    k, kvi, F1, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        out13 += (f'For the initial guess #{i}:\n'
                  f'\ttolerance of equations: {gnorm}\n'
                  f'\tnumber of iterations: {k}\n'
                  f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
                  f'\tphase mole fractions: {F1}, {1.-F1}')
        break

stabres = stab.run(P, T, y2i)
out14 = str(stabres)

glue('glued_out12', MultilineText(out12))
glue('glued_out13', MultilineText(out13))
glue('glued_out14', MultilineText(out14))
```

Таким образом, вблизи границы двухфазной области метод последовательных подстановок характеризуется медленной сходимостью. Для ускорения могут применяться различные модификации данного метода, представленные в работах \[[Mehra et al, 1983](https://doi.org/10.1002/cjce.5450610414); [Nghiem and Li, 1984](https://doi.org/10.1016/0378-3812(84)80013-8)\].

(pvt-sec-equilibrium-pt-newton)=
### Метод минимизации функции

Как уже было отмечено ранее, задачу определения равновесного состояния термодинамической системы для фиксированного количества фаз можно рассматривать как задачу локальной минимизации функции энергии Гиббса. Такой подход требует получение аналитических выражений для вектора градиента и матрицы гессиана оптимизируемой функции. Ранее в данном разделе было получено выражение элемента вектора градиента функции энергии Гиббса при использовании количеств вещества компонентов в нереференсных фазах в качестве основных переменных:

$$ g_{ji} = \ln f_{ji} - \ln f_{N_pi}, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c. $$

Выразим летучесть $i$-го компонента в $j$-й фазе через его коэффициент летучести и количество вещества:

$$ \ln f_{ji} = \ln \varphi_{ji} y_{ji} P = \ln \varphi_{ji} \frac{n_{ji}}{n_j} P = \ln \varphi_{ji} + \ln n_{ji} - \ln n_j + \ln P, \; j = 1 \, \ldots \, N_p, \; i = 1 \, \ldots \, N_c. $$

Тогда элемент вектора градиента:

$$ g_{ji} = \ln \varphi_{ji} + \ln n_{ji} - \ln n_j - \left( \ln \varphi_{N_pi} + \ln n_{N_pi} - \ln n_{N_p} \right), \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c. $$

Получим выражение для элемента матрицы гессиана:

$$ \begin{alignat}{1}
H_{jikl}
&= && \, \frac{\partial g_{ji}}{\partial n_{kl}} \\
&= && \, \frac{\partial \ln \varphi_{ji}}{\partial n_{kl}} + \frac{\partial \ln n_{ji}}{\partial n_{kl}} - \frac{\partial \ln n_j}{\partial n_{kl}} - \left( \frac{\partial \ln \varphi_{N_pi}}{\partial n_{kl}} + \frac{\partial \ln n_{N_pi}}{\partial n_{kl}} - \frac{\partial \ln n_{N_p}}{\partial n_{kl}} \right) , \\
& && \, j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c.
\end{alignat} $$

Рассмотрим и преобразуем некоторые слагаемые. Частная производная логарифма коэффициента летучести $i$-го компонента в $j$-й фазе, где $i = 1 \, \ldots \, N_c$ и $j = 1 \, \ldots \, N_p - 1$, по количесту вещества $l$-го компонента в $k$-й фазе, где $l = 1 \, \ldots \, N_c$ и $k = 1 \, \ldots \, N_p - 1$, будет зависеть от компонентного состава этой же фазы. Следовательно,

$$ \begin{align}
\frac{\partial \ln \varphi_{ji}}{\partial n_{kl}}
&= \begin{cases} 0, \; & j \neq k, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \, i = 1 \, \ldots N_c, \, l = 1 \, \ldots \, N_c, \\ \frac{\partial \ln \varphi_{ji}}{\partial n_{kl}}, \; & j = k, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c. \end{cases} \\
&= \delta_{jk} \frac{\partial \ln \varphi_{ji}}{\partial n_{jl}}, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \, i = 1 \, \ldots N_c, \, l = 1 \, \ldots \, N_c.
\end{align} $$

В выражении выше символом $\delta_{jk}, \, j = 1 \, \ldots \, N_p - 1, \, k = 1 \, \ldots \, N_p - 1,$ обозначается элемент [единичной матрицы](https://en.wikipedia.org/wiki/Identity_matrix) ([дельты Кронекера](https://en.wikipedia.org/wiki/Kronecker_delta)).

Аналитические выражения логарифмов коэффициента летучести компонента фазы по давлению, температуре и количеству вещества компонентов, находящихся в этой же фазе, с использованием уравнений состояния были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.md).

Рассмотрим частную производную логарифма коливечества вещества $i$-го компонента в $j$-й фазе по количеству вещества $l$-го компонента в $k$-й фазе:

$$ \begin{align}
\frac{\partial \ln n_{ji}}{\partial n_{kl}}
&= \begin{cases} 0, \; & j \neq k, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c, \\ \frac{1}{n_{ji}} \frac{\partial n_{ji}}{\partial n_{kl}}, \; & j = k, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c. \end{cases} \\
&= \delta_{jk} \frac{1}{n_{ji}} \frac{\partial n_{ji}}{\partial n_{jl}}, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c, \\
&= \delta_{jk} \frac{1}{n_{ji}} \delta_{il} , \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c.
\end{align} $$

Рассмотрим частную производную логарифма количества вещества $j$-й фазы по количеству вещества $l$-го компонента в $k$-й фазе:

$$ \begin{align}
\frac{\partial \ln n_j}{\partial n_{kl}}
&= \begin{cases} 0, \; & j \neq k, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c, \\ \frac{1}{n_j} \frac{\partial n_j}{\partial n_{kl}}, \; & j = k, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c. \end{cases} \\
&= \delta_{jk} \frac{1}{n_j}, \; j = 1 \, \ldots \, N_p - 1, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots N_c.
\end{align} $$

При получении данного выражения было использовано следующее преобразование частной производной логарифма количества вещества $j$-й фазы по количеству вещества $k$-го компонента в этой же фазе ($k=j$):

$$ \begin{align}
\frac{\partial \ln n_j}{\partial n_{jl}} = & \, \frac{1}{n_j} \frac{\partial n_{j}}{\partial n_{jl}} = \frac{1}{n_j} \frac{\partial}{\partial n_{jl}} \left( \sum_{i=1}^{N_c} n_{ji} \right) = \frac{1}{n_j} \sum_{i=1}^{N_c} \frac{\partial n_{ji}}{\partial n_{jl}} = \frac{1}{n_j} \sum_{i=1}^{N_c} \delta_{il} = \frac{1}{n_j}, \\ & j = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c.
\end{align} $$

Получим аналитическое выражение частной производной логарифма коэффициента летучести $i$-го компонента в $N_p$-й фазе по количеству вещества $l$-го компонента в $k$-й фазе:

$$ \frac{\partial \ln \varphi_{N_pi}}{\partial n_{kl}} = \frac{\partial \ln \varphi_{N_pi}}{\partial n_{N_pl}} \frac{\partial n_{N_pl}}{\partial n_{kl}}, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots \, N_c. $$

В свою очередь, с учетом

$$ n_{N_pl} = n_l - \sum_{j=1}^{N_p-1} n_{jl}, \; l = 1 \, \ldots \, N_c, $$

получим частную производную количества вещества $l$-го компонента в $N_p$-й фазе по количеству вещества этого же компонента в $k$-й фазе:

$$ \frac{\partial n_{N_pl}}{\partial n_{kl}} = \frac{\partial n_l}{\partial n_{kl}} - \frac{\partial}{\partial n_{kl}} \left( \sum_{j=1}^{N_p-1} n_{jl} \right) = - \sum_{j=1}^{N_p-1} \frac{\partial n_{jl}}{\partial n_{kl}} = -1, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots N_c. $$

Рассмотрим частную производную логарифма количества вещества $i$-го компонента в $N_p$-й фазе по количеству вещества $L$-го компонента в $k$-й фазе:

$$ \begin{align}
\frac{\partial \ln n_{N_pi}}{\partial n_{kl}}
&= \frac{1}{n_{N_pi}} \frac{\partial n_{N_pi}}{\partial n_{kl}} \\
&= \frac{1}{n_{N_pi}} \frac{\partial}{\partial n_{kl}} \left( n_i - \sum_{j=1}^{N_p-1} n_{ji} \right) \\
&= -\frac{1}{n_{N_pi}} \sum_{j=1}^{N_p-1} \frac{\partial n_{ji}}{\partial n_{kl}} \\
&= -\frac{1}{n_{N_pi}} \delta_{il}, \; k = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots N_c, \; l = 1 \, \ldots N_c.
\end{align} $$

Получим аналитическое выражение для частной производной логарифма количества вещества $N_p$-й фазы по количеству вещества $l$-го компонента в $k$-й фазе:

$$ \begin{align}
\frac{\partial \ln n_{N_p}}{\partial n_{kl}}
&= \frac{1}{n_{N_p}} \frac{\partial n_{N_p}}{\partial n_{kl}} \\
&= \frac{1}{n_{N_p}} \frac{\partial}{\partial n_{kl}} \left( n - \sum_{j=1}^{N_p-1} n_j \right) \\
&= - \frac{1}{n_{N_p}} \frac{\partial}{\partial n_{kl}} \left( \sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} n_{ji} \right) \\
&= - \frac{1}{n_{N_p}} \sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} \frac{\partial n_{ji}}{\partial n_{kl}} \\
&= - \frac{1}{n_{N_p}}, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots N_c.
\end{align} $$

С учетом изложенного выше получим аналитическое выражение для элемента матрицы гессиана:

$$ \begin{align}
H_{jikl}
&= \delta_{jk} \left( \frac{\partial \ln \varphi_{ji}}{\partial n_{jl}} + \frac{1}{n_j} \left( \frac{\delta_{il}}{y_{ji}} - 1 \right) \right) + \left( \frac{\partial \ln \varphi_{N_pi}}{\partial n_{N_pl}} + \frac{1}{n_{N_p}} \left( \frac{\delta_{il}}{y_{N_pi}} - 1 \right) \right) \\
&= \delta_{jk} \frac{\partial \ln f_{ji}}{\partial n_{jl}} + \frac{\partial \ln f_{N_pi}}{\partial n_{N_pl}} \\
&= \delta_{jk} \frac{\partial \ln \varphi_{ji}}{\partial n_{jl}} + \frac{\partial \ln \varphi_{N_pi}}{\partial n_{N_pl}} + \delta_{jk} \left( \frac{\delta_{il}}{n_{ji}} - \frac{1}{n_j} \right) + \frac{\delta_{il}}{n_{N_pi}} - \frac{1}{n_{N_p}} \\
&= \Phi_{jikl} + U_{jikl}, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c,
\end{align} $$

где:

$$ \begin{alignat}{1}
& \Phi_{jikl} = && \, \delta_{jk} \frac{\partial \ln \varphi_{ji}}{\partial n_{jl}} + \frac{\partial \ln \varphi_{N_pi}}{\partial n_{N_pl}}, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c, \\
& U_{jikl} = && \, \delta_{jk} \left( \frac{\delta_{il}}{n_{ji}} - \frac{1}{n_j} \right) + \frac{\delta_{il}}{n_{N_pi}} - \frac{1}{n_{N_p}}, \\ &&& j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c.
\end{alignat} $$

Для двухфазной системы, состоящей из газовой $V$ и референсной жидкой $L$ фаз, элемент матрицы гессиана:

$$ \begin{align}
H_{il}
&= \frac{\partial \ln \varphi_{Vi}}{\partial n_{Vl}} + \frac{\partial \ln \varphi_{Li}}{\partial n_{Ll}} + \delta_{il} \left( \frac{1}{n_{Vi}} + \frac{1}{n_{Li}} \right) - \left( \frac{1}{n_V} + \frac{1}{n_L} \right) \\
&= \frac{\partial \ln \varphi_{Vi}}{\partial n_{Vl}} + \frac{\partial \ln \varphi_{Li}}{\partial n_{Ll}} + \delta_{il} \frac{n_i}{n_{Vi} n_{Li}} - \frac{n}{n_V n_L} \\
&= \Phi_{il} + U_{il}, \; i = 1 \, \ldots \, N_c, \; l = 1 \, \ldots \, N_c.
\end{align} $$

где:

$$ \begin{align}
& \Phi_{il} = \frac{\partial \ln \varphi_{Vi}}{\partial n_{Vl}} + \frac{\partial \ln \varphi_{Li}}{\partial n_{Ll}}, \; i = 1 \, \ldots \, N_c, \; l = 1 \, \ldots \, N_c, \\
& U_{il} = \delta_{il} \frac{n_i}{n_{Vi} n_{Li}} - \frac{n}{n_V n_L} = \frac{1}{n_V n_L} \left( \delta_{il} \frac{n_i}{y_{Vi} y_{Li}} - n \right), \; i = 1 \, \ldots \, N_c, \; l = 1 \, \ldots \, N_c.
\end{align} $$

При использовании метода минимизации функции энергии Гиббса относительно основных переменных – вектора количеств вещества компонентов в нереференсных фазах – вектор обновления основных переменных определяется при решении следующей системы линейных уравнений:

$$ \mathbf{H} \Delta \mathbf{n} = - \mathbf{g}, $$

где $\mathbf{H} \in {\rm I\!R}^{\left(N_c \cdot \left(N_p - 1\right) \right) \times \left(N_c \cdot \left(N_p - 1\right) \right)}$ представляет собой матрицу гессиана, $\Delta \mathbf{n} \in {\rm I\!R}^{N_c \cdot \left(N_p - 1\right)}$ – вектор обновления основных переменных (количеств вещества компонентов в нереференсных фазах), $\mathbf{g} \in {\rm I\!R}^{N_c \cdot \left(N_p - 1\right)}$ – вектор значений решаемой системы нелинейных уравнений, определяющих положение стационарных точек функции энергии Гиббса (ее градиент).

Помимо вектора количеств вещества компонентов в нереференсных фазах в качестве основных переменных может выступить вектор логарифмов констант фазового равновесия компонентов в нереференсных фазах. В этом случае на каждой итерации метода Ньютона для нахождения вектора обновления основных переменных необходимо решать следующую систему линейных уравнений:

$$ \mathbf{J} \Delta \ln \mathbf{k} = - \mathbf{g}, $$

где $\Delta \ln \mathbf{k} \in {\rm I\!R}^{N_c \cdot \left(N_p - 1\right)}$ – вектор обновления основных переменных (логарифмов констант фазового равновесия компонентов в нереференсных фазах), $\mathbf{J} \in {\rm I\!R}^{\left(N_c \cdot \left(N_p - 1\right) \right) \times \left(N_c \cdot \left(N_p - 1\right) \right)}$ представляет собой якобиан – матрицу частных производных, элемент которой определяется следующим выражением:

$$ \begin{align}
J_{jikl} = & \, \frac{\partial g_{ji}}{\partial \ln K_{kl}} = \sum_{r=1}^{N_p-1} \sum_{s=1}^{N_c} \frac{\partial g_{ji}}{\partial n_{rs}} \frac{\partial n_{rs}}{\partial \ln K_{kl}} = \sum_{r=1}^{N_p-1} \sum_{s=1}^{N_c} H_{jirs} U_{rskl}^{-1}, \\
& j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c.
\end{align} $$

При преобразовании данного выражения использовалось [правило нахождения производной сложной функции от нескольких переменных](https://en.wikipedia.org/wiki/Chain_rule#General_rule:_Vector-valued_functions_with_multiple_inputs).

Покажем, что частная производная логарифма константы фазового равновесия $i$-го компонента в $j$-й фазе, $i = 1 \, \ldots \, N_c, \, j = 1 \, \ldots \, N_p - 1$, по количеству вещества $l$-го компонента в $k$-й фазе, $l = 1 \, \ldots \, N_c, \, k = 1 \, \ldots \, N_p - 1$, соответствует элементу матрицы $U_{jikl}$, представляющей собой слагаемое в полученном ранее выражении для элемента гессиана.

$$ \begin{align}
\frac{\partial \ln K_{ji}}{\partial n_{kl}}
&= \frac{\partial}{\partial n_{kl}} \left( \ln \frac{y_{ji}}{y_{N_pi}} \right) \\
&= \frac{\partial}{\partial n_{kl}} \left( \ln \frac{n_{ji} n_{N_p}}{n_j n_{N_pi}} \right) \\
&= \frac{\partial}{\partial n_{kl}} \left( \ln n_{ji} - \ln n_j - \ln n_{N_pi} + \ln n_{N_p} \right) \\
&= \frac{\partial \ln n_{ji}}{\partial n_{kl}} - \frac{\partial \ln n_j}{\partial n_{kl}} - \frac{\partial \ln n_{N_pi}}{\partial n_{kl}} + \frac{\partial \ln n_{N_p}}{\partial n_{kl}} \\
&= \delta_{jk} \frac{1}{n_{ji}} \delta_{il}  - \delta_{jk} \frac{1}{n_j} + \frac{1}{n_{N_pi}} \delta_{il} - \frac{1}{n_{N_p}} \\
&= \delta_{jk} \left( \frac{\delta_{il}}{n_{ji}} - \frac{1}{n_j} \right) + \frac{\delta_{il}}{n_{N_pi}} - \frac{1}{n_{N_p}} \\
&= U_{jikl}, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \; k = 1 \, \ldots \, N_p - 1, \; l = 1 \, \ldots \, N_c.
\end{align} $$

Следовательно, частная производная количества вещества $i$-го компонента в $j$-й фазе, $i = 1 \, \ldots \, N_c, \, j = 1 \, \ldots \, N_p - 1$, по логарифму константы фазового равновесия $l$-го компонента в $k$-й фазе, $l = 1 \, \ldots \, N_c, \, k = 1 \, \ldots \, N_p - 1$, соответствует элементу обратной матрицы $U_{jikl}^{-1}$.

Важно отметить, что матрица гессиана $H \in {\rm I\!R}^{\left(N_c \cdot \left(N_p - 1\right) \right) \times \left(N_c \cdot \left(N_p - 1\right) \right)}$ представляет собой симметричную матрицу, в отличие от матрицы якобиана $\mathbf{J} \in {\rm I\!R}^{\left(N_c \cdot \left(N_p - 1\right) \right) \times \left(N_c \cdot \left(N_p - 1\right) \right)}$. Согласно \[[Michelsen, 1982](https://doi.org/10.1016/0378-3812(82)85002-4)\], данное свойство матрицы гессиана можно учитывать для более быстрого решения системы линейных уравнений, используя подходящие алгоритмы, например, [разложение Шолески](https://en.wikipedia.org/wiki/Cholesky_decomposition) или, если матрица гессиана не является положительно определенной матрицей, модифицированное разложение Шолески, алгоритмы которого представлены в работах \[[Schnabel and Eskow, 1990](https://doi.org/10.1137/0911064)\] и \[[Schnabel and Eskow, 1999](https://doi.org/10.1137/S105262349833266X)\]. После нахождения вектора обновления количеств вещества компонентов в нереференсных фазах $\Delta \mathbf{n} \in {\rm I\!R}^{N_c \cdot \left(N_p - 1\right)}$ вектор обновления логарифмов констант фазового равновесия компонентов в нереференсных фазах $\Delta \ln \mathbf{k} \in {\rm I\!R}^{N_c \cdot \left(N_p - 1\right)}$ может быть определен с использованием следующего выражения:

$$ \Delta \ln \mathbf{k} = - \mathbf{J}^{-1} \mathbf{g} = - \left( \mathbf{H} \mathbf{U}^{-1} \right)^{-1} \mathbf{g} = - \mathbf{U} \mathbf{H}^{-1} \mathbf{g} = \mathbf{U} \left( -\mathbf{H}^{-1} \mathbf{g} \right) = \mathbf{U} \Delta \mathbf{n}. $$

В процессе преобразования использовались некоторые [свойства обратных матриц](https://en.wikipedia.org/wiki/Invertible_matrix#Other_properties).

````{margin}
```{admonition} Дополнительно
Число обусловленности некоторой матрицы $\mathbf{A}$ является ее свойством и, грубо говоря, характеризует то, насколько сильно изменится решение системы линейных уравнений $\mathbf{A} \mathbf{x} = \mathbf{b}$ при некотором изменении вектора $\mathbf{b}$. При больших значениях числа обусловленности даже небольшие изменения вектора $\mathbf{b}$ могут вызвать существенные изменения вектора $\mathbf{x}$, поэтому если число обусловленности матрицы $\mathbf{A}$ невелико, то такая матрица называется *хорошо обусловленной (well-conditioned)*, и наоборот при большом числе обусловленности матрица называется *плохо обусловленной (ill-conditioned)*. Подробнее данное свойство матриц было рассмотрено [ранее](../../0-Math/0-LAB/LAB-11-ConditionNumber.md).
```
````

Сравнивая представленные формулировки метода локальной минимизации функции энергии Гиббса, необходимо отметить, что очевидным преимуществом использования вектора количеств вещества компонентов в качестве основных переменных является отсутствие необходимости решения уравнения (или системы уравнений) Речфорда-Райса: в этом случае мольные доли компонентов в фазах и мольные доли фаз могут быть получены напрямую из вектора основных переменных. С другой стороны, преимуществом использования логарифмов констант фазового равновесия в качестве основных переменных является то обстоятельство, что матрица $\mathbf{J}$ характеризуется меньшим значением [числа обусловленности](https://en.wikipedia.org/wiki/Condition_number), по сравнению с матрицей $\mathbf{H}$, то есть матрица $\mathbf{J}$ является хорошо обусловленной \[[Petitfrere and Nichita, 2015](https://doi.org/10.1016/j.fluid.2014.11.017)\].

<!-- TODO: Показать изменение числа обусловленности матриц J и H в зависимости от давления. См.: 10.1016/j.fluid.2014.11.017. -->

<!-- TODO: Показать изменение количества итераций с ростом давления для методов последовательных подстановок, Ньютона с гессианом H и Ньютона с якобианом J. См.: 10.1016/j.fluid.2014.11.017. Показать, почему метод последовательных подстановок плохо сходится вблизи линии насыщения или критической точки. -->

(pvt-sec-equilibrium-pt-newton-init)=
#### Начальные приближения

````{margin}
```{admonition} Дополнительно
Алгоритмы оптимизации могут быть классифицированы по порядку производной целевой функции, которую они используют в расчете изменения основных переменных. Так, например, метод градиентного спуска использует производные первого порядка целевой функции (градиент), поэтому его относят к методам первого порядка. Метод Ньютона использует производные второго порядка целевой функции (гессиан), поэтому он относится к методам второго порядка. Существуют также и алгоритмы нулевого порядка, при применении которых производные вообще не используются *(derivative-free methods)*. При этом порядок алгоритма *(order of an algorithm)* в теории соотносится с порядком (скоростью) сходимости *(rate of convergence)*, однако, зачастую, это справедливо только вблизи решения (подробнее рассматривалось в [разделе про методы оптимизации функций](../../0-Math/1-OM/OM-0-Introduction.md)). Реальная же скорость сходимости зависит от начального приближения, вида оптимизируемой функции и др. На практике для оценки скорости сходимости используют [это](https://en.wikipedia.org/wiki/Rate_of_convergence#Order_estimation) выражение.
```
````

Таким образом, после получения аналитических выражений для элементов матриц гессиана и якобиана необходимо отметить, что к преимуществам использования метода Ньютона для локальной минимизации функции энергии Гиббса в процессе нахождения равновесного состояния относят его более быструю сходимость, поскольку данный метод является методом второго порядка, по сравнению с методом последовательных подстановок, относимым к методам первого порядка. Однако недостаток метода Ньютона заключается в меньшей стабильности и необходимости получения достаточно хорошего начального приближения. Как и для метода последовальных подстановок, начальное приближение для метода Ньютона может быть получено из результатов стабильности $\left(N_p - 1 \right)$-фазового состояния. Кроме того, могут быть использованы комбинированные методы, заключающиеся в выполнении нескольких итераций метода последовательных подстановок перед переключением на метод Ньютона. В качестве критериев переключения авторами работы \[[Mehra et al, 1982](https://doi.org/10.2118/9232-PA)\] были предложены следующие:

$$ \begin{cases}
\sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} g_{ji} \leq 10^{-4}, \\
\max_j \left| \frac{F_j^k - F_j^{k-1}}{F_j^{k-1}} \right| \leq 0.01 \lambda^k, \\
\sum_{j=1}^{N_p-1} L_j^2 \leq 10^{-8},
\end{cases} $$

где:

$$ \begin{align}
& g_{ji} = \ln K_{ji} + \ln \varphi_{ji} - \ln \varphi_{N_pi}, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c, \\
& L_j = \sum_{i=1}^{N_c} \left( y_{ji} - y_{N_pi} \right), \; j = 1 \, \ldots \, N_p - 1.
\end{align} $$

Несколько иные условия переключения на методы второго порядка были предложены авторами работы \[[Nghiem et al, 1983](https://doi.org/10.2118/8285-PA)\]:

$$ \begin{cases}
\frac{\left( \mathbf{g}^\top \mathbf{g} \right)^k}{\left( \mathbf{g}^\top \mathbf{g} \right)^{k-1}} > \epsilon_R, \\
\left| F_V^k - F_V^{k-1} \right| < \epsilon_V, \\
\epsilon_L < \left( \mathbf{g}^\top \mathbf{g} \right)^k < \epsilon_U, \\
0 < F_V^k < 1.
\end{cases} $$

Первые два критерия свидетельствуют о медленной сходимости метода последовательных подстановок. Третий критерий показывает диапазон длины вектора невязок, при котором, по мнению авторов, наиболее рационально осуществлять переключение на метод Ньютона. Верхняя граница диапазона предотвращает преждевременное переключение на методы второго порядка. Несмотря на то что вблизи решения порядок сходимости метода Ньютона превышает порядок сходимости метода последовательных подстановок, переключение с одного метода на другой не является целесообразным непосредственно вблизи решения, что обуславливает введение нижней границы в третьем условии. Наконец, последний критерий не допускает переключение на метод Ньютона, когда значения мольной доли одной из фаз равны нулю, принимая во внимание особенности задания начальных приближений, а также выражение для матрицы $\mathbf{U}$. Типовые значения параметров: $\epsilon_R = 0.6, \, \epsilon_V = 10^{-2}, \, \epsilon_L = 10^{-5}, \, \epsilon_U = 10^{-3}$.

(pvt-sec-equilibrium-pt-newton-examples)=
#### Примеры

Проиллюстрируем применение метода локальной минимизации энергии Гиббса для нахождения равновесного состояния системы следующим примером.

````{admonition} Пример
:class: exercise
Пусть имеется $1 \; моль$ смеси следующего мольного состава:

```{table}
:width: 100%
:widths: "1, 1, 1, 1, 1, 1"
| Компонент | $\mathrm{CH_4}$ | $\mathrm{C_2 H_6}$ | $\mathrm{C_3 H_8}$ | $\mathrm{n \mbox{-} C_4}$ | $\mathrm{C_{5+}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Мольная доля | 0.7167 | 0.0895 | 0.0917 | 0.0448 | 0.0573 |
```

<br>

Необходимо определить равновесное состояние системы при давлении $17 \; МПа$ и температуре $68 \; ^{\circ} C$.
````

````{dropdown} Решение
Зададим исходные термобарические условия и компонентный состав.

``` python
P = 17e6 # Pressure [Pa]
T = 68. + 273.15 # Temperature [K]
yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573]) # Mole fractions [fr.]
```

Для выполнения вычислений будем использовать свойства компонентов, а также проинициализированные классы с уравнением состояния и тестом стабильности при рассмотрении предыдущего примера.

Для данной термодинамической системы однофазное состояние системы не является стабильным:

``` python
stabres = stab.run(P, T, yi)
print(stabres)
```

```{glue:} glued_out15
```

Выполним расчет двухфазного равновесного состояния с использованием метода локальной минимизации энергии Гиббса. Для этого создадим функцию, которая будет принимать на вход результаты предыдущей итерации в виде кортежа и рассчитывать результаты для новой итерации. Для расчета логарифмов летучести компонентов и их частных производных по количеству вещества компонентов будем использовать метод `getPT_lnphii_Z_dnj` инициализированного класса с уравнением состояния, принимающий на вход давление (в Па), температуру (в K), компонентный состав в виде одномерного массива (размерностью `(Nc,)`) и количество вещества фазы (в моль) и возвращающий кортеж из массива логарифмов коэффициентов летучести компонентов (размерностью `(Nc,)`), коэффициент сверхсжимаемости фазы, а также матрицу частных производных логарифмов коэффициентов летучести компонентов по их количеству вещества (размерностью `(Nc, Nc)`). Для решения системы линейных уравнений будем использовать [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html).

``` python
def update_newton_2p(carry, yi, plnphi):
    k, lnkvi_k, _, H_k, U_k, gi_k = carry
    dn1i = np.linalg.solve(H_k, -gi_k)
    dlnkvi = U_k.dot(dn1i)
    lnkvi_kp1 = lnkvi_k + dlnkvi
    kvi_kp1 = np.exp(lnkvi_kp1)
    F1_kp1 = solve2p_FGH(kvi_kp1, yi)
    F2_kp1 = 1. - F1_kp1
    y2i = yi / (F1_kp1 * (kvi_kp1 - 1.) + 1.)
    y1i = y2i * kvi_kp1
    lnphi2i, Z2, dlnphi2idn2j = plnphi(yi=y2i, n=F2_kp1)
    lnphi1i, Z1, dlnphi1idn1j = plnphi(yi=y1i, n=F1_kp1)
    gi_kp1 = lnkvi_kp1 + lnphi1i - lnphi2i
    U_kp1 = (np.diagflat(yi / (y1i * y2i)) - 1.) / (F1_kp1 * F2_kp1)
    H_kp1 = dlnphi1idn1j + dlnphi2idn2j + U_kp1
    return k + 1, lnkvi_kp1, F1_kp1, H_kp1, U_kp1, gi_kp1

pupdate_newton_2p = partial(
  update_newton_2p,
  yi=yi,
  plnphi=partial(pr.getPT_lnphii_Z_dnj, P=P, T=T),
)
```

Также зададим максимальное число итераций $N_{iter}$, точность решения системы нелинейных уравнений $\epsilon$.

``` python
maxiter = 50 # Maximum number of iterations
eps = 1e-6 # Tolerance
```

Создадим функцию, которая будет принимать на вход кортеж из результатов предыдущей итерации, точность и максимальное число итераций, и возвращать необходимость расчета следующей итерации цикла локальной минимизации функции энергии Гиббса. Проиницилизируем данную функцию.

``` python
def condit_newton(carry, tol, maxiter):
    k, lnkvi, _, _, _, gi = carry
    return k < maxiter and np.linalg.norm(gi) > tol

pcondit_newton = partial(condit_newton, tol=eps, maxiter=maxiter)
```

Найдем положение локального минимума функции энергии Гиббса с использованием метода Ньютона.

``` python
for i, kvi in enumerate(stabres.kvji):
    lnkvi = np.log(kvi)
    F1 = solve2p_FGH(kvi, yi)
    F2 = 1. - F1
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i, Z2, dlnphi2idn2j = pr.getPT_lnphii_Z_dnj(P, T, y2i, F2)
    lnphi1i, Z1, dlnphi1idn1j = pr.getPT_lnphii_Z_dnj(P, T, y1i, F1)
    gi = lnkvi + lnphi1i - lnphi2i
    U = (np.diagflat(yi / (y1i * y2i)) - 1.) / (F1 * F2)
    H = dlnphi1idn1j + dlnphi2idn2j + U
    carry = (1, lnkvi, F1, H, U, gi)
    while pcondit_newton(carry):
        carry = pupdate_newton_2p(carry)
    k, lnkvi, F1, _, _, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        kvi = np.exp(lnkvi)
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        print(f'For the initial guess #{i}:\n'
              f'\ttolerance of equations: {gnorm}\n'
              f'\tnumber of iterations: {k}\n'
              f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
              f'\tphase mole fractions: {F1}, {1.-F1}')
        break
```

```{glue:} glued_out16
```

Метод Ньютона нашел состояние, характеризующееся равенством летучестей соответствующих компонентов в фазах, всего за пять итераций. Проверим, является ли это состояние стабильным. Для этого проведем тест стабильности для одной из фаз:

``` python
stabres = stab.run(P, T, y2i)
print(stabres)
```

```{glue:} glued_out17
```

Тест стабильности показал, что найденное решение соответствует равновесному состоянию.

````

```{code-cell} python
:tags: [remove-cell]

P = 17e6 # Pressure [Pa]
T = 68. + 273.15 # Temperature [K]
yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573]) # Mole fractions [fr.]

stabres = stab.run(P, T, yi)
out15 = str(stabres)

def update_newton_2p(carry, yi, plnphi):
    k, lnkvi_k, _, H_k, U_k, gi_k = carry
    dn1i = np.linalg.solve(H_k, -gi_k)
    dlnkvi = U_k.dot(dn1i)
    lnkvi_kp1 = lnkvi_k + dlnkvi
    kvi_kp1 = np.exp(lnkvi_kp1)
    F1_kp1 = solve2p_FGH(kvi_kp1, yi)
    F2_kp1 = 1. - F1_kp1
    y2i = yi / (F1_kp1 * (kvi_kp1 - 1.) + 1.)
    y1i = y2i * kvi_kp1
    lnphi2i, Z2, dlnphi2idn2j = plnphi(yi=y2i, n=F2_kp1)
    lnphi1i, Z1, dlnphi1idn1j = plnphi(yi=y1i, n=F1_kp1)
    gi_kp1 = lnkvi_kp1 + lnphi1i - lnphi2i
    U_kp1 = (np.diagflat(yi / (y1i * y2i)) - 1.) / (F1_kp1 * F2_kp1)
    H_kp1 = dlnphi1idn1j + dlnphi2idn2j + U_kp1
    return k + 1, lnkvi_kp1, F1_kp1, H_kp1, U_kp1, gi_kp1

pupdate_newton_2p = partial(
  update_newton_2p,
  yi=yi,
  plnphi=partial(pr.getPT_lnphii_Z_dnj, P=P, T=T),
)

maxiter = 50 # Maximum number of iterations
eps = 1e-6 # Tolerance

def condit_newton(carry, tol, maxiter):
    k, lnkvi, _, _, _, gi = carry
    return k < maxiter and np.linalg.norm(gi) > tol

pcondit_newton = partial(condit_newton, tol=eps, maxiter=maxiter)

out16 = ''
for i, kvi in enumerate(stabres.kvji):
    lnkvi = np.log(kvi)
    F1 = solve2p_FGH(kvi, yi)
    F2 = 1. - F1
    y2i = yi / (F1 * (kvi - 1.) + 1.)
    y1i = y2i * kvi
    lnphi2i, Z2, dlnphi2idn2j = pr.getPT_lnphii_Z_dnj(P, T, y2i, F2)
    lnphi1i, Z1, dlnphi1idn1j = pr.getPT_lnphii_Z_dnj(P, T, y1i, F1)
    gi = lnkvi + lnphi1i - lnphi2i
    U = (np.diagflat(yi / (y1i * y2i)) - 1.) / (F1 * F2)
    H = dlnphi1idn1j + dlnphi2idn2j + U
    carry = (1, lnkvi, F1, H, U, gi)
    while pcondit_newton(carry):
        carry = pupdate_newton_2p(carry)
    k, lnkvi, F1, _, _, gi = carry
    gnorm = np.linalg.norm(gi)
    if gnorm < eps:
        kvi = np.exp(lnkvi)
        y2i = yi / (F1 * (kvi - 1.) + 1.)
        y1i = y2i * kvi
        out16 += (f'For the initial guess #{i}:\n'
                  f'\ttolerance of equations: {gnorm}\n'
                  f'\tnumber of iterations: {k}\n'
                  f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
                  f'\tphase mole fractions: {F1}, {1.-F1}')
        break

stabres = stab.run(P, T, y2i)
out17 = str(stabres)

glue('glued_out15', MultilineText(out15))
glue('glued_out16', MultilineText(out16))
glue('glued_out17', MultilineText(out17))
```

В данном примере начальные приближения для метода локальной минимизации функции энергии Гиббса были получены на основе результатов теста стабильности. Пример реализации алгоритма определения равновесного состояния системы, основанного на применении метода последовательных подстановок для уточнения начальных приближений, представлен [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/flash.py).

Таким образом, в рамках данного подраздела были подробно рассмотрены различные алгоритмы расчета равновесного состояния для известных давления, температуры и компонентного состава, а также их реализации. Следующий подраздел будет посвящен нахождению равновесного состояния для известных объема, температуры и компонентного состава.

(pvt-esc-equilibrium-vt)=
## VT-термодинамика

<!-- TODO: Показать, как учитывать условие электронейтральности при расчете равновесного состояния. -->

<!-- TODO: Показать пример расчета равновесного состояния с учетом распределения ионов в полярных фазах. -->

Стоит отметить, что в данном разделе основное внимание уделялось алгоритмам нахождения равновесного состояния для различных условий. Однако оптимизация времени, затрачиваемого на работу того или иного алгоритма, зависит не только от самого алгоритма, но и от его реализации: работы с памятью, распараллеливания, векторизации вычислений. Подробнее это было рассмотрено авторами работы \[[Haugen and Beckner, 2013](https://doi.org/10.2118/163583-MS)\].

Данный раздел, был посвящен нахождению равновесного состояния для различных формулировок. В следующих разделах будут рассмотрены алгоритмы нахождения предельных состояний, характерных для многокомпонентных систем. В том числе речь пойдет про поиск линий насыщения, конденсации, а также про определение критического состояния системы, являющегося пределом фазового равновесия.
