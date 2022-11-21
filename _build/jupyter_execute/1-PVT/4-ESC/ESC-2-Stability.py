#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-esc-stability'></a>
# # Определение стабильности фазового состояния системы
# В [предыдущем разделе](ESC-1-Equilibrium.html#pvt-esc-equilibrium) рассматривались подходы к решению задачи определение равновесного состояния системы при известных давлении и температуре или давлении и энтальпии. Однако, помимо данных термодинамических параметров, на вход подавалась еще одна характеристика системы - количество фаз. На практике, количество фаз, как и их мольные доли, и мольные доли компонентнов в этих фазах, неизвестно. Следовательно, необходимо с использованием численных методов определить стабильна ли система при заданном количестве фаз. Данный раздел предназначен для рассмотрения наиболее распространенных методов проверки системы на стабильность. Задача определения стабильности фазового состояния системы исследовалась в различных научных работах, среди которых следует отметить следующие: \[[Baker et al, 1982](https://doi.org/10.2118/9806-PA); [Michelsen, 1982](https://doi.org/10.1016/0378-3812(82)85001-2); [Nghiem and Li, 1984](https://doi.org/10.1016/0378-3812(84)80013-8); [Nichita et al, 2009](https://doi.org/10.1080/10916460802686681); [Stone and Nolen, 2009](https://doi.org/10.2118/118893-MS); [Li and Firoozabadi, 2012](https://doi.org/10.2118/129844-PA); [Petitfrere and Nichita, 2014](https://doi.org/10.1016/j.fluid.2013.08.039); [Zhu et al, 2018](https://doi.org/10.2118/175060-PA); [Li and Li, 2019](https://doi.org/10.1016/j.fuel.2019.02.026)\].

# [Ранее](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium) было показано, что в процессе релаксации системы энергия Гиббса снижается до достижения своего минимума в равновесном состоянии. Следовательно, к определению стабильности фазового равновесия системы можно подойти с точки зрения минимизации энергии Гиббса. Однако, прежде, чем переходить к рассмотрению вопроса стабильности фазового состояния, необходимо изложить несколько положений теории анализа энергии Гиббса \[[Baker et al, 1982](https://doi.org/10.2118/9806-PA)\]. Некоторые из данных положений будут перекликаться с рассмотренными ранее темами, но при этом являются более обобщенными.

# Сначала необходимо сформулировать ряд определений, которые будут необходимы для последующего вывода положений анализа энергии Гиббса.
# 
# ```{prf:определение}
# :nonumber:
# Пусть имеется некоторая $j$-ая фаза, состоящая из $N_C$ компонентов, многофазной системы, при этом количество вещества компонентов в фазе $j$ определяется следующим вектором:
# +++
# $$\vec{n^j} = \begin{bmatrix} n_1^j \\ n_2^j \\ \vdots \\ n_{N_c}^j \end{bmatrix}.$$
# +++
# Если количество вещества компонентов в системе определяется вектором
# +++
# $$\vec{n} = \begin{bmatrix} n_1 \\ n_2 \\ \vdots \\ n_{N_c} \end{bmatrix},$$
# +++
# то фаза $j$ является ***приемлемой***, если
# +++
# $$ 0 < n_i^j < n_i, \; i = 1 \ldots N_c.$$
# ```
# 
# Таким образом, все дальнейшие выкладки будут касаться *приемлемых* фаз, в которых представлены все компоненты. С учетом закона сохранения масс можно записать следующее соотношение:
# 
# $$\sum_{j=1}^{N_p} n_i^j = n_i,$$
# 
# где $N_p$ – количество фаз в системе. При этом предполагается, что компоненты не вступают в реакции.

# При данных и постоянных давлении и температуре [энергия Гиббса](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy) фазы зависит от изменяющегося компонентного состава фазы, то есть можно записать:
# 
# $$G^j = G^j \left( \vec{n^j} \right).$$
# 
# Энергия Гиббса является непрерывной функцией [первого порядка](https://en.wikipedia.org/wiki/First-order). При этом, энергия Гиббса многофазной системы определяется суммой энергий Гиббса каждой фазы:
# 
# $$G = \sum_{j=1}^{N_p} G^j.$$
# 
# ```{prf:определение}
# :nonumber:
# Некоторое $\hat{K}$ состояние системы, характеризующееся $N_p$ количеством *приемлемых* фаз, удовлетворяющих закону сохранения масс, является ***равновесным***, если:
# +++
# $$G \left( \hat{K} \right) = \min_l G_l \left( \hat{L} \right),$$
# +++
# где минимум берется по всем состояниям $\hat{L}$, характеризующимся $K_p$ количеством *приемлемых* фаз, удовлетворяющих закону сохранения масс.
# +++
# ```
# 
# Более подробно данное определение рассматривалось [ранее](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium).
# 
# ```{admonition} NB
# $N_p$ может не равняться $K_p$, так как равновесное состояние определяется глобальным минимумом функции энергии Гиббса, рассчитываемой для всех возможных состояний $\hat{L}$, а $N_p$ характеризует количество фаз только в равновесном состоянии $\hat{K}$.
# ```
# 
# Экстремум функции (локальные и глобальные минимумы и максимумы функции, а также седловая точка) определяется равенством нулю производной функции. Условие нахождения глобального минимума функции рассматривалось в предыдущем [разделе](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-gem-hessian).

# Также ранее было [показано](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-gem-fugacity_equlaity), что каждый экстремум функции энергии Гиббса (*стационарное состояние*) характеризуется равенством нулю частных производных энергии Гиббса по количеству вещества $i$-го компонента в фазе $j$, что эквивалентно постоянству летучести (или химического потенциала) $i$-го компонента по всем фазам $j$.

# Центральным для анализа стабильности фазового состояния системы является положение, определяющее, что стационарное состояние системы может быть определено путем идентификации общих точек касательной [гиперплоскости](https://en.wikipedia.org/wiki/Hyperplane) к [гиперповерхности](https://en.wikipedia.org/wiki/Hypersurface), представленной функцией энергии Гиббса. Приставки "гипер" необходимы при рассмотрении $N_c$-мерного пространства. Например, для двух компонентов функция энергии Гиббса представляет собой кривую на плоскости, касательной к которой является прямая. Для трех компонентов функция энергии Гиббса представляет собой поверхность, а касательная к ней – плоскость. Для $N_c$ компонентов функция энергии Гиббса – гиперповерхность, касательная к ней – гиперплоскость. Далее под поверхностью и плоскостью будут пониматься именно гиперповерхность и гиперплоскость соответственно.
# 
# ```{prf:лемма}
# :nonumber:
# Пусть $\hat{K}$ является некоторым состоянием системы, состоящей из $N_p$ приемлемых фаз, удовлетворяющих закону сохранения масс, с известным количеством вещества компонентов в фазах $\vec{n^j}, \; j = 1 \ldots N_p$. Тогда $G \left( \hat{K} \right)$ является стационарной точкой тогда и только тогда, когда $G$ дифференциируема для всех $\vec{n^j}$ и поверхность $G$ имеет одинаковую касательную плоскость для каждого $\vec{n^j}$.
# ```
# 
# Приведем доказательство данного положения.
# 
# ```{prf:proof}
# [Уравнение касательной](https://en.wikipedia.org/wiki/Tangent) плоскости $L^j \left( \vec{r} \right)$ к поверхности $G \left( \vec{r} \right)$ в точке $\vec{n^j}$ описывается следующим выражением:
# +++
# $$ L^j \left( \vec{r} \right) = G \left( \vec{n^j} \right) + \sum_{i=1}^{N_c} \frac{\partial G}{\partial r_i} \bigg|_{n_i^j} \left( r_i - n_i^j \right).$$
# +++
# Так как энергия Гиббса является экстенсивным параметром, то для нее выполняется следующее [свойство](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy-partial_molar_observables):
# +++
# $$G \left( \vec{n^j} \right) = \sum_{i=1}^{N_c} \frac{\partial G}{\partial r_i} \bigg|_{n_i^j} n_i^j = \sum_{i=1}^{N_c} \mu_i^j n_i^j.$$
# +++
# Подставляя данное выражение в уравнение касательной поверхности, получим:
# +++
# $$ L^j \left( \vec{r} \right) = \sum_{i=1}^{N_c} \frac{\partial G}{\partial r_i} \bigg|_{n_i^j} n_i^j + \sum_{i=1}^{N_c} \frac{\partial G}{\partial r_i} \bigg|_{n_i^j} r_i - \sum_{i=1}^{N_c} \frac{\partial G}{\partial r_i} \bigg|_{n_i^j} n_i^j = \sum_{i=1}^{N_c} \frac{\partial G}{\partial r_i} \bigg|_{n_i^j} r_i = \sum_{i=1}^{N_c} \mu_i^j r_i.$$
# +++
# Поскольку для стационарного состояния выполняется условие постоянства химического потенциала компонентов, то есть химический потенциал каждого компонента не зависит от фазы, то $L^j \left( \vec{r} \right)$ также не будет зависеть от фазы $j$, следовательно, для всех $j$ в стационарном состоянии уравнение касательной плоскости к поверхности энергии Гиббса одинаково. Верно и обратное: если все $L^j$ одинаковы, то химические потенциалы компонентов постоянны, и состояние является стационарным.
# ```
# 
# ```{prf:лемма}
# :nonumber:
# Пусть $D \left( \vec{r} \right)$ является функцией разности между энергией Гиббса $G \left( \vec{r} \right)$ и функцией касательной поверхности к ней $L \left( \vec{r} \right)$:
# +++
# $$ D \left( \vec{r} \right) = G \left( \vec{r} \right) - L \left( \vec{r} \right).$$
# +++
# Тогда для равновесного состояния (которое также является стационарным):
# +++
# $$D \left( \vec{r} \right) = G \left( \vec{r} \right) - \sum_{i=1}^{N_c} \mu_i r_i.$$
# +++
# Следовательно, для равновесного состояния $D \left( \vec{n^j} \right) = 0.$ Тогда при любых значениях $\vec{r}$ в равновесном состоянии функция $D \left( \vec{r} \right)$ не принимает отрицательных значений.
# ```
# 
# Приведем доказательство данного положения.

# <a id='pvt-esc-stability-multiphase_system'></a>
# ```{prf:proof}
# Сначала докажем, что разница между энергией Гиббса системы в любом состоянии, состоящей из $K_p$ приемлемых фаз, компонентный состав которых $\vec{m^k}, \; k = 1 \ldots K_p$, и удовлетворяющих закону сохранения масс, и энергией Гиббса в стационарном состоянии с $N_p$ приемлемыми фазами, компонентный состав которых $\vec{n^j}, \; j = 1 \ldots N_p$, причем $K_p$ может быть неравно $N_p$, есть сумма $D \left( \vec{m^k} \right)$ по количеству фаз $K_p$, то есть:
# +++
# $$ \sum_{k=1}^{K_p} G \left( \vec{m^k} \right) - \sum_{j=1}^{N_p} G \left( \vec{n^j} \right) = \sum_{k=1}^{K_p} D \left( \vec{m^k} \right).$$
# +++
# Поскольку система одна и та же, то выполняется закон сохранения масс:
# +++
# $$\sum_{j=1}^{N_p} n_i^j = \sum_{k=1}^{K_p} m_i^k, \; i = 1 \ldots N_c.$$
# +++
# Для стационарного состояния характерно постоянство химического потенциала компонента $\mu_i^j$ среди всех фаз $j = 1 \ldots N_p$. То есть справедливо следующее равенство:
# +++
# $$\mu_i^j = \mu_i^1, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p.$$
# +++
# Следовательно:
# +++
# $$ \begin{align}
# \sum_{j=1}^{N_p} G \left( \vec{n^j} \right)
# &= \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} \mu_i^j n_i^j \\
# &= \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} \mu_i^1 n_i^j \\
# &= \sum_{i=1}^{N_c} \mu_i^1 \sum_{j=1}^{N_p} n_i^j \\
# &= \sum_{i=1}^{N_c} \mu_i^1 \sum_{k=1}^{K_p} m_i^k \\
# &= \sum_{k=1}^{K_p} \sum_{i=1}^{N_c} \mu_i^1 m_i^k \\
# &= \sum_{k=1}^{K_p} L \left( \vec{m^k} \right).
# \end{align} $$
# +++
# Тогда
# +++
# $$\begin{align}
# \sum_{k=1}^{K_p} G \left( \vec{m^k} \right) - \sum_{j=1}^{N_p} G \left( \vec{n^j} \right)
# & = \sum_{k=1}^{K_p} G \left( \vec{m^k} \right) - \sum_{k=1}^{K_p} L \left( \vec{m^k} \right) \\
# & = \sum_{k=1}^{K_p} \left( G \left( \vec{m^k} \right) - L \left( \vec{m^k} \right) \right) \\
# & = \sum_{k=1}^{K_p} D \left( \vec{m^k} \right).
# \end{align} $$
# ```
# 
# ```{prf:лемма}
# :nonumber:
# Пусть система находится в некотором стационарном состоянии с количестом вещества в фазах $\vec{n^j}, j = 1 \ldots N_p$, причем значение энергии Гиббса для всех значений количества вещества $\vec{r^k}$ компонентов в фазах $k, \; k = 1 \ldots K_p$ больше или равно касательной плоскости, то есть $D \left( \vec{r^k} \right) \geq 0, \; k = 1 \ldots K_p$, тогда данное стационарное состояние является равновесным.
# ```
# 
# Докажем данное утверждение.
# 
# ```{prf:proof}
# Рассмотрим любое состояние системы с количеством вещества $\vec{m^k}, \; k = 1 \ldots K_p$. Тогда из условия $D \left( \vec{r^k} \right) \geq 0, \; k = 1 \ldots K_p$ следует, что
# +++
# $$\sum_{k=1}^{K_p} D \left( \vec{m^k} \right) \geq 0.$$
# +++
# Согласно предыдущему положению, данное неравество можно записать следующим образом:
# +++
# $$ \sum_{k=1}^{K_p} G \left( \vec{m^k} \right) - \sum_{j=1}^{N_p} G \left( \vec{n^j} \right) \geq 0.$$
# +++
# Или:
# +++
# $$\sum_{j=1}^{N_p} G \left( \vec{n^j} \right) \leq \sum_{k=1}^{K_p} G \left( \vec{m^k} \right).$$
# +++
# То есть для любых $\vec{m^k}, \; k = 1 \ldots K_p$ энергия Гиббса системы получается больше, чем при $\vec{n^j}$, то есть состояние с $\vec{n^j}$ характеризуется наименьшей энергией Гиббса, следовательно, оно является равновесным.
# ```
# 
# ```{prf:определение}
# :nonumber:
# Таким образом, ***равновесным состоянием*** можно назвать такое состояние системы, при котором касательная к поверхности энергии Гиббса не пересекает ее в любых других точках.
# ```
# 
# Кроме того, необходимо отметить, что
# 
# $$\sum_{j=1}^{N_p} G \left( \vec{n^j} \right) = \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} \mu_i^j n_i^j = \sum_{j=1}^{N_p} n^j \sum_{i=1}^{N_c} \mu_i^j y_i^j = n \sum_{j=1}^{N_p} F^j \sum_{i=1}^{N_c} \mu_i^j y_i^j = n R T \sum_{j=1}^{N_p} F^j \sum_{i=1}^{N_c} y_i^j \ln f_i^j.$$
# 
# Тогда приведенная энергия Гиббса:
# 
# $$\sum_{j=1}^{N_p} \tilde{G} \left( \vec{n^j} \right) = n \sum_{j=1}^{N_p} F^j \sum_{i=1}^{N_c} y_i^j \ln f_i^j.$$

# <a id='pvt-esc-stability-ex1'></a>
# Рассмотрим следующую задачу. Пусть имеется смесь из метана и диоксида углерода при температуре $-42 \; \unicode{xB0} C$ и давлении $2 \; МПа$. Построим функцию приведенной энергии Гиббса $\tilde{G} = \frac{G}{RT}$ для $1 \; моль$ вещества в системе при различных компонентных составах этой смеси. Для расчета будем использовать [уравнение состояния Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr). Кроме того, выполним расчет равновесного состояния системы с использованием [изотермико-изобарического метода последовательных подстановок](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-ssi) для системы, в которой общая мольная доля метана составляет $0.2$.

# Для решения данной задачи импортируем нужные библиотеки.

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append('../../SupportCode/')
from PVT import eos_srk_pr, eos_sw, derivatives_eos_2param, flash_isothermal_ssi, flash_isothermal_gibbs


# Задаем исходные данные.

# In[2]:


P = 2.0 * 10**6
T = 231.15


# In[3]:


Pc = np.array([4.6, 7.4])*10**6
Tc = np.array([190.6, 304.2])
w = np.array([0.008, 0.225])
comp_type = None


# In[4]:


df_a = pd.read_excel(io='../../SupportCode/BIPCoefficients.xlsx', sheet_name='A', usecols='D:Y', skiprows=[0, 1, 2], index_col=0)
df_b = pd.read_excel(io='../../SupportCode/BIPCoefficients.xlsx', sheet_name='B', usecols='D:Y', skiprows=[0, 1, 2], index_col=0)


# In[5]:


groups = [5, 12]
Akl = df_a.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
Bkl = df_b.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
alpha_matrix = np.identity(2)


# Инициализируем класс для расчета уравнения состояния Пенга-Робинсона.

# In[6]:


mr1 = eos_srk_pr(Pc, Tc, w, comp_type, c=1, Akl=Akl, Bkl=Bkl, alpha_matrix=alpha_matrix)


# Выполняем расчет энергии Гиббса системы для каждой фазы.

# In[7]:


points = 200
z = np.linspace(1e-8, 1 - 1e-8, points).reshape(points, 1)
z = np.append(z, 1 - z, axis=1)
G1 = np.sum(z * mr1.eos_run(z, P, T, points * 'o').lnf, axis=1)
G2 = np.sum(z * mr1.eos_run(z, P, T, points * 'g').lnf, axis=1)


# Выполняем расчет равновесного состояния с использованием изотермического метода последовательных подстановок. Рассчитываем уравнения касательных для каждой фазы по летучестям (или химическим потенциалам) компонентов в равновесном состоянии. Определяем разницу между энергией Гиббса системы в каждой точке и значением касательной к ней в равновесном состоянии.

# In[8]:


flash1 = flash_isothermal_ssi(mr1, np.array([0.2, 0.8])).flash_isothermal_ssi_run(P, T, phases='og')
G = np.sum(np.array([G1, G2]) * flash1.F, axis=0)
L1 = np.sum(z * flash1.eos.lnf[0], axis=1)
L2 = np.sum(z * flash1.eos.lnf[1], axis=1)
L = np.sum(np.array([L1, L2]) * flash1.F, axis=0)
D = G - L


# In[ ]:


plt.rcParams.update({'figure.max_open_warning': False})
get_ipython().run_line_magic('matplotlib', 'widget')
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 6))
fig1.canvas.header_visible = False
ax1[0].plot(z[:,0].reshape(points), G.reshape(points), label=r'Приведенная энергия Гиббса $\sum_{j=1}^{N_p} \tilde{G} \vec{\left( r^j \right)}$')
ax1[0].plot(z[:,0].reshape(points), L1.reshape(points), label=r'Касательная в равновесном состоянии $\sum_{j=1}^{N_p} \tilde{L} \vec{ \left( r^j \right)}$')
ax1[1].plot(z[:,0].reshape(points), D.reshape(points), c='g')
ax1[1].plot([flash1.y[0][0], flash1.y[1][0]], [0.0, 0.0], '.', c='r', ms=10)

ax1[0].grid()
ax1[0].set_ylabel(r'Приведенная энергия Гиббса $\tilde{G}$')
ax1[0].set_xlabel('Мольная доля метана в системе')
ax1[0].set_xlim(0, 1)
ax1[0].legend(loc='best')
ax1[1].grid()
ax1[1].set_ylabel(r'$\sum_{j=1}^{N_p} \tilde{D} \vec{ \left( r^j \right)} = \sum_{j=1}^{N_p} \tilde{G} \vec{ \left( r^j \right)} - \sum_{j=1}^{N_p} \tilde{L} \vec{ \left( r^j \right)}$')
ax1[1].set_xlabel('Мольная доля метана в системе')
ax1[1].set_xlim(0, 1)
ax1[1].set_ylim(0, 1)

fig1.tight_layout()


# In[ ]:


flash1.comp_mole_frac


# In[ ]:


flash1.residuals


# Анализируя представленные выше расчеты и рисунок, можно сделать следующие выводы. Во-первых, в равновесном состоянии касательные $L \left( \vec{r^j} \right), \; j = 1 \ldots N_p$ к функции энергии Гиббса одинаковы, поскольку логарифм летучести каждого компонента постоянен. Во-вторых, в равновесном состоянии функция $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$ неотрицательна при любых значениях компонентного состава.

# Изложенные выше положения теории анализа энергии Гиббса можно применить для определения стабильности фаз. Например, предположим, что рассматриваемая система, в которой мольная доля метана составляет $0.2$, является однофазной. Тогда если провести касательную в точке $0.2$, то она пересечет функцию энергии Гиббса, следовательно, функция $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$ не будет неотрицательной, следовательно, данное однофазное состояние не будет являться равновесным, поэтому необходимо рассмотреть систему, в которой на одну фазу больше. Для этой системы – провести расчет равновесного состояния, по результатам которого видно, что функция $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$ неотрицательна, следовательно, данное состояние является равновесным. Аналогично, если для некоторой системы, выполнив расчет двухфазного равновесного состояния, функция $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$ не является неотрицательной, то необходимо добавить еще одну фазу и выполнить расчет равновесного состояния, как для трехфазной системы. Данную итерацию необходимо повторять до тех пор, пока не будет достигнуто равновесное состояние.

# <a id='pvt-esc-stability-ex2'></a>
# Проиллюстрируем данный алгоритм на следующем примере. Пусть имеется система с количеством вещества $1 \; моль$, состоящая из трех компонентов: метана, нормального октана и воды, причем мольные доли компонентов в системе: $0.2, \; 0.3, \; 0.5$ соответственно. С использованием изложенного выше подхода к проверке стабильности фазового сосостояния системы определим количество фаз в ней при давлении $5 \; МПа$ и температуре $70 \; \unicode{xB0} C$. Для расчета летучести компонентов будем использовать [уравнение состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw).

# Сначала зададим все исходные данные.

# In[ ]:


P = 5.0 * 10**6
T = 343.15


# In[ ]:


Pc = np.array([4.6, 2.95, 22.05])*10**6
Tc = np.array([190.6, 570.5, 647.3])
w = np.array([0.008, 0.351, 0.344])
z = np.array([0.2, 0.3, 0.5])
comp_type = np.array([1, 1, 2])


# <style type="text/css">
# .tb  {border-color:#ccc;border-spacing:0px;margin:20px auto}
# .tb td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
#   font-family:Palatino, sans-serif;font-size:14px;overflow:hidden;padding:10px 16px}
# .tb th{border-color:#ccc;border-style:solid;border-width:1px;color:#333;
#   font-family:Palatino, sans-serif;font-size:14px;font-weight:normal;padding:10px 16px}
# .tb .tb-0pky{border-color:inherit;text-align:center;vertical-align:center}
# .tb .tb-1pky{background-color:#f0f0f0;border-color:inherit;text-align:center;vertical-align:center;font-weight:bold}
# .tb .tb-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:center}
# </style>
# <table class="tb">
#     <thead>
#         <tr>
#             <th class="tb-1pky" colspan="2" rowspan="2"></th>
#             <th class="tb-1pky" colspan="4">Группы</th>
#         </tr>
#         <tr>
#             <th class="tb-abip">CH<sub>3</sub></th>
#             <th class="tb-abip">CH<sub>2</sub></th>
#             <th class="tb-abip">CH<sub>4</sub></th>
#             <th class="tb-abip">H<sub>2</sub>O</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tb-1pky" rowspan="4" width="140px">Компоненты</td>
#             <td class="tb-abip" width="80px">CH<sub>4</sub></td>
#             <td class="tb-0pky" width="146.25px">&alpha;<sub>11</sub> = 0</td>
#             <td class="tb-0pky" width="146.25px">&alpha;<sub>12</sub> = 0</td>
#             <td class="tb-0pky" width="146.25px">&alpha;<sub>13</sub> = 1</td>
#             <td class="tb-0pky" width="146.25px">&alpha;<sub>14</sub> = 0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>8</sub>H<sub>18</sub></td>
#             <td class="tb-0pky">&alpha;<sub>21</sub> =  2 / 8</td>
#             <td class="tb-0pky">&alpha;<sub>22</sub> =  6 / 8</td>
#             <td class="tb-0pky">&alpha;<sub>23</sub> =  0</td>
#             <td class="tb-0pky">&alpha;<sub>24</sub> =  0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">H<sub>2</sub>O</td>
#             <td class="tb-0pky">&alpha;<sub>31</sub> =  0</td>
#             <td class="tb-0pky">&alpha;<sub>32</sub> =  0</td>
#             <td class="tb-0pky">&alpha;<sub>33</sub> =  0</td>
#             <td class="tb-0pky">&alpha;<sub>34</sub> =  1</td>
#         </tr>
#     </tbody>
# </table>
# +++

# In[ ]:


alpha_matrix = np.array([[0.0, 0.0, 1.0, 0.0], [2/8, 6/8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


# In[ ]:


groups = [1, 2, 5, 21]
Akl = df_a.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
Bkl = df_b.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6


# In[ ]:


mr2 = eos_sw(Pc, Tc, w, comp_type, Akl=Akl, Bkl=Bkl, alpha_matrix=alpha_matrix)


# Пусть данная система является однофазной, тогда мольные доли в равновесном состоянии однофазной системы соответствуют заданным глобальным мольным долям $z_i$. Для данной системы с использованием уравнения состояния Сорейде-Уитсона выполним расчет логарифмов летучести компонентов.

# In[ ]:


eos2 = mr2.eos_run(np.array([z]), P, T, 'w', cw=0.0)


# Построим поверхность приведенной энергии Гиббса.

# In[ ]:


zi, zj = np.mgrid[1e-8:1-1e-8:50j, 1e-8:1-1e-8:50j]
mask = zi + zj <= 1
zi, zj = zi[mask], zj[mask]
points = len(zi)
zi = zi.reshape(points, 1)
zj = zj.reshape(points, 1)
zk = 1 - zi - zj
zz = np.append(zi, np.append(zj, zk, axis=1), axis=1)
zz = np.where(zz < 1e-8, 1e-8, zz)
zz = zz / np.sum(zz, axis=1).reshape(points, 1)
G = np.sum(zz * mr2.eos_run(zz, P, T, points * 'w', cw=0.0).lnf, axis=1)


# In[ ]:


from plotly import figure_factory as ff
from plotly import graph_objects as go
ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), G, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian', showscale=True,
                          title=r'$ \text{Приведенная энергия Гиббса} \; \tilde{G} \left( \vec{r} \right)$')


# Рассчитаем функцию $D \left( \vec{r^j} \right)$ и построим ее в тех же координатах.

# In[ ]:


D = G - np.sum(eos2.lnf * zz, axis=1)
fig2 = ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), D, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian',
                                 showscale=True, title=r'$\tilde{D} \left( \vec{r} \right) = \tilde{G} \left( \vec{r} \right) - \tilde{L} \left( \vec{r} \right)$')
fig2.add_trace(go.Scatterternary(a=[z[0]], b=[z[1]], c=[z[2]], mode='markers', marker={'color': 'red', 'size': 10, 'symbol': 'circle-dot'}))


# Из данного рисунка видно, что несмотря на то, что в точке предполагаемого равновесного состояния функция $D \left( \vec{r} \right)$ равняется нулю, она имеет отрицательные значения в других значениях компонентного состава. То есть касательная к поверхности энергии Гиббса пересекает ее, следовательно, сделанное предположение о том, что система находится в равновесии при наличии одной фазы, неверно. Тогда пусть система находится в равновесии при ее двухфазном состоянии. Выполним расчет равновесного состояния с использованием метода минимизации энергии Гиббса. Также посчитаем приведенную энергию Гиббса и функцию $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$.

# In[ ]:


flash2 = flash_isothermal_ssi(mr2, z).flash_isothermal_ssi_run(P, T, phases='ow', cw=0.0)


# In[ ]:


flash2.residuals


# In[ ]:


G1 = np.sum(zz * mr2.eos_run(zz, P, T, points * 'o', cw=0.0).lnf, axis=1)
G2 = np.sum(zz * mr2.eos_run(zz, P, T, points * 'w', cw=0.0).lnf, axis=1)


# In[ ]:


G = np.sum(np.array([G1, G2]) * flash2.F, axis=0)


# In[ ]:


ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), G, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian',
                          showscale=True, title=r'$ \text{Приведенная энергия Гиббса} \; \sum_{j=1}^{N_p} \tilde{G} \left( \vec{r^j} \right)$')


# In[ ]:


L = np.sum(flash2.eos.lnf[0] * zz, axis=1)
L = np.sum(np.array([L, L]) * flash2.F, axis=0)
D = G - L


# In[ ]:


fig2 = ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), D, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian', showscale=True,
                                 title=r'$\sum_{j=1}^{N_p} \tilde{D} \left( \vec{r^j} \right) = \sum_{j=1}^{N_p} \tilde{G} \left( \vec{r^j} \right) - \sum_{j=1}^{N_p} \tilde{L} \left( \vec{r^j} \right)$')
fig2.add_trace(go.Scatterternary(a=flash2.y.T[0], b=flash2.y.T[1], c=flash2.y.T[2], mode='markers', marker={'color': 'red', 'size': 15, 'symbol': 'circle-dot'}))


# Функция $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$ имеет отрицательные значения. Следовательно, рассматриваемое двухфазное состояние не является равновесным. Выполним расчет равновесного состояния, предположив наличие трех фаз в системе.

# In[ ]:


flash3 = flash_isothermal_ssi(mr2, z).flash_isothermal_ssi_run(P, T, phases='gow', cw=0.0)


# In[ ]:


flash3.residuals


# In[ ]:


G = np.sum(np.array([G1, G1, G2]) * flash3.F, axis=0)


# In[ ]:


ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), G, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian',
                          showscale=True, title=r'$ \text{Приведенная энергия Гиббса} \; \sum_{j=1}^{N_p} \tilde{G} \left( \vec{r^j} \right)$')


# In[ ]:


L = np.sum(flash3.eos.lnf[0] * zz, axis=1)
L = np.sum(np.array([L, L, L]) * flash3.F, axis=0)
D = G - L


# In[ ]:


fig2 = ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), D, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian', showscale=True,
                                 title=r'$\sum_{j=1}^{N_p} \tilde{D} \left( \vec{r^j} \right) = \sum_{j=1}^{N_p} \tilde{G} \left( \vec{r^j} \right) - \sum_{j=1}^{N_p} \tilde{L} \left( \vec{r^j} \right)$')
fig2.add_trace(go.Scatterternary(a=flash3.y.T[0], b=flash3.y.T[1], c=flash3.y.T[2], mode='markers', marker={'color': 'red', 'size': 20, 'symbol': 'circle-dot'}))


# Функция $\sum_{j=1}^{N_p} D \left( \vec{r^j} \right)$ для трехфазного состояния неотрицательна, следовательно, оно является равновесным. Красным на рисунке отмечены компонентные составы фаз:

# In[ ]:


flash3.comp_mole_frac


# In[ ]:


np.all(D >= 0)


# Таким образом, предложенный подход может быть использован для проверки стабильности фазового состояния системы и обоснования количества фаз. Однако такая графическая интерпретация данного метода, когда значения энергии Гиббса и касательной к ней находятся для всего диапазона мольных долей компонентов, достаточно неудобна. В связи с этим необходим численный подход к проверке стабильности фазового состояния системы. Пример данного подхода приводится в работе \[[Michelsen, 1982](https://doi.org/10.1016/0378-3812(82)85001-2)\].

# Рассмотрим функцию *TPD – (tangent plane distance)* в некоторой точке $\vec{n^j}, \; j = 1 \ldots N_p$. Эта функция представляет собой разницу между приведенными энергией Гиббса и касательной к ней:
# 
# $$TPD \left( \vec{n^j} \right) = \sum_{j=1}^{N_p} \tilde{D} \left( \vec{n^j} \right) = \sum_{j=1}^{N_p} \tilde{G} \left( \vec{n^j} \right) - \sum_{j=1}^{N_p} \tilde{L} \left( \vec{n^j} \right) = \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} n_i^j \ln f_i^j - \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} n_i^j \ln {f^0}_i^j. $$
# 
# С учетом доказанных ранее положений в равновесном состоянии:
# 
# $$TPD \left( \vec{n^j} \right) \geq 0.$$
# 
# Функция $TPD \left( \vec{n^j} \right)$ будет больше или равна нулю, если она будет больше или равна нулю во всех ее стационарных точках – точках, в которых ее частная производная по независимым переменным равна нулю.
# 
# Если система является однофазной, то условие стабильности фазового состояния системы можно записать следующим образом:
# 
# $$ \begin{align}
# TPD \left( \vec{n} \right) & \geq 0 \\
# \sum_{i=1}^{N_c}  n_i \left( \ln f_i - \ln f_i^0 \right) & \geq 0 \\
# \left( \ln f_i - \ln f_i^0 \right) & \geq 0, \; i = 1 \ldots N_c. \\
# \end{align} $$
# 
# То есть сумма попарных произведений координат двух векторов (их [скалярное произведение](../../0-Math/1-LAB/LAB-2-VectorOperations.html#math-lab-vector_operations)) будет точно больше или равна нулю, если координаты двух векторов будут больше или равны нулю. Тогда условие стационарности можно записать следующим образом:
# 
# $$ \begin{align}
# \frac{\partial}{\partial n_l} \left( \ln f_i - \ln f_i^0 \right) &= 0, \; i = 1 \ldots N_c \\
# \ln f_i - \ln f_i^0 &= C, \; i = 1 \ldots N_c,
# \end{align} $$
# 
# где $C$ не зависит от количества вещества $l$-го компонента. Причем, из условия стабильности следует, что $C \geq 0$. Преобразуем условие стационарности к следующему виду:
# 
# $$ \ln \phi_i + \ln y_i - \ln \phi_i^0 - \ln z_i - C = 0, \; i = 1 \ldots N_c.$$
# 
# Введем новую переменную $Y_i = y_i e^{-C}$, тогда условие стационарности:
# 
# $$ \ln Y_i + \ln \phi_i - \ln \phi_i^0 - \ln z_i = 0. $$
# 
# Условие стабильности $C \geq 0$ можно заменить на
# 
# $$\sum_{i=1}^{N_c} Y_i \leq 1,$$
# 
# где $Y_i$ – решение условия стационарности. Для решения нелинейного уравнения
# 
# $$ \ln Y_i + \ln \phi_i - \ln \phi_i^0 - \ln z_i = 0 $$
# 
# относительно $Y_i$ необходимо сначала устранить неизвестную $C$. Для этого запишем:
# 
# $$\sum_{i=1}^{N_c} y_i = 1.$$
# 
# Следовательно,
# 
# $$\sum_{i=1}^{N_c} \frac{Y_i}{e^{-C}} = 1.$$
# 
# Тогда:
# 
# $$e^{-C} = \sum_{i=1}^{N_c} Y_i.$$
# 
# Отсюда следует, что
# 
# $$y_i = \frac{Y_i}{\sum_{i=1}^{N_c} Y_i}.$$
# 
# Тогда система уравнений
# 
# $$ \begin{cases}
# \ln Y_i + \ln \phi_i \left( y_i \right) - \ln \phi_i^0 - \ln z_i = 0 \\
# y_i = \frac{Y_i}{\sum_{i=1}^{N_c} Y_i}
# \end{cases} $$
# 
# разрешима относительно $Y_i$. Для решения могут быть использованы два похода: метод последовательных подстановок и метод Ньютона.

# При реализации метода последовательных подстановок значения $Y_i$ на $j+1$ итерации рассчитываются следующим образом:
# 
# $$Y_i^{j+1} = e^{\ln z_i + \ln \phi_i^0 - \ln \phi \left( y_i^j \right)}.$$
# 
# Для [метода градиентного спуска](../../0-Math/5-OM/OM-1-GradientDescent.html#math-om-gradient) необходимо составить Якобиан:
# 
# $$ J = \frac{\partial}{\partial Y_l} \left( \ln Y_i + \ln \phi_i \left( y_i \right) - \ln \phi_i^0 - \ln z_i \right) = \frac{1}{Y_i} \frac{\partial Y_i}{\partial Y_l} + \frac{\partial \ln \phi_i}{\partial y_l} \frac{\partial y_l}{\partial Y_l} = \frac{1}{Y_i} \frac{\partial Y_i}{\partial Y_l} + \frac{\partial \ln \phi_i}{\partial n_l} \frac{\partial n_l}{\partial y_l} \frac{\partial y_l}{\partial Y_l}.$$
# 
# Частную производную количества вещества $l$-го компонента по мольной доли $l$-го компонента можно выразить следующим образом. Запишем дифференциал количества вещества $l$-го компонента:
# 
# $$ d n_l = d \left( y_l n \right) = n d y_l + y_l d n = n d y_l + y_l \sum_{k=1}^{N_c} d n_k.$$
# 
# Преобразуем данное выражение к следующему виду:
# 
# $$\begin{align}
# n d y_l &= d n_l - y_l \sum_{k=1}^{N_c} d n_k \\
# n d y_l &= \sum_{k=1}^{N_c} \left(d n_l - y_l d n_k \right) \\
# d y_l &= \frac{1}{n} \sum_{k=1}^{N_c} \left( \frac{\partial n_l}{\partial n_k} - y_l \right) d n_k.
# \end{align}$$
# 
# По определению [дифференциала функции](https://en.wikipedia.org/wiki/Differential_of_a_function#Differentials_in_several_variables):
# 
# $$ d y_l = \sum_{k=1}^{N_c} \frac{\partial y_l}{\partial n_k} d n_k.$$
# 
# Тогда:
# 
# $$\frac{\partial y_l}{\partial n_k} = \frac{1}{n} \left( \frac{\partial n_l}{\partial n_k} - y_l \right). $$
# 
# С учетом [правила нахождения производной обратной функции](https://en.wikipedia.org/wiki/Inverse_functions_and_differentiation):
# 
# $$ \frac{\partial n_k}{\partial y_l} =\frac{n}{I_{lk} - y_l}. $$
# 
# Если $k = l$, то:
# 
# $$ \frac{\partial n_l}{\partial y_l} =\frac{n}{1 - y_l}. $$
# 
# Частные производные логарифма коэффициента летучести $i$-го компонента по количеству вещества $k$-го компонента с использованием уравнений состояния были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd). Стоит также отметить, что под $n$ в данном случае понимается количества вещества фазы, что для случая однофазной системы соответствует количеству вещества системы. Получим также частную производную $y_l$ по $Y_l$:
# 
# $$ \frac{\partial y_l}{\partial Y_l} = \frac{\partial}{\partial Y_l} \left( \frac{Y_l}{\sum_{l=1}^{N_c} Y_l} \right) = \frac{\frac{\partial Y_l}{\partial Y_l} \sum_{l=1}^{N_c} Y_l - Y_l \sum_{l=1}^{N_c} \frac{\partial Y_l}{\partial Y_l}}{\left( \sum_{l=1}^{N_c} Y_l \right)^2} = \frac{1}{\sum_{l=1}^{N_c} Y_l} - \frac{Y_l}{\sum_{l=1}^{N_c} Y_l} \frac{1}{\sum_{l=1}^{N_c} Y_l} = \frac{1 - y_l}{\sum_{l=1}^{N_c} Y_l}. $$
# 
# Тогда Якобиан можно записать в следующем виде:
# 
# $$J =\frac{1}{Y_i} I_{il} + \frac{\partial \ln \phi_i}{\partial n_l} \frac{n}{\sum_{l=1}^{N_c} Y_l}.$$

# <a id='pvt-esc-stability-initial_guess'></a>
# Стоит отметить, что равенство нулю частных производных функции $TPD \left( \vec{n} \right)$ по количеству вещества компонентов характеризует не только положение точек минимума функции, но также и максимума или седловой точки. То есть приближение к тому или иному виду стационарной точки будет определяться начальным приближением. В работе \[[Michelsen, 1982](https://doi.org/10.1016/0378-3812(82)85001-2)\] предлагается использовать два набора начальных приближений констант фазового равновесия $\left\{ K_i, \; \frac{1}{K_i} \right\}$, где $K_i$ рассчитывается по [модифицированной корреляции Уилсона](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-ssi-kvalues_init). В работе \[[Li and Firoozabadi, 2012](https://doi.org/10.2118/129844-PA)\] предлагается использовать более расширенный набор начальных приближений: $\left\{ K_i, \; \frac{1}{K_i}, \; \sqrt[3]{K_i}, \; \frac{1}{\sqrt[3]{K_i}}, \; K_i^* \right\}$, где
# 
# $$ \begin{cases} \begin{align} K_i^* &= \frac{0.9}{z_i}; \\ K_l^* &= \frac{0.1}{N_c - 1} \frac{1}{z_l}, \; l \neq i. \end{align} \end{cases}$$
# 
# То есть предполагается, что один из компонентов в фазе имеет значительно большую мольную долю, чем остальные. Кроме того, различные комбинации начальных приближений констант фазового равновесия для проведения проверки стабильности фазового состояния системы рассматривались в работе \[[Imai et al, 2018](https://doi.org/10.3997/2214-4609.201802111)\].

# Таким образом, в случае однофазной системы алгоритм проверки стабильности фазового состояния заключается в инициализации констант фазого равновесия на основе мольных долей компонентов в системе $z_i$, решении уравнения стабильности относительно $Y_i$ и проверке условия стабильности – сопоставления суммы $Y_i$ с единицей.

# Для того чтобы определить стационарные точки функции $TPD$ в случае многофазной системы $N_p > 1$, необходимо ее частные производные по независимым переменным $n_i^j, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p - 1$ приравнять к нулю. Рассмотрим частную производную функции $TPD$ по $n_i^j, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p - 1$.
# 
# $$ \begin{align}
# \frac{\partial TPD}{\partial n_i^j}
# &= \frac{\partial}{\partial n_i^j} \left( \sum_{k=1}^{N_p} \sum_{l=1}^{N_c} n_l^k \ln f_l^k \right) - \frac{\partial}{\partial n_i^j} \left( \sum_{k=1}^{N_p} \sum_{l=1}^{N_c} n_l^k \ln {f^0}_l^k \right) \\
# &= g_i^j - \frac{\partial}{\partial n_i^j} \left( \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} n_l^k \ln {f^0}_l^k + \sum_{l=1}^{N_c} n_l^{N_p} \ln {f^0}_l^{N_p} \right) \\
# &= g_i^j - \left( \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} \frac{\partial n_l^k}{\partial n_i^j} \ln {f^0}_l^k + \sum_{l=1}^{N_c} \frac{\partial n_l^{N_p}}{\partial n_i^j} \ln {f^0}_l^{N_p} \right),
# \end{align} $$
# 
# где выражение для $g_i^j$ было получено [ранее](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-gem-fugacity_equlaity).
# Так как
# 
# $$ n_l^{N_p} = n z_l - \sum_{k=1}^{N_p-1} n_l^k,$$
# 
# то:
# 
# $$ \frac{\partial n_l^{N_p}}{\partial n_i^j} = - \sum_{k=1}^{N_p-1} \frac{\partial n_l^k}{\partial n_i^j}.$$
# 
# Необходимо отметить, что $\frac{\partial n_l^k}{\partial n_i^j} = 0, \; k \neq j,$ поскольку в этом случае $n_l^k$ и $n_i^j$ являются независимыми переменными. Следовательно,
# 
# $$ \frac{\partial n_l^{N_p}}{\partial n_i^j} = - \frac{\partial n_l^k}{\partial n_i^j}, \; k=j.$$
# 
# С учетом этого, частные производные $TPD$ по $n_i^j, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p - 1$:
# 
# $$ \begin{align}
# \frac{\partial TPD}{\partial n_i^j}
# &= g_i^j - \left( \sum_{l=1}^{N_c} \frac{\partial n_l^j}{\partial n_i^j} \ln {f^0}_l^k - \sum_{l=1}^{N_c} \frac{\partial n_l^j}{\partial n_i^j} \ln {f^0}_l^{N_p} \right) \\
# &= g_i^j - \left( \sum_{l=1}^{N_c} \frac{\partial n_l^j}{\partial n_i^j} \left( \ln {f^0}_l^k - \ln {f^0}_l^{N_p} \right) \right) \\
# &= g_i^j.
# \end{align} $$
# 
# Стационарные точки функции $TPD$ для многофазной системы определяются следующей системой уравнений:
# 
# $$\frac{\partial TPD}{\partial n_i^j} = \ln f_i^j - \ln f_i^{N_p} = 0, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p - 1.$$
# 
# Для решения этой системы уравнений [методом градиентного спуска](../../0-Math/5-OM/OM-1-GradientDescent.html#math-om-gradient) необходимо составить Якобиан из вторых частных производных функции $TPD$ по $n_i^j, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p - 1$ и $n_s^m, \; s = 1 \ldots N_c, \; m = 1 \ldots N_p - 1$. Данный Якобиан был получен [ранее](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-gem-jacobian) при рассмотрении определения равновесного состояния изотермическим методом минимизации энергии Гиббса.

# Стоит отметить, что в подавляющем большинстве работ \[[Baker et al, 1982](https://doi.org/10.2118/9806-PA); [Michelsen, 1982](https://doi.org/10.1016/0378-3812(82)85001-2); [Li and Firoozabadi, 2012](https://doi.org/10.2118/129844-PA); [Petitfrere et al, 2020](https://doi.org/10.1016/j.petrol.2020.107241)\] проверка стабильности многофазной системы рассматривается эквивалентной рассмотренной выше однофазной проверке стабильности любой из фаз. В качестве обоснования автором работы \[[Michelsen, 1982](https://doi.org/10.1016/0378-3812(82)85001-2)\] приводятся следующие выкладки. [Ранее](#pvt-esc-stability-multiphase_system) было показано, что разница между энергией Гиббса системы в любом другом состоянии $G \left( \vec{m^k} \right), \; k = 1 \ldots K_p,$ и энергией Гиббса системы в стационарном состоянии $G \left( \vec{n^j} \right), \; j = 1 \ldots N_p,$ есть величина $\sum_{k=1}^{K_p} D \left( \vec{m^k} \right),$ то есть:
# 
# $$ \begin{align}
# \sum_{k=1}^{K_p} G \left( \vec{m^k} \right) - \sum_{j=1}^{N_p} G \left( \vec{n^j} \right)
# &= \sum_{k=1}^{K_p} D \left( \vec{m^k} \right) \\
# &= \sum_{k=1}^{K_p} G \left( \vec{m^k} \right) - \sum_{i=1}^{N_c} n_i \mu_i^0 \\
# &= \sum_{k=1}^{K_p} \sum_{i=1}^{N_c} n_i^k \mu_i^k - \sum_{i=1}^{N_c} n_i \mu_i^0 \\
# &= \sum_{k=1}^{K_p} \sum_{i=1}^{N_c} n_i^k \mu_i^k - \sum_{k=1}^{K_p} \sum_{i=1}^{N_c} n_i^k \mu_i^0 \\
# &= \sum_{k=1}^{K_p} \sum_{i=1}^{N_c} n_i^k \left( \mu_i^k - \mu_i^0 \right) \\
# &= \sum_{k=1}^{K_p} n^k \sum_{i=1}^{N_c} y_i^k \left( \mu_i^k - \mu_i^0 \right) \\
# &= R T \sum_{k=1}^{K_p} n^k TPD \left( \vec{y^k} \right).
# \end{align} $$
# 
# Затем, анализируя данное выражение, автор приходит к выводу о том, что сумма $\sum_{k=1}^{K_p} D \left( \vec{m^k} \right)$ будет отрицательной, если хотя бы один из $TPD \left( \vec{y^k} \right)$ будет отрицательным, следовательно, условия $TPD \left( \vec{y^k} \right) \geq 0$ достаточно для того, чтобы рассматриваемое стационарное состояние $G \left( \vec{n^j} \right), \; j = 1 \ldots N_p$ являлось равновесным. Во-первых, это справедливо в том случае, когда функция $D \left( \vec{m^k} \right) = G \left( \vec{m^k} \right) - L \left( \vec{m^k} \right)$ не будет зависеть от выбранной фазы $\vec{m^k}$, то есть в том случае, если расчет энергии Гиббса для каждого состояния фазы $G \left( \vec{m^k} \right)$ не зависит от выбранной фазы. Такая ситуация наблюдается, когда расчет логарифма летучести не зависит от выбранной фазы, например, при использовании [уравнений состояния Суаве-Редлиха-Квонга или Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr). В этом случае, действительно, проверка стабильности многофазной системы математически эквивалентна проверке стабильности любой из ее фаз. Однако для [уравнения состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw) данное предположение неверно, поскольку летучести компонентов в водной и неводной фазах рассчитываются по-разному (используются разные зависимости коэффициентов попарного взаимодействия). Кроме того, в общем случае условие минимимальной энергии Гиббса (как и условие максимальной энтропии) применяется для изолированной системы, то есть [второе начало термодинамики](../1-TD/TD-6-Entropy.html#pvt-td-entropy-second_law-entropy) формулируется для всей изолированной системы, для которой, действительно, в процессе релаксации энтропия увеличивается (а энергия Гиббса снижается), однако в то же время энтропия подсистем в результате обмена энергией, частицами может снижаться, поскольку они не являются изолированными системами. Таким образом, в общем случае проверку стабильности системы необходимо рассматривать с точки зрения неотрицательности суммы $\sum_{j=1}^{N_p} D \left( \vec{m^k} \right)$.

# Проиллюстрируем данный вывод на следующем примере. Ранее было показано, что трехфазное состояние системы из метана, нормального октана и воды является равновесным, поскольку в этом состоянии касательная к энергии Гиббса $\sum_{j=1}^{N_p} G \left( \vec{m^k} \right)$ не пересекает саму функцию энергии Гиббса, то есть функция $\sum_{j=1}^{N_p} D \left( \vec{m^k} \right)$ неотрицательна. Построим функцию $ D \left( \vec{m^k} \right)$ для водной фазы.

# In[ ]:


Dw = G2 - L
fig2 = ff.create_ternary_contour(np.stack((zz[:,0], zz[:,1], zz[:,2])), Dw, pole_labels=['$CH_4$', '$C_8H_{18}$', '$H_2O$'], ncontours=20, colorscale='Viridis', interp_mode='cartesian', showscale=True,
                                 title=r'$\tilde{D} \left( \vec{r^w} \right) = \tilde{G} \left( \vec{r^w} \right) - \tilde{L} \left( \vec{r^w} \right)$')
fig2.add_trace(go.Scatterternary(a=[flash3.y[-1][0]], b=[flash3.y[-1][1]], c=[flash3.y[-1][2]], mode='markers', marker={'color': 'red', 'size': 20, 'symbol': 'circle-dot'}))


# На данном рисунке видно, что касательная, проведенная к энергии Гиббса в точке равновесного состояния системы, пересекает поверхность энергии Гиббса для водной фазы. При этом, минимальная энергия Гиббса водной фазы характеризуется следующим компонентным составом:

# In[ ]:


zz[np.argmin(G2)]


# То есть в водной фазе минимальная энергия Гиббса получается в точке, где мольная доля воды $0.2$, а нормального октана – $0.8$. Это свидетельствует о том, что в данном случае нельзя определять стабильность системы по одной лишь фазе, поскольку эта подсистема не является изолированной системой, то есть в процессе релаксации всей системы энтропия подсистемы может снижаться, а энергия Гиббса увеличиваться.

# Таким образом, в общем виде определение стабильности фазового равновесия многофазной системы основывается на определении стационарных точек функции $TPD \left( \vec{n^j} \right)$ независимо от найденного компонентного состава фаз в ходе алгоритмов итерационного расчета равновесного состояния. После определения стационарных точек проверяются значения функции $TPD \left( \vec{n^j} \right)$ в них. Если они близки к нулю, то данное состояние считается равновесным. В процессе применения данного подхода ([метод градиентного спуска](../../0-Math/5-OM/OM-1-GradientDescent.html#math-om-gradient)) к нахождению стационарных точек функции $TPD \left( \vec{n^j} \right)$ могут получаться отрицательные количества вещества компонентов (или отрицательные мольные доли компонентов в референсной фазе), поскольку значения $n_i^j$ не ограничены. В связи с этим, рекомендуется несколько первых итераций нахождения $\vec{n^j}$ выполнить методом последовательных подстановок для локализации стационарной точки ближайшей к начальному приближению. Тогда определение стабильности фазового равновесия многофазной системы будет эквивалентно нахождению равновесного состояния методом минимизации энергии Гиббса с предварительными итерациями метода последовательных подстановок для всех [начальных приближений](#pvt-esc-stability-initial_guess) и проверкой значений функции $TPD \left( \vec{n^j} \right)$ для каждого найденного решения. Подробнее критерии переключения от метода последовательных подстановок и методу минимизации энергии Гиббса рассматривались в [предыдущем разделе](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-gem-switch_criteria). Кроме того, для повышения стабильности [метода градиентного спуска](../../0-Math/5-OM/OM-1-GradientDescent.html#math-om-gradient) при решении задачи определения стабильности фазового состояния авторами работ \[[Petitfrere and Nichita, 2014](https://doi.org/10.1016/j.fluid.2013.08.039); [Pan et al, 2019](https://doi.org/10.1021/acs.iecr.8b05229)\] рекомендуется использовать рассмотренный ранее [метод доверительной области](../../0-Math/5-OM/OM-4-TR.html#math-om-tr). В качестве критериев переключения для проверки стабильности однофазным методом авторы работы \[[Pan et al, 2019](https://doi.org/10.1021/acs.iecr.8b05229)\] приводят следующие:
# 
# $$Y_i < 0 \; or \; TPD^{k+1} > TPD^{k},$$
# 
# где $k$ – номер итерации. Для проверки стабильности многфазной системы критерии переключения с метода Ньютона на метод доверительной области аналогичны рассмотренным [ранее](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-gem-switch_criteria) критериям переключения с метода Ньютона на метод доверительной области для задачи определения равновесного состояния.
# 
# С использованием изложенного выше алгоритма выполним проверку стабильности фазового состояния для рассмотренного ранее [примера](#pvt-esc-stability-ex2). Для этого создадим класс для определения стабильности фазового состояния системы.

# In[ ]:


class equilibrium_isothermal(flash_isothermal_gibbs):
    def __init__(self, mr, eos_ders, z, stab_update_kv=False, stab_max_phases=3, stab_onephase_only=False, stab_kv_init_levels=range(0, 5, 1),
                 stab_max_iter=15, stab_eps=1e-8, stab_ssi_max_iter=10, stab_include_water=False, stab_onephase_calc_condition=False, **kwargs):
        super(equilibrium_isothermal, self).__init__(mr, eos_ders, z, **kwargs)
        self.stab_onephase_only = stab_onephase_only
        self.stab_kv_init_levels = stab_kv_init_levels
        self.stab_max_iter = stab_max_iter
        self.stab_ssi_max_iter = stab_ssi_max_iter
        self.stab_eps = stab_eps
        self.stab_update_kv = stab_update_kv
        self.stab_max_phases = stab_max_phases
        self.stab_onephase_calc_condition = stab_onephase_calc_condition
        if stab_include_water or mr.sw:
            self.stab_all_phases = ['g', 'w', 'o']
            self.stab_phases = ['g', 'w']
        else:
            self.stab_all_phases = ['g', 'o']
            self.stab_phases = ['g']
        if not stab_update_kv:
            self.stab_phase_states = self.stab_phases.copy()
            p1 = 'g' if self.stab_phases[-1] == 'w' else 'w'
            for i in range(0, stab_max_phases - 1, 1):
                self.stab_phase_states.append(p1 + i * 'o' + self.stab_phases[-1])
        pass

    @staticmethod
    def calc_tpd(flash, flash0):
        return np.sum(flash.F * flash.y * flash.eos.lnf - flash0.F * flash0.y * flash0.eos.lnf)

    @staticmethod
    def calc_stab_onephase_jacobian(Y, ders, n):
        return np.identity(Y.shape[1]) / Y + ders.dlnphi_dnk * n / np.sum(Y)

    @staticmethod
    def calc_stab_onephase_equation(Y, lnphi, z, lnphi0):
        return np.log(Y) + lnphi - np.log(z) - lnphi0

    def calc_stab_onephase(self, P, T, state, flash0, checking_levels, **kwargs):
        check_phases = self.stab_all_phases.copy()
        for phase in self.stab_phases:
            if phase in state and phase in check_phases:
                check_phases.remove(phase)
        # print(state, 'stab_phases', self.stab_phases,  'check_phases', check_phases)
        if self.stab_update_kv:
            Ycurr = None
            PHcurr = None
            KVcurr = None
        for new_phase in check_phases:
            phases = new_phase + state[-1]
            for level in checking_levels:
                kv = self.calc_kv_init(P, T, phases, level)
                # print(state, phases, level, 'kv0', kv)
                Y = kv * flash0.y[-1]
                # print(state, phases, level, 'Y0', Y)
                y = Y / np.sum(Y)
                # print(state, phases, level, 'y0', y)
                eos = self.mr.eos_run(y, P, T, new_phase, **kwargs)
                residuals = self.calc_stab_onephase_equation(Y, eos.lnphi[0], flash0.y[-1], flash0.eos.lnphi[-1]).reshape((self.mr.Nc, 1))
                i = 0
                while np.any(np.abs(residuals) > self.stab_eps) and i < self.stab_max_iter:
                    if i < self.stab_ssi_max_iter:
                        Y = np.array([np.exp(np.log(flash0.y[-1]) + flash0.eos.lnphi[-1] - eos.lnphi[0])])
                    else:
                        nj = np.array([flash0.F[-1]]) * self.mr.n
                        Y = Y - np.linalg.inv(self.calc_stab_onephase_jacobian(Y, self.eos_ders(self.mr, eos, nj, der_nk=True).get('dlnphi_dnk'), nj)).dot(residuals).reshape((1, self.mr.Nc))
                    y = Y / np.sum(Y)
                    # print('iter', i, 'y', y)
                    eos = self.mr.eos_run(y, P, T, new_phase, **kwargs)
                    residuals = self.calc_stab_onephase_equation(Y, eos.lnphi[0], flash0.y[-1], flash0.eos.lnphi[-1]).reshape((self.mr.Nc, 1))
                    i += 1
                Ysum = np.sum(Y)
                # print(state, phases, level, 'Ysum', Ysum, 'iters', i, 'max_residuals', np.max(np.abs(residuals)))
                if self.stab_onephase_calc_condition:
                    nj = np.array([flash0.F[-1]]) * self.mr.n
                    condition = 1 + np.sum(self.stab_eps / np.diag(self.calc_stab_onephase_jacobian(Y, self.eos_ders(self.mr, eos, nj, der_nk=True).get('dlnphi_dnk'), nj)[0]))
                else:
                    condition = 1 + self.stab_eps
                # print('condition', condition)
                # if i < self.stab_max_iter and Ysum > condition:
                if Ysum > condition:
                    if not self.stab_update_kv:
                        return False
                    else:
                        if Ycurr is None:
                            Ycurr = Ysum
                            PHcurr = phases
                            if new_phase == 'g' or new_phase == 'w' or 'o' in state:
                                KVcurr = Y / flash0.y[-1]
                            elif new_phase == 'o':
                                KVcurr = flash0.y[-1] / Y
                        elif Ycurr < Ysum:
                            Ycurr = Ysum
                            PHcurr = phases
                            if new_phase == 'g' or new_phase == 'w' or 'o' in state:
                                KVcurr = Y / flash0.y[-1]
                            elif new_phase == 'o':
                                KVcurr = flash0.y[-1] / Y
        if not self.stab_update_kv:
            return True
        elif PHcurr is None:
            return True, None, self.stab_kv_init_levels[0]
        else:
            return False, PHcurr, KVcurr if len(state) == 1 else np.append(flash0.kv, KVcurr, axis=0)

    def calc_stab_multiphase(self, P, T, state, flash0, checking_levels, **kwargs):
        if self.stab_update_kv:
            tpds = []
            flashs = []
        for level in checking_levels:
            flash = self.flash_isothermal_gibbs_run(P, T, state, kv0=level, **kwargs)
            if not flash.isnan:
                tpd = self.calc_tpd(flash, flash0)
                if tpd < -self.gibbs_eps:
                    if not self.stab_update_kv:
                        return False
                    else:
                        tpds.append(tpd)
                        flashs.append(flash)
        if not self.stab_update_kv:
            return True
        elif tpds:
            return False, flashs[np.argmin(tpds)]
        else:
            return True, flash0

    def equilibrium_isothermal_run(self, P, T, **kwargs):
        states_checked = {}
        if not self.stab_update_kv:
            for state in self.stab_phase_states:
                for i, kv0 in enumerate(self.stab_kv_init_levels):
                    flash0 = self.flash_isothermal_gibbs_run(P, T, state, kv0, **kwargs)
                    if not flash0.isnan:
                        break
                else:
                    continue
                checking_levels = np.delete(self.stab_kv_init_levels, i).tolist()
                if self.stab_onephase_only or len(state) == 1:
                    stability = self.calc_stab_onephase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                else:
                    stability = self.calc_stab_multiphase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                if stability:
                    break
        else:
            i = 0
            state = self.stab_phases[i]
            kv0 = self.stab_kv_init_levels[0]
            flash0 = None
            stability = False
            while not stability or len(state) <= self.stab_max_phases:
                # print('state', state)
                flash0 = self.flash_isothermal_gibbs_run(P, T, state, kv0, **kwargs)
                # print(state, 'flash0.isnan', flash0.isnan)
                # print('kv0', kv0)
                # print('flash0_res', np.max(np.abs(flash0.residuals)), 'flash0_gibbs_it', flash0.it, 'flash0_ssi_it', flash0.ssi_it)
                if flash0.isnan:
                    for j, kv0 in enumerate(self.stab_kv_init_levels):
                        flash0 = self.flash_isothermal_gibbs_run(P, T, state, kv0, **kwargs)
                        if not flash0.isnan:
                            break
                    else:
                        state = state[:-1] + 'o' + state[-1]
                        kv0 = self.stab_kv_init_levels[0]
                        continue
                    checking_levels = np.delete(self.stab_kv_init_levels, j).tolist()
                else:
                    checking_levels = self.stab_kv_init_levels
                if len(state) == 1 or self.stab_onephase_only:
                    stability, splitted_phases, kv0 = self.calc_stab_onephase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                    # print('state', state, 'stability', stability)
                    if not stability:
                        state = state[:-1] + splitted_phases
                else:
                    stability, flash = self.calc_stab_multiphase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                    if isinstance(kv0, int):
                        kv0 = self.calc_kv_init(P, T, state, kv0)
                    state = state[:-1] + 'o' + state[-1]
                    kv0 = np.append(kv0, [flash.y[-1] / flash.y[-2]], 0)
                if len(state) > self.stab_max_phases and i < len(self.stab_phases) - 1:
                    i += 1
                    state = self.stab_phases[i]
                    kv0 = self.stab_kv_init_levels[0]
                if stability:
                    break
        flash0.states_checked = states_checked
        return flash0


# In[ ]:


equil1 = equilibrium_isothermal(mr2, derivatives_eos_2param, z, ssi_switch=True)


# Проверка стабильности фазового состояния будет проводиться для следующих вариантов фазового состояния данной системы.

# In[ ]:


equil1.stab_phase_states


# Если ни одно из них не удовлетворит условиям стабильности, следовательно, рассматриваемая система будет являться трехфазной, согласно [правилу фаз Гиббса](../1-TD/TD-11-GibbsPhaseRule.html#pvt-td-gibbs_phase_rule).

# In[ ]:


flash4 = equil1.equilibrium_isothermal_run(P, T, cw=0.0)


# In[ ]:


flash4.states_checked


# Все возможные фазовые состояния системы были рассмотрены и оказались нестабильными. Следовательно, система представлена тремя фазами.

# In[ ]:


flash4.comp_mole_frac


# In[ ]:


flash4.residuals


# Необходимо отметить, что основной сложностью при проведении расчетов определения равновесного состояния методами, рассмотренными [ранее](ESC-1-Equilibrium.html#pvt-esc-equilibrium), является задание достаточно приближенных к глобальному минимуму функции констант фазового равновесия. Использование рассмотренных [корреляций](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-ssi-kvalues_init) для расчета равновесного состояния хоть и используется достаточно широко, но может приводить к сходимости алгоритма на локальном минимуме функции. Для решения данной проблемы может использоваться инициализация констант фазового равновесия, полученных после проверки стабильности фазового состояния, в ходе которой $TPD < 0$. То есть в процессе проверки предполагаемого фазового состояния системы определяются стационарные точки функции энергии Гиббса с использованием различных [начальных приближений констант фазового равновесия](#pvt-esc-stability-initial_guess). Если найденная стационарная точка характеризуется более отрицательной энергией Гиббса (или отрицательностью функции $TPD$), то делается вывод о том, что предполагаемое фазовое состояние системы нестабильно, причем значения компонентного состава в такой найденной стационарной точке можно использовать для генерации начальных приближений констант фазового равновесия для проведения расчета равновесного состояния с другим предположением количества фаз в системе. Такая идея использовалась авторами работы \[[Hoteit and Firoozabadi, 2006](https://doi.org/10.1002/aic.10908)\] при проверке стабильности однофазной системе. После проверки стабильности однофазной системы, если в качестве мнимой фазы рассматривалась газовая, и функция $TPD < 0$, то константы фазового равновесия для следующей двухфазной итерации расчета равновесного состояния предлагалось рассчитывать по следующей формуле:
# 
# $$ K_i^{go} = \frac{Y_i}{z_i}.$$
# 
# Если же в качестве мнимой фазы рассматривалась жидкая углеводородная, и функция $TPD < 0$, то:
# 
# $$ K_i^{og} = \frac{z_i}{Y_i}.$$
# 
# Если функция $TPD$ отрицательна в обоих предположениях, то константы фазового равновесия берутся из того предположения фазового состояния, при котором функция $TPD$ наименьшая. В случае проверки стабильности многофазной системы для новой добавленной жидкой углеводородной фазы константы фазового равновесия рассчитываются с использованием следующего выражения:
# 
# $${K_i^{trial}}^{k+1} = \frac{ y_i^R }{{ y_i^{trial}}^k },$$
# 
# где $k$ – номер итерации, то есть константы фазого равновесия добавленной фазы на $k+1$ итерации получаются из компонентного состава найденной стационарной точки с минимальным $TPD$ на предыдущей итерации.
# 
# Зачастую на практике пластовые системы в нефтегазовой отрасли редко состоят из более, чем трех фаз. Однако возможны ситуации, когда жидкая углеводородная фаза разделяется на две. В этом случае, при наличии водной и газовой фаз система будет состоять из четырех фаз. Рассмотрим такой пример. Пусть имеется система, состоящая из следующих компонентов: метана, октана $C_8H_{18}$, $FC_{45}$, и воды в следующем мольном составе: $0.35, \; 0.25, \; 0.15, \; 0.25$ соответственно при давлении $20 \; МПа$ и температуре $50 \; \unicode{xB0} C$. Необходимо определить количество фаз в системе. Корреляции расчета коэффициента попарного взаимодействия [уравнения состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw) не подходят для компонента $FC_{45}$; также не подходит корреляция [GCM](../2-EOS/EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip), так как детальная структура углеводородных псевдокомпонентов неизвестна, поэтому в данном случае будем использовать формулу расчета коэффициентов попарного взаимодействия через критические объемы, рассмотренную [ранее](../2-EOS/EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-bip). Следовательно, можно использовать проверку стабильности многофазной системы однофазным методом.
# 
# Для начала зададим исходные данные.

# In[ ]:


P = 20.0 * 10**6
T = 323.15


# In[ ]:


Pc = np.array([4.600155, 2.950584, 0.7234605, 22.04832]) * 10**6
Tc = np.array([190.6, 570.5, 957.8, 647.3])
vc = np.array([0.099, 0.421, 1.955, 0.056]) * 10**(-3)
w = np.array([0.008, 0.351327, 1.329531, 0.344])
z = np.array([0.35, 0.25, 0.15, 0.25])
comp_type = np.array([0, 0, 0, 2])


# In[ ]:


mr3 = eos_srk_pr(Pc, Tc, w, comp_type, c=1, vc=vc)


# In[ ]:


equil2 = equilibrium_isothermal(mr3, derivatives_eos_2param, z, ssi_switch=True, stab_update_kv=True, stab_include_water=True, stab_onephase_only=True, stab_max_phases=4, ssi_use_opt=True)


# In[ ]:


flash5 = equil2.equilibrium_isothermal_run(P, T)
flash5.comp_mole_frac


# In[ ]:


flash5.states_checked


# Таким образом, для заданного компонентного состава равновесное условие соответствует четырехфазному состоянию.

# Выше представлен подход, при котором неотрицательность функции $TPD \left( \vec{n^j} \right)$ проверяется в ее стационарных точках. Недостаток данного метода заключается в необходимости рассмотрения достаточного количества вариантов инициализации констант фазового равновесия для подтверждения стабильности или нестабильности системы. При этом, зачастую система уравнений из равенства летучестей компонентов (то есть определение стационарной точки для многофазной системы) просчитывается для большого количества вариантов инициализации констант фазового равновесия, а отрицательное значение функции $TPD \left( \vec{n^j} \right)$ наблюдается только в одном – двух из этих вариантов, следовательно, значительная доля вариантов инициализации констант фазового равновесия просто сводится к ранее предположенному равновесному состоянию системы. Все это в совокупности приводит к тому, что на проверку стабильности фазового состояния системы, особенно в случае, когда летучести компонентов в фазах считаются по-разному, требуется существенное количество вычислительных ресурсов и времени. С другой стороны, функция $TPD \left( \vec{n^j} \right)$ будет больше или равна нулю, если она больше или равна нулю в своем глобальном минимуме, то есть вместо проверки значения функции $TPD \left( \vec{n^j} \right)$ во всех стационарных точках достаточно проверить значение функции $TPD \left( \vec{n^j} \right)$ в ее глобальном минимуме. Поиск глобального минимума функции в качестве способа определения равновесного состояния и его стабильности рассматривался в работе \[[Nichita et al, 2004](https://doi.org/10.2118/04-05-TN2)\]. Авторами предлагается использовать туннельный алгоритм оптимизации функции, рассмотренный [ранее](../../0-Math/5-OM/OM-5-Tunneling.html#math-om-tunneling). Подробнее остановимся на этом подходе к оптимизации.

# In[ ]:





# In[ ]:





# In[ ]:





# Выше рассматривались подходы для определения стабильности фазового состояния системы в изотермическом подходе, когда температура системы постоянна. В случае [изоэнтальпийно-изобарической формулировки равновесного состояния](ESC-1-Equilibrium.html#pvt-esc-equilibrium-isoenthalpic) ...

# In[ ]:





# In[ ]:




