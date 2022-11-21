#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-parameters-internal_energy'></a>
# # Внутренняя энергия

# Рассмотрим систему, состоящую из одной фазы и нескольких компонентов. В соответствии с [правилом фаз Гиббса](../1-TD/TD-11-GibbsPhaseRule.html#pvt-td-gibbs_phase_rule), состояние такой системы определяется интенсивными параметрами (например, давлением и температурой) и количеством вещества компонентов. Тогда внутренняя энергия:
# 
# $$ U = U \left( P, T, n_i \right). $$
# 
# С учетом этого дифференциал внутренней энергии:
# 
# $$ dU = \left( \frac{\partial U}{\partial T} \right)_{P, n_i} dT + \left( \frac{\partial U}{\partial P} \right)_{T, n_i} dP + \sum_i \left( \frac{\partial U}{\partial n_i} \right)_{P, T} dn_i. $$
# 
# По определению [изобарной теплоемкости](../1-TD/TD-4-HeatCapacity.html#pvt-td-heat_capacity-isobaric):
# 
# $$ C_P = \left( \frac{\partial U}{\partial T} \right)_{P} + P \left( \frac{\partial V}{\partial T} \right)_P. $$
# 
# Тогда первое слагаемое дифференциала внутренней энергии:
# 
# $$ \left( \frac{\partial U}{\partial T} \right)_{P, n_i} = C_P -  P \left( \frac{\partial V}{\partial T} \right)_{P, n_i}. $$
# 
# Для системы с постоянным количеством частиц [можно](../1-TD/TD-6-Entropy.html#pvt-td-entropy-thermodynamic_identity) записать следующее выражение:
# 
# $$ dU = T dS - P dV. $$
# 
# Разделим левую и правую части на $dP$ и будем рассматривать изотермический процесс с постоянном количеством вещества компонентов:
# 
# $$ \left( \frac{\partial U}{\partial P} \right)_{T, n_i} = T \left( \frac{\partial S}{\partial P} \right)_{T, n_i} - P \left( \frac{\partial V}{\partial P} \right)_{T, n_i}. $$
# 
# С учетом [четвертого соотношения Максвелла](../1-TD/TD-13-MaxwellRelations.html#pvt-td-maxwell_relations-fourth) данное выражение преобразуется к следующему:
# 
# $$ \left( \frac{\partial U}{\partial P} \right)_{T, n_i} = - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} - P \left( \frac{\partial V}{\partial P} \right)_{T, n_i}. $$
# 
# Следовательно, второе слагаемое выражения для дифференциала внутренней энергии:
# 
# $$ \left( \frac{\partial U}{\partial P} \right)_{T, n_i} dP = - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} dP - P \left( \frac{\partial V}{\partial P} \right)_{T, n_i} dP. $$
# 
# Если рассматривать процесс с постоянным количеством частиц, то дифференциал давления $P = P \left( V, T, n_i \right)$ можно записать следующим образом:
# 
# $$ dP = \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV + \left( \frac{\partial P}{\partial T} \right)_{V, n_i} dT. $$
# 
# Тогда второе слагаемое в выражении для дифференциала внутренней энергии:
# 
# $$ \begin{align}
# \left( \frac{\partial U}{\partial P} \right)_{T, n_i} dP = 
# &- T \left( \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV + \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial T} \right)_{V, n_i} dT \right) \\
# & - P \left( \left( \frac{\partial V}{\partial P} \right)_{T, n_i} \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV + \left( \frac{\partial V}{\partial P} \right)_{T, n_i} \left( \frac{\partial P}{\partial T} \right)_{V, n_i} dT \right).
# \end{align} $$
# 
# Упростим полученное выражение. Для частных производных выполняется следующее [правило](https://en.wikipedia.org/wiki/Triple_product_rule):
# 
# $$ \left( \frac{\partial x}{\partial y} \right)_z \left( \frac{\partial y}{\partial z} \right)_x \left( \frac{\partial z}{\partial x} \right)_y = -1. $$
# 
# Для рассматриваемых термодинамических параметров его можно записать в следующем виде:
# 
# $$ \left( \frac{\partial V}{\partial T} \right)_P \left( \frac{\partial T}{\partial P} \right)_V \left( \frac{\partial P}{\partial V} \right)_T = -1. $$
# 
# Следовательно,
# 
# $$ \left( \frac{\partial V}{\partial T} \right)_P \left( \frac{\partial P}{\partial V} \right)_T = - \left( \frac{\partial P}{\partial T} \right)_V. $$
# 
# Кроме того, применяя [правило нахождения производной обратной функции](https://en.wikipedia.org/wiki/Inverse_functions_and_differentiation), получим
# 
# $$ \left( \frac{\partial V}{\partial P} \right)_T \left( \frac{\partial P}{\partial T} \right)_V = - \left( \frac{\partial V}{\partial T} \right)_P. $$
# 
# С учетом этого, второе слагаемое выражения для дифференциала внутренней энергии:
# 
# $$ \left( \frac{\partial U}{\partial P} \right)_{T, n_i} dP = T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} dV - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial T} \right)_{V, n_i} dT - P dV + P \left( \frac{\partial V}{\partial T} \right)_{T, n_i} dT. $$
# 
# Третье слагаемое дифференциала внутренней энергии отвечает за процесс изменения внутренней энергии при изменении количества вещества $i$-го компонента при постоянных давлении и температуры. К таким процессам относятся диффузия (выравнивание количества частиц при их стохастическом движении), а также химические реакции. Если пренебречь данными процессами, то есть если считать, что при постоянных давлении и температуре количество частиц $i$-го компонента остается постоянным, тогда дифференциал внутренней энергии:
# 
# $$ dU = \left( C_P - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) dT + \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) dV .$$
# 
# Если рассматривать внутреннюю энергию системы (при постоянном количестве вещества компонентов) как функцию от объема и температуры $U = U \left( V, T \right)$, тогда дифференциал внутренней энергии можно записать в следующем виде:
# 
# $$ dU = \left( \frac{\partial U}{\partial V} \right)_{T, n_i} dV + \left( \frac{\partial U}{\partial T} \right)_{V, n_i} dT. $$
# 
# По [определению изохорной теплоемкости](../1-TD/TD-4-HeatCapacity.html#pvt-td-heat_capacity-isochoric):
# 
# $$ C_V = \left( \frac{\delta Q}{dT} \right)_V = \left( \frac{\partial U}{\partial T} \right)_V. $$
# 
# Выражение для [thermodynamic identity](../1-TD/TD-6-Entropy.html#pvt-td-entropy-thermodynamic_identity):
# 
# $$ dU = TdS - PdV. $$
# 
# Разделим левую и правую части на $dV$, принимая постоянным $T$:
# 
# $$ \left( \frac{\partial U}{\partial V} \right)_{T, n_i} = T \left( \frac{\partial S}{\partial V} \right)_{T, n_i} - P. $$
# 
# В соответствии со [вторым соотношением Максвелла](../1-TD/TD-13-MaxwellRelations.html#pvt-td-maxwell_relations-second):
# 
# $$ \left( \frac{\partial S}{\partial V} \right)_{T} = \left( \frac{\partial P}{\partial T} \right)_{V}. $$
# 
# Следовательно:
# 
# $$ \left( \frac{\partial U}{\partial V} \right)_{T, n_i} = T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P. $$
# 
# Тогда дифференциал внутренней энергии:
# 
# $$ dU = C_V dT + \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) dV. $$

# <a id='pvt-parameters-internal_energy-isobaric_isochoric_heat_capacities'></a>
# ```{admonition} NB
# Сравнивая с полученным ранее выражением для дифференциала внутренней энергии, отметим, что:
# +++
# $$C_V = C_P - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial T} \right)_{V, n_i}.$$
# +++
# ```
# 
# Если рассматривается идеальный газ, для которого применимо уравнение состояния в виде $PV = n R T$, тогда:
# 
# $$ C_V = C_P - T \frac{n R}{P} \frac{n R}{V} = C_P - T \frac{n^2 R^2}{P V} = C_P - n R = C_P - N k. $$
# 
# Полученное выражение соответствует выведенному [ранее](../1-TD/TD-4-HeatCapacity.html#pvt-td-heat_capacity-ideal_gas) выражению для разницы изобарной и изохорной теплоемкостей идеального газа.

# Таким образом, изменение внутренней энергии системы в процессе перехода из состояния $\left( V_1, T_1 \right)$ в состояние $\left( V_2, T_2 \right)$:
# 
# $$ \Delta U = \int_{T_1}^{T_2} C_V dT + \int_{V_1}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) dV . $$

# <a id='pvt-parameters-internal_energy-isochoric_heat_capacity'></a>
# Преобразуем первое слагаемое в выражении для изменения внутренней энергии. Поскольку теплоемкость можно представить как функцию от объема и температуры, то есть $C_V = C_V \left( V, T \right)$, тогда, применяя [свойство частных производных](../1-TD/TD-13-MaxwellRelations.html#pvt-td-maxwell_relations), получим:
# 
# $$ \left( \frac{\partial C_V}{\partial V} \right)_{T, n_i} = \frac{\partial}{\partial T} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right)_{V, n_i}. $$
# 
# Данное равенство преобразуется к слудующему:
# 
# $$ \left( \frac{\partial C_V}{\partial V} \right)_{T, n_i} = T \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i}. $$
# 
# Следовательно, в изотермическом процессе:
# 
# $$ \int_{ideal \; gas}^{real \; gas} dC_V = C_V - C_V^{*} = T \int_{ideal \; gas}^{real \; gas} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV = T \int_{\infty}^{V} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV .$$
# 
# Здесь под пределом интегрирования $ideal \; gas$ понимается $P \rightarrow 0$ и следовательно $V \rightarrow \infty$. При этом, если при таких условиях система достаточно точно описывается уравнением состояния идеального газа, то изохорная и изобарная теплоемкости связаны следующим соотношением:
# 
# $$ C_V^{*} = C_P^{*} - n R. $$
# 
# Несмотря на то что большинство веществ имеют достаточно высокие значения давления, при котором происходит переход из жидкой фазы в газообразную, существуют вещества, которые даже при предельно низком давлении могут находиться в жидком состоянии. Такие вещества имеют достаточную [энергию связи](https://en.wikipedia.org/wiki/Binding_energy) для того, чтобы данное вещество находилось в виде жидкой фазы. Кроме того, при низких давлениях вещества также могут находиться в твердом состоянии в зависимости от температуры системы. Таким образом, данное соотношение между изобарной и изохорной теплоемкостями стоит применять с осторожностью. С учетом представленных выше преобразований первое слагаемое в выражении для изменения внутренней энергии:
# 
# $$ \int_{T_1}^{T_2} C_V dT = \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{V} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT. $$
# 
# Следовательно, изменение внутренней энергии:
# 
# $$ \Delta U = \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{V} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT + \int_{V_1}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) dV . $$
# 
# Рассмотрим, каким образом может быть вычислено данное выражение. Существует три подхода, графическая интерпетация которых изображена на рисунке ниже:

# In[1]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'widget')

plt.rcParams.update({'figure.max_open_warning': False})
fig, axes = plt.subplots(1, 3, figsize=(6, 2.25))
fig.canvas.header_visible = False
for ax in axes[:-1]:
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(['$T_1$', '$T_2$'])
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(['$V_1$', '$V_2$'])
    ax.set_xlabel('T')
    ax.set_ylabel('V')
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.plot([-1.0, 0.0], [0.0, 0.0], c='k', ls='--', lw=0.5)
    ax.plot([-1.0, 1.0], [1.0, 1.0], c='k', ls='--', lw=0.5)
    ax.plot([0.0, 0.0], [-1.0, 0.0], c='k', ls='--', lw=0.5)
    ax.plot([1.0, 1.0], [-1.0, 1.0], c='k', ls='--', lw=0.5)
axes[0].arrow(0.0, 0.0, 0.0, 0.5, color='k', head_width=0.1)
axes[0].arrow(0.0, 0.5, 0.0, 0.5, color='k', head_width=0.0)
axes[0].arrow(0.0, 1.0, 0.5, 0.0, color='k', head_width=0.1)
axes[0].arrow(0.5, 1.0, 0.5, 0.0, color='k', head_width=0.0)
axes[0].scatter([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], color='k')
axes[1].arrow(0.0, 0.0, 0.5, 0.0, color='k', head_width=0.1)
axes[1].arrow(0.5, 0.0, 0.5, 0.0, color='k', head_width=0.0)
axes[1].arrow(1.0, 0.0, 0.0, 0.5, color='k', head_width=0.1)
axes[1].arrow(1.0, 0.5, 0.0, 0.5, color='k', head_width=0.0)
axes[1].scatter([0.0, 1.0, 1.0], [0.0, 0.0, 1.0], color='k')
axes[2].set_xticks([0.0, 1.0])
axes[2].set_xticklabels(['$T_1$', '$T_2$'])
axes[2].set_yticks([0.0, 1.0, 2.0])
axes[2].set_yticklabels(['$V_1$', '$V_2$', '$V_3$'])
axes[2].set_xlabel('T')
axes[2].set_ylabel('V')
axes[2].set_xlim(-1, 2)
axes[2].set_ylim(-1, 3)
axes[2].plot([-1.0, 0.0], [0.0, 0.0], c='k', ls='--', lw=0.5)
axes[2].plot([-1.0, 1.0], [1.0, 1.0], c='k', ls='--', lw=0.5)
axes[2].plot([0.0, 0.0], [-1.0, 0.0], c='k', ls='--', lw=0.5)
axes[2].plot([1.0, 1.0], [-1.0, 2.0], c='k', ls='--', lw=0.5)
axes[2].plot([-1.0, 1.0], [2.0, 2.0], c='k', ls='--', lw=0.5)
axes[2].arrow(0.0, 0.0, 0.0, 1.0, color='k', head_width=0.1)
axes[2].arrow(0.0, 1.0, 0.0, 1.0, color='k', head_width=0.0)
axes[2].arrow(0.0, 2.0, 0.5, 0.0, color='k', head_width=0.1)
axes[2].arrow(0.5, 2.0, 0.5, 0.0, color='k', head_width=0.0)
axes[2].arrow(1.0, 2.0, 0.0, -0.5, color='k', head_width=0.1)
axes[2].arrow(1.0, 1.5, 0.0, -0.5, color='k', head_width=0.0)
axes[2].scatter([0.0, 0.0, 1.0, 1.0], [0.0, 2.0, 2.0, 1.0], color='k')
axes[0].set_title('A')
axes[1].set_title('B')
axes[2].set_title('C')
fig.tight_layout()


# Рассмотрим подход $A$. Согласно условию, нам необходимо найти изменение внутренней энергии при переходе системы из точки $1: \; \left( T_1, V_1 \right)$ в точку $2: \; \left( T_2, V_2 \right)$. В этом процессе изменение внутренней энергии будет равно:
# 
# $$ \Delta U_{1-2} = U_2 - U_1. $$
# 
# Однако вместо перехода сразу из точки $1$ в точку $2$ рассмотрим переход с промежуточной точкой $2': \; \left( T_1, V_2 \right).$ То есть будто рассматриваемая система сначала перешла в точку $2'$ и только после этого - в точку $2$. Первый этап $1 - 2'$ является изотермическим, так как он проходит при постоянной температуре $T = T_1.$ В этом случае изменение внутренней энергии:
# 
# $$ \Delta U_{1-2'} = U_{2'} - U_1 = \int_{V_1}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_1} dV. $$
# 
# Второй этап $2' - 2$ является изохорным, так как он проходит при постоянном объеме $V = V_2.$ Тогда изменение внутренней энергии на данном этапе:
# 
# $$ \Delta U_{2' - 2} = U_2 - U_{2'} = \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{V_2} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT. $$
# 
# Сложим изменения внутренних энергий на рассмотренных этапах:
# 
# $$ \begin{align}
# \Delta U_{1-2'} + \Delta U_{2' - 2}
# &= \left( U_{2'} - U_1 \right) + \left( U_2 - U_{2'} \right) \\
# &= U_2 - U_1 \\
# &= \Delta U_{1 - 2} \\
# &= \int_{V_1}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_1} dV + \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{V_2} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT.
# \end{align} $$
# 
# Данный подход к нахождению изменения параметра значительно облегчает нахождение аналитического выражения интеграла.

# При подходе $A$ изменение внутренней энергии:
# 
# $$ \Delta U = \int_{V_1}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_1} dV + \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{V_2} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT. $$
# 
# При подходе $B$:
# 
# $$ \Delta U = \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{V_1} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT + \int_{V_1}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_2} dV. $$
# 
# Рассмотрим подход $C$, при этом под $V_3$ будем понимать условия идеального газа, то есть $V_3 = \infty$. Тогда измение внутренней энергии:
# 
# $$ \begin{alignat}{1}
# \Delta U 
# &= & \; \int_{V_1}^{\infty} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_1} dV + \int_{T_1}^{T_2} \left( C_V^{*} + T \int_{\infty}^{\infty} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV \right) dT \\
# && \; + \int_{\infty}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_2} dV \\
# &= & \; \int_{V_1}^{\infty} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_1} dV - \int_{V_2}^{\infty} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_2} dV + \int_{T_1}^{T_2} C_V^{*} dT .
# \end{alignat} $$

# <a id='pvt-parameters-internal_energy-srk_pr'></a>
# ## Вывод выражения внутренней энергии системы на основе уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона

# Получим выражение для частной производной давления по температуре при постоянных объеме и количестве вещества компонентов, используя [уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr): 
# 
# $$ \begin{align}
# \left( \frac{\partial P}{\partial T} \right)_{V, n_i}
# &= \frac{\partial}{\partial T} \left( \frac{n R T}{V - n b_m} - \frac{\alpha_m n^2}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2} \right)_{V, n_i} \\
# &= \frac{n R}{V - n b_m} - \frac{n^2 \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i}}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2}.
# \end{align} $$ 
# 
# С учетом этого подынтегральное выражение во втором слагаемом изменения внутренней энергии: 
# 
# $$ T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P = \frac{n^2 \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right)}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2}. $$ 
# 
# Тогда: 
# 
# $$\begin{align}
# \int_{V}^{\infty} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) dV
# &= n^2 \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right) \int_{V_1}^{V_2} \frac{dV}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2} \\
# &= \frac{n \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right)}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + b_m n \delta_1}{V + b_m n \delta_2} \bigg\rvert_{V}^{\infty} \\
# &= \frac{n \left( T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} - \alpha_m \right)}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + b_m n \delta_1}{V + b_m n \delta_2}.
# \end{align} $$
# 
# Таким образом, изменение внутренней энергии:
# 
# $$ \begin{align}
# \Delta U = & \frac{n \left( T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} - \alpha_m \right) \bigg\rvert_{T_1}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V_1 + b_m n \delta_1}{V_1 + b_m n \delta_2} - \frac{n \left( T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} - \alpha_m \right) \bigg\rvert_{T_2}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V_2 + b_m n \delta_1}{V_2 + b_m n \delta_2} \\ & + \int_{T_1}^{T_2} C_V^{*} dT. 
# \end{align} $$
# 
# Изменение удельной (приведенной к единице количества вещества) внутренней энергии, выраженной через коэффициент сверхсжимаемости:
# 
# $$ \Delta u = \frac{\left( T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} - \alpha_m \right) \bigg\rvert_{T_1}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z_1 + \delta_1 B_1}{Z_1 + \delta_2 B_1} - \frac{\left( T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} - \alpha_m \right) \bigg\rvert_{T_2}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z_2 + \delta_1 B_2}{Z_2 + \delta_2 B_2} + \int_{T_1}^{T_2} c_V^{*} dT.$$
# 
# Если рассматривается изотермический процесс $\left( T_1 = T_2 = T \right)$, при этом в качестве референсных условий принимаются условия $\left( V_1 = \infty, P_1 = 0, Z_1 = 1 \right)$, тогда удельная внутренняя энергия:
# 
# $$ u \left(P, T \right) = u \left(0, T\right) + \frac{\alpha_m -  T \left(\frac{\partial \alpha_m}{\partial T} \right)_{V, n_i}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z + \delta_1 B}{Z + \delta_2 B}. $$
# 
# Частная производная параметра $\alpha_m$ по температуре при постоянном объеме и количестве вещества компонентов для уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона была рассмотрена [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-srk_pr).

# <a id='pvt-parameters-internal_energy-sw'></a>
# ## Вывод выражения внутренней энергии системы на основе уравнения состояния Сорейде-Уитсона

# Поскольку [уравнение состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw) отличается от [уравнения состояния Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr) только расчетом парамета $a_i$ для воды, как компонента, а также определением коэффициентов попарного взаимодействия между водой и растворенными компонентами, то определение внутренней энергии системы с использованием уравнения состояния Сорейде-Уитсона будет отличаться от рассмотренного [ранее](#pvt-parameters-internal_energy-srk_pr) определения внутренней энергии системы с использованием уравнения состояния Пенга-Робинсона только нахождением частной производной параметра $\alpha_m$ по температуре. Для уравнения состояния Сорейде-Уитсона частные производные параметров были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-sw).

# <a id='pvt-parameters-internal_energy-cpa'></a>
# ## Вывод выражения внутренней энергии системы на основе уравнения состояния CPA

# In[ ]:




