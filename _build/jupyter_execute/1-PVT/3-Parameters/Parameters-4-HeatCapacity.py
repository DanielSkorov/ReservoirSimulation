#!/usr/bin/env python
# coding: utf-8

# <a id='pvt.parameters.heat_capacity'></a>
# # Теплоемкость

# <a id='pvt-parameters-heat_capacity-ideal_gas'></a>
# Для получения значения изменения внутренней энергии и энтальпии в неизотермическом процессе необходимо вычислить интеграл изохорной и изобарной теплоемкостей системы при $P \rightarrow 0$ по температуре соотвественно. Если рассматриваемая система при низких значениях давления и рассматриваемых температурах может описываться уравнением состояния идеального газа, тогда:
# 
# $$ C_V^* = C_P^* - n R. $$
# 
# Изобарная теплоемкость при низких давлениях для некоторых веществ может рассматриваться как функция температуры в следующем виде:
# 
# $$ C_P^* = a + b T + c T^2 + d T^3 + e T^4. $$
# 
# Коэффициенты в данном уравнении для некоторых чистых веществ могут быть найдены в различных публикациях и [базах данных](https://www.cheric.org/research/kdb/hcprop/cmpsrch.php). Кроме того, корреляции для расчета изобарной теплоемкости, энтальпии и энтропии компонентов, находящихся в состоянии идеального газа, могут быть найдены в работе \[[Passut and Danner, 1972](https://doi.org/10.1021/i260044a016)\]. При этом, поскольку теплоемкость является экстенсивным параметром, то теплоемкость смеси на практике может быть рассчитана на основе известных теплоемкостей компонентов. При этом, данное правило применимо в том случае, когда на теплоемкость смеси не оказывает влияние межмолекулярное взаимодействие, или им можно пренебречь. Покажем, что приведенное выше выражение, связывающее между собой изохорную и изобарную теплоемкости, справедливо также и для [кубических уравнений состояния](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr).

# В общем виде соотношение между изохорной и изобарной теплоемкостями [записывается](./Parameters-1-InternalEnergy.html#pvt-parameters-internal_energy-isobaric_isochoric_heat_capacities) в следующем виде:
# 
# $$ C_V = C_P - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial T} \right)_{V, n_i}. $$
# 
# Если для идеального газа уравнение состояния записывается в виде $PV = n R T$, то для реальных систем вводится понятие коэффициента сверхсжимаемости, определяемого выражением:
# 
# $$ Z = \frac{PV}{nRT}. $$
# 
# Запишем частные производные коэффициента сверхсжимаемости по температуре при постоянных давлении и объеме:
# 
# $$ \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} = \frac{P}{nR} \frac{\partial}{\partial T} \left( \frac{V}{T} \right)_{P, n_i} = \frac{P}{nRT} \left( \frac{\partial V}{\partial T} \right)_{P, n_i} - \frac{PV}{nrT^2} = \frac{P}{nRT} \left( \frac{\partial V}{\partial T} \right)_{P, n_i} - \frac{Z}{T}; \\ \left( \frac{\partial Z}{\partial T} \right)_{V, n_i} = \frac{V}{nR} \frac{\partial}{\partial T} \left( \frac{P}{T} \right)_{V, n_i} = \frac{V}{nRT} \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - \frac{PV}{nrT^2} = \frac{V}{nRT} \left( \frac{\partial P}{\partial T} \right)_{P, n_i} - \frac{Z}{T}.$$
# 
# Тогда частные производные объема и давления по температуре:
# 
# $$ \left( \frac{\partial V}{\partial T} \right)_{P, n_i} = \frac{nR}{P} \left( Z + T \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \right); \\ \left( \frac{\partial P}{\partial T} \right)_{V, n_i} = \frac{nR}{V} \left( Z + T \left( \frac{\partial Z}{\partial T} \right)_{V, n_i} \right). $$
# 
# Тогда соотношение между изохорной и изобарной теплоемкостями можно записать в следующем виде:
# 
# $$ C_V = C_P - \frac{nRT}{Z} \left( Z + T \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \right) \left( Z + T \left( \frac{\partial Z}{\partial T} \right)_{V, n_i} \right). $$
# 
# [Кубические уравнения состояния](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr-Z_PT) относительно коэффициента сверхсжимаемости:
# 
# $$ Z^3 - \left( 1 - c B \right) Z^2 + \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right) Z^2 - \left( A B - c \left( B^2 + B^3 \right) \right) = 0. $$
# 
# Параметры $A$ и $B$:
# 
# $$ \begin{align*} A &= \frac{\alpha_m P}{R^2 T^2}; \\ B &= \frac{b_m P}{R T}. \end{align*} $$
# 
# При $P \rightarrow 0$ параметры $A \rightarrow 0$ и $B \rightarrow 0$. Тогда уравнение состояния относительно коэффициента сверхсжимаемости:
# 
# $$ Z^3 - Z^2 = 0. $$
# 
# То есть коэффициент сверхсжимаемости $Z \rightarrow 1$ при $P \rightarrow 0$. Поскольку коэффициент сверсхжимаемости становится равным константе, то его частные производные по температуре становятся равными нулю. Тогда соотношение между изохорной и изобарной теплоемкостью будет записываться аналогично соотношению, полученному при использовании уравнения состояния идеального газа. Таким образом, $C_V^* = C_P^* - n R$ свойственно и для систем, для которых применимы кубические уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона.
# 
# ```{admonition} NB
# Сделанный вывод также имеет следующее следствие. Если для рассматриваемой системы межмолекулярное взаимодействие достаточно велико, что им нельзя пренебречь, то коэффициент сверхсжимаемости можно записать:
# +++
# $$Z = Z_{EOS} + Z_{ImIn}.$$
# +++
# $ImIn$ – intermolecular interaction (межмолекулярное взаимодействие). На этом соотношении, например, основано CPA (cubic plus association) EOS, рассмотренное [ранее](../2-EOS/EOS-6-CPA.html#pvt-eos-cpa).
# ```

# <a id='pvt-parameters-heat_capacity-srk_pr-isochoric'></a>
# ## Вывод выражения изохорной теплоемкости с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона

# Получим выражение для изохорной теплоемкости с использованием [уравнений состояния Пенга-Робинсона и Суаве-Редлиха-Квонга](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr). Для этого необходимо получить вторую частную производную давления по температуре при постоянном объеме и количестве вещества компонентов:
# 
# $$ \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} = - \frac{n^2 \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{V, n_i}}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2}.$$
# 
# Тогда интеграл в выражении для изохорной теплоемкости:
# 
# $$ \begin{align}
# \int_{\infty}^{V} \left( \frac{\partial^2 P}{\partial T^2} \right)_{V, n_i} dV
# &= - n^2 \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{V, n_i} \int_{\infty}^{V} \frac{dV}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2} \\
# &= - \frac{n \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{V, n_i}}{b_m \left(\delta_2 - \delta_1 \right)} \ln \frac{V + b_m n \delta_1}{V + b_m n \delta_2} \\
# &= - \frac{n \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{V, n_i}}{b_m \left(\delta_2 - \delta_1 \right)} \ln \frac{Z + \delta_1 B}{Z + \delta_2 B}.
# \end{align} $$
# 
# С учетом этого изохорная теплоемкость:
# 
# $$C_V = C_V^* - T \frac{n \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{V, n_i}}{b_m \left(\delta_2 - \delta_1 \right)} \ln \frac{Z + \delta_1 B}{Z + \delta_2 B}.$$
# 
# Аналогичное выражение может быть получено при рассмотрении производной внутренней энергии в изотермическом процессе по температуре при постоянном объеме.
# 
# Вторая частная производная $\alpha_m$ были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.md).

# <a id='pvt-parameters-heat_capacity-srk_pr-isobaric'></a>
# ## Вывод выражения изобарной теплоемкости с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона

# Получим выражение для изобарной теплоемкости с использованием [уравнений состояния Пенга-Робинсона и Суаве-Редлиха-Квонга](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr). С учетом [определения](../1-TD/TD-5-Enthalpy.html#pvt-td-enthalpy-isobaric_heat_capacity) изобарной теплоемкости:
# 
# $$ \begin{alignat}{1}
# C_P
# &= & \; C_P^* + n R \left( Z - 1 \right) + n R T \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} + \frac{\partial}{\partial T} \left( \frac{n \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right)}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z + \delta_1 B}{Z + \delta_2 B} \right)_{P, n_i} \\
# &= & \; C_P^* + n R \left( Z - 1 \right) + n R T \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} - \frac{n T \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{V, n_i}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z + \delta_1 B}{Z + \delta_2 B} + \frac{n \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right)}{b_m \left( \delta_2 - \delta_1 \right)} \\
# && \; + \frac{\partial}{\partial T} \left( \ln \frac{Z + \delta_1 B}{Z + \delta_2 B} \right)_{P, n_i}. 
# \end{alignat}$$
# 
# Выражения для частных производных коэффициента сверхсжимаемости и параметра $B$ по температуре представлены в [приложении A в предыдущем разделе](../2-EOS/EOS-Appendix-A-PD.md).

# In[ ]:





# <a id='pvt.parameters.heat_capacity.sw.isochoric'></a>
# ## Вывод выражения изохорной теплоемкости с использованием уравнения состояния Сорейде-Уитсона

# In[ ]:





# <a id='pvt.parameters.heat_capacity.sw.isobaric'></a>
# ## Вывод выражения изобарной теплоемкости с использованием уравнения состояния Сорейде-Уитсона

# In[ ]:





# <a id='pvt.parameters.heat_capacity.cpa.isochoric'></a>
# ## Вывод выражения изохорной теплоемкости с использованием уравнения состояния CPA

# In[ ]:





# <a id='pvt.parameters.heat_capacity.cpa.isobaric'></a>
# ## Вывод выражения изобарной теплоемкости с использованием уравнения состояния CPA

# In[ ]:




