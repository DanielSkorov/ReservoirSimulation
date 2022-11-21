#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-td-fugacity'></a>
# # Летучесть

# ## Определение летучести
# Выражение [thermodynamic identity](./TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs_partials), записанное через энергию Гиббса, при постоянном количестве частиц в системе записывается следующим образом:
# 
# $$ dG = -S dT + V dP. $$
# 
# Если рассматривать изотермический процесс, то данное выражение преобразуется следующим образом:
# 
# $$ dG = V dP. $$
# 
# ```{admonition} NB
# Важно отметить, что, принимая температуру постоянной, мы предполагаем, что в системе температура всех фаз одинакова.
# ```
# 
# Для идеального газа с учетом [уравнения состояния](./TD-1-Basics.html#pvt-td-basics-ideal_gas_eos):
# 
# $$ dG = \frac{N k T}{P} dP = N k T d \ln P = \frac{N}{N_A} \left(k N_A \right) T d \ln P = n R T d \ln P. $$
# 
# Здесь $N_A = 6.022 \cdot 10^{23} \; \frac{1}{моль}$ – число Авогадро, $R = k \cdot N_A = 8.314 \; \frac{Дж}{моль K}$ – универсальная газовая постоянная, $n$ – количество вещества (моль). Стоит отметить, что полученные равнее уравнения относительно количества частиц $N$ справедливы и для количества вещества $n$.

# Полученное выражение дифференциала энергии Гиббса справедливо для идеального газа с постоянным количеством молекул в изотермическом квази-стационарном процессе. Для реального газа вместо давления используют такой параметр, как ***летучесть***:
# 
# $$ dG = n R T d \ln f. $$
# 
# ```{prf:определение}
# :nonumber:
# ***Летучесть*** определяется следующим выражением:
# +++
# $$\lim_{P \rightarrow 0} \left( \frac{f}{P} \right) = 1. $$
# +++
# ```
# 
# При этом,
# 
# ```{prf:определение}
# :nonumber:
# Отношение летучести к давлению называют ***коэффициентом летучести***:
# +++
# $$\phi = \frac{f}{P}.$$
# +++
# ```
# 
# Для компонента $i$, находящегося в термодинамическом равновесии (или в квази-стационарном изотермическом процессе), летучесть определяется следующим выражением (единицы измерения химического потенциала – $\frac{Дж}{моль}$):
# 
# $$ d \mu_i = R T d \ln f_i. $$
# 
# Данное выражение справедливо для процесса с постоянным количеством молекул компонента $i$ в системе. При этом, определение летучести дается на основании следующего выражения:
# 
# $$ \lim_{P \rightarrow 0} \left( \frac{f_i}{x_i P} \right) = 1. $$
# 
# Здесь $x_i$ – мольная доля компонента в фазе.

# <a id='pvt-td-fugacity-component_fugacity'></a>
# Преобразуем уравнение, определяющее летучесть $i$-го компонента, к следующему виду:
# 
# $$ \begin{align} d \mu_i - R T d \ln \left( x_i P \right) &= R T d \ln f_i - R T d \ln \left( x_i P \right); \\ d \mu_i - R T d \ln \left( x_i P \right) &= R T d \ln \phi_i; \\ d \mu_i - R T \left( d \ln x_i + d \ln P \right) &= R T d \ln \phi_i. \end{align}$$
# 
# Поскольку количество молекул $i$-го компонента в системе зафиксировано, то:
# 
# $$ d \mu_i - R T d \ln P = R T d \ln \phi_i. $$
# 
# Пусть
# 
# ```{prf:определение}
# :nonumber:
# ***Коэффициент сверхсжимаемости*** – параметр, который определяется следующим выражением:
# +++
# $$Z = \frac{P V}{n R T}.$$
# +++
# ```
# 
# Получим дифференциал коэффициента сверхсжимаемости, рассматривая изотермический процесс с постоянным количеством молекул в системе:
# 
# $$ dZ = \frac{d \left( P V \right)}{n R T} = \frac{V dP + P dV}{n R T}. $$
# 
# Разделим левую и правую части уравнения на $Z$:
# 
# $$ \frac{dZ}{Z} = \frac{dP}{P} + \frac{dV}{V}. $$
# 
# С учетом этого, [выражение](#pvt-td-fugacity-component_fugacity) для определения летучести компонента будет иметь следующий вид:
# 
# $$ R T d \ln \phi_i = d \mu_i - R T \left( \frac{dZ}{Z} - \frac{dV}{V} \right). $$

# <a id='pvt-td-fugacity-chemical_potential_relation'></a>
# Для многокомпонентной системы [thermodynamic identity](TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-helmholtz_partials), выраженное через энергию Гельмгольца, будет иметь следующий вид:
# 
# $$ dF = -P dV - S dT + \sum_i \mu_i dn_i. $$
# 
# Отсюда следует, что:
# 
# $$ \begin{align} \mu_i &= \left( \frac{\partial F}{\partial n_i} \right)_{V, T, n_{j \neq i}}; \\ -P &= \left( \frac{\partial F}{\partial V} \right)_{T, n_i}. \end{align} $$
# 
# Запишем вторую частную производную энергии Гельмгольца $F$ по объему $V$ и количеству вещества $i$-го компонента $n_i$:
# 
# $$ \frac{\partial^2 F}{\partial n_i \partial V} = \frac{\partial}{\partial n_i} \left( \frac{\partial F}{\partial V} \right) = \frac{\partial}{\partial V} \left( \frac{\partial F}{\partial n_i} \right). $$
# 
# С учетом полученных частных производных энергиии Гельмгольца по количеству вещества $i$-го компонента и объему:
# 
# $$ - \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \left( \frac{\partial \mu_i}{\partial V} \right)_{T, n_i}. $$
# 
# Следовательно, расссматривая изотермический процесс с постоянным количеством молекул $i$-го компонента в системе можно записать:
# 
# $$ d\mu_i = - \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} dV. $$
# 
# С учетом дифференциала химического потенциала $i$ компонента в изотермическом процессе с постоянным количеством вещества $i$-го компонента в системе получим дифференциал логарифма коэффициента летучести $i$-го компонента:
# 
# $$ d \ln \phi_i = \left( \frac{1}{V} - \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right) dV - d \ln Z. $$
# 
# Следовательно, интегрируя данное выражение, получим:
# 
# $$ \int_0^{\ln \phi_i} d \ln \phi_i = \int_\infty^V \left( \frac{1}{V} - \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right) dV - \int_0^{\ln Z} d \ln Z. $$
# 
# Или при замене пределов интегрирования:
# 
# $$ \ln \phi_i = \int_\infty^V \left( \frac{1}{V} - \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right) dV - \ln Z = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} - \frac{1}{V} \right) dV - \ln Z . $$

# <a id='pvt-td-fugacity-equilibrium'></a>
# ```{admonition} NB
# Таким образом, если рассматривать квази-стационарный изотермический процесс с постоянным количеством вещества в системе, состоящей из нескольких фаз, то равновесное состояние каждого компонента вместо равенства химических потенциалов компонентов будет определяться равенством летучестей компонентов в каждой из фаз соответственно:
# +++
# $${f_1}_i = {f_2}_i, \; i=1...N_c.$$
# +++
# ```
# 
# При этом, поскольку рассматривается равновесное состояние, то количество вещества каждого компонента постоянно, следовательно, летучесть компонента может быть рассчитана по коэффициенту летучести, определяемому следующим выражением:
# 
# $$ \ln \phi_i = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} - \frac{1}{V} \right) dV - \ln Z.$$
# 
# Кроме того, полученное выражение также справедливо для систем, состояющих из одного компонента. В таком случае выражение для логарифма коэффициента летучести записывается следующим образом:
# 
# $$ \ln \phi = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n} \right)_{V, T} - \frac{1}{V} \right) dV - \ln Z.$$

# <a id='pvt-td-fugacity-ideal_gas'></a>
# ## Летучесть идеального газа
# Рассмотрим применение полученного выражения для идеального газа, описываемого уравнением состояния:
# 
# $$ PV = n R T. $$
# 
# Для идеального газа частная производная давления по количеству вещества компонента:
# 
# $$ \left( \frac{\partial P}{\partial n} \right)_{V, T} = \frac{RT}{V}. $$
# 
# Тогда логарифм коэффициента летучести:
# 
# $$ \ln \phi = \int_V^\infty \left( \frac{1}{R T} \frac{RT}{V} - \frac{1}{V} \right) dV - \ln Z = - \ln Z. $$
# 
# Поскольку для идеального газа коэффициент сверхсжимаемости $Z = 1,$ тогда
# 
# $$ \ln \phi = 0. $$
# 
# То есть коэффициент летучести:
# 
# $$ \phi = 1. $$
# 
# Следовательно, летучесть:
# 
# $$ f = P. $$

# <a id='pvt-td-fugacity-ideal_gas_mixture'></a>
# ## Летучесть смеси идеальных газов
# Для смеси идеальных газов также выполняется уравнение состояния:
# 
# $$ PV = n R T, $$
# 
# где $n = \sum_{i=1}^{N_c} n_i.$ В данном выражении $N_c$ – количество компонентов в системе. Следовательно, частная производная давления по количеству вещества компонента $i$:
# 
# $$ \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \frac{RT}{V} \sum_{i=1}^{N_c} \left( \frac{\partial n_i}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \frac{RT}{V} . $$
# 
# Тогда логарифм коэффициента летучести $i$-го компонента:
# 
# $$ \ln \phi_i = 0. $$
# 
# Тогда летучесть $i$-го компонента:
# 
# $$ f_i = x_i P. $$
# 
# ```{admonition} NB
# Полученное выражение является частным случаем соотношения, выполняемого для ***идеальных смесей***:
# +++
# $$f_i \left( P, T, x_i \right) = x_i f \left( P, T \right).$$
# +++
# ```
# 
# Докажем данное утверждение. Для начала дадим определение идеальной смеси.
# 
# ```{prf:определение}
# :nonumber:
# Смесь является ***идеальной***, если для нее выполняется следующее соотношение:
# +++
# $$ V \left(P, T, n_1, n_2, \ldots, n_{N_c} \right) = \sum_{i=1}^{N_c} n_i v_i \left( P, T \right).$$
# +++
# Здесь $N_c$ - количество компонентов в системе, $v_i \left( P, T \right)$ - молярный (удельный) объем $i$-го компонента.
# ```
# 
# То есть объем для идеальных смесей является экстенсивным параметров.
# 
# ```{prf:proof}
# [Ранее](TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy-partial_molar_observables) для всех экстенсивных параметров было показано, что:
# +++
# $$ V \left(P, T, n_1, n_2, \ldots, n_{N_c} \right) = \sum_{i=1}^{N_c} n_i \bar{V_i}.$$
# +++
# Следовательно,
# +++
# $$ \bar{V_i} = v_i \left( P, T \right).$$
# +++
# Дифференциал логарфима коэффициента летучести $i$-го компонента можно выразить из полученного [ранее](#pvt-td-fugacity-component_fugacity) выражения:
# +++
# $$ RT d \ln \phi_i = d \mu_i - \frac{RT}{P} dP.$$
# +++
# С учетом соотношений частных производных химического потенциала и давления по объему и количеству вещества компонентов:
# +++
# $$ RT d \ln \phi_i = \left( \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{RT}{P} \right) dP = \left( \bar{V_i} - \frac{RT}{P} \right) dP = \left( v_i \left( P, T \right) - \frac{RT}{P} \right) dP .$$
# +++
# Из данного выражения видно, что для идеальной смеси логарифм летучести не зависит от компонентного состава и определяется только термобарическими условиями и свойствами компонентов. Аналогичный результат получается, если рассматривать систему, состоящую из одного компонента. Следовательно, коэффициент летучести $i$-го компонента в идеальной смеси равен коэффициенту летучести этого же компонента при отсутствии других компонентов. Из определения коэффициента летучести и сформулированного вывода вытекает доказываемое равенство.
# ```
# 
# <a id='pvt-td-fugacity-ideal_gas_mixture-chemical_potential'></a>
# Получим еще одно важное свойство смесей идеальных газов.
# 
# ```{admonition} NB
# Пусть имеется идеальный однокомпонентный газ при давлении $P$. Если данный газ изобарно смешать с другими идеальными газами, то давление рассматриваемого газа в смеси в соответствии с полученным ранее выражением будет равняться $x_i P$. Поскольку для данного компонента произошло изменение давления (температура рассматривается постоянной), то дифференциал химического потенциала:
# +++
# $$d \mu^{ig}_i = \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i} dP.$$
# +++
# Интегрируя данное выражение от однокомпонентного состояния к многокомпонентной смеси, получим:
# +++
# $$ \mu^{ig}_i \left( P, T, n_i \right) - \mu^{ig}_i \left( P, T, \right) = \int_{P}^{x_i P} \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i} dP.$$
# +++
# Рассмотрим вторую частную производную энергии Гиббса по количеству вещества $i$-го компонента и давлению:
# +++
# $$\frac{\partial^2 G}{\partial n_i \partial P} = \frac{\partial}{\partial n_i} \left( \left( \frac{\partial G}{\partial P} \right)_{T,n_i} \right)_{P,T} = \frac{\partial}{\partial P} \left( \left( \frac{\partial G}{\partial n_i} \right)_{P,T} \right)_{T,n_i}.$$
# +++
# С учетом [частных производных энергии Гиббса](TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs_partials) получим:
# +++
# $$\left( \frac{\partial V}{\partial n_i} \right)_{P,T} = \left( \frac{\partial \mu_i}{\partial P} \right)_{T,n_i}.$$
# +++
# Тогда рассматриваемая разница химических потенциалов может быть преобразована следующим образом, применяя уравнение состояния идеального газа:
# +++
# $$ \begin{align}
# \mu^{ig}_i \left( P, T, n_i \right) - \mu^{ig}_i \left( P, T \right)
# &= \int_{P}^{x_i P} \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i} dP \\
# &= \int_{P}^{x_i P} \left( \frac{\partial V}{\partial n_i} \right)_{P,T} dP \\
# &= \int_{P}^{x_i P} \frac{\partial}{\partial n_i} \left( \frac{nRT}{P} \right)_{P,T} dP \\
# &= RT \int_{P}^{x_i P} \frac{dP}{P} \\
# &= RT \ln x_i .
# \end{align} $$
# +++
# Следовательно,
# +++
# $$\mu^{ig}_i \left( P, T, n_i \right) = \mu^{ig}_i \left( P, T \right) + RT \ln x_i .$$
# +++
# Данное уравнение может быть использоваться для расчета химического потенциала компонента идеального газа в идеальной смеси, зная химический потенциал чистого компонента при тех же термобарических условиях и компонентный состав смеси.
# ```
# 
# При давлении $P_1$ и компонентном составе ${x_i}_1$ химический потенциал $i$-го компонента в смеси:
# 
# $$ \mu_i \left(P_1, T, {x_i}_1 \right) = \mu_i \left( P_1, T \right) + RT \ln {x_i}_1.$$
# 
# При давлении $P_2$ и компонентном составе ${x_i}_2$ химический потенциал $i$-го компонента в смеси:
# 
# $$ \mu_i \left(P_2, T, {x_i}_2 \right) = \mu_i \left( P_2, T \right) + RT \ln {x_i}_2.$$
# 
# Тогда разница данных выражений:
# 
# $$\mu_i \left(P_2, T, {x_i}_2 \right) - \mu_i \left(P_1, T, {x_i}_1 \right) = \mu_i \left( P_2, T \right) - \mu_i \left( P_1, T \right) + R T \ln \frac{{x_i}_2}{{x_i}_1}.$$
# 
# При этом, разница химических потенциалов чистого компонента, являющегося идеальным газом,:
# 
# $$\mu_i \left( P_2, T \right) - \mu_i \left( P_1, T \right) = RT \ln \frac{P_2}{P_1}.$$
# 
# Тогда:
# 
# $$\mu_i \left(P_2, T, {x_i}_2 \right) - \mu_i \left(P_1, T, {x_i}_1 \right) = RT \ln \frac{P_2 {x_i}_2}{P_1 {x_i}_1}.$$

# <a id='pvt-td-fugacity-real_gas_mixture'></a>
# ## Летучесть смеси реальных газов
# По аналогии с идеальным газом, для реальных газов данное соотношение записывается следующим образом:
# 
# $$\mu_i \left(P_2, T, {x_i}_2 \right) - \mu_i \left(P_1, T, {x_i}_1 \right) = RT \ln \frac{ f_i \left(P_2, T, {x_i}_2 \right)}{f_i \left( P_1, T, {x_i}_1 \right)}.$$
# 
# Тогда разница химических потенциалов между реальным и идеальным газами:
# 
# $$\mu_i^{rg} \left(P_2, T, {x_i}_2 \right) - \mu_i^{ig} \left(P_2, T, {x_i}_2 \right) = \mu_i^{rg} \left(P_1, T, {x_i}_1 \right) - \mu_i^{ig} \left(P_1, T, {x_i}_1 \right) + RT \ln \left( \frac{ f_i \left(P_2, T, {x_i}_2 \right)}{f_i \left( P_1, T, {x_i}_1 \right)} \frac{{x_i}_1 P_1}{{x_i}_2 P_2} \right).$$
# 
# ```{admonition} NB
# При этом, если давление $P_1$ достаточно низкое настолько, что при нем рассматриваемая система ведет себя, как идеальный газ, то:
# +++
# $$\mu_i^{rg} \left(P_2, T, {x_i}_2 \right) - \mu_i^{ig} \left(P_2, T, {x_i}_2 \right) = RT \ln \frac{ f_i \left(P_2, T, {x_i}_2 \right)}{{x_i}_2 P_2} = RT \ln \phi_i \left( P_2, T, {x_i}_2 \right).$$
# +++
# ```
# 
# Данное соотношение между химическими потенциалами реального и идеального газов достаточно полезно, так как будет использоваться при определении других [термодинамических параметров](../3-Parameters/Parameters-0-Introduction.html#pvt-parameters) реальных систем с использованием [уравнений состояния](../2-EOS/EOS-0-Introduction.html#pvt-eos).
