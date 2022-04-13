#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-td-enthalpy'></a>
# # Энтальпия
# Для любого изохорного процесса мы можем использовать [первое начало термодинамики](TD-3-Heat-Work.html#pvt-td-heat_and_work-first_law) для оценки количества тепла, переданного системе: $Q = \Delta U$. То есть, определяя величину внутренней энергии в изохорном процессе, мы однозначно определяем количество тепла, переданное рассматриваемой системе. Однако, какую величину использовать для системы в неизохорном процессе, когда работа может совершаться над системой?

# <a id='pvt-td-enthalpy-definition'></a>
# ```{prf:определение}
# :nonumber:
# Параметр, который может быть использован для определения количества тепла, переданного системе в изобарном процессе, называется ***энтальпией***. Энтальпия системы, как параметр, определяется следующим выражением:
# +++
# $$ H = U + PV. $$
# +++
# ```

# Энтальпия имеет единицы измерения энергии (Дж). Поскольку энтальпия является параметром, то она определена для каждого термодинамического состояния системы. Однако, энтальпия определена вне зависимости от того, является ли рассматриваемый квази-стационарный процесс изобарным.

# Рассмотрим квази-стационарный изобарный процесс, который приводит к изменению состояния системы. В соответствии с первым началом термодинамики, изменение внутренней энергии $\Delta U$ рассматриваемой системы в ходе данного процесса будет происходить посредством передачи системе тепла $Q$, а также в результате совершения над ней работы $W = - P \Delta V + W'$. То есть работа, совершаемая над системой, будет состоять из механической работы (работы расширения-сжатия) и немеханической. В общем виде, изменение энтальпии системы в таком процессе можно рассмотреть следующим образом:

# $$ \Delta H = \Delta U + \Delta \left( P V \right) = \Delta U + P \Delta V = \Delta U - \left( - P \Delta V \right) = Q + W' . $$

# Таким образом, изменение энтальпии системы будет соответствовать передаче энергии любым способом, за исключением механической работы. В частности, если над системой совершается только механическая работа $ \left( W' = 0 \right) $, то изменение энтальпии будет равно количеству переданного системе тепла.

# <a id='pvt-td-enthalpy-isobaric_heat_capacity'></a>
# Другим подходом к определению понятия энтальпии является ее понимание как параметра, чье изменение по отношению к изменению температуры в изобарном процессе определяет теплоемкость системы:

# $$ C_P = \left( \frac{\delta Q}{dT} \right)_P = \left( \frac{\partial U}{\partial T} \right)_P + P \left( \frac{\partial V}{\partial T} \right)_P = \left( \frac{\partial}{\partial T} \left( U + PV \right) \right)_P = \left( \frac{\partial H}{\partial T} \right)_P. $$

# Существует еще одно несколько абстрактное представление об энтальпии, согласно которому, энтальпия системы, находящейся в некотором равновесном состоянии, представляет собой то количество энергии, которое необходимо затратить, чтобы *из ничего* создать данную систему с внутренней энергией $U$ и объемом $V$, пребывающем в равновесном состоянии при давлении $P$. То есть, что необходимо сделать, чтобы создать некоторую систему, находящуюся в пространстве в равновесном состоянии? Сначала необходимо создать систему, затратив на это количество энергии, равное ее внутренней энергии $U$, а затем поместить данную энергию в пространство, затратив на это количество энергии, равное работе $PV$. Следовательно, энтальпия также определяет количество энергии, высвобождаемое при уничтожении системы.
