#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-esc-equilibrium'></a>
# # Определение равновесного состояния системы

# Пусть имеется система, состоящая из $N_c$ компонентов и $N_p$ фаз $\left( N_p > 1 \right)$. Количество вещества $i$-го компонента в фазе $j$ обозначим через $n_i^j$. Тогда количество вещества $i$-го компонента в системе:

# $$\sum_{j=1}^{N_p} n_i^j = n_i.$$

# Количество вещества $j$-ой фазы:

# $$\sum_{i=1}^{N_c} n_i^{j} = n^j.$$

# Количество вещества системы:

# $$\sum_{i=1}^{N_c} n_i = \sum_{j=1}^{N_p} n^j = \sum_{i=1}^{N_c} \sum_{j=1}^{N_p} n_i^j = n.$$

# Пусть мольная доля компонента в системе обозначается $z_i$, при этом:

# $$z_i = \frac{n_i}{n}.$$

# Мольная доля компонента в фазе $j$:

# $$y_i^j = \frac{n_i^j}{n^j}.$$

# Мольная доля фазы:

# $$F^j = \frac{n^j}{n}.$$

# С учетом этого, выполняются следующие соотношения:

# $$ \begin{align} &\sum_{i=1}^{N_c} z_i = 1 ; \\ &\sum_{i=1}^{N_c} y_i^j = 1; \\ &\sum_{j=1}^{N_p} F^j = 1. \end{align}$$

# ```{prf:определение}
# :nonumber:
# Введем понятие ***константы (коэффициента) фазового равновесия*** (*k-value*) $i$-го компонента, характеризующей отношение мольной доли компонента в фазе $j$ к мольной доли компонета в референсной фазе $R$:
# +++
# $$ K_i^{jR} = \frac{y_i^j}{y_i^R}.$$
# +++
# ```

# При этом,

# $$K_i^{RR} = \frac{y_i^R}{y_i^R} = 1.$$

# <a id='pvt-esc-equilibrium-mole_fractions'></a>
# Рассмотрим следующее соотношение:

# $$\sum_{j=1}^{N_p} n_i^j = n_i.$$

# Разделим левую и правую части данного выражения на количество вещества системы $n$:

# $$\sum_{j=1}^{N_p} \frac{n_i^j}{n} = \frac{n_i}{n}.$$

# Преобразуем левую часть уравнения:

# $$ \sum_{j=1}^{N_p} \frac{n_i^j}{n^j} \frac{n^j}{n} = \frac{n_i}{n}.$$

# С учетом представленных выше определений, получим:

# $$ \sum_{j=1}^{N_p} y_i^j F^j = z_i.$$

# Преобразуем левую часть уравнения, применяя определение константы фазового равновесия компонента:

# $$\sum_{j=1}^{N_p} y_i^R K_i^{jR} F^j = z_i.$$

# Тогда мольная доля компонента $i$ в референсной фазе $R$:

# $$y_i^R = \frac{z_i}{\sum_{j=1}^{N_p} K_i^{jR} F^j} = \frac{z_i K_i^{RR}}{\sum_{j=1}^{N_p} K_i^{jR} F^j}.$$

# Следовательно, для остальных фаз:

# $$y_i^j = \frac{z_i K_i^{jR}}{\sum_{k=1}^{N_p} K_i^{kR} F^k}.$$

# Для любой фазы $j$, в том числе для референсной $R$, выполняется следующее соотношение:

# $$\sum_{i=1}^{N_c} y_i^j = 1.$$

# Следовательно,

# $$\sum_{i=1}^{N_c} y_i^j - \sum_i y_i^R = 0.$$

# Подставляя в данное выражение полученные ранее уравнения для мольной доли $i$-го компонента в фазе $j$ и референсной фазе $R$, получим:

# $$ \begin{align} \sum_{i=1}^{N_c} y_i^j - \sum_{i=1}^{N_c} y_i^R
# &= \sum_{i=1}^{N_c} \left( y_i^j - y_i^R \right) \\
# &= \sum_{i=1}^{N_c} \left( \frac{z_i K_i^{jR}}{\sum_{k=1}^{N_p} K_i^{kR} F^k} - \frac{z_i K_i^{RR}}{\sum_{k=1}^{N_p} K_i^{kR} F^k} \right) \\
# &= \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p} K_i^{kR} F^k} .\end{align}$$

# Таким образом, имеется $\left( N_p - 1 \right)$ уравнений, содержащих $\left( N_p - 1 \right) N_c + N_p$ неизвестных: $K_i^{jR}, \; i=1 \ldots N_c, \; j=1 \ldots \left( N_p - 1 \right)$ и $F^k, \; k=1 \ldots N_p.$ При этом, мольные доли компонентов в системе $z_i$ считаются известными.

# $$\sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p} K_i^{kR} F^k} = 0, \; j=1 \ldots N_p - 1.$$

# Рассмотрим выражение:

# $$\sum_{j=1}^{N_p} F^j = 1.$$

# Выделим из данного выражения мольную долю референсной фазы:

# $$\sum_{j=1}^{N_p-1} F^j + F^R = 1.$$

# Тогда:

# $$F^R = 1 - \sum_{j=1}^{N_p-1} F^j.$$

# <a id='pvt-esc-equilibrium-rachford_rice'></a>
# С учетом этого преобразуем систему из $\left( N_p - 1 \right)$ уравнений:

# $$ \begin{align}
# \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p} K_i^{kR} F^k}
# &= \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} K_i^{kR} F^k + K_i^{RR} F^R} \\
# &= \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} K_i^{kR} F^k + 1 - \sum_{k=1}^{N_p-1} F^k} \\
# &= \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1} \\
# &= 0, \; j=1 \ldots N_p-1.
# \end{align} $$

# Полученное уравнение называется уравнение [Речфорда-Райса](https://doi.org/10.2118/952327-G) (*Rachford-Rice*); оно записывается для $N_p-1$ фаз. В данной системе уравнений $\left( N_p - 1 \right) N_c + \left( N_p - 1 \right)$ неизвестных. Следовательно, необходимо добавить к данной системе дополнительные уравнения. Выбор дополнительных уравнений зависит от известных параметров системы. Если помимо мольных долей компонентов в системе $z_i$ известны давление и температура, то используется изотермико-изобарический подход к определению равновесного состояния. Если вместо температуры известна энтальпия системы, то – изоэнтальпийно-изобарический подход.

# <a id='pvt-esc-equilibrium-isothermal'></a>
# ## Изотермико-изобарические подходы к определению равновесного состояния системы
# Среди изотермико-изобарических подходов к определению равновесного состояния рассмотрим метод последовательной подстановки (*successive substitution iteration method*) и метод, основанный на минимизации энергии Гиббса. На практике зачастую данные методы используются в комбинации, поскольку каждый из них характеризуется своими преимуществами и недостатками. Сначала методом последовательной подстановки корректируются начальные значения констант фазового равновесия (метод последовательной подстановки сравнительно медленно доводит решение до нужной точности), после чего, получив необходимой точности начальное приближение, решается задача минимизации энергии Гиббса (в ходе которой с использованием численных методов удается быстрее достигнуть нужной точности решения уравнений).

# <a id='pvt-esc-equilibrium-isothermal-ssi'></a>
# ### Метод последовательных подстановок

# <a id='pvt-esc-equilibrium-isothermal-ssi-equilibrium_condition'></a>
# Рассматривая квази-стационарный изотермический процесс, равновесное состояние системы определяется [равенством химических потенциалов](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium-rule). С другой стороны, для квази-стационарного изотермического процесса равенство химических потенциалов заменяется равенством летучестей (следует из выражения [дифференциала химического потенциала](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity)). Рассмотрим отношение коэффициентов летучести $i$-го компонента в $j$-ой и референсной $R$ фазах с учетом равновесного состояния системы:

# $$ \frac{\phi_i^j}{\phi_i^R} = \frac{f_i^j}{y_i^j P} \frac{y_i^R P}{f_i^R} = \frac{y_i^R}{y_i^j} = \frac{1}{K_i^{jR}}.$$

# Преобразуем данное выражение:

# $$ \ln K_i^{jR} + \ln \phi_i^j - \ln \phi_i^R = 0, \; i=1 \ldots N_c, \; j=1 \dots N_p - 1.$$

# Тогда система

# $$ \begin{cases} \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1} = 0, \; j=1 \ldots N_p-1; \\ \ln K_i^{jR} + \ln \phi_i^j - \ln \phi_i^R = 0, \; i=1 \ldots N_c, \; j=1 \dots N_p - 1. \end{cases} $$

# имеет $\left( N_p - 1 \right) N_c + \left( N_p - 1 \right)$ уравнений, необходимых для нахождения такого же количества неизвестных при условии, что коэффициенты летучести компонента в фазе $j$ определяются по уравнению состояния с учетом давления, температуры и компонентного состава фазы $j$. Для задач нефтегазовой отрасли данная система уравнений решается для трех фаз: газовой $j=g$, нефтяной $j=o$ и водной $j=w$ (иногда также рассматриваются четыре фазы – с учетом выпадение твердых веществ в осадок, например, парафинов или асфальтенов; такой пример будет рассмотрен в разделе, посвященном $\color{#E32636}{моделированию~выпадения~АСПО}$).

# Для того чтобы система уравнений Речфорда-Райса имело решение $F^j$ необходимо, чтобы хотя бы для одного компонента в фазе $j$ константа фазового равновесия $K_i^{jR} < 1$ и хотя бы для одного компонента в фазе $j$ константа фазового равновесия $K_i^{jR} > 1$. В противном случае сумма в уравнении Речфорда-Райса будет положительной (или отрицательной) и будет стремиться к нулю при $F^j \rightarrow \infty$ или при $F^j \rightarrow -\infty.$ Поскольку система Речфорда-Райса содержит нелинейные уравнения относительно $F^j$, то для ее решения необходимо использовать численные методы решения систем нелинейных уравнений, например, [метод Ньютона](../../0-Math/5-OM/OM-1-Newton.html#math-om-newton):

# $$ \begin{bmatrix} F^1 \\ F^2 \\ \vdots \\ F^j \end{bmatrix}^{k+1} = \begin{bmatrix} F^1 \\ F^2 \\ \vdots \\ F^j \end{bmatrix}^k - J^{-1} \begin{bmatrix} g^1 \\ g^2 \\ \vdots \\ g^j \end{bmatrix}^k, \; j=1 \ldots N_p-1, $$

# где $k$ – номер итерации, $g^j$ – уравнение [Речфорда-Райса](#pvt-esc-equilibrium-rachford_rice) для $j$-ой фазы, $J$ – якобиан, матрица $ \left( N_p - 1 \right) \times \left( N_p - 1 \right) $, определяемая следующим выражением:

# $$ J^{-1} = \begin{bmatrix} \frac{\partial g^1}{\partial F^1} & \frac{\partial g^1}{\partial F^2} & \cdots & \frac{\partial g^1}{\partial F^{N_p-1}} \\ \frac{\partial g^2}{\partial F^1} & \frac{\partial g^2}{\partial F^2} & \cdots & \frac{\partial g^2}{\partial F^{N_p-1}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial g^{N_p-1}}{\partial F^1} & \frac{\partial g^{N_p-1}}{\partial F^2} & \cdots & \frac{\partial g^{N_p-1}}{\partial F^{N_p-1}} \end{bmatrix}^{-1}. $$

# Получим частную производную $\frac{\partial g^j}{\partial F^l}, \; j = 1 \ldots N_p-1, \; l = 1 \ldots N_p-1$:

# $$ \begin{align}
# \frac{\partial g^j}{\partial F^l}
# &= \frac{\partial}{\partial F^l} \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1} \\
# &= \sum_{i=1}^{N_c} \frac{\partial}{\partial F^l} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1} \\
# &= \sum_{i=1}^{N_c} - \frac{z_i \left( K_i^{jR} - 1 \right)}{ \left( \sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1 \right)^2} \frac{\partial}{\partial F^l} \left( \sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1 \right) \\
# &= - \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right) \sum_{k=1}^{N_p - 1} \frac{\partial F^k}{\partial F^l} \left(K_i^{kR} - 1 \right) }{ \left( \sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1 \right)^2}.
# \end{align} $$

# Частная производная $\frac{\partial F^k}{\partial F^l}$ определяется аналогично рассмотренной [ранее](../2-EOS/EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-partials) частной производной количества вещества $i$-го компонента по количеству вещества $j$-го компонента. Вместо использования метода Ньютона для экономии вычислительных ресурсов (поскольку нахождение якобиана и обратной матрицы на каждой итерации может занимать большую часть расчетного времени) используются [квазиньютоновские методы](../../0-Math/5-OM/OM-2-QNewton.html#math-om-qnewton) ([Quasi-Newton methods](https://en.wikipedia.org/wiki/Quasi-Newton_method)), когда матрица якобиана заменяется аппроксимацией, получаемой по якобиану из нулевой итерации.

# <a id='pvt-esc-equilibrium-isothermal-ssi-algorithm'></a>
# Рассмотрим алгоритм последовательных подстановок для определения равновесного состояния системы.

# ```{prf:алгоритм}
# :nonumber:
# <a id='pvt-esc-equilibrium-isothermal-ssi-kvalues_init'></a>
# +++
# 
# На первом этапе осуществляется начальная оценка констант фазового равновесия компонентов. Начальные значения $K_i^{go}$ рассчитываются с использованием модифицированной корреляции Уилсона \[[Peng and Robinson, 1976](https://doi.org/10.1002/cjce.5450540541)\]:
# 
# +++
# 
# $$K_i^{go} = \frac{P}{{P_c}_i} e^{5.3727 \left( 1 + \omega_i \right) \left( 1 - \frac{T}{{T_c}_i} \right)},$$
# 
# +++
# 
# где $\omega_i$ – ацентрический фактор компонента, для расчета которого можно использовать корреляцию Эдмистера: 
# 
# +++
# 
# $\omega_i = \frac{3}{7} \frac{\ln {P_c}_i}{\frac{{T_c}_i}{{T_b}_i} - 1} - 1 ,$$
# 
# +++
# 
# где ${T_b}_i$ – температура кипения при атмосферном давлении. В данном уравнении единицы измерения критического давления компонента ${P_c}_i$ – атмосферы. Начальные значения $K_i^{gw}$ определяются с использованием следующего выражения \[[Peng and Robinson, 1976](https://doi.org/10.1002/cjce.5450540541)\]:
# 
# +++
# 
# $$ K_i^{gw} = 10^6 \frac{{P_c}_i}{P} \frac{T}{{T_c}_i}.$$
# 
# +++
# 
# Тогда:
# 
# +++
# 
# $$ K_i^{ow} = \frac{K_i^{gw}}{K_i^{go}}.$$
# 
# +++
# 
# Однако для воды, как компонента, такая начальная оценка констант фазового рановесия неприменима, поскольку будет давать сильно завышенные значения. Следовательно, для воды, как компонента начальные значения констант фазового равновесия $K_i^{gw}$ и $K_i^{ow}$ следует задавать в диапазоне $0 < K_i^{gw} \left( K_i^{ow} \right) < 1.$
# 
# +++
# 
# После задания начального предположения констант фазового равновесия решается система уравнений [Речфорда-Райса](#pvt-esc-equilibrium-rachford_rice).
# 
# +++
# 
# После этого мольные доли каждого компонента рассчитываются по [ранее](#pvt-esc-equilibrium-mole_fractions) полученным формулам.
# 
# +++
# 
# <a id='pvt-esc-equilibrium-isothermal-ssi-iteration'></a>
# Зная давление, температуру и компонентый состав каждой фазы, с использованием уравнения состояния определяются коэффициенты летучести компонентов и проверяется [условие равновесия](#pvt-esc-equilibrium-isothermal-ssi-equilibrium_condition). Если условие не выполняется, то константы фазого равновесия на следующей итерации обновляются, согласно следующему:
# 
# +++
# 
# $${K_i^{jR}}^{k+1} = {K_i^{jR}}^{k} \left( \frac{f_i^j}{f_i^R} \right)^{k},$$
# 
# +++
# 
# где $k$ – номер итерации. Данный подход к определению констант фазового равновесия компонентов называется методом последовательных подстановок (*successive substitution iterations*). В ряде работ \[[Mehra et al, 1982](https://doi.org/10.2118/9232-PA); [Mehra et al, 1983](https://doi.org/10.1002/cjce.5450610414)\] предлагается ускорение нахождения констант фазового равновесия в ходе итерационной процедуры путем введения параметра $\lambda$ в качестве показателя степени отношения летучестей компонентов в выражении выше. При этом на первом шаге $\lambda = 1.$
# ```

# <a id='pvt-esc-equilibrium-isothermal-example_1'></a>
# Рассмотрим данный алгоритм на следующем примере. Пусть имеется компонентный состав \[[Barrufet et al, 1996](https://doi.org/10.1021/je9600616)\], представленный в таблице:

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
#             <th class="tb-1pky" width="161px">Компонент</th>
#             <th class="tb-1pky" width="161px">Мольная доля<br>z<sub>i</sub></th>
#             <th class="tb-1pky" width="161px">Критическое давление, МПа<br>Pc<sub>i</sub></th>
#             <th class="tb-1pky" width="161px">Критическая температура, K<br>Tc<sub>i</sub></th>
#             <th class="tb-1pky" width="161px">Ацентрический фактор<br>&omega;<sub>i</sub></th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tb-abip">C<sub>3</sub>H<sub>8</sub></td>
#             <td class="tb-0pky">0.1292</td>
#             <td class="tb-0pky">4.246</td>
#             <td class="tb-0pky">369.8</td>
#             <td class="tb-0pky">0.152</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>5</sub>H<sub>12</sub></td>
#             <td class="tb-0pky">0.0544</td>
#             <td class="tb-0pky">3.374</td>
#             <td class="tb-0pky">469.6</td>
#             <td class="tb-0pky">0.251</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>8</sub>H<sub>18</sub></td>
#             <td class="tb-0pky">0.0567</td>
#             <td class="tb-0pky">2.951</td>
#             <td class="tb-0pky">570.5</td>
#             <td class="tb-0pky">0.351</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">H<sub>2</sub>O</td>
#             <td class="tb-0pky">0.7597</td>
#             <td class="tb-0pky">22.05</td>
#             <td class="tb-0pky">647.3</td>
#             <td class="tb-0pky">0.344</td>
#         </tr>
#     </tbody>
# </table>

# Необходимо определить компонентные составы фаз при температуре $448.0 \; K$ и давлении $5.15 \; МПа$. Для нахождения летучестей компонентов будем использовать [уравнение состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw). Коэффициенты попарного взаимодействия в водной фазе будут рассчитываться с использованием [корреляций](../2-EOS/EOS-3-SW.html#pvt-eos-sw), а в газовой и нефтяной – с использованием [GCM](../2-EOS/EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip-gcm). Предположим, что рассматриваемая система при заданных термобарических условиях раздялется на три фазы: нефтяную (жидкую преимущественно углеводородную), газовую и водную (жидкую преимущественно неуглеводородную). В качестве референсной фазы выберем водную фазу. Подробнее проблема определения количества фаз в системе будет рассматриваться в [следующем разделе](./ESC-2-Stability.html#pvt-esc-stability).

# Для решения данной задачи необходимо импортировать классы для расчета [параметров](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr-mix_rules) $\alpha_m$ и $b_m$, а также для расчета летучести компонентов с использованием уравнения состояния Сорейде-Уитсона:

# In[1]:


import numpy as np
import sys
sys.path.append("../../SupportCode/")
from PVT import core, eos_srk_pr, eos_sw, derivatives_eos_2param, parameters_2param, derivatives_parameters_2param


# In[2]:


P = 5.15 * 10**6
T = 448.0
cw = 0.0


# In[3]:


Pc = np.array([4.246, 3.374, 2.951, 22.05])*10**6
Tc = np.array([369.8, 469.6, 570.5, 647.3])
w = np.array([0.152, 0.251, 0.351, 0.344])
z = np.array([0.1292, 0.0544, 0.0567, 0.7597])
phases = 'ogw'
comp_type = np.array([1, 1, 1, 2])


# Для использования метода [GCM](../2-EOS/EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip-gcm) составим матрицу долей структурных групп в молекулах компонентов.

# <table class="tb">
#     <thead>
#         <tr>
#             <th class="tb-1pky" colspan="2" rowspan="2"></th>
#             <th class="tb-1pky" colspan="4">Группы</th>
#         </tr>
#         <tr>
#             <th class="tb-abip">CH<sub>3</sub></th>
#             <th class="tb-abip">CH<sub>2</sub></th>
#             <th class="tb-abip">H<sub>2</sub>O</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tb-1pky" rowspan="4" width="140px">Компоненты</td>
#             <td class="tb-abip" width="80px">C<sub>3</sub>H<sub>8</sub></td>
#             <td class="tb-0pky" width="195px">&alpha;<sub>11</sub> = 2 / 3</td>
#             <td class="tb-0pky" width="195px">&alpha;<sub>12</sub> = 1 / 3</td>
#             <td class="tb-0pky" width="195px">&alpha;<sub>13</sub> = 0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>5</sub>H<sub>12</sub></td>
#             <td class="tb-0pky">&alpha;<sub>21</sub> = 2 / 5</td>
#             <td class="tb-0pky">&alpha;<sub>22</sub> = 3 / 5</td>
#             <td class="tb-0pky">&alpha;<sub>23</sub> = 0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>8</sub>H<sub>18</sub></td>
#             <td class="tb-0pky">&alpha;<sub>31</sub> = 2 / 8</td>
#             <td class="tb-0pky">&alpha;<sub>32</sub> = 6 / 8</td>
#             <td class="tb-0pky">&alpha;<sub>33</sub> = 0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">H<sub>2</sub>O</td>
#             <td class="tb-0pky">&alpha;<sub>41</sub> = 0</td>
#             <td class="tb-0pky">&alpha;<sub>42</sub> = 0</td>
#             <td class="tb-0pky">&alpha;<sub>44</sub> = 1</td>
#         </tr>
#     </tbody>
# </table>

# In[4]:


alpha_matrix = np.array([[2/3, 1/3, 0.0], [2/5, 3/5, 0.0], [2/8, 6/8, 0.0], [0.0, 0.0, 1.0]])
alpha_matrix


# Также сформируем матрицы $A_{kl}$ и $B_{kl}$.

# In[5]:


import pandas as pd
df_a = pd.read_excel(io='../../SupportCode/BIPCoefficients.xlsx', sheet_name='A', usecols='D:Y', skiprows=[0, 1, 2], index_col=0)
df_b = pd.read_excel(io='../../SupportCode/BIPCoefficients.xlsx', sheet_name='B', usecols='D:Y', skiprows=[0, 1, 2], index_col=0)


# In[6]:


groups = [1, 2, 21]
Akl = df_a.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
Bkl = df_b.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
Akl, Bkl


# In[7]:


mr1 = eos_sw(Pc, Tc, w, comp_type, Akl=Akl, Bkl=Bkl, alpha_matrix=alpha_matrix)


# Теперь создадим класс для расчета равновесных мольных долей компонентов.

# In[8]:


class flash_isothermal_ssi(core):
    def __init__(self, mr, z, ssi_rr_eps=1e-8, ssi_eq_eps=1e-8, ssi_use_opt=False, qnss=False, ssi_negative_flash=False, ssi_eq_max_iter=30, save_residuals=False):
        self.z = z
        self.mr = mr
        self.ssi_rr_eps = ssi_rr_eps
        self.ssi_eq_eps = ssi_eq_eps
        self.ssi_use_opt = ssi_use_opt
        self.qnss = qnss
        self.ssi_negative_flash = ssi_negative_flash
        self.ssi_eq_max_iter = ssi_eq_max_iter
        self.save_residuals = save_residuals
        pass

    def k_values_init(self, P, T, phases):
        kv_go = self.mr.Pc * np.exp(5.3727 * (1 + self.mr.w) * (1 - self.mr.Tc / T)) / P
        kv_gw = np.where(self.mr.comp_type == 2, 0.1, 10**6 * self.mr.Pc * T / (P * self.mr.Tc))
        kv_ow = np.where(self.mr.comp_type == 2, 0.01, kv_gw / kv_go)
        if phases == 'ogw':
            return np.array([kv_ow, kv_gw])
        elif phases == 'gow':
            return np.array([kv_gw, kv_ow])
        elif phases == 'gwo':
            return np.array([kv_gw / kv_ow, 1 / kv_ow])
        elif phases == 'wgo':
            return np.array([1 / kv_ow, kv_gw / kv_ow])
        elif phases == 'wog':
            return np.array([1 / kv_gw, kv_ow / kv_gw])
        elif phases == 'owg':
            return np.array([kv_ow / kv_gw, 1 / kv_gw])
        elif phases == 'og':
            return np.array([kv_ow / kv_gw])
        elif phases == 'go':
            return np.array([kv_gw / kv_ow])
        elif phases == 'gw':
            return np.array([kv_gw])
        elif phases == 'wg':
            return np.array([1 / kv_gw])
        elif phases == 'wo':
            return np.array([1 / kv_ow])
        elif phases == 'ow':
            return np.array([kv_ow])

    def rachford_rice(self, F, kv):
        return np.sum(self.z * (kv - 1) / (1 + np.sum(F * (kv - 1), axis=0)), axis=1).reshape(self.Np_1, 1)

    def rachford_rice_jacobian(self, F, kv):
        return (-1) * np.sum(self.repeat(self.repeat(self.z, 0, times=self.Np_1), 1, times=self.Np_1) * self.repeat(kv - 1, 0, times=self.Np_1) *                              self.repeat(np.sum(self.repeat(np.identity(self.Np_1), 2, times=self.mr.Nc) * self.repeat(kv - 1, 0, times=self.Np_1), axis=1), axis=1) / 
                             self.repeat(self.repeat((1 + np.sum(F * (kv - 1), axis=0))**2, 0, times=self.Np_1), 1, times=self.Np_1), axis=2)

    def rachford_rice_newton(self, x0, kv):
        x = x0.copy()
        rr_eq_val = np.ones_like(x0)
        while np.all(np.abs(rr_eq_val) > self.ssi_rr_eps):
            rr_eq_val = self.rachford_rice(x, kv)
            if self.Np_1 > 1:
                x = x - np.linalg.inv(self.rachford_rice_jacobian(x, kv)).dot(rr_eq_val)
            else:
                x = x - rr_eq_val / self.rachford_rice_jacobian(x, kv)
        return x

    def rr_negative_limits(self, kv):
        return 1 / (1 - np.amax(kv, axis=1).reshape(self.Np_1, 1)), 1 / (1 - np.amin(kv, axis=1).reshape(self.Np_1, 1))

    def full_F(self, F):
        return np.append(F, 1 - np.sum(F, 0)).reshape(self.Np_1 + 1, 1)

    def flash_isothermal_ssi_run(self, P, T, phases, kv0=None, **kwargs):
        if kv0 is None:
            kv = self.k_values_init(P, T, phases)
            self.Np_1 = len(phases) - 1
        else:
            kv = kv0
            self.Np_1 = len(kv0)
        F = np.array([self.Np_1 * [1 / (self.Np_1 + 1)]]).reshape(self.Np_1, 1)
        residuals = np.ones(shape=(self.Np_1, self.mr.Nc))
        it = 0
        lambda_pow = 1
        Fmin = np.zeros_like(F)
        Fmax = np.ones_like(F)
        res = self.empty_object()
        if self.save_residuals:
            res.residuals_all = []
        while np.any(np.abs(residuals) > self.ssi_eq_eps) and it < self.ssi_eq_max_iter:
            F = self.rachford_rice_newton(F, kv)
            if self.ssi_negative_flash:
                Fmin, Fmax = self.rr_negative_limits(kv)
            F = np.where(F < Fmin, Fmin, F)
            F = np.where(F > Fmax, Fmax, F)
            y = self.z * kv / (1 + np.sum(F * (kv - 1), axis=0))
            yref = self.z / (1 + np.sum(F * (kv - 1), axis=0))
            eos = self.mr.eos_run(np.append(y, np.array([yref]), axis=0), P, T, phases, **kwargs)
            if self.ssi_use_opt:
                residuals_prev = residuals
                residuals = np.log(kv) + eos.lnphi[:-1] - self.repeat(eos.lnphi[-1], axis=0, times=self.Np_1)
                lambda_pow = - lambda_pow * np.sum(residuals_prev**2) / (np.sum(residuals_prev * residuals) - np.sum(residuals_prev**2))
            else:
                residuals = np.log(kv) + eos.lnphi[:-1] - self.repeat(eos.lnphi[-1], axis=0, times=self.Np_1)
            kv = kv * (self.repeat(eos.f[-1], axis=0, times=self.Np_1) / eos.f[:-1])**lambda_pow
            it += 1
            if self.save_residuals:
                res.residuals_all.append(np.max(np.abs(residuals)))
        res.kv = kv
        res.it = it
        res.y = np.append(self.z * kv / (1 + np.sum(F * (kv - 1), axis=0)), np.array([self.z / (1 + np.sum(F * (kv - 1), axis=0))]), axis=0)
        res.eos = eos
        res.F = self.full_F(F)
        res.comp_mole_frac = dict(zip(phases, res.y))
        res.phase_mole_frac = dict(zip(phases, res.F.ravel()))
        res.residuals = residuals
        return res


# In[9]:


flash1 = flash_isothermal_ssi(mr1, z, ssi_use_opt=True, ssi_eq_max_iter=30, save_residuals=True).flash_isothermal_ssi_run(P, T, phases, cw=0.0)
flash1.comp_mole_frac, flash1.it


# In[10]:


flash1.residuals


# Сопоставление расчетных и экспериментальных мольных долей фаз представлено на рисунке ниже.

# In[11]:


from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({'figure.max_open_warning': False})
get_ipython().run_line_magic('matplotlib', 'widget')
fig1, ax1 = plt.subplots(1, 3, figsize=(10, 4))
fig1.canvas.header_visible = False

y_exp = [[0.37724, 0.23217, 0.27938, 0.11121],
         [0.68519, 0.11729, 0.04293, 0.15459],
         [1.76E-03, 7.69E-04, 4.40E-06, 9.97E-01]]

width = 0.9
pos = np.linspace(0.0, 4 * 2 * width, 4)

ax1[0].bar(pos, y_exp[0], align='edge', width=width, zorder=3, label='Экспериментальные\nданные')
ax1[0].bar(pos + width, flash1.comp_mole_frac['o'], align='edge', width=width, zorder=3, label='Расчетные\nданные')
ax1[0].set_xticks(pos + width)
ax1[0].set_xticklabels(['$C_3H_8$', '$C_5H_{12}$', '$C_8H_{18}$', '$H_2O$'])
ax1[0].grid(zorder=0)
ax1[0].set_ylabel('Мольная доля компонента в нефтяной фазе')
ax1[0].set_ylim(0.0, 0.5)

ax1[1].bar(pos, y_exp[1], align='edge', width=width, zorder=3, label='Экспериментальные\nданные')
ax1[1].bar(pos + width, flash1.comp_mole_frac['g'], align='edge', width=width, zorder=3, label='Расчетные\nданные')
ax1[1].set_xticks(pos + width)
ax1[1].set_xticklabels(['$C_3H_8$', '$C_5H_{12}$', '$C_8H_{18}$', '$H_2O$'])
ax1[1].grid(zorder=0)
ax1[1].legend(loc='best')
ax1[1].set_ylabel('Мольная доля компонента в газовой фазе')
ax1[1].set_ylim(0.0, 0.7)

ax1[2].bar(pos, y_exp[2], align='edge', width=width, zorder=3, label='Экспериментальные\nданные')
ax1[2].bar(pos + width, flash1.comp_mole_frac['w'], align='edge', width=width, zorder=3, label='Расчетные\nданные')
ax1[2].set_xticks(pos + width)
ax1[2].set_xticklabels(['$C_3H_8$', '$C_5H_{12}$', '$C_8H_{18}$', '$H_2O$'])
ax1[2].grid(zorder=0)
ax1[2].set_ylabel('Мольная доля компонента в водной фазе')
ax1[2].set_ylim(0.0, 1.0)

fig1.tight_layout()

ax12_det = fig1.add_axes([.785, .2, .135, .7])
ax12_det.bar(pos, y_exp[2], align='edge', width=width, zorder=3, label='Экспериментальные\nданные')
ax12_det.bar(pos + width, flash1.comp_mole_frac['w'], align='edge', width=width, zorder=3, label='Расчетные\nданные')
ax12_det.set_xticks(pos + width)
ax12_det.set_xticklabels(['$C_3H_8$', '$C_5H_{12}$', '$C_8H_{18}$', '$H_2O$'])
ax12_det.grid(zorder=0)
ax12_det.set_ylim(0.0, 2e-3)
ax12_det.set_xlim(-width/2, pos[-1])
ax12_det.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
ax12_det.tick_params(axis='y', which='major', labelsize=8)
ax12_det.set_yticks(np.linspace(0.0, 2e-3, 5));


# <a id='pvt-esc-equilibrium-isothermal-example_2'></a>
# Таким образом, сопоставляя расчетные и фактические данные, можно сделать вывод о том, что предложенных подход может быть применим для оценки трехфазного равновесного состояния. Недостатком данного подхода является необходимость достаточно точного опреления начальных констант фазового равновесия компонентов, а также выбор (и [обоснование](ESC-2-Stability.html#pvt-esc-stability)) референсной фазы. Кроме того, решение [уравнения Речфорда-Райса](#pvt-esc-equilibrium-rachford_rice) может привести к появлению нефизичных корней. В качестве примера рассмотрим следующий компонентный состав:

# <table class="tb">
#     <thead>
#         <tr>
#             <th class="tb-1pky" width="161px">Компонент</th>
#             <th class="tb-1pky" width="161px">Мольная доля<br>z<sub>i</sub></th>
#             <th class="tb-1pky" width="161px">Критическое давление, МПа<br>Pc<sub>i</sub></th>
#             <th class="tb-1pky" width="161px">Критическая температура, K<br>Tc<sub>i</sub></th>
#             <th class="tb-1pky" width="161px">Ацентрический фактор<br>&omega;<sub>i</sub></th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tb-abip">CH<sub>4</sub></td>
#             <td class="tb-0pky">0.3</td>
#             <td class="tb-0pky">4.600</td>
#             <td class="tb-0pky">190.6</td>
#             <td class="tb-0pky">0.008</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>5</sub>H<sub>12</sub></td>
#             <td class="tb-0pky">0.3</td>
#             <td class="tb-0pky">3.374</td>
#             <td class="tb-0pky">469.6</td>
#             <td class="tb-0pky">0.251</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>16</sub>H<sub>34</sub></td>
#             <td class="tb-0pky">0.4</td>
#             <td class="tb-0pky">1.738</td>
#             <td class="tb-0pky">734.5</td>
#             <td class="tb-0pky">0.684</td>
#         </tr>
#     </tbody>
# </table>

# Необходимо определить компонентные составы фаз при температуре $373.15 \; K$ и давлении $5.0 \; МПа$. Поскольку в данной задаче отсутствует водная фаза, то будем использовать уравнение состояния [Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr).

# In[12]:


P = 5.0 * 10**6
T = 373.15


# In[13]:


Pc = np.array([4.6, 3.374, 1.738])*10**6
Tc = np.array([190.6, 469.6, 734.5])
w = np.array([0.008, 0.251, 0.684])
z = np.array([0.3, 0.3, 0.4])
phases = 'go'
comp_type = np.array([0, 0, 0])


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
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tb-1pky" rowspan="4" width="140px">Компоненты</td>
#             <td class="tb-abip" width="80px">CH<sub>4</sub></td>
#             <td class="tb-0pky" width="195px">&alpha;<sub>11</sub> = 0</td>
#             <td class="tb-0pky" width="195px">&alpha;<sub>12</sub> = 0</td>
#             <td class="tb-0pky" width="195px">&alpha;<sub>13</sub> = 1</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>5</sub>H<sub>12</sub></td>
#             <td class="tb-0pky">&alpha;<sub>21</sub> = 2 / 5</td>
#             <td class="tb-0pky">&alpha;<sub>22</sub> = 3 / 5</td>
#             <td class="tb-0pky">&alpha;<sub>23</sub> = 0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">C<sub>16</sub>H<sub>34</sub></td>
#             <td class="tb-0pky">&alpha;<sub>31</sub> = 2 / 16</td>
#             <td class="tb-0pky">&alpha;<sub>32</sub> = 14 / 16</td>
#             <td class="tb-0pky">&alpha;<sub>33</sub> = 0</td>
#         </tr>
#     </tbody>
# </table>

# In[14]:


alpha_matrix = np.array([[0.0, 0.0, 1.0], [2/5, 3/5, 0.0], [2/16, 14/8, 0.0]])
alpha_matrix


# In[15]:


groups = [1, 2, 5]
Akl = df_a.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
Bkl = df_b.loc[groups, groups].to_numpy(dtype=np.float64) * 10**6
Akl, Bkl


# In[16]:


mr2 = eos_srk_pr(Pc, Tc, w, comp_type, c=1, Akl=Akl, Bkl=Bkl, alpha_matrix=alpha_matrix)


# In[17]:


flash2 = flash_isothermal_ssi(mr2, z, ssi_use_opt=False, ssi_eq_max_iter=30).flash_isothermal_ssi_run(P, T, phases)
flash2.comp_mole_frac, flash2.it


# In[18]:


flash2.residuals


# С учетом найденных констант фазового равновесия построим график зависимости уравнения [уравнения Речфорда-Райса](#pvt-esc-equilibrium-rachford_rice) от мольной доли газовой фазы.

# In[19]:


Fv = np.arange(-1.5, 2.5, 0.001)
rr_eq = np.sum(z * (flash2.kv[0] - 1) / (1 + np.outer(Fv, flash2.kv[0] - 1)), axis=1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
fig2.canvas.header_visible = False

ax2.plot(Fv, rr_eq, label='Уравнение Речфорда-Райса')
ax2.set_ylim(-10, 10)
ax2.grid()
ax2.set_ylabel('Значение уравнения Речфорда-Райса')
ax2.set_xlabel('Мольная доля газовой фазы')
ax2.fill([0.0, 0.0, 1.0, 1.0], [-10.0, 10.0, 10.0, -10.0], c='g', alpha=0.1)

asymp = 1 / (1 - flash2.kv[0])

ax2.plot([asymp[0], asymp[0]], [-10.0, 10.0], label=r'Асимптота $\frac{1}{1 - K_1}$')
ax2.plot([asymp[1], asymp[1]], [-10.0, 10.0], label=r'Асимптота $\frac{1}{1 - K_2}$')
ax2.plot([asymp[2], asymp[2]], [-10.0, 10.0], label=r'Асимптота $\frac{1}{1 - K_3}$')

ax2.legend(loc='best')
fig2.tight_layout()


# <a id='pvt-esc-equilibrium-isothermal-ssi-negative_flash'></a>
# Таким образом, для $N_c$-компонентной системы существует $N_c - 1$ решений [уравнения Речфорда-Райса](#pvt-esc-equilibrium-rachford_rice) на $N_c - 1$ интервалах, разделенных $N_c$ асимптотами. При этом, только один корень, находящийся на отрезке $\left[ 0, 1 \right]$ удовлетворяет физическому смыслу мольной доли фазы. Если в ходе решения уравнения Речфорда-Райса численным методом определяется корень, не удовлетворяющий физическому смыслу мольной доли фазы, то это значений необходимо скорректировать: если корень меньше нуля, то следует приравнять его к нулю, если же корень больше единицы, то – к единице. В этом случае итерационный подход определения равновесного состояния называется положительным (*positive flash calculations*). Если же коррекция корней уравнения Речфорда-Райса определяется отрезком $\left[ \frac{1}{1 - K_{max}}, \frac{1}{1 - K_{min}} \right]$, то итерационный подход определения равновесного состояния называется отрицательным (*negative flash calculations*) \[[Whitson and Michelsen, 1989](https://doi.org/10.1016/0378-3812(89)80072-X)\]. Поскольку *negative flash calculations* не приводят к появлению отрицательных мольных долей компонентов, то после коррекции значения мольной доли фазы определяются мольные доли компонентов в них (по [ранее](#pvt-esc-equilibrium-mole_fractions) полученным формулам) и рассчитываются коэффициенты летучести компонентов в фазах, то есть полностью повторяется алгоритм, изложенный [выше](#pvt-esc-equilibrium-isothermal-ssi-algorithm). Отрицательный итерационный подход определения равновесного состояния системы может использоваться для определения констант фазового равновесия компонентов в однофазной области, а также для ряда других алгоритмов, в том числе определения давления минимальной смешиваемости $\color{#E32636}{!!!}$ \[[Yan et al, 2014](https://doi.org/10.1021/ie5012454)\].

# Для рассмотренного [выше](#pvt-esc-equilibrium-isothermal-example_2) примера выполним два определения равновесного состояния: один с учетом *negative flash calculations*, другой – без.

# In[20]:


P = 12.0 * 10**6
T = 323.15


# Определение равновесного состояния системы без учета *negative flash calculations*:

# In[21]:


flash3 = flash_isothermal_ssi(mr2, z, ssi_use_opt=False, ssi_eq_max_iter=30, ssi_negative_flash=False).flash_isothermal_ssi_run(P, T, phases)
flash3.phase_mole_frac, flash3.it


# In[22]:


flash3.residuals


# In[23]:


flash3.kv


# Определение равновесного состояния системы с учетом *negative flash calculations*:

# In[24]:


flash4 = flash_isothermal_ssi(mr2, z, ssi_use_opt=False, ssi_eq_max_iter=30, ssi_negative_flash=True).flash_isothermal_ssi_run(P, T, phases=phases)
flash4.phase_mole_frac, flash4.it


# In[25]:


flash4.residuals


# In[26]:


flash4.kv


# При *negative flash calculations* значение мольной доли газовой фазы получилось отрицательным, несмотря на это в ходе алгоритма удалось достичь равновесного состояния системы. Используя *negative flash calculations*, построим зависимости констант фазового равновесия от давления при фиксированной температуре.

# In[27]:


flash4_kv = flash_isothermal_ssi(mr2, z, ssi_use_opt=True, ssi_eq_max_iter=100, ssi_negative_flash=True)

P_range = np.exp(np.linspace(np.log(101325), np.log(40*10**6), 50))
kv_table = []
y_table = []

kv = flash4_kv.k_values_init(P_range[0], T, phases)

for P in P_range:
    fl = flash4_kv.flash_isothermal_ssi_run(P, T, phases, kv0=kv)
    kv = fl.kv
    y_table.append(fl.y)
    kv_table.append(kv[0])

kv_table = np.array(kv_table)
y_table = np.array(y_table)

fig3, ax3 = plt.subplots(1, 3, figsize=(8, 3))
fig3.canvas.header_visible = False

for i, comp in enumerate(['$CH_4$', '$C_5H_{12}$', '$C_{16}H_{34}$']):
    ax3[0].plot(P_range / 10**6, kv_table.T[i], label=comp)
    ax3[1].plot(P_range / 10**6, y_table.T[i][0], label=comp)
    ax3[2].plot(P_range / 10**6, y_table.T[i][1], label=comp)

ax3[0].set_xscale('log')
ax3[0].set_yscale('log')
ax3[0].grid()
ax3[1].grid()
ax3[2].grid()
ax3[1].set_xlim(0.0, 40)
ax3[2].set_xlim(0.0, 40)
ax3[1].set_ylim(0.0, 1.0)
ax3[2].set_ylim(0.0, 1.0)
ax3[0].set_xlabel('Давление, МПа')
ax3[1].set_xlabel('Давление, МПа')
ax3[2].set_xlabel('Давление, МПа')
ax3[0].set_ylabel('Константа фазового равновесия')
ax3[1].set_ylabel('Мольная доля компонента\nв газовой фазе')
ax3[2].set_ylabel('Мольная доля компонента\nв нефтяной фазе')
ax3[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

fig3.tight_layout()


# Из данного рисунка видно, что с увеличением давления константы фазого равновесия всех компонентов стремятся к единице, следовательно, мольные доли компонентов в каждой фазе становятся равными друг другу.

# ```{prf:определение}
# :nonumber:
# Такое давление, при котором константы фазого равновесия всех компонентов сходятся в единице называется ***давлением сходимости***.
# ```

# Давление сходимости, по своей сути, является границей применимости *negative flash calculations*. По аналогии с давлением сходимости, существует также *температура сходимости*:
# 
# ```{prf:определение}
# :nonumber:
# ***Температура сходимости*** – это такая температура, при которой константы фазового равновесия компонентов становятся равными единице.
# ```

# Более подробно вопрос определения границы сходимости будет рассматриваться в разделе, посвященном [критическому состоянию системы](./ESC-3-Criticality.html#pvt-esc-criticality).

# Таким образом, [метод последовательных подстановок](#pvt-esc-equilibrium-isothermal-ssi) может быть использован для определения равновесного состояния системы. При этом, данный метод характеризуется медленной сходимостью к необходимой точности. На рисунке ниже представлена динамика максимальной абсолютной ошибки [уравнений равновесия компонентов системы](#pvt-esc-equilibrium-isothermal-ssi-equilibrium_condition) для [первого примера](#pvt-esc-equilibrium-isothermal-example_1) (с учетом оптимизации в виде расчета [степени](#pvt-esc-equilibrium-isothermal-ssi-iteration) $\lambda$).

# In[28]:


iters_ssi = np.arange(1, flash1.it + 1, 1)

from matplotlib.ticker import LogLocator

fig4, ax4 = plt.subplots(figsize=(8, 6))
fig4.canvas.header_visible = False

ax4.plot(iters_ssi, flash1.residuals_all)
ax4.set_ylabel('Максимальная абсолютная ошибка\nуравнений равновесного состояния')
ax4.set_xlabel('Итерации')
ax4.set_yscale('log')
ax4.set_ylim(1e-9, 100)
ax4.set_xlim(0.0, 16)
ax4.set_yticks(10**np.arange(-8.0, 2.0, 1.0))
ax4.grid(True, which='major', axis='y')
ax4.grid(True, which='minor', axis='y', ls='--', lw=0.5)
ax4.grid(True, which='major', axis='x')
ax4.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12))

fig4.tight_layout()


# <a id='pvt-esc-equilibrium-isothermal-gem'></a>
# ### Метод минимизации энергии Гиббса
# С другой стороны, к решению определения равновесного состояния с учетом изотермического процесса можно подойти с точки зрения алгоритма оптимизации, а именно – минимизации функции энергии Гиббса системы. Среди численных методов оптимизации рассмотрим оптимизацию [методом Ньютона](../../0-Math/5-OM/OM-1-Newton.html#math-om-newton). Использование [туннельного метода](../../0-Math/5-OM/OM-4-Tunneling.html#math-om-tunneling) определению глобального минимума функции энергии Гиббса будет рассматриваться в [следующем разделе](ESC-2-Stability.html#pvt-esc-stability).

# [Ранее](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity) было показано, что дифференциал энергия Гиббса однокомпонентной системы определяется выражением:

# $$dG = nRT d \ln f.$$

# Поскольку энергия Гиббса является [экстенсивным параметром](../1-TD/TD-9-Observables.html#pvt-td-observables-extensive), то для многокомпонентной системы можно записать:

# $$dG = \sum_{i=1}^{N_c} n_i RT d \ln f_i = RT \sum_{i=1}^{N_c} n_i d \ln f_i .$$

# Интегрируя данное выражение, получим:

# $$G - G_0 = RT \sum_{i=1}^{N_c} n_i \ln f_i - RT \sum_{i=1}^{N_c} n_i \ln {f_i}_0.$$

# Следовательно,

# $$G = RT \sum_{i=1}^{N_c} n_i \ln f_i.$$

# Введем понятие приведенной энергии Гиббса, которая также является [экстенсивным параметром](../1-TD/TD-9-Observables.html#pvt-td-observables-extensive):

# $$\tilde{G} = \frac{G}{RT}.$$

# Тогда:

# $$\tilde{G} = \sum_{i=1}^{N_c} n_i \ln f_i.$$

# Энергия Гиббса многофазной системы равна сумме энергий Гиббса каждой фазы:

# $$\tilde{G} = \sum_{j=1}^{N_p} \tilde{G}^j = \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} n_i^j \ln f_i^j.$$

# В точке глобального минимума функции:

# $$g_l^k = \frac{\partial \tilde{G}}{\partial n_l^k} = 0, \; k=1 \ldots N_p-1, \; l=1 \dots N_c.$$

# В данном выражении переменные $n_l^k, \; k=1 \ldots N_p-1, \; l=1 \dots N_c$ являются независимыми, следовательно, относительно них записывается условие достижения минимума функции. Количество вещества компонента в фазе $N_p$ (будем считать ее референсной) является зависимой переменной относительно количества вещества в фазах $1 \ldots N_p-1$ и определяется следующим выражением:

# $$n_l^{N_p} = n z_l - \sum_{k=1}^{N_p-1} n_l^k.$$

# Следовательно, система уравнений, включающая $\left( N_p - 1 \right) N_c$ неизвестных и $\left( N_p - 1 \right) N_c$ уравнений:

# $$ g_l^k = 0, \; k=1 \ldots N_p-1, \; l=1 \dots N_c,$$

# является разрешимой относительно количеств веществ $N_c$ компонентов в $\left( N_p - 1 \right)$ фазах $n_l^k$. Таким образом, задачу поиска минимума функции можно заменить на задачу решения системы нелинейных уравнений равенства нулю частных производных функции энергии Гиббса по независимым переменным.

# <a id='pvt-esc-equilibrium-isothermal-gem-hessian'></a>
# Однако равенство нулю частных производных функции нескольких переменных является необходимым, но недостаточным условием, поскольку оно определяет наличие локального экстремума или [седловой точки](https://en.wikipedia.org/wiki/Saddle_point). В общем виде, локальный минимум функции [характеризуется](https://en.wikipedia.org/wiki/Second_partial_derivative_test#Functions_of_many_variables) равенством нулю частных производных функции и положительными [собственными значениями](../../0-Math/1-LAB/LAB-7-EigenValues-Eigenvectors.html#math-lab-eigen) (*eigenvalues*) [гессианов](https://en.wikipedia.org/wiki/Hessian_matrix):

# $$ H = \begin{bmatrix} \frac{\partial^2 \tilde{G}}{\partial {n_l^1}^2} & \frac{\partial^2 \tilde{G}}{\partial n_l^1 \partial n_i^2} & \cdots & \frac{\partial^2 \tilde{G}}{\partial n_l^1 \partial n_l^{N_p-1}} \\ \frac{\partial^2 \tilde{G}}{\partial n_l^2 \partial n_i^1} & \frac{\partial^2 \tilde{G}}{\partial {n_l^2}^2} & \cdots & \frac{\partial^2 \tilde{G}}{\partial n_l^2 \partial n_l^{N_p-1}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 \tilde{G}}{\partial n_l^1 \partial n_l^{N_p-1}} & \frac{\partial^2 \tilde{G}}{\partial n_l^2 \partial n_l^{N_p-1}} & \cdots & \frac{\partial^2 \tilde{G}}{\partial {n_l^{N_p-1}}^2} \end{bmatrix}, \; l = 1 \ldots N_c.$$

# Или же его можно записать в следующем виде:

# $$ \vec{H} = \frac{\partial \vec{g}}{\partial \vec{n}} = \begin{bmatrix} \frac{\partial \vec{g^1}}{\partial \vec{n^1}} & \frac{\partial \vec{g^1}}{\partial \vec{n^2}} & \cdots & \frac{\partial \vec{g^1}}{\partial \overrightarrow{n^{N_p-1}}} \\ \frac{\partial \vec{g^2}}{\partial \vec{n^1}} & \frac{\partial \vec{g^2}}{\partial \vec{n^2}} & \cdots & \frac{\partial \vec{g^2}}{\partial \overrightarrow{n^{N_p-1}}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial \overrightarrow{g^{N_p-1}}}{\partial \vec{n^1}} & \frac{\partial \overrightarrow{g^{N_p-1}}}{\partial \vec{n^2}} & \cdots & \frac{\partial \overrightarrow{g^{N_p-1}}}{\partial \overrightarrow{n^{N_p-1}}} \end{bmatrix}, $$

# где $\vec{g^k} = \begin{bmatrix} g^k_1 & g^k_2 & \ldots & g^k_{N_c} \end{bmatrix}, \; k=1 \ldots N_p-1$ и $\vec{n^k} = \begin{bmatrix} n^k_1 & n^k_2 & \ldots & n^k_{N_c} \end{bmatrix}, \; k=1 \ldots N_p-1$. Таким образом, $H$ имеет размерность $\left( N_p-1 \right) \times N_c \times \left( N_p-1 \right) \times N_c.$

# <a id='pvt-esc-equilibrium-isothermal-gem-fugacity_equlaity'></a>
# Рассмотрим подробнее функцию $g_l^k$:

# $$ \begin{align}
# g_l^k
# &= \frac{\partial \tilde{G}}{\partial n_l^k} \\
# &= \frac{\partial}{\partial n_l^k} \left( \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} n_i^j \ln f_i^j \right) \\
# &= \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} \left( \ln f_i^j \frac{\partial n_l^j}{\partial n_l^k} + n_i^j \frac{\partial \ln f_i^j}{\partial n_l^k} \right) \\
# &= \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} \ln f_i^j \frac{\partial n_i^j}{\partial n_l^k} + \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} n_i^j \frac{\partial \ln f_i^j}{\partial n_l^k}, k = 1 \dots N_p-1.
# \end{align} $$

# Поскольку летучесть компонента в фазе определяется компонентным составом самой фазы, то:

# $$ \frac{\partial \ln f_i^j}{\partial n_l^k} = 0, \; k \neq j.$$

# Тогда рассмотрим следующее выражение:

# $$\sum_{i=1}^{N_c} n_i^j \frac{\partial \ln f_i^j}{\partial n_l^j}.$$

# Докажем, что данная сумма будет равна нулю в равновесном состоянии. Рассматривая равновесное состояние при фиксированных давлении и температуре [дифференциал энергии Гиббса](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy):

# $$ dG = \sum_{i=1}^{N_c} \mu_i dn_i + \sum_{i=1}^{N_c} n_i d \mu_i.$$

# Поскольку химический потенциал компонента [определяется](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy) давлением и температурой, то второе слагаемое данного выражения равно нулю:

# $$\sum_{i=1}^{N_c} n_i d \mu_i = 0.$$

# Дифференциал химического потенциала $i$-го компонента [можно выразить](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity) через его летучесть, тогда:

# $$\sum_{i=1}^{N_c} n_i d \ln f_i = 0.$$

# Разделим левую и правую часть уравнения на $d n_l$:

# $$\sum_{i=1}^{N_c} n_i \frac{\partial \ln f_i}{\partial n_l} = 0.$$

# С учетом изложенного выше:

# $$g_l^k = \frac{\partial \tilde{G}}{\partial n_l^k} = \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} \ln f_i^j \frac{\partial n_i^j}{\partial n_l^k}.$$

# Рассмотрим значения частной производной $\frac{\partial n_i^j}{\partial n_l^k}$:

# $$\frac{\partial n_i^j}{\partial n_l^k} = \begin{cases} 1, \; i=l, \; j=k, \; j= 1 \ldots N_p - 1, \; k=1 \ldots N_p-1; \\ -1, \; i = l, \; j = N_p, \; k = 1 \ldots N_p - 1; \\ 0, \; otherwise. \end{cases}$$

# То есть, если частная производная $\frac{\partial n_i^j}{\partial n_l^k}$ записывается для одного и того же компонента в одной и той же фазе, то она равняется единице. Если же частная производная записывается одного и того же компонента для референсной фазы, то с учетом $n_i^{N_p} = n z_i - \sum_{j=1}^{N_p-1} n_i^j$ частная производная равняется $-1$. В остальных случаях частная производная равняется нулю, поскольку либо рассматриваются разные фазы, либо независимые переменные. Тогда $g_l^k$ можно записать в следующем виде:

# $$g_l^k = \ln f_l^k - \ln f_l^{N_p}, \; l = 1 \ldots N_c, \; k = 1 \ldots N_p-1.$$

# Таким образом, частная производная энергии Гиббса по количеству вещества компонентов $l = 1 \dots N_c$ в фазах $k = 1 \ldots N_p - 1$ равняется разнице между логарифмами летучестей компонентов $l$ в фазе $k$ и в референсной фазе $N_p$.

# С учетом этого преобразуем значения элементов [гессиана](#pvt-esc-equilibrium-isothermal-gem-hessian):

# $$ \begin{align}
# \frac{\partial g_l^k}{\partial n_i^j}
# &= \frac{\partial \ln f_l^k}{\partial n_i^j} - \frac{\partial \ln f_l^{N_p}}{\partial n_i^j} \\
# &= \frac{1}{f_l^k} \frac{\partial f_l^k}{\partial n_i^j} - \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^j} \\
# &= \frac{1}{f_l^k} \frac{\partial f_l^k}{\partial n_i^j} - \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}} \frac{\partial n_i^{N_p}}{\partial n_i^j} \\
# &= \frac{1}{f_l^k} \frac{\partial f_l^k}{\partial n_i^j} + \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}}, \; k = 1 \ldots N_p - 1, \; j = 1 \ldots N_p - 1, \; l = 1 \ldots N_c, \; i = 1 \ldots N_c.
# \end{align} $$

# При этом, если $k \neq j$, то:

# $$ \frac{\partial g_l^k}{\partial n_i^j} = \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}}.$$

# Выражения для частных производных логарифма летучести компонента с использованием различных уравнений состояния были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd).

# <a id='pvt-esc-equilibrium-isothermal-gem-jacobian'></a>
# Производная энергии Гиббса по количеству вещества компонентов $\frac{\partial \tilde{G}}{\partial n_l^k} = g_l^k, \; k=1 \ldots N_p-1, \; l=1 \dots N_c$ представляет собой двумерную матрицу размерностью $\left( N_p - 1 \right) \times N_c.$ Для решения уравнения $g_l^k = 0$ используется гессиан $H = \frac{\partial g_l^k}{\partial n_i^j}, \;k = 1 \ldots N_p - 1, \; j = 1 \ldots N_p - 1, \; l = 1 \ldots N_c, \; i = 1 \ldots N_c,$ представляющий собой четырехмерную матрицу $\left( N_p - 1 \right) \times N_c \times \left( N_p - 1 \right) \times N_c.$ Подобная запись системы уравнений выглядит весьма неудобно. Поскольку целью данной задачи является определеление количества вещества $N_c$ компонентов в $N_p - 1$ фазах, то рассматриваемую систему уравнений предлагается записать в виде вектора, в котором последовательно для каждого компонента в каждой фазе записывается уравнение $g_l^k = 0$:

# $$ g_l^k = \begin{bmatrix} g_1^1 = 0 \\ g_2^1 = 0 \\ \vdots \\ g_{N_c}^1 = 0 \\ g_1^2 = 0 \\ g_2^2 = 0 \\ \vdots \\ g_{N_c}^2 = 0 \\ g_1^{N_p-1} = 0 \\ g_2^{N_p - 1} = 0 \\ \vdots \\ g_{N_c}^{N_p-1} = 0 \end{bmatrix}, \; k=1 \ldots N_p-1, \; l=1 \dots N_c.$$

# Всего получится $\left( N_p - 1 \right) \times N_c$ уравнений, необходимых для нахождения такого же количества неизвестных. Для решения данной системы уравнений методом Ньютона необходимо сформировать Якобиан:

# $$\begin{bmatrix}
# \color{#87A96B}{ \frac{\partial g_1^1}{\partial n_1^1} } & \color{#87A96B}{ \frac{\partial g_1^1}{\partial n_2^1} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^1}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#87A96B}{ \frac{\partial g_2^1}{\partial n_1^1} } & \color{#87A96B}{ \frac{\partial g_2^1}{\partial n_2^1} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^1}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#87A96B}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } \\ 
# \color{#87A96B}{ \frac{\partial g_{N_c}^1}{\partial n_1^1} } & \color{#87A96B}{ \frac{\partial g_{N_c}^1}{\partial n_2^1} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^1}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_{N_c}^1} } & \color{#87A96B}{ \frac{\partial g_1^2}{\partial n_1^2} } & \color{#87A96B}{ \frac{\partial g_1^2}{\partial n_2^2} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^2}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_{N_c}^1} } & \color{#87A96B}{ \frac{\partial g_2^2}{\partial n_1^2} } & \color{#87A96B}{ \frac{\partial g_2^2}{\partial n_2^2} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^2}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } \\
# \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_{N_c}^1} } & \color{#87A96B}{ \frac{\partial g_{N_c}^2}{\partial n_1^2} } & \color{#87A96B}{ \frac{\partial g_{N_c}^2}{\partial n_2^2} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^2}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } \\
# \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^{N_p-1}}{\partial n_1^{N_p-1}} } & \color{#87A96B}{ \frac{\partial g_1^{N_p-1}}{\partial n_2^{N_p-1}} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^{N_p-1}}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^{N_p-1}}{\partial n_1^{N_p-1}} } & \color{#87A96B}{ \frac{\partial g_2^{N_p-1}}{\partial n_2^{N_p-1}} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^{N_p-1}}{\partial n_{N_c}^{N_p-1}} } \\
# \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#87A96B}{ \vdots } \\
# \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_1^{N_p-1}} } & \color{#87A96B}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_2^{N_p-1}} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_{N_c}^{N_p-1}} }
# \end{bmatrix}$$

# При этом, зеленым цветом выделены элементы Якобиана, где $k = j$:

# $$  \frac{\partial g_l^k}{\partial n_i^j} = \frac{1}{f_l^k} \frac{\partial f_l^k}{\partial n_i^j} + \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}}.$$

# Остальные элементы, где $ k \neq j$:

# $$ \frac{\partial g_l^k}{\partial n_i^j} = \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}}.$$

# Таким образом, основным отличием метода минимизации энергии Гиббса от [метода последовательных подстановок](#pvt-esc-equilibrium-isothermal-ssi) является то, что он позволяет решить систему нелинейных уравнений относительно количеств веществ компонентов, а не констант фазового равновесия. Рассмотрим применение данного метода на практике.

# Сначала импортируем класс для расчета производных летучести компонентов с использованием уравнения состояния Сорейде-Уитсона.

# Далее создадим класс для определения количества вещества компонентов в фазах с использованием метода минимизации энергии Гиббса.

# In[29]:


class flash_isothermal_gibbs(flash_isothermal_ssi):
    def __init__(self, mr, eos_ders, z,
                 ssi_rr_eps=1e-8, ssi_eq_eps=1e-8, ssi_use_opt=False, qnss=False, ssi_negative_flash=False, ssi_eq_max_iter=30,
                 gibbs_eps=1e-8, gibbs_max_iter=10, save_residuals=False):
        self.z = z
        self.mr = mr
        self.eos_ders = eos_ders
        self.ssi_rr_eps = ssi_rr_eps
        self.ssi_eq_eps = ssi_eq_eps
        self.ssi_use_opt = ssi_use_opt
        self.qnss = qnss
        self.ssi_negative_flash = ssi_negative_flash
        self.ssi_eq_max_iter = ssi_eq_max_iter
        self.gibbs_eps = gibbs_eps
        self.gibbs_max_iter = gibbs_max_iter
        self.save_residuals = save_residuals
        pass

    def gibbs_equation(self, lnf):
        return lnf[:-1] - self.repeat(lnf[-1], axis=0, times=self.Np_1)

    def gibbs_jacobian(self, dlnf_dnk):
        jac = np.empty(shape=(self.Np_1*self.mr.Nc, self.Np_1*self.mr.Nc))
        for j in range(self.Np_1):
            for k in range(self.Np_1):
                if k == j:
                    jac[j*self.mr.Nc:self.mr.Nc*(j+1),k*self.mr.Nc:(k+1)*self.mr.Nc] = dlnf_dnk[j] + dlnf_dnk[-1]
                else:
                    jac[j*self.mr.Nc:self.mr.Nc*(j+1),k*self.mr.Nc:(k+1)*self.mr.Nc] = dlnf_dnk[-1]
        return jac

    def flash_isothermal_gibbs_run(self, P, T, phases, kv0=None, **kwargs):
        res = self.empty_object()
        if kv0 is None:
            ssi_res = self.flash_isothermal_ssi_run(P, T, phases, **kwargs)
            res.ssi_it = ssi_res.it
            if self.save_residuals:
                res.residuals_all = ssi_res.residuals_all
            F = ssi_res.F
            y = ssi_res.y
            nij0 = y[:-1] * F[:-1] * self.mr.n
        else:
            res.ssi_it = 0
            self.Np_1 = len(kv0)
            F = self.rachford_rice_newton(np.array([self.Np_1 * [1 / (self.Np_1 + 1)]]).reshape(self.Np_1, 1), kv0)
            if self.ssi_negative_flash:
                Fmin, Fmax = np.zeros_like(F), np.ones_like(F)
            else:
                Fmin, Fmax = self.rr_negative_limits(kv0)
            F = np.where(F < Fmin, Fmin, F)
            F = np.where(F > Fmax, Fmax, F)
            y = self.z * kv / (1 + np.sum(F * (kv - 1), axis=0))
            nij0 = y * F * self.mr.n
            if self.save_residuals:
                res.residuals_all = []
        nij0 = nij0.reshape((self.Np_1*self.mr.Nc, 1))
        nij = nij0
        residuals = np.ones_like(nij)
        it = 0
        while np.any(np.abs(residuals) > self.gibbs_eps) and it < self.gibbs_max_iter:
            nij_reshape = nij.reshape((self.Np_1, self.mr.Nc))
            F = np.sum(nij_reshape, axis=1).reshape((self.Np_1, 1)) / self.mr.n
            y = nij_reshape / (self.mr.n * F)
            y = np.append(y, np.array([self.z - np.sum(nij_reshape, axis=0) / self.mr.n]) / (1 - np.sum(F)), axis=0)
            eos = self.mr.eos_run(y, P, T, phases, **kwargs)
            residuals = self.gibbs_equation(eos.lnf).reshape(self.Np_1 * self.mr.Nc, 1)
            nj = self.full_F(F) * self.mr.n
            nij = nij - np.linalg.inv(self.gibbs_jacobian(self.eos_ders(self.mr, eos, nj, der_nk=True).get('dlnf_dnk').dlnf_dnk)).dot(residuals)
            it += 1
            if self.save_residuals:
                res.residuals_all.append(np.max(np.abs(residuals)))
        nij_reshape = nij.reshape((self.Np_1, self.mr.Nc))
        res.y = np.append(nij.reshape((self.Np_1, self.mr.Nc)) / (self.mr.n * F), np.array([self.z - np.sum(nij_reshape, axis=0) / self.mr.n]) / (1 - np.sum(F)), axis=0)
        res.eos = eos
        res.kv = res.y[:-1] / self.repeat(res.y[-1], axis=0, times=self.Np_1)
        res.it = it
        res.F = np.sum(nij_reshape, axis=1).reshape((self.Np_1, 1)) / self.mr.n
        res.F = self.full_F(res.F)
        res.comp_mole_frac = dict(zip(phases, res.y))
        res.phase_mole_frac = dict(zip(phases, np.append(res.F, 1 - res.F)))
        res.residuals = residuals
        return res


# С использованием данного метода рассмотрим решение [первой задачи](#pvt-esc-equilibrium-isothermal-example_1).

# In[30]:


P = 5.15 * 10**6
T = 448.0
z = np.array([0.1292, 0.0544, 0.0567, 0.7597])
phases = 'ogw'


# Начальные значения $n_i^j$ для использования метода Ньютона при методе минимизации энергии Гиббса сгенерируем путем проведения пяти итераций [метода последовательных подстановок](#pvt-esc-equilibrium-isothermal-ssi). Это позволит достаточно точно приблизиться к глобальному минимуму функции и не применять [методы поиска глобального минимума функции](../../0-Math/5-OM/OM-0-Introduction.html#math-om).

# In[31]:


flash5 = flash_isothermal_gibbs(mr1, derivatives_eos_2param, z, ssi_use_opt=True, save_residuals=True, ssi_eq_max_iter=5, gibbs_eps=1e-8).flash_isothermal_gibbs_run(P, T, phases, cw=0.0)


# In[32]:


flash5.comp_mole_frac


# In[33]:


flash5.residuals


# In[34]:


flash5.ssi_it, flash5.it


# Достигнуть требуемой точности удалось путем проведения дополнительных четырех итераций метода минимизации энергии Гиббса. Сравним полученные расчеты с проведенными ранее.

# In[35]:


flash1.comp_mole_frac


# In[36]:


flash1.residuals


# In[37]:


flash1.it


# In[38]:


iters_gibbs = np.arange(1, flash5.ssi_it + flash5.it +1, 1)

fig5, ax5 = plt.subplots(figsize=(8, 6))
fig5.canvas.header_visible = False

ax5.plot(iters_ssi, flash1.residuals_all, label='Метод последовательных подстановок')
ax5.plot(iters_gibbs, flash5.residuals_all, label='Метод минимизации энергии Гиббса')
ax5.set_ylabel('Максимальная абсолютная ошибка\nуравнений равновесного состояния')
ax5.set_xlabel('Итерации')
ax5.set_yscale('log')
ax5.set_ylim(1e-8, 100)
ax5.set_xlim(0.0, 16)
ax5.set_yticks(10**np.arange(-9.0, 2.0, 1.0))
ax5.grid(True, which='major', axis='y')
ax5.grid(True, which='minor', axis='y', ls='--', lw=0.5)
ax5.grid(True, which='major', axis='x')
ax5.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12))
ax5.legend(loc='best')

fig5.tight_layout()


# <a id='pvt-esc-equilibrium-isothermal-gem-switch_criteria'></a>
# Таким образом, метод минимизации энергии Гиббса позволяет за меньшее количество итераций достигнуть требуемой точности решения уравнений равновесия. Однако метод минимизации энергии Гиббса, основанный на решении системы нелинейных уравнений методом Ньютона, является менее устойчивым с точки зрения численных схем и может привести к появлению нефизичных значений, например, отрицательных значений количества вещества компонента или к сходимости к локальному минимимуму энергии Гиббса вместо глобального. Кроме того, метод минимизации энергии Гиббса требует большего объема вычислений: частных производных логарифма летучести по независимым переменным. В связи с этим вместо использования метода минимизации, основанного на численном методе Ньютона решения системы нелинейных уравнений, можно использовать [метод Пауэлла](https://en.wikipedia.org/wiki/Powell%27s_method), изложенный авторами работы \[[Nghiem et al, 1983](https://doi.org/10.2118/8285-PA)\] для задачи определения равновесного состояния системы. Необходимо также отметить, что авторы работы \[[Nghiem et al, 1983](https://doi.org/10.2118/8285-PA)\] использовали следующие критерии перехода от метода последовательных подстановок к методу Пауэлла, характеризующемуся более быстрой сходимостью, чем метод последовательных подстановок:

# $$ \begin{cases}
# \frac{ \sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} \left( \left( \frac{f_i^j}{f_i^R} \right)^k - 1 \right)^2 }{ \sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} \left( \left( \frac{f_i^j}{f_i^R} \right)^{k-1} - 1 \right)^2 } > \epsilon_R \\
# {F^j}^k - {F^j}^{k-1} < \epsilon_V, \; j=1 \ldots N_p \\
# \epsilon_L < \sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} \left( \left( \frac{f_i^j}{f_i^R} \right)^k - 1 \right)^2 < \epsilon_U \\
# 0 < {F^j}^k < 1, \; j=1 \ldots N_p ,
# \end{cases}$$

# где $k$ – номер итерации. Первые два критерия отвечают за переход к более быстрому методу из-за сравнительно медленной сходимости метода последовательных подстановок. Третье условие обуславливает наиболее оптимальный диапазон смены метода решения системы нелинейных уравнений. С одной стороны, смена на метода Пауэллаа (или метод Ньютона) должна происходить после локализации стационарной точки, поэтому верхняя граница $\epsilon_U$ предупреждает слишком раннюю смену метода. С другой стороны, если начальное приближение было достаточно точным, то не необходимости изменять используемый подход на более ресурсоемкий вблизи решения. За это отвечает наличие минимальной границы $\epsilon_U$. Кроме того, смену метода не рекомендуется проводить в тех случаях, когда $F^j, \; j=1 \ldots N_p$ близко к единице, поскольку в этом случае могут появиться отрицательные количества вещества компонентов. Авторы работы \[[Nghiem et al, 1983](https://doi.org/10.2118/8285-PA)\] приводят следующие значения для указанных выше условий смены используемого метода:

# $$ \begin{align}
# \epsilon_R &= 0.6; \\
# \epsilon_V &= 10^{-2}; \\
# \epsilon_L &= 10^{-5}; \\
# \epsilon_U &= 10^{-3}.
# \end{align} $$

# Стоит также отметить, что несмотря на более быстрое схождение к искомому решению, [метод Ньютона](../../0-Math/5-OM/OM-1-Newton.html#math-om-newton) является менее устойчивым, в результате чего могут получаться отрицательные количества вещества компонентов, поскольку значения $n_i^j$ не ограничены. В этом случае авторами работ \[[Petitfrere and Nichita, 2014](https://doi.org/10.1016/j.fluid.2013.08.039); [Pan et al, 2019](https://doi.org/10.1021/acs.iecr.8b05229)\] рекомендуется переключиться с метода Ньютона на [метод доверительной области](../../0-Math/5-OM/OM-3-TR.html#math-om-tr), рассмотренный ранее. При этом, в качестве критериев перехода на метод доверительной области авторами работы \[[Pan et al, 2019](https://doi.org/10.1021/acs.iecr.8b05229)\] приводятся следующие:

# $$ n_i^j < 0, \; i = 1 \ldots N_c, \; j = 1 \ldots N_p - 1 \; or \; \tilde{G}^{k+1} > \tilde{G}^k,$$

# где $k$ – номер итерации.

# <a id='pvt-esc-equilibrium-isoenthalpic'></a>
# ## Изоэнтальпийно-изобарические подходы к определению равновесного состояния системы
# Изложенные выше изотермико-изобарические подходы имеют широкое применение при решении задачи определения равновесного состояния многофазной многокомпонентной системы при известных и постоянных давлении и температуре системы. Однако большинство процессов при разработке месторождений углеводородов не являются изотермическими. В связи с этим возникает задача разработки метода определения равновесного состояния многофазной многокомпонентной системы в условиях, когда известны начальные термобарические условия системы и изменение ее энергитических параметров (энтальпии или энтропии). Для решения данной задачи был разработан изоэнтальпийно-изобарический (и изоэнтропно-изобарический) подход к определению равновесного состояния многофазной многокомпонентной системы \[[Mischelsen, 1987](https://doi.org/10.1016/0378-3812(87)87002-4); [Agarwal et al, 1991](https://doi.org/10.2118/91-03-07); [Heidari et al, 2014](https://doi.org/10.2118/170029-MS)\]. В качестве исходных данных для такого подхода используются давление и энтальпия системы, а также мольные доли компонентов в системе. Рассмотрим алгоритм и применение изоэнтальпийно-изобарического подхода к определению равновесного состояния системы.

# <a id='pvt-esc-equilibrium-isoenthalpic-ssi'></a>
# ### Метод последовательных подстановок
# Метод последовательных подстановок также может быть применен для определения равновесного состояния системы с использованием изоэнтальпийно-изобарического подхода. В отличие от [последовательных подстановок в изотермико-изобарическом подходе](#pvt-esc-equilibrium-isothermal-ssi) в изоэнтальпийно-изобарическом в качестве неизвестных рассматриваются $\left( N_p - 1 \right) N_c$ неизвестных констант фазового равновесия компонентов $K_i^{jR}$, $N_p - 1$ неизвестных мольных долей фаз $F^j$, а также температура системы $T$. Общее количество неизвестных составляет $\left( N_p - 1 \right) N_c + \left( N_p - 1 \right) + 1.$ Следовательно, в дополнении к представленной [ранее](#pvt-esc-equilibrium-isothermal-ssi-equilibrium_condition) системе уравнений

# $$ \begin{cases} \sum_{i=1}^{N_c} \frac{z_i \left( K_i^{jR} - 1 \right)}{\sum_{k=1}^{N_p - 1} F^k \left(K_i^{kR} - 1 \right) + 1} = 0, \; j=1 \ldots N_p-1; \\ \ln K_i^{jR} + \ln \phi_i^j - \ln \phi_i^R = 0, \; i=1 \ldots N_c, \; j=1 \dots N_p - 1, \end{cases} $$

# которая включает $\left( N_p - 1 \right) N_c + \left( N_p - 1 \right)$ уравнений, необходимо добавить доплнительное уравнение теплового баланса, позволяющего определить температуру системы. В качестве такого уравнения в изоэнтальпийно-изобарическом подходе используется следующее:

# $$ H \left( P, T, n_i \right) - H_{spec} = 0,$$

# где $H_{spec}$ – исходная заданная энтальпия системы, $H \left( P, T, n_i \right)$ – рассчитываемая энтальпия системы. Поскольку энтальпия системы является [экстенсивным параметром](../1-TD/TD-9-Observables.html#pvt-td-observables), то данное уравнение можно преобразовать следующим образом:

# $$ \sum_{j=1}^{N_p} n^j h^j \left( P, T, y_i^j \right) - H_{spec} = 0,$$

# где $h^j \left( P, T \right)$ – удельная энтальпия $j$-ой фазы.

# <a id='pvt-esc-equilibrium-isoenthalpic-ssi-schemes'></a>
# Существует несколько схем для изоэнтальпийно-изобарического подхода к определению равновесного состояния системы. В каждой из них в качестве исходных данных используются давление и энтальпия системы, мольные доли компонентов в ней, а также начальное приближение температуры. В рамках первой схемы, как и в изотермическом подходе, определяется начальное приближение констант фазового равновесия компонентов с использованием представленных ранее [выражений](#pvt-esc-equilibrium-isothermal-ssi-algorithm) для начального приближения температуры. После этого для каждой итерации, на которой решается уравнение теплового баланса, предварительно проводится изотермический расчет равновесного состояния для температуры и констант фазового равновесия, полученных на предыдущей итерации. То есть уравнение теплового баланса решается во внешнем цикле по отношению к решению уравнения равновесного состояния. Недостатком данной схемы является ее существенная ресурсоемкость, обусловленная вложенностью циклов – значительным количеством итераций изотермического расчета равновесного состояния, проводимого на каждой итерации приближения к искомой температуре. Вторая схема заключается в одновременном решении [системы уравнений Речфорда-Райса](#pvt-esc-equilibrium-rachford_rice) и уравнения теплового баланса на каждой итерации. В этом случае отсутствует вложенность циклов – для расчета температуры нет необходимости в достижении равновесного состояния, что является более эффективным, по сравнению с первой схемой. Однако, так как не существует каких-то ограничений для температуры при решении уравнения теплового баланса (как, например, ограничения в мольных долях фазы, при решении системы уравнений Речфорда-Райса, рассмотренные [ранее](#pvt-esc-equilibrium-isothermal-ssi-negative_flash)), то в процессе одновременного приближения к решению констант фазового равновесия и температуры значения последней могут получаться нефизичными, что негативно скажется на численной стабильности алгоритма. Данная ситуация характерна для систем, находящихся вблизи границы фазового перехода (появления или исчезновения фазы). Третья схема представляет собой некоторую комбинацию из первых двух схем, когда изотермический расчет равновесного состояния проводится после некоторого количества (например, после пяти) итераций второй схемы.

# Для определения энтальпии фазы можно использовать полученное [ранее](../3-Parameters/Parameters-2-Enthalpy.html#pvt-parameters-enthalpy-isobaric_isothermal) выражение:

# $$ H^j \left( P, T, n_i^j \right) =  \sum_{i=1}^{N_c} n_i^j h_i^{ig} - R T^2 \sum_{i=1}^{N_c} n_i^j \left( \frac{\partial \ln \phi_i^j}{\partial T} \right)_{P,n_i}.$$

# Таким образом, уравнение теплового баланса можно записать следующим образом:

# $$ g^H = \sum_{j=1}^{N_c} H^j - H_{spec} =  \sum_{j=1}^{N_p} \sum_{i=1}^{N_c} n_i^j h_i^{ig} - R T^2 \sum_{j=1}^{N_p}  \sum_{i=1}^{N_c} n_i^j \left( \frac{\partial \ln \phi_i^j}{\partial T} \right)_{P,n_i} - H_{spec} = 0.$$

# Данное уравнение является нелинейным относительно температуры – для его решения можно воспользоваться [методом Ньютона](https://en.wikipedia.org/wiki/Newton%27s_method), для которого необходимо определить частную производную функции $g^H$ по температуре:

# $$ \left( \frac{\partial g^H}{\partial T} \right)_{P,n_i} = \sum_{j=1}^{N_p} \left( \frac{\partial H^j }{\partial T} \right)_{P,n_i^j}.$$

# Частные производные энтальпии по температуре с использованием уравнений состояния Суаве-Редлиха-Квонга, Пенга-Робинсона и Сорейде-Уитсона были рассмотрены [ранее](../3-Parameters/Parameters-Appendix-A-PD.html#pvt-parameters-appendix-pd-enthalpy-srk_pr_sw).

# Рассмотрим решение [первой задачи](#pvt-esc-equilibrium-isothermal-example_1) с использованием изоэнтальпийного метода последовательных подстановок.

# In[39]:


P = 5.15 * 10**6
T = 448.0
cw = 0.0
z = np.array([0.1292, 0.0544, 0.0567, 0.7597])


# Сначала определим энтальпию системы, чтобы данное значение использовать в качестве исходных данных. Для этого импортируем класс для расчета параметров системы.

# Инициализируем класс, передав на вход матрицу коэффициентов для расчета теплоемкости системы в состоянии идеального газа.

# In[40]:


cp_ig = np.array([[2.959520E+01, 3.378045E+01, 3.720896E+01, 3.376336E+01],
                  [8.379912E-02, 2.485010E-01, 4.809830E-01, -5.945958E-03],
                  [3.255759E-04, 2.534802E-04, 2.418717E-04, 2.235754E-05],
                  [-3.957572E-07, -3.838002E-07, -5.16685E-07, -9.962009E-09],
                  [1.312889E-10, 1.297666E-10, 1.884531E-10, 1.097487E-12]])
params = parameters_2param(cp_ig)


# Определим энтальпию системы:

# In[41]:


Hspec = np.sum(params.enthalpy(mr1, flash1.eos, flash1.F * mr1.n, Tref=273.15))
Hspec


# Таким образом, энтальпия рассматриваемой системы составляет $-20.4 \; \frac{кДж}{моль}$ при референсной точке $P = 0 \; Па, \; T = 273.15 \; K.$ Поскольку для решения уравнения теплового баланса методом Ньютона необходимо определить частную производную энтальпии по температуре, то импортируем класс для расчета частных производных термодинамических параметров.

# In[42]:


params_der = derivatives_parameters_2param()


# Теперь создадим класс для расчета равновесных мольных долей компонентов с использованием изоэнтальпийного метода последовательных подстановок.

# In[43]:


class flash_isenthalpic_ssi(flash_isothermal_ssi):
    def __init__(self, mr, params, params_der, z, Tref=273.15,
                 ssi_rr_eps=1e-8, ssi_eq_eps=1e-8, ssi_therm_eps=1e-8, ssi_use_opt=False, qnss=False, ssi_eq_max_iter=30, ssi_therm_max_iter=200, save_residuals=False, ssi_therm_scheme=1):
        self.z = z
        self.mr = mr
        self.params = params
        self.params_der = params_der
        self.Tref = Tref
        self.ssi_rr_eps = ssi_rr_eps
        self.ssi_eq_eps = ssi_eq_eps
        self.ssi_therm_eps = ssi_therm_eps
        self.ssi_use_opt = ssi_use_opt
        self.qnss = qnss
        self.ssi_negative_flash = False
        self.ssi_eq_max_iter = ssi_eq_max_iter
        self.ssi_therm_max_iter = ssi_therm_max_iter
        self.ssi_therm_scheme = ssi_therm_scheme
        self.save_residuals = save_residuals
        pass

    def thermal_balance(self, eos, nj, H):
        return np.sum(self.params.enthalpy(self.mr, eos, nj, self.Tref)) - H

    def flash_isenthalpic_ssi_run(self, P, H, phases, kv0=None, T0=273.15, **kwargs):
        if kv0 is None:
            kv = self.k_values_init(P, T0, phases)
            self.Np_1 = len(phases) - 1
        else:
            kv = kv0
            self.Np_1 = len(kv0)
        T = T0
        it_ssi = 0
        it_t = 0
        if self.ssi_therm_scheme == 1:
            residuals = 1
            if self.save_residuals:
                residuals_all = []
                it_ssi_all = []
            while np.abs(residuals) > self.ssi_therm_eps and it_t < self.ssi_therm_max_iter:
                ssi = self.flash_isothermal_ssi_run(P, T, phases, kv, **kwargs)
                kv = ssi.kv
                it_ssi += ssi.it
                residuals = self.thermal_balance(ssi.eos, ssi.F * self.mr.n, H)
                dH_dT = np.sum(self.params_der.dH_dT(self.mr, ssi.eos, self.params, ssi.F * self.mr.n))
                T = T - residuals / dH_dT
                it_t += 1
                if self.save_residuals:
                    residuals_all.append(np.max(np.abs(residuals)))
                    it_ssi_all.append(it_ssi)
            ssi.it = it_t
            ssi.it_ssi = it_ssi
            ssi.T = T
            ssi.residuals_enthalpy = residuals
            if self.save_residuals:
                ssi.residuals_enthalpy_all = residuals_all
                ssi.it_ssi_all = it_ssi_all
            return ssi
        else:
            res = self.empty_object()
            residuals = np.ones(shape=(self.Np_1, self.mr.Nc))
            residuals_H = 1
            if self.save_residuals:
                res.residuals_all = []
                res.residuals_enthalpy_all = []
                res.it_ssi_all = []
            F = np.array([self.Np_1 * [1 / (self.Np_1 + 1)]]).reshape(self.Np_1, 1)
            lambda_pow = 1
            Fmin = np.zeros_like(F)
            Fmax = np.ones_like(F)
            while (np.any(np.abs(residuals) > self.ssi_eq_eps) or np.abs(residuals_H) > self.ssi_therm_eps) and it_t < self.ssi_therm_max_iter:
                if self.ssi_therm_scheme == 3:
                    if it_t % 5 == 0:
                        ssi = self.flash_isothermal_ssi_run(P, T, phases, kv, **kwargs)
                        kv = ssi.kv
                        it_ssi += ssi.it
                F = self.rachford_rice_newton(F, kv)
                F = np.where(F < Fmin, Fmin, F)
                F = np.where(F > Fmax, Fmax, F)
                y = self.z * kv / (1 + np.sum(F * (kv - 1), axis=0))
                yref = self.z / (1 + np.sum(F * (kv - 1), axis=0))
                eos = self.mr.eos_run(np.append(y, np.array([yref]), axis=0), P, T, phases, **kwargs)
                if self.ssi_use_opt:
                    residuals_prev = residuals
                    residuals = np.log(kv) + eos.lnphi[:self.Np_1] - self.repeat(eos.lnphi[-1], axis=0, times=self.Np_1)
                    lambda_pow = - lambda_pow * np.sum(residuals_prev**2) / (np.sum(residuals_prev * residuals) - np.sum(residuals_prev**2))
                else:
                    residuals = np.log(kv) + eos.lnphi[:self.Np_1] - self.repeat(eos.lnphi[-1], axis=0, times=self.Np_1)
                kv = kv * (self.repeat(eos.f[-1], axis=0, times=self.Np_1) / eos.f[:self.Np_1])**lambda_pow
                nj = self.full_F(F) * self.mr.n
                residuals_H = self.thermal_balance(eos, nj, H)
                dH_dT = np.sum(self.params_der.dH_dT(self.mr, eos, self.params, nj))
                T = T - residuals_H / dH_dT
                it_t += 1
                if self.save_residuals:
                    res.residuals_all.append(np.max(np.abs(residuals)))
                    res.residuals_enthalpy_all.append(np.max(np.abs(residuals_H)))
                    res.it_ssi_all.append(it_ssi)
            res.T = T
            res.kv = kv
            res.it = it_t
            res.it_ssi = it_ssi
            res.y = np.append(self.z * kv / (1 + np.sum(F * (kv - 1), axis=0)), np.array([self.z / (1 + np.sum(F * (kv - 1), axis=0))]), axis=0)
            res.eos = eos
            res.F = self.full_F(F)
            res.comp_mole_frac = dict(zip(phases, res.y))
            res.phase_mole_frac = dict(zip(phases, res.F.ravel()))
            res.residuals = residuals
            res.residuals_enthalpy = residuals_H
            return res


# Выполним расчет равновесного состояния, применив изоэнтальпийный подход. Поскольку на данный момент не были рассмотрены алгоритмы определения количества фаз системы, то в качестве начального приближения температуры зададим достаточно близкую к конечному значению, чтобы в процессе поиска решения не выйти за пределы трехфазной области.

# In[44]:


flash6 = flash_isenthalpic_ssi(mr1, params, params_der, z, save_residuals=True, ssi_therm_scheme=1).flash_isenthalpic_ssi_run(P, Hspec, 'ogw', T0=440, cw=0.0)


# In[45]:


flash6.comp_mole_frac


# Сравним полученное решение с найденным изотермическим подходом:

# In[46]:


flash1.comp_mole_frac


# При этом, найденная температура соответствует искомой:

# In[47]:


flash6.T


# Погрешности решения уравнения теплового баланса и уравнений равновесного состояния:

# In[48]:


flash6.residuals_enthalpy, flash6.residuals


# Стоит также отметить, что для нахождения решения применялась [первая схема](#pvt-esc-equilibrium-isoenthalpic-ssi-schemes) изоэнтальпийного подхода. Количество итераций решения уравнения теплового баланса и общее количество итераций изотермического расчета равновесного состояния:

# In[49]:


flash6.it, flash6.it_ssi


# При реализации второй схемы:

# In[50]:


flash7 = flash_isenthalpic_ssi(mr1, params, params_der, z, save_residuals=True, ssi_therm_scheme=2).flash_isenthalpic_ssi_run(P, Hspec, 'ogw', T0=440, cw=0.0)


# In[51]:


flash7.comp_mole_frac


# In[52]:


flash7.T


# In[53]:


flash7.it, flash7.it_ssi


# При реализации третьей схемы:

# In[54]:


flash8 = flash_isenthalpic_ssi(mr1, params, params_der, z, save_residuals=True, ssi_therm_scheme=3).flash_isenthalpic_ssi_run(P, Hspec, 'ogw', T0=440, cw=0.0)


# In[55]:


flash8.comp_mole_frac


# In[56]:


flash8.T


# In[57]:


flash8.it, flash8.it_ssi


# <a id='pvt-esc-equilibrium-isenthalpic-gem'></a>
# ### Метод минимизации энергии Гиббса
# Метод минимизации энергии Гиббса может быть также применен для определения равновесного состояния в изоэнтальпийно-изобарической формулировке. Рассмотрим алгоритм и пример изоэнтальпийно-изобарического метода минимизации энергии Гиббса.

# В отличие от изотермико-изобарического подхода, рассмотренного [ранее](#pvt-esc-equilibrium-isothermal-gem), где система, состоящая из $\left( N_p - 1 \right) \times N_c$ уравнений, решалась методом Ньютона относительно $\left( N_p - 1 \right) \times N_c$ количеств вещества $i$ компонентов в $j$ фазе $n_i^j, \; i = 1 \ldots N_c, \; j = 1 \dots \left( N_p - 1 \right)$, в изоэнтальпийно-изобарическом подходе добавляется дополнительная неизвестная – температура $T$, для нахождения которой записывается уравнение теплового баланса:

# $$g^H = H \left( P, T, n_i^j \right) - H_{spec} = 0.$$

# Таким образом, вектор решаемой системы уравнений записывается следующим образом:

# $$ \begin{bmatrix} g_1^1 = 0 \\ g_2^1 = 0 \\ \vdots \\ g_{N_c}^1 = 0 \\ g_1^2 = 0 \\ g_2^2 = 0 \\ \vdots \\ g_{N_c}^2 = 0 \\ g_1^{N_p-1} = 0 \\ g_2^{N_p - 1} = 0 \\ \vdots \\ g_{N_c}^{N_p-1} = 0 \\ g^H = 0\end{bmatrix}, \; k=1 \ldots N_p-1, \; l=1 \dots N_c.$$

# Для решения этой системы уравнения [методом Ньютона](../../0-Math/5-OM/OM-1-Newton.html#math-om-newton) необходимо составить Якобиан:

# $$\begin{bmatrix}
# \color{#87A96B}{ \frac{\partial g_1^1}{\partial n_1^1} } & \color{#87A96B}{ \frac{\partial g_1^1}{\partial n_2^1} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^1}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^1}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_1^1}{\partial T} } \\
# \color{#87A96B}{ \frac{\partial g_2^1}{\partial n_1^1} } & \color{#87A96B}{ \frac{\partial g_2^1}{\partial n_2^1} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^1}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^1}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_2^1}{\partial T} } \\
# \color{#87A96B}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#6699CC}{ \vdots } \\ 
# \color{#87A96B}{ \frac{\partial g_{N_c}^1}{\partial n_1^1} } & \color{#87A96B}{ \frac{\partial g_{N_c}^1}{\partial n_2^1} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^1}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^1}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_{N_c}^1}{\partial T} } \\
# \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_{N_c}^1} } & \color{#87A96B}{ \frac{\partial g_1^2}{\partial n_1^2} } & \color{#87A96B}{ \frac{\partial g_1^2}{\partial n_2^2} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^2}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^2}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_1^2}{\partial T} } \\
# \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_{N_c}^1} } & \color{#87A96B}{ \frac{\partial g_2^2}{\partial n_1^2} } & \color{#87A96B}{ \frac{\partial g_2^2}{\partial n_2^2} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^2}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^2}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_1^1}{\partial T} } \\
# \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#6699CC}{ \vdots } \\
# \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_{N_c}^1} } & \color{#87A96B}{ \frac{\partial g_{N_c}^2}{\partial n_1^2} } & \color{#87A96B}{ \frac{\partial g_{N_c}^2}{\partial n_2^2} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^2}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_1^{N_p-1}} } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_2^{N_p-1}} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^2}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_{N_c}^2}{\partial T} } \\
# \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#6699CC}{ \vdots } \\
# \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_1^{N_p-1}}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^{N_p-1}}{\partial n_1^{N_p-1}} } & \color{#87A96B}{ \frac{\partial g_1^{N_p-1}}{\partial n_2^{N_p-1}} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_1^{N_p-1}}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_1^{N_p-1}}{\partial T} } \\
# \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_2^{N_p-1}}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^{N_p-1}}{\partial n_1^{N_p-1}} } & \color{#87A96B}{ \frac{\partial g_2^{N_p-1}}{\partial n_2^{N_p-1}} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_2^{N_p-1}}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_2^{N_p-1}}{\partial T} } \\
# \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#CD9575}{ \vdots } & \color{#CD9575}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \vdots } & \color{#87A96B}{ \ddots } & \color{#87A96B}{ \vdots } & \color{#6699CC}{ \vdots } \\
# \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_1^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_2^1} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_{N_c}^1} } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_1^2} } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_2^2} } & \color{#CD9575}{ \cdots } & \color{#CD9575}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_{N_c}^2} } & \color{#CD9575}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_1^{N_p-1}} } & \color{#87A96B}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_2^{N_p-1}} } & \color{#87A96B}{ \cdots } & \color{#87A96B}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial n_{N_c}^{N_p-1}} } & \color{#6699CC}{ \frac{\partial g_{N_c}^{N_p-1}}{\partial T} } \\
# \color{#F0DC82}{ \frac{\partial g^H}{\partial n_1^1} } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_2^1} } & \color{#F0DC82}{ \cdots } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_{N_c}^1} } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_1^2} } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_2^2} } & \color{#F0DC82}{ \cdots } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_{N_c}^2} } & \color{#F0DC82}{ \cdots } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_1^{N_p-1}} } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_2^{N_p-1}} } & \color{#F0DC82}{ \cdots } & \color{#F0DC82}{ \frac{\partial g^H}{\partial n_{N_c}^{N_p-1}} } & \color{#BD33A4}{ \frac{\partial g^H}{\partial T} }
# \end{bmatrix}$$

# При этом, зеленым цветом выделены элементы Якобиана:

# $$  \frac{\partial g_l^k}{\partial n_i^j} = \frac{1}{f_l^k} \frac{\partial f_l^k}{\partial n_i^j} + \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}}, \; k = j.$$

# Красным выделены элементы, где:

# $$ \frac{\partial g_l^k}{\partial n_i^j} = \frac{1}{f_l^{N_p}} \frac{\partial f_l^{N_p}}{\partial n_i^{N_p}}, \; k \neq j.$$

# Синие элементы Якобиана представляют собой частную производную $g_l^k$ по температуре при постоянных давлении и количествах вещества компонентов:

# $$ \frac{\partial g_l^k}{\partial T} = \frac{\partial \ln f_l^k}{\partial T} - \frac{\partial \ln f_l^{N_p}}{\partial T}.$$

# Желтые элементы Якобиана - частная производная уравнения теплового баланса по количеству вещества компонентов при постоянных давлении и температуре:

# $$ \frac{\partial g^H}{\partial n_i^j} = \frac{\partial H \left( P, T, n_l^k \right)}{\partial n_i^j}.$$

# Наконец, розовый элемент Якобиана - частная производная уравнения теплового баланса по температуре при постоянных давлении и количествах вещества компонентов:

# $$ \frac{\partial g^H}{\partial T} = \frac{\partial H \left( P, T, n_l^k \right)}{\partial T}.$$

# Остановимся подробнее на частной производной $g^H$ по $n_i^j$ и распишем данное выражение с учетом выбранных независимых переменных. Энтальпия системы в соответствии с [ранее](../3-Parameters/Parameters-2-Enthalpy.html#pvt-parameters-enthalpy-isobaric_isothermal) полученным выражением записывается следующим образом:

# $$ \begin{align}
# H
# &= \sum_{k=1}^{N_p} \sum_{l=1}^{N_c} n_l^k h_l^k \\
# &= \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} n_l^k h_l^k + \sum_{l=1}^{N_c} n_l^{N_p} h_l^{N_p} \\
# &= \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} n_l^k h_l^{ig} - R T^2 \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} n_l^k \frac{\partial \ln \phi_l^k}{\partial T} + \sum_{l=1}^{N_c} n_l^{N_p} h_l^{ig}  - R T^2 \sum_{l=1}^{N_c} n_l^{N_p} \frac{\partial \ln \phi_l^{N_p}}{\partial T}.
# \end{align} $$

# С учетом данного выражения запишем частную производную $g^H$ по $n_i^j, \; i=1 \dots N_c, \; j=1 \dots N_p - 1$:

# $$ \begin{alignat}{1}
# \frac{\partial g^H}{\partial n_i^j} = \frac{\partial H}{\partial n_i^j}
# &= & \; \frac{\partial}{\partial n_i^j} \left( \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} n_l^k h_l^{ig} \right) - R T^2 \frac{\partial}{\partial n_i^j} \left( \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} n_l^k \frac{\partial \ln \phi_l^k}{\partial T} \right) + \frac{\partial}{\partial n_i^j} \left( \sum_{l=1}^{N_c} n_l^{N_p} h_l^{ig} \right) \\
# && \; - R T^2 \frac{\partial}{\partial n_i^j} \left( \sum_{l=1}^{N_c} n_l^{N_p} \frac{\partial \ln \phi_l^{N_p}}{\partial T} \right) \\
# &= & \; \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} h_l^{ig} \frac{\partial n_l^k}{\partial n_i^j} - R T^2 \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} \left( \frac{\partial \ln \phi_l^k}{\partial T} \frac{\partial n_l^k}{\partial n_i^j} + n_l^k \frac{\partial^2 \ln \phi_l^k}{\partial T \partial n_i^j} \right) + \sum_{l=1}^{N_c} h_i^{ig} \frac{\partial n_l^{N_p}}{\partial n_i^j} \\
# && \; - R T^2 \sum_{l=1}^{N_c} \left( \frac{\partial \ln \phi_l^{N_p}}{\partial T} \frac{\partial n_l^{N_p}}{\partial n_i^j} + n_l^{N_p} \frac{\partial^2 \ln \phi_l^{N_p}}{\partial T \partial n_i^j} \right) \\
# &= & \; \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} h_l^{ig} \frac{\partial n_l^k}{\partial n_i^j} - R T^2 \sum_{k=1}^{N_p-1} \sum_{l=1}^{N_c} \left( \frac{\partial \ln \phi_l^k}{\partial T} \frac{\partial n_l^k}{\partial n_i^j} + n_l^k \frac{\partial^2 \ln \phi_l^k}{\partial T \partial n_i^k} \frac{\partial n_i^k}{\partial n_i^j} \right) + \sum_{l=1}^{N_c} h_i^{ig} \frac{\partial n_l^{N_p}}{\partial n_i^j} \\
# && \; - R T^2 \sum_{l=1}^{N_c} \left( \frac{\partial \ln \phi_l^{N_p}}{\partial T} \frac{\partial n_l^{N_p}}{\partial n_i^j} + n_l^{N_p} \frac{\partial^2 \ln \phi_l^{N_p}}{\partial T \partial n_i^{N_p}} \frac{\partial n_i^{N_p}}{\partial n_i^j} \right).
# \end{alignat}$$

# Так как

# $$ n_l^{N_p} = n z_l - \sum_{k=1}^{N_p-1} n_l^k,$$

# то:

# $$ \frac{\partial n_l^{N_p}}{\partial n_i^j} = - \sum_{k=1}^{N_p-1} \frac{\partial n_l^k}{\partial n_i^j}.$$

# Необходимо отметить, что $\frac{\partial n_l^k}{\partial n_i^j} = 0, \; k \neq j,$ поскольку в этом случае $n_l^k$ и $n_i^j$ являются независимыми переменными. Следовательно,

# $$ \frac{\partial n_l^{N_p}}{\partial n_i^j} = - \frac{\partial n_l^k}{\partial n_i^j}, \; k=j.$$

# С учетом этого выражение частной производной $g^H$ по $n_i^j$ можно свести к следующему виду:

# $$ \begin{alignat}{1}
# \frac{\partial g^H}{\partial n_i^j}
# &= & \; \sum_{l=1}^{N_c} h_l^{ig} \frac{\partial n_l^j}{\partial n_i^j} - R T^2 \sum_{l=1}^{N_c} \left( \frac{\partial \ln \phi_l^j}{\partial T} \frac{\partial n_l^j}{\partial n_i^j} + n_l^j \frac{\partial^2 \ln \phi_l^j}{\partial T \partial n_i^j} \frac{\partial n_i^j}{\partial n_i^j} \right) - \sum_{l=1}^{N_c} h_i^{ig} \frac{\partial n_l^j}{\partial n_i^j} \\
# && \; - R T^2 \sum_{l=1}^{N_c} \left( - \frac{\partial \ln \phi_l^{N_p}}{\partial T} \frac{\partial n_l^j}{\partial n_i^j} - n_l^{N_p} \frac{\partial^2 \ln \phi_l^{N_p}}{\partial T \partial n_i^{N_p}} \frac{\partial n_i^j}{\partial n_i^j} \right) \\
# &= & \;  - R T^2 \sum_{l=1}^{N_c} \left( \frac{\partial \ln \phi_l^j}{\partial T} I_{li}^j + n_l^j \frac{\partial^2 \ln \phi_l^j}{\partial T \partial n_i^j}  \right) + R T^2 \sum_{l=1}^{N_c} \left(\frac{\partial \ln \phi_l^{N_p}}{\partial T} I_{li}^j + n_l^{N_p} \frac{\partial^2 \ln \phi_l^{N_p}}{\partial T \partial n_i^{N_p}} \right) \\
# &= & \; - R T^2 \sum_{l=1}^{N_c} \left( \left( \frac{\partial \ln \phi_l^j}{\partial T} - \frac{\partial \ln \phi_l^{N_p}}{\partial T} \right) I_{li}^j + n_l^j \frac{\partial^2 \ln \phi_l^j}{\partial T \partial n_i^j} - n_l^{N_p} \frac{\partial^2 \ln \phi_l^{N_p}}{\partial T \partial n_i^{N_p}} \right).
# \end{alignat}$$

# В выражении выше $\frac{\partial n_l^j}{\partial n_i^j} = I_{li}^j$ представляет собой единичную матрицу размерностью $N_c \times N_c$ для фазы $j$.

# Для нахождения элементов Якобиана частные производные летучести компонентов по количеству вещества компонентов и температуре были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd). Кроме того, частные производные энтальпии по температуре рассматривались в соответствующем [разделе](../3-Parameters/Parameters-Appendix-A-PD.html#pvt-parameters-appendix-pd-enthalpy-srk_pr_sw). Создадим класс для расчета равновесного состояния изоэнтальпийным методом минимизации энергии Гиббса.

# In[58]:


class flash_isenthalpic_gibbs(flash_isenthalpic_ssi):
    def __init__(self, mr, eos_der, params, params_der, z, Tref=273.15,
                 ssi_rr_eps=1e-8, ssi_eq_eps=1e-8, ssi_therm_eps=1e-8, ssi_use_opt=False, qnss=False, ssi_eq_max_iter=30,
                 ssi_therm_max_iter=30, save_residuals=False, ssi_therm_scheme=1,
                 gibbs_eps=1e-8, gibbs_max_iter=50):
        self.z = z
        self.mr = mr
        self.eos_der = eos_der
        self.params = params
        self.params_der = params_der
        self.Tref = Tref
        self.ssi_rr_eps = ssi_rr_eps
        self.ssi_eq_eps = ssi_eq_eps
        self.ssi_therm_eps = ssi_therm_eps
        self.ssi_use_opt = ssi_use_opt
        self.qnss = qnss
        self.ssi_negative_flash = False
        self.ssi_eq_max_iter = ssi_eq_max_iter
        self.save_residuals = save_residuals
        self.ssi_therm_max_iter = ssi_therm_max_iter
        self.ssi_therm_scheme = ssi_therm_scheme
        self.gibbs_eps = gibbs_eps
        self.gibbs_max_iter = gibbs_max_iter
        pass

    def gibbs_equation(self, eos, nj, H):
        return np.append(np.reshape(eos.lnf[:-1] - self.repeat(eos.lnf[-1], axis=0, times=self.Np_1), newshape=(self.Np_1 * self.mr.Nc, 1)),
                                    [[np.sum(self.params.enthalpy(self.mr, eos, nj, self.Tref)) - H]], axis=0)

    def gibbs_jacobian(self, eos, n):
        ders = self.eos_der(self.mr, eos, n, der_T=True, der_nk=True).get('dlnf_dnk', 'dlnphi_dT', 'dlnf_dT', 'd2lnphi_dnkdT')
        jac = np.empty(shape=(self.Np_1*self.mr.Nc, self.Np_1*self.mr.Nc))
        for j in range(self.Np_1):
            for k in range(self.Np_1):
                if k == j:
                    jac[j*self.mr.Nc:self.mr.Nc*(j+1),k*self.mr.Nc:(k+1)*self.mr.Nc] = ders.dlnf_dnk[j] + ders.dlnf_dnk[-1]
                else:
                    jac[j*self.mr.Nc:self.mr.Nc*(j+1),k*self.mr.Nc:(k+1)*self.mr.Nc] = ders.dlnf_dnk[-1]
        dgf_dT = np.reshape(ders.dlnf_dT[:-1] - ders.dlnf_dT[-1], newshape=(self.Np_1 * self.mr.Nc, 1))
        dgH_dni = np.sum(self.repeat(ders.dlnphi_dT[:-1] - ders.dlnphi_dT[-1], 1, self.mr.Nc) * self.repeat(np.identity(self.mr.Nc), 0, eos.Np - 1) +                          self.repeat(n[:-1] * eos.y[:-1], 1, self.mr.Nc) * ders.d2lnphi_dnkdT[:-1] -                          self.repeat(self.repeat(n[-1] * eos.y[-1], 0, self.mr.Nc) * ders.d2lnphi_dnkdT[-1], 0, eos.Np - 1), axis=2).reshape(1, self.Np_1 * self.mr.Nc) *                   ( - self.R) * eos.T**2
        dgH_dT = np.array([[np.sum(self.params_der.dH_dT(self.mr, eos, self.params, n))]])
        return np.append(np.append(jac, dgf_dT, axis=1), np.append(dgH_dni, dgH_dT, axis=1), axis=0)

    def flash_isenthalpic_gibbs_run(self, P, H, phases, kv0=None, T0=273.15, **kwargs):
        res = self.empty_object()
        if kv0 is None:
            ssi_res = self.flash_isenthalpic_ssi_run(P, H, phases, T0=T0, **kwargs)
            res.ssi_it = ssi_res.it
            if self.save_residuals:
                res.residuals_all = ssi_res.residuals_all
                res.residuals_enthalpy_all = ssi_res.residuals_enthalpy_all
            F = ssi_res.F
            y = ssi_res.y
            nij0 = y[:-1] * F[:-1] * self.mr.n
            T = ssi_res.T
        else:
            res.ssi_it = 0
            self.Np_1 = len(kv0)
            F = self.rachford_rice_newton(np.array([self.Np_1 * [1 / (self.Np_1 + 1)]]).reshape(self.Np_1, 1), kv0)
            if self.ssi_negative_flash:
                Fmin, Fmax = np.zeros_like(F), np.ones_like(F)
            else:
                Fmin, Fmax = self.rr_negative_limits(kv0)
            F = np.where(F < Fmin, Fmin, F)
            F = np.where(F > Fmax, Fmax, F)
            y = self.z * kv / (1 + np.sum(F * (kv - 1), axis=0))
            nij0 = y * F * self.mr.n
            T = T0
            if self.save_residuals:
                res.residuals_all = []
                res.residuals_enthalpy_all = []
        nij0 = nij0.reshape((self.Np_1*self.mr.Nc, 1))
        nijT = np.append(nij0, [[T]], axis=0)
        residuals = np.ones_like(nijT)
        it = 0
        while np.any(np.abs(residuals) > self.gibbs_eps) and it < self.gibbs_max_iter:
            nij = nijT[:-1]
            T = nijT[-1]
            nij_reshape = nij.reshape((self.Np_1, self.mr.Nc))
            F = np.sum(nij_reshape, axis=1).reshape((self.Np_1, 1)) / self.mr.n
            y = nij_reshape / (self.mr.n * F)
            y = np.append(y, np.array([self.z - np.sum(nij_reshape, axis=0) / self.mr.n]) / (1 - np.sum(F)), axis=0)
            eos = self.mr.eos_run(y, P, T, phases, **kwargs)
            nj = self.mr.n * np.append(F, np.array([[1 - np.sum(F)]]), axis=0)
            residuals = self.gibbs_equation(eos, nj, H)
            nijT = nijT - np.linalg.inv(self.gibbs_jacobian(eos, nj)).dot(residuals)
            it += 1
            if self.save_residuals:
                res.residuals_all.append(np.max(np.abs(residuals[:-1])))
                res.residuals_enthalpy_all.append(np.abs(residuals[-1][0]))
        nij_reshape = nij.reshape((self.Np_1, self.mr.Nc))
        res.T = T
        res.y = np.append(nij.reshape((self.Np_1, self.mr.Nc)) / (self.mr.n * F), np.array([self.z - np.sum(nij_reshape, axis=0) / self.mr.n]) / (1 - np.sum(F)), axis=0)
        res.eos = eos
        res.kv = res.y[:-1] / self.repeat(res.y[-1], axis=0, times=self.Np_1)
        res.it = it
        res.F = self.full_F(F)
        res.comp_mole_frac = dict(zip(phases, res.y))
        res.phase_mole_frac = dict(zip(phases, np.append(res.F, 1 - res.F)))
        res.residuals = residuals
        return res


# Выполним расчет равновесного состояния с использованием изоэнтальпийного метода минимизации энергии Гиббса, предварительно проведя 20 итераций изоэнтальпийного метода последовательных подстановок для приближения к минимимуму функции.

# In[59]:


flash9 = flash_isenthalpic_gibbs(mr1, derivatives_eos_2param, params, params_der, z, save_residuals=True, ssi_therm_max_iter=20, ssi_therm_scheme=3).flash_isenthalpic_gibbs_run(P, Hspec, 'ogw', T0=440, cw=0.0)


# In[60]:


flash9.T


# In[61]:


flash9.comp_mole_frac


# Сравним реализации представленных выше схем и методов. Для этого построим графики изменения максимальной абсолютной ошибки уравнения теплового баланса, максимальной абсолютной ошибки уравнений равновесия и количества итераций изотермического расчета.

# In[62]:


fig6, ax6 = plt.subplots(1, 3, figsize=(10.5, 4))
fig6.canvas.header_visible = False

iters_ssi_1 = np.arange(1, flash6.it + 1, 1)
iters_ssi_2 = np.arange(1, flash7.it + 1, 1)
iters_ssi_3 = np.arange(1, flash8.it + 1, 1)
iters_gibbs = np.arange(1, flash9.it + flash9.ssi_it + 1, 1)

ax6[0].plot(iters_ssi_1, flash6.residuals_enthalpy_all, label='SSI. Схема 1', c='b')
ax6[2].plot(iters_ssi_1, flash6.it_ssi_all, label='SSI. Схема 1', c='b')
ax6[0].plot(iters_ssi_2, flash7.residuals_enthalpy_all, label='SSI. Схема 2', c='cyan')
ax6[1].plot(iters_ssi_2, flash7.residuals_all, label='SSI. Схема 2', c='cyan')
ax6[0].plot(iters_ssi_3, flash8.residuals_enthalpy_all, label='SSI. Схема 3', c='g')
ax6[1].plot(iters_ssi_3, flash8.residuals_all, label='SSI. Схема 3', c='g')
ax6[2].plot(iters_ssi_3, flash8.it_ssi_all, label='SSI. Схема 3', c='g')
ax6[0].plot(iters_gibbs, flash9.residuals_enthalpy_all, label='Gibbs minimization', c='orange')
ax6[1].plot(iters_gibbs, flash9.residuals_all, label='Gibbs minimization', c='orange')

ax6[0].set_ylabel('Максимальная абсолютная ошибка\nуравнения теплового баланса')
ax6[1].set_ylabel('Максимальная абсолютная ошибка\nуравнений равновесного состояния')
ax6[2].set_ylabel('Накопленное количество итераций\nизотермического расчета\nравновесного состояния')
ax6[0].set_xlabel('Итерации решения уравнения\nтеплового баланса')
ax6[1].set_xlabel('Итерации решения уравнения\nтеплового баланса')
ax6[2].set_xlabel('Итерации решения уравнения\nтеплового баланса')
ax6[0].set_yscale('log')
ax6[1].set_yscale('log')
ax6[0].grid(True)
ax6[1].grid(True)
ax6[2].grid(True)
ax6[0].legend(loc='best')
ax6[1].legend(loc='best')
ax6[2].legend(loc='best')
ax6[0].set_xlim(0, 200)
ax6[1].set_xlim(0, 200)
ax6[2].set_xlim(0, 200)
ax6[2].set_ylim(0, 3500)

fig6.tight_layout()


# Из данного сопоставления видно, что, во-первых, в методе последовательных подстановок именно из-за условия достижения требуемой погрешности уравнения теплового баланса требуется существенное увеличение количества итераций. Во-вторых, комбинирование первой и второй схем позволяет наиболее оптимально подойти к решению поставленной задачи. В-третьих, применение метода энергии Гиббса позволяет значительно быстрее приблизиться к решению с учетом требуемой точности. В дополнении к проведенным двадцати итерациям метода последовательных подстановок потребовалось всего четыре итерации метода минимизации энергии Гиббса.

# Стоит отметить, что существуют и другие формулировки определения равновесного состояния системы. Например, в дополнении к $PT$ и $PH$ формулировкам также есть определение равновесного состояния при известных объеме $V$ и температуре $T$ системы. В этой формулировке определение равновесного состояния основано на поиске минимуме [энергии Гельмгольца](../1-TD/TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-helmholtz) методом Ньютона. Более подробно с данной формулировкой можно ознакомиться в работах \[[Nagarajan et al, 1991](https://doi.org/10.1016/0378-3812(91)80010-S); [Jindrova and Mikyska, 2013](https://doi.org/10.1016/j.fluid.2013.05.036); [Jindrova and Mikyska, 2015](https://doi.org/10.1016/j.fluid.2015.02.013)\].

# Таким образом в данном разделе были изложены основы расчета равновесного состояния многофазной многокомпонентной системы при заданных ее компонентном составе, а также термодинамических параметрах (давлении и температуре или давлении и энтальпии). При изучении метода [минимизации энергии Гиббса](#pvt-esc-equilibrium-isothermal-gem) было показано, что не только равновесное состояние характеризуется равенством летучестей компонентов (в изотермической постановке для квази-стационарного процесса). В общем виде это условие характерно для всех стационарных точек функции энергии Гиббса. Отсюда следует, что найденное таким образом решение может и не характеризоваться минимальной энергией Гиббса (или Гельмгольца), условием [необходимым](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium) для достижения равновесного состояния. В связи с этим необходимо проверить, является ли найденное решение равновесным, то есть понять, характеризуется ли оно минимальной энергией Гиббса (или Гельмгольца). Варианты численного подхода к решению данной задачи будут рассматриваться в следующем [разделе](ESC-2-Stability.html#pvt-esc-stability).

# Кроме того, основной сложностью изложенных выше методов является необходимость задания достаточно близких к искомому решению констант фазового равновесия. Именно от начального приближения, по сути, зависит, будет ли найден глобальный минимум функции энергии Гиббса (или Гельмгольца) в процессе итеративного приближения. При этом, если для трехфазных систем (жидкая преимущественно водная фаза, жидкая преимущественно углеводородная фаза и газовая фаза) еще существуют методы расчета начальных значений констант фазового равновесия, рассмотренные [ранее](#pvt-esc-equilibrium-isothermal-ssi-kvalues_init), то для четырехфазных (и более) систем, когда жидкая углеводородная фаза разделяется на две, инициализация констант фазового равновесия таким образом невозможна. Как задавать начальные значения констант фазового равновесия в таком случае, также будет рассмотрено в следующем [разделе](ESC-2-Stability.html#pvt-esc-stability).
