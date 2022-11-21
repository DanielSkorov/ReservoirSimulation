#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-eos-appendix-bip'></a>
# # Универсальный расчет коэффициентов попарного взаимодействия
# В данном приложении изложены ряд подходов, в том числе универсальный (погрупповой \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\]), для расчета коэффициентов попарного взаимодействия при использовании [уравнений состояния Суаве-Редлиха-Квонга, Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr) и [правил смешивания Ван-дер-Ваальса](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-mix_rules).

# Параметр $\delta_{jk}$, коэффициент попарного взаимодействия компонентов, играет важную роль при расчете PVT-свойств флюидов с использованием уравнений состояния. [Ранее](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-bip) было показано, что коэффициент попарного взаимодействия между углеводородными компонентами можно оценить, используя их критические объемы $V_c$:
# 
# $$\delta_{jk} = 1 - \left( \frac{2 {V_c}_j^{\frac{1}{6}} {V_c}_k^{\frac{1}{6}}}{{V_c}_j^{\frac{1}{3}} + {V_c}_k^{\frac{1}{3}}} \right)^c.$$
# 
# В данном выражении параметр $c$ подбирается при адаптации модели на лабораторные исследования. В качестве начального значения можно использовать $c = 1.2$, согласно \[[Oellrich et al, 1981](https://api.semanticscholar.org/CorpusID:94056056)\]. Расчет коэффициентов попарного взаимодействия между углеводородными компонентами (и углеводородными гетероатомными, например, с включениями серы – меркаптаны) также рассматривается в работах \[[Slot-Petersen, 1983](https://doi.org/10.2118/16941-PA); [Gao et al, 1992](https://doi.org/10.1016/0378-3812(92)85054-C) [Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059); [Jaubert et al, 2005](https://doi.org/10.1016/j.fluid.2005.09.003); [Vitu et al, 2006](https://doi.org/10.1016/j.fluid.2006.02.004); [Privat et al, 2008](https://doi.org/10.1016/j.jct.2008.05.013); [Privat and Jaubert, 2012](https://doi.org/10.1016/j.fluid.2012.08.007); [Qian et al, 2013](https://doi.org/10.1016/j.fluid.2013.06.040); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\]. Кроме того, коэффициенты попарного взаимодействия исследовались и для уравнения состояния Суаве-Редлиха-Квонга: \[[Mathias, 1983](https://doi.org/10.1021/i200022a008); [Coutinho et al, 1994](https://doi.org/10.1016/0378-3812(94)87090-X)\].

# <a id='pvt-eos-appendix-bip-kato'></a>
# Для неуглеводородных компонентов коэффициенты попарного взаимодействия можно оценить с использованием корреляций. Например, в работе \[[Kato et al, 1981](https://doi.org/10.1016/0378-3812(81)80009-X)\] приводится корреляция для расчета коэффициента попарного взаимодействия между диоксидом углерода и углеводородными компонентами для уравнения состояния Пенга-Робинсона:
# 
# $$ \delta_{ij} = a \cdot \left( T - b \right)^2 + c, $$
# 
# где коэффициенты $a, \; b, \; c$ рассчитываются по функции от ацентрического фактора углеводородного компонента:
# 
# $$ \begin{align} a &= -0.70421 \cdot 10^{-5} \lg \omega - 0.132 \cdot 10^{-7}; \\ b &= 301.58 \omega + 226.57; \\ c &= -0.0470356 \left( \lg \omega + 1.08884 \right)^2 + 0.13040. \end{align} $$
# 
# Данная корреляция получена регрессией для алканового ряда до нормального декана. Также, зависимости коэффициента попарного взаимодействия между диоксидом углерода и углеводородными компонентами исследуются в работах \[[Mulliken and Sandler, 1980](https://doi.org/10.1021/i260076a033); [Turek et al, 1984](https://doi.org/10.2118/9231-PA); [Nishiumi et al, 1988](https://doi.org/10.1016/0378-3812(88)80049-9); [Pedersen et al, 2001](https://doi.org/10.1016/S0378-3812(01)00562-3); [Mutelet et al, 2005](https://doi.org/10.1016/j.fluid.2005.10.001); [Vitu et al, 2008](https://doi.org/10.1016/j.supflu.2007.11.015); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\].

# <a id='pvt-eos-appendix-bip-nishumi'></a>
# В работе \[[Nishiumi et al, 1988](https://doi.org/10.1016/0378-3812(88)80049-9)\] приводится обобщенная зависимость коэффициентов попарного взаимодействия для углеводородных (алканов до $C_{20}$, алкенов, циклоалканов, ароматических углвеодородов) и неуглеводородных компонентов (диоксида углерода, азота и сероводорода) для уравнения состояния Пенга-Робинсона в следующем виде:
# 
# $$ m_{ij} = 1 - \delta_{ij} = C + D \frac{{V_c}_i}{{V_c}_j} + E \left( \frac{{V_c}_i}{{V_c}_j} \right)^2. $$
# 
# В данном уравнении коэффициенты $C, \; D, \; E$ зависят от ацентрических факторов, а также типа компонента:
# 
# $$ \begin{align} C &= c_1 + c_2 \left| \omega_i - \omega_j \right|; \\ D &= d_1 + d_2 \left| \omega_i - \omega_j \right|. \end{align}$$
# 
# В следующей таблице приводятся группы коэффициентов $c_1, \; c_2, \; d_1, \; d_2, \; E$ в зависимости от типа компонента:

# ````{div} full-width
# <style type="text/css">
# .tg  {border-color:#ccc;border-spacing:0;margin:20px auto;}
# .tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
#   font-family:Palatino, sans-serif;font-size:14px;overflow:hidden;padding:10px 16px;word-break:normal;}
# .tg th{border-color:#ccc;border-style:solid;border-width:1px;color:#333;
#   font-family:Palatino, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 16px;word-break:normal;}
# .tg .tg-0pky{border-color:inherit;text-align:center;vertical-align:center}
# .tg .tg-1pky{background-color:#f0f0f0;border-color:inherit;text-align:center;vertical-align:center;font-weight:bold}
# .tg .tg-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:center}
# </style>
# <table class="tg">
#     <thead>
#         <tr>
#             <th class="tg-1pky"></th>
#             <th class="tg-1pky">Алканы<br>C<sub>1</sub> – C<sub>16</sub></th>
#             <th class="tg-1pky">Циклоалканы</th>
#             <th class="tg-1pky">Алкены</th>
#             <th class="tg-1pky">Ароматические<br>у/в</th>
#             <th class="tg-1pky">Алканы<br>C<sub>18</sub> – C<sub>20</sub></th>
#             <th class="tg-1pky">Диоксид<br>углерода</th>
#             <th class="tg-1pky">Сероводород</th>
#             <th class="tg-1pky">Азот</th>
#             <th class="tg-1pky">Ацетилен</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tg-1pky">Алканы<br>C<sub>1</sub> – C<sub>16</sub></td>
#             <td class="tg-0pky">1</td>
#             <td class="tg-0pky">1</td>
#             <td class="tg-0pky">2</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">4</td>
#             <td class="tg-0pky">5</td>
#             <td class="tg-0pky">7</td>
#             <td class="tg-0pky">8</td>
#             <td class="tg-0pky">10</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Циклоалканы</td>
#             <td class="tg-0pky">1</td>
#             <td class="tg-0pky">1</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">5</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Алкены</td>
#             <td class="tg-0pky">2</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">2</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">4</td>
#             <td class="tg-0pky">6</td>
#             <td class="tg-0pky">7</td>
#             <td class="tg-0pky">8</td>
#             <td class="tg-0pky">9</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Ароматические<br>у/в</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">3</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">6</td>
#             <td class="tg-0pky">7</td>
#             <td class="tg-0pky">12</td>
#             <td class="tg-0pky">11</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Алканы<br>C<sub>18</sub> – C<sub>20</sub></td>
#             <td class="tg-0pky">4</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">4</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">4</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Диоксид<br>углерода</td>
#             <td class="tg-0pky">5</td>
#             <td class="tg-0pky">5</td>
#             <td class="tg-0pky">6</td>
#             <td class="tg-0pky">6</td>
#             <td class="tg-0pky">4</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">12</td>
#             <td class="tg-0pky">13</td>
#             <td class="tg-0pky">-</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Азот</td>
#             <td class="tg-0pky">7</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">7</td>
#             <td class="tg-0pky">7</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">12</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">14</td>
#             <td class="tg-0pky">-</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Сероводород</td>
#             <td class="tg-0pky">8</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">8</td>
#             <td class="tg-0pky">12</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">13</td>
#             <td class="tg-0pky">14</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#         </tr>
#         <tr>
#             <td class="tg-1pky">Ацетилен</td>
#             <td class="tg-0pky">10</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">9</td>
#             <td class="tg-0pky">11</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#             <td class="tg-0pky">-</td>
#         </tr>
#     </tbody>
# </table>
# ````

# Для каждой группы значения коэффициентов $c_1, \; c_2, \; d_1, \; d_2, \; E$:

# <style type="text/css">
# .tb  {border-color:#ccc;border-spacing:0;margin:20px auto;}
# .tb td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
#   font-family:Palatino, sans-serif;font-size:14px;overflow:hidden;padding:10px 44px;word-break:normal;}
# .tb th{border-color:#ccc;border-style:solid;border-width:1px;color:#333;
#   font-family:Palatino, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 44px;word-break:normal;}
# .tb .tb-0pky{border-color:inherit;text-align:center;vertical-align:center}
# .tb .tb-1pky{background-color:#f0f0f0;border-color:inherit;text-align:center;vertical-align:center;font-weight:bold}
# .tb .tb-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:center}
# </style>
# <table class="tb">
#     <thead>
#         <tr>
#             <th class="tb-1pky">Группа</th>
#             <th class="tb-1pky">c<sub>1</sub></th>
#             <th class="tb-1pky">c<sub>2</sub></th>
#             <th class="tb-1pky">d<sub>1</sub></th>
#             <th class="tb-1pky">d<sub>2</sub></th>
#             <th class="tb-1pky">E</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="tb-abip">1</td>
#             <td class="tb-0pky">1.041</td>
#             <td class="tb-0pky">0.11</td>
#             <td class="tb-0pky">-0.0403</td>
#             <td class="tb-0pky">0.0367</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">2</td>
#             <td class="tb-0pky">1.017</td>
#             <td class="tb-0pky">-0.417</td>
#             <td class="tb-0pky">-0.0124</td>
#             <td class="tb-0pky">0.0852</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">3</td>
#             <td class="tb-0pky">1.025</td>
#             <td class="tb-0pky">0.317</td>
#             <td class="tb-0pky">-0.0385</td>
#             <td class="tb-0pky">-0.0258</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">4</td>
#             <td class="tb-0pky">0.823</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0.0673</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">-0.0051</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">5</td>
#             <td class="tb-0pky">0.883</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0.0023</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">6</td>
#             <td class="tb-0pky">0.948</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">-0.0084</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">7</td>
#             <td class="tb-0pky">0.982</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">-0.0241</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">8</td>
#             <td class="tb-0pky">0.907</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0.0109</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">9</td>
#             <td class="tb-0pky">1.090</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">-0.1435</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">10</td>
#             <td class="tb-0pky">0.855</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">11</td>
#             <td class="tb-0pky">0.965</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">12</td>
#             <td class="tb-0pky">1.016</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">13</td>
#             <td class="tb-0pky">0.894</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#         <tr>
#             <td class="tb-abip">14</td>
#             <td class="tb-0pky">0.848</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#             <td class="tb-0pky">0</td>
#         </tr>
#     </tbody>
# </table>

# Также зависимости коэффициента попарного взаимодействия между азотом и углеводородными компонентами исследуются в работах \[[Katz and Firoozabadi, 1978](https://doi.org/10.2118/6721-PA); [Mehra, 1981](http://dx.doi.org/10.11575/PRISM/13997); [Pedersen et al, 2001](https://doi.org/10.1016/S0378-3812(01)00562-3); [Privat et al, 2008](https://doi.org/10.1021/ie800636h); [Privat et al, 2008](https://doi.org/10.1021/ie071524b); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\].

# Зависимости коэффициента попарного взаимодействия между кислородом, углеводородными и неуглеводородными компонентами рассматриваются в работе \[[Xu et al, 2015](https://doi.org/10.1021/acs.iecr.5b02639)\]. Зависимости коэффициента попарного взаимодействия между водородом, сероводородом и другими углеводородоными и неуглеводородными компонентами приводятся в работах \[[Valderama et al, 1990](https://doi.org/10.1016/0009-2509(90)87079-8); [Qian et al, 2013](https://doi.org/10.1021/ie402541h); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\].

# Самостоятельный подход для описания фазового поведения воды с учетом ее минерализации был рассмотрен [ранее](./EOS-3-SW.html#pvt-eos-sw). Зависимости коэффициента попарного взаимодействия между водой, углеводородными и неуглеводородными компонентами для уравнения состояния Пенга-Робинсона изучаются в работах \[[Qian et al, 2013](https://doi.org/10.1021/ie402541h)\].

# Таким образом, учитывая колоссальное количество корреляций для коэффициентов попарного взаимодействия, возникает задача разработки универсального подхода к расчету коэффициентов попарного взаимодействия. Данная задача решается в рамках работы \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\]. Авторами предложен погрупповой, по аналогии с рассмотренным [ранее](#pvt-eos-appendix-bip-nishumi), подход к определению коэффициентов попарного взаимодействия для [уравнения состояния Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr) и [правил смешивания Ван-дер-Ваальса](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-mix_rules) с тем отличием, что в рамках универсального подхода учитывается температурная зависимость коэффициентов попарного взаимодействия. При этом, в качестве исходных данных для расчета коэффициентов попарного взаимодействия необходимы критические давления компонентов ${P_c}_i$, критические температуры компонентов ${T_c}_i$, а также ацентрические факторы компонентов $\omega_i$, то есть те же самые параметры, необходимые для использования [уравнения состояния Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr).

# <a id='pvt-eos-appendix-bip-gcm'></a>
# Согласно \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\], температурная зависимость коэффициента попарного взаимодействия определяется следующим выражением:
# 
# $$ k_{ij} \left( T \right) = \frac{-\frac{1}{2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 1} - \left( \delta_i - \delta_j \right)^2}{2 \delta_i \delta_j},$$
# 
# где $\delta_i = \frac{\sqrt{\alpha_i}}{b_i}$ (параметры $\alpha_i, \; b_i$ определяются в соответствии с [уравнением состояния Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr)), $N_g$ – количество групп (соответственно $k, \; l$ – индексы групп), $\alpha_{ik}$ – доля молекулы $i$, занятая группой $k$ (то есть количество появлений группы $k$ в молекуле $i$, деленное на общее количество групп в молекуле $i$), параметры $A_{kl} = A_{lk}, \; B_{kl} = B_{lk}$ – коэффициенты, подобранные авторами работы \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\] для различных групп ($A_{kk} = B_{kk} = 0$). Следовательно, с учетом данной формулы для расчета коэффициента попарного взаимодействия молекул необходимо знать критические свойства компонентов, их ацентрические факторы и структуру молекулы.

# Приведенное выше выражение берет свое начало из описания термодинамической системы решеточной моделью ([lattice model](https://en.wikipedia.org/wiki/Lattice_model_(physics))). Фундамент для вывода данного выражения приводится в работах \[[Redlich et al, 1959](https://doi.org/10.1021/ja01519a001); [Kehiaian et al, 1971](https://doi.org/10.1051/jcp/1971680922); [Kehiaian, 1985](https://doi.org/10.1351/pac198557010015); [Tine and Kehiaian, 1987](https://doi.org/10.1016/0378-3812(87)85056-2); [Abdoul et al, 1991](https://doi.org/10.1016/0378-3812(91)85010-R)\].

# Коэффициенты $A_{kl} = A_{lk}, \; B_{kl} = B_{lk}$ в МПа представлены в работе \[[Qian et al, 2013](https://doi.org/10.1021/ie402541h)\] и следующих таблицах:

# In[1]:


import pandas as pd
pd.options.display.max_columns = None
cols = ['CH3', 'CH2', 'CH', 'C', 'CH4', 'C2H6', 'CH (aro)', 'C (aro)', 'C (fused aro rings)', 'CH2 (cyclic)', 'CH (cyclic) | C (cyclic)', 'CO2', 'N2',
        'H2S', 'SH', 'H2', 'C2H4', 'CH2 (alkenic) | CH (alkenic)', 'C (alkenic)', 'CH (cycloalkenic) | C (cycloalkenic)', 'H2O']
df_a = pd.read_excel(io='../../SupportCode/BIPCoefficients.xlsx', sheet_name='A', usecols='D:Y', skiprows=[0, 1, 2], index_col=0)
dct = dict(zip(list(df_a.columns), cols))
df_a_mod = df_a.rename(index=dct)
df_a_mod = df_a_mod.rename(columns=dct)
df_a_mod


# In[2]:


df_b = pd.read_excel(io='../../SupportCode/BIPCoefficients.xlsx', sheet_name='B', usecols='D:Y', skiprows=[0, 1, 2], index_col=0)
df_b_mod = df_b.rename(index=dct)
df_b_mod = df_b_mod.rename(columns=dct)
df_b_mod


# Рассмотрим пример расчета коэффициента попарного взаимодействия с использованием данного выражения. В качестве примера возьмем смесь нормального бутана $ \left( CH_3 - CH_2 - CH_2 - CH_3 \right) $ и пропана $ \left( CH_3 - CH_2 - CH_3 \right) $ при температуре $T = 303.15 \; K$.

# In[3]:


R = 8.314472
T = 303.15


# Пропан состоит из двух групп $\#1:CH_3$ и одной группы $\#2:CH_2$, следовательно, общее количество групп ${N_g}_1 = 3$. Нормальный бутан состоит из двух групп $\#1:CH_3$ и двух групп $\#2:CH_2$, следовательно, общее количество групп ${N_g}_2 = 4$. Поскольку в данном примере используются всего две группы, то матрицы $A_{kl}$ и $B_{kl}$ можно упростить до:

# In[4]:


import numpy as np
Akl = np.array([[0.0, 74.81], [74.81, 0.0]])*10**6
Bkl = np.array([[0.0, 165.7], [165.7, 0.0]])*10**6


# Доля группы $\#1:CH_3$ в составе молекулы пропана:
# 
# $$ \alpha_{11} = \frac{2}{3}. $$
# 
# Аналогично доля группы $\#2:CH_2$ в составе молекулы пропана:
# 
# $$ \alpha_{12} = \frac{1}{3}. $$
# 
# Для остальных групп $k = 3 \ldots 21$ доля в составе молекулы пропана:
# 
# $$ \alpha_{1k} = 0, \; k = 3 \dots 21. $$
# 
# Для молекулы нормального бутана:
# 
# $$ \begin{align} \alpha_{21} &= \frac{2}{4} = \frac{1}{2}; \\ \alpha_{22} &= \frac{2}{4} = \frac{1}{2}; \\ \alpha_{2k} &= 0, \; k = 3 \dots 2. \end{align} $$
# 
# Тогда сформируем два вектора следующим образом:

# In[5]:


alpha1 = np.array([[2/3], [1/3]])
alpha2 = np.array([[1/2], [1/2]])


# Определим значение параметра $DS$ (double sum):
# 
# $$DS = -\frac{1}{2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 1}.$$

# In[6]:


DS = np.sum(np.outer(alpha1 - alpha2, alpha1 - alpha2) * Akl * (298.15 / T) ** (np.divide(Bkl, Akl, out=np.ones_like(Akl), where=Akl!=0) - 1)) / (-2)
DS


# Определим значения параметров $\alpha_i$ и $b_i$ для компонентов с учетом их критических свойств и ацентрического фактора:

# In[7]:


Pc = np.array([4.248, 3.796])*10**6
Tc = np.array([369.83, 425.12])
w = np.array([0.152, 0.2])


# In[8]:


ai = 0.45724 * (R * Tc)**2 / Pc
alphai = ai * (1 + (0.37464 + 1.54226 * w - 0.26992 * w**2) * (1 - (T / Tc)**0.5))**2
bi = 0.07780 * R * Tc / Pc


# Определим значение $\delta_i$ для каждого компонента:

# In[9]:


deltai = alphai**0.5 / bi
deltai


# Тогда коэффициент попарного взаимодействия:

# In[10]:


kij = (DS - (deltai[0] - deltai[1])**2) / (2 * deltai[0] * deltai[1])
kij


# ```{admonition} NB
# Очевидно, что изложенный выше подход к расчету коэффициентов попарного взаимодействия применим для компонентов с известной молекулярной структурой. Для псевдокомпонентов, характеризующих, скорее фракцию, чем конкретное вещество, данный подход может быть использован для определеления начальных значений коэффициентов попарного взаимодействия с целью последующего уточнения значений в процессе адаптации модели.
# ```
