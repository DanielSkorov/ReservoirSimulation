---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<a id='pvt-eos-sw'></a>
# Уравнение состояния Сорейде-Уитсона
Рассмотренные [ранее](./EOS-2-SRK-PR.html#pvt-eos-srk_pr) уравнения состояния характеризуются достаточно широким применением при рассмотрении вопросов, связанных с поведением углеводородов. Однако данные уравнения состония достаточно [плохо](https://doi.org/10.1002/aic.690200504) подходят для описания взаимодействия воды (растворов солей) и углеводородов. Для моделирования трехфазной системы (в том числе для учета влияния растворенных в водной фазе солей на растворимость в ней других компонентов) Сорейде и Уитсоном была [предложена](http://dx.doi.org/10.1016/0378-3812(92)85105-H) модификация уравнения состояния Пенга-Робинсона. В данном разделе рассматривается применение уравнения состояния Сорейде-Уитсона.

+++

Сорейде и Уитсоном была предложена модификация [уравнения состояния Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr), включающая следующие изменения. [Параметр](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-mix_rules) $\alpha_m$ рассчитывается для водной и неводных (нефтяной и газовой) фаз по-разному. Для нефтяной и газовой фаз ($NA$) параметр $\alpha_m^{NA}$:

$$ \alpha_m^{NA} = \sum_j \sum_k x_j x_k \left( \alpha_i \alpha_j \right)^{0.5} \left( 1 - \delta_{jk}^{NA} \right). $$

Коэффициент попарного взаимодействия для нефтяной и газовой фаз $\delta_{jk}^{NA}$ может быть рассчитан в соответствии с [ранее изложенным подходом](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-bip). Для водной фазы $\left( AQ \right)$ параметр $\alpha_m^{AQ}$:

$$ \alpha_m^{AQ} = \sum_j \sum_k x_j x_k \left( \alpha_i \alpha_j \right)^{0.5} \left( 1 - \delta_{jk}^{AQ} \right). $$

Для расчета коэффициента попарного взаимодействия компонентов в водной фазе предлагается использовать корреляцию от безразмерной температуры ${T_r}_j = \frac{T}{{T_c}_j}$ и моляльности $c_w$ (моль $NaCl$ / кг воды):

$$ \delta_{jk}^{AQ} = A_0 \left( 1 + \alpha_0 c_w \right) + A_1 {T_r}_j \left( 1 + \alpha_1 c_w \right) + A_2 {{T_r}_j}^2 \left( 1 + \alpha_2 c_w \right). $$

В данном уравнении индекс $k$ обозначает воду, как компонент, с растворенной в ней солью, а индекс $j$ - все остальные компоненты. Коэффициенты в данном уравнении были подобраны для компонентов метана, этана, пропана и нормального бутана:

+++

$A_0$|$A_1$|$A_2$
----|----|----
$1.1120 - 1.7369 \omega_j^{-0.1}$|$1.001 + 0.8360 \omega_j$|$-0.15742-1.0988 \omega_j$

+++

$\alpha_0$|$\alpha_1$|$\alpha_2$
----|----|----
$0.017407$|$0.033516$|$0.011478$

+++

Приведенные выше коэффициенты были подобраны для следующих диапазонов параметров:

+++

<style type="text/css">
.tg  {border-color:#ccc;border-spacing:0;margin:20px auto;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Palatino, sans-serif;font-size:14px;overflow:hidden;padding:10px 38px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Palatino, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-1pky{border-color:inherit;text-align:center;vertical-align:top;font-weight:bold}
.tg .tg-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
    <thead>
        <tr>
            <th class="tg-1pky">Параметр</th>
            <th class="tg-1pky">Метан</th>
            <th class="tg-1pky">Этан</th>
            <th class="tg-1pky">Пропан</th>
            <th class="tg-1pky">н-Бутан</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="tg-abip">Давление (бар)</td>
            <td class="tg-0pky">14 – 690</td>
            <td class="tg-0pky">14 – 690</td>
            <td class="tg-0pky">14 – 207</td>
            <td class="tg-0pky">14 – 690</td>
        </tr>
        <tr>
            <td class="tg-abip">Температура (°C)</td>
            <td class="tg-0pky" colspan="4">38 – 204</td>
        </tr>
        <tr>
            <td class="tg-abip">Моляльность NaCl (моль/кг)</td>
            <td class="tg-0pky" colspan="4">0 – 5</td>
        </tr>
    </tbody>
</table>

+++

Для неуглеводородных компонентов (диоксида углерода, азота и сероводорода) коэффициент попарного взаимодействия рекомендуется рассчитывать по следующим уравнениям:

$$ \begin{align}
\delta_{jk}^{AQ} \left( N_2 \right) &= -1.70235 \left( 1 + 0.025587 c_w^{0.75} \right) + 0.44338 \left( 1 + 0.08126 c_w^{0.75} \right) {T_r}_j; \\
\delta_{jk}^{AQ} \left( CO_2 \right) &= -0.31092 \left( 1 + 0.15587 c_w^{0.7505} \right) + 0.23580 \left( 1 + 0.17837 c_w^{0.979} \right) {T_r}_j - 21.2566 e^{-6.7222 {T_r}_j - c_w}; \\
\delta_{jk}^{AQ} \left( H_2S \right) &= -0.20441 + 0.23426 {T_r}_j.
\end{align} $$

Приведенные уравнения были подобраны регрессией для следующих диапазонов давления и температуры:

+++

<style type="text/css">
.tb  {border-color:#ccc;border-spacing:0;margin:20px auto;}
.tb td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Palatino, sans-serif;font-size:14px;overflow:hidden;padding:10px 93.5px;word-break:normal;}
.tb th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Palatino, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tb .tb-0pky{border-color:inherit;text-align:center;vertical-align:top}
.tb .tb-1pky{border-color:inherit;text-align:center;vertical-align:top;font-weight:bold}
.tb .tb-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tb">
    <thead>
        <tr>
            <th  class="tb-1pky">Система</th>
            <th  class="tb-1pky">Давление (бар)</th>
            <th  class="tb-1pky">Температура (°C)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="tb-abip">N<sub>2</sub> / H<sub>2</sub>O</td>
            <td class="tb-0pky">14 – 1035</td>
            <td class="tb-0pky">25 – 100</td>
        </tr>
        <tr>
            <td class="tb-abip">N<sub>2</sub> / H<sub>2</sub>O + NaCl</td>
            <td class="tb-0pky">100 – 600</td>
            <td class="tb-0pky">52 – 125</td>
        </tr>
        <tr>
            <td class="tb-abip">CO<sub>2</sub> / H<sub>2</sub>O</td>
            <td class="tb-0pky">25 – 620</td>
            <td class="tb-0pky">12 – 50</td>
        </tr>
        <tr>
            <td class="tb-abip">CO<sub>2</sub> / H<sub>2</sub>O + NaCl</td>
            <td class="tb-0pky">145 – 970</td>
            <td class="tb-0pky">150 – 350</td>
        </tr>
        <tr>
            <td class="tb-abip">H<sub>2</sub>S / H<sub>2</sub>O</td>
            <td class="tb-0pky">10 – 345</td>
            <td class="tb-0pky">38 – 204</td>
        </tr>
    </tbody>
</table>

+++

Параметр $\alpha_j$ для компонентов, за исключением воды, предлагается рассчитывать аналогично подходу, применяемому Пенгом и Робинсоном:

$$ \alpha_j^{NA} = a_j \left( 1 + \kappa_j^{NA} \left( 1 - \sqrt{{T_r}_j} \right) \right)^2. $$

Здесь параметр $a_j$ определяется так же, как и для уравнения состояния Пенга-Робинсона:

$$ a_j = \Omega_a \frac{R^2 {T_c}_j^2}{{P_c}_j}. $$

Для расчета параметра $\kappa_j^{NA}$ можно использовать одну из [ранее](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-kappa) рассмотренных корреляций от ацентрического фактора компонента $\omega_j$. 

+++

А для воды, как компонента, с растворенным в ней содержанием хлорида натрия параметр $\alpha_j$ определяется следующим выражением:

$$ \alpha_w = a_w \left( 1 + 0.4530 \left( 1 - {T_r}_w \left( 1 - 0.0103 c_w^{1.1} \right) \right) + 0.0034 \left({T_r}_w^{-3} - 1 \right) \right)^2. $$

Данное выражени было настроено на лабораторные исследования давления насыщения водной фазы при изменении температуры в диапазоне $ 0 \; \unicode{xB0} C - 325 \; \unicode{xB0} C$ и изменении моляльности хлорида натрия $ 0 \; \frac{моль}{кг} - 5  \; \frac{моль}{кг}$.

+++

Очевидно, что использование различных подходов к расчету коэффициентов попарного взаимодействия компонентов в водной и неводных фазах (нефтяной и газовой) ограничивается ситуацией, когда взаимное растворение настолько велико, что фазы могут содержать практически равное количество вещества компонентов \[[Pedersen et al, 2001](https://doi.org/10.1016/S0378-3812(01)00562-3)\]. Тем не менее уравнение состояния Сорейде-Уитсона может использоваться для решения практических задач в области низкой растворимости компонентов в водной фазе.

+++

Кроме того, в методике Сорейде-Уитсона есть другие недостатки. Авторами описывается расчет коэффициентов попарного взаимодействия между компонентами и водой в водной фазе, в то время как расчет коэффициентов попарного взаимодействия между самими компонентами в водной фазе не излагается. В связи с этим, для расчета коэффициентов попарного взаимодействия будет использоваться методика [GCM](./EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip-gcm). Кроме того, в связи с малым количеством данных Сорейде и Уитсоном не предложен подход для описания коэффициентов попарного взаимодействия между компонентами и водой в неводных фазах (нефтяной и газовой). Следовательно, в этом случае также будет использоваться методика [GCM](./EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip-gcm). Таким образом, уравнение состояния Сорейде-Уитсона полезно для моделирования изменения растворимости легкий углеводородных компонентов и неуглеводородных газов (диоксида углерода, сероводорода, азота) в водной фазе с учетом ее минерализации.

+++

Поскольку внесенные Сорейде и Уитсоном модификации не повлияют на нахождение частных производных от параметров $\alpha_m$ и $b_m$ по количеству вещества $i$-го компонента, рассмотренные [ранее](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-mix_rules), то логарифм коэффициента летучести будет соответствовать полученному в [предыдущем разделе](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-fugacity_coeff-pt) выражению:

$$ \ln \phi_i = -\ln \left( Z - B \right) + \frac{b_i}{b_m} \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right) $$

Здесь параметр $\alpha_{ij}$ определяется следующим выражением:

$$ \alpha_{ij} = \left( \alpha_i \alpha_j \right)^{0.5} \left( 1 - \delta_{ij} \right). $$

С учетом изложенного выше рассмотрим задачу определения давления насыщения водной фазы при различных температурах и солености.

+++

[Ранее](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium-chemical_potential) было показано, что фазы находятся в равновесии, если равны их химические потенциалы:

$$ \mu_1 \left(P, T \right) = \mu_2 \left(P, T \right). $$

Дифференциал химического потенциала компонента определяется следующим выражением:

$$ d \mu = RT d \ln f. $$

Интегрируя данное выражение, получим:

$$ \mu = \mu_0 + RT \ln \frac{f}{f_0}. $$

Следовательно, для чистого компонента на линии насыщения равенство химических потенциалов заменяется равеноством летучестей этого компонента в жидкой и газовой фазах:

$$ f_1 \left(P, T \right) = f_2 \left(P, T \right). $$

Получим выражение для расчета летучести чистого компонента. Дифференциал энергии Гиббса в изотермическом процессе с постоянным количеством частиц ([thermodynamic identity](../1-TD/TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs_partials)):

$$ dG = VdP. $$

С другой стороны, для изотермического процесса с постоянным количеством частиц дифференциал энергии Гиббса [определяется](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity):

$$ dG = nRT d \ln f. $$

Следовательно,

$$ d \ln f = \frac{1}{n R T} V dP. $$

Интегрируя данное выражение, получим:

$$ \ln \frac{f}{f_0} = \frac{1}{nRT} \int_{P_0}^{P} V dP. $$

Поскольку из кубического уравнения состояния Сорейде-Уитсона проще выразить давление $P$, то применим к рассматриваемому интегралу правило [интегрирования по частям](https://en.wikipedia.org/wiki/Integration_by_parts):

$$ \int_{P_0}^P V dP = PV \bigg\rvert_{P_0 V_0}^{PV} - \int_{V_0}^{V} P dV = \left( PV - P_0 V_0 \right) - \int_{V_0}^{V} P dV .$$

Тогда выражение для летучести можно записать следующим образом:

$$ \ln f = \ln f_0 + \frac{1}{nRT} \int_{P_0}^P V dP = \ln f_0 + \frac{1}{nRT} \left( PV - P_0 V_0 \right) - \frac{1}{nRT} \int_{V_0}^{V} P dV. $$

Рассмотрим выражение:

$$ f_1 \left(P, T \right) - f_2 \left( P, T \right) = 0. $$

Оно равносильно следующему:

$$ \ln f_1 \left(P, T \right) - \ln f_2 \left( P, T \right) = 0. $$

Левую часть этого выражения можно преобразовать:

$$ \begin{alignat}{1}
\ln f_1 \left(P, T \right) - \ln f_2 \left( P, T \right)
&= & \; \ln f_0 + \frac{1}{nRT} \left( P V_1 - P_0 V_0 \right) - \frac{1}{nRT} \int_{V_0}^{V_1} P dV - \ln f_0 \\
&& \; - \frac{1}{nRT} \left( P V_2 - P_0 V_0 \right) + \frac{1}{nRT} \int_{V_0}^{V_2} P dV \\
&= & \; \frac{P}{nRT} \left( V_1 - V_2 \right) - \frac{1}{nRT} \int_{V_2}^{V_1} P dV.
\end{alignat}$$

Первообразная интеграла $\int P dV$ с использованием уравнения состояния Пенга-Робинсона была получена [ранее](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-fugacity_coefficient). Следовательно,

$$ \begin{alignat}{1}
\ln f_1 \left(P, T \right) - \ln f_2 \left( P, T \right)
&= & \; \frac{P}{nRT} \left( V_1 - V_2 \right) - \frac{1}{nRT} \left( n R T \ln \left( V_1 - n b \right) - \frac{\alpha n}{b \left( \delta_2 - \delta_1 \right)} \ln \frac{V_1 + b n \delta_1}{V_1 + b n \delta_2} \right. \\
&& \; \left. - n R T \ln \left( V_2 - n b \right) + \frac{\alpha n}{b \left( \delta_2 - \delta_1 \right)} \ln \frac{V_2 + b n \delta_1}{V_2 + b n \delta_2} \right) \\
&= & \; \left( Z_1 - Z_2 \right) - \left( \ln \frac{Z_1 - B}{Z_2 - B} - \frac{A}{B \left( \delta_2 - \delta_1 \right)} \ln \left( \frac{Z_1 + B \delta_1}{Z_1 + B \delta_2} \frac{Z_2 + B \delta_2}{Z_2 + B \delta_1} \right) \right).
\end{alignat} $$

Данное выражение можно вывести, если использовать полученное [ранее](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-fugacity_coeff-pt) выражение для логарифма коэффициента летучести чистого компонента.

```{code-cell} python
import numpy as np
import sys
sys.path.append('../../SupportCode/')
from PVT import core
```

```{code-cell} python
Tcw = 647.096
Pcw = 22.064 * 10**6
R = 8.314
a = 0.45724 * R**2 * Tcw**2 / Pcw
b = 0.07780 * R * Tcw / Pcw
d1 = 1 - 2**0.5
d2 = 1 + 2**0.5
def Psat(P, T, cw):
    alpha = a * (1 + 0.4530 * (1 - T * (1 - 0.0103 * cw**1.1) / Tcw) + 0.0034 * ((T / Tcw)**(-3) - 1))**2
    A = alpha * P / (R**2 * T**2)
    B = b * P / (R * T)
    Zs = core.calc_cardano(B - 1, A - 2 * B - 3 * B**2, -(A * B - B**2 - B**3))
    Z1 = max(Zs)
    Z2 = min(Zs)
    return (Z1 - Z2) - (np.log((Z1 - B) / (Z2 - B)) - A * np.log(((Z1 + B * d1) / (Z1 + B * d2)) * ((Z2 + B * d2) / (Z2 + B * d1))) / (B * (d2 - d1)))
```

```{code-cell} python
from iapws import IAPWS97
from scipy.optimize import fsolve
tt = np.linspace(0, 325, 50)
pp_sw = []
pp_iapws = []
for t in tt:
    p0 = IAPWS97(T=t + 273.15, x=0).P
    pp_iapws.append(p0)
    pp_sw.append(fsolve(Psat, x0=p0*10**6, args=(t + 273.15, 0))[0] / 10**6)
```

```{code-cell} python
:tags: [hide-input]

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': False})
%matplotlib widget
fig1, ax1 = plt.subplots(figsize=(6, 4))
fig1.canvas.header_visible = False
ax1.grid()
ax1.set_xlabel('Температура, °C')
ax1.set_ylabel('Давление, МПа')
ax1.set_xlim(tt[0], tt[-1])
ax1.set_ylim(pp_sw[0], pp_sw[-1])
ax1.plot(tt, pp_iapws, label='IAPWS97')
ax1.plot(tt, pp_sw, '.', label='Расчет')
ax1.legend(loc='best');

fig1.tight_layout()
```

Уравнение состояния Сорейде-Уитсона достаточно точно определяет значение давления насыщения чистой воды. Проверим соответствие расчетных и экспериментальных значений \[[Haas, 1976](https://doi.org/10.3133/b1421B); [Hubert et al, 1995](https://doi.org/10.1021/je00020a034)\] давления насыщения для соленой воды.

```{code-cell} python
import pandas as pd
df_haas = pd.read_excel(io='../../SupportCode/SaltWaterProperties.xlsx', sheet_name='Haas1976')
df_haas
```

```{code-cell} python
:tags: [hide-input]

fig2, ax2 = plt.subplots(figsize=(6, 4))
fig2.canvas.header_visible = False
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for i, salt_molal in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
    df_temp = df_haas[df_haas['Salt Molality, gmole/kg'] == salt_molal]
    tt = df_temp['Temperature, C']
    pp = df_temp['Pressure, bar'] / 10
    ax2.plot(tt, pp, '.', label='{} моль/кг (Haas, 1976)'.format(salt_molal), c=colors[i])
    pp_calc = []
    for p, t in zip(pp, tt):
        pp_calc.append(fsolve(Psat, x0=p*10**6, args=(t + 273.15, salt_molal))[0] / 10**6)
    ax2.plot(tt, pp_calc, label='{} моль/кг (SW EOS)'.format(salt_molal), c=colors[i])
ax2.legend(loc='best')
ax2.grid()
ax2.set_ylabel('Давление, МПа')
ax2.set_xlabel('Температура, °C')
ax2.set_ylim(0, 13)
ax2.set_xlim(70, 330)

fig2.tight_layout()
```

```{code-cell} python
df_hubert = pd.read_excel(io='../../SupportCode/SaltWaterProperties.xlsx', sheet_name='Hubert1995')
df_hubert
```

```{code-cell} python
:tags: [hide-input]

fig3, ax3 = plt.subplots(figsize=(6, 4))
fig3.canvas.header_visible = False
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for i, salt_molal in enumerate(list(set(df_hubert['Salt Molality, gmole/kg'].values.tolist()))):
    df_temp = df_hubert[df_hubert['Salt Molality, gmole/kg'] == salt_molal]
    tt = df_temp['Temperature, K'] - 273.15
    pp = df_temp['Pressure, Pa'] / 10**3
    ax3.plot(tt, pp, '.', label='{} моль/кг (Hubert, 1995)'.format(salt_molal), c=colors[i])
    pp_calc = []
    for p, t in zip(pp, tt):
        pp_calc.append(fsolve(Psat, x0=p*10**3, args=(t + 273.15, salt_molal))[0] / 10**3)
    ax3.plot(tt, pp_calc, label='{} моль/кг (SW EOS)'.format(salt_molal), c=colors[i])
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax3.grid()
ax3.set_ylabel('Давление, кПа')
ax3.set_xlabel('Температура, °C')
ax3.set_ylim(0, 90)
ax3.set_xlim(20, 100)

fig3.tight_layout()
```

Из приведенных выше рисунков видно, что уравнение состояния Сорейде-Уитсона позволяет достаточно точно рассчитать давление насыщения водной фазы с растворенным в ней количеством соли (хлорида натрия). Сопоставление фактических и расчетных значений [энтальпии](../3-Parameters/Parameters-2-Enthalpy.html#pvt-parameters-enthalpy-sw) и [энтропии](../3-Parameters/Parameters-3-Entropy.html#pvt-parameters-entropy-sw) водной фазы, содержащей соль в растворенном виде, будет рассмотрено в следующем разделе. Также уравнение состояния Сорейде-Уитсона [будет использоваться](../4-ESC/ESC-1-Equilibrium.html#pvt-esc-equilibrium-isothermal-example_1) для определения мольных долей компонентов в водной фазе при различных термобарических условиях.

+++

Из представленных на рисунках зависимостях видно, что с увеличением количества растворенных в водной фазе соли хлорида натрия снижается давление насыщения при одной и той же температуре. Это объясняется тем, что процесс испарения – это процесс, который протекает на границе раздела фаз. Если в воде растворить некоторое количество солей, то концентрация молекул воды как во всем объеме, так и на поверхности снизится. Это приведет к тому, что часть молекул воды на поверхности будет замещена на молекулы (ионы) растворенного вещества, следовательно, меньше молекул растворителя будет в газовой фазе, что приведет к снижению давления насыщения. Иными словами, для того чтобы молекула растворителя в растворе перешла в газовую фазу, ей необходимо сообщить большее количество кинетической энергии (путем, например, нагрева), чтобы она перешла в газовую фазу. При этом, чем больше количество растворенного вещества, тем ниже кривая насыщения. Данная закономерность описывается [законом Рауля](https://en.wikipedia.org/wiki/Raoult%27s_law):

```{prf:закон}
:nonumber:
Если нелетучее вещество (с давлением насыщения близким к нулю) растворить в некотором растворителе и при этом образуется идеальный раствор, то давление насыщения раствора будет меньше давления насыщения растворителя; снижение давления насыщения будет прямо пропорционально мольной доле растворенного вещества.
```

Стоит отметить, что закон Рауля справедлив только для идеальных растворов – таких растворов, компоненты которых близки по своим физическим и химическим свойствам. В этом случае силы межмолекулярного взаимодействия близки между собой, и образование раствора обусловлено [вторым началом термодинамики](../1-TD/TD-6-Entropy.html#pvt-td-entropy-second_law-entropy). Для реальных растворов закон Рауля справедлив в области низких концентраций. При больших концентрациях в реальных растворах взаимодействие между разнородными частицами и однородными различно. Поэтому факт снижения давления насыщения (при постоянной температуре) или увеличения температуры насыщения (при постоянном давлении) можно рассмотреть с точки зрения межмолекулярного (межатомного) взаимодействия. Существует несколько типов сил, возникающих между атомами и молекулами в растворе. Первый тип – [ионные силы](https://en.wikipedia.org/wiki/Intermolecular_force#Ionic_bonding), возникающие между заряженным частицами (ионами). Такие силы описываются [законом Кулона](https://en.wikipedia.org/wiki/Coulomb's_law), определяющим зависимость величины силы от значений зарядов и расстояния между ними. Ионное взаимодействие является наиболее сильным, поэтому ионизация раствора приводит к увеличению температуры насыщения при одном и том же давлении. [Водородные силы](https://en.wikipedia.org/wiki/Intermolecular_force#Hydrogen_bonding) возникают между молекулами, имеющими в своем составе [электроотрицательные](https://en.wikipedia.org/wiki/Electronegativity) элементы (например, кислород, фтор, азот), напрямую связанные с водородом. Поскольку электроотрицательность водорода (2.2) значительно больше электроотрицательности натрия (0.93), то такого типа связи характеризуются меньшей полярностью, чем ионные связи, однако все равно формирует [дипольную структуру молекулы](https://en.wikipedia.org/wiki/Dipole#Molecular_dipoles). [Диполь-диполь взаимодействие](https://en.wikipedia.org/wiki/Intermolecular_force#Dipole%E2%80%93dipole_and_similar_interactions) происходит между молекулами, в которых присутствует сильно электроотрицательный элемент, связанный с другим, более электроотрицательным, чем водородод (например, углерод с электроотрицательностью 2.55), элементом. Таким молекулам также свойственно дипольное строение, однако в этом случае говорят о "частично отрицательном заряде" и "частично положительном заряде" в молекуле. Природа возникновения диполь-диполь взаимодействия схожа с водородными силами, однако характеризуется меньшей энергией из-за большей электроотрицательности углерода по сравнению с водородом. Кроме того, существуют Ван-Дер-Ваальсовы силы, возникающие между молекулами и атомами. К таким относится [взаимодействие между полярной и неполярной молекулами](https://en.wikipedia.org/wiki/Intermolecular_force#Debye_(permanent%E2%80%93induced_dipoles)_force), когда в результате действия дипольной молекулы происходит поляризация изначально неполярной молекулы. Такое взаимодействие слабее, чем диполь-диполь взаимодействие. Наконец, также имеются [дисперсионные силы](https://en.wikipedia.org/wiki/London_dispersion_force), возникающие в результате мгновенной поляризации атома или молекулы, вызванной рандомным изменением ([флуктуацией](https://en.wikipedia.org/wiki/Thermal_fluctuations)) электронной плотности атома или молекулы. Такое взаимодействие слабее взаимодействия между полярной и неполярной молекулами.

+++

Таким образом, на величину давления насыщения оказывает вляиние не только концентрация растворенных солей, но и тип взаимодействия между молекулами. Для учета межмолекулярного взаимодействия рассмотрим уравнение состояния с учетом ассоциации молекул.
