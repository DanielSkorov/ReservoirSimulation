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

<a id='pvt-parameters-enthalpy'></a>
# Энтальпия

+++

Энтальпия рассматриваемой системы, как и внутренняя энергия, определяется давлением, температурой и количеством вещества компонентов:

+++

$$H = H \left( P, T, n_i \right).$$

+++

Следовательно, дифференциал энтальпии можно записать следубщим образом:

+++

$$dH = \left( \frac{\partial H}{\partial P} \right)_{T, n_i} dP + \left( \frac{\partial H}{\partial T} \right)_{P, n_i} dT + \sum_i \left( \frac{\partial H}{\partial n_i} \right)_{P, T} dn_i.$$

+++

По [определению](../1-TD/TD-5-Enthalpy.html#pvt-td-enthalpy-isobaric_heat_capacity), изобарная теплоемкость:

+++

$$C_P = \left( \frac{\partial H}{\partial T} \right)_P.$$

+++

Запишем выражение [thermodynamic identity](../1-TD/TD-6-Entropy.html#pvt-td-entropy-thermodynamic_identity) выраженное через энтальпию системы:

+++

$$dH = T dS + V dP.$$

+++

Разделим левую и правую части на $dP$ и будем рассматривать изотермический процесс с постоянном количеством вещества компонентов:

+++

$$ \left( \frac{\partial H}{\partial P} \right)_{T, n_i} = T \left( \frac{\partial S}{\partial P} \right)_{T, n_i} + V.$$

+++

Используя [четвертое соотношение Максвелла](../1-TD/TD-13-MaxwellRelations.html#pvt-td-maxwell_relations-fourth), преобразуем данное выражение к следующему виду:

+++

$$ \left( \frac{\partial H}{\partial P} \right)_{T, n_i} = V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i}.$$

+++

Тогда дифференциал энтальпии:

+++

$$ dH = C_P dT + \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) dP + \sum_i \left( \frac{\partial H}{\partial n_i} \right)_{P, T} dn_i.$$

+++

<a id='pvt-parameters-enthalpy-partial_molar_enthalpy'></a>
Теперь рассмотрим третье слагаемое в данном выражении. [Ранее](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy) было показано, что при постоянных давлении и температуре энергия Гиббса системы определяется выражением:

+++

$$ G_i = \mu_i n_i.$$

+++

С учетом [определения энергии Гиббса](../1-TD/TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs) получим:

+++

$$ G_i = \mu_i n_i = H_i - T S_i.$$

+++

Рассмотрим частные производные по количеству вещества $i$-го компонента:

+++

$$ \left( \frac{\partial G_i}{\partial n_i} \right)_{P,T} = \mu_i = \left( \frac{\partial H_i}{\partial n_i} \right)_{P,T} - T \left( \frac{\partial S_i}{\partial n_i} \right)_{P,T}.$$

+++

Обозначим частную производную энергии Гиббса по количеству вещества $i$-го компонента как $\bar{G_i}$. Аналогично обозначим и частные производные энтальпии и энтропии. Тогда:

+++

$$\bar{G_i} = \mu_i = \bar{H_i} - T \bar{S_i}.$$

+++

Запишем вторую частную производную энергии Гиббса по температуре и количеству вещества $i$-го компонента, рассматривая изобарный процесс:

+++

$$ \left( \frac{\partial^2 G_i}{\partial n_i \partial T} \right)_P = \frac{\partial}{\partial n_i} \left( \left( \frac{\partial G_i}{\partial T} \right)_{P,n_i} \right)_{P,T} = \frac{\partial}{\partial T} \left( \left( \frac{\partial G_i}{\partial n_i} \right)_{P,T} \right)_{P,n_i}.$$

+++

Используя полученные [ранее](../1-TD/TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs_partials) частные производные энергии Гиббса, получим:

+++

$$-\left( \frac{\partial S}{\partial n_i} \right)_{P,T} = \left( \frac{\partial \mu_i}{\partial T} \right)_{P,n_i}.$$

+++

Тогда:

+++

$$\mu_i = \bar{H_i} + T \left( \frac{\partial \mu_i}{\partial T} \right)_{P,n_i}.$$

+++

Выразим из данного уравнения $\bar{h_i}$:

+++

$$ \bar{H_i} = \mu_i - T \left( \frac{\partial \mu_i}{\partial T} \right)_{P,n_i}.$$

+++

Разделим левую и правую части данного уравнения на $-T^2$ и немного преобразуем:

+++

$$-\frac{\bar{H_i}}{T^2} = \frac{T \left( \frac{\partial \mu_i}{\partial T} \right)_{P,n_i} - \mu_i \left( \frac{\partial T}{\partial T} \right)_{P,n_i}}{T^2} = \frac{\partial}{\partial T} \left( \frac{\mu_i}{T} \right)_{P,n_i} .$$

+++

Следовательно,

+++

$$\bar{H_i} = - T^2 \frac{\partial}{\partial T} \left( \frac{\mu_i}{T} \right)_{P,n_i}.$$

+++

С учетом этого, дифференциал энтальпии:

+++

$$ dH = C_P dT + \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) dP - T^2 \sum_i \frac{\partial}{\partial T} \left( \frac{\mu_i}{T} \right)_{P,n_i} dn_i.$$

+++

По аналогии с рассмотрением дифференциала внутренней энергии пренебрежем процессами, приводящими к изменению энтальпии системы при постоянных давлении и температуры. С учетом этого

+++

$$ dH = C_P dT + \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) dP.$$

+++

<a id='pvt-parameters-enthalpy-isobaric_heat_capacity'></a>
Исходя из полученного выражения дифференциала энтальпии, применяя [свойство частных производных](../1-TD/TD-13-MaxwellRelations.html#pvt-td-maxwell_relations), можно записать следующее выражение:

+++

$$ \left( \frac{\partial C_P}{\partial P} \right)_{T, n_i} = \frac{\partial}{\partial T} \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right)_P.$$

+++

Преобразуя выражение справа, получим:

+++

$$ \left( \frac{\partial C_P}{\partial P} \right)_{T, n_i} = - T \left( \frac{\partial^2 V}{\partial T^2} \right)_{P, n_i}.$$

+++

Тогда в изотермическом процессе:

+++

$$d C_P = - T \left( \frac{\partial^2 V}{\partial T^2} \right)_{P, n_i} dP.$$

+++

Интегрируя данное выражение от условий $ideal \; gas$ к $real \; gas$, получим:

+++

$$\int_{ideal \; gas}^{real \; gas} d C_P = C_P - C_P^* = - T \int_{0}^{P} \left( \frac{\partial^2 V}{\partial T^2} \right)_{P, n_i} dP.$$

+++

Тогда дифференциал энтальпии:

+++

$$ dH = \left( C_P^* - T \int_{0}^{P} \left( \frac{\partial^2 V}{\partial T^2} \right)_{P, n_i} dP \right) dT + \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) dP.$$

+++

Рассмотрим следующий подход к интегрированию данного выражения. Допустим, необходимо вычислить изменение энтальпии системы при ее переходе из точки $1: \; \left(P_1, T_1 \right)$ в точку $2: \; \left(P_2, T_2 \right)$. При этом, данный переход будет искусственно проходить через $P_3=0$, как показано на рисунке ниже.

+++

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import pyplot as plt

%matplotlib widget

fig0, ax0 = plt.subplots(figsize=(2.25, 2.25))
fig0.canvas.header_visible = False
ax0.set_xticks([0.0, 1.0])
ax0.set_xticklabels(['$T_1$', '$T_2$'])
ax0.set_yticks([0.0, 1.0, 2.0])
ax0.set_yticklabels(['$P_3$', '$P_1$', '$P_2$'])
ax0.set_xlabel('T')
ax0.set_ylabel('P')
ax0.set_xlim(-1, 2)
ax0.set_ylim(-1, 3)
ax0.plot([-1.0, 1.0], [0.0, 0.0], c='k', ls='--', lw=0.5)
ax0.plot([-1.0, 0.0], [1.0, 1.0], c='k', ls='--', lw=0.5)
ax0.plot([0.0, 0.0], [-1.0, 0.0], c='k', ls='--', lw=0.5)
ax0.plot([1.0, 1.0], [-1.0, 1.0], c='k', ls='--', lw=0.5)
ax0.plot([-1.0, 1.0], [2.0, 2.0], c='k', ls='--', lw=0.5)
ax0.arrow(0.0, 1.0, 0.0, -0.5, color='k', head_width=0.1)
ax0.arrow(0.0, 0.5, 0.0, -0.5, color='k', head_width=0.0)
ax0.arrow(0.0, 0.0, 0.5, 0.0, color='k', head_width=0.1)
ax0.arrow(0.5, 0.0, 0.5, 0.0, color='k', head_width=0.0)
ax0.arrow(1.0, 0.0, 0.0, 1.0, color='k', head_width=0.1)
ax0.arrow(1.0, 1.0, 0.0, 1.0, color='k', head_width=0.0)
ax0.scatter([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 2.0], color='k')
fig0.tight_layout()
```

+++

Тогда изменение энтальпии можно записать следующим образом:

+++

$$ \begin{alignat}{1}
\Delta H
&= & \; \int_{P_1}^{P_3} \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) \bigg\rvert_{T_1} dP + \int_{T_1}^{T_2} \left( C_P^* - T \int_{0}^{P_3} \left( \frac{\partial^2 V}{\partial T^2} \right)_{P, n_i} dP \right) dT \\
&& \; + \int_{P_3}^{P_2} \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) \bigg\rvert_{T_2} dP \\
&= & \; \int_{P_1}^0 \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) \bigg\rvert_{T_1} dP + \int_{T_1}^{T_2} C_P^* dT + \int_0^{P_2} \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) \bigg\rvert_{T_2} dP.
\end{alignat}$$

+++

В полученном выражении содержится частная производная объема по температуре при постоянном давлении и количестве вещества компонентов. Поскольку аналитическое выражение данной производной достаточно сложно получить, используя уравнения состояния, невыражаемые явно относительно объема, то необходимо преобразовать полученное выражение.

+++

Давление можно записать как функцию от температуры и объема (при постоянном количестве вещества компонентов) $P = P \left(V, T \right).$ В рассматриваемых изотермических условиях дифференциал давления:

+++

$$dP = \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV.$$

+++

Тогда:

+++

$$ \begin{align}
\left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) dP
&= \left( V - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \right) \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV \\
&= V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV.
\end{align} $$

+++

Второе слагаемое в полученном выражении можно упростить, применяя [свойство частных производных](https://en.wikipedia.org/wiki/Triple_product_rule):

+++

$$ V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV - T \left( \frac{\partial V}{\partial T} \right)_{P, n_i} \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV = V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} dV.$$

+++

С учетом этого изменение энтальпии:

+++

$$ \begin{align}
\Delta H = & \int_{V_1}^{\infty} \left( V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_1} dV + \int_{T_1}^{T_2} C_P^* dT + \\ & \int_{\infty}^{V_2} \left( V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_2} dV.
\end{align} $$

+++

Полученное выражение также можно несколько упростить.

+++

Для этого рассмотрим следующий интеграл:

+++

$$\int_{\infty}^{V} \left( V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T} dV. $$

+++

Пределы интегрирования $\infty$ и $V$, по сути, означают $ideal \; gas$ и $real \; gas$ соответственно. Раскроем скобки в данном интеграле и рассмотрим первое слагаемое:

+++

$$\int_{ideal \; gas}^{real \; gas} V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV.$$

+++

Согласно [правилу интегрирования по частям](https://en.wikipedia.org/wiki/Integration_by_parts), данное выражение можно преобразовать следующим образом:

+++

$$\int_{ideal \; gas}^{real \; gas} V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV = PV \bigg\rvert_{ideal \; gas}^{real \; gas} - \int_{ideal \; gas}^{real \; gas} \left( \frac{\partial V}{\partial V} \right)_{T, n_i} P dV = PV \bigg\rvert_{ideal \; gas}^{real \; gas} - \int_{ideal \; gas}^{real \; gas} P dV .$$

+++

С учетом выражения для коэффициента сверхсжимаемости получим:

+++

$$\int_{ideal \; gas}^{real \; gas} V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} dV = nRT \left(Z - 1 \right) - \int_{ideal \; gas}^{real \; gas} P dV.$$

+++

Тогда:

+++

$$ \int_{\infty}^{V} \left( V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T} dV = nRT \left(Z - 1 \right) + \int_{\infty}^{V} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_T dV . $$

+++

После проведенных преобразований изменение энтальпии:

+++

$$ \begin{alignat}{1}
\Delta H
&= & \; \int_{V_1}^{\infty} \left( V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_1} dV + \int_{T_1}^{T_2} C_P^* dT \\
&& \; + \int_{\infty}^{V_2} \left( V \left( \frac{\partial P}{\partial V} \right)_{T, n_i} + T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_2} dV \\
&= & \; - n R T_1 \left(Z_1 - 1 \right) - \int_{\infty}^{V_1} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_1} dV + \int_{T_1}^{T_2} C_P^* dT + n R T_2 \left( Z_2 - 1 \right) \\
&& \; + \int_{\infty}^{V_2} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_{T_2} dV.
\end{alignat} $$

+++

<a id='pvt-parameters-enthalpy-isobaric_isothermal'></a>
Если процесс происходит при постоянных давлении и температуре, то дифференциал энтальпии:

+++

$$dH = - T^2 \sum_i \frac{\partial}{\partial T} \left( \frac{\mu_i}{T} \right)_{P,n_i} dn_i.$$

+++

Разделив левую и правую часть на $dn_i$, получим:

+++

$$\left( \frac{\partial H}{\partial n_i} \right)_{P,T} = - T^2 \sum_i \frac{\partial}{\partial T} \left( \frac{\mu_i}{T} \right)_{P,n_i}.$$

+++

Рассмотрим следующее выражение, в котором обозначим химический потенциал $i$-го компонента при таких термобарических условиях, что его можно считать идеальным газом, $\mu_i^{ig}$:

+++

$$ \frac{\partial}{\partial T} \left( \frac{\mu_i - \mu_i^{ig}}{T} \right)_{P,n_i} = \frac{\partial}{\partial T} \left( \frac{\mu_i}{T} \right)_{P,n_i} - \frac{\partial}{\partial T} \left( \frac{\mu_i^{ig}}{T} \right)_{P,n_i}. $$

+++

С учетом выражения для парциальной молярной энтальпии компонента, полученной [ранее](#pvt-parameters-enthalpy-partial_molar_enthalpy):

+++

$$ \frac{\partial}{\partial T} \left( \frac{\mu_i - \mu_i^{ig}}{T} \right)_{P,n_i} = -\frac{\bar{H_i} - \bar{H_i^{ig}}}{T^2}.$$

+++

С другой стороны, преобразуем отношение разности химических потенциалов $i$-го компонента в реальных и идеальных условиях к температуре с учетом химического потенциала $i$-го компонента с использованием полученного [ранее](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity-real_gas_mixture) уравнения:

+++

$$\frac{\partial}{\partial T} \left( \frac{\mu_i - \mu_i^{ig}}{T} \right)_{P,n_i} = R \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i}.$$

+++

Объединяя данные выражения, получим:

+++

$$\bar{H_i} = \bar{H_i^{ig}} - R T^2 \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i}.$$

+++

Поскольку рассматривается процесс при постоянных давлении и температуре, а также учитывая то, что энтальпия является [экстенсивным параметром](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy-partial_molar_observables), энтальпия системы:

+++

$$ H = \sum_{i=1}^{N_c} n_i \bar{H_i} = \sum_{i=1}^{N_c} n_i \bar{H_i^{ig}} - R T^2 \sum_{i=1}^{N_c} n_i \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i}.$$

+++

Покажем, что $\bar{H_i^{ig}} = h^{ig}_i$, где $h^{ig}_i = h^{ig}_i \left( P, T \right)$ – удельная мольная энтальпия чистого компонента. Для этого в соответствии с [выражением парциальной молярной энтальпии компонента](#pvt-parameters-enthalpy-partial_molar_enthalpy) запишем:

+++

$$-\frac{\bar{H_i}}{T^2} = \frac{\partial}{\partial T} \left( \frac{\mu^{ig}_i \left( P, T, y_i \right)}{T} \right)_{P, n_i}.$$

+++

Применяя полученное [ранее](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity-ideal_gas_mixture-chemical_potential) уравнение для химического потенциала компонента идеального газа в смеси, а также учитывая свойство частной производной экстенсивного параметра по количеству вещества компонента идеального газа, рассмотренное [ранее](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy-partial_molar_observables-ideal_gas), преобразуем данное выражение:

+++

$$ \begin{align}
-\frac{\bar{H^{ig}_i}}{T^2}
&= \frac{\partial}{\partial T} \left( \frac{\mu^{ig}_i \left( P, T, y_i \right)}{T} \right)_{P, n_i} \\
&= \frac{\partial}{\partial T} \left( \frac{\mu^{ig}_i \left( P, T \right) + RT \ln y_i}{T} \right)_{P, n_i} \\
&= \frac{\partial}{\partial T} \left( \frac{\mu^{ig}_i \left( P, T \right)}{T} \right)_{P, n_i} \\
&= -\frac{h_i^{ig} \left(P, T \right)}{T^2}.
\end{align} $$

+++

Тогда энтальпия системы:

+++

$$ H = \sum_{i=1}^{N_c} n_i \bar{H_i} = \sum_{i=1}^{N_c} n_i h_i^{ig} - R T^2 \sum_{i=1}^{N_c} n_i \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i}.$$

+++

В свою очередь, удельная энтальпия:

+++

$$ h = \sum_{i=1}^{N_c} y_i \bar{H_i} = \sum_{i=1}^{N_c} y_i h_i^{ig} - R T^2 \sum_{i=1}^{N_c} y_i \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i}.$$

+++

Для систем, состоящих из одного компонента, полученное выше выражение преобразуется к следующему: 

+++

$$ h = h^{ig} - R T^2 \left( \frac{\partial \ln \phi}{\partial T} \right)_{P,n}.$$

+++

Для расчета энтальпии с использованием данных выражений необходимо определить частные производные логарифма коэффициента летучести $i$-го компонента по температуре. Как для смеси, так и для чистого компонента частные производные логарифма коэффициента летучести были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-srk_pr).

+++

<a id='pvt-parameters-enthalpy-srk_pr'></a>
## Вывод выражения энтальпии с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона

+++

Интеграл $\int_{\infty}^{V} \left( T \left( \frac{\partial P}{\partial T} \right)_{V, n_i} - P \right) \bigg\rvert_T dV$ был рассмотрен [ранее](./Parameters-1-InternalEnergy.html#pvt-parameters-internal_energy-srk_pr).

+++

Следовательно, изменение энтальпии:

+++

$$ \begin{align}
\Delta H = & \int_{T_1}^{T_2} C_P^* dT + n R \left( T_2 \left(Z_2 - 1 \right) - T_1 \left(Z_1 - 1 \right) \right) + \frac{n \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_2}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V_2 + b_m \delta_1}{V_2 + b_m \delta_2} \\ & - \frac{n \left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_1}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V_1 + b_m \delta_1}{V_1 + b_m \delta_2}.
\end{align} $$

+++

Изменение удельной (приведенной к единице количества вещества) энтальпии, выраженной через коэффициент сверхсжимаемости:

+++

$$ \begin{align}
\Delta h = & \int_{T_1}^{T_2} c_P^* dT + R \left( T_2 \left(Z_2 - 1 \right) - T_1 \left(Z_1 - 1 \right) \right) + \frac{\left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_2}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z_2 + \delta_1 B_2}{Z_2 + \delta_2 B_2} \\ & - \frac{\left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right) \bigg\rvert_{T_1}}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z_1 + \delta_1 B_1}{Z_1 + \delta_2 B_1}.
\end{align} $$

+++

Если рассматривается изотермический процесс $\left( T_1 = T_2 = T \right)$, при этом в качестве референсных условий принимаются условия идеального газа $\left( V_1 = \infty, P_1 = 0, Z_1 = 1 \right)$, тогда удельная энтальпия:

+++

$$ h \left( P, T, y_i \right) = h \left( 0, T, y_i \right) + R T \left( Z - 1 \right) + \frac{\left( \alpha_m - T \left( \frac{\partial \alpha_m}{\partial T} \right)_{V, n_i} \right)}{b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{Z + \delta_1 B}{Z + \delta_2 B}.$$

+++

Частная производная параметра $\alpha_m$ по температуре была выведена [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-srk_pr). Таким образом, данное выражение может быть использовано для расчета энтальпии фазы (системы), если известна величина $h \left( 0, T \right)$, обозначающую энтальпию при нулевом давлении и температуре $T$, которую можно интерпретировать как энтальпию идеального газа, являющуюся функцией температуры и не зависящую от давления и объема системы. Для расчета энтальпии компонента идеального газа чаще всего используется выражение:

+++

$$ h_i \left( 0, T \right) = \int_{T_{ref}}^{T} {c_P}^{ig}_i dT.$$

+++

То есть при некоторой температуре $T_{ref}$ и давлении $P=0$ энтальпия компонента принимается равной нулю, и относительно данного базиса с использованием изобарной теплоемкости компонента в состоянии идеального газа ${c_P}^{ig}_i$ рассчитывается его энтальпия $ h_i \left( 0, T \right)$. Подробнее подход к расчету изобарной теплоемкости системы, как идеального газа, будет показан в [соответствующем разделе](./Parameters-4-HeatCapacity.html#pvt-parameters-heat_capacity-ideal_gas). Энтальпия смеси компонентов идеального газа определяется с учетом экстенсивности энтальпии как параметра:

+++

$$h \left(0, T, y_i \right) = \sum_{i=1}^{N_c} y_i h_i \left( 0, T \right). $$

+++

Таким образом, для энтальпии смеси референсными условиями являются $P = 0, \; T = T_{ref}.$ В основном, для расчета энтальпии системы используется приведенное выше выражение, поскольку оно показывает, насколько отличается энтальпия системы при неком давлении $P$, температуре $T$ и компонентном составе $y_i$ от энтальпии системы при ее идеализированном состоянии (давлении $P=0$, температуре $T=T_{ref}$ и компонентном составе $y_i$), то есть оно показывает изменение энтальпии системы в изотермическом процессе при ее переходе от идеального состояния к реальному.

+++

<a id='pvt-parameters-enthalpy-sw'></a>
## Вывод выражения энтальпии с использованием уравнения состояния Сорейде-Уитсона

+++

Поскольку [уравнение состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw) отличается от [уравнения состояния Пенга-Робинсона](../2-EOS/EOS-2-SRK-PR.html#pvt-eos-srk_pr) только расчетом парамета $a_i$ для воды, как компонента, а также определением коэффициентов попарного взаимодействия между водой и растворенными компонентами, то определение энтальпии системы с использованием уравнения состояния Сорейде-Уитсона будет отличаться от рассмотренного [ранее](#pvt-parameters-enthalpy-srk_pr) определения энтальпии системы с использованием уравнения состояния Пенга-Робинсона только нахождением частной производной параметра $\alpha_m$ по температуре. Для уравнения состояния Сорейде-Уитсона частные производные параметров были рассмотрены [ранее](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-sw).

+++

Рассмотрим нахождение энтальпии соленой воды с использованием уравнения состояния Сорейде-Уитсона. Для сравнения экспериментальные данные были взяты из работы \[[Haas, 1976](https://doi.org/10.3133/b1421B)\].

+++

```{code-cell} ipython3
import sys
sys.path.append("../../SupportCode/")
from PVT import core
```

+++

```{code-cell} ipython3
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
```

+++

```{code-cell} ipython3
df_haas = pd.read_excel(io='../../SupportCode/SaltWaterProperties.xlsx', sheet_name='Haas1976')
```

+++

```{code-cell} ipython3
:tags: [hide-input]

Tcw = 647.096
Pcw = 22.064 * 10**6
R = 8.314
a = 0.45724 * R**2 * Tcw**2 / Pcw
b = 0.07780 * R * Tcw / Pcw
d1 = 1 - 2**0.5
d2 = 1 + 2**0.5

def Psat_func(P, T, cw):
    alpha = a * (1 + 0.4530 * (1 - T * (1 - 0.0103 * cw**1.1) / Tcw) + 0.0034 * ((T / Tcw)**(-3) - 1))**2
    A = alpha * P / (R**2 * T**2)
    B = b * P / (R * T)
    Zs = core.calc_cardano(B - 1, A - 2 * B - 3 * B**2, -(A * B - B**2 - B**3))
    Z1 = max(Zs)
    Z2 = min(Zs)
    return (Z1 - Z2) - (np.log((Z1 - B) / (Z2 - B)) - A * np.log(((Z1 + B * d1) / (Z1 + B * d2)) * ((Z2 + B * d2) / (Z2 + B * d1))) / (B * (d2 - d1)))

# from EOS import mix_rules_sw, eos_sw
# from PD import derivatives_sw
# y = np.array([[1]])
# Pc = np.array([22.05])*10**6
# Tc = np.array([647.3])
# w = np.array([0.344])
# z = np.array([0.7597])
# phases = 'w'
# comp_type = np.array([2])
# Akl = Bkl = np.array([[0]])
# alpha_matrix = np.array([[1]])
# def dlnphi_dT(P, T, cw, Zl=False, Zv=False):
#     mr_pure = mix_rules_sw(T, Pc, Tc, w, comp_type, Akl, Bkl, alpha_matrix, phases='w', sw=True, cw=cw)
#     eos = eos_sw().eos_run(mr_pure, y, P, 'w', Zl, Zv)
#     return derivatives_sw(eos, y, P, n=y, phases='w', der_T=True).dlnphi_dT

# sys.path.append('./Code/')
# from Parameters import parameters_2param
# from EOS import eos_sw
# from EOS_PD import derivatives_eos_2param
# Pc = np.array([22.064 * 10**6])
# Tc = np.array([647.096])
# w = np.array([0.344])
# comp_type = np.array([2])
# Akl = np.array([[0.0]])
# Bkl = np.array([[0.0]])
# alpha_matrix = np.array([[1.0]])
# mr = eos_sw(Pc, Tc, w, comp_type, Akl, Bkl, alpha_matrix)
# y = np.array([[1.0]])
# cp_matrix = cp_ig = np.array([[3.376336E+01], [-5.945958E-03], [2.235754E-05], [-9.962009E-09], [1.097487E-12]])
# enthalpy_l = lambda P, T, cw: parameters_2param(cp_matrix).enthalpy(mr, mr.eos_run(y, P, T, 'w', cw=cw, Zl=True), y, 273.15)
# enthalpy_v = lambda P, T, cw: parameters_2param(cp_matrix).enthalpy(mr, mr.eos_run(y, P, T, 'w', cw=cw, Zv=True), y, 273.15)

fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))
fig2.canvas.header_visible = False
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

cp1 = 3.376336E+01
cp2 = -5.945958E-03
cp3 = 2.235754E-05
cp4 = -9.962009E-09
cp5 = 1.097487E-12

def h_ig(T, Tr):
    return (cp1 * (T - Tr) + cp2 * (T**2 - Tr**2) / 2 + cp3 * (T**3 - Tr**3) / 3 + cp4 * (T**4 - Tr**4) / 4 + cp5 * (T**5 - Tr**5) / 5)

for i, cw in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
    df_temp = df_haas[df_haas['Salt Molality, gmole/kg'] == cw]
    tt = df_temp['Temperature, C']
    pp = df_temp['Pressure, bar'] / 10
    hhv = df_temp['Vapour Enthalpy, J/gmole'].to_numpy(dtype=np.float64)
    hhl = df_temp['Liquid Enthalpy, J/gmole'].to_numpy(dtype=np.float64)
    hhv = (hhv - hhv[0]) / 10**3
    hhl = (hhl - hhl[0]) / 10**3
    ax2[0].plot(tt, hhl, '.', label='{} моль/кг (Haas, 1976)'.format(cw), c=colors[i])
    ax2[1].plot(tt, hhv, '.', label='{} моль/кг (Haas, 1976)'.format(cw), c=colors[i])
    hvs = []
    hls = []
    for p, t in zip(pp, tt):
        T = t + 273.15
        Psat = fsolve(Psat_func, x0=p*10**6, args=(T, cw))[0]
        alpha = a * (1 + 0.4530 * (1 - T * (1 - 0.0103 * cw**1.1) / Tcw) + 0.0034 * ((T / Tcw)**(-3) - 1))**2
        dalpha_dT = 2 * np.sqrt(a * alpha) * (-0.4530 * (1 - 0.0103 * cw**1.1) / Tcw - 0.0102 * Tcw**3 / T**4)
        A = alpha * Psat / (R**2 * T**2)
        B = b * Psat / (R * T)
        Zs = core.calc_cardano(B - 1, A - 2 * B - 3 * B**2, -(A * B - B**2 - B**3))
        Zv = max(Zs)
        Zl = min(Zs)
        hvs.append(h_ig(T, 273.15) + R * T * (Zv - 1) + (alpha - T * dalpha_dT) * np.log((Zv + d1 * B) / (Zv + d2 * B)) / (b * (d2 - d1)))
        hls.append(h_ig(T, 273.15) + R * T * (Zl - 1) + (alpha - T * dalpha_dT) * np.log((Zl + d1 * B) / (Zl + d2 * B)) / (b * (d2 - d1)))
        # hvs.append(h_ig(T, 273.15) - R * T**2 *  dlnphi_dT(Psat, T, cw, Zv=True)[0][0])
        # hls.append(h_ig(T, 273.15) - R * T**2 *  dlnphi_dT(Psat, T, cw, Zl=True)[0][0])
        # hvs.append(enthalpy_v(Psat, T, cw)[0][0])
        # hls.append(enthalpy_l(Psat, T, cw)[0][0])
    hvs = (np.array(hvs) - hvs[0]) / 10**3
    hls = (np.array(hls) - hls[0]) / 10**3
    ax2[0].plot(tt, hls, label='{} моль/кг (SW EOS)'.format(cw), c=colors[i])
    ax2[1].plot(tt, hvs, label='{} моль/кг (SW EOS)'.format(cw), c=colors[i])
ax2[0].grid()
ax2[0].set_ylabel('Энтальпия на линии испарения, кДж/моль\nреференсные условия: ' + r'$H_{sat} \; \left( P=P_{sat}, T = 80 \; °C \right)$')
ax2[0].set_xlabel('Температура, °C')
ax2[0].set_ylim(0, 25)
ax2[0].set_xlim(80, 330)
ax2[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax2[1].grid()
ax2[1].set_ylabel('Энтальпия на линии конденсации, кДж/моль\nреференсные условия: ' + r'$H_{due} \; \left( P=P_{sat}, T = 80 \; °C \right)$')
ax2[1].set_xlabel('Температура, °C')
ax2[1].set_ylim(0, 5)
ax2[1].set_xlim(80, 330)
ax2[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
fig2.tight_layout()
```

+++

Представленное на рисунке выше сопоставление расчетных и фактических данных позволяет сделать вывод о том, что использование [уравнения состояния Сорейде-Уитсона](../2-EOS/EOS-3-SW.html#pvt-eos-sw), несмотря на достаточно точное воспроизведение зависимости давления насыщения воды от температуры и солености, весьма ограничено. Следовательно, для более точного моделирования термодинамических параметров систем необходимо учитывать межмолекулярное взаимодействие.

+++

<a id='pvt-parameters-enthalpy-cpa'></a>
## Вывод выражения энтальпии с использованием уравнения состояния CPA

+++

```{code-cell} ipython3

```

+++