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

<a id='pvt-eos-srk_pr'></a>
# Уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона
В 1949 году Редлих и Квонг предложили модификацию [уравнения состояния Ван-дер-Ваальса](EOS-1-VanDerWaals.html#pvt-eos-van_der_waals):

+++

$$ \left( P + \frac{a_m}{T^{0.5} v \left( v + b_m \right)} \right) \left(v - b_m \right) = R T.$$

+++

Позднее, в 1972 году Суаве улучшил данное уравнение состояния, заменив отношение $\frac{a_m}{T^{0.5}}$ более общим коэффициентом $\alpha_m$, зависящим от температуры и свойств компонента, в том числе *ацентрического фактора*.

+++

```{prf:определение}
:nonumber:
***Ацентрический фактор*** – параметр компонента (вещества), который характеризует отклонение формы молекулы компонента от сферической молекулы идеального газа.
```

+++

В общем виде уравнение состояния Суаве-Редлиха-Квонга можно записать в следующем виде:

+++

$$ \left( P + \frac{\alpha_m}{v \left( v + b_m \right)} \right) \left(v - b_m \right) = R T. $$

+++

Позднее, в 1976 году, Пенг и Робинсон внесли модификацию в данное уравнение состояния для лучшего воспроизведения плотности жидкой фазы:

+++

$$ \left( P + \frac{\alpha_m}{v \left( v + b_m \right) + b_m \left( v - b_m \right)} \right) \left(v - b_m \right) = R T. $$

+++

В общем виде, оба уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона можно записать в следующем виде:

+++

$$ \left( P + \frac{\alpha_m}{v^2 + v b_m \left(1 + c \right) - c b_m^2} \right) \left(v - b_m \right) = R T. $$

+++

Если параметр $c = 0$, то уравнение приводится к уравнению состояния Суаве-Редлиха-Квонга, если же $c = 1$, то – к уравнению состояния Пенга-Робинсона.

+++

<a id='pvt-eos-srk_pr-fugacity_coefficient'></a>
Начнем с нахождения логарифма коэффициента летучести $i$-го компонента. Для этого преобразуем полученное [ранее](../1-TD/TD-15-Fugacity.md) выражение следующим образом:

+++

$$ \ln \phi_i = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} - \frac{1}{V} \right) dV - \ln Z = \frac{\partial}{\partial n_i} \left( \int_V^\infty \left( \frac{P}{RT} - \frac{n}{V} \right) dV \right)_{V, T, n_{j \neq i}} - \ln Z.$$

+++

Запишем рассматриваемые уравнения состояния относительно давления:

+++

$$ P = \frac{RT}{v - b_m} - \frac{\alpha_m}{v^2 + \left( c + 1 \right) b_m v - c b_m^2} = \frac{n R T}{V - n b_m} - \frac{\alpha_m n^2}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2}. $$

+++

Получим выражение для первообразной подынтегральной функции:

+++

$$ \begin{align}
F \left( V \right)
&= \int \left( \frac{P}{RT} - \frac{n}{V} \right) dV \\
&= \frac{1}{RT} \int P dV - n \int \frac{dV}{V} \\
&= n \int \frac{dV}{V - n b_m} - \frac{n^2 \alpha_m}{RT} \int \frac{dV}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2} - n \int \frac{dV}{V}.
\end{align} $$

+++

Для преобразования данного выражения распишем подробнее интегралы:

+++

$$ \begin{align}
\int \frac{dV}{V - n b_m} = \int \frac{dV - n b_m}{V - n b_m}
&= \ln \lvert V - n b_m \rvert = \ln \left( V - n b_m \right); \\
\int \frac{dV}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2}
&= \int \frac{dV}{ \left( V + \frac{c+1}{2} b_m n \right)^2 - b_m^2 n^2 \left( \frac{\left( c+1 \right)^2}{4} + c \right) } \\
&= \frac{1}{2 b_m n \sqrt{\frac{\left( c + 1 \right)^2}{4} + c }} \ln \left| \frac{ V + \frac{c+1}{2} b_m n - b_m n \sqrt{ \frac{\left( c+1 \right)^2}{4} + c }}{V + \frac{c+1}{2} b_m n + b_m n \sqrt{ \frac{\left( c+1 \right)^2}{4} + c }} \right| \\
&= \frac{1}{b_m n \left( \delta_2 - \delta_1 \right) } \ln \frac{V + b_m n \delta_1}{V + b_m n \delta_2}; \\
\int \frac{dV}{V}
&= \ln \lvert V \rvert = \ln V.
\end{align} $$

+++

Здесь $\delta_1 = \frac{c + 1}{2} - \sqrt{ \frac{\left( c+1 \right)^2}{4} + c }$ и $\delta_2 = \frac{c + 1}{2} + \sqrt{ \frac{\left( c+1 \right)^2}{4} + c }.$ С учетом этого выражение для первообразной подынтегральной функции примет следующий вид:

+++

$$ F \left( V \right) = n \ln \frac{V - n b_m}{V} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2}.$$

+++

Следовательно, интеграл в выражении для коэффициента летучести $i$-го компонента:

+++

$$ \begin{alignat}{1}
\int_V^\infty \left( \frac{P}{RT} - \frac{n}{V} \right) dV
&= & \; F \left( V \right) \bigg\rvert_V^\infty \\
&= & \; \lim_{b \rightarrow \infty} F \left( V \right) \bigg\rvert_V^b \\
&= & \; \lim_{b \rightarrow \infty} \left( n \ln \frac{b - n b_m}{b} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{b + n b_m \delta_1}{b + n b_m \delta_2} \right) \\
&& \; - \left( n \ln \frac{V - n b_m}{V} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= & \; n \lim_{b \rightarrow \infty} \ln \left( 1 - \frac{n b_m}{b} \right) - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \lim_{b \rightarrow \infty} \ln \left( 1 - \frac{n b_m \left( \delta_2 - \delta_1 \right)}{b + n b_m \delta_2} \right) \\
&& \; - \left( n \ln \frac{V - n b_m}{V} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= & \; - n \ln \frac{V - n b_m}{V} + \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2}
\end{alignat} $$

+++

Для выражения логарифма коэффициента летучести $i$-го компонента необходимо получить производную по количеству вещества $i$-го компонента при постоянных объеме, температуре и количествах веществ других компонентов:

+++

$$ \begin{align} & \frac{\partial}{\partial n_i} \left( - n \ln \frac{V - n b_m}{V} + \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right)_{V, T, n_{j \neq i}} \\ & = - \frac{\partial}{\partial n_i} \left( n \ln \frac{V - n b_m}{V} \right) + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \frac{\partial}{\partial n_i} \left( \frac{n \alpha_m}{b_m} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right). \end{align} $$

+++

Распишем подробнее данные производные.

+++

$$ \begin{align}
\frac{\partial}{\partial n_i} \left( n \ln \frac{V - n b_m}{V} \right)
&= \ln \frac{V - n b_m}{V} + n \frac{\partial}{\partial n_i} \ln \frac{V - n b_m}{V} \\
&= \ln \frac{V - n b_m}{V} - \frac{n}{V - n b_m} \frac{\partial n b_m}{\partial n_i}.
\end{align}$$

+++

$$\begin{alignat}{1}
\frac{\partial}{\partial n_i} \left( \frac{n \alpha_m}{b_m} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right)
&= & \; \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \frac{\partial}{\partial n_i} \left( \frac{n \alpha_m}{b_m} \right) + \frac{n \alpha_m}{b_m} \frac{\partial}{\partial n_i} \left( \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= & \; \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \\
&& \; + \frac{n \alpha_m}{b_m} \frac{\partial n b_m}{\partial n_i} \frac{V \left( \delta_1 - \delta_2 \right)}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)}.
\end{alignat} $$

+++

С учетом этого выражение для логарифма коэффициента летучести $i$-го компонента:

+++

$$ \begin{alignat}{1}
\ln \phi_i
&= & \; -\ln \frac{V - n b_m}{V} + \frac{n}{V - n b_m} \frac{\partial n b_m}{\partial n_i} \\
&& \; + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \right. \\
&& \; \left. - \frac{n \alpha_m}{b_m} \frac{\partial n b_m}{\partial n_i} \frac{V \left( \delta_2 - \delta_1 \right)}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) - \ln Z \\
&= & \; -\ln \frac{V - n b_m}{V}  + \frac{n}{V - n b_m} \frac{\partial n b_m}{\partial n_i} - \frac{n \alpha_m}{R T b_m} \frac{\partial n b_m}{\partial n_i} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \\
&& \; + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z \\
&= & \; -\ln \frac{V - n b_m}{V} + \frac{\partial n b_m}{\partial n_i} \left( \frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) \\
&& \; + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z
\end{alignat}$$

+++

Полученное выражение позволяет определить логарифм коэффициента летучести $i$-го компонента.

+++

<a id='pvt-eos-srk_pr-mix_rules'></a>
Пусть расчет параметров $\alpha_m$ и $b_m$ соответствует используемым [ранее](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-mix_rules) правилам смешивания, то есть:

+++

$$ \begin{align} \alpha_m &= \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} x_j x_k \alpha_{jk}; \\ b_m &= \sum_{j=1}^{N_c} x_j b_j. \end{align}$$

+++

Здесь параметр $\alpha_{jk}$ рассчитывается следующим образом:

+++

$$\alpha_{jk} = \left( \alpha_j \alpha_k \right)^{0.5} \left( 1 - \delta_{jk} \right).$$

+++

В свою очередь параметр $\alpha_j$:

+++

$$ \alpha_j = a_j \left( 1 + \kappa_j \left( 1 - \sqrt{{T_r}_j} \right) \right)^2.$$

+++

Параметр $\kappa$ является функцией ацентрического фактора компонента, то есть $\kappa = \kappa \left( \omega \right)$, а параметр $T_r$ рассчитывается как отношение температуры к критической температуре компонента. Методы расчета коэффициентов попарного взаимодействия $\delta_{jk}$ более детально обсуждаются в [приложении B](./EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip).

+++

Задав правила смешивания, определим оставшиеся производные в выражении логарифма коэффициента летучести $i$-го компонента:

+++

$$ \begin{align*} \frac{\partial n b_m}{\partial n_i} &= b_i; \\ \frac{\partial n^2 \alpha_m}{\partial n_i} &= 2 n \sum_{j=1}^{N_c} \alpha_{ij} x_j. \end{align*}$$

+++

<a id='pvt-eos-srk_pr-fugacity_coeff-tv'></a>
С учетом этого преобразуем выражение для логарифма коэффициента летучести $i$-го компонента:

+++

$$ \begin{align} \ln \phi_i = &-\ln \frac{V - n b_m}{V} + b_i \left( \frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) \\ &+ \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{b_m} - \frac{\alpha_m b_i}{b_m^2} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z. \end{align} $$

+++

Аналогично может быть получено выражение для логарифма коэффициента летучести системы, состоящей из одного компонента:

+++

$$ \begin{align} \ln \phi = &-\ln \frac{V - n b}{V} + b \left( \frac{n}{V - n b} - \frac{n \alpha}{R T b} \frac{V}{\left( V + n b \delta_1 \right) \left( V + n b \delta_2 \right)} \right) + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \frac{\alpha}{b} \ln \left( \frac{V + n b \delta_1}{V + n b \delta_2} \right) \\ &- \ln Z. \end{align} $$

+++

Данные выражения могут быть использованы для определения логарифма коэффициента летучести компонента систем, состоящих из нескольких компонентов и одного компонента соответственно, при известных объеме и температуре. На практике зачастую известными параметрами системы являются давление и температура. Следовательно, необходимо преобразовать полученные выражения относительно давления и температуры.

+++

Для начала обратим внимание на то, что:

+++

$$ \left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right) = V^2 + n b_m \left( c + 1 \right) V - c b_m^2 n^2 = \frac{\alpha_m n^2}{\frac{n R T}{V - n b_m} - P}. $$

+++

<a id='pvt-eos-srk_pr-Z_PT'></a>
Уравнения состояния Пенга-Робинсона и Суаве-Редлиха-Квонга относительно коэффициента сверхсжимаемости можно записать в следующем виде:

+++

$$ Z^3 - \left( 1 - c B \right) Z^2 + \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right) Z - \left( A B - c \left( B^2 + B^3 \right) \right) = 0. $$

+++

Здесь параметры $A$ и $B$:

+++

$$ \begin{align} A &= \frac{\alpha_m P}{R^2 T^2}; \\ B &= \frac{b_m P}{R T}. \end{align}$$

+++

Теперь с учетом коэффициента сверхсжимаемости $Z = \frac{P v}{R T}$ и параметров $A$ и $B$ преобразуем выражение для логарифма коэффициента летучести $i$-го компонента. Сгруппируем первое и последнее слагаемые в этом выражении:

+++

$$ - \ln \frac{V - n b_m}{V} - \ln Z = - \ln \frac{v - b_m}{v} - \ln Z = - \ln \frac{Z - B}{Z} - \ln Z = -\ln \left( Z - B \right). $$

+++

Предпоследнее слагаемое можно преобразовать следующим образом:

+++

$$ \begin{align}
& \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{b_m} - \frac{\alpha_m b_i}{b_m^2} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \frac{\alpha_m}{b_m} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right) \\
&= \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right).
\end{align} $$

+++

Второе слагаемое можно упростить следующим образом:

+++

$$ \begin{align}
\frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)}
&= \frac{1}{v - b_m} - \frac{\alpha_m}{R T b_m} \frac{v}{v^2 + \left( c + 1 \right) b_m v - c b_m} \\
&= \frac{1}{v - b_m} - \frac{v}{R T b_m} \left( \frac{R T}{v - b_m} - P \right) \\
&= \frac{1}{b_m} \left( Z - 1 \right).
\end{align} $$

+++

<a id='pvt-eos-srk_pr-fugacity_coeff-pt'></a>
С учетом приведенных выше преобразований выражение для логарифма коэффициента летучести компонента смеси:

+++

$$ \ln \phi_i = -\ln \left( Z - B \right) + \frac{b_i}{b_m} \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right). $$

+++

Для чистого компонента:

+++

$$ \ln \phi = -\ln \left( Z - B \right) + \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right). $$

+++

Далее необходимо определить выражения для параметров $a$ и $b$ для чистых компонентов. Для этого рассмотрим критическую точку чистого компонента. [Ранее](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium-critical_point) было показано, что для критической точки характерно равенство нулю первой и второй частной производной давления по объему при постоянной температуре. Исходя из данных условий для уравнения состояния Ван-дер-Ваальса [были найдены](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-coefficients) значения параметров $a$ и $b$ для чистых компонентов, выраженные через давление и температуру в критической точке чистого компонента. Однако для уравнений состояния Пенга-Робинсона и Суаве-Редлиха-Квонга данный подход достаточно трудоемок (хотя абсолютно возможен). Вместо этого рассмотрим уравнение состояния, записанные относительно коэффициента сверхсжимаемости:

+++

$$ Z^3 - \left( 1 - c B \right) Z^2 + \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right) Z^2 - \left( A B - c \left( B^2 + B^3 \right) \right) = 0. $$

+++

Данное уравнение является кубическим, то есть имеет максимум три действительных корня. Более подробно о том, почему в принципе для заданных давления и температуры могут быть получены различные значения объема и коэффициента сверсжимаемости, будет рассматриваться в [главе "Равновесие. Стабильность. Критичность"](../4-ESC/ESC-0-Introduction.html#pvt-esc). Также будет показано, что в критической точке данное уравнение имеет три действительных равных друг другу корня. То есть:

+++

$$ \left( Z - Z_c \right)^3 = 0. $$

+++

Раскрыв скобки в данном выражении, получим:

+++

$$ Z^3 - 3 Z_c Z^2 + 3 Z_c^2 Z - Z_c^3 = 0.$$

+++

Тогда, поскольку данное уравнение и уравнение состояния характеризуют одно и то же состояние системы, можно записать следующие соотношения:

+++

$$ \begin{align} -3 Z_c &= - \left( 1 - c B \right); \\ 3 Z_c^2 &= \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right); \\ - Z_c^3 &= - \left( A B - c \left( B^2 + B^3 \right) \right). \end{align} $$

+++

Полученные уравнения можно решить как систему уравнений относительно параметров $Z_c$, $A$ и $B$. Выразим из первого уравнения $Z_c$, а из второго – $A$ и подставим в третье. Тогда полученная система уравнений будет выглядеть следующим образом:

+++

$$ \begin{cases} Z_c = \frac{1 - c B}{3}; \\ A = \left( c + 1 \right) B + \left( 2 c + 1\right) B^2 + \frac{\left(1 - c B\right)^2}{3}; \\ \left(c^3 + 9 c^2 + 27 c + 27 \right) B^3 + \left( -3 c^2 - 18 c + 27 \right) B^2 + \left( 3 c + 9 \right) B -1 = 0. \end{cases}$$

+++

Данное кубическое уравнение относительно $B$ может быть решено с использованием [формулы Кардано](https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula). Для уравнения состояния Суаве-Редлиха-Квонга, определяемого $c = 0$:


+++

```{code-cell} ipython3
import numpy as np
import sys
sys.path.append('../../SupportCode/')
from PVT import core
```

+++

```{code-cell} ipython3
c = 0
k1 = (-3 * c**2 - 18 * c + 27) / (c**3 + 9 * c**2 + 27 * c + 27)
k2 = (3 * c + 9) / (c**3 + 9 * c**2 + 27 * c + 27)
k3 = -1 / (c**3 + 9 * c**2 + 27 * c + 27)
B_srk = core.calc_cardano(k1, k2, k3)[0]
B_srk
```

+++

Для уравнения состояния Пенга-Робинсона, определяемого $c = 1$:

+++

```{code-cell} ipython3
c = 1
k1 = (-3 * c**2 - 18 * c + 27) / (c**3 + 9 * c**2 + 27 * c + 27)
k2 = (3 * c + 9) / (c**3 + 9 * c**2 + 27 * c + 27)
k3 = -1 / (c**3 + 9 * c**2 + 27 * c + 27)
B_pr = core.calc_cardano(k1, k2, k3)[0]
B_pr
```

+++

В критической точке параметр $B$ определяется следующим выражением:

+++

$$ B = \frac{b P_c}{R T_c}.$$

+++

Тогда для чистого компонента значение параметра $b$ можно выразить:

+++

$$ b = \Omega_b \frac{R T_c}{P_c}. $$

+++

Здесь коэффициент $\Omega_b = 0.08664$ для уравнения состояния Суаве-Редлиха-Квонга и $\Omega_b = 0.07780$ для уравнения состояния Пенга-Робинсона. Получим значения для параметра $A$. Для уравнения состояния Суаве-Редлиха-Квонга:

+++

```{code-cell} ipython3
c = 0
A_srk = (c + 1) * B_srk + (2 * c + 1) * B_srk**2 + (1 - c * B_srk)**2 / 3
A_srk
```

+++

Для уравнения состояния Пенга-Робинсона:

+++

```{code-cell} ipython3
c = 1
A_pr = (c + 1) * B_pr + (2 * c + 1) * B_pr**2 + (1 - c * B_pr)**2 / 3
A_pr
```

+++

В критической точке параметр $A$ определяется следующим выражением:

+++

$$A = \frac{\alpha P_c}{R^2 T_c^2}.$$

+++

Параметр $\alpha$ в критической точке при $ T = T_c$:

+++

$$\alpha = a \left( 1 + \kappa \left( 1 - \sqrt{\frac{T}{T_c}} \right) \right)^2 = a.$$

+++

С учетом этого, выражение для параметра $a$:

+++

$$a = \Omega_a \frac{R^2 T_c^2}{P_c}. $$

+++

Здесь коэффициент для уравнения состояния Суаве-Редлиха-Квонга $\Omega_a = 0.42748$, для уравнения состояния Пенга-Робинсона $\Omega_a = 0.45724.$

+++

<a id='pvt-eos-srk_pr-kappa'></a>
Как было отмечено ранее, коэффициент $\kappa$ рассчитывается по корреляции от ацентрического фактора. В своей [работе](https://doi.org/10.1021/i160057a011) Пенг и Робинсон в 1976 году предложили использовать следующую корреляцию:

+++

$$\kappa = 0.37464 + 1.54226 \omega - 0.26992 \omega^2.$$

+++

Данная корреляция с удовлетворительной точностью воспроизводит значения $\kappa$ для компонентов из углеводородного ряда до декана, а также для диоксида углерода, азота и сероводорода. Позднее, в 1978 году, Пенг и Робинсон для более тяжедых компонентов предложили следующую корреляцию:

+++

$$ \kappa = 0.379642 + 1.48503 \omega - 0.164423 \omega^2 + 0.016666 \omega^3.$$

+++

Для уравнения состояния Суаве-Редлиха-Квонга в соответствии со [статьей](https://doi.org/10.1016/0009-2509(72)80096-4) Суаве рекомендуется использование следующей корреляции:

+++

$$\kappa = 0.480 + 1.574 \omega - 0.176 \omega^2.$$

+++

В 1978 году для уравнения состояния Суаве-Редлиха-Квонга [было предложено](https://doi.org/10.1021/i260068a010) использовать следующую зависимость:

+++

$$ \kappa = 0.48508 + 1.55171 \omega - 0.15613 \omega^2.$$

+++

Решение кубического уравнения состояния относительно коэффициента сверхсжимаемости может привести к возникновению нескольких действительных корней. Как было показано [ранее](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-root_selection), выбор нужного корня осуществляется на основе сравнения величин энергий Гиббса для каждого корня. Пусть $Z_1$ и $Z_2$ – корни кубического уравнения состояния относительно коэффициента сверсхжимаемости. Тогда:

+++

$$ G_1 - G_2 = n R T \left( \sum_{i=1}^{N_c} x_i \ln {f_1}_i - \sum_{i=1}^{N_c} x_i \ln {f_2}_i \right).$$

+++

Логарифм летучести компонента может быть выражен через логарифм коэффициента летучести компонента:

+++

$$ \ln f_i = \ln \phi_i + \ln x_i P. $$

+++

С учетом выражения для логарифма коэффициента летучести $i$-го компонента, полученного при использовании уравнений состояния Пенга-Робинсона или Суаве-Редлиха-Квонга, получим для $ \sum_{i=1}^{N_c} x_i \ln \phi_i$:

+++

$$ \begin{alignat}{1}
\sum_{i=1}^{N_c} x_i \ln \phi_i
&= & \; - \sum_{i=1}^{N_c} x_i \ln \left( Z - B \right) + \sum_{i=1}^{N_c} x_i \frac{b_i}{b_m} \left( Z - 1 \right) \\
&& \; + \sum_{i=1}^{N_c} x_i \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right) \\
&= & \; - \ln \left( Z - B \right) + \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right).
\end{alignat} $$

+++

Тогда:

+++

$$ G_1 - G_2 = - \ln \left( \frac{Z_1 - B}{Z_2 - B} \right) + \left( Z_1 - Z_2 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \ln \left( \frac{Z_1 + B \delta_1}{Z_1 + B \delta_2} \frac{Z_2 + B \delta_2}{Z_2 + B \delta_1} \right).$$

+++

Следовательно, если $G_1 - G_2 > 0$, то коэффициент сверхсжимаемости равен $Z_2$.

+++

Теперь рассмотрим применение уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона. Для этого определим летучести компонентов рассмотренной [ранее](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-exercise) системы.

+++

```{code-cell} ipython3
Pc = np.array([7.37646, 4.600155]) * 10**6
Tc = np.array([304.2, 190.6])
w = np.array([0.225, 0.008])
z = np.array([0.15, 0.85])
dij = np.array([[0, 0.025], [0.025, 0]])
R = 8.314
```

+++

```{code-cell} ipython3
P = 20 * 10**5
T = 40 + 273.15
```

+++

```{code-cell} ipython3
:tags: [hide-input]

class mix_rules_pr_srk(core):
    def __init__(self, Pc, Tc, w, n=1, dij=None, eos='PR'):
        self.select_eos(eos, w)
        self.Pc = Pc
        self.Tc = Tc
        self.n = n
        self.dij = dij
        self.ai = self.omegaa * (R * Tc)**2 / Pc
        self.bi = self.omegab * R * Tc / Pc
        pass

    def select_eos(self, eos, w):
        if eos == 'PR':
            self.c = 1
            self.omegaa = 0.45724
            self.omegab = 0.07780
            self.delta1 = 1 - 2**0.5
            self.delta2 = 1 + 2**0.5
            self.kappa = np.where(w <= 0.491, 0.37464 + 1.54226 * w - 0.26992 * w**2, 0.379642 + 1.48503 * w - 0.164423 * w**2 + 0.016666 * w**3)
        elif eos == 'SRK':
            self.c = 0
            self.omegaa = 0.42748
            self.omegab = 0.08664
            self.delta1 = 0
            self.delta2 = 1
            self.kappa = 0.48508 + 1.55171 * w - 0.15613 * w**2
#             self.kappa = 0.48 + 1.574 * w - 0.176 * w**2
        pass

    def calc_mix_params(self, T, z, calc_der=False):
        self.alphai = self.ai * (1 + self.kappa * (1 - (T / self.Tc)**0.5))**2
        self.alphaij = np.outer(self.alphai, self.alphai)**0.5
        if self.dij is not None:
            self.alphaij = self.alphaij * (1 - self.dij)
        self.alpham = np.sum(np.outer(z, z) * self.alphaij)
        self.bm = np.sum(z * self.bi)
        if calc_der:
            self.dalphamdn = 2 * (np.sum(self.alphaij * self.repeat(self.z, 0), 1) - self.alpham) / self.n
            self.damdn = (self.bi - self.bm) / self.n
        return self
```

+++

```{code-cell} ipython3
:tags: [hide-input]

class eos_pr_srk(core):
    def __init__(self, mr, z, T, P=None, v=None):
        self.mr = mr.calc_mix_params(T, z)
        self.z = z
        self.T = T
        if v is not None:
            self.v = v
            self.Z = self.calc_Z_V()
            self.P = self.Z * R * T / v
            self.lnphi = self.calc_fug_coef_V()
        elif P is not None:
            self.P = P
            self.A = mr.alpham * P / (R**2 * T**2)
            self.B = mr.bm * P / (R * T)
            self.Z = self.calc_Z_P()
            self.v = self.Z * R * T / P
            self.lnphi = self.calc_fug_coef_P()
        self.lnf = self.lnphi + np.log(z * self.P)
        pass

    def calc_Z_V(self):
        return self.v / (self.v - self.mr.bm) - self.mr.alpham * self.v / (R * self.T * \
               (self.v**2 + (self.mr.c + 1) * self.mr.bm * self.v - self.mr.c * self.mr.bm**2))

    def calc_Z_P(self):
        Zs = self.calc_cardano(-(1 - self.mr.c * self.B), self.A - (self.mr.c + 1) * self.B - (2 * self.mr.c + 1) * self.B**2,
                               -(self.A * self.B - self.mr.c * (self.B**2 + self.B**3)))
        Z = Zs[0]
        if len(Zs) > 1:
            for i in range(1, 3):
                if self.calc_dG(Z, Zs[i]) > 0:
                    Z = Zs[i]
        return Z

    def calc_fug_coef_V(self):
        return -np.log((self.v - self.mr.bm) / self.v) + self.mr.bi * (1 / (self.v - self.mr.bm) - \
               self.mr.alpham * self.v / (R * self.T * self.mr.bm * (self.v + self.mr.bm * self.mr.delta1) * \
               (self.v + self.mr.bm * self.mr.delta2))) + (2 * np.sum(self.mr.alphaij * self.repeat(self.z, 0), 1) / \
               self.mr.bm - self.mr.alpham * self.mr.bi / self.mr.bm**2) * np.log((self.v + self.mr.bm * self.mr.delta1) / \
               (self.v + self.mr.bm * self.mr.delta2)) / (R * self.T * (self.mr.delta2 - self.mr.delta1)) - np.log(self.Z)

    def calc_fug_coef_P(self):
        return -np.log(self.Z - self.B) + self.mr.bi * (self.Z - 1) / self.mr.bm + \
               self.A * (2 * np.sum(self.mr.alphaij * self.repeat(self.z, 0), 1) / self.mr.alpham - self.mr.bi / self.mr.bm) * \
               np.log((self.Z + self.B * self.mr.delta1) / (self.Z + self.B * self.mr.delta2)) / \
               (self.B * (self.mr.delta2 - self.mr.delta1))

    def calc_dG(self, Z1, Z2):
        return - np.log((Z1 - self.B) / (Z2 - self.B)) + (Z1 - Z2) + \
               self.A * np.log(((Z1 + self.B * self.mr.delta1) * (Z2 + self.B * self.mr.delta2)) / \
                               ((Z1 + self.B * self.mr.delta2) * (Z2 + self.B * self.mr.delta1))) / \
               (self.B * (self.mr.delta2 - self.mr.delta1))
```

+++

Коэффициент сверхсжимаемости с использованием уравнения состояния Пенга-Робинсона:

+++

```{code-cell} ipython3
mr = mix_rules_pr_srk(Pc, Tc, w, dij=dij)
```

+++

```{code-cell} ipython3
eos = eos_pr_srk(mr, z, T=T, P=P)
eos.Z
```

+++

```{code-cell} ipython3
eosv = eos_pr_srk(mr, z, T=T, v=eos.v)
eosv.Z
```

+++

Логарифм летучести компонентов с использованием уравнения состояния Пенга-Робинсона:

+++

```{code-cell} ipython3
eos.lnphi
```

+++

```{code-cell} ipython3
eosv.lnf
```

+++

Коэффициент сверхсжимаемости с использованием уравнения состояния Суаве-Редлиха-Квонга:

+++

```{code-cell} ipython3
mr = mix_rules_pr_srk(Pc, Tc, w, dij=dij, eos='SRK')
```

+++

```{code-cell} ipython3
eos = eos_pr_srk(mr, z, T=T, P=P)
eos.Z
```

+++

```{code-cell} ipython3
eosv = eos_pr_srk(mr, z, T=T, v=eos.v)
eosv.Z
```

+++

Логарифм летучести компонентов с использованием уравнения состояния Суаве-Редлиха-Квонга:

+++

```{code-cell} ipython3
eos.lnf
```

+++

```{code-cell} ipython3
eosv.lnf
```

+++
