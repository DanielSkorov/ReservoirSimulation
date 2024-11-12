---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 3.0.1
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(pvt-eos-srkpr)=
# Уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона

## Двухпараметрические уравнение состояния

В 1949 году Редлих и Квонг предложили модификацию [уравнения состояния Ван-дер-Ваальса](EOS-1-VanDerWaals.md#pvt-eos-vdw):

$$ \left( P + \frac{a_m}{T^{0.5} v \left( v + b_m \right)} \right) \left(v - b_m \right) = R T.$$

Обозначения и единицы измерения аналогичны [уравнению состояния Ван-дер-Ваальса](EOS-1-VanDerWaals.md#pvt-eos-vdw). Позднее, в 1972 году Суаве улучшил данное уравнение состояния, заменив отношение $\frac{a_m}{T^{0.5}}$ более общим коэффициентом $\alpha_m$, зависящим от температуры и свойств компонента, в том числе *ацентрического фактора*.

```{admonition} Определение
:class: tip
***Ацентрический фактор*** – параметр компонента (вещества), который характеризует отклонение формы молекулы компонента от сферической молекулы идеального газа.
```

В общем виде уравнение состояния Суаве-Редлиха-Квонга можно записать в следующем виде:

$$ \left( P + \frac{\alpha_m}{v \left( v + b_m \right)} \right) \left(v - b_m \right) = R T. $$

Позднее, в 1976 году, Пенг и Робинсон внесли модификацию в данное уравнение состояния для лучшего воспроизведения плотности жидкой фазы:

$$ \left( P + \frac{\alpha_m}{v \left( v + b_m \right) + b_m \left( v - b_m \right)} \right) \left(v - b_m \right) = R T. $$

В общем виде, оба уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона можно записать в следующем виде:

$$ \left( P + \frac{\alpha_m}{v^2 + v b_m \left(1 + c \right) - c b_m^2} \right) \left(v - b_m \right) = R T. $$

Если параметр $c = 0$, то уравнение приводится к уравнению состояния Суаве-Редлиха-Квонга, если же $c = 1$, то – к уравнению состояния Пенга-Робинсона. Записанные в данном формате уравнения состояния принято называть *двухпараметрическими*.

<a id='pvt-eos-srkpr-fugacitycoefficient'></a>
Начнем с нахождения логарифма коэффициента летучести $i$-го компонента. Для этого преобразуем полученное [ранее](../1-TD/TD-15-Fugacity.md#pvt-td-fugacity-PT) выражение следующим образом:

$$ \ln \phi_i = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} - \frac{1}{V} \right) dV - \ln Z = \frac{\partial}{\partial n_i} \left( \int_V^\infty \left( \frac{P}{RT} - \frac{n}{V} \right) dV \right)_{V, T, n_{j \neq i}} - \ln Z. $$

Запишем рассматриваемые уравнения состояния относительно давления:

$$ P = \frac{RT}{v - b_m} - \frac{\alpha_m}{v^2 + \left( c + 1 \right) b_m v - c b_m^2} = \frac{n R T}{V - n b_m} - \frac{\alpha_m n^2}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2}. $$

Получим выражение для первообразной подынтегральной функции:

$$ \begin{align}
F \left( V \right)
&= \int \left( \frac{P}{RT} - \frac{n}{V} \right) dV \\
&= \frac{1}{RT} \int P dV - n \int \frac{dV}{V} \\
&= n \int \frac{dV}{V - n b_m} - \frac{n^2 \alpha_m}{RT} \int \frac{dV}{V^2 + \left( c + 1 \right) b_m n V - c b_m^2 n^2} - n \int \frac{dV}{V}.
\end{align} $$

Для преобразования данного выражения распишем подробнее интегралы:

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

Здесь $\delta_1 = \frac{c + 1}{2} - \sqrt{ \frac{\left( c+1 \right)^2}{4} + c }$ и $\delta_2 = \frac{c + 1}{2} + \sqrt{ \frac{\left( c+1 \right)^2}{4} + c }.$ С учетом этого выражение для первообразной подынтегральной функции примет следующий вид:

$$ F \left( V \right) = n \ln \frac{V - n b_m}{V} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2}. $$

Следовательно, интеграл в выражении для коэффициента летучести $i$-го компонента:

$$ \begin{alignat}{3}
\int_V^\infty \left( \frac{P}{RT} - \frac{n}{V} \right) dV
&= && \; F \left( V \right) \bigg\rvert_V^\infty \\
&= && \; \lim_{b \rightarrow \infty} F \left( V \right) \bigg\rvert_V^b \\
&= && \; \lim_{b \rightarrow \infty} \left( n \ln \frac{b - n b_m}{b} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{b + n b_m \delta_1}{b + n b_m \delta_2} \right) \\
& && \; - \left( n \ln \frac{V - n b_m}{V} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= && \; n \lim_{b \rightarrow \infty} \ln \left( 1 - \frac{n b_m}{b} \right) - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \lim_{b \rightarrow \infty} \ln \left( 1 - \frac{n b_m \left( \delta_2 - \delta_1 \right)}{b + n b_m \delta_2} \right) \\
& && \; - \left( n \ln \frac{V - n b_m}{V} - \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= && \; - n \ln \frac{V - n b_m}{V} + \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2}
\end{alignat} $$

Для выражения логарифма коэффициента летучести $i$-го компонента необходимо получить производную по количеству вещества $i$-го компонента при постоянных объеме, температуре и количествах веществ других компонентов:

$$ \begin{align} & \frac{\partial}{\partial n_i} \left( - n \ln \frac{V - n b_m}{V} + \frac{n \alpha_m}{R T b_m \left( \delta_2 - \delta_1 \right)} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right)_{V, T, n_{j \neq i}} \\ & = - \frac{\partial}{\partial n_i} \left( n \ln \frac{V - n b_m}{V} \right) + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \frac{\partial}{\partial n_i} \left( \frac{n \alpha_m}{b_m} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right). \end{align} $$

Распишем подробнее данные производные.

$$ \begin{align}
\frac{\partial}{\partial n_i} \left( n \ln \frac{V - n b_m}{V} \right)
&= \ln \frac{V - n b_m}{V} + n \frac{\partial}{\partial n_i} \ln \frac{V - n b_m}{V} \\
&= \ln \frac{V - n b_m}{V} - \frac{n}{V - n b_m} \frac{\partial n b_m}{\partial n_i}.
\end{align} $$

$$ \begin{alignat}{3}
\frac{\partial}{\partial n_i} \left( \frac{n \alpha_m}{b_m} \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right)
&= && \; \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \frac{\partial}{\partial n_i} \left( \frac{n \alpha_m}{b_m} \right) + \frac{n \alpha_m}{b_m} \frac{\partial}{\partial n_i} \left( \ln \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= && \; \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \\
&&& \; + \frac{n \alpha_m}{b_m} \frac{\partial n b_m}{\partial n_i} \frac{V \left( \delta_1 - \delta_2 \right)}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)}.
\end{alignat} $$

С учетом этого выражение для логарифма коэффициента летучести $i$-го компонента:

$$ \begin{alignat}{3}
\ln \phi_i
&= && \; -\ln \frac{V - n b_m}{V} + \frac{n}{V - n b_m} \frac{\partial n b_m}{\partial n_i} \\
&&& \; + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \right. \\
&&& \; \left. - \frac{n \alpha_m}{b_m} \frac{\partial n b_m}{\partial n_i} \frac{V \left( \delta_2 - \delta_1 \right)}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) - \ln Z \\
&= && \; -\ln \frac{V - n b_m}{V}  + \frac{n}{V - n b_m} \frac{\partial n b_m}{\partial n_i} - \frac{n \alpha_m}{R T b_m} \frac{\partial n b_m}{\partial n_i} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \\
&&& \; + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z \\
&= && \; -\ln \frac{V - n b_m}{V} + \frac{\partial n b_m}{\partial n_i} \left( \frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) \\
&&& \; + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{1}{n b_m} \frac{\partial n^2 \alpha_m}{\partial n_i} - \frac{\alpha_m}{b_m^2} \frac{\partial n b_m}{\partial n_i} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z
\end{alignat} $$

Полученное выражение позволяет определить логарифм коэффициента летучести $i$-го компонента.

<a id='pvt-eos-srkpr-mixrules'></a>
Пусть расчет параметров $\alpha_m$ и $b_m$ соответствует используемым [ранее](./EOS-1-VanDerWaals.md#pvt-eos-vdw-mixrules) правилам смешивания, то есть:

$$ \begin{align} \alpha_m &= \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} x_j x_k \alpha_{jk}; \\ b_m &= \sum_{j=1}^{N_c} x_j b_j. \end{align} $$

Здесь параметр $\alpha_{jk}$ рассчитывается следующим образом:

$$ \alpha_{jk} = \left( \alpha_j \alpha_k \right)^{0.5} \left( 1 - \delta_{jk} \right). $$

В свою очередь параметр $\alpha_j$:

$$ \alpha_j = a_j \left( 1 + \kappa_j \left( 1 - \sqrt{{T_r}_j} \right) \right)^2. $$

Параметр $\kappa$ является функцией ацентрического фактора компонента, то есть $\kappa = \kappa \left( \omega \right)$, а параметр $T_r$ рассчитывается как отношение температуры к критической температуре компонента. Методы расчета коэффициентов попарного взаимодействия $\delta_{jk}$ более детально обсуждаются в [приложении B](./EOS-Appendix-B-BIP.md#pvt-eos-appendix-bip).

Задав правила смешивания, определим оставшиеся производные в выражении логарифма коэффициента летучести $i$-го компонента:

$$ \begin{align} \frac{\partial n b_m}{\partial n_i} &= b_i; \\ \frac{\partial n^2 \alpha_m}{\partial n_i} &= 2 n \sum_{j=1}^{N_c} \alpha_{ij} x_j. \end{align} $$

<a id='pvt-eos-srkpr-fugacitycoeff-tv'></a>
С учетом этого преобразуем выражение для логарифма коэффициента летучести $i$-го компонента:

$$ \begin{align} \ln \phi_i =
&-\ln \frac{V - n b_m}{V} + b_i \left( \frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) \\
&+ \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{b_m} - \frac{\alpha_m b_i}{b_m^2} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z. \end{align} $$

Аналогично может быть получено выражение для логарифма коэффициента летучести системы, состоящей из одного компонента:

$$ \begin{align} \ln \phi =
&-\ln \frac{V - n b}{V} + b \left( \frac{n}{V - n b} - \frac{n \alpha}{R T b} \frac{V}{\left( V + n b \delta_1 \right) \left( V + n b \delta_2 \right)} \right) + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \frac{\alpha}{b} \ln \left( \frac{V + n b \delta_1}{V + n b \delta_2} \right) \\
&- \ln Z. \end{align} $$

Данные выражения могут быть использованы для определения логарифма коэффициента летучести компонента систем, состоящих из нескольких компонентов и одного компонента соответственно, при известных объеме и температуре. На практике зачастую известными параметрами системы являются давление и температура. Следовательно, необходимо преобразовать полученные выражения относительно давления и температуры.

Для начала обратим внимание на:

$$ \left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right) = V^2 + n b_m \left( c + 1 \right) V - c b_m^2 n^2 = \frac{\alpha_m n^2}{\frac{n R T}{V - n b_m} - P}. $$

<a id='pvt-eos-srkpr-ZPT'></a>
Уравнения состояния Пенга-Робинсона и Суаве-Редлиха-Квонга относительно коэффициента сверхсжимаемости можно записать в следующем виде:

$$ Z^3 - \left( 1 - c B \right) Z^2 + \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right) Z - \left( A B - c \left( B^2 + B^3 \right) \right) = 0. $$

Здесь параметры $A$ и $B$:

$$ \begin{align} A &= \frac{\alpha_m P}{R^2 T^2}; \\ B &= \frac{b_m P}{R T}. \end{align} $$

Теперь с учетом коэффициента сверхсжимаемости $Z = \frac{P v}{R T}$ и параметров $A$ и $B$ преобразуем выражение для логарифма коэффициента летучести $i$-го компонента. Сгруппируем первое и последнее слагаемые в этом выражении:

$$ - \ln \frac{V - n b_m}{V} - \ln Z = - \ln \frac{v - b_m}{v} - \ln Z = - \ln \frac{Z - B}{Z} - \ln Z = -\ln \left( Z - B \right). $$

Предпоследнее слагаемое можно преобразовать следующим образом:

$$ \begin{align}
& \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{b_m} - \frac{\alpha_m b_i}{b_m^2} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) \\
&= \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \frac{\alpha_m}{b_m} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right) \\
&= \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right).
\end{align} $$

Второе слагаемое можно упростить следующим образом:

$$ \begin{align}
\frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)}
&= \frac{1}{v - b_m} - \frac{\alpha_m}{R T b_m} \frac{v}{v^2 + \left( c + 1 \right) b_m v - c b_m} \\
&= \frac{1}{v - b_m} - \frac{v}{R T b_m} \left( \frac{R T}{v - b_m} - P \right) \\
&= \frac{1}{b_m} \left( Z - 1 \right).
\end{align} $$

<a id='pvt-eos-srkpr-fugacitycoeff-pt'></a>
С учетом приведенных выше преобразований выражение для логарифма коэффициента летучести компонента смеси:

$$ \ln \phi_i = -\ln \left( Z - B \right) + \frac{b_i}{b_m} \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right). $$

Для чистого компонента:

$$ \ln \phi = -\ln \left( Z - B \right) + \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right). $$

Далее необходимо определить выражения для параметров $a$ и $b$ для чистых компонентов. Для этого рассмотрим критическую точку чистого компонента. [Ранее](../1-TD/TD-14-PhaseEquilibrium.md#pvt-td-phaseequilibrium-criticalpoint) было показано, что для критической точки характерно равенство нулю первой и второй частной производной давления по объему при постоянной температуре. Исходя из данных условий для уравнения состояния Ван-дер-Ваальса [были найдены](./EOS-1-VanDerWaals.md#pvt-eos-vdw-coefficients) значения параметров $a$ и $b$ для чистых компонентов, выраженные через давление и температуру в критической точке чистого компонента. Однако для уравнений состояния Пенга-Робинсона и Суаве-Редлиха-Квонга данный подход достаточно трудоемок (хотя абсолютно возможен). Вместо этого рассмотрим уравнение состояния, записанные относительно коэффициента сверхсжимаемости:

$$ Z^3 - \left( 1 - c B \right) Z^2 + \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right) Z - \left( A B - c \left( B^2 + B^3 \right) \right) = 0. $$

Данное уравнение является кубическим, то есть имеет максимум три действительных корня. Более подробно о том, почему в принципе для заданных давления и температуры могут быть получены различные значения объема и коэффициента сверсжимаемости, будет рассматриваться в [главе "Равновесие. Стабильность. Критичность"](../4-ESC/ESC-0-Introduction.md#pvt-esc). Также будет показано, что в критической точке данное уравнение имеет три действительных равных друг другу корня. То есть:

$$ \left( Z - Z_c \right)^3 = 0. $$

Раскрыв скобки в данном выражении, получим:

$$ Z^3 - 3 Z_c Z^2 + 3 Z_c^2 Z - Z_c^3 = 0.$$

Тогда, поскольку данное уравнение и уравнение состояния характеризуют одно и то же состояние системы, можно записать следующие соотношения:

$$ \begin{align} -3 Z_c &= - \left( 1 - c B \right); \\ 3 Z_c^2 &= \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right); \\ - Z_c^3 &= - \left( A B - c \left( B^2 + B^3 \right) \right). \end{align} $$

Полученные уравнения можно решить как систему уравнений относительно параметров $Z_c$, $A$ и $B$. Выразим из первого уравнения $Z_c$, а из второго – $A$ и подставим в третье. Тогда полученная система уравнений будет выглядеть следующим образом:

$$ \begin{cases} Z_c = \frac{1 - c B}{3}; \\ A = \left( c + 1 \right) B + \left( 2 c + 1\right) B^2 + \frac{\left(1 - c B\right)^2}{3}; \\ \left(c^3 + 9 c^2 + 27 c + 27 \right) B^3 + \left( -3 c^2 - 18 c + 27 \right) B^2 + \left( 3 c + 9 \right) B -1 = 0. \end{cases} $$

Данное кубическое уравнение относительно $B$ может быть решено с использованием [формулы Кардано](https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula):

```{code-cell} python
import numpy as np

def cardano(b: float, c: float, d: float) -> float:
    p = (3. * c - b * b) / 3.
    q = (2. * b * b * b - 9. * b * c + 27. * d) / 27.
    s = q * q / 4. + p * p * p / 27.
    if s >= 0.:
        s_ = np.sqrt(s)
        u1 = np.cbrt(-q / 2. + s_)
        u2 = np.cbrt(-q / 2. - s_)
        return u1 + u2 - b / 3.
```

Для уравнения состояния Суаве-Редлиха-Квонга, определяемого $c = 0$, коэффициент $B$:

```{code-cell} python
c = 0
k1 = (-3 * c**2 - 18 * c + 27) / (c**3 + 9 * c**2 + 27 * c + 27)
k2 = (3 * c + 9) / (c**3 + 9 * c**2 + 27 * c + 27)
k3 = -1 / (c**3 + 9 * c**2 + 27 * c + 27)
B_srk = cardano(k1, k2, k3)
B_srk
```

Для уравнения состояния Пенга-Робинсона, определяемого $c = 1$, коэффициент $B$:

```{code-cell} python
c = 1
k1 = (-3 * c**2 - 18 * c + 27) / (c**3 + 9 * c**2 + 27 * c + 27)
k2 = (3 * c + 9) / (c**3 + 9 * c**2 + 27 * c + 27)
k3 = -1 / (c**3 + 9 * c**2 + 27 * c + 27)
B_pr = cardano(k1, k2, k3)
B_pr
```

В критической точке параметр $B$ определяется следующим выражением:

$$ B = \frac{b P_c}{R T_c}. $$

<a id='pvt-eos-srkpr-b'></a>
Тогда для чистого компонента значение параметра $b$ можно выразить:

$$ b = \Omega_b \frac{R T_c}{P_c}. $$

Здесь коэффициент $\Omega_b = 0.08664$ для уравнения состояния Суаве-Редлиха-Квонга и $\Omega_b = 0.07780$ для уравнения состояния Пенга-Робинсона. Получим значения для параметра $A$. Для уравнения состояния Суаве-Редлиха-Квонга:

```{code-cell} python
c = 0
A_srk = (c + 1) * B_srk + (2 * c + 1) * B_srk**2 + (1 - c * B_srk)**2 / 3
A_srk
```

Для уравнения состояния Пенга-Робинсона:

```{code-cell} python
c = 1
A_pr = (c + 1) * B_pr + (2 * c + 1) * B_pr**2 + (1 - c * B_pr)**2 / 3
A_pr
```

В критической точке параметр $A$ определяется следующим выражением:

$$ A = \frac{\alpha P_c}{R^2 T_c^2}. $$

Параметр $\alpha$ в критической точке при $ T = T_c$:

$$ \alpha = a \left( 1 + \kappa \left( 1 - \sqrt{\frac{T}{T_c}} \right) \right)^2 = a. $$

<a id='pvt-eos-srkpr-a'></a>
С учетом этого, выражение для параметра $a$:

$$ a = \Omega_a \frac{R^2 T_c^2}{P_c}. $$

Здесь коэффициент для уравнения состояния Суаве-Редлиха-Квонга $\Omega_a = 0.42748$, для уравнения состояния Пенга-Робинсона $\Omega_a = 0.45724.$

<a id='pvt-eos-srkpr-kappa'></a>
Как было отмечено ранее, коэффициент $\kappa$ рассчитывается по корреляции от ацентрического фактора. В своей [работе](https://doi.org/10.1021/i160057a011) Пенг и Робинсон в 1976 году предложили использовать следующую корреляцию:

$$ \kappa = 0.37464 + 1.54226 \omega - 0.26992 \omega^2. $$

Данная корреляция с удовлетворительной точностью воспроизводит значения $\kappa$ для компонентов из углеводородного ряда до декана, а также для диоксида углерода, азота и сероводорода. Позднее, в 1978 году, Пенг и Робинсон для более тяжедых компонентов (с ацентрическим фактором больше 0.491) предложили следующую корреляцию:

$$ \kappa = 0.379642 + 1.48503 \omega - 0.164423 \omega^2 + 0.016666 \omega^3. $$

Для уравнения состояния Суаве-Редлиха-Квонга в соответствии со [статьей](https://doi.org/10.1016/0009-2509(72)80096-4) Суаве рекомендуется использование следующей корреляции:

$$ \kappa = 0.480 + 1.574 \omega - 0.176 \omega^2. $$

В 1978 году для уравнения состояния Суаве-Редлиха-Квонга [было предложено](https://doi.org/10.1021/i260068a010) использовать следующую зависимость:

$$ \kappa = 0.48508 + 1.55171 \omega - 0.15613 \omega^2. $$

Решение кубического уравнения состояния относительно коэффициента сверхсжимаемости может привести к возникновению нескольких действительных корней. Как было показано [ранее](./EOS-1-VanDerWaals.md#pvt-eos-vdw-rootselection), выбор нужного корня осуществляется на основе сравнения величин энергий Гиббса для каждого корня. Пусть $Z_1$ и $Z_2$ – корни кубического уравнения состояния относительно коэффициента сверсхжимаемости. Тогда:

$$ \bar{G_1} - \bar{G_2} = R T \left( \sum_{i=1}^{N_c} x_i \ln {\phi_1}_i - \sum_{i=1}^{N_c} x_i \ln {\phi_2}_i \right). $$

С учетом выражения для логарифма коэффициента летучести $i$-го компонента, полученного при использовании уравнений состояния Пенга-Робинсона или Суаве-Редлиха-Квонга, получим для $\sum_{i=1}^{N_c} x_i \ln \phi_i$:

$$ \begin{alignat}{3}
\sum_{i=1}^{N_c} x_i \ln \phi_i
&= && \; - \sum_{i=1}^{N_c} x_i \ln \left( Z - B \right) + \sum_{i=1}^{N_c} x_i \frac{b_i}{b_m} \left( Z - 1 \right) \\
&&& \; + \sum_{i=1}^{N_c} x_i \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right) \\
&= && \; - \ln \left( Z - B \right) + \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right).
\end{alignat} $$

Тогда:

$$ \bar{G_1} - \bar{G_2} = - \ln \left( \frac{Z_1 - B}{Z_2 - B} \right) + \left( Z_1 - Z_2 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \ln \left( \frac{Z_1 + B \delta_1}{Z_1 + B \delta_2} \frac{Z_2 + B \delta_2}{Z_2 + B \delta_1} \right). $$

Следовательно, если $\bar{G_1} - \bar{G_2} > 0$, то коэффициент сверхсжимаемости равен $Z_2$.

В качестве примера рассмотрим применение уравнения состояния Пенга-Робинсона для рассмотренной [ранее](./EOS-1-VanDerWaals.md#pvt-eos-vdw-exercise) задачи.

```{admonition} Пример
:class: exercise
Необходимо определить летучести компонентов в смеси диоксида углерода и метана с мольными долями $0.15$ и $0.85$ соответственно, находящейся при давлении $2 \; МПа$ и температуре $40 \; ^{\circ} C$.
```

````{dropdown} Решение
``` python
import numpy as np
P = 2e6 # Pressure [Pa]
T = 40. + 273.15 # Temperature [K]
yi = np.array([.15, .85]) # Mole fractions [fr.]
R = 8.3144598 # Universal gas constant [J/mol/K]
```

В дополнении к критическим давлениям ${P_c}_i$, температурам ${T_c}_i$ и коэффициентам попарного взаимодействия компонентов $\delta_{ij}$ необходимо задать их ацентрические факторы $\omega_i$:

``` python
Pci = np.array([7.37646, 4.600155]) * 1e6 # Critical pressures [Pa]
Tci = np.array([304.2, 190.6]) # Critical temperatures [K]
wi = np.array([.225, .008]) # Acentric factors
dij = np.array([[0., .025], [.025, 0.]]) # Binary interaction parameters
```

Поскольку в данном примере используется уравнением состояния Пенга-Робинсона, то коэффициенты $\Omega_A$, $\Omega_B$, $\delta_1$ и $\delta_2$:

``` python
OmegaA = 0.4572355289213823
OmegaB = 0.07779607390388851
d1 = -0.41421356
d2 = 2.41421356
```

Рассчитаем коэффициенты $a_i$ и $b_i$ для каждого компонента:

``` python
ai = OmegaA * R**2 * Tci**2 / Pci
bi = OmegaB * R * Tci / Pci
```

Теперь необходимо учесть температурную поправку в коэффициенте межмолекулярного взаимодействия каждого компонента $\alpha_i$. Для этого получим значения коэффициентов $\kappa_i$, воспользовавшись [корреляцией](#pvt-eos-srkpr-kappa) от ацентрического фактора:

``` python
kappai = 0.37464 + 1.54226 * wi - 0.26992 * wi**2
```

Рассчитаем значения $\alpha_i$:

``` python
alphai = ai * (1. + kappai * (1. - np.sqrt(T / Tci)))**2
```

Затем определим параметр $\alpha_{ij}$:

``` python
alphaij = np.outer(np.sqrt(alphai), np.sqrt(alphai)) * (1. - dij)
```

Теперь рассчитаем энергетический и объемный коэффициенты $\alpha_m$ и $b_m$ для рассматриваемой смеси:

``` python
alpham = np.sum(alphaij * np.outer(yi, yi))
bm = yi.dot(bi)
```

Найдем коэффициенты $A$ и $B$, необходимые для решения кубического уравнения:

``` python
A = alpham * P / R**2 / T**2
B = bm * P / R / T
```

Для решения уравнения состояния относительно коэффициента сверхсжимаемости будем использовать метод [`np.roots`](https://numpy.org/doc/stable/reference/generated/numpy.roots.html):

``` python
Zs = np.roots(np.array([1., B - 1., A - 2. * B - 3. * B**2, -A * B + B**2 * (1. + B)]))
print(Zs)
```

```{glue:} glued_Zs
```

В результате решения получен один действительный корень и два мнимых. Действительный корень является коэффициентом сверхсжимаемости рассматриваемой системы.

``` python
Z = Zs[0].real
```

В общем случае выбор корней уравнения основывается на сопоставлении энергий Гиббса в соответствии с изложенным выше выражением. Определив коэффициент сверхсжимаемости, рассчитаем логарфим коэффициентов летучести компонентов:

``` python
gZ = np.log(Z - B)
gphii = (2. * alphaij.dot(yi) - alpham * bi / bm) / (R * T * bm)
fZ = np.log((Z + B * d1) / (Z + B * d2))
lnphii = -gZ + bi * (Z - 1.) / bm + gphii * fZ / (d2 - d1)
```

Тогда логарифм летучести компонентов:

``` python
lnfi = lnphii + np.log(yi * P)
print(lnfi)
```

```{glue:} glued_lnfi
```

````

```{code-cell} python
:tags: [remove-cell]

import numpy as np
P = 2e6 # Pressure [Pa]
T = 40. + 273.15 # Temperature [K]
yi = np.array([.15, .85]) # Mole fractions [fr.]
R = 8.3144598 # Universal gas constant [J/mol/K]

Pci = np.array([7.37646, 4.600155]) * 1e6 # Critical pressures [Pa]
Tci = np.array([304.2, 190.6]) # Critical temperatures [K]
wi = np.array([.225, .008]) # Acentric factors
dij = np.array([[0., .025], [.025, 0.]]) # Binary interaction parameters

OmegaA = 0.4572355289213823
OmegaB = 0.07779607390388851
d1 = -0.41421356
d2 = 2.41421356

ai = OmegaA * R**2 * Tci**2 / Pci
bi = OmegaB * R * Tci / Pci

kappai = 0.37464 + 1.54226 * wi - 0.26992 * wi**2

alphai = ai * (1. + kappai * (1. - np.sqrt(T / Tci)))**2

alphaij = np.outer(np.sqrt(alphai), np.sqrt(alphai)) * (1. - dij)

alpham = np.sum(alphaij * np.outer(yi, yi))
bm = yi.dot(bi)

A = alpham * P / R**2 / T**2
B = bm * P / R / T

Zs = np.roots(np.array([1., B - 1., A - 2. * B - 3. * B**2, -A * B + B**2 * (1. + B)]))

Z = Zs[0].real

gZ = np.log(Z - B)
gphii = (2. * alphaij.dot(yi) - alpham * bi / bm) / (R * T * bm)
fZ = np.log((Z + B * d1) / (Z + B * d2))
lnphii = -gZ + bi * (Z - 1.) / bm + gphii * fZ / (d2 - d1)

lnfi = lnphii + np.log(yi * P)

from myst_nb import glue
glue('glued_Zs', Zs)
glue('glued_lnfi', lnfi)
```

## Трехпараметрические уравнения состояния

В предыдущем разделе были подробно изложены особенности расчета коэффициента сверхсжимаемости смеси компонентов и их летучестей с использованием *двухпараметрических* уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона. Однако следует отметить, что зачастую для построения PVT-моделей реальных пластовых систем используют *трехпараметрические* уравнения состояния. Необходимость введения дополнительных коэффициентов обуславливается отклонением расчетных и фактических плотностей газовой и жидкой фаз. В 1982 году в работе \[[Peneloux et al, 1982](https://doi.org/10.1016/0378-3812(82)80002-2)\] было предложено ввести дополнительный ***параметр объемного сдвига*** *(volume shift parameter)* для коррекции рассчитанного по двухпараметрическому уравнению состояния значения объема смеси:

$$ V = V_{2p} - \sum_{i=1}^{N_c} c_i n_i. $$

Здесь $V$ – объем рассматриваемой системы для трехпараметрического уравнения состояния $\left[м^3 \right]$, $V_{2p}$ – молярный объем рассматриваемой системы для двухпараметрического уравнения состояния $\left[м^3 \right]$, $c_i$ – параметр объемного сдвига $i$-го компонента $\left[м^3 / моль \right]$, $n_i$ – количество вещества $i$-го компонента в рассматриваемой системе $\left[ моль \right]$. Данное выражение может быть записано относительно молярных объемов:

$$ v = v_{2p} - \sum_{i=1}^{N_c} c_i x_i. $$

В данном выражении $v$ – молярный объем рассматриваемой системы для трехпараметрического уравнения состояния $\left[м^3 / моль \right]$, $v_{2p}$ – молярный объем рассматриваемой системы для двухпараметрического уравнения состояния $\left[м^3 / моль \right]$, $x_i$ – мольная доля $i$-го компонента в рассматриваемой системе $\left[ д.ед. \right]$. При этом, зачастую параметр объемного сдвига выражают через *безразмерный* коэффициент объемного сдвига $s_i$ следующим образом:

````{margin}
```{admonition} Дополнительно
:class: note
Для моделей реальных смесей значения безразмерного коэффициента объемного вдига $s_i$ зачастую меньше нуля для компонентов, легче гексана, и больше нуля для компонентов, тяжелее гексана, при использовании уравнения состояния Пенга-Робинсона. Таким образом, на практике введение объемного сдвига позволяет увеличить молярный объем газовой фазы, преимущественно состоящей из легких компонентов, то есть сделать ее еще более "газовой", и соответственно уменьшить молярный объем жидкой фазы, преимущественно состоящей из тяжелых компонентов, то есть сделать ее еще более "жидкой".
```
````

$$ c_i = b_i s_i \Rightarrow v = v_{2p} - \sum_{i=1}^{N_c} b_i s_i x_i, $$

где $b_i$ – объемный коэффициент в соответствующем уравнении состояния для $i$-го компонента, определяемый полученным [ранее](#pvt-eos-srkpr-b) выражением.

Каким образом введение данного коэффициента влияет на полученные ранее выражения? С учетом параметра объмного коэффициента сдвига выражение для коэффициента сверхсжимаемости получается из представленного выше выражения молярного объема:

$$ Z = Z_{2p} - \frac{P \sum_{i=1}^{N_c} b_i s_i x_i}{RT}. $$

Здесь $Z$ – коэффициент сверхсжимаемости рассматриваемой системы для трехпараметрического уравнения состояния, $Z_{2p}$ – коэффициент сверхсжимаемости рассматриваемой системы для двухпараметрического уравнения состояния, определенный путем решения кубического уравнения состояния, представленного выше.

Для того чтобы определить, как изменилось выражение для расчета коэффициента летучести $i$-го компонента при добавлении объемного коэффициента сдвига, рассмотрим полученное [ранее](../1-TD/TD-15-Fugacity.md#pvt-td-fugacity-VT) выражение:

$$ \begin{align}
\ln \phi_i &= \int_0^P \left( \frac{1}{RT} \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{1}{P} \right) dP \\
&= \int_0^P \left( \frac{1}{RT} \frac{\partial}{\partial n_i} \left( V_{2p} - \sum_{j=1}^{N_c} c_j n_j \right)_{P, T, n_{j \neq i}} - \frac{1}{P} \right) dP \\
&= \int_0^P \left( \frac{1}{RT} \left( \frac{\partial V_{2p}}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{c_i}{RT} - \frac{1}{P} \right) dP \\
&= \int_0^P \left( \frac{1}{RT} \left( \frac{\partial V_{2p}}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{1}{P} \right) dP - \frac{c_i P}{RT} \\
&= \ln {\phi_i}_{2p} - \frac{b_i s_i P}{RT} . \\
\end{align} $$

Необходимо также отметить, что введение параметров объемного сдвига не оказывает влияния на равновесное состояние, [определяемое](../1-TD/TD-15-Fugacity.md#pvt-td-fugacity-equilibrium) равенством летучестей компонентов в соответствующих фазах. То есть в равновесном состоянии при условии $\ln f_i^1 = \ln f_i^2$ также будет выполняться соотношение $\ln {f_i^1}_{2p} = \ln {f_i^2}_{2p}$. Докажем данное утверждение.

```{admonition} Доказательство
:class: proof
Рассмотрим две фазы $1$ и $2$, в которых компоненты характеризуются следующими летучестями:

$$ \begin{align}
\ln f_i^1 &= \ln \phi_i^1 + \ln y_i^1 P = \ln {\phi_i^1}_{2p} - \frac{b_i s_i P}{RT} + \ln y_i^1 P = \ln {f_i^1}_{2p} - \frac{b_i s_i P}{RT}, \\
\ln f_i^2 &= \ln \phi_i^2 + \ln y_i^2 P = \ln {\phi_i^2}_{2p} - \frac{b_i s_i P}{RT} + \ln y_i^2 P = \ln {f_i^2}_{2p} - \frac{b_i s_i P}{RT}.
\end{align} $$

Равновесное состояние характеризуется равенством летучестей компонентов для соответствующих фаз, следовательно:

$$ \begin{align}
\ln f_i^1 &= ln f_i^2 ,\\
\ln {f_i^1}_{2p} - \frac{b_i s_i P}{RT} &= \ln {f_i^2}_{2p} - \frac{b_i s_i P}{RT}, \\
\ln {f_i^1}_{2p} &= \ln {f_i^2}_{2p} .
\end{align} $$

```

Таким образом учет параметров объемного сдвига не приводит к изменению равновесного состояния, в том числе не оказывает влияния на фазовые диаграммы.

Каким образом могут быть определены значения параметров объемного сдвига? Зачастую они подбираются путем адаптации на результаты лабораторных исследований, в частности с помощью них настраивается изменение плотностей и объемов фаз в процессе лабораторных исследований. Однако для адаптации модели необходимы начальные приближения. Они могут быть получены, например, используя корреляции, представленные в \[[Pedersen et al, 2004](https://doi.org/10.2118/88364-PA)\]. Авторы работы предложили использовать линейную зависимость от температуры для расчета параметров объемного сдвига широкого ряда углеводородных компонентов, а также диоксида углерода и азота. Значения безразмерных коэффициетов объемного сдвига для хорошо изученных компонентов (метан – гексан) и уравнения состояния Пенга-Робинсона также приводятся в работе \[[Jhaveri et al, 1988](https://doi.org/10.2118/13118-PA)\], кроме того в этой работе приводятся корреляция для расчета безразмерных коэффициентов объемного сдвига по зависимости от молярной массы. Значения параметров объемного сдвига также могут быть получены с использованием корреляции, представленной в работе \[[Peneloux et al, 1982](https://doi.org/10.1016/0378-3812(82)80002-2)\], относительно критических давлений и температур компонентов, а также коэффициента сжимаемости Ракетта $Z_{Ra}$ – константы, используемой в [уравнении состояния Ракетта](https://doi.org/10.1021/je60047a012) и изученной для широкого ряда компонентов:

$$ c = 0.40768 \frac{R T_c}{P_c} \left( 0.29441 - Z_{Ra} \right). $$

Величина $Z_{Ra}$, в свою очередь, может быть приблизительно принята (согласно \[[Spencer et al, 1973](https://doi.org/10.1021/je60057a007)\]) равной коэффициенту сверхсжимаемости компонента в критической точке, либо определена по корреляции относительно ацентрического фактора:

$$ Z_{Ra} = 0.29056 - 0.08775 \omega. $$

Кроме того, авторами работы \[[Kokal et al, 1990](https://doi.org/10.2118/90-05-07)\] был предложен алгоритм расчета параметров объемного сдвига, основанный на сопоставлении молярных объемов, рассчитанных с использованием корреляции, например, через $Z_{Ra}$ (или коэффициент сверхсжимаемости компонента в критической точке), а также с использованием настраемого уравнения состояния.

Пример реализации уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона для расчета коэффициента сверхсжимаемости и летучестей компонентов представлен [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/eos.py).

Таким образом, рассмотренные в данном разделе уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона характеризуются достаточно широким применением при рассмотрении вопросов, связанных с поведением углеводородных систем. Однако данные уравнения состония достаточно плохо подходят для описания поведения флюидов, молекулы которых имеют тенденцию к образованию ассоциаций (например, воды, метанола и других полярных молекул) – кластеров или аггрегатов, состоящих из нескольких молекул, сформированных в результате электростатических взаимодействий, недостаточных для образования химических связей.
