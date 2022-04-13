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

<a id='pvt-eos-appendix-fugacity_pd'></a>
# Частные производные летучести компонентов

+++

````{div} full-width
Целью данного приложения является получение частных производных логарифма коэффициента летучести (и логарифма летучести) по давлению, объему, температуре и количеству вещества компонента с использованием уравнений состояния. Некоторые из представленных частных производных уже были рассмотрены в предыдущих разделах. Рассмотренные производные будут полезны при изучении последующих задач.
````

+++

````{div} full-width
Рассмотрим нахождение частных производных логарифма коэффициента летучести (и логарифма летучести) по давлению, объему, температуре и количеству вещества компонента компонента с использованием [уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr).
````

+++

<a id='pvt-eos-appendix-fugacity_pd-srk_pr-p'></a>
````{div} full-width
Логарифм коэффициента летучести $i$-го компонента с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона, [записанный](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-fugacity_coeff-pt) относительно давления $P$, температуры $T$ и коэффициента сверхсжимаемости $Z$:

+++

$$ \ln \phi_i = -\ln \left( Z - B \right) + \frac{b_i}{b_m} \left( Z - 1 \right) + \frac{1}{\left( \delta_2 - \delta_1 \right)} \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right)  \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right) $$

+++

Для упрощения работы с данным выражением запишем его в следующем виде:

+++

$$ \ln \phi_i = -g_Z + \frac{b_i}{b_m} \left( Z - 1 \right) + \frac{1}{\delta_2 - \delta_1} g_{\phi_i} f_Z,$$

+++

где:

+++

$$ \begin{align} g_Z &= \ln \left( Z - B \right); \\ g_{\phi_i} &= \frac{A}{B} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{\alpha_m} - \frac{b_i}{b_m} \right) = \frac{1}{R T b_m} \left( 2 \sum_{j=1}^{N_c} \alpha_{ij} x_j - \alpha_m \frac{b_i}{b_m} \right); \\ f_Z &= \ln \left( \frac{Z + B \delta_1}{Z + B \delta_2} \right). \end{align}$$

+++

Здесь и далее принимается, что параметры $\alpha_m$ и $b_m$ рассчитываются с использованием [правил смешивания Ван-дер-Ваальса](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-mix_rules).

````

+++

## Частные производные летучести компонентов с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона (в PT-формулировке)

+++

````{div} full-width

Тогда частные производные логарифма коэффициента летучести $i$-го компонента по температуре, давлению и количеству вещества $k$-го компонента:

+++

$$\begin{align}
\left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P, n_i} &= - \left( \frac{\partial g_Z}{\partial T} \right)_{P, n_i} + \frac{\partial}{\partial T} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P, n_i} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial T} \right)_{P, n_i}; \\
\left( \frac{\partial \ln \phi_i}{\partial P} \right)_{T, n_i} &= - \left( \frac{\partial g_Z}{\partial P} \right)_{T, n_i} + \frac{\partial}{\partial P} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{T, n_i} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial P} \right)_{T, n_i}; \\
\left( \frac{\partial \ln \phi_i}{\partial n_k} \right)_{P, T} &= - \left( \frac{\partial g_Z}{\partial n_k} \right)_{P, T} + \frac{\partial}{\partial n_k} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P, T} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial n_k} \right)_{P, T}.
\end{align}$$

+++

Необходимо отметить, что частные производные логарфима коэффициента летучести $i$-го компонента по количеству вещества $i$-го компонента равняются частным производным логарфима коэффициента летучести $i$-го компонента по количеству вещества $k$-го компонента, находящимся на главной диагонали.

+++

Сначала рассмотрим второе слагаемое каждого выражения:

+++

$$ \begin{align}
\frac{\partial}{\partial T} \left( \left( Z-1 \right) \frac{b_i}{b_m} \right)_{P,n_i}
&= \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} + \left( Z - 1 \right) \frac{\partial}{\partial T} \left( \frac{b_i}{b_m} \right)_{P,n_i} \\
&= \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i}; \\
\frac{\partial}{\partial P} \left( \left( Z-1 \right) \frac{b_i}{b_m} \right)_{T,n_i}
&= \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} + \left( Z - 1 \right) \frac{\partial}{\partial P} \left( \frac{b_i}{b_m} \right)_{T,n_i} \\
&= \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i}; \\
\frac{\partial}{\partial n_k} \left( \left( Z-1 \right) \frac{b_i}{b_m} \right)_{P,T}
&= \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} + \left( Z - 1 \right) \frac{\partial}{\partial n_k} \left( \frac{b_i}{b_m} \right)_{P,T} \\
&= \frac{b_i}{b_m} \left(\frac{\partial Z}{\partial n_k}\right)_{P,T} + \left( Z - 1 \right) \frac{b_m \left( \frac{\partial b_i}{\partial n_k} \right)_{P,T} - b_i \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} }{b_m^2} \\
&= \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} - \left( Z - 1 \right) \frac{b_i}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

Для упрощения данных выражений требуется определить частные производные коэффициента сверхсжимаемости по температуре, давлению и количеству вещества $k$-го компонента. Для нахождения частной производной коэффициента сверхсжимаемости запишем частную производную уравнения состояния, выраженного через коэффициент сверхсжимаемости, являющего функцией коэффициента сверхсжимаемости (который, в свою очередь, также является функцией термодинамических параметров), давления, температуры и количества вещества $i$-го компонента $ q = q \left( Z, P, T, n_i \right)$:

+++

$$ \left( \frac{\partial q}{\partial T} \right)_{P, n_i} = \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \left( \frac{\partial Z}{\partial T} \right)_{P, n_i}  + \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i}.$$

+++

При известном коэффициенте сверхсжимаемости:

+++

$$ q \left( Z, P, T, n_i \right) = 0.$$

+++

Тогда левая часть производной уравнения состояния:

+++

$$ \left( \frac{\partial q}{\partial T} \right)_{P, n_i} = 0.$$

+++

Следовательно,

+++

$$ \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} = - \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i}\left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1}.$$

+++

Аналогично можно получить частные производные коэффициента сверхсжимаемости по давлению и количеству вещества $k$-го компонента:

+++

$$\begin{align} \left( \frac{\partial Z}{\partial P} \right)_{T, n_i} &= - \left( \frac{\partial q}{\partial P} \right)_{Z, T, n_i} \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} ; \\ \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} &= - \left( \frac{\partial q}{\partial n_k} \right)_{Z, P, T} \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1}. \end{align}$$

+++

Получим частную производную уравнения состояния по коэффициенту сверхсжимаемости при постоянных давлении, температуре и количеству вещества $i$-го компонента:

+++

$$ \left( \frac{\partial q}{\partial Z} \right)_{P,T,n_i} = \frac{\partial}{\partial Z} \left( Z^3 + d_2 Z^2 + d_1 Z + d_0 \right)_{P,T,n_i} = 3 Z^2 + 2 d_2 Z + d_1 = 3 Z^2 + \sum_{m=1}^2 m d_m Z^{m-1}.$$

+++

В выражении выше $\{d_2, \; d_1, \; d_0\}$ представляют собой коэффициенты перед коэффициентом сверхсжимаемости в уравнении состояния:

+++

$$\begin{align}
d_2 &= - \left( 1 - c B \right); \\
d_1 &= \left( A - \left( c + 1 \right) B - \left( 2 c + 1\right) B^2 \right); \\
d_0 &= - \left( A B - c \left( B^2 + B^3 \right) \right).
\end{align}$$

+++

Частные производные уравнения состояния по давлению, температуре и количеству вещества $k$-ко компонента при постоянном коэффициенте сверхсжимаемости:

+++

$$\begin{align}
\left( \frac{\partial q}{\partial T} \right)_{Z,P,n_i} &= \sum_{m=0}^2 \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^m; \\
\left( \frac{\partial q}{\partial P} \right)_{Z,T,n_i} &= \sum_{m=0}^2 \left( \frac{\partial d_m}{\partial P} \right)_{T,n_i} Z^m; \\
\left( \frac{\partial q}{\partial n_k} \right)_{Z,P,T} &= \sum_{m=0}^2 \left( \frac{\partial d_m}{\partial n_k} \right)_{P,T} Z^m.
\end{align}$$

+++

Поскольку коэффициенты $\{d_2, \; d_1, \; d_0\}$ являются функциями параметров $A$ и $B$, то их частные производные по давлению, температуре и количеству вещества $k$-го компонента:

+++

$$\begin{align}
\left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} &= \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}; \\
\left( \frac{\partial d_m}{\partial P} \right)_{T,n_i} &= \left( \frac{\partial d_m}{\partial A} \right)_{P,T,n_i} \left( \frac{\partial A}{\partial P} \right)_{T,n_i} + \left( \frac{\partial d_m}{\partial B} \right)_{P,T,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}; \\
\left( \frac{\partial d_m}{\partial n_k} \right)_{P,T} &= \left( \frac{\partial d_m}{\partial A} \right)_{n_k,P,T} \left( \frac{\partial A}{\partial n_k} \right)_{P,T} + \left( \frac{\partial d_m}{\partial B} \right)_{n_k,P,T} \left( \frac{\partial B}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

Частные производные коэффициентов $\{d_2, \; d_1, \; d_0\}$ по параметрам $A$ и $B$ при постоянных давлении, температуре и количестве вещества $i$-го компонента:

+++

$$\begin{align}
\left( \frac{\partial d_2}{\partial A} \right)_{T,P,n_i} &= 0; & \left( \frac{\partial d_2}{\partial B} \right)_{T,P,n_i} &= c; \\
\left( \frac{\partial d_1}{\partial A} \right)_{T,P,n_i} &= 1; & \left( \frac{\partial d_1}{\partial B} \right)_{T,P,n_i} &= - \left(c + 1\right) - 2 \left( 2c + 1 \right) B;\\
\left( \frac{\partial d_0}{\partial A} \right)_{T,P,n_i} &= -B; & \left( \frac{\partial d_0}{\partial B} \right)_{T,P,n_i} &= - A + c \left( 2 B + 3 B^2 \right).
\end{align}$$

+++

Получим частные производные параметров $A$ и $B$ по давлению, температуре и количеству вещества $k$-го компонента:

+++

$$\begin{align}
\left( \frac{\partial A}{\partial T} \right)_{P,n_i} &= \frac{P}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} - 2 \frac{A}{T}; & \left( \frac{\partial B}{\partial T} \right)_{P,n_i} &= - \frac{b_m P}{R T^2}; \\
\left( \frac{\partial A}{\partial P} \right)_{T,n_i} &= \frac{\alpha_m}{R^2 T^2}; & \left( \frac{\partial B}{\partial P} \right)_{T,n_i} &= \frac{b_m}{RT};\\
\left( \frac{\partial A}{\partial n_k} \right)_{P,T} &= \frac{P}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T}; & \left( \frac{\partial B}{\partial n_k} \right)_{P,T} &= \frac{P}{RT} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

Частная производная параметра смешивания $\alpha_m$ по температуре:

+++

$$\left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} = \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} x_i x_j \left( \frac{\partial \alpha_{ij} }{\partial T} \right)_{P,n_i}.$$

+++

В свою очередь,

+++

$$\begin{align} \left( \frac{\partial \alpha_{ij} }{\partial T} \right)_{P,n_i} &= \frac{\partial}{\partial T} \left( \left( \alpha_i \alpha_j \right)^{0.5} \left( 1 - \delta_{ij} \right) \right)_{P,n_i} \\ &= \left( 1 - \delta_{ij} \right) \frac{\partial}{\partial T} \left( \left( \alpha_i \alpha_j \right)^{0.5} \right)_{P,n_i} + \left( \alpha_i \alpha_j \right)^{0.5} \frac{\partial}{\partial T} \left( 1 - \delta_{ij} \right)_{P,n_i} \\ &= \left( 1 - \delta_{ij} \right) \frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{2 \sqrt{\alpha_i \alpha_j}} - \left( \alpha_i \alpha_j \right)^{0.5} \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i}. \end{align}$$

+++

Частная производная параметра $\alpha_i$ (и $\alpha_j$ соответственно) по температуре:

+++

$$\begin{align}
\left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i}
&= \frac{\partial}{\partial T} \left( a_i \left( 1 + \kappa_i \left( 1 - \sqrt{\frac{T}{{T_c}_i}} \right) \right)^2 \right)_{P,n_i} \\
&= 2 a_i \left( 1 + \kappa_i \left( 1 - \sqrt{\frac{T}{{T_c}_i}} \right) \right) \frac{\partial}{\partial T} \left( 1 + \kappa_i \left( 1 - \sqrt{\frac{T}{{T_c}_i}} \right) \right)_{P,n_i} \\
&= - a_i \left( 1 + \kappa_i \left( 1 - \sqrt{\frac{T}{{T_c}_i}} \right) \right) \kappa_i \frac{1}{\sqrt{{T_c}_i T}} \\
&= - \sqrt{\frac{a_i \alpha_i}{{T_c}_i T}} \kappa_i.
\end{align}$$

+++

Частная производная параметра $\alpha_i$ для воды (как компонента), используемого в [уравнении состояния Сорейде-Уитсона](./EOS-3-SW.html#pvt-eos-sw), будет рассмотрена [ниже](#pvt-eos-appendix-fugacity_pd-sw). Также необходимо получить частные производные коэффициентов попарного взаимодействия по температуре при использовании [GCM](EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip-gcm) для их расчета:

+++

$$\begin{alignat}{1}
\left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \frac{-\frac{1}{2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 1} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)^2}{2 \frac{\sqrt{\alpha_i}}{b_i} \frac{\sqrt{\alpha_j}}{b_j}} \right)_{P,n_i} \\
&= & \; \frac{b_i b_j}{2} \frac{\partial}{\partial T} \left( \frac{DS - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)^2}{\sqrt{\alpha_i \alpha_j}} \right)_{P,n_i} \\
&= & \; \frac{b_i b_j}{2} \frac{\sqrt{\alpha_i \alpha_j} \frac{\partial}{\partial T} \left( DS - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)^2 \right)_{P,n_i} - \left( DS - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)^2 \right) \frac{\partial}{\partial T} \left( \sqrt{\alpha_i \alpha_j} \right)_{P,n_i}}{\alpha_i \alpha_j} \\
&= & \; \frac{b_i b_j}{2 \alpha_i \alpha_j} \left( \sqrt{\alpha_i \alpha_j} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \right. \\
&& \; \left. - \left( DS - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)^2 \right) \frac{1}{2\sqrt{\alpha_i \alpha_j}} \left(\alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \\
&= & \; \frac{1}{2} \left( \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \right. \\
&& \; \left. - \frac{\delta_{ij}}{\alpha_i \alpha_j} \left(\alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right).
\end{alignat}$$

+++

Частная производная двойной суммы $DS$ по температуре:

+++

$$\begin{align}
\left( \frac{\partial DS}{\partial T} \right)_{P,n_i}
&= \frac{\partial}{\partial T} \left( -\frac{1}{2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 1} \right)_{P,n_i} \\
&= \frac{149.075}{T^2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2}.
\end{align}$$

+++

Частные производные параметров $\alpha_m$ и $b_m$ по количеству вещества $k$-го компонента были рассмотрены [ранее](./EOS-1-VanDerWaals.html#pvt-eos-van_der_waals-am_bm_derivative). Поэтому здесь приведем только конечные выражения:

+++

$$\begin{align} \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} &= \frac{2}{n} \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right); \\ \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} &= \frac{b_k - b_m}{n}. \end{align}$$

+++

Теперь частная производная коэффициента сверхсжимаемости по давлению, температуре и количеству вещества $k$-го компонента определена, следовательно, определены вторые слагаемые в выражении для частной производной логарифма коэффициента летучести $i$-го компонента.

+++

Рассмотрим первое слагаемое:

+++

$$\begin{align}
\left( \frac{\partial g_Z}{\partial T} \right)_{P,n_i} &= \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}; \\
\left( \frac{\partial g_Z}{\partial P} \right)_{T,n_i} &= \left( \frac{\partial g_Z}{\partial Z} \right)_{P,T,n_i} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} + \left( \frac{\partial g_Z}{\partial B} \right)_{P,T,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}; \\
\left( \frac{\partial g_Z}{\partial n_k} \right)_{P,T} &= \left( \frac{\partial g_Z}{\partial Z} \right)_{P,T,n_k} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} + \left( \frac{\partial g_Z}{\partial B} \right)_{P,T,n_k} \left( \frac{\partial B}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

При этом,

+++

$$\begin{align}
\left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} &= \frac{1}{Z - B}; \\
\left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} &= - \frac{1}{Z - B}.
\end{align}$$

+++

Распишем последнее слагаемое в выражении частной производной логарифма коэффициента летучести $i$-го компонента:

+++

$$\begin{align}
\left( \frac{\partial g_{\phi_i} f_Z}{\partial T} \right)_{P,n_i} &= f_Z \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} + g_{\phi_i} \left( \frac{\partial f_Z}{\partial T} \right)_{P,n_i}; \\
\left( \frac{\partial g_{\phi_i} f_Z}{\partial P} \right)_{T,n_i} &= f_Z \left( \frac{\partial g_{\phi_i}}{\partial P} \right)_{T,n_i} + g_{\phi_i} \left( \frac{\partial f_Z}{\partial P} \right)_{T,n_i}; \\
\left( \frac{\partial g_{\phi_i} f_Z}{\partial n_k} \right)_{P,T} &= f_Z \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} + g_{\phi_i} \left( \frac{\partial f_Z}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

Поскольку параметр $g_{\phi_i}$ не зависит от давления, то:

+++

$$\left( \frac{\partial g_{\phi_i}}{\partial P} \right)_{T,n_i} = 0.$$

+++

Получим частную производную $g_{\phi_i}$ по температуре:

+++

$$\begin{align}
\left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i}
&= \frac{\partial}{\partial T} \left( \frac{1}{R T b_m} \left( 2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_m \frac{b_i}{b_m} \right) \right)_{P,n_i} \\
&= \frac{1}{R b_m} \left( \frac{2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_m \frac{b_i}{b_m}}{T} \right)_{P,n_i} \\
&= \frac{1}{R b_m} \frac{T \frac{\partial}{\partial T} \left( 2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_m \frac{b_i}{b_m} \right)_{P,n_i} - \left( 2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_m \frac{b_i}{b_m} \right)}{T^2} \\
&= \frac{1}{R b_m} \frac{T \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)  - g_{\phi_i} R T b_m}{T^2} \\
&= \frac{1}{T} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) - g_{\phi_i} \right)
\end{align}$$

+++

Рассмотрим частную производную параметра $g_{\phi_i}$ по количеству вещества $k$-го компонента:

+++

$$ \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} = \frac{\partial}{\partial n_k} \left( \frac{1}{R T b_m} \left( 2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_m \frac{b_i}{b_m} \right) \right)_{P,T} = \frac{2}{R T} \frac{\partial}{\partial n_k} \left( \frac{1}{b_m} \sum_{j=1}^{N_c} x_j \alpha_{ij} \right)_{P,T} - \frac{1}{R T}\frac{\partial}{\partial n_k} \left( \frac{\alpha_m b_i}{b_m^2} \right)_{P,T}. $$

+++

Первое слагаемое выражения выше можно преобразовать следующим образом:

+++

$$\begin{align}
\frac{\partial}{\partial n_k} \left( \frac{1}{b_m} \sum_{j=1}^{N_c} x_j \alpha_{ij} \right)_{P,T} 
&= \frac{b_m \frac{\partial}{\partial n_k} \left( \sum_{j=1}^{N_c} \frac{n_j}{n} \alpha_{ij} \right)_{P,T} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} }{b_m^2} \\
&= \frac{1}{b_m} \sum_{j=1}^{N_c} \alpha_{ij} \frac{\partial}{\partial n_k} \left( \frac{n_j}{n} \right)_{P,T} - \frac{\sum_{j=1}^{N_c} x_j \alpha_{ij}}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \\
&= \frac{1}{n b_m} \sum_{j=1}^{N_c} \alpha_{ij} \left( E_{jk} - x_j \right) - \frac{\sum_{j=1}^{N_c} x_j \alpha_{ij}}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \\
&= \frac{\alpha_{ik} - \sum_{j=1}^{N_c} x_j a_{ij}}{n b_m} - \frac{\sum_{j=1}^{N_c} x_j \alpha_{ij}}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

Второе слагаемое выражения частной производной $g_{\phi_i}$ по количеству вещества $k$-го компонента можно преобразовать:

+++

$$ \frac{\partial}{\partial n_k} \left( \frac{\alpha_m b_i}{b_m^2} \right)_{P,T} = b_i \frac{b_m^2 \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - 2 \alpha_m b_m \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T}}{b_m^4} = \frac{b_i}{b_m^2} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - 2 \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right). $$

+++

С учетом этого выражение для частной производной $g_{\phi_i}$ по количеству вещества $k$-го компонента:

+++

$$\begin{align}
\left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T}
&= \frac{1}{R T} \left( 2 \frac{\alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij}}{n b_m} - 2 \frac{\sum_{j=1}^{N_c} x_j \alpha_{ij}}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} - \frac{b_i}{b_m^2} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - 2 \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right) \\
&= \frac{1}{R T b_m} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - 2 \frac{\sum_{j=1}^{N_c} x_j \alpha_{ij}}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} + 2 \alpha_m \frac{b_i}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \\
&= \frac{1}{R T b_m} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - \frac{1}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( 2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_m \frac{b_i}{b_m} \right) - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} + \alpha_m \frac{b_i}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \\
&= \frac{1}{R T b_m} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} g_{\phi_i} R T - \frac{b_i}{b_m} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right).
\end{align}$$

+++

Получим частные производные параметра $f_Z$:

+++

$$\begin{align}
\left( \frac{\partial f_Z}{\partial T} \right)_{P,n_i} &= \left( \frac{\partial f_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} + \left( \frac{\partial f_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}; \\
\left( \frac{\partial f_Z}{\partial P} \right)_{T,n_i} &= \left( \frac{\partial f_Z}{\partial Z} \right)_{P,T,n_i} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} + \left( \frac{\partial f_Z}{\partial B} \right)_{P,T,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}; \\
\left( \frac{\partial f_Z}{\partial n_k} \right)_{P,T} &= \left( \frac{\partial f_Z}{\partial Z} \right)_{P,T,n_k} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} + \left( \frac{\partial f_Z}{\partial B} \right)_{P,T,n_k} \left( \frac{\partial B}{\partial n_k} \right)_{P,T}.
\end{align}$$

+++

При этом,

+++

$$\begin{align}
\left( \frac{\partial f_Z}{\partial Z} \right)_{T,P,n_i} &= \frac{1}{Z + B \delta_1} - \frac{1}{Z + B \delta_2}; \\
\left( \frac{\partial f_Z}{\partial B} \right)_{T,P,n_i} &= \frac{\delta_1}{Z + B \delta_1} - \frac{\delta_2}{Z + B \delta_2}.
\end{align}$$

+++

Зная частные производные логарифма коэффициента летучести $i$-го компонента по давлению, температуре и количеству вещества $k$-го компонента, получим выражения частных производных логарифма летучести $i$-го компонента по давлению, температуре и количеству вещества $k$-го компонента.

+++

Распишем частную производную логарифма коэффициента летучести $i$-го компонента по температуре:

+++

$$\left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i} = \frac{1}{\phi_i} \left( \frac{\partial \phi_i}{\partial T} \right)_{P,n_i} = \frac{1}{\phi_i x_i P} \left( \frac{\partial f_i}{\partial T} \right)_{P,n_i} = \frac{1}{f_i} \left( \frac{\partial f_i}{\partial T} \right)_{P,n_i} = \left( \frac{\partial \ln f_i}{\partial T} \right)_{P,n_i}.$$

+++

Аналогично преобразуем частную производную логарифма коэффициента летучести $i$-го компонента по давлению:

+++

$$\left( \frac{\partial \ln \phi_i}{\partial P} \right)_{T,n_i} = \frac{1}{\phi_i} \left( \frac{\partial \phi_i}{\partial P} \right)_{T,n_i} = \frac{1}{\phi_i x_i} \frac{\partial}{\partial T} \left( \frac{f_i}{P} \right)_{P,n_i} = \frac{1}{\phi_i x_i} \left( \frac{1}{P} \left( \frac{\partial f_i}{\partial P} \right)_{T,n_i} - \frac{f_i}{P^2} \right) = \left( \frac{\partial \ln f_i}{\partial P} \right)_{T,n_i} - \frac{1}{P}. $$

+++

Следовательно,

+++

$$\left( \frac{\partial \ln f_i}{\partial P} \right)_{T,n_i} = \left( \frac{\partial \ln \phi_i}{\partial P} \right)_{T,n_i} + \frac{1}{P}.$$

+++

Частная производная логарифма коэффициента летучести $i$-го компонента по количеству вещества $k$-го компонента:

+++

$$ \begin{align}
\left( \frac{\partial \ln \phi_i}{\partial n_k} \right)_{P,T}
& = \frac{1}{\phi_i} \left( \frac{\partial \phi_i}{\partial n_k} \right)_{P,T} \\
& = \frac{1}{\phi_i P} \frac{\partial}{\partial n_k} \left( \frac{f_i}{x_i} \right)_{P,T} \\
& = \frac{1}{\phi_i P} \left( \frac{1}{x_i} \left( \frac{\partial f_i}{\partial n_k} \right)_{P,T} - \frac{f_i}{x_i^2} \frac{\partial}{\partial n_k} \left( \frac{n_i}{n}  \right)_{P,T} \right) \\
& = \frac{x_i}{f_i} \left( \frac{1}{x_i} \left( \frac{\partial f_i}{\partial n_k} \right)_{P,T} - \frac{f_i}{x_i^2} \frac{E_{ik}}{n} + \frac{f_i}{x_i n} \right) \\
& = \left( \frac{\partial \ln f_i}{\partial n_k} \right)_{P,T} - \frac{E_{ik}}{x_i n} + \frac{1}{n}.
\end{align} $$

+++

Тогда:

+++

$$ \left( \frac{\partial \ln f_i}{\partial n_k} \right)_{P,T} = \left( \frac{\partial \ln \phi_i}{\partial n_k} \right)_{P,T} + \frac{E_{ik}}{x_i n} - \frac{1}{n}.$$

+++

При этом, необходимо отметить, что под $n$ понимается в данном случае количество вещества фазы, поскольку летучести компонентов, коэффициент сверхсжимаемости и т.д. определяются с использованием уравнения состояния для фазы.
````

+++

## Вторые частные производные летучести компонентов с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона (в PT-формулировке)

+++

````{div} full-width

Получим также вторые частные производные логарифма коэффициента летучести по давлению, температуре и количеству вещества $l$-го комонента:

+++

$$\begin{align}
\left( \frac{\partial^2 \ln \phi_i}{\partial T^2} \right)_{P, n_i} &= - \left( \frac{\partial^2 g_Z}{\partial T^2} \right)_{P, n_i} + \frac{\partial^2}{\partial T^2} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P, n_i} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial T^2} \right)_{P, n_i}; \\
\left( \frac{\partial^2 \ln \phi_i}{\partial P^2} \right)_{T, n_i} &= - \left( \frac{\partial^2 g_Z}{\partial P^2} \right)_{T, n_i} + \frac{\partial^2}{\partial P^2} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{T, n_i} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial P^2} \right)_{T, n_i}; \\
\left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial n_l} \right)_{P, T} &= - \left( \frac{\partial^2 g_Z}{\partial n_k \partial n_l} \right)_{P, T} + \frac{\partial^2}{\partial n_k \partial n_l} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P, T} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial n_k \partial n_l} \right)_{P, T}; \\
\left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial T} \right)_{P} &= - \left( \frac{\partial^2 g_Z}{\partial n_k \partial T} \right)_{P} + \frac{\partial^2}{\partial n_k \partial T} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial n_k \partial T} \right)_{P}; \\
\left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial P} \right)_{T} &= - \left( \frac{\partial^2 g_Z}{\partial n_k \partial P} \right)_{T} + \frac{\partial^2}{\partial n_k \partial P} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{T} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial n_k \partial P} \right)_{T}; \\
\left( \frac{\partial^2 \ln \phi_i}{\partial P \partial T} \right)_{n_i} &= - \left( \frac{\partial^2 g_Z}{\partial P \partial T} \right)_{n_i} + \frac{\partial^2}{\partial P \partial T} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{n_i} + \frac{1}{\delta_2 - \delta_1} \left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial P \partial T} \right)_{n_i}.
\end{align}$$

+++

Распишем подробнее вторые слагаемые в приведенных выше выражениях:

+++

$$\begin{alignat}{1}
\frac{\partial^2}{\partial T^2} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \frac{\partial}{\partial T} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{P,n_i} \right)_{P,n_i} = \frac{\partial}{\partial T} \left( \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial T}  \right)_{P,n_i} \right)_{P,n_i} = \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial T^2}  \right)_{P,n_i} ; \\
\frac{\partial^2}{\partial P^2} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{T,n_i}
&= & \; \frac{\partial}{\partial P} \left( \frac{\partial}{\partial P} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{T,n_i} \right)_{T,n_i} = \frac{\partial}{\partial P} \left( \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial P}  \right)_{T,n_i} \right)_{T,n_i} =  \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial P^2}  \right)_{T,n_i}; \\
\frac{\partial^2}{\partial n_k \partial n_l} \left( \frac{b_i}{b_m} \left( Z-1 \right) \right)_{P,T}
&= & \; \frac{\partial}{\partial n_l} \left( \frac{\partial}{\partial n_k} \left( \frac{b_i}{b_m} \left( Z-1 \right) \right)_{P,T} \right)_{P,T} \\
&= & \; \frac{\partial}{\partial n_l} \left( \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} - \left( Z-1 \right) \frac{b_i}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right)_{P,T} \\
&= & \; \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial n_k \partial n_l} \right)_{P,T} - \frac{b_i}{b_m^2} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} - \frac{b_i}{b_m^2} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial Z}{\partial n_l} \right)_{P,T} \\
&& \; + \frac{2 \left( Z-1 \right) b_i}{b_m^3} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} - \left( Z-1 \right) \frac{b_i}{b_m^2} \left( \frac{\partial^2 b_m}{\partial n_k \partial n_l} \right)_{P,T} \\
&= & \; \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial n_k \partial n_l} \right)_{P,T} - \frac{b_i}{b_m^2} \left( \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} + \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial Z}{\partial n_l} \right)_{P,T} \right) \\
&& \; + \left( Z-1 \right) \frac{b_i}{b_m^2} \left( \frac{2}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} - \left( \frac{\partial^2 b_m}{\partial n_k \partial n_l} \right)_{P,T} \right) ; \\
\frac{\partial^2}{\partial T \partial n_k} \left( \frac{b_i}{b_m} \left( Z-1 \right) \right)_{P}
&= & \; \frac{\partial}{\partial n_k} \left( \frac{\partial}{\partial T} \left( \frac{b_i}{b_m} \left( Z-1 \right) \right)_{P,n_i} \right)_{P,T} = \frac{\partial}{\partial n_k} \left( \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \right)_{P,T} \\
&= & \; \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial T \partial n_k} \right)_{P} - \frac{b_i}{b_m^2} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T}; \\
\frac{\partial^2}{\partial P \partial n_k} \left( \frac{b_i}{b_m} \left( Z-1 \right) \right)_{T}
&= & \; \frac{\partial}{\partial n_k} \left( \frac{\partial}{\partial P} \left( \frac{b_i}{b_m} \left( Z-1 \right) \right)_{T,n_i} \right)_{P,T} = \frac{\partial}{\partial n_k} \left( \frac{b_i}{b_m} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \right)_{P,T} \\
&= & \; \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial P \partial n_k} \right)_{T} - \frac{b_i}{b_m^2} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T}; \\
\frac{\partial^2}{\partial P \partial T} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{n_i}
&= & \; \frac{\partial}{\partial T} \left( \frac{\partial}{\partial P} \left( \frac{b_i}{b_m} \left( Z - 1 \right) \right)_{T, n_i} \right)_{P, n_i} = \frac{b_i}{b_m} \left( \frac{\partial^2 Z}{\partial P \partial T} \right)_{n_i}.
\end{alignat}$$

+++

Для определения значений данных выражений необходимо знать вторые частные производные коэффициента сверхсжимаемости. Рассмотрим вывод выражения второй частной производной коэффициента сверхсжимаемости по температуре. Для этого запишем вторую частную производную по температуре уравнения состояния $ q = q \left( Z, P, T, n_i \right)$

+++

$$\begin{align}
\left( \frac{\partial^2 q}{\partial T^2} \right)_{P,n_i}
&= \left( \frac{\partial}{\partial T} \left( \frac{\partial q}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \left( \frac{\partial Z}{\partial T} \right)_{P, n_i}  + \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i} \right)_{P,n_i} \\
&= \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{P, n_i} + \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \left( \frac{\partial^2 Z}{\partial T^2} \right)_{P, n_i} + \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i} \right)_{P, n_i}.
\end{align}$$

+++

При известном коэффициенте сверхсжимаемости $q \left( Z, P, T, n_i \right) = 0$, следовательно:

+++

$$\left( \frac{\partial^2 q}{\partial T^2} \right)_{P,n_i} = 0.$$

+++

Тогда:

+++

$$\left( \frac{\partial^2 Z}{\partial T^2} \right)_{P, n_i} = - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{P, n_i} + \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i} \right)_{P, n_i} \right).$$

+++

Преобразуем слагаемые в скобках. Второе слагаемое:

+++


$$\begin{align}
\frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i} \right)_{P, n_i} &= \frac{\partial}{\partial T} \left( \sum_{m=0}^2 \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^m \right)_{P, n_i} \\
&= \sum_{m=0}^2 Z^m \left( \frac{\partial^2 d_m}{\partial T^2} \right)_{P,n_i} + \sum_{m=0}^2 m \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^{m-1} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \\
&= \sum_{m=0}^2 Z^m \left( \frac{\partial^2 d_m}{\partial T^2} \right)_{P,n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^{m-1} \\
&= \left( \frac{\partial^2 q}{\partial T^2} \right)_{Z, P, n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^{m-1}
\end{align}$$ 

+++

Для дальнейшего преобразования рассмотрим следующее выражение: 

+++

$$\frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} = \frac{\partial}{\partial T} \left( 3 Z^2 + \sum_{m=1}^2 m d_m Z^{m-1} \right)_{Z, P, n_i} = \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^{m-1}.$$

+++

Тогда: 

+++

$$ \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i} \right)_{P, n_i} = \left( \frac{\partial^2 q}{\partial T^2} \right)_{Z, P, n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i}. $$ 

+++

Преобразуем первое слагаемое в скобках в выражении второй частной производной коэффициента сверхсжимаемости по температуре:

+++

$$\begin{align}
\frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{P, n_i} &= \frac{\partial}{\partial T} \left( 3 Z^2 + 2 d_2 Z + d_1 \right)_{P, n_i} \\
&= 6Z \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} + 2 Z \left( \frac{\partial d_2}{\partial T} \right)_{P, n_i} + 2 d_2 \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} + \left( \frac{\partial d_1}{\partial T} \right)_{P, n_i} \\
&= \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \left( 6Z + 2 d_2 \right) + \left( 2 Z \left( \frac{\partial d_2}{\partial T} \right)_{P, n_i} + \left( \frac{\partial d_1}{\partial T} \right)_{P, n_i} \right) \\
&= \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} + \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial T} \right)_{P, n_i} Z^{m-1} \\
&= \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} + \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i}.
\end{align}$$ 

+++

Следовательно, вторая частная производная коэффициента сверхсжимаемости по температуре:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 Z}{\partial T^2} \right)_{P, n_i}
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{P, n_i} + \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial T} \right)_{Z, P, n_i} \right)_{P, n_i} \right) \\
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial Z}{\partial T} \right)_{P, n_i}^2 \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P, n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} \right. \\
&& \; \left. + \left( \frac{\partial^2 q}{\partial T^2} \right)_{Z, P, n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} \right) \\
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial^2 q}{\partial T^2} \right)_{Z, P, n_i} + 2 \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P, n_i}^2 \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} \right).
\end{alignat}$$

+++

Аналогично получаются и другие вторые частные производные коэффициента сверхсжимаемости:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 Z}{\partial P^2} \right)_{T, n_i}
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial^2 q}{\partial P^2} \right)_{Z, T, n_i} + 2 \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \frac{\partial}{\partial P} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, T, n_i} + \left( \frac{\partial Z}{\partial P} \right)_{T, n_i}^2 \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} \right); \\
\left( \frac{\partial^2 Z}{\partial n_k \partial n_l} \right)_{P,T}
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial^2 q}{\partial n_k \partial n_l} \right)_{Z, P,T} + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \frac{\partial}{\partial n_l} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P,T} + \left( \frac{\partial Z}{\partial n_l} \right)_{P,T} \frac{\partial}{\partial n_k} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P,T} \right. \\
&& \; \left. + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial Z}{\partial n_l} \right)_{P,T} \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} \right); \\
\left( \frac{\partial^2 Z}{\partial n_k \partial T} \right)_{P}
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial^2 q}{\partial n_k \partial T} \right)_{Z, P} + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial n_k} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, T} \right. \\
&& \; \left. + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} \right); \\
\left( \frac{\partial^2 Z}{\partial n_k \partial P} \right)_{T}
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial^2 q}{\partial n_k \partial P} \right)_{Z, T} + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \frac{\partial}{\partial P} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, T, n_i} + \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \frac{\partial}{\partial n_k} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, T} \right. \\
&& \; \left. + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} \right); \\
\left( \frac{\partial^2 Z}{\partial P \partial T} \right)_{n_i}
&= & \; - \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i}^{-1} \left( \left( \frac{\partial^2 q}{\partial P \partial T} \right)_{Z, n_i} + \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial P} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, T, n_i} \right. \\
&& \; \left. + \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} \right).
\end{alignat}$$

+++

В выражениях выше распишем подробнее слагаемые:

+++

$$\begin{align}
\left( \frac{\partial^2 q}{\partial T^2} \right)_{Z, P, n_i} &= \sum_{m=0}^2 \left( \frac{\partial^2 d_m}{\partial T^2} \right)_{P,n_i} Z^m; & \left( \frac{\partial^2 q}{\partial P^2} \right)_{Z, T, n_i} &= \sum_{m=0}^2 \left( \frac{\partial^2 d_m}{\partial P^2} \right)_{T,n_i} Z^m; \\
\left( \frac{\partial^2 q}{\partial n_k \partial n_l} \right)_{Z, P,T} &= \sum_{m=0}^2 \left( \frac{\partial^2 d_m}{\partial n_k \partial n_l} \right)_{P,T} Z^m; & \left( \frac{\partial^2 q}{\partial n_k \partial T} \right)_{Z, P} &= \sum_{m=0}^2 \left( \frac{\partial^2 d_m}{\partial n_k \partial T} \right)_{P} Z^m; \\
\left( \frac{\partial^2 q}{\partial n_k \partial P} \right)_{Z, T} &= \sum_{m=0}^2 \left( \frac{\partial^2 d_m}{\partial n_k \partial P} \right)_{T} Z^m; & \left( \frac{\partial^2 q}{\partial P \partial T} \right)_{Z, n_i} &= \sum_{m=0}^2 \left( \frac{\partial^2 d_m}{\partial P \partial T} \right)_{n_i} Z^m. \\
\end{align}$$

+++

Кроме того, 

+++

$$\left( \frac{\partial^2 q}{\partial Z^2} \right)_{P,T,n_i} = 6Z + 2 d_2.$$

+++

В свою очередь,

+++

$$\begin{align}
\frac{\partial}{\partial T} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, n_i} &= \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} Z^{m-1}; \\
\frac{\partial}{\partial P} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, T, n_i} &= \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial P} \right)_{P,n_i} Z^{m-1}; \\
\frac{\partial}{\partial n_k} \left( \left( \frac{\partial q}{\partial Z} \right)_{P, T, n_i} \right)_{Z, P, T} &= \sum_{m=1}^2 m \left( \frac{\partial d_m}{\partial n_k} \right)_{P,T} Z^{m-1}.
\end{align}$$

+++

Рассмотрим вторую частную производную коэффициентов $\{d_2, \; d_1, \; d_0\}$ по температуре:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 d_m}{\partial T^2} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial d_m}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \right)_{P,n_i} + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial T^2} \right)_{P,n_i} + \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T^2} \right)_{P,n_i} \\
&= & \; \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \left( \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} + \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \right) + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial T^2} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \left( \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} + \left( \frac{\partial^2 d_m}{\partial B \partial A} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \right) + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T^2} \right)_{P,n_i} \\
&= & \; \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i}^2 + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial T^2} \right)_{P,n_i} + \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}^2 + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T^2} \right)_{P,n_i} \\
&& \; + 2 \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}
\end{alignat}$$

+++

Аналогично получаются остальные вторые частные производные коэффициентов $\{d_2, \; d_1, \; d_0\}$:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 d_m}{\partial P^2} \right)_{T,n_i}
&= & \; \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial P} \right)_{T,n_i}^2 + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial P^2} \right)_{T,n_i} + \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}^2 + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial P^2} \right)_{T,n_i} \\
&& \; + 2 \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}; \\
\left( \frac{\partial^2 d_m}{\partial n_k \partial n_l} \right)_{P,T}
&= & \; \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial n_k} \right)_{P,T} \left( \frac{\partial A}{\partial n_l} \right)_{P,T} + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial n_k \partial n_l} \right)_{P,T} + \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial n_l} \right)_{P,T} \\
&& \; + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial n_k \partial n_l} \right)_{P,T} + \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial A}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial n_l} \right)_{P,T} + \left( \frac{\partial A}{\partial n_l} \right)_{P,T} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \right); \\
\left( \frac{\partial^2 d_m}{\partial n_k \partial T} \right)_{P}
&= & \; \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial n_k} \right)_{P,T} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial n_k \partial T} \right)_{P} + \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial n_k \partial T} \right)_{P} + \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial A}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} + \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \right); \\
\left( \frac{\partial^2 d_m}{\partial n_k \partial P} \right)_{T}
&= & \; \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial n_k} \right)_{P,T} \left( \frac{\partial A}{\partial P} \right)_{T,n_i} + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial n_k \partial P} \right)_{T} + \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \\
&& \; + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial n_k \partial P} \right)_{T} + \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial A}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} + \left( \frac{\partial A}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \right); \\
\left( \frac{\partial^2 d_m}{\partial P \partial T} \right)_{n_i}
&= & \; \left( \frac{\partial^2 d_m}{\partial A^2} \right)_{T,P,n_i} \left( \frac{\partial A}{\partial P} \right)_{T,n_i} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} + \left( \frac{\partial d_m}{\partial A} \right)_{T,P,n_i} \left( \frac{\partial^2 A}{\partial P \partial T} \right)_{n_i} + \left( \frac{\partial^2 d_m}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial d_m}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial P \partial T} \right)_{n_i} + \left( \frac{\partial^2 d_m}{\partial A \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial A}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} + \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \right); \\
\end{alignat}$$

+++

Вторые частные производные коэффициентов $\{d_2, \; d_1, \; d_0\}$ по параметрам $A$ и $B$:

+++

$$\begin{align}
\left( \frac{\partial^2 d_2}{\partial A^2} \right)_{T,P,n_i} &= 0; & \left( \frac{\partial^2 d_1}{\partial A^2} \right)_{T,P,n_i} &= 0; & \left( \frac{\partial^2 d_0}{\partial A^2} \right)_{T,P,n_i} &= 0; \\
\left( \frac{\partial^2 d_2}{\partial B^2} \right)_{T,P,n_i} &= 0; & \left( \frac{\partial^2 d_1}{\partial B^2} \right)_{T,P,n_i} &= -2 \left( 2c+1 \right); & \left( \frac{\partial^2 d_0}{\partial B^2} \right)_{T,P,n_i} &= c \left(2 + 6 B \right); \\
\left( \frac{\partial^2 d_2}{\partial A \partial B} \right)_{T,P,n_i} &= 0; & \left( \frac{\partial^2 d_1}{\partial A \partial B} \right)_{T,P,n_i} &= 0; & \left( \frac{\partial^2 d_0}{\partial A \partial B} \right)_{T,P,n_i} &= -1.
\end{align}$$

+++

Вторые частные производные параметров $A$ и $B$ по давлению, температуре и количеству вещества $k$-го и $l$-го компонентов:

+++

$$\begin{align}
\left( \frac{\partial^2 A}{\partial T^2} \right)_{P,n_i} &= \frac{\partial}{\partial T} \left( \frac{P}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} - 2 \frac{A}{T} \right)_{P,n_i} & \left( \frac{\partial^2 B}{\partial T^2} \right)_{P,n_i} &= 2 \frac{b_m P}{R T^3}; \\
&= \frac{P}{R^2 T^2} \left( \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{P,n_i} - \frac{2}{T} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) - \frac{2}{T} \left( \frac{\partial A}{\partial T} \right)_{P,n_i} + 2 \frac{A}{T^2} & \left( \frac{\partial^2 B}{\partial P^2} \right)_{T,n_i} &= 0; \\
&= \frac{P}{R^2 T^2} \left( \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{P,n_i} - \frac{1}{T} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) - \frac{3}{T} \left( \frac{\partial A}{\partial T} \right)_{P,n_i}; & \left( \frac{\partial^2 B}{\partial n_k \partial n_l} \right)_{P,T} &= \frac{P}{R T} \left( \frac{\partial^2 b_m}{\partial n_k \partial n_l} \right)_{P,T}; \\
\left( \frac{\partial^2 A}{\partial P^2} \right)_{T,n_i} &= 0; & \left( \frac{\partial^2 B}{\partial n_k \partial T} \right)_{P} &= - \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \frac{P}{R T^2}; \\
\left( \frac{\partial^2 A}{\partial n_k \partial n_l} \right)_{P,T} &= \frac{P}{R^2 T^2} \left( \frac{\partial^2 \alpha_m}{\partial n_k \partial n_l} \right)_{P,T}; & \left( \frac{\partial^2 B}{\partial n_k \partial P} \right)_{T} &= \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \frac{1}{R T}; \\
\left( \frac{\partial^2 A}{\partial n_k \partial T} \right)_{P} &= \frac{\partial}{\partial n_k} \left( \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \right)_{P,T} = \frac{\partial}{\partial n_k} \left( \frac{P}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} - 2 \frac{A}{T} \right)_{P,T} & \left( \frac{\partial^2 B}{\partial P \partial T} \right)_{n_i} &= - \frac{b_m}{R T^2}. \\
&= \frac{P}{R^2 T^2} \frac{\partial}{\partial n_k} \left( \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)_{P,T} - \frac{2}{T} \left( \frac{\partial A}{\partial n_k} \right)_{P,T}; & \\
\left( \frac{\partial^2 A}{\partial n_k \partial P} \right)_{T} &= \frac{\partial}{\partial n_k} \left( \left( \frac{\partial A}{\partial P} \right)_{T,n_i} \right)_{P,T} = \frac{1}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T}; & \\
\left( \frac{\partial^2 A}{\partial P \partial T} \right)_{n_i} &= \frac{\partial}{\partial P} \left( \left( \frac{\partial A}{\partial T} \right)_{P,n_i} \right)_{T,n_i} = \frac{\partial}{\partial P} \left( \frac{P}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} - 2 \frac{A}{T} \right)_{T,n_i} & \\
&= \frac{1}{R^2 T^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} - \frac{2}{T} \left( \frac{\partial A}{\partial P} \right)_{T,n_i}. &
\end{align}$$

+++

Рассмотрим вторые частные производные параметров смешивания $\alpha_m$ и $b_m$:

+++

$$\begin{align}
\left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{P,n_i} &= \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} x_i x_j \left( \frac{\partial^2 \alpha_{ij} }{\partial T^2} \right)_{P,n_i}; & \left( \frac{\partial^2 b_m}{\partial T^2} \right)_{P,n_i} &= 0; \\
\left( \frac{\partial^2 \alpha_m}{\partial P^2} \right)_{T,n_i} &= 0; & \left( \frac{\partial^2 b_m}{\partial P^2} \right)_{T,n_i} &= 0; \\
\left( \frac{\partial^2 \alpha_m}{\partial n_k \partial T} \right)_{P} &= \frac{\partial}{\partial T} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} \right)_{P,n_i} & \left( \frac{\partial^2 b_m}{\partial n_k \partial T} \right)_{P} &= 0 ; \\
&= \frac{\partial}{\partial T} \left( \frac{2}{n} \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right)_{P,n_i} = \frac{2}{n} \left( \sum_{i=1}^{N_c} \left( \frac{\partial \alpha_{ik}}{\partial T} \right)_{P,n_i} x_i - \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) ; & \left( \frac{\partial^2 b_m}{\partial n_k \partial P} \right)_{T} &= 0 ; \\
\left( \frac{\partial^2 \alpha_m}{\partial n_k \partial P} \right)_{T} &= 0; & \left( \frac{\partial^2 b_m}{\partial P \partial T} \right)_{n_i} &= 0. \\
\left( \frac{\partial^2 \alpha_m}{\partial P \partial T} \right)_{n_i} &= 0; & \\
\left( \frac{\partial^2 \alpha_m}{\partial n_i \partial T} \right)_{P} &= \frac{\partial}{\partial T} \left( \left( \frac{\partial \alpha_m}{\partial n_i} \right)_{P,T} \right)_{P,n_i} & \\
&= \frac{1}{n} \left( \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} x_j + \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} \left( \frac{\partial \alpha_{ij, \; i=j}}{\partial T} \right)_{P,n_i} x_i - 2 \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right).
\end{align}$$

+++

Отдельно получим вторые частные производные параметров смешивания по количеству вещества $k$-го и $l$-го компонентов:

+++

$$\begin{align}
\left( \frac{\partial^2 \alpha_m}{\partial n_k \partial n_l} \right)_{P,T} &= \frac{\partial}{\partial n_l} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} \right)_{P,T} \\
&= \frac{\partial}{\partial n_l} \left( \frac{2}{n} \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right)_{P,T} \\
&= 2 \frac{n \frac{\partial}{\partial n_l} \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right)_{P,T} - \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \left( \frac{\partial n}{\partial n_l} \right)_{P,T}}{n^2} \\
&= \frac{2}{n^2} \left( n \frac{\partial}{\partial n_l} \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right)_{P,T} - \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right) \\
&= \frac{2}{n^2} \left( n \left( \sum_{i=1}^{N_c} \alpha_{ik} \left( \frac{\partial x_i}{\partial n_l} \right)_{P,T} - \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} \right) - \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right) \\
&= \frac{2}{n^2} \left( n \left( \sum_{i=1}^{N_c} \alpha_{ik} \frac{n \left( \frac{\partial n_i}{\partial n_l} \right)_{P,T} - n_i \left( \frac{\partial n}{\partial n_l} \right)_{P,T}}{n^2} - \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} \right) - \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right) \\
&= \frac{2}{n^2} \left( n \left( \frac{1}{n} \sum_{i=1}^{N_c} \alpha_{ik} E_{il} - \frac{1}{n} \sum_{i=1}^{N_c} \alpha_{ik} x_i - \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} \right) - \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right) \\
&= \frac{2}{n^2} \left( n \left( \frac{1}{n} \alpha_{kl} - \frac{1}{n} \sum_{i=1}^{N_c} \alpha_{ik} x_i - \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} \right) - \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) \right) \\
&= \frac{2}{n^2} \left( \alpha_{kl} - \sum_{i=1}^{N_c} \alpha_{ik} x_i - n \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} - \sum_{i=1}^{N_c} \alpha_{ik} x_i + \alpha_m \right) \\
&= \frac{2}{n^2} \left( \alpha_{kl} - n \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} - n \frac{2}{n} \left( \sum_{i=1}^{N_c} \alpha_{ik} x_i - \alpha_m \right) - \alpha_m \right) \\
&= \frac{2}{n^2} \left( \alpha_{kl} - n \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} - n \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \alpha_m \right). \\
\left( \frac{\partial^2 b_m}{\partial n_k \partial n_l} \right)_{P,T}
&= \frac{\partial}{\partial n_l} \left( \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right)_{P,T} \\
&= \frac{\partial}{\partial n_l} \left( \frac{b_k - b_m}{n} \right)_{P,T} \\
&= \frac{n \frac{\partial}{\partial n_l} \left( b_k - b_m \right)_{P,T} - \left( b_k - b_m \right) \left( \frac{\partial n}{\partial n_l} \right)_{P,T}}{n^2} \\
&= \frac{-n \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} - b_k + b_m}{n^2} \\
&= \frac{2 b_m - b_l - b_k}{n^2}.
\end{align} $$

+++

Получим вторую частную производную параметра $\alpha_{ij}$ по температуре:

+++

$$ \begin{alignat}{1}
\left( \frac{\partial^2 \alpha_{ij}}{\partial T^2} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \frac{\partial}{\partial T} \left( \left( 1 - \delta_{ij} \right) \frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{2 \sqrt{\alpha_i \alpha_j}} - \sqrt{\alpha_i \alpha_j} \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; -\frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{2 \sqrt{\alpha_i \alpha_j}} \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} + \left( 1 - \delta_{ij} \right) \frac{\partial}{\partial T} \left( \frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{2 \sqrt{\alpha_i \alpha_j}} \right)_{P,n_i} \\
&& \; - \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \sqrt{\alpha_i \alpha_j} \right)_{P,n_i} - \sqrt{\alpha_i \alpha_j} \left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i} \\
&= & \; -\frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{\sqrt{\alpha_i \alpha_j}} \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} + \left( 1 - \delta_{ij} \right) \frac{\partial}{\partial T} \left( \frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{2 \sqrt{\alpha_i \alpha_j}} \right)_{P,n_i} - \sqrt{\alpha_i \alpha_j} \left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i}.
\end{alignat}$$

+++

Распишем второе слагаемое:

+++

$$ \begin{align}
\frac{\partial}{\partial T} \left( \frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{2 \sqrt{\alpha_i \alpha_j}} \right)_{P,n_i}
&= \frac{\sqrt{\alpha_i \alpha_j} \frac{\partial}{\partial T} \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right)_{P,n_i} - \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) \frac{\partial}{\partial T} \left( \sqrt{\alpha_i \alpha_j} \right)_{P,n_i}}{2 \alpha_i \alpha_j} \\
&= \frac{\left( \alpha_j \left( \frac{\partial^2 \alpha_i}{\partial T^2} \right)_{P,n_i} + \alpha_i \left( \frac{\partial^2 \alpha_j}{\partial T^2} \right)_{P,n_i} + 2 \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) - \frac{\left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right)^2}{2 \alpha_i \alpha_j}}{2 \sqrt{\alpha_i \alpha_j}}
\end{align} $$

+++

Тогда вторая частная производная параметра $\alpha_{ij}$ по температуре:

+++

$$ \begin{align}
\left( \frac{\partial^2 \alpha_{ij}}{\partial T^2} \right)_{P,n_i} =
&-\frac{\alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i}}{\sqrt{\alpha_i \alpha_j}} \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} \\
&+ \left( 1 - \delta_{ij} \right) \frac{\left( \alpha_j \left( \frac{\partial^2 \alpha_i}{\partial T^2} \right)_{P,n_i} + \alpha_i \left( \frac{\partial^2 \alpha_j}{\partial T^2} \right)_{P,n_i} + 2 \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) - \frac{1}{2 \alpha_i \alpha_j} \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right)^2}{2 \sqrt{\alpha_i \alpha_j}} \\
&- \sqrt{\alpha_i \alpha_j} \left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i}.
\end{align} $$

+++

Вторая частная производная параметра $\alpha_i$ (и $\alpha_j$) по температуре:

+++

$$ \begin{align}
\left( \frac{\partial^2 \alpha_i}{\partial T^2} \right)_{P,n_i}
&= \frac{\partial}{\partial T} \left( \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= \frac{\partial}{\partial T} \left( - \kappa_i \sqrt{\frac{a_i \alpha_i}{{T_c}_i T}} \right)_{P,n_i} \\
&= -\kappa_i \sqrt{\frac{a_i }{{T_c}_i}} \frac{\partial}{\partial T} \left( \sqrt{\frac{\alpha_i}{T}} \right)_{P,n_i} \\
&= -\kappa_i \sqrt{\frac{a_i }{{T_c}_i}} \frac{1}{2\sqrt{\frac{\alpha_i}{T}}} \frac{T \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} - \alpha_i}{T^2} \\
&= -\frac{\kappa_i}{2} \sqrt{\frac{a_i T}{{T_c}_i \alpha_i}} \left( \frac{1}{T} \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} - \frac{\alpha_i}{T^2} \right).
\end{align}$$

+++

Вторая частная производная параметра $\alpha_i$ для воды (как компонента), используемого в [уравнении состояния Сорейде-Уитсона](EOS-3-SW.html#pvt-eos-sw), будет рассмотрена [ниже](#pvt-eos-appendix-fugacity_pd-sw). Также необходимо получить вторые частные производные коэффициентов попарного взаимодействия по температуре при использовании [GCM](EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip-gcm) для их расчета:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \frac{\partial}{\partial T} \left( \frac{1}{2} \left( \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \right. \right. \\
&& \; \left. \left. - \frac{\delta_{ij}}{\alpha_i \alpha_j} \left(\alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \right)_{P,n_i} \\
&= & \; \frac{1}{2} \frac{\partial}{\partial T} \left( \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \right)_{P,n_i} \\
&& \; - \frac{1}{2} \frac{\partial}{\partial T} \left( \frac{\delta_{ij}}{\alpha_i \alpha_j} \left(\alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right)_{P,n_i}.
\end{alignat}$$

+++

Рассмотрим подробнее первое и второе слагаемые. Для начала обозначим $S$ следующее выражение:

+++

$$S = \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right).$$ 

+++

Тогда первое слагаемое можно преобразовть следующим образом:

+++

$$ \frac{\partial}{\partial T} \left( \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} S \right)_{P,n_i} = S \frac{\partial}{\partial T} \left( \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} \right)_{P,n_i} + \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} \left( \frac{\partial S}{\partial T} \right)_{P,n_i} = - \frac{S}{2} \frac{b_i b_j}{\sqrt{\alpha_i^3 \alpha_j^3}} \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) + \frac{b_i b_j}{\sqrt{\alpha_i \alpha_j}} \left( \frac{\partial S}{\partial T} \right)_{P,n_i}.$$ 

+++

Преобразуем частную производную $\left( \frac{\partial S}{\partial T} \right)_{P,n_i}$:

+++

$$\begin{alignat}{1}
\left( \frac{\partial S}{\partial T} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right)_{P,n_i} \\
&= & \; \left( \frac{\partial^2 DS}{\partial T^2} \right)_{P,n_i} - \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \frac{\partial}{\partial T} \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)_{P,n_i} \\
&& \; - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \frac{\partial}{\partial T} \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \left( \frac{\partial^2 DS}{\partial T^2} \right)_{P,n_i} - \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \left( \frac{1}{2 b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{2 b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) - \\
&& \;  - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i} \frac{\sqrt{\alpha_i} \left( \frac{\partial^2 \alpha_i }{\partial T^2} \right)_{P,n_i} - \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} \frac{1}{2 \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i}}{\alpha_i} - \frac{1}{b_j} \frac{\sqrt{\alpha_j} \left( \frac{\partial^2 \alpha_j }{\partial T^2} \right)_{P,n_i} - \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \frac{1}{2 \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i}}{\alpha_j} \right) \\
&= & \; \left( \frac{\partial^2 DS}{\partial T^2} \right)_{P,n_i} - \frac{1}{2} \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right)^2 \\
&& \; - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)  \left( \frac{\left( \frac{\partial^2 \alpha_i }{\partial T^2} \right)_{P,n_i} - \frac{1}{2 \alpha_i} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i}^2}{b_i \sqrt{\alpha_i}} - \frac{\left( \frac{\partial^2 \alpha_j }{\partial T^2} \right)_{P,n_i} - \frac{1}{2 \alpha_j} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i}^2}{b_j \sqrt{\alpha_j}} \right).
\end{alignat}$$

+++

Второе слагаемое частной производной $\left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i}$:

+++

$$\begin{alignat}{1}
\frac{\partial}{\partial T} \left( \frac{\delta_{ij}}{\alpha_i \alpha_j} \left(\alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right)_{P,n_i}
&= & \; \frac{1}{\left( \alpha_i \alpha_j \right)^2} \left( \alpha_i \alpha_j \frac{\partial}{\partial T} \left( \delta_{ij} \left( \alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right)_{P,n_i} \right. \\
&& \; \left. - \delta_{ij} \left( \alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \frac{\partial}{\partial T} \left( \alpha_i \alpha_j \right)_{P,n_i} \right) \\
&= & \; \frac{1}{\left( \alpha_i \alpha_j \right)^2} \left( \alpha_i \alpha_j \left( \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} \right. \right. \\
&& \; \left. \left. + \delta_{ij} \frac{\partial}{\partial T} \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \right) \right. \\
&& \; \left. - \delta_{ij} \left( \alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right)^2 \right) \\
&= & \; \frac{1}{\alpha_i \alpha_j} \left( \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} \right. \\
&& \; \left. + \delta_{ij} \left( \alpha_j \left( \frac{\partial^2 \alpha_i}{\partial T^2} \right)_{P,n_i} + \alpha_i \left( \frac{\partial^2 \alpha_j}{\partial T^2} \right)_{P,n_i} + 2 \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) \right) \\
&& \; - \frac{\delta_{ij}}{\left( \alpha_i \alpha_j \right)^2} \left( \alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right)^2.
\end{alignat}$$

+++

Таким образом, финальное выражение для второй частной производной коэффициентов попарного взаимодействия по температуре:

+++

$$\begin{align}
\left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i} = 
&- \frac{1}{4} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right) \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right) \right) \frac{b_i b_j}{\sqrt{\alpha_i^3 \alpha_j^3}} \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) + \\ 
& + \frac{b_i b_j}{2\sqrt{\alpha_i \alpha_j}} \left( \left( \frac{\partial^2 DS}{\partial T^2} \right)_{P,n_i} - \frac{1}{2} \left( \frac{1}{b_i \sqrt{\alpha_i}} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} - \frac{1}{b_j \sqrt{\alpha_j}} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right)^2 - \left( \frac{\sqrt{\alpha_i}}{b_i} - \frac{\sqrt{\alpha_j}}{b_j} \right)  \left( \frac{\left( \frac{\partial^2 \alpha_i }{\partial T^2} \right)_{P,n_i} - \frac{1}{2 \alpha_i} \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i}^2}{b_i \sqrt{\alpha_i}} - \frac{\left( \frac{\partial^2 \alpha_j }{\partial T^2} \right)_{P,n_i} - \frac{1}{2 \alpha_j} \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i}^2}{b_j \sqrt{\alpha_j}} \right) \right) - \\ 
& - \frac{1}{2 \alpha_i \alpha_j} \left( \left( \alpha_j \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) \left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} + \delta_{ij} \left( \alpha_j \left( \frac{\partial^2 \alpha_i}{\partial T^2} \right)_{P,n_i} + \alpha_i \left( \frac{\partial^2 \alpha_j}{\partial T^2} \right)_{P,n_i} + 2 \left( \frac{\partial \alpha_i}{\partial T} \right)_{P,n_i} \left( \frac{\partial \alpha_j}{\partial T} \right)_{P,n_i} \right) \right) + \\ 
& + \frac{\delta_{ij}}{2 \left( \alpha_i \alpha_j \right)^2} \left( \alpha_j \left( \frac{\partial \alpha_i }{\partial T} \right)_{P,n_i} + \alpha_i \left( \frac{\partial \alpha_j }{\partial T} \right)_{P,n_i} \right)^2.
\end{align}$$

+++

Также получим вторую частную производную удвоенной суммы $DS$ по температуре:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 DS}{\partial T^2} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \frac{\partial}{\partial T} \left( \frac{149.075}{T^2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2} \right)_{P,n_i} \\
&= & \; \frac{149.075}{T^4} \left( T^2 \frac{\partial}{\partial T} \left( \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2} \right)_{P,n_i} \right. \\
&& \; \left. - 2 T \left( \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2} \right) \right) \\
&= & \; \frac{149.075}{T^2} \frac{\partial}{\partial T} \left( \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2} \right)_{P,n_i} \\
&& \; - \frac{2}{T} \frac{149.075}{T^2} \left( \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2} \right) \\
&= & \; \frac{149.075}{T^2} \frac{\partial}{\partial T} \left( \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 2} \right)_{P,n_i} - \frac{2}{T} \left( \frac{\partial DS}{\partial T} \right)_{P,n_i} \\
&= & \; -\frac{298.15^2}{2 T^4} \left( \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) \left( B_{kl} - A_{kl} \right) \left( \frac{B_{kl}}{A_{kl}} - 2 \right) \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 3} \right) - \frac{2}{T} \left( \frac{\partial DS}{\partial T} \right)_{P,n_i}.
\end{alignat}$$

+++

Рассмотрим вторую частную производную $g_Z$ по температуре:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 g_Z}{\partial T^2} \right)_{P, n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial g_Z}{\partial T} \right)_{P, n_i} \right)_{P, n_i} = \frac{\partial}{\partial T} \left( \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \right)_{P, n_i} \\
&= & \; \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \right)_{P, n_i} + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial T^2} \right)_{P, n_i} + \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \frac{\partial}{\partial T} \left( \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \right)_{P, n_i} \\
&& \; + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T^2} \right)_{P, n_i} \\
&= & \; \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} + \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \right) + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial T^2} \right)_{P, n_i} \\
&& \; + \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \left( \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} + \left( \frac{\partial^2 g_Z}{\partial B \partial Z} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \right) + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T^2} \right)_{P, n_i} \\
&= & \; \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i}^2 + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial T^2} \right)_{P, n_i} + \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}^2 + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T^2} \right)_{P, n_i} \\
&& \; + 2 \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i}.
\end{alignat}$$

+++

Аналогично могут быть получены остальные частные производные $g_Z$:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 g_Z}{\partial P^2} \right)_{T, n_i}
&= & \; \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i}^2 + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial P^2} \right)_{T, n_i} + \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}^2 + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial P^2} \right)_{T, n_i} \\
&& \; + 2 \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i}; \\
\left( \frac{\partial^2 g_Z}{\partial n_k \partial n_l} \right)_{P,T}
&= & \; \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial Z}{\partial n_l} \right)_{P,T} + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial n_k \partial n_l} \right)_{P,T} + \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial n_l} \right)_{P,T} \\
&& \; + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial n_k \partial n_l} \right)_{P,T} + \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial n_l} \right)_{P,T} + \left( \frac{\partial Z}{\partial n_l} \right)_{P,T} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \right); \\
\left( \frac{\partial^2 g_Z}{\partial T \partial n_k} \right)_{P}
&= & \; \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial T \partial n_k} \right)_{P} + \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \\
&& \; + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial T \partial n_k} \right)_{P} +  \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \right); \\
\left( \frac{\partial^2 g_Z}{\partial P \partial n_k} \right)_{T}
&= & \; \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial P \partial n_k} \right)_{T} + \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} \\
&& \; + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial P \partial n_k} \right)_{T} + \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial n_k} \right)_{P,T} + \left( \frac{\partial Z}{\partial n_k} \right)_{P,T} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \right); \\
\left( \frac{\partial^2 g_Z}{\partial P \partial T} \right)_{n_i}
&= & \; \left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} + \left( \frac{\partial g_Z}{\partial Z} \right)_{T,P,n_i} \left( \frac{\partial^2 Z}{\partial P \partial T} \right)_{n_i} + \left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial g_Z}{\partial B} \right)_{T,P,n_i} \left( \frac{\partial^2 B}{\partial P \partial T} \right)_{n_i} + \left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} \left( \left( \frac{\partial Z}{\partial P} \right)_{T,n_i} \left( \frac{\partial B}{\partial T} \right)_{P,n_i} + \left( \frac{\partial Z}{\partial T} \right)_{P,n_i} \left( \frac{\partial B}{\partial P} \right)_{T,n_i} \right).
\end{alignat}$$

+++

При этом, вторые частные производные $g_Z$ по $Z$ и $B$:

+++

$$\begin{align}
\left( \frac{\partial^2 g_Z}{\partial Z^2} \right)_{T,P,n_i} &= - \frac{1}{\left( Z - B \right)^2}; \\
\left( \frac{\partial^2 g_Z}{\partial B^2} \right)_{T,P,n_i} &= - \frac{1}{\left( Z - B \right)^2}; \\
\left( \frac{\partial^2 g_Z}{\partial Z \partial B} \right)_{T,P,n_i} &= \frac{1}{\left( Z - B \right)^2}.
\end{align}$$

+++

Рассмотрим последние слагаемые в выражениях вторых частных производных логарифма коэффициента летучести.

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial T^2} \right)_{P, n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial T} \right)_{P, n_i} \right)_{P,n_i} = f_z \left( \frac{\partial^2 g_{\phi_i}}{\partial T^2} \right)_{P, n_i} + g_{\phi_i} \left( \frac{\partial^2 f_Z}{\partial T^2} \right)_{P, n_i} + 2 \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P, n_i} \left( \frac{\partial f_Z}{\partial T} \right)_{P, n_i};\\
\left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial P^2} \right)_{T, n_i}
&= & \; \frac{\partial}{\partial P} \left( \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial P} \right)_{T, n_i} \right)_{T,n_i} = f_z \left( \frac{\partial^2 g_{\phi_i}}{\partial P^2} \right)_{T, n_i} + g_{\phi_i} \left( \frac{\partial^2 f_Z}{\partial P^2} \right)_{T, n_i} + 2 \left( \frac{\partial g_{\phi_i}}{\partial P} \right)_{T, n_i} \left( \frac{\partial f_Z}{\partial P} \right)_{T, n_i};\\
\left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial n_k \partial n_l} \right)_{P, T}
&= & \; \frac{\partial}{\partial n_l} \left( \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial n_k} \right)_{P,T} \right)_{P,T} = f_z \left( \frac{\partial^2 g_{\phi_i}}{\partial n_k \partial n_l} \right)_{P, T} + g_{\phi_i} \left( \frac{\partial^2 f_Z}{\partial n_k \partial n_l} \right)_{P, T} + \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} \left( \frac{\partial f_Z}{\partial n_l} \right)_{P,T} \\
&& \; + \left( \frac{\partial g_{\phi_i}}{\partial n_l} \right)_{P,T} \left( \frac{\partial f_Z}{\partial n_k} \right)_{P,T};\\
\left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial n_k \partial T} \right)_{P}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial n_k} \right)_{P,T} \right)_{P,n_i} = f_z \left( \frac{\partial^2 g_{\phi_i}}{\partial n_k \partial T} \right)_{P} + g_{\phi_i} \left( \frac{\partial^2 f_Z}{\partial n_k \partial T} \right)_{P} + \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} \left( \frac{\partial f_Z}{\partial T} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \left( \frac{\partial f_Z}{\partial n_k} \right)_{P,T};\\
\left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial n_k \partial P} \right)_{T}
&= & \; \frac{\partial}{\partial P} \left( \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial n_k} \right)_{P,T} \right)_{T,n_i} = f_z \left( \frac{\partial^2 g_{\phi_i}}{\partial n_k \partial P} \right)_{T} + g_{\phi_i} \left( \frac{\partial^2 f_Z}{\partial n_k \partial P} \right)_{T} + \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} \left( \frac{\partial f_Z}{\partial P} \right)_{T,n_i} \\
&& \; + \left( \frac{\partial g_{\phi_i}}{\partial P} \right)_{T,n_i} \left( \frac{\partial f_Z}{\partial n_k} \right)_{P,T};\\
\left( \frac{\partial^2 \left( g_{\phi_i} f_Z \right)}{\partial P \partial T} \right)_{n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial \left( g_{\phi_i} f_Z \right)}{\partial P} \right)_{T,n_i} \right)_{P,n_i} = f_z \left( \frac{\partial^2 g_{\phi_i}}{\partial P \partial T} \right)_{n_i} + g_{\phi_i} \left( \frac{\partial^2 f_Z}{\partial P \partial T} \right)_{n_i} + \left( \frac{\partial g_{\phi_i}}{\partial P} \right)_{T,n_i} \left( \frac{\partial f_Z}{\partial T} \right)_{P,n_i} \\
&& \; + \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \left( \frac{\partial f_Z}{\partial P} \right)_{T,n_i}.
\end{alignat}$$

+++

Получим выражения для вторых частных производных $g_{\phi_i}$:

+++

$$\begin{alignat}{1}
\left( \frac{\partial^2 g_{\phi_i}}{\partial T^2} \right)_{P,n_i}
&= & \; \frac{\partial}{\partial T} \left( \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \right)_{P,n_i} \\
&= & \; \frac{\partial}{\partial T} \left( \frac{1}{T} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) - g_{\phi_i} \right) \right)_{P,n_i} \\
&= & \; \frac{1}{T} \frac{\partial}{\partial T} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)_{P,n_i} - g_{\phi_i} \right) \\
&& \; - \frac{1}{T^2} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) - g_{\phi_i} \right) \\
&= & \; \frac{1}{T} \frac{\partial}{\partial T} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)_{P,n_i} - g_{\phi_i} \right) - \frac{1}{T} \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \\
&= & \; \frac{1}{R T b_m^2} \left( b_m \frac{\partial}{\partial T} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)_{P,n_i} - \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right) \right) \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \right) \\
&& \; - \frac{2}{T} \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \\
&= & \; \frac{1}{R T b_m^2} \left( b_m \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial^2 \alpha_{ij}}{\partial T^2} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{P,n_i} + \frac{b_i}{b_m^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \right) \right. \\
&& \; \left. - \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right) \right) \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \right) - \frac{2}{T} \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \\
&= & \; \frac{1}{R T b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial^2 \alpha_{ij}}{\partial T^2} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{P,n_i} + \frac{b_i}{b_m^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \right. \\
&& \; \left. - R \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \left( g_{\phi_i} + T \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \right) \right) - \frac{2}{T} \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \\
&= & \; \frac{1}{T} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial^2 \alpha_{ij}}{\partial T^2} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial^2 \alpha_m}{\partial T^2} \right)_{P,n_i} + \frac{b_i}{b_m^2} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \right. \right. \\
&& \; \left. \left. - R \left( \frac{\partial b_m}{\partial T} \right)_{P,n_i} \left( g_{\phi_i} + T \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \right) \right) - 2 \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} \right); \\
\left( \frac{\partial^2 g_{\phi_i}}{\partial n_k \partial n_l} \right)_{P, T}
&= & \; \frac{\partial}{\partial n_l} \left( \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P, T} \right)_{P, T} \\
&= & \; \frac{\partial}{\partial n_l} \left( \frac{1}{R T b_m} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} g_{\phi_i} R T - \frac{b_i}{b_m} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right) \right)_{P, T} \\
&= & \; \frac{1}{RTb_m^2} \left( b_m \frac{\partial}{\partial n_l} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} g_{\phi_i} R T - \frac{b_i}{b_m} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right)_{P,T} \right. \\
&& \; \left. - \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} g_{\phi_i} R T - \frac{b_i}{b_m} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right) \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} \right) \\
&= & \; \frac{1}{RT b_m} \left( \frac{\partial}{\partial n_l} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) - \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} g_{\phi_i} R T - \frac{b_i}{b_m} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right)_{P,T} \right. \\
&& \; \left. - RT \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} \right) \\
&= & \; \frac{1}{RT b_m} \left( \frac{\partial}{\partial n_l} \left( \frac{2}{n} \left( \alpha_{ik} - \sum_{j=1}^{N_c} x_j \alpha_{ij} \right) \right)_{P,T} - RT \frac{\partial}{\partial n_l} \left( \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} g_{\phi_i} \right)_{P,T} \right. \\
&& \; \left. - \frac{\partial}{\partial n_l} \left( \frac{b_i}{b_m} \left( \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right)_{P,T} - RT \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} \right) \\
&= & \; \frac{1}{RT b_m} \left( \frac{2}{n^2} \left( 2 \sum_{j=1}^{N_c} x_j \alpha_{ij} - \alpha_{ik} - \alpha_{il} \right) - RT g_{\phi_i} \left( \frac{\partial^2 b_m}{\partial n_k \partial n_l} \right)_{P, T} \right. \\
&& \; \left. - \frac{b_i}{b_m} \left( \left( \frac{\partial^2 \alpha_m}{\partial n_k \partial n_l} \right)_{P, T} - \frac{1}{b_m} \left( \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial \alpha_m}{\partial n_l} \right)_{P,T} + \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} \left( \frac{\partial \alpha_m}{\partial n_k} \right)_{P,T} - 2 \frac{\alpha_m}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} \right) \right. \right. \\
&& \; \left. \left. - \frac{\alpha_m}{b_m} \left( \frac{\partial^2 b_m}{\partial n_k \partial n_l} \right)_{P, T} \right) - RT \left( \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \frac{\partial g_{\phi_i}}{\partial n_l} \right)_{P,T} + \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P,T} \left( \frac{\partial b_m}{\partial n_l} \right)_{P,T} \right) \right); \\
\left( \frac{\partial^2 g_{\phi_i}}{\partial n_k \partial T} \right)_{P}
&= & \; \frac{\partial}{\partial n_k} \left( \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P, n_i} \right)_{P, T} \\
&= & \; \frac{\partial}{\partial n_k} \left( \frac{1}{T} \left( \frac{1}{R b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) - g_{\phi_i} \right) \right)_{P, T} \\
&= & \; \frac{\partial}{\partial n_k} \left( \frac{1}{R T b_m} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) \right)_{P, T} - \frac{1}{T} \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P, T} \\
&= & \; \frac{1}{R T} \frac{b_m \frac{\partial}{\partial n_k} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)_{P,T} - \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right) \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} }{b_m^2} \\
&& \; - \frac{1}{T} \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P, T} \\
&= & \; \frac{1}{R T b_m} \frac{\partial}{\partial n_k} \left( 2 \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} - \frac{b_i}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \right)_{P,T} - \frac{1}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( T \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} + g_{\phi_i} \right) - \frac{1}{T} \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P, T} \\
&= & \; \frac{1}{R T b_m} \left( \frac{2}{n} \left( \left( \frac{\partial \alpha_{ik}}{\partial T} \right)_{P,n_i} - \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} \right) - \frac{b_i}{b_m} \left( \left( \frac{\partial^2 \alpha_m}{\partial T \partial n_k} \right)_{P} - \frac{1}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \right) \right) \\
&& \; - \frac{1}{b_m} \left( \frac{\partial b_m}{\partial n_k} \right)_{P,T} \left( \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} + \frac{g_{\phi_i}}{T} \right) - \frac{1}{T} \left( \frac{\partial g_{\phi_i}}{\partial n_k} \right)_{P, T}; \\
\left( \frac{\partial^2 g_{\phi_i}}{\partial n_i \partial T} \right)_{P}
&= & \; \frac{1}{R T b_m} \left( \frac{2}{n} \left( \sum_{j=1}^{N_c} \left( \frac{\partial \alpha_{ij, \; i=j}}{\partial T} \right)_{P,n_i} - \sum_{j=1}^{N_c} x_j \left( \frac{\partial \alpha_{ij}}{\partial T} \right)_{P,n_i} \right) - \frac{b_i}{b_m} \left( \left( \frac{\partial^2 \alpha_m}{\partial T \partial n_i} \right)_{P} - \frac{1}{b_m} \left( \frac{\partial \alpha_m}{\partial T} \right)_{P,n_i} \left( \frac{\partial b_m}{\partial n_i} \right)_{P,T} \right) \right) \\
&& \; - \frac{1}{b_m} \left( \frac{\partial b_m}{\partial n_i} \right)_{P,T} \left( \left( \frac{\partial g_{\phi_i}}{\partial T} \right)_{P,n_i} + \frac{g_{\phi_i}}{T} \right) - \frac{1}{T} \left( \frac{\partial g_{\phi_i}}{\partial n_i} \right)_{P, T}.
\end{alignat}$$

+++

Остальные частные производные $g_{\phi_i}$ равны нулю, поскольку $g_{\phi_i}$ не зависит от давления.

+++

Вторые частные производные $f_Z$ по давлению, температуре и количеству вещества $k$-го и $l$-го компонентов определяются аналогично вторым частным производным $g_Z$. Поэтому приведем вторые частные производные $f_Z$ по коэффициенту сверхсжимаемости $Z$ и параметру $B$:

+++

$$\begin{align}
\left( \frac{\partial^2 f_Z}{\partial Z^2} \right)_{T,P,n_i} &= - \frac{1}{\left( Z + B \delta_1 \right)^2} + \frac{1}{\left( Z + B \delta_2 \right)^2}; \\
\left( \frac{\partial^2 f_Z}{\partial Z \partial B} \right)_{T,P,n_i} &= - \frac{\delta_1}{\left( Z + B \delta_1 \right)^2} + \frac{\delta_2}{\left( Z + B \delta_2 \right)^2}; \\
\left( \frac{\partial^2 f_Z}{\partial B^2} \right)_{T,P,n_i} &= - \left( \frac{\delta_1}{Z + B \delta_1} \right)^2 + \left( \frac{\delta_2}{Z + B \delta_2} \right)^2.
\end{align}$$

+++

Зная вторые частные производные логарифма коэффициента летучести $i$-го компонента, получим вторые частные производные логарифма летучести $i$-го компонента. Вторая частная производная логарифма коэффициента летучести по температуре:

+++

$$ \left( \frac{\partial^2 \ln \phi_i}{\partial T^2} \right)_{P,n_i} = \frac{\partial}{\partial T} \left( \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i} \right)_{P,n_i} = \frac{\partial}{\partial T} \left( \left( \frac{\partial \ln f_i}{\partial T} \right)_{P,n_i} \right)_{P,n_i} = \left( \frac{\partial^2 \ln f_i}{\partial T^2} \right)_{P,n_i}.$$ 

+++

Аналогично вторая частная производная логарифма коэффициента летучести по давлению:

+++

$$ \left( \frac{\partial^2 \ln \phi_i}{\partial P^2} \right)_{T,n_i} = \frac{\partial}{\partial P} \left( \left( \frac{\partial \ln \phi_i}{\partial P} \right)_{T,n_i} \right)_{T,n_i} = \frac{\partial}{\partial P} \left( \left( \frac{\partial \ln f_i}{\partial P} \right)_{P,n_i} + \frac{1}{P} \right)_{T,n_i} = \left( \frac{\partial^2 \ln f_i}{\partial P^2} \right)_{T,n_i} - \frac{1}{P^2}.$$

+++

Следовательно,

+++

$$\left( \frac{\partial^2 \ln f_i}{\partial P^2} \right)_{T,n_i} = \left( \frac{\partial^2 \ln \phi_i}{\partial P^2} \right)_{T,n_i} + \frac{1}{P^2}.$$ 

+++

Вторая частная производная логарифма коэффициента летучести $i$-го компонента по количеству вещества $k$-го и $l$-го компонентов: 

+++

$$ \left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial n_l} \right)_{P,T} = \frac{\partial}{\partial n_l} \left( \left( \frac{\partial \ln \phi_i}{\partial n_k} \right)_{P,T} \right)_{P,T} = \frac{\partial}{\partial n_l} \left( \left( \frac{\partial \ln f_i}{\partial n_k} \right)_{P,T} - \frac{E_{ik}}{n_i} + \frac{1}{n} \right)_{P,T} = \left( \frac{\partial^2 \ln f_i}{\partial n_k \partial n_l} \right)_{P,T} + \frac{E_{ik} E_{il}}{n_i^2} - \frac{1}{n^2}.$$

+++

Тогда:

+++

$$\left( \frac{\partial^2 \ln f_i}{\partial n_k \partial n_l} \right)_{P,T} = \left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial n_l} \right)_{P,T} - \frac{E_{ik} E_{il}}{n_i^2} + \frac{1}{n^2}.$$

+++


Вторая частная производная логарифма коэффициента летучести $i$-го компонента по температуре $T$ и количеству вещества $k$-го компонента:

+++

$$ \left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial T} \right)_{P} = \frac{\partial}{\partial T} \left( \left( \frac{\partial \ln \phi_i}{\partial n_k} \right)_{P,T} \right)_{P,n_i} = \frac{\partial}{\partial T} \left( \left( \frac{\partial \ln f_i}{\partial n_k} \right)_{P,T} - \frac{E_{ik}}{n_i} + \frac{1}{n} \right)_{P,n_i} = \left( \frac{\partial^2 \ln f_i}{\partial n_k \partial T} \right)_{P}.$$

+++

Аналогично вторая частная производная логарифма коэффициента летучести $i$-го компонента по давлению $P$ и количеству вещества $k$-го компонента:

+++

$$ \left( \frac{\partial^2 \ln \phi_i}{\partial n_k \partial P} \right)_{T} = \frac{\partial}{\partial P} \left( \left( \frac{\partial \ln \phi_i}{\partial n_k} \right)_{P,T} \right)_{T,n_i} = \frac{\partial}{\partial P} \left( \left( \frac{\partial \ln f_i}{\partial n_k} \right)_{P,T} - \frac{E_{ik}}{n_i} + \frac{1}{n} \right)_{T,n_i} = \left( \frac{\partial^2 \ln f_i}{\partial n_k \partial T} \right)_{T}.$$

+++

Наконец, вторая частная производная логарифма коэффициента летучести $i$-го компонента по давлению $P$ и температуре $T$:

+++

$$ \left( \frac{\partial^2 \ln \phi_i}{\partial P \partial T} \right)_{n_i} = \frac{\partial}{\partial P} \left( \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i} \right)_{T,n_i} = \frac{\partial}{\partial P} \left( \left( \frac{\partial \ln f_i}{\partial T} \right)_{P,n_i} \right)_{T,n_i} = \left( \frac{\partial^2 \ln f_i}{\partial P \partial T} \right)_{n_i} .$$

````

+++

<a id='pvt-eos-appendix-fugacity_pd-srk_pr-v'></a>
## Частные производные летучести компонентов с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона (при постоянных объеме и температуре)

+++

````{div} full-width
Логарифм коэффициента летучести $i$-го компонента с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона, [записанный](./EOS-2-SRK-PR.html#pvt-eos-srk_pr-fugacity_coeff-tv) относительно объема $V$, температуры $T$ и коэффициента сверхсжимаемости $Z$:

+++

$$ \begin{align} \ln \phi_i = &-\ln \frac{V - n b_m}{V} + b_i \left( \frac{n}{V - n b_m} - \frac{n \alpha_m}{R T b_m} \frac{V}{\left( V + n b_m \delta_1 \right) \left( V + n b_m \delta_2 \right)} \right) \\ & + \frac{1}{R T \left( \delta_2 - \delta_1 \right)} \left( \frac{2 \sum_{j=1}^{N_c} \alpha_{ij} x_j}{b_m} - \frac{\alpha_m b_i}{b_m^2} \right) \ln \left( \frac{V + n b_m \delta_1}{V + n b_m \delta_2} \right) - \ln Z. \end{align} $$
````

+++

```{code-cell} ipython3

```

+++

## Вторые частные производные летучести компонентов с использованием уравнений состояния Суаве-Редлиха-Квонга и Пенга-Робинсона (при постоянных объеме и температуре)

+++

```{code-cell} ipython3

```

+++

<a id='pvt-eos-appendix-fugacity_pd-sw'></a>
## Частные производные летучести компонентов с использованием уравнения состояния Сорейде-Уитсона

+++
````{div} full-width
Рассмотрим нахождение частных производных логарифма коэффициента летучести (и логарифма летучести) по давлению, объему температуре и количеству вещества компонента компонента с использованием [уравнения состояния Сорейде-Уитсона](./EOS-3-SW.html#pvt-eos-sw). Для уравнения состояния Сорейде-Уитсона частные производные логарифма коэффициента летучести (и логарифма летучести) находятся аналогично рассмотренным ранее [частным производным для уравнения состояния Пенга-Робинсона](#pvt-eos-appendix-fugacity_pd-srk_pr) за исключением частных производных коэффициента попарного взаимодействия по температуре для ряда компонентов, а также частной производной параметра $\alpha_i$ для воды как компонента.

+++

Для углеводородов $i$ частная производная коэффициента попарного взаимодействия с водой $j$ по температуре определяется следующим выражением:

+++

$$\left( \frac{\partial \delta_{ij}}{\partial T} \right)_{P,n_i} = \frac{A_1}{{T_c}_i} \left( 1 + \alpha_1 c_w \right) + 2 A_2 \frac{T}{{T_c}_i} \left( 1 + \alpha_2 c_w \right).$$

+++

Коэффициенты в данном уравнении были представлены [ранее](./EOS-3-SW.html#pvt-eos-sw). Для неуглеводородных компонентов $i$ частная производная коэффициента попарного взаимодействия с водой $j$ по температуре определяется следующими выражениями:

+++

$$ \begin{align}
\left( \frac{\partial \delta_{ij}^{AQ} \left( N_2 \right)}{\partial T} \right)_{P,n_i} &= \frac{0.44338}{{T_c}_i} \left( 1 + 0.08126 c_w^{0.75} \right); \\
\left( \frac{\partial \delta_{ij}^{AQ} \left( CO_2 \right)}{\partial T} \right)_{P,n_i} &= \frac{1}{{T_c}_i} \left( 0.23580 \left( 1 + 0.17837 c_w^{0.979} \right) + 142.8911 e^{-6.7222 {T_r}_i - c_w} \right); \\
\left( \frac{\partial \delta_{ij}^{AQ} \left( H_2S \right)}{\partial T} \right)_{P,n_i} &= \frac{0.23426}{{T_c}_i}.
\end{align} $$

+++

Частная производная параметра $\alpha_i$ для воды по температуре:

+++

$$\left( \frac{\partial \alpha_w}{\partial T} \right)_{P,n_i} = 2 \sqrt{a_w \alpha_w} \left( - \frac{0.4530}{{T_c}_w} \left( 1 - 0.0103 c_w^{1.1} \right) - 0.0102  \frac{{T_c}_w^3}{T^4} \right).$$
````

+++

## Вторые частные производные летучести компонентов с использованием уравнения состояния Сорейде-Уитсона

+++

````{div} full-width

Вторая частная производная коэффициента попарного взаимодействия между углеводородом $i$ и водой $j$:

+++

$$\left( \frac{\partial^2 \delta_{ij}}{\partial T^2} \right)_{P,n_i} = \frac{2 A_2}{{T_c}_i} \left( 1 + \alpha_2 c_w \right).$$

+++

Для неуглеводородных компонентов вторые частные производные коэффициентов попарного взаимодействия:

+++

$$ \begin{align}
\left( \frac{\partial^2 \delta_{ij}^{AQ} \left( N_2 \right)}{\partial T^2} \right)_{P,n_i} &= 0; \\
\left( \frac{\partial^2 \delta_{ij}^{AQ} \left( CO_2 \right)}{\partial T^2} \right)_{P,n_i} &= -\frac{960.5426}{{T_c}_i^2} e^{-6.7222 {T_r}_i - c_w}; \\
\left( \frac{\partial^2 \delta_{ij}^{AQ} \left( H_2S \right)}{\partial T^2} \right)_{P,n_i} &= 0.
\end{align} $$

+++

Вторая частная производная параметра $\alpha_i$ по температуре для воды:

+++

$$\left( \frac{\partial^2 \alpha_w}{\partial T^2} \right)_{P,n_i} = \frac{1}{2 \alpha_w} \left( \frac{\partial \alpha_w}{\partial T} \right)_{P,n_i}^2 + 0.0816 \frac{\sqrt{a_w \alpha_w} {T_c}_w^3}{T^5}.$$

````
