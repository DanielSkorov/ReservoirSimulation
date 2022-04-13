#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-eos-van_der_waals'></a>
# # Уравнение состояния Ван-дер-Ваальса
# В 1873 году Ван-дер-Ваальс предложил использовать уравнение состояние:

# $$ \left( P + \frac{a_m}{v^2} \right) \left( v - b_m \right) = R T.$$

# В данном уравнении $v = \frac{V}{n}$ – молярный объем. Параметры $a_m$ и $b_m$ являются коэффициентами, со следующими физическими смыслами. Параметр $b_m$ определяет объем полностью сжатой системы при бесконечно большом давлении. Поэтому условием применения уравнения состояния Ван-дер-Ваальса является $v > b_m$. Параметр $a_m$ характеризует межмолекулярное взаимодействие. При рассмотрении системы, состоящей из нескольких компонентов, (смеси) параметры $a_m$ и $b_m$ будут зависеть от видов молекул системы (свойств компонентов), а также ее компонентного состава.

# Выразим из уравнения состояния Ван-дер-Ваальса давление:

# $$P = \frac{n R T}{V - n b_m} - \frac{n^2 a_m}{V^2}.$$

# Для того чтобы определить летучесть компонента, необходимо получить частную производную давления по количеству вещества $i$-го компонента при постоянных давлении, температуре и количествах вещества остальных компонентов.  Стоит сразу отметить, что эта производная будет представлять собой вектор из $N_c$ значений, где $N_c$ – количество компонентов в системе. При этом, от количества вещества $i$-го компонента будут зависеть параметры $a_m$ и $b_m$, а также общее количество вещества $n$. С учетом этого, частная производная давления по количеству вещества $i$-го компонента:

# $$ \begin{align}
# \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}}
# &= \frac{\partial}{\partial n_i} \left( \frac{n R T}{V - n b_m} - \frac{n^2 a_m}{V^2} \right)_{V, T, n_{j \neq i}} \\
# &= \frac{\partial}{\partial n_i} \left( \frac{n R T}{V - n b_m} \right)_{V, T, n_{j \neq i}} - \frac{\partial}{\partial n_i} \left( \frac{n^2 a_m}{V^2} \right)_{V, T, n_{j \neq i}} \\
# &= \frac{RT}{V - n b_m} \frac{\partial n}{\partial n_i} + n \frac{\partial}{\partial n_i} \left( \frac{R T}{V - n b_m} \right) - \frac{1}{V^2} \frac{\partial \left( n^2 a_m \right)}{\partial n_i}. \end{align} $$

# <a id='pvt-eos-van_der_waals-partials'></a>
# Распишем подробнее производную количества вещества $n$ по количеству вещества $i$-го компонента:

# $$ \frac{\partial n}{\partial n_i} = \frac{\partial}{\partial n_i} \left( \sum_{j=1}^{N_c} n_j \right) = \sum_{j=1}^{N_c} \frac{\partial n_j}{\partial n_i} .$$

# Частная производная количества вещества $i$-го компонента по количеству вещества $k$-го компонента:

# $$ \frac{\partial n_j}{\partial n_i} = \begin{bmatrix} \frac{\partial n_1}{\partial n_1} & \frac{\partial n_2}{\partial n_1} & \dots & \frac{\partial n_{N_c}}{\partial n_1} \\ \frac{\partial n_1}{\partial n_2} & \frac{\partial n_2}{\partial n_2} & \dots & \frac{\partial n_{N_c}}{\partial n_2} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial n_1}{\partial n_{N_c}} & \frac{\partial n_2}{\partial n_{N_c}} & \dots & \frac{\partial n_{N_c}}{\partial n_{N_c}} \end{bmatrix} = \begin{bmatrix} 1 & 0 & \dots & 0 \\ 0 & 1 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & 1 \end{bmatrix} = I_{ij}. $$

# Следовательно, в результате суммы столбцов (по $j$) матрицы $I_{ij}$ получится вектор размерностью $\left(N_c \times 1 \right)$:

# $$\sum_{j=1}^{N_c} I_{ij} = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} = \vec{e}.$$

# Таким образом, умножение числа на вектор $\vec{e}$ будет означать, что это число одинаково для всех компонентов. Поэтому умножение на $\vec{e}$ можно опустить.

# С учетом этого частная производная давления по количеству вещества $i$-го компонента:

# $$ \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \frac{RT}{V - n b_m} + \frac{n R T}{\left( V - n b_m \right)^2} \frac{\partial \left( n b_m \right)}{\partial n_i} - \frac{1}{V^2} \frac{\partial \left( n^2 a_m \right)}{\partial n_i}.$$

# Подставим полученное выражение в уравнение для логарифма коэффициента летучести:

# $$ \ln \phi_i = \int_V^\infty \left( \frac{1}{V - n b_m} + \frac{n}{\left( V - n b_m \right)^2} \frac{\partial \left( n b_m \right)}{\partial n_i} - \frac{1}{R T V^2} \frac{\partial \left( n^2 a_m \right)}{\partial n_i} - \frac{1}{V} \right) dV - \ln Z.$$

# Найдем первообразную подынтегральной функции:

# $$ \begin{align}
# F \left( V \right)
# &= \int f \left( V \right) dV \\
# &= \int \left( \frac{1}{V - n b_m} + \frac{n}{\left( V - n b_m \right)^2} \frac{\partial \left( n b_m \right)}{\partial n_i} - \frac{1}{R T V^2} \frac{\partial \left( n^2 a_m \right)}{\partial n_i} - \frac{1}{V} \right) dV \\
# &= \int \frac{dV}{V - n b_m} + n \frac{\partial \left( n b_m \right)}{\partial n_i} \int \frac{dV}{\left( V - n b_m \right)^2} - \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{RT} \int \frac{dV}{V^2} - \int \frac{dV}{V} \\
# &= \int \frac{d \left(V - n b_m \right)}{V - n b_m} + n \frac{\partial \left( n b_m \right)}{\partial n_i} \int \frac{d \left( V - n b_m \right)}{\left( V - n b_m \right)^2} - \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{RT} \int \frac{dV}{V^2} - \int \frac{dV}{V} \\
# &= \ln \lvert V - n b_m \rvert - n \frac{\partial \left( n b_m \right)}{\partial n_i} \frac{1}{V - n b_m} + \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{R T} \frac{1}{V} - \ln \lvert V \rvert \\
# &= \ln \frac{V - n b_m}{V} - n \frac{\partial \left( n b_m \right)}{\partial n_i} \frac{1}{V - n b_m} + \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{R T V}.
# \end{align} $$

# Формула [Ньютона-Лейбница](https://en.wikipedia.org/wiki/Leibniz_integral_rule) для [несобственных интегралов](https://en.wikipedia.org/wiki/Improper_integral):

# $$ \int_a^\infty f \left( x \right) dx = \lim_{b \rightarrow \infty} \int_a^b f \left( x \right) dx = \lim_{b \rightarrow \infty} F \left( x \right) \bigg\rvert_a^b = \lim_{b \rightarrow \infty} \left( F \left( b \right) - F \left( a \right)\right). $$

# С учетом данного выражения несобственный интеграл в выражении для логарифма коэффициента летучести:

# $$ \begin{alignat}{1}
# \int_V^\infty f \left( V \right) dV
# &= &\lim_{V \rightarrow \infty} \left( \ln \frac{V - n b_m}{V} - n \frac{\partial \left( n b_m \right)}{\partial n_i} \frac{1}{V - n b_m} + \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{R T V} \right) \\
# && - \left( \ln \frac{V - n b_m}{V} - n \frac{\partial \left( n b_m \right)}{\partial n_i} \frac{1}{V - n b_m} + \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{R T V} \right) \\
# &= &\ln \frac{V}{V - n b_m} + n \frac{\partial \left( n b_m \right)}{\partial n_i} \frac{1}{V - n b_m} - \frac{\partial \left( n^2 a_m \right)}{\partial n_i} \frac{1}{R T V}.
# \end{alignat} $$

# При преобразовании данного выражения было учтено:

# $$ \begin{align}
# \lim_{V \rightarrow \infty} \left( \ln \frac{V - n b_m}{V} \right) &= 0; \\
# \lim_{V \rightarrow \infty} \left( \frac{1}{V - n b_m} \right) &= 0; \\
# \lim_{V \rightarrow \infty} \left( \frac{1}{V} \right) &= 0.
# \end{align}$$

# Тогда логарифм коэффициента летучести $i$-го компонента:

# $$ \ln \phi_i = \ln \frac{V}{V - n b_m} + \frac{n}{V - n b_m} \frac{\partial \left( n b_m \right)}{\partial n_i} - \frac{1}{R T V} \frac{\partial \left( n^2 a_m \right)}{\partial n_i} - \ln Z. $$

# <a id='pvt-eos-van_der_waals-mix_rules'></a>
# ```{admonition} NB
# Для того чтобы преобразовать это выражение, то есть взять частные производные от параметров $a_m$ и $b_m$ по количеству вещества $i$-го компонента, необходимо ввести ***правила смешивания*** – допущения, позволяющие рассчитать параметры $a_m$ и $b_m$ по компонентному составу системы и свойствам компонентов.
# ```

# Одними из наиболее часто использующихся правил смешивания являются следующие соотношения:

# $$ \begin{align} a_m &= \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} x_j x_k a_{jk}; \\ b_m &= \sum_{j=1}^{N_c} x_j b_j. \end{align}$$

# Здесь $x$ – мольная доля компонента, $a_{jk}$ – параметр, характеризующий степень взаимодействия молекул $j$-го и $k$-го компонентов. Если $j=k$, то параметр $a_{jk}$ должен соответствовать параметру $a$ по Ван-дер-Ваальсу, если же $j \neq k$, то параметр $a_{jk}$ в отсутствие экспериментальных данных должен выражаться из известных значений $a_j$ и $a_k$. Это является одной из основных проблем при использовании уравнения состояния и правил смешивания – правильно учесть взаимовлияние компонентов. Зная взаимодействие молекул чистых компонентов, каким образом выразить взаимодействие молекул разных компонентов между собой? Однозначного ответа на данный вопрос нет. Поэтому на практике используются различные правила смешивания в зависимости от поставленной задачи, а также в качестве инструмента адаптации на лабораторные исследования. Наиболее часто для расчета параметра $a_{jk}$ используется следующее выражение:

# $$a_{jk} = \left( a_j a_k \right) ^ {0.5}.$$

# Также можно встретить дополнительный множитель в этом выражении:

# $$a_{jk} = \left( a_j a_k \right) ^ {0.5} \left(1 - \delta_{jk} \right).$$

# <a id='pvt-eos-van_der_waals-bip'></a>
# В приведенном выше уравнении $\delta_{jk}$ – коэффициент попарного взаимодействия. Данный коэффициент был введен авторами работы \[[Chueh and Prausnitz, 1967](https://doi.org/10.1002/aic.690130612)\] с целью коррекции правил смешивания, основанных на геометрическом осреднении параметров компонентов при расчете параметра смеси. Одним из распространненных подходов к определению коэффициентов попарного взаимодействия является пренебрежние коэффициентами попарного взаимодействие между углеводородными компонентами, поскольку их можно рассматривать как неполярные компоненты, и использование табличных констант для задания коэффициентов попарного взаимодействия между углеводородными и неуглеводородными компонентами \[[Pedersen et al, 1984](https://doi.org/10.1021/i200024a027); [Pedersen et al, 2004](https://doi.org/10.2118/88364-PA)\]. Однако использование данного подхода может быть недостаточно для адаптации PVT-модели к лабораторным данным. Так, авторами работы \[[Katz and Firoozabadi, 1978](https://doi.org/10.2118/6721-PA)\] не удалось достичь удовлетворительной адаптации PVT-модели, пока не были изменены коэффициенты попарного взаимодействия между легкими и тяжелыми компонентами, оказывающиими зачастую наибольшее влияние на фазовое поведение системы. Одной из распространенных корреляций коэффициента попарного взаимодействия двух углеводородных компонентов является:

# $$\delta_{jk} = 1 - \left( \frac{2 {V_c}_j^{\frac{1}{6}} {V_c}_k^{\frac{1}{6}}}{{V_c}_j^{\frac{1}{3}} + {V_c}_k^{\frac{1}{3}}} \right)^c.$$

#  ${V_c}_j$ – критический объем $j$-го компонента, $c$ – коэффициент, обычно принимающий значение $1.2$ \[[Oellrich et al, 1981](https://api.semanticscholar.org/CorpusID:94056056)\], но также выступающий в качестве инструмента адаптации на лабораторные данные. Для неуглеводородных компонентов коэффициент попарного взаимодействия можно найти в научных работах. Выбор корреляции для коэффициентов попарного взаимодействия зависит в том числе от используемых уравнения состояния и правил смешивания, поскольку зачастую коэффиициенты в корреляциях коэффициентов попарного взаимодействия подбираются путем регрессии расчетных значений PVT-свойств флюида к фактическим данным. Поэтому на практике на нулевой итерации начальные коэффициенты попарного взаимодействия определяются по корреляциям, которые будут представлены для уравнений состояния [Суаве-Редлиха-Квонга и Пенга-Робинсона](./EOS-2-SRK-PR.html#pvt-eos-srk_pr) в [приложении B](./EOS-Appendix-B-BIP.html#pvt-eos-appendix-bip), а затем варьируются в процессе адаптации модели к экспериментальным данным. Значения коэффициентов попарного взаимодействия могут быть как положительными, так и отрицательными. Однако в ряде случаев использование положительных и отрицательных коэффициентов попарного взаимодействия компонентов, близких по молекулярной массе, подобранных в результате адаптации модели к лабораторным данным, может приводить к появлению нефизичности в PVT-модели \[[Whitson et al, 2019](http://dx.doi.org/10.15530/urtec-2019-551)\]. Стоит отметить, что коэффициенты попарного взаимодействия здесь и далее рассматриваются симметричными, то есть $\delta_{jk} \approx \delta_{kj}$, поскольку использование несимметричных коэффициентов попарного взаимодействия может приводить к нереалистному моделированию вблизи критической области \[[Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\].

# <a id='pvt-eos-van_der_waals-am_bm_derivative'></a>
# Принимая данные правила смешивания, получим конечное выражение для логарифма коэффициента летучести $i$-го компонента. Для этого необходимо получить производные от параметров $a_m$ и $b_m$.

# $$\frac{\partial n b_m}{\partial n_i} = b_m \frac{\partial n}{\partial n_i} + n \frac{\partial b_m}{\partial n_i}.$$

# В свою очередь, частная производная параметра $b_m$ по количеству вещества $i$-го компонента:

# $$ \begin{align}
# \frac{\partial b_m}{\partial n_i}
# &= \frac{\partial}{\partial n_i} \left( \sum_{j=1}^{N_c} x_j b_j \right) \\
# &= \sum_{j=1}^{N_c} b_j \frac{\partial}{\partial n_i} \left( \frac{n_j}{n} \right) \\
# &= \sum_{j=1}^{N_c} b_j \frac{n \frac{\partial n_j}{\partial n_i} - n_j \frac{\partial n}{\partial n_i}}{n^2} \\
# &= \frac{1}{n} \sum_{j=1}^{N_c} b_j I_{ij} - \frac{1}{n} \sum_{j=1}^{N_c} b_j x_j \\
# &= \frac{b_i - b_m}{n}.
# \end{align} $$

# При получении данного выражения было использовано следующее преобразование:

# $$ \sum_{j=1}^{N_c} b_j I_{ij} = \sum_{j=1}^{N_c} \begin{bmatrix} b_1 & 0 & \dots & 0 \\ 0 & b_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & b_{N_c} \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_{N_c} \end{bmatrix} = b_i. $$

# Получим производную для параметра $a_m$.

# $$ \frac{\partial \left( n^2 a_m \right)}{\partial n_i} = 2 n a_m + n^2 \frac{\partial a_m}{\partial n_i}.$$

# В свою очередь, частная производная параметра $a_m$ по количеству вещества $i$-го компонента:

# $$ \begin{align}
# \frac{\partial a_m}{\partial n_i}
# &= \frac{\partial}{\partial n_i} \left( \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} x_j x_k a_{jk} \right) \\
# &= \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} \frac{\partial}{\partial n_i} \left( \frac{n_j n_k}{n^2} \right) \\
# &= \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} \frac{n^2 \frac{\partial}{\partial n_i} \left( n_j n_k \right) - n_j n_k \frac{\partial n^2}{\partial n_i}}{n^4} \\
# &= \frac{1}{n^2} \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} \left( n_k I_{ij} + n_j I_{ik} \right) - \frac{2}{n} \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} x_j x_k \\
# &= \frac{1}{n^2} \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} n_k I_{ij} + \frac{1}{n^2} \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} n_j I_{ik} - \frac{2}{n} a_m.
# \end{align} $$

# Для преобразования данного выражения рассмотрим следующий пример. Параметр $a_{jk}$ представляет собой симметричную матрицу относительно главной диагонали.

# In[1]:


import numpy as np
import sys
sys.path.append('../../SupportCode/')
from PVT import core


# In[2]:


ajk = np.outer([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
ajk


# При умножении "разнонаправленных" матриц, например, $a_{jk} I_{ij}$ получится трехмерная матрица. Для того чтобы правильно выполнить умножение можно использовать пользовательскую функцию *repeat*, которая предназначена для того, чтобы продублировать вектор или матрицу в заданном направлении.

# In[3]:


Iij = core.repeat(np.identity(3), 1)
Iik = core.repeat(np.identity(3), 0)
nk = core.repeat(core.repeat(np.array([1, 2, 3]), 0), 2)
nj = core.repeat(core.repeat(np.array([1, 2, 3]), 1), 2)
ajk = core.repeat(ajk, 2)


# In[4]:


np.sum(np.sum(ajk * nk * Iij, 1), 0)


# In[5]:


np.sum(np.sum(ajk * nj * Iik, 1), 0)


# В результате в обоих случаях получился один и тот же вектор (в направлении $i$, поскольку суммирование проиходило по направлениям $j$ и $k$). Следовательно, можно записать следующее равенство:

# $$ \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} n_k I_{ij} = \sum_{j=1}^{N_c} \sum_{k=1}^{N_c} a_{jk} n_j I_{ik}. $$

# Аналогичный результат получится, если взять $n_j$ умножить на $a_{ij}$ и просуммировать по $j$.

# In[6]:


aij = np.outer([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
nj = core.repeat(np.array([1, 2, 3]), 0)
nj


# In[7]:


np.sum(aij * nj, 1)


# С учетом этого частная производная параметра $a_m$ по количеству вещества $i$-го компонента:

# $$ \frac{\partial a_m}{\partial n_i} = \frac{2}{n} \left( \sum_{j=1}^{N_c} a_{ij} x_j - a_m \right).$$

# <a id='pvt-eos-van_der_waals-fugacity_coeff-tv'></a>
# Тогда логарифм коэффициента летучести $i$-го компонента:

# $$ \begin{alignat}{1}
# \ln \phi_i &= \; &\ln \frac{V}{V - n b_m} + \frac{n}{V - n b_m} \left(b_m + n \frac{b_i - b_m}{n} \right) - \frac{1}{R T V} \left( 2 n a_m + n^2 \frac{2}{n} \left( \sum_{j=1}^{N_c} a_{ij} x_j - a_m \right) \right) \\
# & \; & - \ln Z \\
# &= \; &\ln \frac{V}{V - n b_m} + \frac{n b_i}{V - n b_m} - \frac{2 n \sum_{j=1}^{N_c} a_{ij} x_j}{R T V} - \ln Z \\
# &= \; &\ln \frac{v}{v - b_m} + \frac{b_i}{v - b_m} - \frac{2 \sum_{j=1}^{N_c} a_{ij} x_j}{R T v} - \ln Z.
# \end{alignat}$$

# <a id='pvt-eos-van_der_waals-coefficients'></a>
# Каким образом могут быть определены параметры $a$ и $b$ для чистого компонента, по которым рассчитываются с использованием правил смешивания параметры $a_m$ и $b_m$? Это можно сделать через экспериментально определенные критические давление и температуру. [Ранее](../1-TD/TD-14-PhaseEquilibrium.html#pvt-td-phase_equilibrium-critical_point) было показано, что критическая точка характеризуется следующими соотношениями:

# $$ \left( \frac{\partial P}{\partial v} \right)_{T_c} = \left( \frac{\partial^2 P}{\partial v^2} \right)_{T_c} = 0.$$

# Получим значения параметров $a$ и $b$, выраженные через критические свойства компонентов, из уравнения Ван-дер-Ваальса. Частная производная давления по молярному объему при постоянной температуре:

# $$ \left(\frac{\partial P}{\partial v} \right)_T = \frac{\partial}{\partial v}\left( \frac{R T}{v - b} - \frac{a}{v^2} \right)_T = - \frac{RT}{\left( v - b \right)^2} + \frac{2 a}{v^3} = 0. $$

# $$ a = \frac{R T_с v_с^3}{2 \left( v_с - b \right)^2}.$$

# Вторая частная производная давления по молярному объему при постоянной температуре:

# $$ \left(\frac{\partial^2 P}{\partial v^2} \right)_T = \frac{\partial}{\partial v} \left( \frac{\partial P}{\partial v} \right)_T = \frac{\partial}{\partial v} \left( - \frac{RT}{\left( v - b \right)^2} + \frac{2 a}{v^3} \right)_T = \frac{2 R T}{\left( v - b \right)^3} - \frac{6 a}{v^4} = 0. $$

# $$ b = \frac{v_c}{3}. $$

# $$ a = \frac{9 R T_c v_c}{8}. $$

# Поскольку наиболее часто критическую точку характеризуют давлением и температурой, то выразим параметры $a$ и $b$ через критические давление и температуру компонента. Для этого запишем уравнение состояния Ван-дер-Ваальса для критической точки:

# $$ \left( P_c + \frac{a}{v_c^2} \right) \left( v_c - b \right) = R T_c.$$

# Подставив в данное уравнение полученные ранее значения параметров $a$ и $b$, получим:

# $$ v_c = \frac{3 R T_c}{8 P_c}.$$

# С учетом этого, значения параметров $a$ и $b$:

# $$ \begin{align}
# a &= \frac{27}{64} \frac{R^2 T_c^2}{P_c}; \\
# b &= \frac{1}{8} \frac{R T_c}{P_c}.
# \end{align} $$

# Еще одной неизвестной для расчета логарифма коэффициента летучести $i$-го компонента является коэффициент сверхсжимаемости $Z$, определяемый выражением:

# $$Z = \frac{P V}{n R T} = \frac{P v}{R T}.$$

# Если о системе известны молярный объем $v$ и температура $T$, тогда коэффициент сверхсжимаемости по уравнению Ван-дер-Ваальса:

# $$ Z = \frac{v}{v - b} - \frac{a}{R T v}.$$

# Если о системе известны давление $P$ и температура $T$, тогда уравнение Ван-дер-Ваальса можно записать в следующем виде:

# $$ Z^3 - \left( 1 + B \right) Z^2 + A Z - A B = 0. $$

# $$ A = \frac{a_m P}{R^2 T^2}. $$

# $$ B = \frac{b_m P}{R T}. $$

# Решая кубическое уравнение относительно $Z$, например, с использованием [формулы Кардано](https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula) или численными методами решения нелинейных уравнений, определяется значение коэффициента сверхсжимаемости по уравнению Ван-дер-Ваальса.

# <a id='pvt-eos-van_der_waals-fugacity_coeff-pt'></a>
# С учетом этого выражение для логарифма коэффициента летучести можно преобразовать следующим образом:

# $$ \ln \phi_i = - \ln \left( Z - B \right) + \frac{b_i}{b_m} \frac{B}{Z - B} + \frac{2 A}{Z a_m} \sum_{j=1}^{N_c} a_{ij} x_j.$$

# <a id='pvt-eos-van_der_waals-root_selection'></a>
# Решение кубического уравнения может привести к появлению более одного корня. Поскольку система должна характеризоваться одним значением параметра, то из нескольких значений необходимо выбрать тот, который имеет физический смысл. Это можно сделать, сравнивая между собой энергии Гиббса, рассчитываемые для данного коэффициента сверхсжимаемости. Тогда нужный корень будет характеризоваться наименьшим значением энергии Гиббса. [Ранее](../1-TD/TD-10-MixtureGibbsEnergy.html#pvt-td-mixture_gibbs_energy) было показано, что энергия Гиббса многокомпонентной системы может быть рассчитана следующим образом:

# $$ G = \sum_{i=1}^{N_c} \mu_i x_i.$$

# Данное выражение применимо при постоянных давлении и температуре. [Дифференциал химического потенциала компонента](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity):

# $$ d \mu_i = R T d \ln f_i.$$

# Проинтегрируем данное выражение, при этом, $\mu_0$ будет соответствовать $P \rightarrow 0$. Тогда:

# $$ \mu_i - {\mu_0}_i = R T \ln f_i. $$

# Тогда энергия Гиббса:

# $$ G = \sum_{i=1}^{N_c} x_i {\mu_0}_i + R T \sum_{i=1}^{N_c} x_i \ln f_i. $$

# Пусть $Z_1$ и $Z_2$ – корни кубического уравнения состояния, которым соответствуют энергии Гиббса $G_1$ и $G_2$. Тогда:

# $$ G_1 - G_2 = R T \left( \sum_{i=1}^{N_c} x_i \ln {f_1}_i - \sum_{i=1}^{N_c} x_i \ln {f_2}_i \right). $$

# Логарифм летучести можно получить из логарифма коэффициента летучести:

# $$ \ln f_i = \ln \phi_i + \ln x_i P.$$

# Преобразуем сумму произведений мольной доли компонента и логарифма летучести:

# $$ \begin{align}
# \sum_{i=1}^{N_c} x_i \ln f_i
# &= \sum_{i=1}^{N_c} x_i \ln \phi_i + \sum_{i=1}^{N_c} x_i \ln x_i P \\
# &= - \sum_{i=1}^{N_c} x_i \ln \left( Z - B \right) + \sum_{i=1}^{N_c} x_i \frac{b_i}{b_m} \frac{B}{Z - B} + \sum_{i=1}^{N_c} x_i \frac{2 A}{Z a_m} \sum_{j=1}^{N_c} a_{ij} x_j + \sum_{i=1}^{N_c} x_i \ln x_i P \\
# &= - \ln \left( Z - B \right) \sum_{i=1}^{N_c} x_i + \frac{1}{b_m} \frac{B}{Z - B} \sum_{i=1}^{N_c} x_i b_i + \frac{2 A}{Z a_m} \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} a_{ij} x_i x_j + \sum_{i=1}^{N_c} x_i \ln x_i P \\
# &= - \ln \left( Z - B \right) + \frac{B}{Z - B} + \frac{2 A}{Z} + \sum_{i=1}^{N_c} x_i \ln x_i P.
# \end{align} $$

# С учетом этого разность энергий Гиббса:

# $$ \begin{align} G_1 - G_2 &= R T \left( - \ln \left( Z_1 - B \right) + \frac{B}{Z_1 - B} + \frac{2 A}{Z_1} + \ln \left( Z_2 - B \right) - \frac{B}{Z_2 - B} - \frac{2 A}{Z_2} \right) \\
# &= R T \left(\ln \frac{Z_2 - B}{Z_1 - B} + B \frac{Z_2 - Z_1}{\left( Z_1 - B \right) \left( Z_2 - B \right)} + 2 A \frac{Z_2 - Z_1}{Z_1 Z_2} \right). \end{align}$$

# Следовательно, если $G_1 - G_2 > 0,$ то коэффициент сверхсжимаемости системы равен $Z_2$.

# <a id='pvt-eos-van_der_waals-exercise'></a>
# Теперь можно рассмотреть применение уравнения состояния Ван-дер-Ваальса, например, для нахождения летучестей компонентов в смеси метана и диоксида углерода с мольными долями $0.85$ и $0.15$ для давления $20 \; бар$ и температуры $40 \; \unicode{xB0} C$.

# In[8]:


Pc = np.array([7.37646, 4.600155]) * 10**6
Tc = np.array([304.2, 190.6])
z = np.array([0.15, 0.85])
dij = np.array([[0, 0.025], [0.025, 0]])
R = 8.314


# In[9]:


class mix_rules_vdw(core):
    def __init__(self, z, Pc, Tc, n=1, dij=None, calc_der=False):
        self.z = z
        self.n = n
        self.ai = 27 * R**2 * Tc**2 / (64 * Pc)
        self.bi = R * Tc / (8 * Pc)
        self.aij = np.outer(self.ai, self.ai)**0.5
        self.dij = dij
        if dij is not None:
            self.aij = self.aij * (1 - self.dij)
        self.am = np.sum(np.outer(self.z, self.z) * self.aij)
        self.bm = np.sum(self.z * self.bi)
        if calc_der:
            self.damdn = 2 * (np.sum(self.aij * self.repeat(self.z, 0), 1) - self.am) / self.n
            self.damdn = (self.bi - self.bm) / self.n
        pass


# In[10]:


mr = mix_rules_vdw(z, Pc, Tc)


# In[11]:


class eos_vdw(core):
    def __init__(self, mr, T, P=None, v=None):
        self.mr = mr
        self.T = T
        if v is not None:
            self.v = v
            self.Z = self.calc_Z_V()
            self.P = self.Z * R * T / v
            self.lnphi = self.calc_fug_coef_V()
        elif P is not None:
            self.P = P
            self.A = mr.am * P / (R**2 * T**2)
            self.B = mr.bm * P / (R * T)
            self.Z = self.calc_Z_P()
            self.v = self.Z * R * T / P
            self.lnphi = self.calc_fug_coef_P()
        self.lnf = self.lnphi + np.log(mr.z * self.P)
        pass

    def calc_Z_V(self):
        return self.v / (self.v - self.mr.bm) - self.mr.am / (R * self.T * self.v)

    def calc_Z_P(self):
        Zs = self.calc_cardano(-(1 + self.B), self.A, -self.A * self.B)
        Z = Zs[0]
        if len(Zs) > 1:
            for i in range(1, 3):
                if self.calc_dG(Z, Zs[i]) > 0:
                    Z = Zs[i]
        return Z

    def calc_fug_coef_V(self):
        return np.log(self.v / (self.v - self.mr.bm)) + self.mr.bi / (self.v - self.mr.bm) -                2 * np.sum(self.mr.aij * self.repeat(self.mr.z, 0), 1) / (R * self.T * self.v) - np.log(self.Z)

    def calc_fug_coef_P(self):
        return -np.log(self.Z - self.B) + self.mr.bi * self.B / (self.mr.bm * (self.Z - self.B)) -                2 * self.A * np.sum(self.mr.aij * self.repeat(self.mr.z, 0), 1) / (self.Z * self.mr.am)

    def calc_dG(self, Z1, Z2):
        return np.log((Z2 - self.B) / (Z1 - self.B)) + self.B * (Z2 - Z1) / ((Z1 - self.B) * (Z2 - self.B)) +                2 * self.A * (Z2 - Z1) / (Z1 * Z2)


# Результаты расчета коэффициента сверхсжимаемости газа с использованием уравнения состояния Ван-дер-Ваальса:

# In[12]:


eos = eos_vdw(mr, T=40+273.15, P=20*10**5)
eos.Z


# Аналогичный результат может быть получен при расчете коэффициента сверхсжимаемости газа относительно известных температуры и молярного объема:

# In[13]:


eosv = eos_vdw(mr, T=40+273.15, v=eos.v)
eosv.Z


# Значения логарифма летучести компонентов, рассчитанные относительно известных давления и температуры:

# In[14]:


eos.lnf


# Значения логарифма летучести компонентов, рассчитанные относительно известных молярного объема и температуры:

# In[15]:


eosv.lnf


# Однако использование уравнения состояния Ван-дер-Ваальса может приводить к существенному расхождению с фактическими данными. Например, с учетом ранее полученных соотношений, согласно уравнению состояния Ван-дер-Ваальса коэффиицент сверхсжимаемости глюбого газа в сверхкритическом состоянии составляет $\frac{3}{8} = 0.375 .$ При этом, для большинства реальных газов коэффициент сверхсжимаемости в критическом состоянии [составляет](https://phys.libretexts.org/Bookshelves/Thermodynamics_and_Statistical_Mechanics/Book%3A_Heat_and_Thermodynamics_(Tatum)/06%3A_Properties_of_Gases/6.03%3A_Van_der_Waals_and_Other_Gases) около $0.28$. Таким образом, несмотря на свою простоту, уравнение состояния Ван-дер-Ваальса практически не используется, поэтому далее уравнение состояния Ван-дер-Ваальса не будет рассматриваться. Для повышения точности расчета параметров систем на практике используются уравнения состояния Пенга-Робинсона и Суаве-Редлиха-Квонга, которые также являются двухпараметрическими уравнениями состояния.
