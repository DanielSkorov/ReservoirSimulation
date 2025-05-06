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

(pvt-esc-rr-np)=
# Уравнение Речфорда-Райса для многофазных систем

[Ранее](SEC-2-RR.md) было получено уравнение Речфорда-Райса, решение которого позволяет определить мольные доли фаз и их компонентные составы по известным константам фазового равновесия и глобальному компонентному составу. В [предыдущем разделе](SEC-3-RR-2P.md) был представлен устойчивый и эффективный алгоритм решения уравнения Речфорда-Райса для двухфазной постановки. Данный раздел посвящен подходам к решению уравнения Речфорда-Райса для многофазных систем.

Исследованию и разработке численных схем решения уравнения Речфорда-Райса в многофазной постановке были посвящены работы \[[Leibovici and Nichita, 2008](https://doi.org/10.1016/j.fluid.2008.03.006); [Okuno et al, 2010](https://doi.org/10.2118/117752-PA); [Pan et al, 2020](https://doi.org/10.1016/j.petrol.2020.108150); [Haugen and Firoozabadi, 2010](https://doi.org/10.1002/aic.12452); [Li and Firoozabadi, 2012](https://doi.org/10.1016/j.fluid.2012.06.021); [Iranshahr et al, 2010](https://doi.org/10.1016/j.fluid.2010.09.022)\] и другие.

Стоит отметить, что в подавляющем большинстве случаев при разработке месторождений углеводородов использование двухфазной постановки *(VLE – vapour-liquid equilibrium)* при расчете равновесного состояния достаточно и оправдано. В этом случае пренебрегается растворимостью воды, как компонента, в газовой и жидкой углеводородных фазах, а также растворимостью других компонентов в водной фазе, а равновесное состояние рассматривается только между газовой и жидкой углеводородными фазами. На крайний случай, учитывают растворимость компонентов в водной фазе индивидуально по [закону Генри](https://en.wikipedia.org/wiki/Henry%27s_law) при моделировании закачки, например, диоксида углерода \[[Agarwal et al, 1993](https://doi.org/10.2118/93-02-03)\]. Необходимость решения уравнения Речфорда-Райса в многофазной постановке и соответственно проведения многофазного расчета равновесного состояния возникает, например, при разработке месторождений нефти с аномально низкой пластовой температурой газовыми методами: в этом случае в процессе обмена компонентами может наблюдаться формирование двух жидких углеводородных фаз *(VLLE – vapour-liquid-liquid equilibrium)*. Либо, наоборот, при высоких температурах, когда взаимная растворимость воды и углеводородных компонентов увеличивается. Существует также и ряд других процессов и технологий разработки месторождений углеводородов, для которых многофазный расчет равновесного состояния играет в той или иной степени значимую роль.

Дальнейшее изложение анализа многофазной постановки уравнения Речфорда-Райса и численного метода решения будет основано на работе \[[Okuno et al, 2010](https://doi.org/10.2118/117752-PA)\].

Для многофазной ($N_p$ – количество фаз), многокомпонентной ($N_c$ – количество компонентов) системы, компонентный состав которой задан вектором $\mathbf{y} \in {\rm I\!R}^{N_c}$, система $\left( N_p - 1 \right)$ уравнений Речфорда-Райса записывается следующим образом:

$$ \sum_{i=1}^{N_c} \frac{y_i \left( K_{ji} - 1 \right)}{\sum_{k=1}^{N_p-1} f_k \left( K_{ki} - 1 \right) + 1} = 0, \; j = 1 \, \ldots \, N_p - 1, $$

где $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ – матрица размерностью $\left( N_p - 1 \right) \times N_c$, а вектор $\mathbf{f} \in {\rm I\!R}^{N_p - 1}$ представляет собой вектор основных переменных – мольных долей фаз.

Преобразуем эту систему уравнений к следующему виду:

$$ r_j \left( \mathbf{f} \right) = \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right)}{1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right)} = 0, \; j = 1 \, \ldots \, N_p - 1, $$

В векторном виде:

$$ \mathbf{r} = \left( 1 - \mathbf{K} \right) \left( \mathbf{y} \oslash \left(1 - \left( 1 - \mathbf{K} \right)^\top \mathbf{f} \right) \right) = 0. $$

В выражении выше символ $\oslash$ обозначает поэлементное деление ([деление Адамара](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#Analogous_operations)) двух векторов-столбцов, в результате которого также получается вектор-столбец.

Данная система представляет собой систему нелинейных уравнений, для решения (нахождения корня) которой обычно используют [метод Ньютона](https://en.wikipedia.org/wiki/Newton%27s_method#Multidimensional_formulations). Для его применения необходимо составить [якобиан](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) – матрицу $\mathbf{J} \in {\rm I\!R}^{\left( N_p - 1 \right) \times \left( N_p - 1 \right)}$ производных вектора функций $\mathbf{r} \in {\rm I\!R}^{N_p - 1}$ по вектору основных переменных:

$$ \mathbf{J} = \begin{bmatrix}
\frac{\partial r_1}{\partial f_1} & \frac{\partial r_1}{\partial f_2} & \ldots & \frac{\partial r_1}{\partial f_{N_p-1}} \\
\frac{\partial r_2}{\partial f_1} & \frac{\partial r_2}{\partial f_2} & \ldots & \frac{\partial r_2}{\partial f_{N_p-1}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial r_{N_p-1}}{\partial f_1} & \frac{\partial r_{N_p-1}}{\partial f_2} & \ldots & \frac{\partial r_{N_p-1}}{\partial f_{N_p-1}}
\end{bmatrix}. $$

Получим выражение для элемента матрицы якобиана:

$$ \begin{align}
\frac{\partial r_j}{\partial f_l}
&= \frac{\partial}{\partial f_l} \left( \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right)}{1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right)} \right) \\
&= \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right) \left( \sum_{k=1}^{N_p-1} \frac{\partial f_k}{\partial f_l} \left( 1 - K_{ki} \right) \right)}{\left( 1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right) \right)^2} \\
&= \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right) \left( 1 - K_{li} \right)}{\left( 1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right) \right)^2} .
\end{align} $$

Пусть

$$ t_i = \left( 1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right) \right), $$

тогда выражение для элемента якобиана примет вид:

$$ \frac{\partial r_j}{\partial f_l} = \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right) \left( 1 - K_{li} \right)}{t_i^2}. $$

Стоит отметить, что полученный якобиан симметричен. Действительно,

$$ \begin{align}
\frac{\partial r_j}{\partial f_l} &= \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right) \left( 1 - K_{li} \right)}{t_i^2}, \\
\frac{\partial r_l}{\partial f_j} &= \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{li} \right) \left( 1 - K_{ji} \right)}{t_i^2}.
\end{align} $$

Значение основных переменных на $\left( k + 1 \right)$-й итерации метода Ньютона для нахождения корня системы нелинейных уравнений определяются следующим образом:

$$ \mathbf{f}_{k+1} = \mathbf{f}_k - \mathbf{J}^{-1} \mathbf{r}_k. $$

С другой стороны, аналогичное выражение для итерации можно записать, если рассматривать данную задачу с точки зрения минимизации некоторой функции $F \left( \mathbf{f} \right) \, : \, {\rm I\!R}^{\left( N_p - 1 \right)} \rightarrow {\rm I\!R}$. В этом случае выражение для [минимизации методом Ньютона](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization#Higher_dimensions):

$$ \mathbf{f}_{k+1} = \mathbf{f}_k - \mathbf{H}^{-1} \nabla F \left( \mathbf{f} \right), $$

где матрица $\mathbf{H} \in {\rm I\!R}^{\left( N_p - 1 \right) \times \left( N_p - 1 \right)}$ представляет собой матрицу вторых частных производных функции $F \left( \mathbf{f} \right)$, называемую [гессианом](https://en.wikipedia.org/wiki/Hessian_matrix), а вектор $\nabla F \left( \mathbf{f} \right)$ – вектор частных производных функции $F \left( \mathbf{f} \right)$, называемый [градиентом](https://en.wikipedia.org/wiki/Gradient). При этом эта функция должна быть дважды дифференцируема, а гессиан – симметричной матрицей. Соответственно, поскольку $\mathbf{J}$ также является симметричной матрицей, мы можем поставить знак равенства между двумя этими выражениями. Найдем выражение для функции $F \left( \mathbf{f} \right)$. Для этого необходимо проинтегрировать выражение для элемента градиента этой функции, равного элементу вектора $\mathbf{r}$.

$$ \begin{align}
F \left( \mathbf{f} \right)
&= \int \mathbf{r}^\top \left( \mathbf{f} \right) \, \mathrm{d} \mathbf{f} \\
&= \int \sum_{j=1}^{N_p-1} r_j \left( \mathbf{f} \right) \, \mathrm{d} f_j \\
&= \int \sum_{j=1}^{N_p-1} \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right)}{1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right)} \, \mathrm{d} f_j \\
&= \int \sum_{i=1}^{N_c} \frac{y_i \sum_{j=1}^{N_p-1} \left( 1 - K_{ji} \right) \, \mathrm{d} f_j}{1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right)} \\
&= -\int \sum_{i=1}^{N_c} \frac{y_i \, \mathrm{d} \left(1 - \sum_{j=1}^{N_p-1} \left( 1 - K_{ji} \right) f_j \right)}{1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right)} \\
&= -\int \sum_{i=1}^{N_c} \frac{y_i \, \mathrm{d} t_i}{t_i} \\
&= - \sum_{i=1}^{N_c} y_i \int \frac{\mathrm{d} t_i}{t_i} \\
&= - \sum_{i=1}^{N_c} y_i \ln \left| t_i \right| + C,
\end{align} $$

где $C$ – некоторая константа.

Теперь покажем, что данная функция выпуклая.

````{margin}
```{admonition} Дополнительно
Необходимо отметить, что в процессе доказательства выпуклости функции $F \left( \mathbf{f} \right)$ использовались величины $\sqrt{y_i},$ что допустимо только в том случае, когда глобальные мольные доли компонентов неотрицательны. Однако в ряде задач в PVT-моделировании могут рассматриваться случаи с отрицательными мольными долями компонентов. Например, в задачах определения минимального давления смешивания \[[Wang and Orr, 1997](https://doi.org/10.1016/S0378-3812(97)00179-9)\]. В этом случае гессиан не является положительно полуопределенной матрицей, а сама функция $F \left( \mathbf{f} \right)$ не является выпуклой в чистом виде.

[comment]: <> (Для использования метода Ньютона в случае, когда гессиан не является положительно полуопределенной матрицей, допустимо применять модифицированное разложение Холецкого.)
```
````

```{admonition} Доказательство
:class: proof

Для того чтобы доказать, что функция от нескольких переменных является выпуклой, [необходимо](https://en.wikipedia.org/wiki/Convex_function#Functions_of_several_variables) доказать, что ее гессиан является [положительно полуопределенной матрицей](https://en.wikipedia.org/wiki/Definite_matrix). Матрица является [положительно полуопределенной](../../0-Math/0-LAB/LAB-8-MatrixDefiniteness.md), если все ее [собственные значения](../../0-Math/0-LAB/LAB-6-Eigenvalues-Eigenvectors.md) неотрицательны, что [равносильно](../../0-Math/0-LAB/LAB-8-MatrixDefiniteness.md) следующему неравенству:

$$ \mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0, \; \forall \, \mathbf{x} \in {\rm I\!R}^{\left( N_p - 1 \right)}. $$

Для доказательства данного неравенства для гессиана полученной ранее функции преобразуем выражение для элемента матрицы гессиана к следующему виду:

$$ \frac{\partial r_j}{\partial f_l} = \sum_{i=1}^{N_c} \frac{y_i \left( 1 - K_{ji} \right) \left( 1 - K_{li} \right)}{t_i^2} = \sum_{i=1}^{N_c} \frac{\sqrt{y_i} \left( 1 - K_{ji} \right)}{t_i} \frac{\sqrt{y_i} \left( 1 - K_{li} \right)}{t_i} = \sum_{i=1}^{N_c} P_{ji} P_{li}. $$

Таким образом, гессиан рассматриваемой функции может быть представлен в виде произведения матрицы $\mathbf{P} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и ее транспонированного вида:

$$ \mathbf{H} = \mathbf{P} \mathbf{P}^\top. $$

Тогда

$$ \mathbf{x}^\top \mathbf{H} \mathbf{x} = \mathbf{x}^\top \mathbf{P} \mathbf{P}^\top \mathbf{x} = \left( \mathbf{P}^\top \mathbf{x} \right)^\top \left( \mathbf{P}^\top \mathbf{x} \right) = \mathbf{v}^\top \mathbf{v} = \sum_{i=1}^{N_c} v_i^2 \geq 0. $$

В процессе данного преобразования было использовано свойство транспонирования произведения:

$$ \left( \mathbf{P} \mathbf{B} \right)^\top = \mathbf{B}^\top \mathbf{P}^\top, $$

доказанное [ранее](../../0-Math/1-OM/OM-0-Introduction.md).

```

Таким образом, было показано, что гессиан функции $F \left( \mathbf{f} \right)$ является положительно полуопределенной матрицей, а сама функция – выпуклой. Следовательно, задачу решения системы уравнений Речфорда-Райса можно переформулировать с точки зрения минимизации функции $F \left( \mathbf{f} \right)$. Однако, данная функция, как и в случае [двухфазной постановки](SEC-3-RR-2P.md), характеризуется наличием $N_c$ полюсов, определяемых выражением $t_i = 0, \; i = 1 \, \ldots \, N_c$. Таким образом, для устойчивого нахождения минимума функции необходимо определить [область допустимых решений](https://en.wikipedia.org/wiki/Feasible_region). Эта область, как и в случае с двухфазной постановкой, определяется *negative flash window (NF-window)*, формулируемая из условия неотрицательности мольных долей компонентов в фазах, но допускающая существование отрицательных мольных долей фаз.

Мольная доля $i$-го компонента в референсной фазе, обозначим ее $x_i$, [определяется](SEC-2-RR.md) следующим выражением:

$$ x_i = \frac{y_i}{\sum_{k=1}^{N_p-1} f^k \left( K_{ki} - 1 \right) + 1} = \frac{y_i}{1 - \sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right)} = \frac{y_i}{t_i}. $$

NF-window предполагает, что мольные доли компонентов во всех фазах находятся на отрезке $\left[0; \, 1 \right]$, следовательно:

$$ 0 \leq x_i \leq 1 \, \Rightarrow \, 0 \leq \frac{y_i}{t_i} \leq 1 \, \Rightarrow \, 0 \leq y_i \leq t_i. $$

Аналогично рассмотрим мольные доли компонентов в остальных, нереференсных, фазах:

$$ 0 \leq y_{ji} \leq 1 \, \Rightarrow \, 0 \leq K_{ji} x_i \leq 1 \, \Rightarrow \, 0 \leq K_{ji} \frac{y_i}{t_i} \leq 1 \, \Rightarrow \, 0 \leq y_i K_{ji} \leq t_i. $$

Таким образом, можно записать следующую систему неравенств, определяющую область допустимых решений:

$$ \begin{cases}
t_i \geq y_i, \; i = 1 \, \ldots \, N_c, \\
t_i \geq y_i K_{ji} \; i = 1 \, \ldots \, N_c, \; j = 1 \, \ldots \, N_p - 1.
\end{cases} $$

Подставив выражение для $t_i$ в данную систему неравенств, получим:

$$ \begin{cases}
\sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right) \leq 1 - y_i, \; i = 1 \, \ldots \, N_c, \\
\sum_{k=1}^{N_p-1} f_k \left( 1 - K_{ki} \right) \leq 1 - K_{ji} y_i,  \; i = 1 \, \ldots \, N_c, \; j = 1 \, \ldots \, N_p - 1.
\end{cases} $$

Левую часть в данных неравенствах можно представить в виде вектора, $i$-ый элемент которого определяется выражением $\mathbf{f}^\top \mathbf{a}_i$, где $\mathbf{a}_i \in {\rm I\!R}^{\left( N_p - 1 \right)}, \, i = 1 \, \ldots \, N_c,$ представляет собой $i$-ый столбец матрицы $\mathbf{A} = \left( 1 - K_{ji} \right), \, j = 1 \, \ldots \, N_p - 1, \, i = 1 \, \ldots \, N_c$. Кроме того, $i$-ыe элементы этого вектора, согласно записанной выше системе неравенств, должны быть меньше (или равны) соответствующих значений $\left( 1 - y_i \right)$ и меньше (или равны) соотвутствующих значений $\left( 1 - K_{ji} y_i \right)$ среди всех фаз, за исключением референсной, то есть $j = 1 \, \ldots \, N_p - 1$. Поскольку $i$-ыe элементы этого вектора должны быть меньше (или равны) соответствующих значений $\left( 1 - K_{ji} y_i \right)$ среди всех фаз, следовательно, они должны быть меньше или равны минимальных соответствующих значений среди вседи всех фаз, то есть $\min_j \left\{ 1 - K_{ji} y_i \right\}$. С учетом этого, систему неравенств можно записать следующим образом:

$$ \mathbf{f}^\top \mathbf{a}_i \leq b_i = \min \left\{ 1 - y_i, \, \min_j \left\{ 1 - K_{ji} y_i \right\} \right\}, \; i = 1 \, \ldots \, N_c. $$

Таким образом, решение системы уравнений Речфорда-Райса можно эквивалентно заменить задачей оптимизации:

$$ \begin{alignat}{1}
\mathrm{min} & \;\;\;\;\;\;\; && F \left( \mathbf{f} \right) \\
\mathrm{subject\,to} &&& \mathrm{f}^\top \mathrm{a}_i \leq b_i, \, i = 1 \, \ldots \, N_c
\end{alignat} $$

Стоит отметить, что полученная область допустимых решений, определяемая $\mathbf{f}^\top \mathbf{a}_i \leq b_i, \; i = 1 \, \ldots \, N_c$, не содержит полюсы ($t_i = 0, \, i = 1 \, \ldots \, N_c$) и меньше области допустимых решений, используемой в работе \[[Leibovici and Nichita, 2008](https://doi.org/10.1016/j.fluid.2008.03.006)\] и определямой условием $t_i \geq 0$.

Прежде чем переходить к формулированию алгоритма, рассмотрим следующий пример.

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
import numpy as np
Kji = np.array([
    [2.64675, 1.16642, 1.25099E-03],
    [1.83256, 1.64847, 1.08723E-02],
]) # K-values
yi = np.array([0.3, 0.4, 0.3]) # Global component composition
```

Необходимо отобразить область допустимых решений для задачи оптимизации функции $F \left( \mathbf{f} \right)$.

````

```{code-cell} python
:tags: [remove-cell]
import numpy as np
Kji = np.array([
    [2.64675, 1.16642, 1.25099E-03],
    [1.83256, 1.64847, 1.08723E-02],
]) # K-values
yi = np.array([0.3, 0.4, 0.3]) # Global component composition
```

В соответствии с изложенным выше, область допустимых решений определяется следующим выражением:

$$ \mathbf{f}^\top \mathbf{a}_i \leq b_i, \; i = 1 \, \ldots \, N_c. $$

Для трехфазного трехкомпонентного случая данная система неравенств преобразуется в:

$$ \begin{cases}
f_1 \left( 1 - K_{11} \right) + f_2 \left( 1 - K_{21} \right) \leq b_1, \\
f_1 \left( 1 - K_{12} \right) + f_2 \left( 1 - K_{22} \right) \leq b_2, \\
f_1 \left( 1 - K_{13} \right) + f_2 \left( 1 - K_{23} \right) \leq b_3. \\
\end{cases} $$

Для рассматриваемого примера значения матрицы $\mathbf{A} = \left( 1 - \mathbf{K} \right)$:

```{code-cell} python
Aji = 1. - Kji
Aji
```

С учетом знаков $\left( 1 - \mathbf{K}_{2,i} \right), \, i = 1 \, \ldots \, N_c,$ преобразуем систему неравенств для рассматриваемого примера к следующему виду:

$$ \begin{cases}
f_2 \geq - \frac{1 - K_{11}}{1 - K_{21}} f_1 + \frac{b_1}{1 - K_{21}}, \\
f_2 \geq - \frac{1 - K_{12}}{1 - K_{22}} f_1 + \frac{b_2}{1 - K_{22}}, \\
f_2 \leq - \frac{1 - K_{13}}{1 - K_{23}} f_1 + \frac{b_3}{1 - K_{23}}. \\
\end{cases} $$

Также в соответствии с изложенным выше значения вектора $\mathbf{b} \in {\rm I\!R}^{N_c}$ определяются следующими выражениями:

$$
b_1 = \min \left\{ 1 - y_1, \, \min \left\{ 1 - K_{11} y_1, \, 1 - K_{21} y_1 \right\} \right\}, \\
b_2 = \min \left\{ 1 - y_2, \, \min \left\{ 1 - K_{12} y_2, \, 1 - K_{22} y_2 \right\} \right\}, \\
b_3 = \min \left\{ 1 - y_3, \, \min \left\{ 1 - K_{13} y_3, \, 1 - K_{23} y_3 \right\} \right\}. \\
$$

Получим значения элементов вектора $\mathbf{b}$:

```{code-cell} python
bi = np.min([np.min(1. - Kji * yi, axis=0), 1. - yi], axis=0)
```

Тогда прямые, ограничивающие область допустимых решений:

```{code-cell} python
f1 = np.array([-3., 4.])
f2 = (-Aji[0] / Aji[1])[:,None] * f1 + (bi / Aji[1])[:,None]
```

Кроме того, вычислим значения функции $F \left( \mathbf{f} \right)$ для рассматриваемой задачи с целью последующего отображения на графике в виде контуров:

```{code-cell} python
fs1 = np.linspace(-3., 4., 100, endpoint=True)
fs2 = np.linspace(-3., 4., 100, endpoint=True)
fssj = np.dstack(np.meshgrid(fs1, fs2))
tssi = 1. - fssj.dot(Aji)
Fss = - np.log(np.abs(tssi)).dot(yi)
```

Как было отмечено ранее, эта функция характеризуется наличием полюсов, определяемых уравнением $t_i = 0, \, i = 1 \, \ldots \, N_c$. Получим координаты прямых, представляющих собой эти полюсы:

```{code-cell} python
p2 = (-Aji[0] / Aji[1])[:,None] * f1 + (1. / Aji[1])[:,None]
```

Представим все это в графическом виде:

```{code-cell} python
from matplotlib import pyplot as plt
fig1, ax1 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax1.fill_between(f1, f2[0], [4., 4.], color='b', alpha=0.2, zorder=3)
ax1.fill_between(f1, f2[1], [4., 4.], color='m', alpha=0.2, zorder=3)
ax1.fill_between(f1, f2[2], [-3., -3.], color='g', alpha=0.2, zorder=3)
ax1.plot(f1, f2[0], lw=2.5, c='b', zorder=4, label='Ограничение #1')
ax1.plot(f1, f2[1], lw=2.5, c='m', zorder=4, label='Ограничение #2')
ax1.plot(f1, f2[2], lw=2.5, c='g', zorder=4, label='Ограничение #3')
ax1.plot(f1, p2[0], ls='--', lw=2.5, c='b', zorder=4, label='Полюс #1')
ax1.plot(f1, p2[1], ls='--', lw=2.5, c='m', zorder=4, label='Полюс #2')
ax1.plot(f1, p2[2], ls='--', lw=2.5, c='g', zorder=4, label='Полюс #3')
ax1.plot(0.1625, 0.1257, 'o', mec='r', mfc='r', ms=4., lw=0., zorder=4, label='Решение')
ax1.contour(fssj[:,:,0], fssj[:,:,1], Fss, 15, linewidths=0.5, zorder=2)
ax1.set_xlabel('$f_1$')
ax1.set_xlim(-3., 4.)
ax1.set_ylabel('$f_2$')
ax1.set_ylim(-3., 4.)
ax1.legend(loc=1, fontsize=9)
ax1.grid(zorder=1)
```

Серый треугольник в центре, ограниченный сплошными прямыми является областью допустимых решений, определяемой выражением $\mathbf{f}^\top \mathbf{a}_i \leq b_i, \, i = 1 \, \ldots \, N_c$, причем эта область не содержит полюсы и несколько меньше области допустимых решений (треугольника), ограниченной пунктирными линиями и определяемой выражением $t_i \geq 0, \, i = 1 \, \ldots \, N_c$.

Рассмотрим одну итерацию оптимизации функции $F \left( \mathbf{f} \right)$ для рассматриваемого примера:

$$ \mathbf{f}_{k+1} = \mathbf{f}_k - \lambda \mathbf{H}^{-1} \nabla F \left( \mathbf{f}_k \right). $$

Пусть длина шага $\lambda = 1$, а значения мольных долей фаз на $k$-й итерации $\mathbf{f}_k$ заданы следующим вектором:

```{code-cell} python
fjk = np.array([-0.75, 1.35])
```

В этой точке значение функции $F \left( \mathbf{f}_k \right)$:

```{code-cell} python
ti = 1. - fjk.dot(Aji)
F = - np.log(np.abs(ti)).dot(yi)
F
```

Определим значения вектора мольных долей фаз на $\left( k + 1 \right)$-й итерации. Для этого вычислим значения градиента минимизируемой функции:

```{code-cell} python
dFdfj = Aji.dot(yi / ti)
dFdfj
```

Получим значения гессиана:

```{code-cell} python
Pji = np.sqrt(yi) / ti * Aji
Hjl = Pji.dot(Pji.T)
Hjl
```

Определим направление оптимизации $\Delta \mathbf{f} = -\mathbf{H}^{-1} \nabla F \left( \mathbf{f}_k \right)$:

```{code-cell} python
dfj = -np.linalg.inv(Hjl).dot(dFdfj)
dfj
```

Тогда значения мольных долей фаз на $\left( k + 1 \right)$-й итерации:

```{code-cell} python
fjkp1 = fjk + dfj
fjkp1
```

Отобразим эту итерацию графически:

```{code-cell} python
fig2, ax2 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax2.plot(f1, f2[0], lw=2.5, c='b', zorder=2, label='Ограничение #1')
ax2.plot(f1, f2[1], lw=2.5, c='m', zorder=2, label='Ограничение #2')
ax2.plot(f1, f2[2], lw=2.5, c='g', zorder=2, label='Ограничение #3')
ax2.plot(f1, p2[0], ls='--', lw=2.5, c='b', zorder=2, label='Полюс #1')
ax2.plot(f1, p2[1], ls='--', lw=2.5, c='m', zorder=2, label='Полюс #2')
ax2.plot(f1, p2[2], ls='--', lw=2.5, c='g', zorder=2, label='Полюс #3')
ax2.plot(0.1625, 0.1257, 'o', mec='r', mfc='r', ms=4., lw=0., zorder=4, label='Решение')
ax2.plot(*fjk, 'o', mec='c', mfc='c', ms=4., lw=0., zorder=4, label='$f_j^k$')
ax2.plot(*fjkp1, 'o', mec='m', mfc='m', ms=4., lw=0., zorder=4,
         label=r'$f_j^{k+1}$ $\left(\lambda = 1\right)$')
ax2.annotate('', fjk, fjkp1, arrowprops=dict(arrowstyle='<-'))
ax2.set_xlabel('$f_1$')
ax2.set_xlim(-2., 2.)
ax2.set_ylabel('$f_2$')
ax2.set_ylim(-2., 3.)
ax2.legend(loc=1, fontsize=9)
ax2.grid(zorder=1)
```

Из данного графика видно, что при несколько большей длине шага в этом же направлении можно еще лучше приблизиться к искомому решению – минимуму функции $F \left( \mathbf{f} \right)$. Действительно, построим зависимость этой функции от длины шага в выбранном направлении. Зададимся диапазоном значений длины шага:

```{code-cell} python
lmbds = np.linspace(0., 2., 100, endpoint=True)
```

Вычислим значения функции $F \left( \mathbf{f} \right)$ для каждой длины шага итерации:

```{code-cell} python
fsjkp1 = fjk + dfj * lmbds[:,None]
tsi = 1. - fsjkp1.dot(Aji)
Fs = - np.log(np.abs(tsi)).dot(yi)
```

Построим график:

```{code-cell} python
fig3, ax3 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax3.plot(lmbds, Fs, lw=2., c='y', zorder=2)
ax3.set_xlim(0., 2.)
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylim(-0.04, 0.1)
ax3.set_ylabel(r'$F \left( \lambda \right)$')
ax3.grid(zorder=1)
```

Видно, что при $\lambda \approx 1.25$ значения функции меньше всего. Таким образом, ньютоновскую итерацию можно дополнить *процедурой поиска длины шага (line search)*, исходя из условия минимизации искомой функции для заданного направления. При этом необходимо соблюдать ограничения области допустимых решений. Это позволит, с одной стороны, сократить количество ньютоновских итераций, с другой, поможет избежать "перелета" в тех случаях, когда минимум функции близок к границам области допустимых решений. Получим максимальное значение длины шага. Для этого подставим выражение для мольных долей фаз на $\left( k + 1 \right)$-й итерации $\mathbf{f}_{k+1} = \mathbf{f}_k + \Delta \mathbf{f}$ в систему $N_c$ неравенств, определяющих область допустимых решений:

$$ \begin{align}
\mathbf{f}_{k+1}^\top \mathbf{a}_i & \leq b_i, \; i = 1 \, \ldots \, N_c, \\
\left( \mathbf{f}_k + \lambda \Delta \mathbf{f} \right)^\top \mathbf{a}_i & \leq b_i, \; i = 1 \, \ldots \, N_c, \\
\mathbf{f}_k^\top \mathbf{a}_i + \lambda \Delta \mathbf{f}^\top \mathbf{a}_i & \leq b_i, \; i = 1 \, \ldots \, N_c, \\
\lambda \Delta \mathbf{f}^\top \mathbf{a}_i & \leq b_i - \mathbf{f}_k^\top \mathbf{a}_i, \; i = 1 \, \ldots \, N_c. \\
\end{align} $$

С учетом знака $i$-ого значения $\Delta \mathbf{f}^\top \mathbf{a}_i, \, i = 1 \, \ldots \, N_c,$ данная система неравенств преобразуется к системе двойных неравенств, определяющих допустимые значения длины шага, исходя из условий области допустимых решений. Поскольку величина длины шага должна быть больше всех значений $i$, удовлетворяющих условию $\Delta \mathbf{f}^\top \mathbf{a}_i < 0$, то величина длины шага должна быть больше максимального среди всех значений, удовлетворяющих данному условию. Поскольку величина длины шага должна быть меньше всех значений $i$, удовлетворяющих условию $\Delta \mathbf{f}^\top \mathbf{a}_i > 0$, то величина длины шага должна быть меньше минимального среди всех значений, удовлетворяющих данному условию. Иными словами, длина шага итерации находится на следующем отрезке:

$$ \max_i \left\{ \frac{b_i - \mathbf{f}_k^\top \mathbf{a}_i}{\Delta \mathbf{f}^\top \mathbf{a}_i} \, : \, \Delta \mathbf{f}^\top \mathbf{a}_i < 0 \right\} \leq \lambda \leq \min_i \left\{ \frac{b_i - \mathbf{f}_k^\top \mathbf{a}_i}{\Delta \mathbf{f}^\top \mathbf{a}_i} \, : \, \Delta \mathbf{f}^\top \mathbf{a}_i > 0 \right\}. $$

Символ двоеточия в данном выражении заменяет фразу "такие, что" или "удовлетворяющие условию".

Получим минимальное и максимальное значения длины шага для рассматриваемого примера:

```{code-cell} python
lmbdi = (bi - fjk.dot(Aji)) / (dfj.dot(Aji))
where_max = dfj.dot(Aji) < 0.
where_min = np.logical_not(where_max)
np.max(lmbdi[where_max]), np.min(lmbdi[where_min])
```

````{margin}
```{admonition} Дополнительно
Необходимо отметить, что существуют также безградиентные методы оптимизации функции, которые могут успешно применяться для рассматриваемой задачи. К таковым, например, относится [метод Брента](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html). Однако зачастую такие методы требуют значительно больше количество итераций для достижения той же точности, что и метод Ньютона.

Кроме того, для процедуры поиска длины шага нет необходимости с большой точностью определять минимум функции – достаточно найти такое значение $\lambda$ при котором значение функции на следующей итерации будет достачным для удовлетворения определенных *условий*. Зачастую такие условия включают условие необходимой степени минимизации функции и условие, не допускающее выполнение слишком малых шагов. К таковым, например, относятся [условия Вольфе](https://en.wikipedia.org/wiki/Wolfe_conditions) и [условия Голдштейна](https://en.wikipedia.org/wiki/Backtracking_line_search). Первые чаще применяются при использовании квази-Ньютоновских методов оптимизации, вторые – для метода Ньютона. Стоит отметить, что неточное нахождение длины шага итерации может приводить к увеличению количества ньютоновских итераций основного цикла (однако для рассматриваемой в рамках данного раздела задачи это увеличение составляет всего 1 – 2 ньютоновские итерации). Подробнее процедура поиска длины шага рассматривалась в [разделе](../../0-Math/1-OM/OM-0-Introduction.md), посвященном методам оптимизации функций.
```
````

Для минимизации функции $F \left( \lambda \right)$ также можно использовать метод Ньютона:

$$ \lambda_{l+1} = \lambda_l - \frac{F'_\lambda \left( \lambda_l \right)}{F''_{\lambda \lambda} \left( \lambda_l \right)}. $$

Значения первой и второй производных получены с использованием [правила нахождения производной сложной функции](https://en.wikipedia.org/wiki/Chain_rule):

$$ \begin{align}
F'_\lambda &= \nabla F^\top \Delta \mathbf{f}, \\
F''_{\lambda \lambda} &= \Delta \mathbf{f}^\top \mathbf{H} \Delta \mathbf{f}.
\end{align} $$

Стоит отметить, что выполнение такой процедуры поиска оптимальной длины шага на каждой итерации может быть излишне ресурсоемким, поскольку предполагает итеративное вычисление градиента и гессиана оптимизируемой функции. Поэтому авторы работы \[[Okuno et al, 2010](https://doi.org/10.2118/117752-PA)\] предлагают более консервативное применение процедуры поиска оптимальной длины только в том случае, когда искомое решение находится вблизи границ области допустимых решений, и в процессе приближения к нему есть риск выйти за эти границы. Учитывая выпуклость оптимизируемой функции, для проверки данного случая можно вычислить знак производной $F'_\lambda$ для начального приближения $\lambda = 1$. Если $F'_{\lambda=1} > 0$ (то есть если функция $F \left( \lambda \right)$ при $\lambda = 1$ возрастающая, что возможно в том случае, когда при $\lambda = 1$ произойдет "перелет"), то осуществляется коррекция длины шага, исходя из условия минимизации. В противном случае, когда $F'_{\lambda=1} \leq 0$ процедура *line search* не выполняется, и принимается $\lambda = 1$. Однако такая проверка все равно может привести к "перелету" в том случае, когда шаг итерации с длиной $\lambda = 1$ пересекает полюс функции. Для иллюстрации данного сценария рассмотрим следующий пример \[[Okuno et al, 2010](https://doi.org/10.2118/117752-PA)\].

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002],
    [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831],
]) # K-values
yi = np.array(
    [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
) # Global component composition
```

Необходимо отобразить область допустимых решений для задачи оптимизации функции $F \left( \mathbf{f} \right)$ и выполнить первую итерацию оптимизации методом Ньютона для начального приближения:

``` python
fjk = np.array([0.33699, 0.4512]) # Initial estimate
```

````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002],
    [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831],
]) # K-values
yi = np.array(
    [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
) # Global component composition
fjk = np.array([0.33699, 0.4512]) # Initial estimate
```

Вычислим значения матрицы $\mathbf{A}$ и вектора $\mathbf{t}$ для начального приближения:

```{code-cell} python
Aji = 1. - Kji
ti = 1. - fjk.dot(Aji)
```

Тогда градиент функции $\nabla F \left( \mathbf{f}_k \right)$:

```{code-cell} python
dFdfj = Aji.dot(yi / ti)
dFdfj
```

И гессиан функции $\mathbf{H}$:

```{code-cell} python
Pji = np.sqrt(yi) / ti * Aji
Hjl = Pji.dot(Pji.T)
Hjl
```

Вектор направления шага итерации $\Delta \mathbf{f}$:

```{code-cell} python
dfj = -np.linalg.inv(Hjl).dot(dFdfj)
```

Значения основных переменных для следующей итерации при $\lambda = 1$:

```{code-cell} python
lmbd = 1.
fjkp1 = fjk + lmbd * dfj
fjkp1
```

Отобразим эту итерацию на графике с нанесением области допустимых решений, ограниченных $\mathbf{f}^\top \mathbf{a}_i \leq b_i$. Для этого вычислим значения вектора $\mathbf{b}$:

```{code-cell} python
bi = np.min([np.min(1. - Kji * yi, axis=0), 1. - yi], axis=0)
```

Зададимся диапазоном изменения мольной доли первой фазы и вычислим для него значения прямых, ограничивающих область допустимых решений:

```{code-cell} python
f1 = np.array([-2., 3.])
f2 = (-Aji[0] / Aji[1])[:,None] * f1 + (bi / Aji[1])[:,None]
```

Также вычислим координаты полюсов:

```{code-cell} python
p2 = (-Aji[0] / Aji[1])[:,None] * f1 + (1. / Aji[1])[:,None]
```

Отобразим результаты расчетов на графике. В данном случае, в связи с большим значением компонентов, цветом будет закрашиваться не оставшаяся область, а, наоборот, отсекаемая. В результате незакрашенная зона на графике будет представлять область допустимых решений:

```{code-cell} python
np.random.seed(5) # 5
fig4, ax4 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
lims = [[-1., -1.], [4., 4.]]
for i in range(yi.shape[0]):
    c = np.random.uniform(size=(3,))
    idx = 0 if Aji[1,i] < 0 else 1
    ax4.fill_between(f1, f2[i], lims[idx], color=c, alpha=0.2, zorder=2)
    ax4.plot(f1, f2[i], lw=2.5, c=c, zorder=4)
    ax4.plot(f1, p2[i], ls='--', lw=.5, c=c, zorder=4)
ax4.plot(*fjk, 'o', mec='c', mfc='c', ms=4., lw=0., zorder=5, label='$f_j^k$')
ax4.plot(*fjkp1, 'o', mec='m', mfc='m', ms=4., lw=0., zorder=5,
         label=r'$f_j^{k+1}$ $\left(\lambda = 1\right)$')
ax4.annotate('', fjk, fjkp1, arrowprops=dict(arrowstyle='<-'), zorder=4)
ax4.plot(0.6868, 0.0602, 'o', mec='r', mfc='r', ms=4., lw=0., zorder=5, label='Решение')
ax4.set_xlabel('$f_1$')
ax4.set_xlim(-2., 3.)
ax4.set_ylabel('$f_2$')
ax4.set_ylim(-1., 4.)
ax4.legend(loc=1, fontsize=9)
ax4.grid(zorder=1)
```

Сплошными линиями на данном графике представлены прямые, удовлетворяющие условию $\mathbf{f}^\top \mathbf{a}_i = b_i, \, i = 1 \, \ldots \, N_c$, а пунктирными – полюсы функции. Анализируя результаты итерации, можно отметить, что наблюдается вылет за пределы области допустимых решений и даже за пределы NF-window. Построим зависимость оптимизируемой функции от длины шага в выбранном направлении. Зададимся диапазоном значений длины шага:

```{code-cell} python
lmbds = np.linspace(0., 2., 500, endpoint=True)
```

Вычислим значения функции $F \left( \mathbf{f} \right)$ для каждой длины шага итерации:

```{code-cell} python
fsjkp1 = fjk + dfj * lmbds[:,None]
tsi = 1. - fsjkp1.dot(Aji)
Fs = - np.log(np.abs(tsi)).dot(yi)
```

Построим график:

```{code-cell} python
fig5, ax5 = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
ax5.plot(lmbds, Fs, lw=2., c='y', zorder=2)
ax5.set_xlim(0., 2.)
ax5.set_xlabel(r'$\lambda$')
ax5.set_ylim(-0.35, -0.05)
ax5.set_ylabel(r'$F \left( \lambda \right)$')
ax5.grid(zorder=1)
```

При $\lambda = 1$ итерация выходит за пределы NF-window, следовательно значение $\lambda_{max}$ должно быть меньше единицы:

```{code-cell} python
lmbdi = (bi - fjk.dot(Aji)) / (dfj.dot(Aji))
where = dfj.dot(Aji) > 0.
np.min(lmbdi[where])
```

Следовательно, условием проведения процедуры поиска оптимальной длины шага итерации является $\lambda_{max} <= 1$.

Таким образом, сформулируем алгоритм решения системы уравнений Речфорда-Райса.

```{eval-rst}
.. role:: comment
    :class: comment
```

```{admonition} Алгоритм. Решение системы уравнений Речфорда-Райса для многофазных систем
:class: algorithm

**Дано:** Матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$; вектор компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$; вектор начальных приближений мольных долей фаз $\mathbf{f}_0 \in {\rm I\!R}^{N_p - 1}$; максимальное число итераций $N_{iter}$; точность решения уравнения $\epsilon$; максимальное число итераций процедуры поиска оптимальной длины шага $N_{ls}$; погрешность для процедуры поиска оптимальной длины шага $\epsilon_{ls}$.

**Определить:** Вектор мольных долей фаз, удовлетворяющий NF-window и являющийся корнем системы уравнений Речфорда-Райса.

**Псевдокод:**  
**def** $\,linesearch \left( \mathbf{f}, \, \Delta \mathbf{f}, \, \mathbf{g}, \, \mathbf{H}, \, \lambda_{max} \right) \rightarrow \lambda$ {comment}`# Функция поиска оптимальной длины шага`  
$\mathbf{A} := 1 - \mathbf{K}$  
$b_i := \min \left\{ 1 - y_i, \, \min_j \left\{ 1 - K_{ji} y_i \right\} \right\}, \; i := 1 \, \ldots \, N_c$  
$\mathbf{f} := \mathbf{f}_0$ {comment}`# Начальное приближение`  
$\mathbf{t} := 1 - \mathbf{f}^\top \mathbf{A}$  
$\mathbf{g} := A \left( \mathbf{y} \oslash \mathbf{t} \right)$ {comment}`# Градиент`  
$k := 1$ {comment}`# Счетчик итерации`  
**while** $\lVert \mathbf{g} \rVert > \epsilon$ **and** $k < N_{iter}$ **do**  
&emsp;$\mathbf{P} := A \left( \sqrt{\mathbf{y}} \oslash \mathbf{t} \right)$  
&emsp;$\mathbf{H} := \mathbf{P} \mathbf{P}^\top$ {comment}`# Гессиан`  
&emsp;$\Delta \mathbf{f} := - \mathbf{H}^{-1} \nabla F$ {comment}`# Направление итерации`  
&emsp;$\lambda_{max} := \min_i \left\{ \left( b_i - \mathbf{f}^\top \mathbf{a}_i \right) \oslash \left( \Delta \mathbf{f}^\top \mathbf{a}_i \right) \, : \, \Delta \mathbf{f}^\top \mathbf{a}_i > 0 \right\}$ {comment}`# Максимальное значение шага итерации`  
&emsp;**if** $\lambda_{max} < 1$ **then** {comment}`# Проверка выхода за пределы области допустимых решений`  
&emsp;&emsp;$\lambda := linesearch \left( \mathbf{f}, \, \Delta \mathbf{f}, \, \mathbf{g}, \, \mathbf{H}, \, \lambda_{max} \right)$ {comment}`# Определение длины шага`  
&emsp;**else**  
&emsp;&emsp;$\lambda := 1$  
&emsp;**end if**  
&emsp;$\mathbf{f} := \mathbf{f} + \lambda \Delta \mathbf{f}$ {comment}`# Обновление вектора основных переменных`  
&emsp;$\mathbf{t} := 1 - \mathbf{f}^\top \mathbf{A}$  
&emsp;$\mathbf{g} := A \left( \mathbf{y} \oslash \mathbf{t} \right)$ {comment}`# Градиент`  
&emsp;$k := k + 1$ {comment}`# Счетчик итерации`  
```

Рассмотрим реализацию данного алгоритма. Процедура поиска оптимальной длины шага выполняется исходя из условия минимизации оптимизируемой функции в заданном направлении методом Ньютона в соответствии с \[[Okuno et al, 2010](https://doi.org/10.2118/117752-PA)\]. Применение процедуры *line search* методом обратного хода с учетом условий Голдштейна реализовано [здесь](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/_src/rr.py).

```{code-cell} python
import numpy.typing as npt

def solveNp(
    Kji: npt.NDArray[np.float64],
    yi: npt.NDArray[np.float64],
    fj0: npt.NDArray[np.float64],
    tol: np.float64 = np.float64(1e-6),
    Niter: int = 30,
    tol_ls: np.float64 = np.float64(1e-5),
    Niter_ls: int = 10,
):
    Aji = 1. - Kji
    bi = np.min([np.min(1. - Kji * yi, axis=0), 1. - yi], axis=0)
    fjk = fj0
    ti = 1. - fjk.dot(Aji)
    gj = Aji.dot(yi / ti)
    gnorm = np.linalg.norm(gj)
    if gnorm < tol:
        return fjk
    print(f'Iteration #0:\n\t{fjk = }\n\t{gnorm = }')
    k: int = 1
    while (gnorm > tol) & (k < Niter):
        Pji = np.sqrt(yi) / ti * Aji
        Hjl = Pji.dot(Pji.T)
        dfj = -np.linalg.inv(Hjl).dot(gj)
        denom = dfj.dot(Aji)
        where = denom > 0.
        lmbdi = ((bi - fjk.dot(Aji)) / denom)[where]
        idx = np.argmin(lmbdi)
        lmbdmax = lmbdi[idx]
        if lmbdmax < 1.:
            plmbdmax = (ti / denom)[where][idx]
            lmbdn = lmbdmax / 2.
            fjkp1 = fjk + lmbdn * dfj
            ti = 1. - fjkp1.dot(Aji)
            gj = Aji.dot(yi / ti)
            dFdlmbd = dfj.dot(gj)
            n: int = 1
            print(f'\tLS-Iteration #{n}:\n\t\t{lmbdn = }\n\t\t{dFdlmbd = }')
            while (np.abs(dFdlmbd) > tol_ls) & (n < Niter_ls):
                Pji = np.sqrt(yi) / ti * Aji
                Hjl = Pji.dot(Pji.T)
                d2Flmbd2 = dfj.dot(Hjl).dot(dfj)
                dlmbd = -dFdlmbd / d2Flmbd2
                lmbdnp1 = lmbdn + dlmbd
                if lmbdnp1 > plmbdmax:
                    print(f'\t\tPerform the bisection step:\n\t\t{lmbdnp1 = } > {plmbdmax = }')
                    lmbdnp1 = (lmbdn + plmbdmax) / 2.
                elif lmbdnp1 < 0:
                    print(f'\t\tPerform the bisection step:\n\t\t{lmbdnp1 = } < 0')
                    lmbdnp1 = lmbdn / 2.
                n += 1
                lmbdn = lmbdnp1
                fjkp1 = fjk + lmbdnp1 * dfj
                ti = 1. - fjkp1.dot(Aji)
                gj = Aji.dot(yi / ti)
                dFdlmbd = dfj.dot(gj)
                print(f'\tLS-Iteration #{n}:\n\t\t{lmbdn = }\n\t\t{dFdlmbd = }')
            gnorm = np.linalg.norm(gj)
            fjk = fjkp1
        else:
            fjk += dfj
            ti = 1. - fjk.dot(Aji)
            gj = Aji.dot(yi / ti)
            gnorm = np.linalg.norm(gj)
        print(f'\nIteration #{k}:\n\t{fjk = }\n\t{gnorm = }')
        k += 1
    return fjk
```

Применим данную реализацию алгоритма к рассматриваемому примеру:

```{code-cell} python
fj = solveNp(Kji, yi, np.array([0.33699, 0.4512]))
fj
```

На самой первой итерации алгоритм выполняет процедуру поиска оптимальной длины шага, после чего находит решение методом Ньютона за три дополнительных итерации.

Применение реализации алгоритма для ранее рассматриваемого примера:

```{code-cell} python
Kji = np.array([
    [2.64675, 1.16642, 1.25099E-03],
    [1.83256, 1.64847, 1.08723E-02],
])
yi = np.array([0.3, 0.4, 0.3])
```

```{code-cell} python
fj = solveNp(Kji, yi, np.array([0.3333, 0.3333]))
fj
```

В данном случае процедура поиска оптимальной длины шага итерации не потребовалась, алгоритм нашел решение за пять итераций.

Таким образом, выше был представлен устойчивый *(robust)* алгоритм решения системы уравнений Речфорда-Райса. Необходимо отметить, что практически значимый численный метод определяет не только его устойчивость, но и его эффективность *(rapidness)* – количество затрачиваемых итераций на поиск решения заданной точности, зависящая от используемого метода решения системы уравнений и качества начального приближения. В представленном алгоритме используется метод Ньютона для поиска минимума функции, что считается достаточно эффективным для небольшой размерности вектора основных переменных (при увеличении размерности начинают возникать сложности с поиском обратной матрицы или с решением системы линейных уравнений). Второе направление увеличения эффективности алгоритма – это повышение точности начального приближения.

Прежде всего необходимо отметить, что система уравнений Речфорда-Райса зачастую решается внутри цикла решения системы уравнений, описывающих термодинамическое равновесие (об этом речь пойдет в [следующем разделе](SEC-5-Equilibrium.md)), следовательно, в качестве начального приближения могут использоваться значения мольных долей фаз на предыдущей итерации внешнего цикла. Кроме того, трехфазный ($\left( N_p + 1 \right)$-фазный) итерационный расчет равновесного состояния зачастую выполняется после двухфазного ($\left( N_p \right)$-фазного), поэтому на первой итерации $\left( N_p + 1 \right)$-фазного расчета в качестве начальных приближений мольных долей фаз могут быть использованы результаты $\left( N_p \right)$-фазного расчета. Стоит отметить, что переход к рассмотрению $\left( N_p + 1 \right)$-фазного расчета равновесного состояния осуществляют после проверки на [стабильность](SEC-1-Stability.md) результатов $\left( N_p \right)$-фазного расчета. В этом случае начальное приближение рекомендуется задавать с учетом ограничений области физичных значений. Так, например, в работе \[[Leibovici and Nichita, 2008](https://doi.org/10.1016/j.fluid.2008.03.006)\] авторы используют начальное приближение $f_j = 1 \, / \, N_p, \, j = 1 \, \ldots \, N_p - 1$. Авторы \[[Okuno et al, 2010](https://doi.org/10.2118/117752-PA)\] в качестве начального приближения рекомендуют среднее от координат вершин пересечения области допустимых решений $\mathbf{f}^\top \mathbf{a}_i = b_i, \, i = 1 \, \ldots \, N_c,$ и области физичных значений, ограниченной прямыми $\left\{ f_j \geq 0, \, j = 1 \, \ldots \, N_p - 1; \, \sum_j f_j \leq 1 \right\}$, однако данный подход может быть ресурсоемким для большого числа компонентов или вблизи области [бикритической точки](SEC-7-Criticality.md), где константы фазового равновесия близки единице, а прямые получаются почти параллельными \[[Li and Firoozabadi, 2012](https://doi.org/10.1016/j.fluid.2012.06.021)\].

Представим также несколько примеров применения данного алгоритма для трех- и четырехфазных систем.

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [26.3059904941, 1.91580344867, 1.42153325608, 3.21966622946, 0.22093634359, 0.01039336513, 19.4239894458],
    [66.7435876079, 1.26478653025, 0.94711004430, 3.94954222664, 0.35954341233, 0.09327536295, 12.0162990083],
]) # K-values
yi = np.array(
    [0.132266176697, 0.205357472415, 0.170087543100, 0.186151796211, 0.111333894738, 0.034955417168, 0.159847699672]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out1
```

````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [26.3059904941, 1.91580344867, 1.42153325608, 3.21966622946, 0.22093634359, 0.01039336513, 19.4239894458],
    [66.7435876079, 1.26478653025, 0.94711004430, 3.94954222664, 0.35954341233, 0.09327536295, 12.0162990083],
]) # K-values
yi = np.array(
    [0.132266176697, 0.205357472415, 0.170087543100, 0.186151796211, 0.111333894738, 0.034955417168, 0.159847699672]
) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

def solveNp_out(Kji, yi, fj0, tol=np.float64(1e-6), Niter=30, tol_ls=np.float64(1e-5), Niter_ls=10):
    out = ''
    Aji = 1. - Kji
    bi = np.min([np.min(1. - Kji * yi, axis=0), 1. - yi], axis=0)
    fjk = fj0
    ti = 1. - fjk.dot(Aji)
    gj = Aji.dot(yi / ti)
    gnorm = np.linalg.norm(gj)
    if gnorm < tol:
        return fjk
    out += f'Iteration #0:\n\t{fjk = }\n\t{gnorm = }\n'
    k: int = 1
    while (gnorm > tol) & (k < Niter):
        Pji = np.sqrt(yi) / ti * Aji
        Hjl = Pji.dot(Pji.T)
        dfj = -np.linalg.inv(Hjl).dot(gj)
        denom = dfj.dot(Aji)
        where = denom > 0.
        lmbdi = ((bi - fjk.dot(Aji)) / denom)[where]
        idx = np.argmin(lmbdi)
        lmbdmax = lmbdi[idx]
        if lmbdmax < 1.:
            plmbdmax = (ti / denom)[where][idx]
            lmbdn = lmbdmax / 2.
            fjkp1 = fjk + lmbdn * dfj
            ti = 1. - fjkp1.dot(Aji)
            gj = Aji.dot(yi / ti)
            dFdlmbd = dfj.dot(gj)
            n: int = 1
            out += f'\tLS-Iteration #{n}:\n\t\t{lmbdn = }\n\t\t{dFdlmbd = }\n'
            while (np.abs(dFdlmbd) > tol_ls) & (n < Niter_ls):
                Pji = np.sqrt(yi) / ti * Aji
                Hjl = Pji.dot(Pji.T)
                d2Flmbd2 = dfj.dot(Hjl).dot(dfj)
                dlmbd = -dFdlmbd / d2Flmbd2
                lmbdnp1 = lmbdn + dlmbd
                if lmbdnp1 > plmbdmax:
                    out += f'\t\tPerform the bisection step:\n\t\t{lmbdnp1 = } > {plmbdmax = }\n'
                    lmbdnp1 = (lmbdn + plmbdmax) / 2.
                elif lmbdnp1 < 0:
                    out += f'\t\tPerform the bisection step:\n\t\t{lmbdnp1 = } < 0\n'
                    lmbdnp1 = lmbdn / 2.
                n += 1
                lmbdn = lmbdnp1
                fjkp1 = fjk + lmbdnp1 * dfj
                ti = 1. - fjkp1.dot(Aji)
                gj = Aji.dot(yi / ti)
                dFdlmbd = dfj.dot(gj)
                out += f'\tLS-Iteration #{n}:\n\t\t{lmbdn = }\n\t\t{dFdlmbd = }\n'
            gnorm = np.linalg.norm(gj)
            fjk = fjkp1
        else:
            fjk += dfj
            ti = 1. - fjk.dot(Aji)
            gj = Aji.dot(yi / ti)
            gnorm = np.linalg.norm(gj)
        out += f'\nIteration #{k}:\n\t{fjk = }\n\t{gnorm = }\n'
        k += 1
    return fjk, out

fj, out1 = solveNp_out(Kji, yi, fj0)

class MultilineText(object):
    def __init__(self, text):
        self.text = text
        self.template = """
<div class="output text_plain highlight-myst-ansi notranslate"><div class="highlight"><pre id="codecell20" tabindex="0"><span></span>{}
</pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell20">
      <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <title>Copy to clipboard</title>
  <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
  <rect x="8" y="8" width="12" height="12" rx="2"></rect>
  <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
</svg>
    </button></div>
</div>
"""
    def _repr_html_(self):
        return self.template.format(self.text.replace('\n', '<br>'))

from myst_nb import glue
glue('glued_out1', MultilineText(out1))

```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [1.64571122126, 1.91627717926, 0.71408616431, 0.28582415424, 0.04917567928, 0.00326226927, 0.00000570946],
    [1.61947897153, 2.65352105653, 0.68719907526, 0.18483049029, 0.01228448216, 0.00023212526, 0.00000003964],
]) # K-values
yi = np.array(
    [0.896646630194, 0.046757914522, 0.000021572890, 0.000026632729, 0.016499094171, 0.025646758089, 0.014401397406]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out2
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [1.64571122126, 1.91627717926, 0.71408616431, 0.28582415424, 0.04917567928, 0.00326226927, 0.00000570946],
    [1.61947897153, 2.65352105653, 0.68719907526, 0.18483049029, 0.01228448216, 0.00023212526, 0.00000003964],
]) # K-values
yi = np.array(
    [0.896646630194, 0.046757914522, 0.000021572890, 0.000026632729, 0.016499094171, 0.025646758089, 0.014401397406]
) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out2 = solveNp_out(Kji, yi, fj0)

glue('glued_out2', MultilineText(out2))
```


````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [0.112359551, 13.72549020, 3.389830508],
    [1.011235955, 0.980392157, 0.847457627],
]) # K-values
yi = np.array([0.08860, 0.81514, 0.09626]) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out3
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [0.112359551, 13.72549020, 3.389830508],
    [1.011235955, 0.980392157, 0.847457627],
]) # K-values
yi = np.array([0.08860, 0.81514, 0.09626]) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out3 = solveNp_out(Kji, yi, fj0)

glue('glued_out3', MultilineText(out3))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [1.46330454, 1.782453544, 0.953866131, 0.560800539, 0.142670434, 0.01174238, 0.000150252],
    [1.513154299, 2.490033379, 0.861916482, 0.323730849, 0.034794391, 0.000547609, 5.54587E-07],
]) # K-values
yi = np.array(
    [0.836206, 0.0115731, 0.0290914, 0.0324648, 0.0524046, 0.0258683, 0.0123914]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out4
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [1.46330454, 1.782453544, 0.953866131, 0.560800539, 0.142670434, 0.01174238, 0.000150252],
    [1.513154299, 2.490033379, 0.861916482, 0.323730849, 0.034794391, 0.000547609, 5.54587E-07],
]) # K-values
yi = np.array(
    [0.836206, 0.0115731, 0.0290914, 0.0324648, 0.0524046, 0.0258683, 0.0123914]
) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out4 = solveNp_out(Kji, yi, fj0)

glue('glued_out4', MultilineText(out4))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [1.5678420840, 3.1505528290, 0.8312143829, 0.4373864613, 0.2335674319, 0.0705088530, 0.0204347990, 0.0039222104, 0.0004477899, 0.0000270002, 0.0000000592],
    [1.5582362780, 2.1752897740, 0.7919216342, 0.5232473203, 0.3492086459, 0.1630735937, 0.0742862374, 0.0263779361, 0.0067511045, 0.0011477841, 0.0000237639]
]) # K-values
yi = np.array(
    [0.96, 0.011756, 0.004076, 0.00334, 0.001324, 0.004816, 0.006324, 0.003292, 0.002112, 0.001104, 0.001856]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out5
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [1.5678420840, 3.1505528290, 0.8312143829, 0.4373864613, 0.2335674319, 0.0705088530, 0.0204347990, 0.0039222104, 0.0004477899, 0.0000270002, 0.0000000592],
    [1.5582362780, 2.1752897740, 0.7919216342, 0.5232473203, 0.3492086459, 0.1630735937, 0.0742862374, 0.0263779361, 0.0067511045, 0.0011477841, 0.0000237639]
]) # K-values
yi = np.array(
    [0.96, 0.011756, 0.004076, 0.00334, 0.001324, 0.004816, 0.006324, 0.003292, 0.002112, 0.001104, 0.001856]
) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out5 = solveNp_out(Kji, yi, fj0)

glue('glued_out5', MultilineText(out5))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [1.248111, 1.566922, 1.338555, 0.86434, 0.722684, 0.629019, 0.596295, 0.522358, 0.501303, 0.413242, 0.295002, 0.170873, 0.08982, 0.042797, 0.018732, 0.006573],
    [1.2793605, 3.5983187, 2.3058527, 0.8619853, 0.4989022, 0.3439748, 0.2907231, 0.1968906, 0.173585, 0.1085509, 0.0428112, 0.0094336, 0.0016732, 0.0002458, 3.4493E-05, 4.7093E-06],
]) # K-values
yi = np.array(
    [0.854161, 0.000701, 0.023798, 0.005884, 0.004336, 0.000526, 0.004803, 0.002307, 0.003139, 0.004847, 0.026886, 0.023928, 0.018579, 0.014146, 0.008586, 0.003373]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out6
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [1.248111, 1.566922, 1.338555, 0.86434, 0.722684, 0.629019, 0.596295, 0.522358, 0.501303, 0.413242, 0.295002, 0.170873, 0.08982, 0.042797, 0.018732, 0.006573],
    [1.2793605, 3.5983187, 2.3058527, 0.8619853, 0.4989022, 0.3439748, 0.2907231, 0.1968906, 0.173585, 0.1085509, 0.0428112, 0.0094336, 0.0016732, 0.0002458, 3.4493E-05, 4.7093E-06],
]) # K-values
yi = np.array(
    [0.854161, 0.000701, 0.023798, 0.005884, 0.004336, 0.000526, 0.004803, 0.002307, 0.003139, 0.004847, 0.026886, 0.023928, 0.018579, 0.014146, 0.008586, 0.003373]
) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out6 = solveNp_out(Kji, yi, fj0)

glue('glued_out6', MultilineText(out6))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [23.75308598, 0.410182283, 0.009451899, 5.20178E-05, 5.04359E-09],
    [28.57470741, 1.5525E-10, 7.7405E-18, 8.0401E-40, 3.8652E-75],
]) # K-values
yi = np.array([0.5, 0.2227, 0.1402, 0.1016, 0.0355]) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out7
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [23.75308598, 0.410182283, 0.009451899, 5.20178E-05, 5.04359E-09],
    [28.57470741, 1.5525E-10, 7.7405E-18, 8.0401E-40, 3.8652E-75],
]) # K-values
yi = np.array([0.5, 0.2227, 0.1402, 0.1016, 0.0355]) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out7 = solveNp_out(Kji, yi, fj0)

glue('glued_out7', MultilineText(out7))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [23.24202564, 0.401865268, 0.00928203, 5.13213E-05, 5.01517E-09],
    [28.55840446, 1.55221E-10, 7.73713E-18, 8.03102E-40, 3.85651E-75],
]) # K-values
yi = np.array([0.5, 0.2227, 0.1402, 0.1016, 0.0355]) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out8
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [23.24202564, 0.401865268, 0.00928203, 5.13213E-05, 5.01517E-09],
    [28.55840446, 1.55221E-10, 7.73713E-18, 8.03102E-40, 3.85651E-75],
]) # K-values
yi = np.array([0.5, 0.2227, 0.1402, 0.1016, 0.0355]) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out8 = solveNp_out(Kji, yi, fj0)

glue('glued_out8', MultilineText(out8))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [4.354749468, 5.514004477, 0.056259268, 0.000465736, 3.83337E-05],
    [9.545344719, 8.93018E-06, 4.53738E-18, 1.15826E-35, 1.07956E-49],
]) # K-values
yi = np.array([0.5, 0.15, 0.1, 0.1, 0.15]) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.3333, 0.3333]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out9
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [4.354749468, 5.514004477, 0.056259268, 0.000465736, 3.83337E-05],
    [9.545344719, 8.93018E-06, 4.53738E-18, 1.15826E-35, 1.07956E-49],
]) # K-values
yi = np.array([0.5, 0.15, 0.1, 0.1, 0.15]) # Global component composition
fj0 = np.array([0.3333, 0.3333]) # Initial estimate

fj, out9 = solveNp_out(Kji, yi, fj0)

glue('glued_out9', MultilineText(out9))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [3.208411, 1.460005, 1.73068, 0.951327, 0.572598, 0.152184, 0.013802, 0.000211],
    [2.259859, 1.531517, 2.892645, 0.814588, 0.243232, 0.016798, 0.000118, 4.27E-08],
    [644.0243063, 0.001829876, 1.51452E-05, 8.05299E-10, 5.65494E-17, 3.81673E-34, 7.23797E-56, 6.58807E-68],
]) # K-values
yi = np.array(
    [0.06, 0.802688, 0.009702, 0.024388, 0.027216, 0.043932, 0.021686, 0.010388]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.25, 0.25, 0.25]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out10
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [3.208411, 1.460005, 1.73068, 0.951327, 0.572598, 0.152184, 0.013802, 0.000211],
    [2.259859, 1.531517, 2.892645, 0.814588, 0.243232, 0.016798, 0.000118, 4.27E-08],
    [644.0243063, 0.001829876, 1.51452E-05, 8.05299E-10, 5.65494E-17, 3.81673E-34, 7.23797E-56, 6.58807E-68],
]) # K-values
yi = np.array(
    [0.06, 0.802688, 0.009702, 0.024388, 0.027216, 0.043932, 0.021686, 0.010388]
) # Global component composition
fj0 = np.array([0.25, 0.25, 0.25]) # Initial estimate

fj, out10 = solveNp_out(Kji, yi, fj0)

glue('glued_out10', MultilineText(out10))
```

````{admonition} Пример
:class: exercise
Пусть матрица констант фазового равновесия $\mathbf{K} \in {\rm I\!R}^{\left( N_p - 1 \right) \times N_c}$ и вектор глобального компонентного состава системы $\mathbf{y} \in {\rm I\!R}^{N_c}$ заданы следующим образом:

``` python
Kji = np.array([
    [1.931794, 1.423945, 2.634586, 0.83815, 0.466729, 0.263498, 0.088209, 0.028575, 0.006425, 9.00E-04, 6.96E-05, 2.14E-07],
    [3.081192, 1.42312, 1.729091, 0.833745, 0.617263, 0.460452, 0.263964, 0.149291, 0.07112, 0.026702, 0.007351, 0.000352],
    [1018.249407, 0.001355733, 2.069E-06, 4.15121E-09, 4.95509E-12, 5.43433E-15, 5.89003E-23, 2.46836E-31, 6.01409E-45, 1.36417E-62, 4.87765E-84, 4.5625E-132],
]) # K-values
yi = np.array(
    [0.0825, 0.7425, 0.051433, 0.017833, 0.014613, 0.005793, 0.02107, 0.027668, 1.44E-02, 9.24E-03, 0.00483, 0.00812]
) # Global component composition
```

Необходимо решить систему уравнений Речфорда-Райса для начального приближения:

``` python
fj0 = np.array([0.25, 0.25, 0.25]) # Initial estimate
```

````

````{dropdown} Решение
``` python
fj = solveNp(Kji, yi, fj0)
```

```{glue:} glued_out11
```
````

```{code-cell} python
:tags: [remove-cell]

Kji = np.array([
    [1.931794, 1.423945, 2.634586, 0.83815, 0.466729, 0.263498, 0.088209, 0.028575, 0.006425, 9.00E-04, 6.96E-05, 2.14E-07],
    [3.081192, 1.42312, 1.729091, 0.833745, 0.617263, 0.460452, 0.263964, 0.149291, 0.07112, 0.026702, 0.007351, 0.000352],
    [1018.249407, 0.001355733, 2.069E-06, 4.15121E-09, 4.95509E-12, 5.43433E-15, 5.89003E-23, 2.46836E-31, 6.01409E-45, 1.36417E-62, 4.87765E-84, 4.5625E-132],
]) # K-values
yi = np.array(
    [0.0825, 0.7425, 0.051433, 0.017833, 0.014613, 0.005793, 0.02107, 0.027668, 1.44E-02, 9.24E-03, 0.00483, 0.00812]
) # Global component composition
fj0 = np.array([0.25, 0.25, 0.25]) # Initial estimate

fj, out11 = solveNp_out(Kji, yi, fj0)

glue('glued_out11', MultilineText(out11))
```

Таким образом, в рамках данного раздела представлен подход к решению системы уравнений Речфорда-Райса в многофазоной постановке. [Следующий раздел](SEC-5-Equilibrium.md) будет посвящен определению равновесного состояния, проводимому, если в результате [проверки стабильности](SEC-1-Stability.md) рассматриваемое фазовое состояние системы оказалось нестабильным.
