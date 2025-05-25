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

(pvt-td-fugacity)=
# Летучесть

## Определение летучести
Выражение [thermodynamic identity](TD-8-Helmholtz-Gibbs.md#pvt-td-helmholtzgibbs-gibbspartials), записанное через энергию Гиббса, при постоянном количестве частиц в системе записывается следующим образом:

$$ dG = -S dT + V dP. $$

Если рассматривать изотермический процесс, то данное выражение преобразуется следующим образом:

$$ dG = V dP. $$

```{admonition} NB
:class: note
Важно отметить, что, принимая температуру постоянной, мы предполагаем, что в системе температура всех фаз одинакова.
```

Для идеального газа с учетом [уравнения состояния](TD-1-Basics.md#pvt-td-basics-idealgaseos):

$$ dG = \frac{N k T}{P} dP = N k T d \ln P = \frac{N}{N_A} \left(k N_A \right) T d \ln P = n R T d \ln P. $$

Здесь $N_A = 6.022 \cdot 10^{23} \; \frac{1}{моль}$ – число Авогадро, $R = k \cdot N_A = 8.314 \; \frac{Дж}{моль K}$ – универсальная газовая постоянная, $n$ – количество вещества (моль). Стоит отметить, что полученные равнее уравнения относительно количества частиц $N$ справедливы и для количества вещества $n$.

Полученное выражение дифференциала энергии Гиббса справедливо для идеального газа с постоянным количеством молекул в изотермическом квази-стационарном процессе. Для реального газа вместо давления используют такой параметр, как ***летучесть***:

$$ dG = n R T d \ln f. $$

```{admonition} Определение
:class: tip
***Летучесть*** определяется следующим выражением:

$$ \lim_{P \rightarrow 0} \left( \frac{f}{P} \right) = 1. $$

```

При этом

```{admonition} Определение
:class: tip
Отношение летучести к давлению называют ***коэффициентом летучести***:

$$ \phi = \frac{f}{P}. $$

```

Для компонента $i$, находящегося в термодинамическом равновесии (или в квази-стационарном изотермическом процессе), летучесть определяется следующим выражением (единицы измерения химического потенциала – $\frac{Дж}{моль}$):

$$ d \mu_i = R T d \ln f_i. $$

Данное выражение справедливо для процесса с постоянным количеством молекул компонента $i$ в системе. При этом определение летучести дается на основании следующего выражения:

$$ \lim_{P \rightarrow 0} \left( \frac{f_i}{x_i P} \right) = 1. $$

Здесь $x_i$ – мольная доля компонента в фазе.


<a id='pvt-td-fugacity-componentfugacity'></a>
Преобразуем уравнение, определяющее летучесть $i$-го компонента, к следующему виду:

$$ \begin{align}
d \mu_i - R T d \ln \left( x_i P \right) &= R T d \ln f_i - R T d \ln \left( x_i P \right); \\
d \mu_i - R T d \ln \left( x_i P \right) &= R T d \ln \phi_i; \\
d \mu_i - R T \left( d \ln x_i + d \ln P \right) &= R T d \ln \phi_i.
\end{align} $$

Поскольку количество молекул $i$-го компонента в системе зафиксировано, то:

$$ d \mu_i - R T d \ln P = R T d \ln \phi_i. $$

Пусть

````{margin}
```{admonition} Дополнительно
:class: note
Физический смысл коэффициента сверхсжимаемости заключается в уточнении поведения реальной системы относительно идеальной, описываемой [уравнением состояния идеального газа](TD-1-Basics.md#pvt-td-basics-idealgaseos). Для газовой фазы реальной пластовой системы значения коэффициента сверхсжимаемости обычно несколько меньше единицы или немного больше в зависимости от термобарических условий и компонентного состава. Значения коэффициента сверхсжимаемости жидкой фазы реальной пластовой системы обычно много меньше единицы, что соответствует состоянию, когда объем реальной системы в жидком состоянии при рассматриваемых термобарических условиях много меньше объема идеального газа при тех же термобарических условиях. Однако при высоких давлениях коэффициент сверхсжимаемости жидкой фазы может стать близким или даже больше двух, поскольку сжимаемость жидкой фазы реальной пластовой системы сильно меньше сжимаемости идеального газа. В этом случае жидкая фаза будет характеризоваться большим объемом, чем объем идеального газа при тех же термобарических условиях.
```
````

```{admonition} Определение
:class: tip
***Коэффициент сверхсжимаемости*** – параметр, который определяется следующим выражением:

$$Z = \frac{P V}{n R T}.$$

```

Получим дифференциал коэффициента сверхсжимаемости, рассматривая изотермический процесс с постоянным количеством молекул в системе:

$$ dZ = \frac{d \left( P V \right)}{n R T} = \frac{V dP + P dV}{n R T}. $$

Разделим левую и правую части уравнения на $Z$:

$$ \frac{dZ}{Z} = \frac{dP}{P} + \frac{dV}{V}. $$

С учетом этого, [выражение](#pvt-td-fugacity-componentfugacity) для определения летучести компонента будет иметь следующий вид:

$$ R T d \ln \phi_i = d \mu_i - R T \left( \frac{dZ}{Z} - \frac{dV}{V} \right). $$

<a id='pvt-td-fugacity-chemicalpotentialrelation'></a>
Для многокомпонентной системы [thermodynamic identity](TD-8-Helmholtz-Gibbs.md#pvt-td-helmholtzgibbs-helmholtzpartials), выраженное через энергию Гельмгольца, будет иметь следующий вид:

$$ dF = -P dV - S dT + \sum_i \mu_i dn_i. $$

Отсюда следует, что:

$$ \begin{align}
\mu_i &= \left( \frac{\partial F}{\partial n_i} \right)_{V, T, n_{j \neq i}}; \\
-P &= \left( \frac{\partial F}{\partial V} \right)_{T, n_i}.
\end{align} $$

Запишем вторую частную производную энергии Гельмгольца $F$ по объему $V$ и количеству вещества $i$-го компонента $n_i$:

$$ \frac{\partial^2 F}{\partial n_i \partial V} = \left( \frac{\partial}{\partial n_i} \left( \frac{\partial F}{\partial V} \right)_{T, n_i} \right)_{V, T, n_{j \neq i}} = \left( \frac{\partial}{\partial V} \left( \frac{\partial F}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right)_{T, n_i}. $$

С учетом полученных частных производных энергиии Гельмгольца по количеству вещества $i$-го компонента и объему:

$$ - \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \left( \frac{\partial \mu_i}{\partial V} \right)_{T, n_i}. $$

Следовательно, рассматривая изотермический процесс с постоянным количеством молекул $i$-го компонента в системе можно записать:

$$ d \mu_i = - \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} dV. $$

С учетом дифференциала химического потенциала $i$ компонента в изотермическом процессе с постоянным количеством вещества $i$-го компонента в системе получим дифференциал логарифма коэффициента летучести $i$-го компонента:

$$ d \ln \phi_i = \left( \frac{1}{V} - \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right) dV - d \ln Z. $$

Следовательно, интегрируя данное выражение, получим:

$$ \int_0^{\ln \phi_i} d \ln \phi_i = \int_\infty^V \left( \frac{1}{V} - \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right) dV - \int_0^{\ln Z} d \ln Z. $$

Или при замене пределов интегрирования:

$$ \ln \phi_i = \int_\infty^V \left( \frac{1}{V} - \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} \right) dV - \ln Z = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} - \frac{1}{V} \right) dV - \ln Z . $$


<a id='pvt-td-fugacity-equilibrium'></a>
```{admonition} NB
:class: note
Таким образом, если рассматривать квази-стационарный изотермический процесс с постоянным количеством вещества в системе, состоящей из нескольких фаз $N_p$, то равновесное состояние каждого компонента вместо равенства химических потенциалов компонентов будет определяться равенством летучестей компонентов в каждой из фаз соответственно:

$$ f_{ji} = f_{N_pi}, \; j = 1 \, \ldots \, N_p - 1, \; i = 1 \, \ldots \, N_c. $$

```

<a id='pvt-td-fugacity-PT'></a>
При этом, поскольку рассматривается равновесное состояние, то количество вещества каждого компонента постоянно, следовательно, летучесть компонента может быть рассчитана по коэффициенту летучести, определяемому следующим выражением:

$$ \ln \phi_i = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} - \frac{1}{V} \right) dV - \ln Z.$$

Кроме того, полученное выражение также справедливо для систем, состояющих из одного компонента. В таком случае выражение для логарифма коэффициента летучести записывается следующим образом:

$$ \ln \phi = \int_V^\infty \left( \frac{1}{R T} \left( \frac{\partial P}{\partial n} \right)_{V, T} - \frac{1}{V} \right) dV - \ln Z.$$

Выражение для коэффициента летучести можно представить в несколько другом равносильном виде. Использование одного или другого выражения определяется удобством выражения частных производных по уравнению состояния. Отправной точкой является [формула](#pvt-td-fugacity-componentfugacity), устанавливающая соотношение между коэффициентом летучести и химическим потенциалом в изотермическом процессе:

$$ d \mu_i - R T d \ln P = R T d \ln \phi_i. $$

Для изотермического процесса дифференциал [энергии Гиббса](TD-8-Helmholtz-Gibbs.md#pvt-td-helmholtzgibbs-gibbspartials) записывается следующим образом:

$$ dG = -S dT + V dP + \sum_{i=1}^{N_c} \mu_i dn_i .$$

Следовательно, можно записать следущие соотношения:

$$ \begin{align}
\mu_i &= \left( \frac{\partial G}{\partial n_i} \right)_{P, T, n_{j \neq i}}; \\
V &= \left( \frac{\partial G}{\partial P} \right)_{T, n_i}.
\end{align} $$

Преобразуем выражение для второй частной производной энергии Гиббса по давлению и количеству вещества $i$-го компонента:

$$ \frac{\partial^2 G}{\partial n_i \partial P} = \left( \frac{\partial}{\partial n_i} \left( \frac{\partial G}{\partial P} \right)_{T, n_i} \right)_{P, T, n_{j \neq i}} = \left( \frac{\partial}{\partial P} \left( \frac{\partial G}{\partial n_i} \right)_{P, T, n_{j \neq i}} \right)_{T, n_i}. $$

Следовательно,

$$ \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} = \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i}. $$

Для изотермического процесса с постоянным количеством частиц $i$-го компонента в системе:

$$ d \mu_i = \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} dP. $$

Тогда дифференциал химического потенциала $i$-го компонента преобразуется к следующему виду:

$$ d \ln \phi_i = \left( \frac{1}{RT} \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{1}{P} \right) dP. $$

<a id='pvt-td-fugacity-VT'></a>
Интегрируя данное выражение, получим:

$$ \ln \phi_i = \int_0^P \left( \frac{1}{RT} \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{1}{P} \right) dP. $$

(pvt-td-fugacity-idealgas)=
## Летучесть идеального газа
Рассмотрим применение полученных выражений для идеального газа, описываемого уравнением состояния:

$$ PV = n R T. $$

Для идеального газа частная производная давления по количеству вещества компонента:

$$ \left( \frac{\partial P}{\partial n} \right)_{V, T} = \frac{RT}{V}. $$

Тогда логарифм коэффициента летучести:

$$ \ln \phi = \int_V^\infty \left( \frac{1}{R T} \frac{RT}{V} - \frac{1}{V} \right) dV - \ln Z = - \ln Z. $$

Поскольку для идеального газа коэффициент сверхсжимаемости $Z = 1,$ тогда

$$ \ln \phi = 0. $$

То есть коэффициент летучести:

$$ \phi = 1. $$

Следовательно, летучесть:

$$ f = P. $$

Аналогичный результат может быть получен при использовании выражения с интегралом по давлению.


(pvt-td-fugacity-idealgasmixture)=
## Летучесть смеси идеальных газов
Для смеси идеальных газов также выполняется уравнение состояния:

$$ PV = n R T, $$

где $n = \sum_{i=1}^{N_c} n_i.$ В данном выражении $N_c$ – количество компонентов в системе. Следовательно, частная производная давления по количеству вещества компонента $i$:

$$ \left( \frac{\partial P}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \frac{RT}{V} \sum_{i=1}^{N_c} \left( \frac{\partial n_i}{\partial n_i} \right)_{V, T, n_{j \neq i}} = \frac{RT}{V} . $$

Тогда логарифм коэффициента летучести $i$-го компонента:

$$ \ln \phi_i = 0. $$

Тогда летучесть $i$-го компонента:

$$ f_i = x_i P. $$

```{admonition} NB
:class: note
Полученное выражение является частным случаем соотношения, выполняемого для ***идеальных смесей***:

$$ f_i \left( P, T, \mathbf{x} \right) = f_i \left( P, T, x_1, x_2, \ldots, x_{N_c} \right) = x_i f \left( P, T \right). $$

```

Докажем данное утверждение. Для начала дадим определение идеальной смеси.

```{admonition} Определение
:class: tip
Смесь является ***идеальной***, если для нее выполняется следующее соотношение:

$$ V \left(P, T, \mathbf{n} \right) = \sum_{i=1}^{N_c} n_i v_i \left( P, T \right). $$

Здесь $N_c$ – количество компонентов в системе, $v_i \left( P, T \right)$ – молярный (удельный) объем $i$-го компонента.
```

То есть объем для идеальных смесей является экстенсивным параметром.

```{admonition} Доказательство
:class: proof
[Ранее](TD-10-MixtureGibbsEnergy.md#pvt-td-mixturegibbsenergy) для всех экстенсивных параметров было показано, что:

$$ V \left(P, T, \mathbf{n} \right) = \sum_{i=1}^{N_c} n_i \bar{V_i}.$$

Следовательно, с учетом определения идеальной смеси получим следующее соотношение:

$$ \bar{V_i} = v_i \left( P, T \right).$$

Запишем и преобразуем с учетом данного соотношения полученное [ранее](#pvt-td-fugacity-VT) выражение для коэффициента летучести $i$-го компонента:

$$ \begin{align}
\ln \phi_i
&= \int_0^P \left( \frac{1}{RT} \left( \frac{\partial V}{\partial n_i} \right)_{P, T, n_{j \neq i}} - \frac{1}{P} \right) dP \\
&= \int_0^P \left( \frac{\bar{V_i}}{RT} - \frac{1}{P} \right) dP \\
&= \int_0^P \left( \frac{v_i \left( P, T \right)}{RT} - \frac{1}{P} \right) dP .
\end{align} $$

Из данного выражения видно, что для идеальной смеси логарифм коэффициента летучести не зависит от компонентного состава и определяется только термобарическими условиями и свойствами компонентов. Получим выражение для летучести $i$-го компонента в идеальной смеси. Для этого преобразуем и проинтегрируем полученное выражение:

$$ f_i \left(P, T, \mathbf{x} \right) = x_i P \exp \left( \int_0^P \left( \frac{v_i \left( P, T \right)}{RT} - \frac{1}{P} \right) dP \right). $$

Для чистого компонента $x_i = 1$, тогда летучесть чистого компонента:

$$ f \left(P, T \right) = P \exp \left( \int_0^P \left( \frac{v \left( P, T \right)}{RT} - \frac{1}{P} \right) dP \right). $$

Следовательно, летучесть $i$-го компонента:

$$ f_i \left(P, T, \mathbf{x} \right) = x_i f \left( P, T \right). $$

Таким образом, летучесть $i$-го компонента в идеальной смеси равна летучести этого же компонента при отсутствии других компонентов.
```

<a id='pvt-td-fugacity-idealgasmixture-chemicalpotential'></a>
Получим еще одно важное свойство смесей идеальных газов.

```{admonition} NB
:class: note
Пусть имеется идеальный однокомпонентный газ при давлении $P$. Если данный газ изобарно смешать с другими идеальными газами, то давление рассматриваемого газа в смеси в соответствии с полученным ранее выражением будет равняться $x_i P$. Поскольку для данного компонента произошло изменение давления (температура рассматривается постоянной), то дифференциал химического потенциала:

$$ d \mu^{ig}_i = \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i} dP. $$

Интегрируя данное выражение от однокомпонентного состояния к многокомпонентной смеси, получим:

$$ \mu^{ig}_i \left( P, T, \mathbf{n} \right) - \mu^{ig}_i \left( P, T \right) = \int_{P}^{x_i P} \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i} dP. $$

Рассмотрим вторую частную производную энергии Гиббса по количеству вещества $i$-го компонента и давлению:

$$ \frac{\partial^2 G}{\partial n_i \partial P} = \frac{\partial}{\partial n_i} \left( \left( \frac{\partial G}{\partial P} \right)_{T,n_i} \right)_{P,T} = \frac{\partial}{\partial P} \left( \left( \frac{\partial G}{\partial n_i} \right)_{P,T} \right)_{T,n_i}. $$

С учетом [частных производных энергии Гиббса](TD-8-Helmholtz-Gibbs.md#pvt-td-helmholtzgibbs-gibbspartials) получим:

$$ \left( \frac{\partial V}{\partial n_i} \right)_{P,T} = \left( \frac{\partial \mu_i}{\partial P} \right)_{T,n_i}. $$

Тогда рассматриваемая разница химических потенциалов может быть преобразована следующим образом, применяя уравнение состояния идеального газа:

$$ \begin{align}
\mu^{ig}_i \left( P, T, \mathbf{n} \right) - \mu^{ig}_i \left( P, T \right)
&= \int_{P}^{x_i P} \left( \frac{\partial \mu_i}{\partial P} \right)_{T, n_i} dP \\
&= \int_{P}^{x_i P} \left( \frac{\partial V}{\partial n_i} \right)_{P,T} dP \\
&= \int_{P}^{x_i P} \frac{\partial}{\partial n_i} \left( \frac{nRT}{P} \right)_{P,T} dP \\
&= RT \int_{P}^{x_i P} \frac{dP}{P} \\
&= RT \ln x_i .
\end{align} $$

Следовательно,

$$ \mu^{ig}_i \left( P, T, \mathbf{n} \right) = \mu^{ig}_i \left( P, T \right) + RT \ln x_i . $$

Данное уравнение может быть использоваться для расчета химического потенциала компонента идеального газа в идеальной смеси, зная химический потенциал чистого компонента при тех же термобарических условиях и компонентный состав смеси.
```

При давлении $P_1$ и компонентном составе ${x_i}_1$ химический потенциал $i$-го компонента в смеси:

$$ \mu_i \left(P_1, T, \mathbf{x}_1 \right) = \mu_i \left( P_1, T \right) + RT \ln {x_i}_1. $$

При давлении $P_2$ и компонентном составе ${x_i}_2$ химический потенциал $i$-го компонента в смеси:

$$ \mu_i \left(P_2, T, \mathbf{x}_2 \right) = \mu_i \left( P_2, T \right) + RT \ln {x_i}_2. $$

Тогда разница данных выражений:

$$ \mu_i \left(P_2, T, \mathbf{x}_2 \right) - \mu_i \left(P_1, T, \mathbf{x}_1 \right) = \mu_i \left( P_2, T \right) - \mu_i \left( P_1, T \right) + R T \ln \frac{{x_i}_2}{{x_i}_1}. $$

При этом разница химических потенциалов чистого компонента, являющегося идеальным газом,:

$$ \mu_i \left( P_2, T \right) - \mu_i \left( P_1, T \right) = RT \ln \frac{P_2}{P_1}. $$

Тогда:

$$ \mu_i \left(P_2, T, \mathbf{x}_2 \right) - \mu_i \left(P_1, T, \mathbf{x}_1 \right) = RT \ln \frac{P_2 {x_i}_2}{P_1 {x_i}_1}. $$


(pvt-td-fugacity-realgasmixture)=
## Летучесть смеси реальных газов
По аналогии с идеальным газом, для реальных газов данное соотношение записывается следующим образом:

$$ \mu_i \left(P_2, T, \mathbf{x}_2 \right) - \mu_i \left(P_1, T, \mathbf{x}_1 \right) = RT \ln \frac{ f_i \left(P_2, T, \mathbf{x}_2 \right)}{f_i \left( P_1, T, \mathbf{x}_1 \right)}. $$

Докажем корректность данного соотношения.

```{admonition} Доказательство
:class: proof
Для квази-стационарного изотермического процесса проинтегрируем дифференциал химического потенциала компонента рассматриваемой многокомпонентной системы:

$$ \begin{align}
d \mu_i \left( P,T, \mathbf{x} \right) &= RT d \ln f_i \left( P,T, \mathbf{x} \right), \; i = 1 \, \ldots \, N_c, \\
\int_{P^*}^{P} d \mu_i \left( P,T, \mathbf{x} \right) &= RT \int_{P^*}^{P} d \ln f_i \left( P,T, \mathbf{x} \right), \; i = 1 \, \ldots \, N_c,
\end{align} $$

В выражении выше в качестве нижнего предела интегрирования используется давление, соответствующее состоянию $i$-го компонента в виде идеального газа. С учетом этого

$$ \mu_i \left( P,T, \mathbf{x} \right) - \mu_i \left( P^*,T, \mathbf{x} \right) = RT \ln \frac{f_i \left( P,T, \mathbf{x} \right)}{f_i \left( P^*,T, \mathbf{x} \right)}, \; i = 1 \, \ldots \, N_c. $$

Учитывая определение летучести компонента многокомпонентной системы, можно записать следующее выражение:

$$ f_i \left( P^*,T, \mathbf{x} \right) = x_i P^*, \; i = 1 \, \ldots \, N_c. $$

Тогда выражение для химического потенциала $i$-го компонента:

$$ \mu_i \left( P,T, \mathbf{x} \right) = \mu_i \left( P^*,T, \mathbf{x} \right) + RT \ln \frac{f_i \left( P,T, \mathbf{x} \right)}{x_i P^*}, \; i = 1 \, \ldots \, N_c. $$

[Ранее](pvt-td-fugacity-idealgasmixture-chemicalpotential) было доказано следующее свойство смеси идеальных газов:

$$ \mu_i \left( P^*,T, \mathbf{x} \right) = \mu_i \left( P^*,T \right) + RT \ln x_i, \; i = 1 \, \ldots \, N_c. $$

С учетом этого преобразуем выражение для химического потенциала $i$-го компонента к следующему виду:

$$ \begin{align}
\mu_i \left( P,T, \mathbf{x} \right)
&= \mu_i \left( P^*,T \right) + RT \ln x_i + RT \ln \frac{f_i \left( P,T, \mathbf{x} \right)}{x_i P^*} \\
&= \mu_i \left( P^*,T \right) + RT \ln \frac{f_i \left( P,T, \mathbf{x} \right)}{P^*}, \; i = 1 \, \ldots \, N_c.
\end{align} $$

Тогда разность химических потенциалов:

$$ \mu_i \left( P_2,T, \mathbf{x}_2 \right) - \mu_i \left( P_1,T, \mathbf{x}_1 \right) = RT \ln \frac{f_i \left( P_2,T, \mathbf{x}_2 \right)}{f_i \left( P_1,T, \mathbf{x}_1 \right)}, \; i = 1 \, \ldots \, N_c. $$

```

Данный раздел завершает главу, посвященную изложению основ термодинамики, необходимых для выполения композиционного PVT-моделирования. В следующем разделе будут рассмотрены некоторые [уравнения состояния](../2-EOS/EOS-0-Introduction.md), используемые на практике для расчета свойств многофазных многокомпонентных термодинамических систем.
