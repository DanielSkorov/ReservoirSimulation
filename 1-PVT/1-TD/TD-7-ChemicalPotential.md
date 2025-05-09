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

(pvt-td-chemicalpotential)=
# Химический потенциал
В соответствии со [вторым началом термодинамики](TD-6-Entropy.md#pvt-td-entropy-secondlaw-entropy) в равновесном состоянии энтропия изолированной системы максимальна. При этом ранее было показано, что если рассматривать термическое равновесие, то показателем его достижения является температура: она одинакова у двух систем, находящихся в термическом равновесии. Для механического равновесия показателем является давление. При этом энтропия, определяемая статистическим весом макросостояния, также зависит от количества частиц (молекул) в системе. Если две системы могут обмениваться молекулами, то они находятся в диффузивном контакте, следовательно, стремятся достичь *диффузивного равновесия*. То есть две системы будут обмениваться молекулами до тех пор, пока не установится равновесие. Система, в которой больше количество молекул, будет стараться передать часть из них, другой системе, у которых этих молекул меньше. Для описания данного процесса необходимо ввести новый параметр, называемый ***химическим потенциалом***, являющийся аналогом давления и температуры для механического и термического равновесий.

```{admonition} Определение
:class: tip
***Химический потенциал*** – это показатель диффузивного равновесия.
```

Рассмотрим следующий пример. Пусть имеется изолированная система, состоящая из двух подсистем $A$ и $B$. Количество молекул в первой подсистеме – $N_A$, во второй – $N_B$. Общая энтропия системы $S = S_A + S_B$. В процессе релаксации (перехода части молекул от отдной подсистемы к другой) системы будет изменяться ее энтропия до достижиения максимума (экстремума). Следовательно, частные производные энтропии системы по количеству молекул в подсистемах $A$ и $B$ будут равны нулю и равны между собой. Следовательно,

$$
\begin{align}
\left( \frac{\partial S}{\partial N_A} \right)_{U_A,V_A} &= \left( \frac{\partial S}{\partial N_B} \right)_{U_B,V_B}; \\
\left( \frac{\partial S_A}{\partial N_A} \right)_{U_A,V_A} + \left( \frac{\partial S_B}{\partial N_A} \right)_{U_A,V_A} &= \left( \frac{\partial S_A}{\partial N_B} \right)_{U_B,V_B} + \left( \frac{\partial S_B}{\partial N_B} \right)_{U_B,V_B}; \\
\left( \frac{\partial S_A}{\partial N_A} \right)_{U_A,V_A} &= \left( \frac{\partial S_B}{\partial N_B} \right)_{U_B,V_B}. \end{align} $$

То есть в равновесном состоянии приращения энтропии подсистем при изменении количества молекул в них равны между собой. Пусть химический потенциал характеризует это приращение так, что:

$$ \mu = - T \left( \frac{\partial S}{\partial N} \right)_{U, V, ...}. $$

Тогда в равновесном состоянии $\mu_A = \mu_B$. Иными словами,

```{admonition} NB
:class: note
Диффузивное равновесное состояние (равновесное состояние, достигаемое в процессе обмена подсистемами молекулами) характеризуется равенством химических потенциалов.
```

Выведем выражение для расчета химического потенциала идеального газа с использованием формулы [Сакура-Тетрода](TD-6-Entropy.md#pvt-td-entropy-idealgasentropy):

$$\mu = -T \frac{\partial S}{\partial N} = -T k \ln \left( \frac{V}{N} \left( \frac{4 \pi m U}{3 N h^2} \right)^\frac{3}{2} \right) = -T k \ln \left( \frac{V}{N} \left( \frac{2 \pi m k T}{h^2} \right)^\frac{3}{2} \right). $$

При увеличении / уменьшении количества молекул химический потенциал увеличивается / уменьшается. Из этого следует, что если в одной системе количество молекул больше (больше химический потенциал), чем в другой (у которой химический потенциал меньше), то для достижения диффузивного равновесия первая система будет отдавать молекулы второй системе. Именно исходя из данного соответствия в выражении химического потенциала используется знак "минус".

<a id='pvt-td-chemicalpotential-thermodynamicidentity'></a>
Дополним дифференциал энтропии, используемый при выводе [thermodynamic identity](TD-6-Entropy.md#pvt-td-entropy-thermodynamicidentity), следующим образом:

$$ dS = \frac{1}{T} dU + \frac{P}{T} dV - \frac{\mu}{T} dN.$$

Данное выражение можно преобразовать к следующему виду:

$$ dU = T dS - P dV + \mu dN. $$

Если рассматривать внутреннюю энергию как функцию $U = U \left( S, V, N \right)$, тогда:

$$ \mu = \left( \frac{\partial U}{\partial N} \right)_{S, V}. $$

Таким образом, химический потенциал может быть интерпретиврован как изменение внутренней энергии системы при добавлении в нее бесконенчно малого количества молекул при постоянных энтропии и объеме.


Если в системе имеется несколько видов молекул (несколько компонентов), тогда выражение для [thermodynamic identity](TD-6-Entropy.md#pvt-td-entropy-thermodynamicidentity) будет записываться следующим образом:

$$ dU = T dS - P dV + \sum_{i} \mu_i dN_i. $$

Здесь $i$ - номер компонента.
