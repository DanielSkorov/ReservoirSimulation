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

<a id='pvt-td-maxwell_relations'></a>
# Соотношения Максвелла
Существует множество соотношений между частными производными термодинамических параметров систем, называемые ***соотношения Максвелла***. Все они основаны на следующих свойствах дифференциала и частных производных. Пусть имеется некоторая функция $\phi \left( x, y \right)$, при этом ее дифференциал записывается следующим образом:

+++

$$d \phi \left( x, y \right) = M \left( x, y \right) dx + N \left( x, y \right) dy.$$

+++

Следовательно, можно записать, что:

+++

$$M \left( x, y \right) = \left( \frac{\partial \phi \left( x, y \right)}{\partial x} \right)_y; \\ N \left( x, y \right) = \left( \frac{\partial \phi \left( x, y \right)}{\partial y} \right)_x.$$

+++

Запишем выражение для второй частной производной функции $\phi \left( x, y \right)$:

+++

$$\frac{\partial^2 \phi \left( x, y \right)}{\partial x \partial y} = \left( \frac{\partial}{\partial x} \left( \frac{\partial \phi \left( x, y \right)}{\partial y} \right)_x \right)_y = \left( \frac{\partial}{\partial y} \left( \frac{\partial \phi \left( x, y \right) }{\partial x} \right)_y \right)_x.$$

+++

Преобразуя, получим:

+++

$$ \left( \frac{\partial N \left( x, y \right)}{\partial x} \right)_y = \left( \frac{\partial M \left( x, y \right)}{\partial y} \right)_x.$$

+++

Используя данное свойство, рассмотрим вывод соотношений Максвелла.

+++

<a id='pvt-td-maxwell_relations-first'></a>
Допустим, что вторая частная производная внутренней энергии $U$ по энтропии $S$ и объему $V$ непрерывна и определена. [Тогда](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives#Theorem_of_Schwarz) можно записать следующее соотношение:

+++

$$ \frac{\partial^2 U}{\partial S \partial V} = \frac{\partial}{\partial S} \left( \frac{\partial U}{\partial V} \right) = \frac{\partial}{\partial V} \left( \frac{\partial U}{\partial S} \right).$$

+++

При выводе [thermodynamic identity](TD-6-Entropy.html#pvt-td-entropy-thermodynamic_identity) было показано, что:

+++

$$ \begin{align} \left( \frac{\partial U}{\partial S} \right)_{V, N} &= T; \\ \left( \frac{\partial U}{\partial V} \right)_{S, N} &= -P. \end{align} $$

+++

С учетом этого можно заключить, что:

+++

$$ \left( \frac{\partial T}{\partial V} \right)_{S, N} = - \left( \frac{\partial P}{\partial S} \right)_{V, N}.$$

+++

С учетом [производной обратной функции](https://en.wikipedia.org/wiki/Inverse_functions_and_differentiation) данное выражение преобразуется к следующему:

+++

$$ \left( \frac{\partial V}{\partial T} \right)_{S, N} = - \left( \frac{\partial S}{\partial P} \right)_{V, N}. $$

+++

Данное выражение называется ***первым соотношением Максвелла***.

+++

<a id='pvt-td-maxwell_relations-second'></a>
Получим второе соотношение Максвелла. Для этого применим правило симметрии для второй частной производной энергии Гельмгольца $F$ по температуре $T$ и объему $V$:

+++

$$ \frac{\partial^2 F}{\partial T \partial V} = \frac{\partial}{\partial T} \left( \frac{\partial F}{\partial V} \right) = \frac{\partial}{\partial V} \left( \frac{\partial F}{\partial T} \right).$$

+++

При выводе [дифференциала энергии Гельмгольца](TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-helmholtz_partials) было показано, что

+++

$$ \left( \frac{\partial F}{\partial T} \right)_{V, N} = -S; \\ \left( \frac{\partial F}{\partial V} \right)_{T, N} = -P.$$

+++

Отсюда следует, что

+++

$$ \left( \frac{\partial S}{\partial V} \right)_{T, N} = \left( \frac{\partial P}{\partial T} \right)_{V, N}.$$

+++

Данное выражение является ***вторым соотношением Максвелла***.

+++

<a id='pvt-td-maxwell_relations-third'></a>
Для получения третьего соотношения Максвелла запишем дифференциал [энтальпии](TD-5-Enthalpy.html#pvt-td-enthalpy):

+++

$$dH = dU + P dV + V dP.$$

+++

С учетом выражения [thermodynamic identity](TD-6-Entropy.html#pvt-td-entropy-thermodynamic_identity) его можно преобразовать к следующему виду: 

+++

$$dH = T dS + V dP.$$

+++

Следовательно,

+++

$$ \left( \frac{\partial H}{\partial S} \right)_{P, N} = T; \\ \left( \frac{\partial H}{\partial P} \right)_{S, N} = V.$$

+++

Применим правило симметрии для второй частной производной энтальпии по энтропии и давлению:

+++

$$ \frac{\partial^2 H}{\partial S \partial P} = \frac{\partial}{\partial S} \left( \frac{\partial H}{\partial P} \right) = \frac{\partial}{\partial P} \left( \frac{\partial H}{\partial S} \right).$$

+++

Следовательно,

+++

$$ \left( \frac{\partial T}{\partial P} \right)_{S, N} = \left( \frac{\partial V}{\partial S} \right)_{P, N}.$$

+++

Или с учетом производной обратной функции:

+++

$$ \left( \frac{\partial P}{\partial T} \right)_{S, N} = \left( \frac{\partial S}{\partial V} \right)_{P, N}.$$

+++

Данное выражение является ***третьим соотношением Максвелла***.

+++

<a id='pvt-td-maxwell_relations-fourth'></a>
Наконец, для получения четвертого соотношения Максвелла запишем правило симметрии для второй частной производной энергии Гиббса $G$ по температуре $T$ и объему $V$:

+++

$$ \frac{\partial^2 G}{\partial T \partial V} = \frac{\partial}{\partial T} \left( \frac{\partial G}{\partial V} \right) = \frac{\partial}{\partial V} \left( \frac{\partial G}{\partial T} \right).$$

+++

При выводе [дифференциала энергии Гиббса](TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs_partials) было показано, что

+++

$$ \begin{align*} \left( \frac{\partial G}{\partial T} \right)_{P, N} &= -S; \\ \left( \frac{\partial G}{\partial P} \right)_{T, N} &= V. \end{align*} $$

+++

С учетом этого получим:

+++

$$ \left( \frac{\partial S}{\partial P} \right)_{T, N} = - \left( \frac{\partial V}{\partial T} \right)_{P, N}.$$

+++

Данное выражение является ***четвертым соотношением Максвелла***.
