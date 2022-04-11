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

<a id='pvt-td-gibbs_duhem_equation'></a>
# Уравнение Гиббса-Дюгема
Рассматривая изотермический квазистационарный процесс, дифференциал энергии Гиббса может быть записан следующим образом:

+++

$$dG = \sum_i \mu_i dN_i + \sum_i N_i d \mu_i.$$

+++

С другой стороны, [ранее](TD-8-Helmholtz-Gibbs.html#pvt-td-helmholtz_gibbs-gibbs_partials) было показано, что дифференциал энергии Гиббса:

+++

$$ dG = -S dT + V dP + \sum_i \mu_i dN_i.$$

+++

Приравнивая правые части записанных выше уравнений, получим:

+++

$$ S dT - V dP + \sum_i N_i d \mu_i = 0.$$

+++

Данное уравнение является уравнением Гиббса-Дюгема.
