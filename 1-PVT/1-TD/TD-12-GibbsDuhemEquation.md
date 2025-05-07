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

(pvt-td-gibbsduhemequation)=
# Уравнение Гиббса-Дюгема
Рассматривая изотермический квазистационарный процесс, дифференциал энергии Гиббса может быть записан следующим образом:

$$ dG = \sum_i \mu_i dN_i + \sum_i N_i d \mu_i. $$

С другой стороны, [ранее](TD-8-Helmholtz-Gibbs.md#pvt-td-helmholtzgibbs-gibbspartials) было показано, что дифференциал энергии Гиббса:

$$ dG = -S dT + V dP + \sum_i \mu_i dN_i. $$

Приравнивая правые части записанных выше уравнений, получим:

$$ S dT - V dP + \sum_i N_i d \mu_i = 0. $$

Данное уравнение является уравнением Гиббса-Дюгема.

Если в системе температура и давление постоянны, то уравнение Гиббса-Дюгема:

$$ \sum_i N_i d \mu_i = 0. $$

Разделив левую и правую части данного выражения на $\partial N_1$, получим:

$$ \sum_i N_i \frac{\partial \mu_i}{\partial N_1} = 0. $$

Аналогичное выражение можно записать для остальных компонентов. Следовательно:

$$ \sum_i N_i \frac{\partial \mu_i}{\partial N_j} = 0, \; j = 1 \, \ldots \, C, $$

где $C$ – количество компонентов.
