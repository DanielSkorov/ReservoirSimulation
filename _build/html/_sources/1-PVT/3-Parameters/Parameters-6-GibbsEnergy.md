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

<a id='pvt-parameters-gibbs_energy'></a>
# Энергия Гиббса

+++

Дифференциал энергии Гиббса для изотермического квазистационарного процесса с постоянным количеством вещества в системе:

$$ dG = \sum_i n_i d \mu_i = R T \sum_i n_i d \ln f_i. $$

С учетом этого, изменение энергии Гиббса в рассматриваемом процессе:

$$ \Delta G = G - G^0 = \sum_i n_i \Delta \mu_i = R T \sum_i n_i \ln \frac{f_i}{f_i^0}. $$

```{code-cell} python

```
