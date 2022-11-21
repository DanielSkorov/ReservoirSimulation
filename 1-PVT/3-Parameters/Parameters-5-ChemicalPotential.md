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

<a id='pvt-parameters-chemical_potential'></a>
# Химический потенциал

+++

Рассматривая изотермический квазистационарный процесс с постоянным количеством молекул в системе, химический потенциал компонента, как было показано [ранее](../1-TD/TD-15-Fugacity.html#pvt-td-fugacity), связан с летучестью следующим соотношением:

$$ d \mu_i = R T d \ln f_i.$$

Следовательно, изменение химического потенциала компонента в некоторой фазе:

$$\Delta \mu_i = \mu_i - \mu_i^0 = R T \ln \frac{f_i}{f_i^0}.$$

Таким образом, зная значения летучести компонента при известных давлении $P$ и температуре $T$, а также при референсных давлении $P^0$ и температуре $T^0$ можно определить изменение химического потенциала компонента относительно референсных условий.

```{code-cell} python

```
