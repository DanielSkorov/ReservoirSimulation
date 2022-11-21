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

<a id='pvt-parameters-appendix-pd'></a>
# Частные производные термодинамических параметров
В данном разделе приводится вывод выражений частных производных термодинамических параметров по давлению, температуре и количеству вещества компонентов на основе использования [уравнений состояния](../2-EOS/EOS-0-Introduction.html#pvt-eos).

```{code-cell} python

```

```{code-cell} python

```

<a id='pvt-parameters-appendix-pd-enthalpy'></a>
## Частные производные энтальпии

+++

<a id='pvt-parameters-appendix-pd-enthalpy-srk_pr_sw'></a>
### Частные производные энтальпии с использованием уравнений состояния Суаве-Редлиха-Квонга, Пенга-Робинсона и Сорейде-Уитсона
[Ранее](Parameters-2-Enthalpy.html#pvt-parameters-enthalpy-isobaric_isothermal) было показано, что энтальпия фазы в изотермическом процессе определяется следующим выражением:

$$ H \left( P, T, n_i \right) = \sum_{i=1}^{N_c} n_i h_i^{ig} - R T^2 \sum_{i=1}^{N_c} n_i \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i}.$$

Рассмотрим частные производные данного выражения по давлению, температуре и количеству вещества $i$-го компонента.

+++

Частная производная энтальпии по температуре:

$$ \begin{align}
\left( \frac{\partial H}{\partial T} \right)_{P,n_i} &= \sum_{i=1}^{N_c} n_i \left( \frac{\partial h^{ig}_i}{\partial T} \right)_{P,n_i} - R \left( 2 T \sum_{i=1}^{N_c} n_i \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i} + T^2 \sum_{i=1}^{N_c} n_i \left( \frac{\partial^2 \ln \phi_i}{\partial T^2} \right)_{P,n_i} \right) \\
&= \sum_{i=1}^{N_c} n_i {c_P}_i^{ig} - R T^2 \left( \frac{2}{T} \sum_{i=1}^{N_c} n_i \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i} + \sum_{i=1}^{N_c} n_i \left( \frac{\partial^2 \ln \phi_i}{\partial T^2} \right)_{P,n_i} \right).
\end{align}$$

Частные производные логарифма коэффициента летучести по температуре с использованием уравнений состояния [Суаве-Редлиха-Квонга и Пенга-Робинсона](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-srk_pr), а также [Сорейде-Уитсона](../2-EOS/EOS-Appendix-A-PD.html#pvt-eos-appendix-fugacity_pd-sw) были рассмотрены ранее.

+++

Частная производная энтальпии по количеству вещества компонентов:

$$ \left( \frac{\partial H}{\partial n_i} \right)_{P,T} = \sum_{i=1}^{N_c} h_i^{ig} - R T^2 \sum_{i=1}^{N_c} \left( \left( \frac{\partial \ln \phi_i}{\partial T} \right)_{P,n_i} + n_i \left( \frac{\partial^2 \ln \phi_i}{\partial T \partial n_i} \right)_{P} \right). $$

```{code-cell} python

```
