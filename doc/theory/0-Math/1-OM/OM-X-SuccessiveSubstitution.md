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

# Метод последовательных подстановок

Метод последовательных подстановок, также известный как [метод простой итерации](https://en.wikipedia.org/wiki/Fixed-point_iteration), на практике используется для решения систем нелинейных уравнений, суть которого заключается в вычислении более точного приближения путем подстановки приближения на предыдущей итерации в некоторую функцию.

Пусть решаемая система нелинейных уравнений записывается следующим образом:

$$ \mathbf{f} \left( \mathbf{x} \right) = 0, $$

где $\mathbf{x} \in \mathbb{R}^n$ представляет собой вектор основных (независимых) переменных, $\mathbf{f} \in \mathbb{R}^n$ – вектор значений зависимых переменных.

Суть метода последовательных подстановок заключается в преобразовании данной системы уравнений к следующему виду:

$$ \mathbf{x} = \boldsymbol{\varphi} \left( \mathbf{x} \right), $$

где $\boldsymbol{\varphi} \in \mathbb{R}^n$ представляет векторную функцию, удовлетворяющую условию эквивалентности перехода от предыдущей записи системы нелинейных уравнений к данной. Выбор данной функции важен для обеспечения сходимости метода последовательных подстановок.

На $\left( k+1 \right)$-й итерации метода последовательных подстановок вектор основных переменных рассчитывается с использованием следующего выражения:

$$ \mathbf{x}_{k+1} = \boldsymbol{\varphi} \left( \mathbf{x}_{k} \right). $$

(convergence)=
## Анализ сходимости

Анализ сходимости метода последовательных подстановок начнем с введения вектора ошибки для следующей, $\left( k+1 \right)$-й итерации, $\mathbf{e}_{k+1} \in \mathbb{R}^n$, определяемого разностью между вектором основных переменных на этой итерации и вектором решения системы нелинейных уравнений $\mathbf{x}_{\infty}$:

$$ \mathbf{e}_{k+1} = \mathbf{x}_{k+1} - \mathbf{x}_{\infty} = \boldsymbol{\varphi} \left( \mathbf{x}_{k} \right) - \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right). $$

Рассмотрим разложение в [ряд Тейлора](https://en.wikipedia.org/wiki/Taylor_series) векторной функции $\boldsymbol{\varphi} \left( \mathbf{x} \right)$ в окрестности решения $\mathbf{x}_{\infty}$:

$$ \boldsymbol{\varphi} \left( \mathbf{x} \right) = \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) + \bar{\nabla} \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) \left( \mathbf{x} - \mathbf{x}_{\infty} \right) + \frac{1}{2} \left( \mathbf{x} - \mathbf{x}_{\infty} \right)^\top \bar{\nabla}^2 \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) \left( \mathbf{x} - \mathbf{x}_{\infty} \right) + \ldots \, , $$

где $\bar{\nabla} \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right)$ представляет собой матрицу частных производных, элементы которой определяются выражением $\left[ \frac{\partial \varphi_i}{\partial x_j} \bigg|_{x_{\infty}}, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n \right]$, а $\bar{\nabla}^2 \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right)$ является тензором, элементы которого представляют собой вторые частные производные $\left[ \frac{\partial^2 \varphi_i}{\partial x_j \partial x_k} \bigg|_{x_{\infty}}, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n, \, k = 1 \, \ldots \, n \right]$.

Пренебрегая элементами ряда Тейлора высших порядков, получим:

$$ \boldsymbol{\varphi} \left( \mathbf{x} \right) \approx \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) + \bar{\nabla} \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) \left( \mathbf{x} - \mathbf{x}_{\infty} \right). $$

Тогда вектор ошибки на $\left( k+1 \right)$-й итерации:

$$ \mathbf{e}_{k+1} \approx \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) + \bar{\nabla} \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) \left( \mathbf{x}_k - \mathbf{x}_{\infty} \right) - \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) = \bar{\nabla} \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right) \left( \mathbf{x}_k - \mathbf{x}_{\infty} \right). $$

Введем обозначение:

$$ \mathbf{J}_{\infty} = \bar{\nabla} \boldsymbol{\varphi} \left( \mathbf{x}_{\infty} \right). $$

Тогда

$$ \mathbf{e}_{k+1} \approx \mathbf{J}_{\infty} \left( \mathbf{x}_k - \mathbf{x}_{\infty} \right) = \mathbf{J}_{\infty} \mathbf{e}_{k}. $$

Таким образом, сходимость метода последовательных подстановок определяется значением следующего предела: $\lim_{k \rightarrow \infty} \mathbf{J}^{k}_{\infty}$.


(algorithm)=
## Алгоритм

(example)=
## Пример
