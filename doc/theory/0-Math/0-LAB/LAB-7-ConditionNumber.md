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

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue

class MultilineText(object):
    def __init__(self, *args, split=False, sep=' '):
        self.text = sep.join([str(obj) for obj in args]).replace('<', '&lt;').replace('>', '&gt;')
        if split:
            words = self.text.split()
            lines = [[]]
            width = 0
            for word in words:
                if width > 90:
                    lines.append([])
                    width = len(word) + 1
                else:
                    width += len(word) + 1
                lines[-1].append(word)
            self.text = '\n'.join([sep.join(line) for line in lines])
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
```

# Число обусловленности

(cond-singular)=
## Максимальное растяжение вектора. Сингулярное значение матрицы

Ранее было введены понятия [линейного преобразования](LAB-4-LinearTransformations.md#def-lintran) и [собственных значения и вектора матрицы](LAB-5-Eigenvalues-Eigenvectors.md#def-eigen) линейного преобразования. Также было доказано, что [собственные векторы матрицы определяют направления растяжения, сжатия или обращения вектора, относящегося к пространству действительных чисел](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-stretching). Следующим шагом является исследование результата произвольного линейного преобразования такого вектора.

Прежде чем переходить к рассмотрении теоремы о наибольшем растяжении вектора, необходимо ввести понятие ***сингулярного значения матрицы***.

<a id='cond-singular-def'></a>
```{admonition} Определение
:class: tip
Пусть [симметричная](LAB-2-Matrices.md#def-matrix-symmetric) и [положительно полуопределенная](LAB-6-Definiteness.md) матрица $\mathbf{B} \in \mathbb{R}^{n \times n}$ определяется следующим выражением:

$$ \mathbf{B} = \mathbf{A}^\top \mathbf{A}, $$

где $\mathbf{A} \in \mathbb{R}^{m \times n}$. Тогда ***сингулярным значением*** матрицы $\mathbf{A}$ является квадратный корень из собственного значения матрицы $\mathbf{B}$.
```

Важно отметить, что несмотря на разную размерность, матрицы $\mathbf{B}_r = \mathbf{A}^\top \mathbf{A}$ и $\mathbf{B}_l = \mathbf{A} \mathbf{A}^\top$ имеют одинаковый набор ненулевых собственных значений.

<a id='lemma-cond-singular'></a>
```{admonition} Лемма
:class: caution
Если $\lambda \neq 0$ является собственным значением матрицы $\mathbf{B}_r = \mathbf{A}^\top \mathbf{A}$, то оно же является собственным значением матрицы $\mathbf{B}_l = \mathbf{A} \mathbf{A}^\top$.
```

```{admonition} Доказательство
:class: proof
Обозначим $\mathbf{v} \in \mathbb{R}^n$ собственный вектор матрицы $\mathbf{B}_r = \mathbf{A}^\top \mathbf{A} \in \mathbb{R}^{n \times n}$ (данный вектор называется *правым сингулярным вектором* матрицы $\mathbf{A}$), где матрица $\mathbf{A} \in \mathbb{R}^{m \times n}$. Пусть $\lambda \neq 0$ является соответствующим собственным значением матрицы $\mathbf{B}_r$. Тогда по [определению собственных значения и вектора матрицы](LAB-5-Eigenvalues-Eigenvectors.md#def-eigen):

$$ \mathbf{B}_r \mathbf{v} = \left( \mathbf{A}^\top \mathbf{A} \right) \mathbf{v} = \lambda \mathbf{v}. $$

Умножим обе части уравнения на матрицу $\mathbf{A}$ слева:

$$ \mathbf{A} \mathbf{A}^\top \mathbf{A} \mathbf{v} = \lambda \mathbf{A} \mathbf{v}. $$

Пусть $\mathbf{A} \mathbf{v} = \sigma \mathbf{u}$ (вектор $\mathbf{u}$ называется *левым сингулярным вектором* матрицы $\mathbf{A}$), тогда:

$$ \left( \mathbf{A} \mathbf{A}^\top \right) \mathbf{u} = \mathbf{B}_l \mathbf{u} = \lambda \mathbf{u}. $$

То есть по определению собственных значения и вектора матрицы $\lambda$ является собственным значением матрицы $\mathbf{B}_l  \in \mathbb{R}^{m \times m}$, а $\mathbf{u} \in \mathbb{R}^m$ – ее собственным вектором.
```

Важно отметить, что поскольку матрицы $\mathbf{B}_r \in \mathbb{R}^{n \times n}$ и $\mathbf{B}_l \in \mathbb{R}^{m \times m}$ имеют один и тот же набор ненулевых собственных значений, то разница в размерностях данных матриц проявляется в количестве нулевых собственных значений.

Проиллюстрируем данную лемму следующим примером.

```{admonition} Пример
:class: exercise
Пусть матрица $\mathbf{A} \in \mathbb{R}^{3 \times 4}$ задана следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 1 & 3 & 14 & -2 \\ 5 & 2 & -7 & 0 \\ 4 & -2 & 7 & 3 \end{bmatrix}. $$

Необходимо найти собственные векторы матриц $\mathbf{A}^\top \mathbf{A}$ и $\mathbf{A} \mathbf{A}^\top$.
```

````{dropdown} Решение
Для работы с матрицами будем использовать библиотеку [numpy](https://numpy.org):

```python
import numpy as np
```

Зададим матрицу $\mathbf{A}$ в виде двумерного массива:

```python
A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])
```

Вычислим матрицы $\mathbf{B}_r = \mathbf{A}^\top \mathbf{A} \in \mathbb{R}^{4 \times 4}$ и $$\mathbf{B}_l = \mathbf{A} \mathbf{A}^\top\mathbb{R}^{3 \times 3}$:

```python
Br = A.T @ A
print(Br)
```

```{glue:} glued_text1
```

```python
Bl = A @ A.T
print(Bl)
```

```{glue:} glued_text2
```

С использованием функции [`numpy.linalg.eigvals`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html) рассчитаем и выведем собственные значения для обеих матриц:

```python
print(np.linalg.eigvals(Br))
```

```{glue:} glued_text3
```

```python
print(np.linalg.eigvals(Bl))
```

```{glue:} glued_text4
```

Из данного сопоставления видно, что собственные значения матрицы $\mathbf{B}_r$ повторяют собственные значения матрицы $\mathbf{B}_l$ за исключением одного нуля.
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])

Br = A.T @ A

glue('glued_text1', MultilineText(Br))

Bl = A @ A.T

glue('glued_text2', MultilineText(Bl))

glue('glued_text3', MultilineText(np.linalg.eigvals(Br)))

glue('glued_text4', MultilineText(np.linalg.eigvals(Bl)))
```

Если умножить равенство $\mathbf{A} \mathbf{v} = \sigma \mathbf{u}$ на $\mathbf{A}^\top$ слева, получим:

$$ \begin{align}
\mathbf{A}^\top \mathbf{A} \mathbf{v} &= \sigma \mathbf{A}^\top \mathbf{u}, \\
\mathbf{B}_r \mathbf{v} &= \sigma \mathbf{A}^\top \mathbf{u}, \\
\lambda \mathbf{v} &= \sigma \mathbf{A}^\top \mathbf{u}, \\
\sigma \mathbf{v} &= \mathbf{A}^\top \mathbf{u}.
\end{align} $$

Таким образом, умножив матрицу $\mathbf{A}$ на ее правый сингулярный вектор $\mathbf{v}$, получим ее левый сингулярный вектор $\mathbf{u}$, умноженный на сингулярное значение $\sigma$. И наоборот, умножив матрицу $\mathbf{A}^\top$ на левый сингулярный вектор $\mathbf{u}$, получим ее правый сингулярный вектор $\mathbf{v}$, умноженный на сингулярное значение $\sigma$:

$$ \begin{align}
\mathbf{A} \mathbf{v} &= \sigma \mathbf{u}, \\
\mathbf{A}^\top \mathbf{u} &= \sigma \mathbf{v}.
\end{align} $$

Следует отметить, что если рассматривается симметричная положительно полуопределенная матрица, то ее сингулярные значения равны собственным.

<a id='lemma-cond-singular-symmetric'></a>
```{admonition} Лемма
:class: caution
Для симметричной положительно полуопределенной матрицы ее сингулярные значения равны собственным значениям.
```

```{admonition} Доказательство
:class: proof
Если матрица $\mathbf{A} \in \mathbb{R}^{n \times n}$ является симметричной, то выражение для матрицы $\mathbf{B}$:

$$ \mathbf{B} = \mathbf{A}^\top \mathbf{A} = \mathbf{A} \mathbf{A}. $$

По определению сингулярное значение матрицы $\mathbf{A}$ представляет собой квадратный корень из собственного значения матрицы $\mathbf{B}$:

$$ \sigma_\mathbf{A} = \sqrt{\lambda_\mathbf{B}}. $$

В соответствии с доказанной [леммой о собственных значениях степени матрицы](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-power):

$$ \lambda_\mathbf{B} = \lambda_\mathbf{A}^2. $$

Тогда:

$$ \sigma_\mathbf{A} = \sqrt{\lambda_\mathbf{B}} = \sqrt{\lambda_\mathbf{A}^2} = \left| \lambda_\mathbf{A} \right| = \lambda_\mathbf{A}. $$

Поскольку собственные значения положительно полуопределенной матрицы [неотрицательны](LAB-6-MatrixDefiniteness.md), то модуль собственного значения матрицы равен самому собственному значению.
```

Перейдем к рассмотрению теоремы о максимальном растяжении и сжатии вектора в результате его линейного преобразования.

<a id='theorem-cond-stretching'></a>
```{admonition} Теорема
:class: danger
Максимальное растяжение произвольного единичного вектора в результате линейного преобразования равняется сингулярному значению матрицы этого линейного преобразования.
```

Доказательство данной теоремы будет похоже на получение [выражения для $L_2$-нормы конечномерной матрицы](LAB-2-Matrices.md#matrix-norm-2).

```{admonition} Доказательство
:class: proof
Максимальное растяжение вектора в результате линейного преобразования определяется его [длиной](LAB-1-Vectors.md#vector-length):

$$ \mathbf{A} \mathbf{v} = \mathbf{b} \implies \lVert \mathbf{b} \rVert_2 = \lVert \mathbf{A} \mathbf{v} \rVert_2, $$

где $\mathbf{A} \in \mathbb{R}^{n \times n}$ представляет собой матрицу линейного преобразования, применяемого к единичному вектору $\mathbf{v} \in \mathbb{R}^{n} \; : \; \mathbf{v}^\top \mathbf{v} = 1$, в результате которого получается вектор $\mathbf{b} \in \mathbb{R}^{n}$.

Для того чтобы определить максимальный коэффициент растяжения в результате линейного преобразования вектора рассмотрим квадрат его длины:

$$ \lVert \mathbf{b} \rVert_2^2 = \mathbf{b}^\top \mathbf{b} = \left( \mathbf{A} \mathbf{v} \right)^\top \left( \mathbf{A} \mathbf{v} \right). $$

Применим доказанное ранее [свойство транспонирования произведения](LAB-2-Matrices.md#theorem-matrix-transp-dot):

$$ \lVert \mathbf{b} \rVert_2^2 = \mathbf{b}^\top \mathbf{b} = \mathbf{v}^\top \mathbf{A}^\top \mathbf{A} \mathbf{v}. $$

Пусть $\mathbf{A}^\top \mathbf{A} = \mathbf{B}$. Матрица $\mathbf{B}$ является [симметричной](LAB-2-Matrices.md#def-matrix-symmetric). Следовательно:

* ее собственные значения [относятся ко множеству действительных чисел](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-symmetric-real);
* [существует ортонормированный базис, в котором симметричная матрица является диагональной](LAB-5-Eigenvalues-Eigenvectors.md#theorem-eigen-symmetric),
* перевод вектора из стандартного (канонического) базиса в ортонормированный, составленный из собственных векторов симметричной матрицы, [осуществляется с сохранением его длины](LAB-4-LinearTransformations.md#theorem-lintran-orthogonal).

Данные свойства будут применяться в процессе доказательства. Пусть столбцы матрицы $\mathbf{P}^\top \in \mathbb{R}^{n \times n}$ представляют собой базисные векторы ортонормированного базиса $\mathbb{B} = \begin{Bmatrix} u_1, \, u_2, \, \ldots, \, u_n \end{Bmatrix}$, в котором матрица $\mathbf{B}$ является диагональной:

$$ \mathbf{D} = \mathbf{P} \mathbf{B} \mathbf{P}^\top, $$

где $\mathbf{D} \in \mathbb{R}^{n \times n}$ является диагональным видом матрицы $\mathbf{B}$ в базисе $\mathbb{B}$, на главной диагонали которой расположены собственные значения матрицы $\mathbf{B}$.

Тогда любой вектор [можно](LAB-4-LinearTransformations.md#theorem-lintran-asmatrix) представить в виде суммы произведений координат данного вектора в этом базисе и базисных векторов:

$$ \mathbf{v} = \mathbf{P}^\top \mathbf{y}, $$

где $\mathbf{y} \in \mathbb{R}^{n}$ – координаты вектора $\mathbf{v}$ в базисе $\mathbb{B}$.

Подставим данное выражение в полученное ранее:

$$ \begin{align}
\lVert \mathbf{b} \rVert_2^2
&= \mathbf{b}^\top \mathbf{b} \\
&= \mathbf{v}^\top \mathbf{A}^\top \mathbf{A} \mathbf{v} \\
&= \mathbf{v}^\top \mathbf{B} \mathbf{v} \\
&= \left( \mathbf{P}^\top \mathbf{y} \right)^\top \mathbf{B} \left( \mathbf{P}^\top \mathbf{y} \right) \\
&= \mathbf{y}^\top \mathbf{P} \mathbf{B} \mathbf{P}^\top \mathbf{y} \\
&= \mathbf{y}^\top \mathbf{D} \mathbf{y} \\
&= \sum_{i=1}^n y_i^2 \lambda_i,
\end{align} $$

где $\lambda_i, \, i = 1 \, \ldots \, n,$ – неотрицательные собственные значения матрицы $\mathbf{B}$. Пусть $\lambda_1 > \lambda_2 > \ldots > \lambda_n$. Тогда справедливы следующие неравенства:

$$ \begin{align}
\sum_{i=1}^n y_i^2 \lambda_i &< \sum_{i=1}^n y_i^2 \lambda_1, \\
\sum_{i=1}^n y_i^2 \lambda_i &< \lambda_1 \sum_{i=1}^n y_i^2. \\
\end{align} $$

Поскольку перевод вектора из стандартного базиса, являющегося ортонормированным, в другой ортонормированный, составленный из собственных векторов симметричной матрицы, [осуществляется](LAB-4-LinearTransformations.md#theorem-lintran-orthogonal) с сохранением его длины, то:

$$ \sum_{i=1}^n v_i^2 = \sum_{i=1}^n y_i^2 = 1. $$

Следовательно,

$$ \lVert \mathbf{b} \rVert_2^2 < \lambda_1. $$

С учетом определения сингулярного значения матрицы $\mathbf{A}$: $\sigma_\mathbf{A} = \sqrt{\lambda_\mathbf{B}}$

$$ \lVert \mathbf{b} \rVert_2 < \sigma_1. $$

где $\sigma_1$ – наибольшее сингулярное значение матрицы $\mathbf{A}$.

Если матрица $\mathbf{A}$ является симметричной, то $\mathbf{B} = \mathbf{A}^\top \mathbf{A} = \mathbf{A} \mathbf{A}$. Следовательно, в соответствии с доказанным [ранее](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-power) собственные значения матрицы $\mathbf{B}$ являются квадратами соответствующих собственных значений матрицы $\mathbf{A}$. Тогда:

$$ \lVert \mathbf{b} \rVert_2 < \left| \lambda_1 \right|, $$

где $\lambda_1$ – наибольшее по модулю собственное значение матрицы $\mathbf{A}$ ([спектральный радиус](LAB-5-Eigenvalues-Eigenvectors.md#def-eigen-spectral-radius) матрицы $\mathbf{A}$).

Если же матрица $\mathbf{A}$ также является положительно полуопределенной, то:

$$ \lVert \mathbf{b} \rVert_2 < \lambda_1. $$

Таким образом, максимальное растяжение любого вектора в результате линейного преобразования равняется наибольшему сингулярному значению этой матрицы и происходит в направлении соответствующего ему *правого сингулярного вектора*.
```

Если максимальное растяжение вектора в результате линейного преобразования определяется наибольшим сингулярным значением его оператора, то минимальное растяжение – наименьшим. Кроме того, следует отметить, что если наибольшее собственное значение симметричной матрицы линейного преобразования меньше единицы, то в результате этого линейного преобразования не происходит растяжение вектора, а только его сжатие.

(condition)=
## Число обусловленности матрицы
