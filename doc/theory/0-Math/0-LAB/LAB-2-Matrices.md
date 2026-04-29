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

# Матрицы

(matrix)=
## Определение и объявление матрицы

<a id='matrix-def'></a>
```{admonition} Определение
:class: definition
***Матрица*** представляет собой упорядоченную совокупность векторов-столбцов (или векторов-строк).
```







(matrix-det)=
## Определитель матрицы

```{admonition} Определение
:class: definition
***Определителем матрицы*** будем называть числовую характеристику *квадратной* матрицы, являющуюся многочленом от ее элементов и определяющую такое свойство матрицы, как [обратимость](LAB-2-Matrices.md#matrix-inv): *матрица является обратимой, если ее определитель отличен от нуля*.
```

Для нахождения определителя матрицы можно использовать функцию [`numpy.linalg.det`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html).

Определитель матрицы первого порядка равен единственному элементу этой матрицы.

```{code-cell} python
A = np.array([[4 + 2j]])
print(np.linalg.det(A))
```

Определитель матрицы размерностью $\left( 2 \times 2 \right)$ вычисляется следующим образом:

$$ \begin{vmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{vmatrix} = a_{11} \cdot a_{22} - a_{12} \cdot a_{21}. $$

```{code-cell} python
A = np.array([
    [11 + 2j, -2 - 3j],
    [-3 + 1j, -4 + 4j],
])
print(np.linalg.det(A))
```

При нахождении определителя матрицы размерностью $\left( 3 \times 3 \right)$ необходимо "раскрыть" определитель по любой строке или столбцу с учетом матрицы знаков:

$$ \begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{vmatrix} = a_{11} \cdot \begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} - a_{12} \cdot \begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} + a_{13} \cdot \begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}. $$

```{code-cell} python
A = np.array([
    [1 + 3j, 3 - 1j, 14 - 5j],
    [5 + 6j, 2 - 2j, -7 + 9j],
    [4 + 2j, 2 - 5j, 11 + 4j],
])
print(np.linalg.det(A))
```

Если определитель матрицы, составленной из координат этих векторов, равен нулю, то данные векторы являются [*линейно зависимыми*](LAB-1-Vectors.md#vector-linear-dependence), а такая матрица называется *вырожденной*. Рассмотрим данное свойство на следующем примере.

```{admonition} Пример
:class: exercise
Пусть координаты векторов $\mathbf{a}, \, \mathbf{b}, \, \mathbf{c} \in \mathbb{R}^3$ определены следующим образом:

$$ \mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \; \mathbf{b} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}, \; \mathbf{c} = \begin{bmatrix} 6 \\ 9 \\ 12 \end{bmatrix}. $$

Являются ли данные векторы линейно независимыми?
```

````{dropdown} Решение
Анализируя координаты данных векторов, можно отметить, что они не являются коллинеарными, то есть не существует таких чисел $\lambda_1, \, \lambda_2, \, \lambda_3$, при которых были бы справедливы равенства $\mathbf{a} = \lambda_1 \mathbf{b}, \, \mathbf{b} = \lambda_2 \mathbf{c}, \, \mathbf{c} = \lambda_3 \mathbf{a}$. Однако при этом эти векторы все равно могут быть линейно зависимыми. Выполним проверку, вычислив определитель матрицы, составленной из координат данных векторов:

```python
M = np.array([
    [1., 4., 6.],
    [2., 5., 9.],
    [3., 6., 12.],
])
print(np.linalg.det(M))
```

```{glue:} det1
```

Определитель матрицы, составленный из координат данных векторов, равен нулю, следовательно, векторы $\mathbf{a}$, $\mathbf{b}$ и $\mathbf{c}$ являются линейно зависимыми.

Действительно, мы можем убедиться в этом, заметив, что:

$$ \mathbf{c} = 2 \mathbf{a} + \mathbf{b}. $$
````

```{code-cell} python
:tags: [remove-cell]

M = np.array([
    [1., 4., 6.],
    [2., 5., 9.],
    [3., 6., 12.],
])

glue('det1', MultilineText(np.linalg.det(M)))
```

Доказательство данного свойства основывается на приведении матрицы к верхнетреугольному виду, которое будет рассмотрено в [следующем разделе](LAB-3-LinearSystems.md).

(matrix-inv)=
## Обратная матрица

```{admonition} Определение
:class: definition
Матрица $\mathbf{A}^{-1} \in \mathbb{C}^{n \times n}$ называется ***обратной*** к матрице $\mathbf{A} \in \mathbb{C}^{n \times n}$, если их произведение (как слева, так и справа) дает единичную матрицу:

$$ \mathbf{A} \mathbf{A}^{-1} = \mathbf{I}, \; \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}, $$

где $\mathbf{I}$ – единичная матрица.
```

Для нахождении обратной матрицы аналитически зачастую используется [правило Крамера](https://en.wikipedia.org/wiki/Cramer%27s_rule):

$$ \mathbf{A}^{-1} = \frac{1}{\left| \mathbf{A} \right|} \cdot \mathbf{A}_{*}^\top, $$

где $\mathbf{A}_{*}^\top$ – транспонированная матрица алгебраических дополнений. Следует отметить, что, исходя из указанного выше выражения, необходимым условием для существования обратной матрицы является $\left| \mathbf{A} \right| \neq 0$.

Однако использование библиотеки [numpy](https://numpy.org/) может значительно упростить данную операцию. Для нахождения обратной матрицы удобно использовать функцию [`numpy.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html):

```{code-cell} python
A = np.array([
    [1 + 3j, 3 - 1j],
    [5 + 6j, 2 - 2j],
])
A_inv = np.linalg.inv(A)
print(A_inv)
```

Убедимся в том, что `A_inv` является обратной матрицей к матрице `A`:

```{code-cell} python
print(A @ A_inv)
```

Матрица, обратная к обратной матрице $\mathbf{A}$, равняется самой матрице $\mathbf{A}$:

$$ \left( \mathbf{A}^{-1} \right)^{-1} = \mathbf{A}. $$

Подтвердим данное свойство рассматриваемым примером:

```{code-cell} python
print(np.linalg.inv(A_inv))
```

В результате получилась матрица, элементы которой равны элементам исходной матрицы `A`.

(matrix-nilpotent)=
## Нильпотентная матрица

```{admonition} Определение
:class: definition
***Нильпотентной*** называется квадратная матрица $\mathbf{A} \in \mathbb{C}^{n \times n}$, для которой справедливо следующее выражение:

$$ \underbrace{\mathbf{A} \mathbf{A} \ldots \mathbf{A}}_\text{k раз} = \mathbf{A}^k = 0, \; k = 1 \, \ldots \, n. $$

Натуральное число $k$ называется показателем (или индексом) нильпотентности.
```

Покажем, что строго верхнетреугольная матрица $\mathbf{A} \in \mathbb{C}^{n \times n} \; : \; a_{ij} = 0, \; i \geq j, \; i = 1 \, \ldots n, \, j = 1 \, \ldots \, n,$ является нильпотентной.

Рассмотрим произведение строго верхнетреугольной матрицы $\mathbf{A}$ и единичного вектора $\mathbf{e}_1$. В результате этого действия получается первый столбец матрицы $\mathbf{A}$, равный нулевому вектору:

$$ \mathbf{A} \mathbf{e}_1 = \begin{bmatrix} 0 & a_{12} & a_{13} & \ldots & a_{1n} \\ 0 & 0 & a_{23} & \ldots & a_{2n} \\ 0 & 0 & 0 & \ldots & a_{3n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = \mathbf{0}. $$

Умножив левую и правую части на $\mathbf{A}^{n-1}$ слева, получим:

$$ \begin{align}
\mathbf{A}^{n-1} \mathbf{A} \mathbf{e}_1 &= \mathbf{A}^{n-1} \mathbf{0}, \\
\mathbf{A}^n \mathbf{e}_1 &= \mathbf{0}.
\end{align} $$

Рассмотрим произведение строго верхнетреугольной матрицы $\mathbf{A}$ и единичного вектора $\mathbf{e}_2$:

$$ \mathbf{A} \mathbf{e}_2 = \begin{bmatrix} 0 & a_{12} & a_{13} & \ldots & a_{1n} \\ 0 & 0 & a_{23} & \ldots & a_{2n} \\ 0 & 0 & 0 & \ldots & a_{3n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = \begin{bmatrix} a_{12} \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = a_{12} \mathbf{e}_1. $$

Если умножим левую и правую части данного равенства на $\mathbf{A}$ слева и воспользуемся свойством коммутативности [умножения матрицы на число](LAB-2-Matrices.md#matrix-mult), то получим:

$$ \begin{align}
\mathbf{A} \mathbf{A} \mathbf{e}_2 &= \mathbf{A} a_{12} \mathbf{e}_1, \\
\mathbf{A}^2 \mathbf{e}_2 &= a_{12} \mathbf{A} \mathbf{e}_1, \\
\mathbf{A}^2 \mathbf{e}_2 &= a_{12} \mathbf{0}, \\
\mathbf{A}^2 \mathbf{e}_2 &= \mathbf{0}. \\
\end{align} $$

Умножим левую и правую части полученного соотношения на $\mathbf{A}^{n-2}$ слева:

$$ \begin{align}
\mathbf{A}^2 \mathbf{e}_2 &= \mathbf{0}, \\
\mathbf{A}^{n-2} \mathbf{A}^2 \mathbf{e}_2 &= \mathbf{A}^{n-2} \mathbf{0}, \\
\mathbf{A}^n \mathbf{e}_2 &= \mathbf{0}.
\end{align} $$

Для наглядности получим аналогичный результат и для третьего единичного вектора:

$$ \mathbf{A} \mathbf{e}_3 = \begin{bmatrix} 0 & a_{12} & a_{13} & \ldots & a_{1n} \\ 0 & 0 & a_{23} & \ldots & a_{2n} \\ 0 & 0 & 0 & \ldots & a_{3n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix} = \begin{bmatrix} a_{13} \\ a_{23} \\ 0 \\ \vdots \\ 0 \end{bmatrix} = a_{13} \mathbf{e}_1 + a_{23} \mathbf{e}_2. $$

Умножим левую и правую части данного соотношения на $\mathbf{A}^2$ слева:

$$ \begin{align}
\mathbf{A}^2 \mathbf{A} \mathbf{e}_3 &= a_{13} \mathbf{A}^2 \mathbf{e}_1 + a_{23} \mathbf{A}^2 \mathbf{e}_2, \\
\mathbf{A}^3 \mathbf{e}_3 &= a_{13} \mathbf{A} \mathbf{0} + a_{23} \mathbf{0}, \\
\mathbf{A}^3 \mathbf{e}_3 &= \mathbf{0}.
\end{align} $$

Теперь умножим на $\mathbf{A}^{n-3}$ для получения матрицы $\mathbf{A}^n$:

$$ \begin{align}
\mathbf{A}^3 \mathbf{e}_3 &= \mathbf{0}, \\
\mathbf{A}^{n-3} \mathbf{A}^3 \mathbf{e}_3 &= \mathbf{A}^{n-3} \mathbf{0}, \\
\mathbf{A}^n \mathbf{e}_3 &= \mathbf{0}.
\end{align} $$

Данные действия можно продолжать вплоть до последнего единичного вектора. Пусть $\mathbf{M} = \mathbf{A}^n$, а ее $i$-й столбец $\mathbf{m}_i = \mathbf{M} \mathbf{e}_i, \, i = 1 \, \ldots n$. Тогда

$$ \begin{align}
& \mathbf{A}^n \mathbf{e}_1 = \mathbf{M} \mathbf{e}_1 = \mathbf{m}_1 = \mathbf{0}, \\
& \mathbf{A}^n \mathbf{e}_2 = \mathbf{M} \mathbf{e}_2 = \mathbf{m}_2 = \mathbf{0}, \\
& \mathbf{A}^n \mathbf{e}_3 = \mathbf{M} \mathbf{e}_3 = \mathbf{m}_3 = \mathbf{0}, \\
& \vdots \\
& \mathbf{A}^n \mathbf{e}_n = \mathbf{M} \mathbf{e}_n = \mathbf{m}_n = \mathbf{0}.
\end{align} $$

Таким образом, все столбцы $\mathbf{m}_i = \mathbf{0}, \, i = 1 \, \ldots n,$ следовательно, матрица $\mathbf{M} = \mathbf{A}^n$ является нулевой матрицей, а строго верхнетреугольная матрица $\mathbf{A}$ – нильпотентной.
