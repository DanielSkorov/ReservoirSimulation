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

# Собственные векторы и значения матриц

Пусть дана квадратная матрица:

$$ \mathbf{A} = \begin{bmatrix} -1 & -6 \\ 2 & 6 \end{bmatrix}. $$

Умножим матрицу $\mathbf{A}$ на вектор $\mathbf{u} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}$:

$$ \mathbf{A} \mathbf{u} = \begin{bmatrix} -1 & -6 \\ 2 & 6 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix}=\begin{bmatrix} -1 \cdot 2 + (-6) \cdot (-1) \\ 2 \cdot 2 + 6 \cdot (-1) \end{bmatrix} = \begin{bmatrix} 4 \\ -2 \end{bmatrix} = 2 \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \lambda \mathbf{u}. $$

В результате умножения матрицы $\mathbf{A}$ на вектор $\mathbf{u}$ получился тот же самый вектор $\mathbf{u}$ с числовым коэффициентом $\lambda = 2$:

$$ \mathbf{A} \mathbf{u} = \lambda \mathbf{u}. $$

Такой вектор $\mathbf{u}$ называется ***собственным вектором*** (*eigenvector*) матрицы $\mathbf{A}$, а $\lambda$ – ***собственным значением*** матрицы $\mathbf{A}$ (*eigenvalue*).

<a id='def-eigen'></a>
```{admonition} Определение
:class: tip
Ненулевой вектор $\mathbf{u} \in \mathbb{R}^{n}$, который при умножении на некоторую квадратную матрицу $\mathbf{A} \in \mathbb{R}^{n \times n}$ преобразуется в самого же себя с числовым коэффициентом $\lambda$, называется ***собственным вектором*** матрицы $\mathbf{A}$, а число $\lambda$ – ***собственным значением*** матрицы $\mathbf{A}$.
```

(eigen-characteristic-polynomial)=
## Характеристическое уравнение матрицы

Рассмотрим квадратную матрицу $\mathbf{A} \in \mathbb{R}^{n \times n}$. Запишем определение для ее собственного вектора $\mathbf{u} \in \mathbb{R}^{n}$ следующим образом:

$$ \mathbf{A} \mathbf{u} - \lambda \mathbf{u} = 0, $$

$$ \begin{bmatrix} \mathbf{A} - \lambda \mathbf{I} \end{bmatrix} \mathbf{u} = 0, $$

где $\mathbf{I} \in \mathbb{R}^{n \times n}$ представляет собой единичную матрицу.

Поскольку тривиальное решение данного уравнения не удовлетворяет условию, указанному в определении собственного вектора $\left( \mathbf{u} \neq 0 \right)$, то необходимо, чтобы:

$$ \det \left( \mathbf{A} - \lambda \mathbf{I} \right) = 0. $$

Данное уравнение называется *характеристическим* для матрицы $\mathbf{A}$, позволяющим определить ее собственные значения $\lambda$. Последующее определение собственных векторов $\mathbf{u}$ основано на решении уравнения $\begin{bmatrix} \mathbf{A} - \lambda \mathbf{I} \end{bmatrix} \mathbf{u} = 0$ относительно $\mathbf{u}$.

```{admonition} Доказательство
:class: proof
Предположим, что $\det \left( \mathbf{A} - \lambda \mathbf{I} \right) \neq 0$. Следовательно, существует такая обратная матрица, что:

$$ \begin{align}
\begin{bmatrix} \mathbf{A} - \lambda \mathbf{I} \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{A} - \lambda \mathbf{I} \end{bmatrix} \mathbf{u} &= \begin{bmatrix} \mathbf{A} - \lambda \mathbf{I} \end{bmatrix}^{-1} \cdot 0, \\
\mathbf{I} \mathbf{u} &= 0,
\end{align} $$

откуда следует:

$$ \mathbf{u} = 0, $$

что противоречит условию $\mathbf{u} \neq 0$.
```

Характеристическое уравнение матрицы $\mathbf{A}$, записанное через определитель, также можно представить в виде полинома:

$$ p_n \left( \lambda \right) = \det \left( \mathbf{A} - \lambda \mathbf{I} \right) = c_{n} \lambda^{n} + c_{n-1} \lambda^{n-1} + \ldots + c_1 \lambda + c_0 = 0, $$

где $c_0, \, c_1, \, \ldots, \, c_{n}$ – коэффициенты полиномиального уравнения, выражаемые через элементы матрицы при раскрытии определителя (при раскрытии определителя используются только операции сложения, вычитания и умножения, что, в конечном итоге, приводит к полиномиальной форме выражения).

По [основной теореме алгебры](https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra) любой полином может быть представлен в виде произведения:

$$ p_n \left( \lambda \right) = \left( -1 \right)^n \left( \lambda - \lambda_1 \right) \cdot \left( \lambda - \lambda_2 \right) \cdot \ldots \cdot \left( \lambda - \lambda_n \right) = 0, $$

где $\lambda_1, \, \lambda_2, \, \ldots, \, \lambda_n$ – корни этого полинома, принадлежащие пространству [комплексных чисел](https://en.wikipedia.org/wiki/Complex_number) (пространство действительных чисел входит в пространство комплексных чисел и характеризуется нулевой мнимной частью). Строго говоря, все дальнейшие утверждения, сформулированные для матриц, как таковых ([без указания на их симметричность](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-real) или на определенные допущения), должны быть записаны для пространства комплексных чисел, поскольку именно для него сформулирована основная теорема алгебры о равенстве степени и количества корней полинома с учетом их кратности.

Рассмотрим пример аналитического определения собственных значений и векторов матрицы.

```{admonition} Пример
:class: exercise

Пусть дана матрица $\mathbf{A}$:

$$ \mathbf{A} = \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix}. $$

Необходимо найти собственные значения и собственные векторы матрицы $\mathbf{A}$.
```

````{dropdown} Решение
Для начала найдем собственные значения. Запишем уравнение и решим его относительно $\lambda$:

$$ \begin{align}
\begin{vmatrix} \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - \lambda \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\end{vmatrix} &= 0 \\ \begin{vmatrix} 4 - \lambda & -5 \\ 2 & -3 - \lambda \end{vmatrix} &= 0 \\ (4 - \lambda) \cdot (-3 - \lambda) + 10 &= 0 \\ {\lambda}^2 - \lambda - 2 &= 0 \\ {\lambda}_{1,2} &= (-1, 2)
\end{align} $$

При $\lambda = -1$:

$$ \begin{align}
\left( \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - (-1) \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \cdot \mathbf{u} &= 0 \\ \begin{bmatrix} 5 & -5 \\ 2 & -2 \end{bmatrix} \cdot \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &= 0.
\end{align} $$

Данное выражение может быть преобразовано в систему линейных уравнений и решено с использованием [метода Гаусса](LAB-3-LinearSystems.md#elimination). Его целью является получение "треугольной матрицы нулей" путем простейших математических преобразований – сложения и умножения. В результате, получится следующее выражение:

$$ \begin{array}{cc|c} 1 & -1 & 0 \\ 0 & 0 & 0 \end{array} $$

Исходя из второй строчки, $u_2$ может принимать любые значения. Поэтому пусть $u_2 = 1$. Тогда $u_1 = 1$. Отнормируем $u_1$ и $u_2$ на величину $\sqrt{{u_1}^2 + {u_2}^2}$ исключительно для сопоставления с результатом, получаемым в [numpy](https://numpy.org/) (далее будет показано, что данная операция, по сути, не играет существенной роли). Тогда $u_1 = u_2 = \frac{1}{\sqrt{2}}$.

При $\lambda = 2$:

$$ \begin{align} \left( \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix} - 2 \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \cdot u &= 0 \\ \begin{bmatrix} 2 & -5 \\ 2 & -5 \end{bmatrix} \cdot \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &= 0 \end{align} $$

После применения метода Гаусса получим:

$$ \begin{array}{cc|c} 2 & -5 & 0 \\ 0 & 0 & 0 \end{array} $$

Исходя из второй строчки, $u_2$ может принимать любые значения. Поэтому пусть $u_2 = 1$. Тогда $u_1 = \frac{5}{2}$. Аналогично отнормируем $u_1$ и $u_2$ на величину $\sqrt{{u_1}^2 + {u_2}^2}$. Тогда $u_1 = \frac{5}{\sqrt{29}}$ и $u_2 = \frac{2}{\sqrt{29}}$.

Проверим правильность решения с использованием [numpy](https://numpy.org/):

```python
import numpy as np
```

Зададим найденные собственные значения в виде одномерного массива и соответствующие им собственные векторы в виде двумерного массива:

``` python
lmbdi = np.array([2., -1.])
U = np.array([
    [5 / 29**0.5, 1 / 2**0.5],
    [2 / 29**0.5, 1 / 2**0.5],
])
print(lmbdi, U, sep='\n')
```

```{glue:} glued_out1
```

Полученные результаты полностью совпадают с расчетом, выполненным [numpy](https://numpy.org/):

``` python
A = np.array([[4., -5.], [2., -3.]])
print(np.linalg.eig(A))
```

```{glue:} glued_out2
```

Выполним проверку:

``` python
lmbd = lmbdi[0]
u = U[:, 0]
print(A.dot(u) - lmbd * u)
```

```{glue:} glued_out3
```

``` python
lmbd = lmbdi[1]
u = U[:, 1]
print(A.dot(u) - lmbd * u)
```

```{glue:} glued_out4
```

Также проверим правильность подобранных значений $\mathbf{u}$ без нормирования:

``` python
lmbd = 2.
u = np.array([5. / 2., 1.])
print(A.dot(u) - lmbd * u)
```

```{glue:} glued_out5
```

``` python
lmbd = -1.
u = np.array([1., 1.])
print(A.dot(u) - lmbd * u)
```

```{glue:} glued_out6
```

Кроме того, правильность решения не зависит от выбора значения $u_2$. При $\lambda = 2$ предположим, что $u_2 = 2$. Тогда $u_1 = 5$.

``` python
lmbd = 2.
u = np.array([5., 2.])
print(A.dot(u) - lmbd * u)
```

```{glue:} glued_out7
```

Также необходимо отметить, что найденные собственные векторы матрицы являются [линейно независимыми](LAB-3-LinearSystems.md#def-ls-linear-independence) – [определитель](LAB-2-Matrices.md#matrix-det) матрицы, составленный из их координат, отличен от нуля:

``` python
print(np.linalg.det(U))
```

```{glue:} glued_out8
```
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

lmbdi = np.array([2., -1.])
U = np.array([
    [5 / 29**0.5, 1 / 2**0.5],
    [2 / 29**0.5, 1 / 2**0.5],
])

glue('glued_out1', MultilineText(lmbdi, U, sep='\n'))

A = np.array([[4., -5.], [2., -3.]])

glue('glued_out2', MultilineText(np.linalg.eig(A)))

lmbd = lmbdi[0]
u = U[:, 0]

glue('glued_out3', MultilineText(A.dot(u) - lmbd * u))

lmbd = lmbdi[1]
u = U[:, 1]

glue('glued_out4', MultilineText(A.dot(u) - lmbd * u))

lmbd = 2.
u = np.array([5. / 2., 1.])

glue('glued_out5', MultilineText(A.dot(u) - lmbd * u))

lmbd = -1.
u = np.array([1., 1.])

glue('glued_out6', MultilineText(A.dot(u) - lmbd * u))

lmbd = 2.
u = np.array([5., 2.])

glue('glued_out7', MultilineText(A.dot(u) - lmbd * u))

glue('glued_out8', MultilineText(np.linalg.det(U)))
```

<a id='eigen-complex'></a>
```{dropdown} Свойства и определения линейной алгебры в комплексном пространстве
Поскольку основная теорема алгебры гарантирует существование как минимум одного комплексного корня характеристического уравнения матрицы, то необходимо отметить ряд свойств и определений, используемых в линейной алгебре при рассмотрении комплексного пространства. Данные избранные понятия будут использоваться для доказательства двух важных лемм о [приведении матрицы к верхнетреугольной форме](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-schur) и о [действительности собственных значений симметричных матриц](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-real).

Итак, вместо ранее определенного [скалярного произведения векторов](LAB-1-Vectors.md#vector-dot) необходимо ввести понятие сопряженного (эрмитового) скалярного произведения, являющегося обобщением скалярного произведения на пространство комплексных чисел.

Сопряженным скалярным произведением двух векторов $\mathbf{a} \in \mathbb{C}^n$ и $\mathbf{b} \in \mathbb{C}^n$ называется комплексное число, равное сумме произведений соответствующих элементов [сопряженно](https://en.wikipedia.org/wiki/Complex_conjugate) транспонированного вектора $\mathbf{a}^\dagger$ и вектора $\mathbf{b}$:

$$ \mathbf{a}^\dagger \mathbf{b} = \sum_{i=1}^n \bar{a}_i b_i, $$

где символом $\bar{a}_i$ обозначено сопряженное число к $i$-му элементу вектора $\mathbf{a}$.

В пространстве комплексных чисел, скалярное произведение векторов:

* сопряженно симметрично: $\mathbf{a}^\dagger \mathbf{b} = \bar{\mathbf{b}}^\dagger \mathbf{a}$,
* сопряженно ассоциативно со скаляром к первому вектору: $\left( \lambda \mathbf{a} \right)^\dagger \mathbf{b} = \bar{\lambda} \left( \mathbf{a}^\dagger \mathbf{b} \right)$,
* ассоциативно со скаляром ко второму вектору: $\mathbf{a}^\dagger \left( \lambda \mathbf{b} \right) = \lambda \left( \mathbf{a}^\dagger \mathbf{b} \right)$,
* дистрибутивно: $\mathbf{a}^\dagger \left( \mathbf{b} + \mathbf{c} \right) = \mathbf{a}^\dagger \mathbf{b} + \mathbf{a}^\dagger \mathbf{c}$.

Длина вектора в пространстве комплексных чисел вычисляется так же, как и [для пространства действительных чисел](LAB-1-Vectors.md#vector-length), за исключением того, что вместо скалярного произведения вектора на самого себя используется сопряженное скалярное произведение:

$$ \lVert \mathbf{a} \rVert_2 = \sqrt{\mathbf{a}^\dagger \mathbf{a}}. $$

Операция сопряженного транспонирования, по сути, состоит из двух действий: из нахождения сопряженных чисел для элементов вектора (или матрицы) и транспонирования. Следовательно, для этой операции будут характерны рассмотренные ранее свойства [транспонирования](LAB-2-Matrices.md#matrix-transp):

* дистрибутивность относительно сложения: $\left( \mathbf{A} + \mathbf{B} \right)^\dagger = \mathbf{A}^\dagger + \mathbf{B}^\dagger$,
* сопряженная дистрибутивность относительно умножения на скаляр: $\left( \lambda \mathbf{A} \right)^\dagger = \bar{\lambda} \mathbf{A}^\dagger$,
* закон обратного порядка: $\left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right)^\dagger = \mathbf{A}_n^\dagger \, \ldots \, .\mathbf{A}_2^\dagger \mathbf{A}_1^\dagger$.

Матричным произведением $\mathbf{A} \in \mathbb{C}^{n \times m}$ и $\mathbf{B} \in \mathbb{C}^{m \times l}$ называется матрица $\mathbf{C} \in \mathbb{C}^{n \times l} \, : \, \begin{Bmatrix} c_{ij} = \sum_{k=1}^m a_{ik} b_{kj} \end{Bmatrix}$. Матричная запись этой операции: $\mathbf{A} \mathbf{B} = \mathbf{C}$. Свойства матричного произведения, записанного для пространства коплексных чисел, аналогичны свойствам [матричного произведения, записанного для пространства действительных чисел](LAB-2-Matrices.md#matrix-dot).

Кроме того, следует ввести понятие унитарной матрицы, которое обобщает понятие [ортогональных](LAB-2-Matrices.md#matrix-orthogonal) матриц на пространство комплексных чисел. Квадратная матрица $\mathbf{U} \in \mathbb{C}^{n \times n}$ называется унитарной, если ее сопряженно транспонированная матрица $\mathbf{U}^\dagger$ равняется ее обратной матрице $\mathbf{U}^{-1}$:

$$ \mathbf{U}^\dagger \mathbf{U} = \mathbf{U} \mathbf{U}^\dagger = \mathbf{I}. $$

При этом векторы-столбцы унитарной матрицы $\left( \mathbf{u}_1, \, \mathbf{u}_2, \, \ldots, \, \mathbf{u}_n \right)$, как и ортогональной, имеют длину, равную единице, и ортогональны друг другу.

Рассмотренное ранее [преобразование Хаусхолдера](LAB-4-LinearTransformations.md#lintran-hauseholder) в пространстве действительных чисел может быть распространено на пространство комплексных чисел путем замены операции транспонирования на сопряженное транспонирование.

Важно отметить, что здесь перечислены лишь избранные свойства и определения линейной алгебры в комплексном пространстве, необходимые для доказательства существования разложения Шура и действительности собственных значений симметричных матриц, которые будут рассмотрены далее в этом разделе.
```

(eigen-spec)=
## Спектр и спектральный радиус матрицы

<a id='def-eigen-spec'></a>
```{admonition} Определение
:class: tip
***Спектром*** квадратной матрицы называется совокупность (множество) всех ее собственных значений.
```

Спектр матрицы обычно обозначают следующим образом: $\mathrm{spec} \left( \cdot \right)$ или $\Lambda \left( \cdot \right)$.

Рассмотрим ряд важных свойств и теорем, касающихся спектра матрицы, которые будем использовать впоследствии.

```{admonition} Свойство
:class: note
Произведение элементов спектра матрицы равняется ее [определителю](LAB-2-Matrices.md#matrix-det) и свободному члену характеристического уравнения.
```

```{admonition} Доказательство
:class: proof
Данное свойство вытекает из равенства форм записи характеристического уравнения матрицы:

$$ \begin{align}
p_n \left( \lambda \right)
&= \det \left( \mathbf{A} - \lambda \mathbf{I} \right) \\
&= c_{n} \lambda^{n} + c_{n-1} \lambda^{n-1} + \ldots + c_1 \lambda + c_0 \\
&= \left( -1 \right)^n \left( \lambda - \lambda_1 \right) \cdot \left( \lambda - \lambda_2 \right) \cdot \ldots \cdot \left( \lambda - \lambda_n \right).
\end{align} $$

Если принять $\lambda = 0$, тогда:

$$ p_n \left( \lambda = 0 \right) = \det \left( \mathbf{A} \right) = c_0 = \prod_{i=1}^n \lambda_i. $$
```

Из этого свойства также следует, что матрица является [обратимой](LAB-2-Matrices.md#matrix-inv), если все ее собственные значения отличны от нуля.

<!-- ```{admonition} Свойство
:class: note
Сумма элементов спектра матрицы равняется ее [следу](LAB-2-Matrices.md#matrix-trace).
```

```{admonition} Доказательство
:class: proof
...
``` -->

<a id='lemma-eigen-power'></a>
```{admonition} Лемма
:class: caution
Если $\lambda$ является собственным значением матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$, а $\mathbf{u}$ – соответствующий ему собственный вектор $\left( \mathbf{A} \mathbf{u} = \lambda \mathbf{u} \right)$, то для любого целого $k \geq 1$:

* $\lambda^k$ является собственным значением матрицы $\mathbf{A}^k$;
* $\mathbf{u}$ является собственным вектором матрицы $\mathbf{A}^k$.
```

```{admonition} Доказательство
:class: proof
Для доказательства данной теоремы воспользуемся [методом математической индукции](https://en.wikipedia.org/wiki/Mathematical_induction). Данный подход к математическому доказательству зачастую используется, когда необходимо доказать истинность утверждения для всех натуральных чисел (в данном случае такая последовательность будет задаваться степенью $k$). Метод математической индукции состоит из двух этапов:

1. База индукции: осуществляется проверка истинности утверждения для минимального значения последовательности (в данном случае $k = 1$).
2. Индукционный переход: утверждение для произвольного $n$ принимается истинным, доказывается истинность утверждения для $\left( n + 1 \right)$.

Выполнив доказательство истинности утверждения для $\left( n + 1 \right)$ с учетом его истинности как для минимального, так и для произвольного $n$ делается вывод об истинности утверждения для всей последовательности натуральных чисел.

**База индукции**

Для $k = 1 \; : \; \mathbf{A}^1 \mathbf{u} = \lambda^1 \mathbf{u}$ верно по условию.

**Индукционный переход**

Пусть доказываемое утверждение верно для $k$:

$$ \mathbf{A}^k \mathbf{u} = \lambda^k \mathbf{u}. $$

Докажем его истинность для $\left( k + 1 \right)$. Умножим левую и правую часть равенства на $\mathbf{A}$ слева:

$$ \mathbf{A} \mathbf{A}^k \mathbf{u} = \lambda^k \mathbf{A} \mathbf{u}. $$

Применим определение собственных вектора и значения матрицы $\mathbf{A}$ к правой части равенства:

$$ \mathbf{A}^{k+1} \mathbf{u} = \lambda^k \lambda \mathbf{u} = \lambda^{k+1} \mathbf{u}. $$

Таким образом, по определению, число $\lambda^{k+1}$ и вектор $\mathbf{u}$ являются соответственно собственными значением и вектором матрицы $\mathbf{A}^{k+1}$.

Важно отметить, что если рассматриваемая матрица $\mathbf{A}$ [невырожденная](LAB-2-Matrices.md#matrix-det), то для $k=-1$ данное утверждение также сохраняется:

$$ \begin{align}
\mathbf{A} \mathbf{u} &= \lambda \mathbf{u}, \\
\mathbf{u} &= \mathbf{A}^{-1} \lambda \mathbf{u}, \\
\lambda^{-1} \mathbf{u} &= \mathbf{A}^{-1} \mathbf{u}.
\end{align} $$

То есть собственным значением матрицы $\mathbf{A}^{-1}$ является $\lambda^{-1}$.
```

Таким образом, при возведении квадратной матрицы в степень (умножении на саму себя), ее собственные векторы остаются теми же самыми, а собственные значения также возводятся в эту степень. Представленную лемму можно рассматривать как частный случай теоремы о спектральном преобразовании *(spectral mapping theorem)*.

<a id='theorem-eigen-function'></a>
```{admonition} Теорема
:class: danger
Обозначим произвольную квадратную матрицу $\mathbf{A} \in \mathbb{C}^{n \times n}$, а ее спектр – $\Lambda \left( \mathbf{A} \right)$. Пусть существует некоторая функция $f \left( \cdot \right)$, определенная для матрицы $\mathbf{A}$ и ее спектра. Тогда спектром матрицы $f \left( \mathbf{A} \right)$ является $f \left( \Lambda \left( \mathbf{A} \right) \right)$:

$$ \Lambda \left( f \left( \mathbf{A} \right) \right) = f \left( \Lambda \left( \mathbf{A} \right) \right). $$
```

Рассмотрим доказательство данной теоремы в предположении, что функция представляет собой полином произвольной степени.

```{admonition} Доказательство
:class: proof
Рассмотрим произведение матрицы $f \left( \mathbf{A} \right) \in \mathbb{C}^{n \times n}$ и собственного вектора $\mathbf{v} \in \mathbb{C}^n$ матрицы $\mathbf{A}$:

$$ \begin{align}
f \left( \mathbf{A} \right) \mathbf{v}
&= \left( c_0 \mathbf{I} + c_1 \mathbf{A} + \ldots + c_k \mathbf{A}^k \right) \mathbf{v} \\
&= c_0 \mathbf{I} \mathbf{v} + c_1 \mathbf{A} \mathbf{v} + \ldots + c_k \mathbf{A}^k \mathbf{v} \\
&= c_0 \mathbf{v} + c_1 \lambda \mathbf{v} + \ldots + c_k \lambda^k \mathbf{v} \\
&= \left( c_0 + c_1 \lambda + \ldots + c_k \lambda^k \right) \mathbf{v} \\
&= f \left( \lambda \right) \mathbf{v}.
\end{align} $$

Таким образом, по определению, собственными значением и вектором матрицы $f \left( \mathbf{A} \right)$ являются $f \left( \lambda \right)$ и $\mathbf{v}$ соответственно.
```

<a id='def-eigen-spectral-radius'></a>
```{admonition} Определение
:class: tip
***Спектральным радиусом*** квадратной матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$ называется наибольшее по модулю ее собственное значение:

$$ \rho \left( \mathbf{A} \right) = \max_i \left| \lambda_i \right|, $$

где $\lambda_i, \, i = 1 \, \ldots \, n,$ – $i$-е собственное значение матрицы $\mathbf{A}$.
```

Теперь перейдем к доказательству важного свойства спектрального радиуса, используемого при анализе сходимости и устойчивости итерационных алгоритмов. Прежде рассмотрим лемму о представлении любой квадратной матрицы в верхнетреугольном виде.

<a id='lemma-eigen-schur'></a>
```{admonition} Лемма
:class: caution
Любая квадратная матрица $\mathbf{A} \in \mathbb{C}^{n \times n}$ может быть представлена в виде:

$$ \mathbf{A} = \mathbf{U} \mathbf{T} \mathbf{U}^\dagger \Leftrightarrow \mathbf{T} = \mathbf{U}^\dagger \mathbf{A} \mathbf{U}, $$

где $\mathbf{U} \in \mathbb{C}^{n \times n}$ является унитарной матрицей; $\mathbf{T} \in \mathbb{C}^{n \times n}$ – верхнетреугольной матрицей, на главной диагонали которой находятся собственные значения исходной матрицы $\mathbf{A}$. Данное представление матрицы называют [разложением Шура](https://en.wikipedia.org/wiki/Schur_decomposition).
```

```{admonition} Доказательство
:class: proof
Для доказательства данной леммы воспользуемся [методом математической индукции](https://en.wikipedia.org/wiki/Mathematical_induction). Последовательность натуральных чисел будет задаваться размерностью $n$ матрицы $\mathbf{A}$.

**База индукции**

Матрица размерностью $\mathbf{A} \in \mathbb{C}^{1 \times 1} = \begin{bmatrix} a \end{bmatrix}$ уже является верхнетреугольной с собственным значением $a$ и матрицей $\mathbf{Q} = \begin{bmatrix} 1 \end{bmatrix}$.

**Индукционный переход**

Пусть утверждение о существовании разложения Шура истинно для всех матриц размерностью $\left( n - 1 \right) \times \left( n - 1 \right)$. Покажем, что оно также справедливо и для матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$. При этом у такой матрицы существует как минимум один единичный собственный вектор $\mathbf{v}_1 \in \mathbb{C}^{n}$ и соответствующее ему собственное значение $\lambda_1$:

$$ \mathbf{A} \mathbf{v}_1 = \lambda_1 \mathbf{v}_1. $$

Используя [преобразование Хаусхолдера](LAB-4-LinearTransformations.md#lintran-hauseholder) или [процесс Грама-Шмидта](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process), на основе собственного вектора $\mathbf{v}_1$ можно создать ортонормированный базис $\mathrm{B} = \begin{Bmatrix} \mathbf{v}_1, \, \mathbf{u_2}, \, \ldots, \, \mathbf{u}_n \end{Bmatrix}$. Это означает, что скалярные произведения различных базисных векторов такого базиса равны нулю, а скалярные произведения базисных векторов на самих себя равны единице:

$$ \begin{align}
\mathbf{v}_1^\dagger \mathbf{u}_j &= 0, \; j = 2 \, \ldots \, n, \\
\mathbf{u}_j^\dagger \mathbf{u}_k &= 0, \; j \neq k, \; j = 2 \, \ldots \, n, \; k = 2 \, \ldots \, n, \; \\
\mathbf{v}_1^\dagger \mathbf{v}_1 &= 1, \\
\mathbf{u}_j^\dagger \mathbf{u}_j &= 1, \; j = 2 \, \ldots \, n.
\end{align} $$

Пусть матрица $\mathbf{P} \in \mathbb{C}^{n \times n}$ составлена из векторов базиса $\mathrm{B}$:

$$ \mathbf{P} = \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{v}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix}. $$

Рассмотрим произведение

$$ \begin{align}
\mathbf{P}^\dagger \mathbf{A} \mathbf{P}
&= \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{v}_1^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_2^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_n^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \mathbf{A} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{v}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix} \\
&= \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{v}_1^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_2^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_n^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{A} \mathbf{v}_1 & \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{A} \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{v}_1^\dagger \mathbf{A} \mathbf{v}_1 & \mathbf{v}_1^\dagger \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{v}_1^\dagger \mathbf{A} \mathbf{u}_n \\
\mathbf{u}_2^\dagger \mathbf{A} \mathbf{v}_1 & \mathbf{u}_2^\dagger \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_2^\dagger \mathbf{A} \mathbf{u}_n \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{u}_n^\dagger \mathbf{A} \mathbf{v}_1 & \mathbf{u}_n^\dagger \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_n^\dagger \mathbf{A} \mathbf{u}_n
\end{bmatrix}
\end{align} $$

Распишем подробнее элементы получившейся матрицы.

$$ \begin{align}
\mathbf{v}_1^\dagger \mathbf{A} \mathbf{v}_1 &= \mathbf{v}_1^\dagger \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{v}_1^\dagger \mathbf{v}_1 = \lambda_1, \\
\mathbf{u}_2^\dagger \mathbf{A} \mathbf{v}_1 &= \mathbf{u}_2^\dagger \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{u}_2^\dagger \mathbf{v}_1 = 0, \\
\mathbf{u}_n^\dagger \mathbf{A} \mathbf{v}_1 &= \mathbf{u}_n^\dagger \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{u}_n^\dagger \mathbf{v}_1 = 0.
\end{align} $$

Таким образом, в результате произведения $\mathbf{P}^\dagger \mathbf{A} \mathbf{P}$ получается следующая матрица:

$$ \mathbf{P}^\dagger \mathbf{A} \mathbf{P}
= \left[
\begin{array}{c|ccc}
\lambda_1 & \mathbf{v}_1^\dagger \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{v}_1^\dagger \mathbf{A} \mathbf{u}_n \\
\hline
0 & \mathbf{u}_2^\dagger \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_2^\dagger \mathbf{A} \mathbf{u}_n \\
\vdots & \vdots & \ddots & \vdots \\
0 & \mathbf{u}_n^\dagger \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_n^\dagger \mathbf{A} \mathbf{u}_n
\end{array}
\right]
= \left[
\begin{array}{c|ccc}
\lambda_1 & w_2 & \ldots & w_n \\
\hline
0 & \\
\vdots & & \huge \mathbf{B} & \\
0 &
\end{array}
\right], $$

где $\mathbf{w} \in \mathbb{C}^{n-1}$ – вектор из комплексных чисел, и для матрицы $\mathbf{B} \in \mathbb{C}^{\left( n - 1 \right) \times \left( n - 1 \right)}$ вследствие ее размерности и в соответствии и индукционным переходом существует разложение Шура. Таким образом, для матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$ также существует разложение Шура.
```

Важно отметить, что разложение Шура существует для любой квадратной матрицы, заданной в пространстве комплексных чисел. Кроме того, из представленного выше доказательства следует, что на главной диагонали верхнетреугольной матрицы располагаются собственные значения исходной матрицы. На практике для вычисления разложения Шура можно использовать функцию [`scipy.linalg.schur`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.schur.html) из библотеки [scipy](https://scipy.org/). В данном учебном пособии разложение Шура будет использоваться для доказательства теоремы о сходимости степеней матрицы.

<a id='theorem-eigen-powerlim'></a>
```{admonition} Теорема
:class: danger
Пусть $\mathbf{A} \in \mathbb{С}^{m \times m}$ представляет собой квадратную матрицу. Тогда необходимым и достаточным условием равенства $\lim_{n \rightarrow \infty} \mathbf{A}^{n} = 0$ является $\rho \left( \mathbf{A} \right) < 1$.
```

```{admonition} Доказательство
:class: proof

Начнем с доказательства *достаточности* условия, то есть если известно, что $\rho \left( \mathbf{A} \right) < 1$, то $\lim_{n \rightarrow \infty} \mathbf{A}^{n} = 0$.

Согласно доказанному ранее [разложению Шура](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-schur), любая квадратная матрица может быть разложена на следующее произведение:

$$ \mathbf{A} = \mathbf{U} \mathbf{T} \mathbf{U}^\dagger, $$

где $\mathbf{U} \in \mathbb{C}^{m \times m}$ является унитарной матрицей; $\mathbf{T} \in \mathbb{C}^{m \times m}$ – верхнетреугольной матрицей, на главной диагонали которой находятся собственные значения исходной матрицы $\mathbf{A}$.

Возведем левую и правую части данного произведения в степень $n$:

$$ \mathbf{A}^n = \left( \mathbf{U} \mathbf{T} \mathbf{U}^\dagger \right)^n = \underbrace{ \left( \mathbf{U} \mathbf{T} \mathbf{U}^\dagger \right) \left( \mathbf{U} \mathbf{T} \mathbf{U}^\dagger \right) \ldots \left( \mathbf{U} \mathbf{T} \mathbf{U}^\dagger \right) }_{\text{n раз}} = \mathbf{U} \mathbf{T} \underbrace{\mathbf{U}^\dagger \mathbf{U}}_{\mathbf{I}} \mathbf{T} \mathbf{U}^\dagger \ldots \mathbf{U} \mathbf{T} \mathbf{U}^\dagger = \mathbf{U} \mathbf{T}^n \mathbf{U}^\dagger. $$

Таким образом, справедливо следующее равенство:

$$ \mathbf{A}^n = \mathbf{U} \mathbf{T}^n \mathbf{U}^\dagger. $$

В свою очередь, верхнетреугольная матрица $\mathbf{T}$ может быть представлена в виде суммы диагональной матрицы $\mathbf{D}$, на главной диагонали которой расположены собственные значения матрицы $\mathbf{A}$, и строго верхнетреугольной матрицы $\mathbf{N}$, являющейся [нильпотентной](LAB-2-Matrices.md#matrix-nilpotent) и представляющей собой внедиагональные элементы верхнетреугольной матрицы $\mathbf{T}$:

$$ \mathbf{T} = \mathbf{D} + \mathbf{N}. $$

Тогда, согласно [биному Ньютона](https://en.wikipedia.org/wiki/Binomial_theorem), матрицу $\mathbf{T}^n$ можно представить в виде:

$$ \mathbf{T}^n = \left( \mathbf{D} + \mathbf{N} \right)^n = \sum_{k=0}^n \frac{n!}{k! \left( n - k \right)!} \mathbf{D}^{n - k} \mathbf{N}^k. $$

Учитывая, что матрица $\mathbf{N}$ является строго верхнетреугольной, то для нее справедливо:

$$ \mathbf{N}^k = 0, \; \forall \; k \geq m. $$

Тогда выражение для $\mathbf{T}^n$ преобразуется к следующему:

$$ \mathbf{T}^n = \sum_{k=0}^n \frac{n!}{k! \left( n - k \right)!} \mathbf{D}^{n - k} \mathbf{N}^k = \sum_{k=0}^{m-1} \frac{n!}{k! \left( n - k \right)!} \mathbf{D}^{n - k} \mathbf{N}^k. $$

То есть в данном выражении были обращены в ноль все слагаемые со степенью $k$ матрицы $\mathbf{N}$, большей или равной ее размерности $m$.

Теперь перейдем к рассмотрению предела:

$$ \begin{align}
\lim_{n \rightarrow \infty} \mathbf{A}^n
&= \lim_{n \rightarrow \infty} \mathbf{U} \mathbf{T}^n \mathbf{U}^\dagger \\
&= \mathbf{U} \left( \lim_{n \rightarrow \infty} \mathbf{T}^n \right) \mathbf{U}^\dagger \\
&= \mathbf{U} \left( \lim_{n \rightarrow \infty} \sum_{k=0}^{m-1} \frac{n!}{k! \left( n - k \right)!} \mathbf{D}^{n - k} \mathbf{N}^k \right) \mathbf{U}^\dagger \\
&= \mathbf{U} \left( \sum_{k=0}^{m-1} \left( \lim_{n \rightarrow \infty} \frac{n!}{k! \left( n - k \right)!} \mathbf{D}^{n - k} \right) \mathbf{N}^k \right) \mathbf{U}^\dagger.
\end{align} $$

Рассмотрим подробнее следующей предел:

$$ \lim_{n \rightarrow \infty} \frac{n!}{k! \left( n - k \right)!} \mathbf{D}^{n-k} = \begin{bmatrix} \lim\limits_{n \rightarrow \infty} \frac{n!}{k! \left( n - k \right)!} \lambda_1^{n-k} & 0 & \ldots & 0 \\ 0 & \lim\limits_{n \rightarrow \infty} \frac{n!}{k! \left( n - k \right)!} \lambda_2^{n-k} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \lim\limits_{n \rightarrow \infty} \frac{n!}{k! \left( n - k \right)!} \lambda_m^{n-k} \end{bmatrix}. $$

Значение коэффициента перед матрицей $\mathbf{D}^{n - k}$ с ростом $n$ растет по степенному закону. В свою очередь, поскольку $\rho \left( \mathbf{A} \right) = \max_i \left| \lambda_i \right| < 1$, то все собственные значения матрицы $\mathbf{A}$, расположенные на главной диагонали матрицы $\mathbf{D}$, по модулю меньше единицы. Следовательно, элементы матрицы $\mathbf{D}^{n - k}$ по мере $n \rightarrow \infty$ будут стремиться к нулю по экспоненциальному закону. Тогда предел произведения функций, первая из которых растет по степенному закону, а вторая уменьшается по экспоненциальному, равен нулю. Данный вывод доказывается с использованием правила Лопиталя ([см. третий пример](https://en.wikipedia.org/wiki/L%27H%C3%B4pital%27s_rule#Examples)). Таким образом, условие $\rho \left( \mathbf{A} \right) < 1$ является достаточным для $\lim_{n \rightarrow \infty} \mathbf{A}^{n} = 0$.

Докажем *необходимость* данного условия, то есть если $\lim_{n \rightarrow \infty} \mathbf{A}^{n} = 0$, то $\rho \left( \mathbf{A} \right) < 1$.

Умножим левую и правую части равенства нулю данного предела на собственный вектор матрицы $\mathbf{A}$. Тогда:

$$ \lim_{n \rightarrow \infty} \mathbf{A}^{n} \mathbf{v} = 0. $$

В соответствии с доказанной выше [леммой о собственных значениях и векторах матрицы, возведенной в степень](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-power), преобразуем данное выражение к следующему виду:

$$ \lim_{n \rightarrow \infty} \lambda^{n} \mathbf{v} = 0. $$

Поскольку собственный вектор $\mathbf{v}$ не является нулевым по определению, то истинность данного равенства возможна только в том случае, если:

$$ \lim_{n \rightarrow \infty} \lambda^{n} = 0. $$

Соответственно, показательная функция при устремлении показателя в бесконечность стремится к нулю в том случае, когда модуль основания меньше единицы: $\left| \lambda \right| < 1$. Поскольку данное неравенство характерно для всех собственных значений матрицы $\mathbf{A}$, то из этого следует, что ее спектральный радиус меньше единицы.
```

(eigen-multiplicity)=
## Алгебраическая и геометрическая кратность собственных значений матрицы

Существуют матрицы, для которых их собственные значения повторяются. В этом случае [характеристическое уравнение](LAB-5-Eigenvalues-Eigenvectors.md#eigen-characteristic-polynomial) может быть записано следующим образом:

$$ p_n \left( \lambda \right) = \left( -1 \right)^n \left( \lambda - \lambda_1 \right)^{m_1} \cdot \left( \lambda - \lambda_2 \right)^{m_2} \cdot \ldots \cdot \left( \lambda - \lambda_n \right)^{m_n} = 0, $$

где $m_1, \, m_2, \, \ldots, \, m_n$ – *алгебраические* кратности собственных значений матрицы.

*Геометрической кратностью* называют количество [линейно независимых](LAB-3-LinearSystems.md#def-ls-linear-independence) собственных векторов матрицы.

Если квадратная матрица $\mathbf{A} \in \mathbb{C}^{n \times n}$ характеризуется $n$ различными собственными значениями, то все ее собственные векторы линейно независимы и образуют базис (что следует из определения собственных вектора и значения матрицы). Однако обратное утверждение о том, что если квадратная матрица $\mathbf{A} \in \mathbb{C}^{n \times n}$ имеет $n$ линейно независимых векторов, то ее собственные значения различны, *неверно*. Контрпримером является единичная матрица размерностью $2 \times 2$, собственные векторы которой различны $\left[ 1, \, 0 \right]^\top$ и $\left[ 0, \, 1 \right]^\top$, но при этом матрица характеризуется собственным значением, равным единице, с алгебраической кратностью, равной двум.

(eigen-diagonalization)=
## Базис из собственных векторов и диагонализация матриц

Ранее было дано определение [базису](LAB-4-LinearTransformations.md#lintran-basis), а также сформулировано условие его определения, выраженное в линейной независимости базисных векторов. Кроме того, было показано, что линейное преобразование $\mathbf{T} \left( \cdot \right)$ в базисе $\mathrm{B}$ выражается матрицей $\mathbf{D}$, определяемой следующим образом:

$$ \mathbf{D} = \mathbf{B}^{-1} \mathbf{A} \mathbf{B}, $$

где: $\mathbf{B}$ – матрица перехода между стандартным (каноническим) базисом и базисом $\mathrm{B}$, $\mathbf{A}$ – матрица линейного преобразования вектора $\mathbf{x}$ в стандартном базисе.

<a id='theorem-eigen-diagonalization'></a>
```{admonition} Теорема
:class: danger
Матрица линейного преобразования в базисе из ее собственных векторов является диагональной, главная диагональ которой представлена соответствующими собственными значениями.
```

Применительно к введенным выше обозначениям данную теорему можно сформулировать следующим образом: если матрица перехода к базису $\mathrm{B}$ составлена из линейно независимых собственных векторов исходной матрицы $\mathbf{A}$, то в этом базисе матрица будет диагональной $\mathbf{D}$. Важно отметить, что условием диагонализируемости квадратной матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$ является наличие $n$ линейно независимых собственных векторов, образующих базис. В противном случае не существует [обратной матрицы](LAB-2-Matrices.md#matrix-inv) для матрицы, столбцы которой линейно зависимы, поскольку [определитель](LAB-2-Matrices.md#matrix-det) такой матрицы равен нулю.

````{admonition} Доказательство
:class: proof
Пусть матрица $\mathbf{B} \in \mathbb{C}^{n \times n}$ составлена из координат линейно независимых собственных векторов $\mathbf{v}_1, \, \mathbf{v}_2, \, \ldots, \, \mathbf{v}_n$ матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$:

$$ \mathbf{B} = \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{v_1} & \mathbf{v_2} & \ldots & \mathbf{v}_n \\ \vert & \vert & & \vert \end{bmatrix}. $$

Умножим левую и правую части равенства $\mathbf{D} = \mathbf{B}^{-1} \mathbf{A} \mathbf{B}$ на единичный вектор $\mathbf{e}_i$, сонаправленный с собственным вектором $\mathbf{v}_i$:

$$ \mathbf{D} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{B} \mathbf{e}_i. $$

Результатом произведения матрицы $\mathbf{D}$ на единичный вектор $\mathbf{e}_i$ будет $i$-ый столбец матрицы.

Аналогично произведение матрицы $\mathbf{B}$ на единичный вектор $\mathbf{e}_i$ даст $i$-ый собственный вектор:

$$ \mathbf{B} \mathbf{e}_i = \mathbf{v}_i. $$

Тогда:

$$ \mathbf{D} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{B} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{v}_i. $$

По [определению собственного вектора матрицы](LAB-5-Eigenvalues-Eigenvectors.md#def-eigen) данное выражение преобразуется в:

$$ \mathbf{D} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{B} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{v}_i= \mathbf{B}^{-1} \lambda_i \mathbf{v}_i. $$

С учетом того, что $\mathbf{e}_i = \mathbf{B}^{-1} \mathbf{v}_i$, получим:

$$ \mathbf{D} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{B} \mathbf{e}_i = \mathbf{B}^{-1} \mathbf{A} \mathbf{v}_i = \mathbf{B}^{-1} \lambda_i \mathbf{v}_i = \lambda_i \mathbf{B}^{-1} \mathbf{v}_i = \lambda_i \mathbf{e}_i. $$

Таким образом, вертикальный столбец матрицы $\mathbf{D}$ представляет собой произведение единичного вектора на скаляр, следовательно, если в качестве базиса выбрать собственные векторы матрицы $\mathbf{A}$, то матрица $\mathbf{D}$ будет являться диагонализированной матрицей $\mathbf{A}$ относительно базиса $\mathrm{B}$. На главной диагонали матрицы $\mathbf{D}$ будут находиться собственные значения матрицы $\mathbf{A}$.
````

Рассмотрим данное свойство на следующем примере.

```{admonition} Пример
:class: exercise
Пусть дана матрица $\mathbf{A}$:

$$ \mathbf{A} = \begin{bmatrix} -1 & 3 & -1 \\ -3 & 5 & -1 \\ -3 & 3 & 1 \end{bmatrix}. $$

Необходимо определить ее диагонализированный вид и соответствующий базис.
```

````{dropdown} Решение
Зададим матрицу в виде двумерного массива и найдем ее собственные значения и векторы:

```python
A = np.array([
    [-1., 3., -1.],
    [-3., 5., -1.],
    [-3., 3., 1.],
])
lmbdi, B = np.linalg.eig(A)
print(B)
```

```{glue:} glued_out10
```

Вычислим определитель матрицы перехода для проверки линейной независимости собственных векторов матрицы:

``` python
print(np.linalg.det(B))
```

```{glue:} glued_out11
```

Определитель матрицы, составленной из координат собственных векторов, не равен нулю, следовательно, собственные векторы матрицы $\mathbf{A}$ образуют базис, в котором рассматриваемая матрица является диагональной:

``` python
B_inv = np.linalg.inv(B)
print(B_inv @ A @ B)
```

```{glue:} glued_out12
```

Главная диагональ представлена из собственных значений исходной матрицы $\mathbf{A}$:

``` python
print(lmbdi)
```

```{glue:} glued_out13
```
````

```{code-cell} python
:tags: [remove-cell]

A = np.array([
    [-1., 3., -1.],
    [-3., 5., -1.],
    [-3., 3., 1.],
])
lmbdi, B = np.linalg.eig(A)
B_inv = np.linalg.inv(B)

glue('glued_out10', MultilineText(B))

glue('glued_out11', MultilineText(np.linalg.det(B)))

glue('glued_out12', MultilineText(B_inv @ A @ B))

glue('glued_out13', MultilineText(lmbdi))
```

Геометрической интерпретацией (следствием) для доказанной теоремы о диагонализации матрицы в базисе, составленном из ее собственных векторов, является определение собственных векторов как инвариантных направлений линейного преобразования, вдоль которых действие этого преобразования сводится к [умножению вектора на скаляр](LAB-1-Vectors.md#vector-mult).

Из определения собственных вектора и значения матрицы:

$$ \mathbf{A} \mathbf{u} = \lambda \mathbf{u}, $$

следует, что $\mathbf{A} \in \mathbb{C}^{n \times n}$ представляет собой матрицу линейного преобразования, действие которого на вектор $\mathbf{u} \in \mathbb{C}^{n}$ сводится к его умножению на скаляр, сохраняя условие [коллинеарности](LAB-1-Vectors.md#vector-collinear). Если же применить данную матрицу линейного преобразования к произвольному вектору, то ее собственные векторы будут указывать направления, вдоль каждого из которых действие преобразования сводится к коллинеарному изменению вектора.

<a id='lemma-eigen-stretching'></a>
```{admonition} Лемма
:class: caution
Линейно независимые собственные векторы матрицы линейного преобразования определяют инвариантные направления, вдоль которых действие преобразования сводится к умножению вектора на соответствующее собственное значение. Для любого вектора, лежащего на таком направлении, образ будет ему коллинеарен, а коэффициент пропорциональности будет равен соответствующему собственному значению матрицы.
```

```{admonition} Доказательство
:class: proof
Пусть линейное преобразование произвольного вектора $v \in \mathbb{C}^{n}$ выражается с использованием матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$, собственные значения и соответствующие им линейно независимые собственные векторы которой обозначаются $\lambda_i \in \mathbb{C}, \, i = 1 \, \ldots \, n,$ и $\mathbf{u}_i \in \mathbb{C}^{n}, \, i = 1 \, \ldots \, n,$ соответственно. Поскольку линейно независимые собственные векторы образуют базис $\mathrm{B} = \begin{Bmatrix} \mathbf{u}_1, \, \mathbf{u}_2, \, \ldots, \, \mathbf{u}_n \end{Bmatrix}$, то произвольный вектор можно представить в следующем виде:

$$ \mathbf{v} = \sum_{k=1}^n w_k \mathbf{u}_k = \mathbf{U} \mathbf{w}, $$

где $\mathbf{w} = \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B} = \begin{bmatrix} w_1, \, w_2, \, \ldots, \, w_n \end{bmatrix}^\top$ – координаты вектора $\mathbf{v}$ относительно базиса, составленного из собственных векторов матрицы линейного преобразования:

$$ \mathrm{B} = \begin{Bmatrix} \mathbf{u}_1, \; \mathbf{u}_2, \; \ldots, \; \mathbf{u}_n \end{Bmatrix}. $$

В свою очередь, матрица $\mathbf{U} \in \mathbb{C}^{n \times n}$ составлена из собственных векторов матрицы линейного преобразования:

$$ \mathbf{U} = \begin{bmatrix} \vert & \vert &  & \vert \\ \mathbf{u}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert &  & \vert \end{bmatrix}. $$

С учетом этого преобразуем произведение матрицы $\mathbf{A}$ и вектора $\mathbf{v}$:

$$ \begin{align}
\mathbf{A} \mathbf{v}
&= \mathbf{A} \mathbf{U} \mathbf{w} \\
&= \mathbf{A} \begin{bmatrix} \vert & \vert &  & \vert \\ \mathbf{u}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert &  & \vert \end{bmatrix} \mathbf{w} \\
&= \begin{bmatrix} \vert & \vert &  & \vert \\ \mathbf{A} \mathbf{u}_1 & \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{A} \mathbf{u}_n \\ \vert & \vert &  & \vert \end{bmatrix} \mathbf{w} \\
&= \begin{bmatrix} \vert & \vert &  & \vert \\ \lambda_1 \mathbf{u}_1 & \lambda_2 \mathbf{u}_2 & \ldots & \lambda_n \mathbf{u}_n \\ \vert & \vert &  & \vert \end{bmatrix} \mathbf{w} \\
&= \mathbf{U} \mathbf{D} \mathbf{w},
\end{align} $$

где на главной диагонали диагональной матрицы $\mathbf{D} \in \mathbb{C}^{n \times n}$ находятся собственные значения матрицы $\mathbf{A}$:

$$ \mathbf{D} = \begin{bmatrix} \lambda_1 & 0 & \ldots & 0 \\ 0 & \lambda_2 & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \lambda_n \end{bmatrix}. $$

Таким образом, произведение матрицы $\mathbf{A}$ и вектора $\mathbf{v}$ может быть представлено в виде линейной комбинации ее собственных векторов, собственных значений и координат вектора относительно базиса, составленного из собственных векторов матрицы линейного преобразования:

$$ \mathbf{A} \mathbf{v} = \lambda_1 w_1 \mathbf{u}_1 + \lambda_2 w_2 \mathbf{u}_2 + \ldots + \lambda_n w_n \mathbf{u}_n. $$

Данная теорема является геометрической интерпретацией (следствием) для сформулированной теоремы о диагонализации матрицы в базисе, составленном из ее собственных векторов, поскольку линейное преобразование $\mathbf{A} \mathbf{v}$ в базисе $\mathrm{B}$ записывается следующим образом:

$$ \begin{bmatrix} \mathbf{A} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{U}^{-1} \mathbf{A} \mathbf{v} = \mathbf{U}^{-1} \mathbf{U} \mathbf{D} \mathbf{w} = \mathbf{D} \mathbf{w} = \lambda_i w_i, \; i = 1 \, \ldots \, n. $$
```

Если рассматривать пространство действительных чисел, то собственные векторы матрицы линейного преобразования произвольного вектора указывают направления, в которых происходит растяжение, сжатие и отражение этого произвольного вектора в зависимости от значений и знаков собственных значений матрицы линейного преобразования и координат вектора в базисе, составленном из собственных векторов матрицы линейного преобразования.

Проиллюстрируем данное следствие следующим примером.

```{admonition} Пример
:class: exercise
С использованием матрицы $\mathbf{A}$ из предыдущего примера необходимо выполнить линейное преобразование вектора $\mathbf{v} = \begin{bmatrix} 3, \, -1, \, 2 \end{bmatrix}^\top$ в базисе, составленном из ее собственных векторов.
```

````{dropdown} Решение
Зададим координаты вектора $\mathbf{v}$ в стандартном базисе в виде одномерного массива:

```python
v = np.array([3., -1., 2.])
```

Получим его координаты в базисе, составленном из собственных векторов матрицы $\mathbf{A}$:

```python
vB = B_inv.dot(v)
```

Выполним линейное преобразование в данном базисе:

```python
rB = lmbdi * vB
print(rB)
```

```{glue:} glued_out14
```

Также покажем, что координаты результирующего вектора $\mathbf{r}$ в стандартном базисе соответствуют произведению матрицы $\mathbf{A}$ и вектора $\mathbf{v}$:

```python
r = B.dot(rB)
print(np.allclose(r, A.dot(v)))
```

```{glue:} glued_out15
```
````

```{code-cell} python
:tags: [remove-cell]

v = np.array([3., -1., 2.])
vB = B_inv.dot(v)
rB = lmbdi * vB

glue('glued_out14', MultilineText(rB))

r = B.dot(rB)

glue('glued_out15', MultilineText(np.allclose(r, A.dot(v))))
```

(eigen-hermitian)=
## Спектральные свойства эрмитовых матриц

Данный подраздел будет посвящен изложению ряда свойств собственных векторов и значений [эрмитовых матриц](LAB-2-Matrices.md#def-matrix-hermitian).

<a id='lemma-eigen-hermitian-real'></a>
```{admonition} Лемма
:class: caution
Собственные значения симметричной матрицы, составленной из действительных чисел $\mathbf{A} \in \mathbb{R}^{n \times n}$, также являются действительными.
```

```{admonition} Доказательство
:class: proof
Пусть $\mathbf{A} \in \mathbb{R}^{n \times n}$ представляет собой симметричную матрицу, составленную из действительных чисел. Обозначим $\lambda$ ее собственное значение и $\mathbf{x} \in \mathbb{R}^{n}$ соответствующее ему ненулевой собственный вектор.

Из определения собственных вектора и значения матрицы:

$$ \mathbf{A} \mathbf{x} = \lambda \mathbf{x}. $$

Применим к левой и правой части данного выражения операцию [комплексно-сопряженного транспонирования](https://en.wikipedia.org/wiki/Conjugate_transpose), состоящую из двух операций: [транспонирования](LAB-2-Matrices.md#matrix-transp) и замены значений элементов матрицы или вектора на их [сопряженные](https://en.wikipedia.org/wiki/Complex_conjugate), имеющих обратный знак перед мнимой частью:

$$ \left( \mathbf{A} \mathbf{x} \right)^\dagger = \left( \lambda \mathbf{x} \right)^\dagger. $$

Операции комплексно-сопряженного транспонирования будет характерно ранее доказанное [свойство транспонирования матриц](LAB-2-Matrices.md#theorem-matrix-transp-dot):

$$ \mathbf{x}^\dagger \mathbf{A}^\dagger = \lambda^* \mathbf{x}^\dagger, $$

где символом $\lambda^*$ обозначено сопряженное число собственному значению $\lambda$. Поскольку матрица $\mathbf{A}$ представляет собой симметричную матрицу, составленную из действительных чисел, то:

$$ \mathbf{A} = \mathbf{A}^\top = \mathbf{A}^\dagger. $$

Тогда:

$$ \mathbf{x}^\dagger \mathbf{A} = \lambda^* \mathbf{x}^\dagger. $$

Домножив данное равенство на $\mathbf{x}$ справа и выражение для определения собственных вектора и числа матрицы $\mathbf{A}$ на $\mathbf{x}^\dagger$ слева, запишем систему уравнений:

$$ \begin{cases}
\mathbf{x}^\dagger \mathbf{A} \mathbf{x} = \lambda \mathbf{x}^\dagger \mathbf{x}, \\
\mathbf{x}^\dagger \mathbf{A} \mathbf{x} = \lambda^* \mathbf{x}^\dagger \mathbf{x}.
\end{cases} $$

Поскольку левые части уравнений равны, то равны и правые:

$$ \lambda \mathbf{x}^\dagger \mathbf{x} = \lambda^* \mathbf{x}^\dagger \mathbf{x}. $$

Перенесем все в левую часть уравнения и вынесем общее за скобки:

$$ \mathbf{x}^\dagger \mathbf{x} \left( \lambda - \lambda^* \right) = 0. $$

Пусть

$$ \mathbf{x} = \begin{bmatrix} a_0 + b_0 i, \, a_1 + b_1 i, \, \ldots, \, a_n + b_n i \end{bmatrix}^\top, $$

тогда

$$ \mathbf{x}^\dagger = \begin{bmatrix} a_0 - b_0 i, \, a_1 - b_1 i, \, \ldots, \, a_n - b_n i \end{bmatrix}. $$

Распишем подробнее [скалярное произведение](LAB-1-Vectors.md#vector-dot) данных векторов:

$$ \mathbf{x}^\dagger \mathbf{x} = \sum_{j=1}^n \left( a_j + b_j i \right) \left( a_j - b_j i \right) = \sum_{j=1}^n \left( a_j^2 + b_j^2 \right) > 0 \; \; \forall \; \; a_j \neq 0, \; b_j \neq 0, \; j = 1 \, \ldots \, n. $$

Элементы вектора $\mathbf{x}$ отличны от нуля по условию. Таким образом, произведение $\mathbf{x}^\dagger \mathbf{x}$ будет больше нуля для ненулевых векторов $\mathbf{x}$, следовательно, собственное значение матрицы будет равно его сопряженному $\lambda = \lambda^*$, что возможно в том случае, если мнимая его часть равна нулю, а само собственное значение относится ко множеству действительных чисел.
```

Таким образом, симметричные матрицы, состоящие из действительных чисел, характеризуются действительными собственными значениями. Однако обратное утверждение не обязательно верно, поскольку квадратная несимметричная матрица, составленная из действительных чисел, также может иметь действительные собственные значения, как, например, матрицы, ранее рассмотренные в примерах данного раздела.

Ранее было введено понятие [*ортогонального* линейного преобразования](LAB-4-LinearTransformations.md#def-lintran-orthogonal), при котором сохраняется [длина вектора](LAB-1-Vectors.md#vector-length). Покажем, что собственные векторы симметричной матрицы, соответствующие различным собственным значениям, ортогональны, а сама матрица *ортогонально диагонализируема*.

<a id='lemma-eigen-hermitian-orthogonal'></a>
```{admonition} Лемма
:class: caution
Собственные векторы симметричной матрицы, соответствующие различным собственным значениям, [ортогональны](LAB-1-Vectors.md#vector-dot).
```

```{admonition} Доказательство
:class: proof
Пусть $\mathbf{A} \in \mathbb{R}^{n \times n}$ представляет собой симметричную матрицу. В соответствии с доказанным выше такая матрица будет характеризоваться действительными собственными значениями и соответствующими им собственными векторами. Рассмотрим два различных собственных значения данной матрицы $\lambda_1 \neq \lambda_2$ и соответствующие им два различных собственных вектора $\mathbf{u}_1 \neq \mathbf{u}_2 \neq 0$. Запишем определение собственных вектора и значения для первой пары:

$$ \mathbf{A} \mathbf{u}_1 = \lambda_1 \mathbf{u}_1. $$

Домножим левую и правую часть данного выражения на $\mathbf{u}_2$ справа:

$$ \left( \mathbf{A} \mathbf{u}_1 \right)^\top \mathbf{u}_2 = \left( \lambda_1 \mathbf{u}_1 \right)^\top \mathbf{u}_2. $$

Применим доказанное ранее [свойство транспонирования произведения](LAB-2-Matrices.md#theorem-matrix-transp-dot):

$$ \mathbf{u}_1^\top \mathbf{A}^\top \mathbf{u}_2 = \lambda_1 \mathbf{u}_1^\top \mathbf{u}_2. $$

Поскольку матрица $\mathbf{A}$ является симметричной, то для нее можно записать равенство: $\mathbf{A}^\top = \mathbf{A}$. Тогда с учетом определения собственных вектора и значения для второй пары, преобразуем данное выражение к следующему виду:

$$ \lambda_2 \mathbf{u}_1^\top \mathbf{u}_2 = \lambda_1 \mathbf{u}_1^\top \mathbf{u}_2. $$

Перенесем в одну сторону и вынесем за скобки общую часть:

$$ \mathbf{u}_1^\top \mathbf{u}_2 \left( \lambda_2 - \lambda_1 \right) = 0. $$

Поскольку $\lambda_1 \neq \lambda_2$, а также $\mathbf{u}_1 \neq 0$ и $\mathbf{u}_2 \neq 0$, то данное равенство возможно, если

$$ \mathbf{u}_1^\top \mathbf{u}_2 = 0. $$

Поскольку скалярное произведение двух ненулевых векторов равно нулю, то они являются *ортогональными* по определению. Если при этом каждый собственный вектор разделить на его длину (то есть сделать их единичными), то их можно считать *ортонормальными*.
```

Леммы о [принадлежности собственных значений симметричной матрицы пространству действительных чисел](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-symmetric-real) и об [ортогональности собственных векторов симметричной матрицы, соответствующих ее различным собственным значениям](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-symmetric-orthogonal) можно использовать для доказательства теоремы об ортогональной диагонализируемости симметричных матриц ([спектральной теоремы](https://en.wikipedia.org/wiki/Spectral_theorem) для симметричных матриц).

<a id='theorem-eigen-hermitian'></a>
```{admonition} Теорема
:class: danger
Пусть $\mathbf{A} \in \mathbb{R}^{n \times n}$ представляет собой симметричную матрицу. Тогда существуют ортогональная матрица $\mathbf{P} \in \mathbb{R}^{n \times n}$ (ее столбцы являются [ортонормальными](LAB-1-Vectors.md#vector-dot) векторами, образующими ортонормированный базис) и диагональная матрица $\mathbf{D} \in \mathbb{R}^{n \times n}$ такие, что:

$$ \mathbf{D} = \mathbf{P}^\top \mathbf{A} \mathbf{P}. $$
```

Следует отметить, что, доказав лемму об ортогональности собственных векторов симметричной матрицы, соответствующих ее различным собственным значениям, теорема об ортогональной диагонализируемости симметричных матриц (следствием из которой является существование полного ортонормированного базиса на основе собственных векторов симметричной матрицы) еще не доказана, поскольку непонятно, можно ли построить базис в пространстве известной размерности при условии, когда кратность собственных значений матрицы больше единицы. Таким образом, необходимо доказать существование ортонормированного базиса для произвольной симметричной матрицы, в котором она является диагональной даже при условии повторяемости собственных значений. Доказательство данной теоремы будет очень похоже на рассмотренное ранее [доказательство существования разложения Шура](LAB-5-Eigenvalues-Eigenvectors.md#lemma-eigen-schur), а саму теорему можно рассматривать как частный случай этого разложения.

```{admonition} Доказательство
:class: proof
Для доказательства данной теоремы воспользуемся [методом математической индукции](https://en.wikipedia.org/wiki/Mathematical_induction). Последовательность натуральных чисел будет задаваться размерностью $n$ симметричной матрицы $\mathbf{A}$.

**База индукции**

Матрица размерностью $\mathbf{A} \in \mathbb{R}^{1 \times 1} = \begin{bmatrix} a \end{bmatrix}$ уже является диагональной с собственным значением $a$ и матрицей $\mathbf{P} = \begin{bmatrix} 1 \end{bmatrix}$.

**Индукционный переход**

Пусть утверждение об ортогональной диагонализируемости истинно для всех симметричных матриц размерностью $\left( n - 1 \right) \times \left( n - 1 \right)$, составленных из действительных чисел. Покажем, что оно также справедливо и для симметричной матрицы размерности $\mathbf{A} \in \mathbb{R}^{n \times n}$. При этом у такой матрицы существует как минимум один единичный собственный вектор $\mathbf{v}_1 \in \mathbb{R}^{n}$ и соответствующее ему собственное значение $\lambda_1$:

$$ \mathbf{A} \mathbf{v}_1 = \lambda_1 \mathbf{v}_1. $$

Используя [преобразование Хаусхолдера](LAB-4-LinearTransformations.md#lintran-hauseholder) или [процесс Грама-Шмидта](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process), на основе собственного вектора $\mathbf{v}_1$ можно создать ортонормированный базис $\mathrm{B} = \begin{Bmatrix} \mathbf{v}_1, \, \mathbf{u_2}, \, \ldots, \, \mathbf{u}_n \end{Bmatrix}$. Пусть матрица $\mathbf{P} \in \mathbb{R}^{n \times n}$ составлена из векторов базиса $\mathrm{B}$:

$$ \mathbf{P} = \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{v}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix}. $$

Рассмотрим произведение

$$ \begin{align}
\mathbf{P}^\top \mathbf{A} \mathbf{P}
&= \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{v}_1^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_2^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_n^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \mathbf{A} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{v}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix} \\
&= \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{v}_1^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_2^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_n^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{A} \mathbf{v}_1 & \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{A} \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{v}_1^\top \mathbf{A} \mathbf{v}_1 & \mathbf{v}_1^\top \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{v}_1^\top \mathbf{A} \mathbf{u}_n \\
\mathbf{u}_2^\top \mathbf{A} \mathbf{v}_1 & \mathbf{u}_2^\top \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_2^\top \mathbf{A} \mathbf{u}_n \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{u}_n^\top \mathbf{A} \mathbf{v}_1 & \mathbf{u}_n^\top \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_n^\top \mathbf{A} \mathbf{u}_n
\end{bmatrix}
\end{align} $$

Распишем подробнее элементы получившейся матрицы.

$$ \begin{align}
\mathbf{v}_1^\top \mathbf{A} \mathbf{v}_1 &= \mathbf{v}_1^\top \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{v}_1^\top \mathbf{v}_1 = \lambda_1, \\
\mathbf{v}_1^\top \mathbf{A} \mathbf{u}_2 &= \left( \mathbf{A}^\top \mathbf{v}_1 \right)^\top \mathbf{u}_2 = \lambda_1 \mathbf{v}_1^\top \mathbf{u}_2 = 0, \\
\mathbf{v}_1^\top \mathbf{A} \mathbf{u}_n &= \left( \mathbf{A}^\top \mathbf{v}_1 \right)^\top \mathbf{u}_n = \lambda_1 \mathbf{v}_1^\top \mathbf{u}_n = 0, \\
\mathbf{u}_2^\top \mathbf{A} \mathbf{v}_1 &= \mathbf{u}_2^\top \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{u}_2^\top \mathbf{v}_1 = 0, \\
\mathbf{u}_n^\top \mathbf{A} \mathbf{v}_1 &= \mathbf{u}_n^\top \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{u}_n^\top \mathbf{v}_1 = 0, \\
\mathbf{u}_2^\top \mathbf{A} \mathbf{u}_n &= \left( \mathbf{u}_2^\top \mathbf{A} \mathbf{u}_n \right)^\top = \mathbf{u}_n^\top \mathbf{A}^\top \mathbf{u}_2 = \mathbf{u}_n^\top \mathbf{A} \mathbf{u}_2.
\end{align} $$

При выводе данных выражений учитывались:

* равенство единице длины единичного собственного вектора $\lVert \mathbf{v}_1 \rVert_2 = 1$ или $\mathbf{v}_1^\top \mathbf{v}_1 = 1$;
* равенство нулю скалярных произведений пар различных базисных векторов базиса $\mathrm{B} = \begin{Bmatrix} \mathbf{v}_1, \, \mathbf{u_2}, \, \ldots, \, \mathbf{u}_n \end{Bmatrix}$ вследствие его ортонормированности;
* доказанный ранее [закон обратного порядка для транспонирования](LAB-2-Matrices.md#theorem-matrix-transp-dot).

Таким образом, в результате произведения $\mathbf{P}^\top \mathbf{A} \mathbf{P}$ получается следующая матрица:

$$ \mathbf{P}^\top \mathbf{A} \mathbf{P}
= \left[
\begin{array}{c|ccc}
\lambda_1 & 0 & \ldots & 0 \\
\hline
0 & \mathbf{u}_2^\top \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_2^\top \mathbf{A} \mathbf{u}_n \\
\vdots & \vdots & \ddots & \vdots \\
0 & \mathbf{u}_n^\top \mathbf{A} \mathbf{u}_2 & \ldots & \mathbf{u}_n^\top \mathbf{A} \mathbf{u}_n
\end{array}
\right]
= \left[
\begin{array}{c|ccc}
\lambda_1 & 0 & \ldots & 0 \\
\hline
0 & \\
\vdots & & \huge \mathbf{B} & \\
0 &
\end{array}
\right] $$

При этом матрица $\mathbf{B} \in \mathbb{R}^{\left( n - 1 \right) \times \left( n - 1 \right)}$ является симметричной $\left( \mathbf{u}_2^\top \mathbf{A} \mathbf{u}_n = \mathbf{u}_n^\top \mathbf{A} \mathbf{u}_2 \right)$, следовательно, диагонализируемой. Таким образом, исходная симметричная матрица $\mathbf{A}$ также является диагонализируемой.
```

(eigen-numerical)=
## Численные методы определения собственных значений и векторов матрицы
