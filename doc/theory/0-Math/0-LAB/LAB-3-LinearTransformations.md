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

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': False})

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

# Линейные преобразования

Определив свойства векторов, операции над ними, а также их определение в базисе, мы переходим к изучению того, как векторы могут изменяться, сохраняя свои фундаментальные свойства. Любое линейное преобразование – будь то поворот, растяжение или отражение – можно рассматривать как отображение, которое переводит один вектор в другой, соблюдая свойства линейности. Ключевым связующим звеном здесь выступает матрица: она служит своеобразным инструментом преобразования, полностью описывая, как изменяются координаты векторов. Таким образом, операции над матрицами становятся удобным языком для управления геометрическими трансформациями и переходом между различными системами координат

<!-- Прежде чем переходить к изложению темы о линейных преобразованиях векторов, необходимо дать определение их линейной независимости.

<a id='lintran-linindep'></a>
Теперь остановимся на достаточных условиях.

Пусть столбцы матрицы $\mathbf{V} \in \mathbb{C}^{n \times n}$ составлены из элементов векторов рассматриваемой совокупности:

$$ \mathbf{V} = \begin{bmatrix} \vert & \vert &  & \vert \\ \mathbf{v}_1 & \mathbf{v}_2 & \ldots & \mathbf{v}_n \\ \vert & \vert &  & \vert \end{bmatrix}. $$

Тогда равенство $\sum_{i=1}^n \alpha_i \mathbf{v}_i = 0$ можно представить в виде системы линейных уравнений:

$$ \mathbf{V} \boldsymbol{\alpha} = 0, $$

где вектор $\boldsymbol{\alpha} \in \mathbb{C}^n$ составлен из коэффициентов: $\boldsymbol{\alpha} = \begin{bmatrix} \alpha_1, \, \alpha_2, \, \ldots, \, \alpha_n \end{bmatrix}^\top$.

Тогда условие линейной независимости векторов будет эквивалентно тривиальности решения ($\boldsymbol{\alpha} = 0$) системы линейных уравнений $\mathbf{V} \boldsymbol{\alpha} = 0$. То есть если данная система линейных уравнений имеет бесконечное множество решений, то векторы-столбцы матрицы коэффициентов являются линейно зависимыми. На практике это условие применяется следующим образом:

* из элементов совокупности векторов составляют матрицу коэффициентов;
* приводят ее к треугольному виду [методом Гаусса](LAB-3-LinearSystems.md#ls-gauss);
* если появляется строка из нулей (уравнение вида $0 \cdot x = 0$), то делается вывод о линейной зависимости рассматриваемых векторов.

Наличие множества решений системы линейных уравнений $\mathbf{V} \boldsymbol{\alpha} = 0$ эквивалентно равенству нулю определителя матрицы $\mathbf{V}$. Действительно, согласно [правилу Крамера](https://en.wikipedia.org/wiki/Cramer%27s_rule), единственность и тривиальность решения данной системы линейных уравнений возможны при условии существования обратной матрицы матрице $\mathbf{V}$. Однако если такая матрица *вырожденная*, то обратная матрица к матрице $\mathbf{V}$ не существует, что обуславливает отсутствие единственного решения для системы линейных уравнений. На практике данное условие применяется следующим образом:

* из элементов совокупности векторов составляют матрицу коэффициентов;
* вычисляют определитель этой матрицы;
* если он равен нулю, то делают вывод о линейной зависимости совокупности векторов.

Для небольших по размерам матриц ($n \leq 3$) рекомендуется использовать аналитические формулы для раскрытия определителя и проверки на линейную зависимость векторов. Для больших матриц рекомендуют использовать подход, основанный на приведении к треугольному виду. -->

<!-- Пусть вектор $\mathbf{v} \in \mathbb{R}^2$ имеет координаты $\begin{bmatrix} 4 \\ 2 \end{bmatrix}$. Тогда скалярные произведения вектора $\mathbf{v}$ и базисных векторов стандартного (канонического) базиса $\mathbb{C} = \begin{Bmatrix} \mathbf{i}, \, \mathbf{j} \end{Bmatrix}$:

```{code-cell} python
v = np.array([4., 2.])
i = np.array([1., 0.])
j = np.array([0., 1.])
print(v.dot(i), v.dot(j))
```

Таким образом, скалярное произведение вектора и единичного вектора, сонаправленного с определенной осью, дает значение координаты вектора на данной оси, иными словами, проекцию вектора на ось. В общем случае геометрическое определение скалярного произведения позволяет утверждать, что скалярное произведение двух векторов определяет скалярную проекцию одного вектора на другой. -->

(lintran-matrix)=
## Матрица линейного преобразования

```{admonition} Определение
:class: definition
Если в некотором линейном пространстве  каждому вектору $\mathbf{v}$ по некоторому правилу $\mathcal{T} \left( \cdot \right)$ поставлен в соответствие вектор $\mathbf{u}$ этого же пространства, то говорят, что в данном пространстве задано ***преобразование***:

$$ \mathbf{u} = \mathcal{T} \left(\mathbf{v} \right). $$

Такое преобразование называется ***линейным***, если оно обладает следующими свойствами:

$$ \begin{align}
\mathcal{T} \left( \mathbf{a} + \mathbf{b} \right) &= \mathcal{T} \left( \mathbf{a} \right) + \mathcal{T} \left( \mathbf{b} \right) , \\
\mathcal{T} \left( \lambda \mathbf{a} \right) &= \lambda \mathcal{T} \left( \mathbf{a} \right).
\end{align} $$
```

Термины *линейное отображение* и *линейное преобразование* синонимичны. В данном материале будет использоваться термин *линейное преобразование*.

Рассмотрим следующие примеры.

```{admonition} Пример
:class: exercise
Пусть преобразование определено так, что:

$$ \mathcal{T} \left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1 + x_2 \\ 3x_1 \end{bmatrix}. $$

Необходимо определить, является ли данное преобразование линейным.
```

````{dropdown} Решение
Для того чтобы доказать, что некоторое преобразование является линейным, необходимо доказать, что оно обладает свойствами линейности.

**Свойство аддитивности**

Рассмотрим два вектора $\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}$ и $\mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}.$ Их сумма:

$$ \mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}. $$

Тогда:

$$ \mathcal{T} \left( \mathbf{a} + \mathbf{b} \right) = \mathcal{T} \left( \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix} \right) = \begin{bmatrix} a_1 + b_1 + a_2 + b_2 \\ 3a_1 + 3b_2 \end{bmatrix}.$$

Линейное преобразование вектора $\mathbf{a}$:

$$ \mathcal{T} \left( \mathbf{a} \right)= \mathcal{T} \left( \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \right) = \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix}. $$

Линейное преобразование вектора $\mathbf{b}$:

$$ \mathcal{T} \left( \mathbf{b} \right) = \mathcal{T} \left( \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} \right) = \begin{bmatrix} b_1 + b_2 \\ 3b_1 \end{bmatrix}. \\ \mathcal{T} \left( \mathbf{a} \right) + \mathcal{T} \left( \mathbf{b} \right) = \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix} + \begin{bmatrix} b_1 + b_2 \\ 3b_1 \end{bmatrix} = \begin{bmatrix} a_1 + a_2 + b_1 + b_2 \\ 3a_1 + 3b_1 \end{bmatrix}. $$

Из этого следует, что

$$ \mathcal{T} \left(\mathbf{a} + \mathbf{b} \right) = \mathcal{T} \left( \mathbf{a} \right) + \mathcal{T} \left( \mathbf{b} \right). $$

**Свойство однородности**

Докажем для данного примера и второе свойство линейности:

$$ \mathcal{T} \left( \lambda \mathbf{a} \right) = \mathcal{T} \left(\lambda \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}\right) = \mathcal{T} \left( \begin{bmatrix} \lambda a_1 \\ \lambda a_2 \end{bmatrix} \right) = \begin{bmatrix} \lambda a_1 + \lambda a_2 \\ 3 \lambda a_1 \end{bmatrix} \\ \lambda \mathcal{T} \left( \mathbf{a} \right) = \lambda \mathcal{T} \left( \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \right) = \lambda \begin{bmatrix} a_1 + a_2 \\ 3a_1 \end{bmatrix} = \begin{bmatrix} \lambda a_1 + \lambda a_2 \\ 3 \lambda a_1 \end{bmatrix}. $$

Из этого следует, что

$$ \mathcal{T} \left( \lambda \mathbf{a} \right) = \lambda \mathcal{T} \left( \mathbf{a} \right). $$

Таким образом, преобразование $\mathcal{T} \left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1 + x_2 \\ 3x_1 \end{bmatrix}$ является линейным.
````

```{admonition} Пример
:class: exercise
Пусть преобразование определено так, что:

$$ \mathcal{T} \left( \mathbf{v} \right) = \mathbf{v} + \mathbf{v}_0, $$

где $\mathbf{v}_0$ представляет собой ненулевой вектор, принадлежащий пространству той же размерности, что и вектор $\mathbf{v}$.

Необходимо определить, является ли данное преобразование линейным.
```

````{dropdown} Решение
Рассмотрим свойство аддитивности для данного линейного преобразования.

$$ \mathcal{T} \left( \mathbf{v}_1 + \mathbf{v}_2 \right) = \mathbf{v}_1 + \mathbf{v}_2 + \mathbf{v}_0. $$

С другой стороны,

$$ \mathcal{T} \left( \mathbf{v}_1 \right) + \mathcal{T} \left( \mathbf{v}_2 \right) = \mathbf{v}_1 + \mathbf{v}_0 + \mathbf{v}_2 + \mathbf{v}_0 = \mathbf{v}_1 + \mathbf{v}_2 + 2 \mathbf{v}_0. $$

Поскольку $\mathbf{v}_0 \neq \mathbf{0}$, то $\mathcal{T} \left( \mathbf{v}_1 + \mathbf{v}_2 \right) \neq \mathcal{T} \left( \mathbf{v}_1 \right) + \mathcal{T} \left( \mathbf{v}_2 \right)$. Следовательно, данное преобразование не является линейным.
````

Рассмотрев данные примеры, можно сделать вывод о том, что не всякое преобразование можно считать линейным. Главное следствие свойств аддитивности и однородности заключается в том, что линейное преобразование полностью определяется тем, как оно действует на элементы вектора. При этом в соответствии с доказанной ранее [теоремой об уникальности разложения вектора по базису](LAB-2-Basis.md#basis-theorem) известно, что любой вектор может быть разложен по некоторому базису с учетом его координат (элементов) в этом базисе и набора базисных векторов единственным образом. Следовательно, принимая во внимание свойства линейности, линейное преобразование произвольного вектора в некотором базисе, по сути, может быть представлено в виде комбинации линейных преобразований базисных векторов. Это наблюдение позволяет нам перейти от абстрактного правила к конкретному инструменту – матрице линейного преобразования.

<a id='lintran-matrix-theorem'></a>
```{admonition} Теорема
:class: theorem
Любое линейное преобразование вектора можно представить в виде произведения ***матрицы преобразования*** на данный вектор.
```

```{admonition} Доказательство
:class: proof
Пусть имеется вектор $\mathbf{x} \in \mathbb{C}^n$, координаты которого заданы в произвольном базисе $\mathrm{B} = \begin{Bmatrix} \mathbf{e}_1, \, \mathbf{e}_2, \, \ldots, \, \mathbf{e}_n \end{Bmatrix}$. Данный вектор может быть представлен в виде следующей суммы (разложен по базису):

$$ \begin{align}
\mathbf{x}
&= \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_n \end{bmatrix} \\
&= x_1 \cdot \mathbf{e}_1 + x_2 \cdot \mathbf{e}_2 + \ldots + x_n \cdot \mathbf{e}_n.
\end{align} $$

Тогда линейное преобразование вектора $\mathbf{x}$:

$$ \begin{align}
\mathcal{T} \left( \mathbf{x} \right)
&= \mathcal{T} \left( x_1 \cdot \mathbf{e}_1 + x_2 \cdot \mathbf{e}_2 + \ldots + x_n \cdot \mathbf{e}_n \right) \\
&= \mathcal{T} \left( x_1 \cdot \mathbf{e}_1 \right) + \mathcal{T} \left( x_2 \cdot \mathbf{e}_2 \right) + \ldots + \mathcal{T} \left( x_n \cdot \mathbf{e}_n \right) \\
&= x_1 \cdot \mathcal{T} \left( \mathbf{e}_1 \right) + x_2 \cdot \mathcal{T} \left( \mathbf{e}_2 \right) + \ldots + x_n \cdot \mathcal{T} \left( \mathbf{e}_n \right).
\end{align}$$

Здесь последовательно были применены свойства аддитивности и однородности линейных преобразований. В свою очередь, $\mathcal{T} \left( \mathbf{e}_i \right), \, i = 1 \, \ldots \, n,$ представляет собой вектор. Следовательно, полученное выражение может быть записано в виде произведения матрицы линейного преобразования и рассматриваемого вектора $\mathbf{x}$:

$$ \begin{align}
\mathcal{T} \left( \mathbf{x} \right)
&= x_1 \cdot \mathcal{T} \left( \mathbf{e}_1 \right) + x_2 \cdot \mathcal{T} \left( \mathbf{e}_2 \right) + \ldots + x_n \cdot \mathcal{T} \left( \mathbf{e}_n \right) \\
&= \begin{bmatrix}
\vert & \vert & & \vert \\
\mathcal{T} \left( \mathbf{e}_1 \right) & \mathcal{T} \left( \mathbf{e}_2 \right) & \ldots & \mathcal{T} \left( \mathbf{e}_n \right) \\
\vert & \vert & & \vert
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}.
\end{align} $$

Таким образом, линейное преобразование любого вектора может быть представлено в виде произведения матрицы на этот же вектор:

$$ \mathcal{T} \left( \mathbf{x} \right) = \mathbf{A} \mathbf{x}. $$

Такая матрица $\mathbf{A}$ называется ***матрицей преобразования*** по отношению к рассматриваемому базису.
```

Следует отметить, что в процессе доказательства теоремы о представлении линейного преобразования с использованием матрицы было введено не только определение матрицы, как упорядоченной совокупности векторов-столбцов, но и ее базовой операции – произведения матрицы и вектора. Подробнее остановимся на матрицах, их свойствах и операциях над ними.

(matrix)=
## Матрицы

(matrix-def)=
### Определение и объявление матрицы

```{admonition} Определение
:class: definition
***Матрица*** представляет собой упорядоченную совокупность векторов-столбцов (или векторов-строк).
```

Для обозначения матрицы здесь и далее будут использоваться заглавные латинские буквы, выделенные жирным: $\mathbf{A}$. Например, следующая запись

$$ \exists ~ \mathbf{A} \in \mathbb{C}^{n \times m} $$

читается так: *существует* $\left( \exists \right)$ *матрица "A"* $\left( \mathbf{A} \right)$ *, принадлежащая* $\left( \in \right)$ *пространству комплексных чисел* $\left( \mathbb{C} \right)$ *размерности* $n \times m$. Данная запись объявляет матрицу $\mathbf{A}$ следующего вида:

$$ \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1m} \\ a_{21} & a_{22} & \ldots & a_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \ldots & a_{nm} \end{bmatrix}. $$

Кроме того, матрица может обозначаться путем указания элементов с индексами, например, для представленного выше примера:

$$ \mathbf{A} = \left\{ a_{ij}, \; i = 1 \ldots n, \; j = 1 \ldots m \right\}. $$

<!-- ```{admonition} Определение
:class: definition
***Рангом матрицы*** с $n$ строками и $m$ столбцами называется максимальное число линейно независимых строк или столбцов матрицы.

https://en.wikipedia.org/wiki/Rank%E2%80%93nullity_theorem
``` -->

Если размерности $m \neq n$, то такая матрица называется *прямоугольной*. Вектор-столбец (вектор-строку) можно рассматривать, как частный случай прямоугольной матрицы с количеством стобцов (строк), равным единице. Если же у матрицы количества строк и столбцов равны, то такая матрица называется *квадратной*.

Далее рассмотрим различные действия, которые можно осуществлять с матрицами, а также их свойства.

(matrix-dot)=
### Произведение матриц

```{admonition} Определение
:class: definition
***Произведением двух матриц*** $\mathbf{A} \in \mathbb{C}^{m \times n}$ и $\mathbf{B} \in \mathbb{C}^{n \times p}$ называется матрица $\mathbf{C} \in \mathbb{C}^{m \times p}$, элемент которой, находящийся на пересечении $i$-ой строки $\left( i = 1 \, \ldots \, m \right)$ и $j$-го столбца $\left( j = 1 \, \ldots \, p \right)$ равен сумме произведений элементов $i$-ой строки матрицы $\mathbf{A}$ на соответствующие (по порядку) элементы $j$-го столбца матрицы $\mathbf{B}$:

$$ c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}, \; i = 1 \, \ldots \, m, \; j = 1 \, \ldots \, p. $$
```

Произведение матриц характеризуется следующими свойствами:

* *некоммутативностью (в общем случае)*: $\mathbf{A} \mathbf{B} \neq \mathbf{B} \mathbf{A}$,
* *ассоциативностью*: $\left( \mathbf{A} \mathbf{B} \right) \mathbf{C} = \mathbf{A} \left( \mathbf{B} \mathbf{C} \right)$,
* *дистрибутивностью*: $\mathbf{A} \left( \mathbf{B} + \mathbf{C} \right) = \mathbf{A} \mathbf{B} + \mathbf{A} \mathbf{C}$ или $\left( \mathbf{B} + \mathbf{C} \right) \mathbf{A} = \mathbf{B} \mathbf{A} + \mathbf{C} \mathbf{A}$,
* *сочетательностью с умножением на число*: $\lambda \left( \mathbf{A} \mathbf{B} \right) = \left( \lambda \mathbf{A} \right) \mathbf{B} = \mathbf{A} \left( \lambda \mathbf{B} \right)$.

Если рассматривать вектор-столбец и вектор-строку как частный случай матрицы, то произведение матрицы и вектора может быть выполнено по тем же правилам.

Важно подчеркнуть, что матричное произведение, несмотря на свой геометрический смысл, выражаемый в линейном преобразовании вектора, по своей сути является алгебраической операцией, выполняемой по определенным правилам и обладающей своими свойствами. Это позволяет отличать данную операцию от эрмитового скалярного произведения векторов.

Элемент $c_{ij}, \, i = 1 \, \ldots \, m, \, j = 1 \, \ldots \, p,$ матрицы $\mathbf{C} \in \mathbb{C}^{m \times p}$, которая получается в результате умножения матрицы $\mathbf{A} \in \mathbb{C}^{m \times n}$ на матрицу $\mathbf{B} \in \mathbb{C}^{n \times p}$, по своей сути является результатом ранее введенной алгебраической операции – [симметричной билинейной формы](LAB-2-Basis.md#bilinear-form), выполняемой над векторами-строками первой матрицы и векторам-столбцами второй. Важно понимать, что в этой чисто алгебраической процедуре (в отличие от эрмитового скалярного произведения векторов) сопряжение не используется. Это обеспечивает симметрию операции на уровне элементов векторов-строк и векторов-столбцов.

Рассмотрим произведение матриц на следующем примере.

```{admonition} Пример
:class: exercise

Пусть матрицы $\mathbf{A} \in \mathbb{C}^{2 \times 3}$ и  $\mathbf{A} \in \mathbb{C}^{3 \times 4}$ заданы следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 2 + 2i & 8 - 3i & -1 + 5i \\ 3 - i & 5 + 2i & 0 \end{bmatrix}, \; \mathbf{B} = \begin{bmatrix} 3 - 1i & 8 + 2i & -2 + 1i & 0 \\ 4 - 3i & 6 - 4i & 9 + 2i & -3 + i \\ 0 & 7 + 2i & 1 - i & 5 + 5i  \end{bmatrix} $$

Необходимо найти матрицу $\mathbf{C} \in \mathbb{C}^{2 \times 4}$, равную произведению матриц $\mathbf{A}$ и $\mathbf{B}$.
```

````{dropdown} Решение
Зададим матрицы $\mathbf{A}$ и $\mathbf{B}$ в виде двумерных массивов:

```python
A = np.array([
[2 + 2j, 8 - 3j, -1 + 5j],
[3 - 1j, 5 + 2j, 0],
])
B = np.array([
[3 - 1j, 8 + 2j, -2 + 1j, 0],
[4 - 3j, 6 - 4j, 9 + 2j, -3 + 1j],
[0, 7 + 2j, 1 - 1j, 5 + 5j],
])
```

Для умножения матрицы $\mathbf{A}$ на матрицу $\mathbf{B}$ можно использовать метод [`dot`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html):

```python
C = A.dot(B)
print(C)
```

```{glue:} dot1
```

Аналогичный результат можно получить, используя символ `@`:

```python
C = A @ B
print(C)
```

```{glue:} dot2
```
````

```{code-cell} python
:tags: [remove-cell]

A = np.array([[2 + 2j, 8 - 3j, -1 + 5j], [3 - 1j, 5 + 2j, 0]])
B = np.array([[3 - 1j, 8 + 2j, -2 + 1j, 0], [4 - 3j, 6 - 4j, 9 + 2j, -3 + 1j], [0, 7 + 2j, 1 - 1j, 5 + 5j]])

C = A.dot(B)

glue('dot1', MultilineText(C))

C = A @ B

glue('dot2', MultilineText(C))
```

```{admonition} Определение
:class: definition
***Нулевой матрицей*** будем называть матрицу, все элементы которой равны нулю.
```

Матричное произведение с нулевой матрицей всегда будет равно нулевой матрице. Для создания нулевой матрицы удобно использовать функцию [`numpy.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html), обязательным аргументом которой является размерность матрицы в виде кортежа:

```{code-cell} python
B = np.zeros((3, 4))
print(B)
```

Соответственно произведение с нулевой матрицей даст нулевую матрицу:

```{code-cell} python
print(A @ B)
```

```{admonition} Определение
:class: definition
***Единичной матрицей*** будем называть квадратную матрицу, на главной диагонали которой расположены единицы, а все остальные элементы равны нулю.
```

Для создания единичной матрицы удобно использовать функции [`numpy.eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) или [`numpy.identity`](https://numpy.org/doc/stable/reference/generated/numpy.identity.html), обязательным аргументом которых является количество строк в единичной матрице:

```{code-cell} python
I = np.eye(3)
print(I)
```

Результатом произведения некоторой матрицы и единичной матрицы будет являться эта же самая матрица:

```{code-cell} python
print(A @ I)
```

(matrix-add)=
### Сложение матриц

```{admonition} Определение
:class: definition
***Сложением*** матриц $\mathbf{A}, \, \mathbf{B} \in \mathbb{C}^{m \times n}$ называется операция, результатом которой является новая матрица $\mathbf{C} \in \mathbb{C}^{m \times n}$, каждый элемент которой равен сумме соответствующих элементов слагаемых матриц:

$$ \mathbf{C} = \mathbf{A} + \mathbf{B} &= \begin{bmatrix} a_{11} & \ldots & a_{1m} \\ \vdots & \ddots & \vdots \\ a_{n1} & \ldots & a_{nm} \end{bmatrix} + \begin{bmatrix} b_{11} & \ldots & b_{1m} \\ \vdots & \ddots & \vdots \\ b_{n1} & \ldots & b_{nm} \end{bmatrix}
= \begin{bmatrix} a_{11} + b_{11} & \ldots & a_{1m} + b_{1m} \\ \vdots & \ddots & \vdots \\ a_{n1} + b_{n1} & \ldots & a_{nm} + b_{nm} \end{bmatrix} $$
```

Данная операция является:

* *коммутативной*: $\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$,
* *ассоциативной*: $\left( \mathbf{A} + \mathbf{B} \right) + \mathbf{C} = \mathbf{A} + \left( \mathbf{B} + \mathbf{C} \right)$.

Рассмотрим сложение матриц на следующем примере:

```{admonition} Пример
:class: exercise

Пусть матрицы $\mathbf{A}, \, \mathbf{B} \in \mathbb{C}^{2 \times 3}$ заданы следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 3 + 2i & 5 - 3i & -17 + 5i \\ -1 - 1i & 0 & 10 - 2i \end{bmatrix}, \; \mathbf{B} = \begin{bmatrix} 1 + 3i & 3 - i & 14 - 5i \\ 5 + 6i & 2 - 2i & -7 + 9i \end{bmatrix}. $$

Необходимо найти матрицу $\mathbf{C} \in \mathbb{C}^{2 \times 3}$, удовлетворяющую условию $\mathbf{C} = \mathbf{A} + \mathbf{B}$.
```

````{dropdown} Решение
Для работы с матрицами будем использовать библиотеку [numpy](https://numpy.org):

```python
import numpy as np
```

Зададим матрицы $\mathbf{A}$ и $\mathbf{B}$ в виде двумерных массивов с использованием функции [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html):

``` python
A = np.array([[3 + 2j, 5 - 3j, -17 + 5j], [-1 - 1j, 0, 10 - 2j]])
B = np.array([[1 + 3j, 3 - 1j, 14 - 5j], [5 + 6j, 2 - 2j, -7 + 9j]])
```

Выполним операцию сложения матриц и выведем полученный результат:

``` python
C = A + B
print(C)
```

```{glue:} add1
```

Аналогичным образом выполняются и другие поэлементные операции с матрицами, в том числе умножение, деление, возведение в степень и пр.
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

A = np.array([[3 + 2j, 5 - 3j, -17 + 5j], [-1 - 1j, 0, 10 - 2j]])
B = np.array([[1 + 3j, 3 - 1j, 14 - 5j], [5 + 6j, 2 - 2j, -7 + 9j]])

C = A + B

glue('add1', MultilineText(C))
```

(matrix-mult)=
### Умножение матрицы на число

```{admonition} Определение
:class: definition
***Произведением матрицы*** $\mathbf{A} \in \mathbb{C}^{m \times n}$ ***на число*** $\lambda \in \mathbb{C}$ называется новая матрица $\mathbf{B} \in \mathbb{C}^{m \times n}$, элементы которой получаются умножением каждого элемента матрицы $\mathbf{A}$ на число $\lambda$:

$$ \mathbf{B} = \lambda \cdot \mathbf{A} = \lambda \mathbf{A} = \lambda \cdot \begin{bmatrix} a_{11} & \ldots & a_{1m} \\ \vdots & \ddots & \vdots \\ a_{n1} & \ldots & a_{nm} \end{bmatrix} = \begin{bmatrix} \lambda \cdot a_{11} & \ldots & \lambda \cdot a_{1m} \\ \vdots & \ddots & \vdots \\ \lambda \cdot a_{n1} & \ldots & \lambda \cdot a_{nm} \end{bmatrix}. $$
```

Данная операция является:
* *коммутативной*: $m \cdot \mathbf{A} = \mathbf{A} \cdot m$,
* *ассоциативной*: $m \cdot \left( n \cdot \mathbf{A} \right) = \left( m \cdot n \right) \cdot \mathbf{A}$,
* *дистрибутивной*: $m \cdot \left( \mathbf{A} + \mathbf{B} \right) = m \cdot \mathbf{A} + m \cdot \mathbf{B}$ или $\left(m + n \right) \cdot \mathbf{A} = m \cdot \mathbf{A} + n \cdot \mathbf{A}$.

Рассмотрим произведение матрицы и числа на следующем примере.

```{admonition} Пример
:class: exercise

Пусть матрица $\mathbf{A} \in \mathbb{R}^{2 \times 3}$ задана следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 2 + 2i & 8 - 3i & -1 + 5i \\ 3 - i & 5 + 2i & 0 \end{bmatrix}. $$

Необходимо найти матрицу $\mathbf{B} \in \mathbb{R}^{2 \times 3}$, равную произведению матрицы $\mathbf{A}$ на число $4 - 2i$.
```

````{dropdown} Решение
Зададим матрицу $\mathbf{A}$ в виде двумерного массива:

```python
A = np.array([[2 + 2j, 8 - 3j, -1 + 5j], [3 - 1j, 5 + 2j, 0]])
```

Умножим данный массив на число `4 - 2j` и выведем полученный результат:

```python
B = (4 - 2j) * A
print(B)
```

```{glue:} mult1
```

Аналогичным образом выполняются и другие операции между скаляром и матрицей: сложение, вычитание, деление, возведение в степень и пр.
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

A = np.array([[2 + 2j, 8 - 3j, -1 + 5j], [3 - 1j, 5 + 2j, 0]])

B = (4 - 2j) * A

glue('mult1', MultilineText(B))
```

(matrix-transp)=
### Транспонирование матрицы

Для того чтобы ***транспонировать матрицу***, нужно ее столбцы записать в строки транспонированной матрицы. Транспонирование обозначается символом $^\top$ или изменением индексов матрицы:

$$ \begin{align}
& \mathbf{A} = \begin{Bmatrix} a_{ij}, \; i = 1 \, \ldots \, n, \; j = 1 \, \ldots \, m \end{Bmatrix}, \\
& \mathbf{A}^{\top} = \begin{Bmatrix} a_{ji}, \; j = 1 \, \ldots \, m, \; i = 1 \, \ldots \, n \end{Bmatrix}.
\end{align} $$

Операция транспонирования является:

* *дистрибутивной относительно сложения*: $\left( \mathbf{A} + \mathbf{B} \right)^\top = \mathbf{A}^\top + \mathbf{B}^\top$,
* *дистрибутивной относительно умножения на скаляр*: $\left( \lambda \mathbf{A} \right)^\top = \lambda \mathbf{A}^\top$.

Операция транспонирования с использованием [numpy](https://numpy.org) осуществляется путем обращения к атрибуту [`T`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html) некоторого массива:

```{code-cell} python
print(A)
```

```{code-cell} python
print(A.T)
```

Одним из свойств транспонирования является следующее:

<a id='matrix-transp-prop'></a>
```{admonition} Свойство
:class: property
Транспонирование произведения $n$ матриц равняется произведению в обратном порядке $n$ транспонированных матриц:

$$ \left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right)^\top = \mathbf{A}_n^\top \, \ldots \, \mathbf{A}_2^\top \mathbf{A}_1^\top. $$
```

```{admonition} Доказательство
:class: proof
Для доказательства данного свойства воспользуемся [методом математической индукции](https://en.wikipedia.org/wiki/Mathematical_induction). Данный подход к математическому доказательству зачастую используется, когда необходимо доказать истинность утверждения для всех натуральных чисел (в данном случае такая последовательность будет задаваться количеством матриц $n$, участвующих в произведении). Метод математической индукции состоит из двух этапов:

1. База индукции: осуществляется проверка истинности утверждения для минимального значения последовательности (в данном случае $n = 2$).
2. Индукционный переход: утверждение для произвольного $n$ принимается истинным, доказывается истинность утверждения для $\left( n + 1 \right)$.

Выполнив доказательство истинности утверждения для $\left( n + 1 \right)$ с учетом его истинности как для минимального, так и для произвольного $n$, делается вывод об истинности утверждения для всей последовательности натуральных чисел.

**База индукции**

Пусть $\mathbf{A} \in \mathbb{C}^{m \times n}$ и $\mathbf{B} \in \mathbb{C}^{n \times l}$ представляют собой матрицы.

Обозначим $\mathbf{a}_i^\top \in \mathbb{C}^n, \, i = 1 \, \ldots \, m,$ векторы-строки матрицы $\mathbf{A}$:

$$ \mathbf{A} = \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_1^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_2^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_m^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix}. $$

Обозначим $\mathbf{b}_j \in \mathbb{C}^n, \, j = 1 \, \ldots \, l,$ векторы-столбцы матрицы $\mathbf{B}$:

$$ \mathbf{B} = \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{b}_1 & \mathbf{b}_2 & \ldots & \mathbf{b}_l \\ \vert & \vert & & \vert \end{bmatrix}. $$

Распишем подробнее транспонированное произведение $\left( \mathbf{A} \mathbf{B} \right)^\top$:

$$ \begin{align}
\left( \mathbf{A} \mathbf{B} \right)^\top
&= \left( \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_1^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_2^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_m^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{b}_1 & \mathbf{b}_2 & \ldots & \mathbf{b}_l \\ \vert & \vert & & \vert \end{bmatrix} \right)^\top \\
&= \begin{bmatrix}
\mathbf{a}_1^\top \mathbf{b}_1 & \mathbf{a}_1^\top \mathbf{b}_2 & \ldots & \mathbf{a}_1^\top \mathbf{b}_l \\
\mathbf{a}_2^\top \mathbf{b}_1 & \mathbf{a}_2^\top \mathbf{b}_2 & \ldots & \mathbf{a}_2^\top \mathbf{b}_l \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a}_m^\top \mathbf{b}_1 & \mathbf{a}_m^\top \mathbf{b}_2 & \ldots & \mathbf{a}_m^\top \mathbf{b}_l \\
\end{bmatrix}^\top = \begin{bmatrix}
\mathbf{a}_1^\top \mathbf{b}_1 & \mathbf{a}_2^\top \mathbf{b}_1 & \ldots & \mathbf{a}_m^\top \mathbf{b}_1 \\
\mathbf{a}_1^\top \mathbf{b}_2 & \mathbf{a}_2^\top \mathbf{b}_2 & \ldots & \mathbf{a}_m^\top \mathbf{b}_2 \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a}_1^\top \mathbf{b}_l & \mathbf{a}_2^\top \mathbf{b}_l & \ldots & \mathbf{a}_m^\top \mathbf{b}_l \\
\end{bmatrix}
\end{align} $$

В свою очередь, произведение транспонированных матриц $\mathbf{A}$ и $\mathbf{B}$ можно преобразовать следующим образом:

$$ \begin{align}
\mathbf{B}^\top \mathbf{A}^\top
&= \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{b}_1^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{b}_2^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{b}_l^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_m \\ \vert & \vert & & \vert \end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{b}_1^\top \mathbf{a}_1 & \mathbf{b}_1^\top \mathbf{a}_2 & \ldots & \mathbf{b}_1^\top \mathbf{a}_m \\
\mathbf{b}_2^\top \mathbf{a}_1 & \mathbf{b}_2^\top \mathbf{a}_2 & \ldots & \mathbf{b}_2^\top \mathbf{a}_m \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{b}_l^\top \mathbf{a}_1 & \mathbf{b}_l^\top \mathbf{a}_2 & \ldots & \mathbf{b}_l^\top \mathbf{a}_m \\
\end{bmatrix} = \begin{bmatrix}
\mathbf{a}_1^\top \mathbf{b}_1 & \mathbf{a}_2^\top \mathbf{b}_1 & \ldots & \mathbf{a}_m^\top \mathbf{b}_1 \\
\mathbf{a}_1^\top \mathbf{b}_2 & \mathbf{a}_2^\top \mathbf{b}_2 & \ldots & \mathbf{a}_m^\top \mathbf{b}_2 \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a}_1^\top \mathbf{b}_l & \mathbf{a}_2^\top \mathbf{b}_l & \ldots & \mathbf{a}_m^\top \mathbf{b}_l \\
\end{bmatrix}
\end{align} $$

В данном преобразовании было применено [коммутативное свойство симметричной билинейной формы](LAB-2-Basis.md#bilinear-form).

Таким образом, транспонирование произведения двух матриц равняется произведению двух транспонированных матриц, записанных в обратном порядке. Следовательно, база индукции истинна.

**Индукционный переход**

Пусть для некоторого натурального $n$ истинно следующее выражение:

$$ \left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right)^\top = \mathbf{A}_n^\top \, \ldots \, \mathbf{A}_2^\top \mathbf{A}_1^\top. $$

Докажем истинность утверждения для $\left( n + 1 \right)$:

$$ \begin{align}
\left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \mathbf{A}_{n+1} \right)^\top
&= \left( \left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right) \mathbf{A}_{n+1} \right)^\top \\
&= \mathbf{A}_{n+1}^\top \left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right)^\top \\
&= \mathbf{A}_{n+1}^\top \mathbf{A}_n^\top \, \ldots \, \mathbf{A}_2^\top \mathbf{A}_1^\top.
\end{align} $$

Следует отметить, что данное свойство будет верно, если в качестве множителей будут участвовать векторы при соблюдении согласованности размерности произведения.
```

Следствием из данного свойства является утверждение о том, что закон обратного порядка справедлив и для комплексно-сопряженного транспонирования произведения:

<a id='matrix-transp-cor-complex'></a>
```{admonition} Следствие
:class: corollary
Комплексно-сопряженное транспонирование произведения $n$ матриц равняется произведению в обратном порядке $n$ комплексно-сопряженных транспонированных матриц:

$$ \left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right)^\dagger = \mathbf{A}_n^\dagger \, \ldots \, \mathbf{A}_2^\dagger \mathbf{A}_1^\dagger. $$
```

```{admonition} Доказательство
:class: proof
Действительно, поскольку операция комплексно-сопряженного транспонирования, по сути, состоит из двух операций: замены знаков у мнимной части на обратные и транспонирования, то можно выполнить следующие преобразования:

$$ \left( \mathbf{A}_1 \mathbf{A}_2 \, \ldots \, \mathbf{A}_n \right)^\dagger = \left( \bar{\mathbf{A}}_1 \bar{\mathbf{A}}_2 \, \ldots \, \bar{\mathbf{A}}_n \right)^\top = \bar{\mathbf{A}}_n^\top \, \ldots \, \bar{\mathbf{A}}_2^\top \bar{\mathbf{A}}_1^\top = \mathbf{A}_n^\dagger \, \ldots \, \mathbf{A}_2^\dagger \mathbf{A}_1^\dagger. $$

В процессе преобразования было учтено равенство сопряжения произведения матриц произведению сопряженных матриц, вытекающее из [свойств операции сопряжения над комплексными числами](LAB-1-Vectors.md#complex-numbers), когда сопряжение суммы или произведения чисел равняется соответственно сумме или произведению сопряженных чисел.
```

Следует отметить, что в литературе данные утвеждения могут называться законом обратного порядка (*reverse order law*) для некоторой операции, например, транспонирования, комплексно-сопряженного транспонирования или даже [обращения матриц](LAB-3-LinearTransformations.md#matrix-inv).

<a id='matrix-transp-symmetric'></a>
```{admonition} Определение
:class: definition
***Симметричной матрицей*** называется *квадратная матрица*, совпадающая со своей транспонированной матрицей:

$$ \mathbf{A} = \mathbf{A}^\top. $$
```

Например, единичная матрица является симметричной:

```{code-cell} python
print(np.allclose(I, I.T))
```

Симметричная матрица получается в результате тензорного произведения двух коллинеарных векторов. Для обозначения тензорного произведения векторов используется символ $\otimes$. Также тензорное произведение двух векторов-стобцов может обозначаться $\mathbf{a} \mathbf{b}^\top$. В общем случае определение тензорному произведению векторов даяется следующим образом:

$$ \forall ~ \mathbf{a} \in \mathbb{C}^{m}, \, \mathbf{b} \in \mathbb{C}^{n} ~ \exists ~ \mathbf{a} \otimes \mathbf{b} = \mathbf{M} \in \mathbb{C}^{m \times n} ~ : ~ M_{ij} = a_i \cdot b_j, \; i = 1 \, \ldots \, m, \; j = 1 \, \ldots \, n. $$

Тензорное произведение:

* является *коммутативным*, если векторы коллинеарны: $\mathbf{a} \otimes \mathbf{b} = \mathbf{a} \otimes \left( \lambda \mathbf{a} \right) = \lambda \left( \mathbf{a} \otimes \mathbf{a} \right)$, то есть в результате получается симметричная матрица,
* *дистрибутивно:* $\mathbf{a} \otimes \left( \mathbf{b} + \mathbf{c} \right) = \mathbf{a} \otimes \mathbf{b} + \mathbf{a} \otimes \mathbf{c}$,
* *ассоциативно:* $\left( \mathbf{a} \otimes \mathbf{b} \right) \otimes \mathbf{c} = \mathbf{a} \otimes \left( \mathbf{b} \otimes \mathbf{c} \right)$.

Для выполнения тензорного произведения в [numpy](https://numpy.org) можно использовать функцию [`numpy.outer`](https://numpy.org/doc/stable/reference/generated/numpy.outer.html). Другим более быстрым и универсальным способом (с его помощью можно выполнять и другие тензорные операции – сложение, вычитание, деление и т.д.) выполнения тензорного произведения является [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html):

```{code-cell} python
b = np.array([1 + 3j, 2 - 2j, 4 + 1j])
print(b[:, None] * b)
```

Кроме того, симметричная матрица образуется при матричном умножении некоторой матрицы на  $\mathbf{A} \mathbf{A}^\top$ или $\mathbf{A}^\top \mathbf{A}$:

<a id='symmetric-transpdot'></a>
```{admonition} Следствие
:class: corollary
Для любой матрицы $\mathbf{A} \in \mathbb{C}^{m \times n}$ (даже не квадратной) произведения $\mathbf{A} \mathbf{A}^\top$ и $\mathbf{A}^\top \mathbf{A}$ всегда являются симметричными матрицами.
```

```{admonition} Доказательство
:class: proof
Для того чтобы доказать, что матрица $\mathbf{B} = \mathbf{A} \mathbf{A}^\top$, где $\mathbf{A} \in \mathbb{C}^{m \times n}$ и $\mathbf{B} \in \mathbb{C}^{m \times m}$, является симметричной, необходимо доказать справедливость равенства $\mathbf{B} = \mathbf{B}^\top$.

Выполним следующие преобразования матрицы $\mathbf{B}^\top$, применив доказанный ранее [закон обратного порядка для транспонирования произведения матриц](LAB-2-Matrices.md#matrix-transp-prop):

$$ \mathbf{B}^\top = \left( \mathbf{A} \mathbf{A}^\top \right)^\top = \left( \mathbf{A}^\top \right)^\top \mathbf{A}^\top = \mathbf{A} \mathbf{A}^\top = \mathbf{B}. $$

Таким образом, доказав справедливость равенства $\mathbf{B} = \mathbf{B}^\top$, мы доказали, что матрица $\mathbf{B} = \mathbf{A} \mathbf{A}^\top$ является симметричной. Аналогично доказывается для случая $\mathbf{B} = \mathbf{A}^\top \mathbf{A}$.
```

Данное свойство можно проиллюстрировать следующими примерами:

```{code-cell} python
A @ A.T
```

```{code-cell} python
A.T @ A
```

Для пространства комплексных чисел также существует понятие *эрмитовой (самосопряженной)* матрицы.

<a id='matrix-transp-hermitian'></a>
```{admonition} Определение
:class: definition
***Эрмитовой матрицей*** называется *квадратная матрица*, совпадающая со своим эрмитовым сопряжением:

$$ \mathbf{A} = \mathbf{A}^\dagger. $$
```

Иными словами, для каждого элемента матрицы $\mathbf{A} \in \mathbb{C}^{n \times n}$ должно выполняться следующее равенство:

$$ a_{ij} = \bar{a}_{ji} \; \forall \; i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n. $$

Из этого следует, что на главной диагонали эрмитовой матрицы находятся действительные числа:

$$ a_{ii} = \bar{a}_{ii} \Leftrightarrow a_{ii} \in \mathbb{R} \; \forall \; i = 1 \, \ldots \, n. $$

Если эрмитова матрица состоит из действительных чисел, то она является симметричной.

Эрмитова матрица получается в результате [тензорного произведения вектора](LAB-1-Vectors.md#vector-outer) на его самосопряженный вектор:

```{code-cell} python
b = np.array([1 + 3j, 2 - 2j, 4 + 1j])
print(np.outer(b, b.conjugate()))
```

Кроме того, эрмитова матрица образуется при умножении $\mathbf{A} \mathbf{A}^\dagger$ или $\mathbf{A}^\dagger \mathbf{A}$:

<a id='matrix-transp-cor-hermitian'></a>
```{admonition} Следствие
:class: corollary
Для любой матрицы $\mathbf{A} \in \mathbb{C}^{m \times n}$ (даже не квадратной) произведения $\mathbf{A} \mathbf{A}^\dagger$ и $\mathbf{A}^\dagger \mathbf{A}$ всегда являются эрмитовыми матрицами.
```

```{admonition} Доказательство
:class: proof
Для того чтобы доказать, что матрица $\mathbf{B} = \mathbf{A} \mathbf{A}^\dagger$, где $\mathbf{A} \in \mathbb{C}^{m \times n}$ и $\mathbf{B} \in \mathbb{C}^{m \times m}$, является эрмитовой, необходимо доказать справедливость равенства $\mathbf{B} = \mathbf{B}^\dagger$.

Выполним следующие преобразования матрицы $\mathbf{B}^\dagger$, применив доказанный ранее [закон обратного порядка для комплексно-сопряженного транспонирования произведения матриц](LAB-2-Matrices.md#matrix-transp-cor-complex):

$$ \mathbf{B}^\dagger = \left( \mathbf{A} \mathbf{A}^\dagger \right)^\dagger = \left( \mathbf{A}^\dagger \right)^\dagger \mathbf{A}^\dagger = \mathbf{A} \mathbf{A}^\dagger = \mathbf{B}. $$

Таким образом, доказав справедливость равенства $\mathbf{B} = \mathbf{B}^\dagger$, мы доказали, что матрица $\mathbf{B} = \mathbf{A} \mathbf{A}^\dagger$ является эрмитовой. Аналогично доказывается для случая $\mathbf{B} = \mathbf{A}^\dagger \mathbf{A}$.
```

Данное свойство можно проиллюстрировать следующими примерами:

```{code-cell} python
A @ A.conjugate().T
```

```{code-cell} python
A.conjugate().T @ A
```

<!-- (matrix-trace)=
## След матрицы

<a id='def-matrix-trace'></a>
```{admonition} Определение
:class: definition
***Следом матрицы*** называют веичину, численно равную сумме компонентов главной диагонали квадратной матрицы.
```

Для нахождения следа матрицы существует функция [`numpy.trace`](https://numpy.org/doc/stable/reference/generated/numpy.trace.html):

```{code-cell} python
A = np.array([[1., 3., 14.], [5., 2., -7.], [4., -2., 7.]])
print(A, np.trace(A), sep='\n')
``` -->

Общая теория линейных преобразований позволяет описывать любые деформации пространства и выражать их в виде матричного произведения. Однако в приложениях наибольший интерес представляют преобразования, обладающие специальной внутренней структурой. В данном разделе подробнее остановимся на унитарных преобразованиях, сохраняющих результат [эрмитового скалярного произведения](LAB-1-Vectors.md#vector-dot) двух векторов.

<a id='lintran-matrix-unitary'></a>
```{admonition} Определение
:class: definition
***Унитарным линейным преобразованием*** называется линейное преобразование, сохраняющее результат [эрмитового скалярного произведения](LAB-1-Vectors.md#vector-dot) двух векторов.
```

Если рассматривать пространство действительных чисел, то данное преобразование сводится к *ортогональному линейному преобразованию*. Унитарное преобразование получило такое название, поскольку в ортонормированном базисе выражется в виде [унитарной матрицы](LAB-2-Matrices.md#matrix-unitary).

```{admonition} Свойство
:class: property
В ортонормированном базисе унитарное линейное преобразование выражается в виде [унитарной матрицы](LAB-2-Matrices.md#matrix-unitary).
```

```{admonition} Доказательство
:class: proof
Пусть $\mathcal{U} \left( \cdot \right)$ является унитарным линейным преобразованием. Тогда, по определению, оно не приводит к изменению результата эрмитового скалярного произведения двух векторов:

$$ \mathbf{a}^\dagger \mathbf{b} = \mathcal{U} \left( \mathbf{a} \right)^\dagger \mathcal{U} \left( \mathbf{b} \right) \; \forall \; \mathbf{a}, \, \mathbf{b} \in \mathbb{C}^n. $$

```

<a id='theorem-lintran-orthogonal'></a>
```{admonition} Теорема
:class: danger
Если квадратная матрица $\mathbf{P} \in \mathbb{R}^{n \times n}$ является [ортогональной](LAB-2-Matrices.md#matrix-orthogonal) (ее столбцы являются векторами, образующими ортонормированный базис), то для любого вектора $\mathbf{x} \in \mathbb{R}^{n}$ выполняется следующее равенство:

$$ \lVert \mathbf{P} \mathbf{x} \rVert_2 = \lVert \mathbf{x} \rVert_2. $$
```

```{admonition} Доказательство
:class: proof
Пусть вектор $\mathbf{x} \in \mathbb{R}^{n}$ является единичным, то есть $\mathbf{x}^\top \mathbf{x} = 1$ и задан в стандартном (каноническом) базисе, составленном из ортонормальных векторов. Рассмотрим базис $\mathrm{B} = \begin{Bmatrix} \mathbf{u}_1, \, \mathbf{u}_2, \, \ldots, \, \mathbf{u}_n \end{Bmatrix}$, также составленный из ортонормальных векторов. Обозначим $\mathbf{y} = \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathrm{B} = \begin{bmatrix} y_1, \, y_2, \, \ldots, \, y_n \end{bmatrix}$ координаты вектора $\mathbf{x}$ в базисе $\mathrm{B}$. В соответствии с доказанным ранее будет справедливо следующее соотношение:

$$ \mathbf{x} = \mathbf{P} \mathbf{y}, $$

где $\mathbf{P} \in \mathbb{R}^{n \times n}$ представляет собой квадратную матрицу, составленную из базисных векторов базиса $\mathrm{B}$:

$$ \mathbf{P} = \begin{bmatrix}
\vert & \vert & & \vert \\
\mathbf{u}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\
\vert & \vert & & \vert
\end{bmatrix}. $$

Тогда произведение вектора $\mathbf{x}$ на самого себя:

$$ \mathbf{x}^\top \mathbf{x} = \left( \mathbf{P} \mathbf{y} \right)^\top \left( \mathbf{P} \mathbf{y} \right). $$

Применим доказанное ранее [свойство транспонирования произведения](LAB-2-Matrices.md#theorem-matrix-transp-dot):

$$ \mathbf{x}^\top \mathbf{x} = \mathbf{y}^\top \mathbf{P}^\top \mathbf{P} \mathbf{y}. $$

Рассмотрим произведение матрицы $\mathbf{P}$ на саму себя:

$$ \mathbf{P}^\top \mathbf{P} = \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_1^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_2^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{u}_n^\top & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{u}_1 & \mathbf{u}_2 & \ldots & \mathbf{u}_n \\ \vert & \vert & & \vert \end{bmatrix}. $$

Поскольку векторы $\mathbf{u}_1, \, \mathbf{u}_2, \, \ldots, \, \mathbf{u}_n$ являются ортонормальными по условию, то в результате произведения матрицы $\mathbf{P}$ на саму себя получится единичная матрица:

$$ \mathbf{P}^\top \mathbf{P} = \mathbf{P} \mathbf{P}^\top = \mathbf{I}, $$

а матрица $\mathbf{P}$ называется [ортогональной](LAB-2-Matrices.md#matrix-orthogonal) $\left( \mathbf{P}^{-1} = \mathbf{P}^\top \right)$. Тогда:

$$ \mathbf{x}^\top \mathbf{x} = \mathbf{y}^\top \mathbf{I} \mathbf{y} = \mathbf{y}^\top \mathbf{y} = 1. $$

Таким образом, в результате линейного преобразования, выраженного ортогональной матрицей, длина вектора не изменилась.
```

(matrix-unitary)=
## Унитарная матрица

```{admonition} Определение
:class: definition
Квадратная матрица $\mathbf{A} \in \mathbb{C}^{n \times n}$ называется унитарной, если все ее векторы-столбцы единичны (имеют [длину](LAB-1-Vectors.md#vector-length), равную единице) и ортогональны друг другу (попарные [эрмитовы скалярные произведения](LAB-1-Vectors.md#vector-dot) ее векторов-столбцов равны нулю).
```

Если рассматривается пространство действительных чисел, то эрмитово скалярное произведение сводится к простому скалярному произведению, а прилагательное "унитарная" заменяется на "ортогональная".

Основным свойством унитарных матриц является равенство ее обратной матрицы и сопряженно транспонированной:

```{admonition} Свойство
:class: property
Если квадратная матрица $\mathbf{A} \in \mathbb{C}^{n \times n}$ является унитарной, то для нее справедливо следующее соотношение:

$$ \mathbf{A} \mathbf{A}^\dagger = \mathbf{A}^\dagger \mathbf{A} = \mathbf{I}. $$
```

```{admonition} Доказательство
:class: proof
Для доказательства распишем подробнее произведение:

$$ \begin{align}
\mathbf{A}^\dagger \mathbf{A}
&= \begin{bmatrix} \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_1^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_2^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \\ & \vdots & \\ \frac{\hspace{0.5cm}}{\hspace{0.5cm}} & \mathbf{a}_n^\dagger & \frac{\hspace{0.5cm}}{\hspace{0.5cm}} \end{bmatrix} \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_n \\ \vert & \vert & & \vert \end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{a}_1^\dagger \mathbf{a}_1 & \mathbf{a}_1^\dagger \mathbf{a}_2 & \ldots & \mathbf{a}_1^\dagger \mathbf{a}_n \\
\mathbf{a}_2^\dagger \mathbf{a}_1 & \mathbf{a}_2^\dagger \mathbf{a}_2 & \ldots & \mathbf{a}_2^\dagger \mathbf{a}_n \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a}_n^\dagger \mathbf{a}_1 & \mathbf{a}_n^\dagger \mathbf{a}_2 & \ldots & \mathbf{a}_n^\dagger \mathbf{a}_n
\end{bmatrix} \\
&= \begin{bmatrix}
1 & 0 & \ldots & 0 \\
0 & 1 & \ldots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \ldots & 1
\end{bmatrix}
\end{align} $$

При преобразовании учитывались следующие свойства векторов:

* эрмитово скалярное произведение единичного вектора на самого себя равняется единице;
* эрмитово скалярное произведение двух ортогональных векторов равняется нулю.

Поскольку $\mathbf{A}^\dagger \mathbf{A} = \mathbf{I}$ и $\mathbf{A}^{-1} \mathbf{A} = \mathbf{I}$, следовательно, $\mathbf{A}^\dagger = \mathbf{A}^{-1}$.
```

(matrix-norm)=
## Норма матрицы

Данный раздел является необязательным, но рекомендуемым к изучению. Ранее было введено понятие [нормы вектора](LAB-1-Vectors.md#vector-norm). Данное понятие можно распространить и на матрицы.

```{admonition} Определение
:class: definition
***Операторной (подчиненной или индуцированной) нормой*** матрицы будем называть:

$$ \lVert \mathbf{A} \rVert_p = \sup_{\lVert \mathbf{x} \rVert_p \neq 0} \frac{ \lVert \mathbf{A} \mathbf{x} \rVert_p }{ \lVert \mathbf{x} \rVert_p } = \sup_{\lVert \mathbf{u} \rVert_p = 1} \lVert \mathbf{A} \mathbf{u} \rVert_p, $$

где $p$ – параметр нормы; вектор $\mathbf{u}$ – единичный вектор; оператор $\mathrm{sup}$ ([супремум](https://en.wikipedia.org/wiki/Infimum_and_supremum)) характеризует наименьшее из чисел, ограничивающих сверху множество $\lVert \mathbf{A} \mathbf{x} \rVert_p \, / \lVert \mathbf{x} \rVert_p$ при различных ненулевых векторах $\mathbf{x}$, и для конечномерных матриц, с которыми будем работать здесь и далее, его можно заменить на оператор $\max$:

$$ \lVert \mathbf{A} \rVert_p = \max_{\lVert \mathbf{x} \rVert_p \neq 0} \frac{ \lVert \mathbf{A} \mathbf{x} \rVert_p }{ \lVert \mathbf{x} \rVert_p } = \max_{\lVert \mathbf{u} \rVert_p = 1} \lVert \mathbf{A} \mathbf{u} \rVert_p, \; \mathbf{A} \in \mathbb{C}^{m \times n}. $$
```

В ходе преобразований использовалось свойство однородности [нормы вектора](LAB-1-Vectors.md#vector-norm). Таким образом, геометрический смысл нормы матрицы заключается в определении максимального изменения вектора (среди множества единичных векторов, образующих единичную сферу) в результате его линейного преобразования, выраженного данной матрицей.

Как и для векторов, одними из наиболее часто использующихся норм матриц являются $L_1$-норма (Манхэттенская норма), $L_2$-норма (Евклидова норма) и $L_\infty$-норма (норма Чебышёва). Для получения выражения для каждой из норм необходимо решить задачу, состоящую из двух шагов:

1. Необходимо найти верхнюю границу $p$-нормы вектора $\mathbf{A} \mathbf{u}$ среди всех возможных единичных векторов $\mathbf{u}$ (доказательство того, что найденное значение $p$-нормы является ограничением сверху).
2. Необходимо найти как минимум один единичный вектор $\mathbf{u}_*$, такой что $\lVert \mathbf{u}_* \rVert_p = 1$ и $\lVert \mathbf{A} \mathbf{u}_* \rVert_p = \max_{\lVert \mathbf{u} \rVert_p = 1} \lVert \mathbf{A} \mathbf{u} \rVert_p$ (доказательство того, что определенная граница сверху достижима в заданных условиях).

<a id='matrix-norm-infty'></a>
Начнем с $L_\infty$-нормы. Обозначим $\mathbf{A} \mathbf{u} = \mathbf{v}$, где $\mathbf{A} \in \mathbb{C}^{m \times n}$, $\mathbf{u} \in \mathbb{C}^n$ и $\mathbf{v} \in \mathbb{C}^m$. Тогда:

$$ \lVert \mathbf{A} \mathbf{u} \rVert_\infty = \lVert \mathbf{v} \rVert_\infty = \max_i \left| v_i \right| = \max_i \left| \sum_{j=1}^n a_{ij} u_j \right| \leq \max_i \sum_{j=1}^n \left| a_{ij} \right| \left| u_j \right| . $$

Поскольку $\lVert \mathbf{u} \rVert_\infty = \max_j \left| u_j \right| = 1$, то есть максимальное значение, которое может принимать величина $\left| u_j \right|, \, j = 1 \, \ldots \, n,$ равняется единице, то:

$$ \lVert \mathbf{A} \rVert_\infty = \max_{\lVert \mathbf{u} \rVert_\infty = 1} \lVert \mathbf{A} \mathbf{u} \rVert_\infty = \max_i \sum_{j=1}^n \left| a_{ij} \right|. $$

То есть $L_\infty$-норма матрицы $\mathbf{A}$ представляет собой максимальное значение среди построчных сумм абсолютных значений этой матрицы.

Для $L_\infty$-нормы вектор $\mathbf{u}_*$ определяется следующим образом:

$$ \mathbf{u}_* = \begin{bmatrix} \frac{\bar{a}_{i1}}{\left| a_{i1} \right|} & \frac{\bar{a}_{i2}}{\left| a_{i2} \right|} & \ldots & \frac{\bar{a}_{in}}{\left| a_{in} \right|} \end{bmatrix}^\top, $$

где $i = 1 \, \ldots \, m$ представляет собой номер строки матрицы $\mathbf{A}$ с наибольшей суммой $\sum_{j=1}^n \left| a_{ij} \right|$.

Если матрица составлена из действительных чисел, то выражение для вектора $\mathbf{u}_*$ сводится к следующему:

$$ \mathbf{u}_* = \begin{bmatrix} \mathrm{sgn} \left( a_{i1} \right) & \mathrm{sgn} \left( a_{i2} \right) & \ldots & \mathrm{sgn} \left( a_{in} \right) \end{bmatrix}^\top, $$

где функция $\mathrm{sgn} \left( \cdot \right)$ возвращает знак действительного числа.

Рассмотрим нахождение $L_\infty$-нормы на следующем примере.

```{admonition} Пример
:class: exercise
Пусть матрица $\mathbf{A} \in \mathbb{R}^{3 \times 4}$ задана следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 1 & 3 & 14 & -2 \\ 5 & 2 & -7 & 0 \\ 4 & -2 & 7 & 3 \end{bmatrix}. $$

Необходимо найти $L_\infty$-норму данной матрицы.
```

````{dropdown} Решение
Зададим матрицу $\mathbf{A}$ в виде двумерного массива:

```python
A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])
```

Для нахождения абсолютных значений можно использовать функцию [`numpy.abs`](https://numpy.org/doc/stable/reference/generated/numpy.abs.html), для нахождения суммы – метод [`sum`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sum.html), а для вычисления максимального значения – метод [`max`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html). С учетом полученного выше выражения посчитаем и выведем $L_\infty$-норму данной матрицы:

```python
print(np.abs(A).sum(axis=1).max())
```

```{glue:} norm1
```

В качестве значения аргумента `axis` метода [`sum`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sum.html) было передано значение `1`, позволяющее задать суммирование вдоль первой оси (то есть элементов в каждой строке), считая от нуля.

Аналогичный результат может быть получен с использованием функции [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html):

```python
print(np.linalg.norm(A, ord=np.inf))
```

```{glue:} norm2
```

Здесь в качестве значения аргумента `ord` была передана [константа, имитирующая бесконечность](https://numpy.org/doc/stable/reference/constants.html#numpy.inf).

Покажем, что данное значение достижимо при определенном векторе $\mathbf{u}_*$. Получим элементы данного вектора в виде одномерного массива:

```python
sums = np.abs(A).sum(axis=1)
row = np.argmax(sums)
u = np.sign(A[row])
print(row, u)
```

```{glue:} norm3
```

Для вычисления индекса наибольшего элемента массива использовалась функция [`numpy.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html), а для получения массива, состоящих из знаков элементов другого массива, – функция [`numpy.sign`](https://numpy.org/doc/stable/reference/generated/numpy.sign.html).

Покажем, что $\lVert \mathbf{A} \mathbf{u}_* \rVert_\infty = \max_i \left| \sum_j a_{ij} {u_*}_j \right|$ равняется найденной $L_\infty$-норме матрицы $\mathbf{A}$:

```python
print(np.abs(A.dot(u)).max())
```

```{glue:} norm4
```
````

```{code-cell} python
:tags: [remove-cell]

A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])

glue('norm1', MultilineText(np.abs(A).sum(axis=1).max()))

glue('norm2', MultilineText(np.linalg.norm(A, ord=np.inf)))

sums = np.abs(A).sum(axis=1)
row = np.argmax(sums)
u = np.sign(A[row])

glue('norm3', MultilineText(row, u))

glue('norm4', MultilineText(np.abs(A.dot(u)).max()))
```

<a id='matrix-norm-1'></a>
Теперь рассмотрим $L_1$-норму. Обозначим $\mathbf{A} \mathbf{u} = \mathbf{v}$, где $\mathbf{A} \in \mathbb{C}^{m \times n}$, $\mathbf{u} \in \mathbb{C}^n$ и $\mathbf{v} \in \mathbb{C}^m$. Тогда:

$$ \lVert \mathbf{A} \mathbf{u} \rVert_1 = \lVert \mathbf{v} \rVert_1 = \sum_{i=1}^m \left| v_i \right| = \sum_{i=1}^m \left| \sum_{j=1}^n a_{ij} u_j \right| \leq \sum_{i=1}^m \sum_{j=1}^n \left| a_{ij} \right| \left| u_j \right| . $$

Обозначим сумму абсолютных значений элементов матрицы в каждом столбце $\sum_{i=1}^m \left| a_{ij} \right| = c_j, \, j = 1 \, \ldots \, n,$ тогда:

$$ \lVert \mathbf{A} \mathbf{u} \rVert_1 \leq \sum_{j=1}^n \left| u_j \right| c_j \leq \sum_{j=1}^n \left| u_j \right| \max_k c_k. $$

Поскольку $\lVert \mathbf{u} \rVert_1 = \sum_{j=1}^n \left| u_j \right| = 1$ по условию, то полученное неравенство преобразуется к следующему виду:

$$ \lVert \mathbf{A} \mathbf{u} \rVert_1 \leq \max_j c_j \Leftrightarrow \lVert \mathbf{A} \mathbf{u} \rVert_1 \leq \max_j \sum_{i=1}^m \left| a_{ij} \right|. $$

Таким образом, верхняя граница $L_1$-нормы вектора $\mathbf{A} \mathbf{u}$ равняется максимуму из сумм абсолютных значений матрицы $\mathbf{A}$ в каждом столбце, то есть:

$$ \lVert \mathbf{A} \rVert_1 = \max_{\lVert \mathbf{u} \rVert_1 = 1} \lVert \mathbf{A} \mathbf{u} \rVert_1 = \max_j \sum_{i=1}^m \left| a_{ij} \right|. $$

Для $L_1$-нормы вектор $\mathbf{u}_* = \mathbf{e}_j$, где $\mathbf{e}_j$ представляет собой единичный вектор, все элементы которого, за исключением $j$-го, равны нулю. Индекс $j$ определяется номером столбца матрицы $\mathbf{A}$ с наибольшей суммой абсолютных значений элементов.

Рассмотрим нахождение $L_1$-нормы на следующем примере.

```{admonition} Пример
:class: exercise

Пусть матрица $\mathbf{A} \in \mathbb{R}^{3 \times 4}$ задана следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 1 & 3 & 14 & -2 \\ 5 & 2 & -7 & 0 \\ 4 & -2 & 7 & 3 \end{bmatrix}. $$

Необходимо найти $L_1$-норму данной матрицы.
```

````{dropdown} Решение
Зададим матрицу $\mathbf{A}$ в виде двумерного массива:

```python
A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])
```

С учетом полученного выше выражения посчитаем и выведем $L_1$-норму данной матрицы:

```python
print(np.abs(A).sum(axis=0).max())
```

```{glue:} norm5
```

Аналогичный результат может быть получен с использованием функции [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html):

```python
print(np.linalg.norm(A, ord=1))
```

```{glue:} norm6
```

Покажем, что данное значение достижимо при определенном векторе $\mathbf{u}_*$. Получим элементы данного вектора в виде одномерного массива:

```python
sums = np.abs(A).sum(axis=0)
col = np.argmax(sums)
u = np.zeros(shape=(4,))
u[col] = 1.
print(col, u)
```

```{glue:} norm7
```

Для создания одномерного массива из нулей была использована функция [`numpy.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html). Затем элемент этого массива с индексом `col` был заменен на единицу.

Покажем, что $\lVert \mathbf{A} \mathbf{u}_* \rVert_1 = \sum_{i=1}^m \left| \sum_{j=1}^n a_{ij} {u_*}_j \right|$ равняется найденной $L_1$-норме матрицы $\mathbf{A}$:

```python
print(np.abs(A.dot(u)).sum())
```

```{glue:} norm8
```
````

```{code-cell} python
:tags: [remove-cell]

A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])

glue('norm5', MultilineText(np.abs(A).sum(axis=0).max()))

glue('norm6', MultilineText(np.linalg.norm(A, ord=1)))

sums = np.abs(A).sum(axis=0)
col = np.argmax(sums)
u = np.zeros(shape=(4,))
u[col] = 1.

glue('norm7', MultilineText(col, u))

glue('norm8', MultilineText(np.abs(A.dot(u)).sum()))
```

<a id='matrix-norm-2'></a>
Рассмотрим $L_2$-норму. Обозначим $\mathbf{A} \mathbf{u} = \mathbf{v}$, где $\mathbf{A} \in \mathbb{C}^{m \times n}$, $\mathbf{u} \in \mathbb{C}^n$ и $\mathbf{v} \in \mathbb{C}^m$. Тогда:

$$ \lVert \mathbf{A} \mathbf{u} \rVert_2 = \lVert \mathbf{v} \rVert_2. $$

Запишем выражение для квадрата $L_2$-нормы вектора $\mathbf{v}$:

$$ \lVert \mathbf{v} \rVert_2^2 = \left( \sqrt{\sum_{i=1}^m \left| v_i \right|^2} \right)^2 = \sum_{i=1}^m \left| v_i \right|^2 = \mathbf{v}^\dagger \mathbf{v} = \left( \mathbf{A} \mathbf{u} \right)^\dagger \left( \mathbf{A} \mathbf{u} \right) = \mathbf{u}^\dagger \mathbf{A}^\dagger \mathbf{A} \mathbf{u} = \mathbf{u}^\dagger \mathbf{M} \mathbf{u}. $$

Матрица $\mathbf{M} = \mathbf{A}^\dagger \mathbf{A} \in \mathbb{C}^{n \times n}$ является [эрмитовой](LAB-2-Matrices.md#matrix-transp-hermitian) в соответствии с доказанным [ранее](LAB-2-Matrices.md#matrix-transp-cor-hermitian). Далее будет показано, что [любая эрмитова матрица унитарно диагонализируема](LAB-5-Eigenvalues-Eigenvectors.md#eigen-hermitian-theorem-spec), то есть существуют такие [унитарная](LAB-2-Matrices.md#matrix-unitary) матрица $\mathbf{P} \in \mathbb{C}^{n \times n}$ и диагональная матрица $\mathbf{D} \in \mathbb{R}^{n \times n}$, на главной диагонали которой расположены [действительные собственные значения](LAB-5-Eigenvalues-Eigenvectors.md#eigen-hermitian-lemma-real) $\lambda_1, \, \lambda_2, \, \ldots, \, \lambda_n$  матрицы $\mathbf{M}$, что справедливо следующее выражение:

$$ \mathbf{M} = \mathbf{P} \mathbf{D} \mathbf{P}^\dagger \implies \mathbf{D} = \mathbf{P}^\dagger \mathbf{M} \mathbf{P}. $$

На данный момент, без доказательства теоремы об унитарной диагонализируемости эрмитовых матриц, это соотношение следует воспринимать как есть. При необходимости, после изучения разделов о [линейных преобразованиях](LAB-4-LinearTransformations.md) и [собственных значениях и векторах матриц](LAB-5-Eigenvalues-Eigenvectors.md), можно вернуться к данному выводу выражения для $L_2$-нормы матрицы.

Кроме того, поскольку матрица $\mathbf{M} = \mathbf{A}^\dagger \mathbf{A}$, то она является [положительно полуопределенной](LAB-6-Definiteness.md), следовательно, все ее собственные значения неотрицательны.

Если матрица $\mathbf{P}$ является унитарной:

$$ \mathbf{P} = \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{v}_1 & \mathbf{v}_2 & \ldots & \mathbf{v}_n \\ \vert & \vert & & \vert \end{bmatrix}, $$

то ее векторы-столбцы $\mathbf{v}_i \in \mathbb{C}^n, \, i = 1 \, \ldots \, n,$ образуют ортонормированный [базис](LAB-4-LinearTransformations.md#lintran-basis), в котором любой вектор, в том числе $\mathbf{u}$, может быть представлен в виде суммы произведений координат этого вектора в данном базисе и базисных векторов. Иными словами, [любой вектор может быть разложен по данному базису](LAB-4-LinearTransformations.md#lintran-basis-theorem):

$$ \mathbf{u} = \sum_{i=1}^n c_i \mathbf{v}_i = \mathbf{P} \mathbf{c}, $$

где вектор $\mathbf{c} = \begin{bmatrix} c_1, \, c_2, \, \ldots, \, c_n \end{bmatrix}^\top$ представляет собой координаты вектора $\mathbf{u}$ в базисе из векторов-столбцов матрицы $\mathbf{P}$. Причем, поскольку этот базис, как и стандартный (канонический), в котором были заданы координаты вектора $\mathbf{u}$, является ортонормированным, то длина вектора $\mathbf{c}$ [также равняется](LAB-4-LinearTransformations.md#lintran-orthogonal-theorem) единице: $\mathbf{c}^\dagger \mathbf{c} = \sum_{i=1}^n \left| c_i \right|^2 = 1$.

Подставим равенство $\mathbf{u} = \mathbf{P} \mathbf{c}$ в полученное ранее выражение для квадрата $L_2$-нормы вектора $\mathbf{v}$:

$$ \lVert \mathbf{v} \rVert_2^2 = \mathbf{u}^\dagger \mathbf{M} \mathbf{u} = \left( \mathbf{P} \mathbf{c} \right)^\dagger \mathbf{M} \left( \mathbf{P} \mathbf{c} \right) = \mathbf{c}^\dagger \underbrace{\mathbf{P}^\dagger \mathbf{M} \mathbf{P}}_\mathbf{D} \mathbf{c} = \sum_{i=1}^n \lambda_i \left| c_i \right|^2 \leq \sum_{i=1}^n \lambda_1 \left| c_i \right|^2, $$

где $\lambda_1$ – наибольшее собственное значение матрицы $\mathbf{M}$. Поскольку $\sum_{i=1}^n \lambda_1 \left| c_i \right|^2 = \lambda_1 \sum_{i=1}^n \left| c_i \right|^2 = \lambda_1$, то верхняя граница $L_2$-нормы вектора $\mathbf{v} = \mathbf{A} \mathbf{u}$:

$$ \lVert \mathbf{A} \rVert_2 = \max_{\lVert \mathbf{u} \rVert_2} \lVert \mathbf{A} \mathbf{u} \rVert_2 = \sqrt{ \lambda_1 } = \sigma_1, $$

где $\sigma_1$ – наибольшее [сингулярное значение](LAB-8-ConditionNumber.md#cond-singular-def) матрицы $\mathbf{A}$.

Для $L_2$-нормы вектор $\mathbf{u}_* = \mathbf{v}_1$, то есть равен собственному вектору, соответствующему наибольшему собственному значению матрицы $\mathbf{M}$. Покажем, что в этом случае $L_2$-норма вектора $\mathbf{A} \mathbf{u}_*$ равняется наибольшему сингулярному значению матрицы $\mathbf{A}$. Для этого рассмотрим квадрат $L_2$-нормы вектора $\mathbf{A} \mathbf{u}_*$:

$$ \lVert \mathbf{A} \mathbf{u}_* \rVert_2^2 = \lVert \mathbf{A} \mathbf{v}_1 \rVert_2^2 = \left( \mathbf{A} \mathbf{v}_1 \right)^\dagger \left( \mathbf{A} \mathbf{v}_1 \right) = \mathbf{v}_1^\dagger \mathbf{A}^\dagger \mathbf{A} \mathbf{v}_1 = \mathbf{v}_1^\dagger \mathbf{M} \mathbf{v}_1 = \mathbf{v}_1^\dagger \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{v}_1^\dagger \mathbf{v}_1 = \lambda_1. $$

Тогда $L_2$-норма вектора $\mathbf{A} \mathbf{u}_*$:

$$ \lVert \mathbf{A} \mathbf{u}_* \rVert_2 = \sqrt{\lambda_1} = \sigma_1. $$

Рассмотрим нахождение $L_2$-нормы на следующем примере.

```{admonition} Пример
:class: exercise

Пусть матрица $\mathbf{A} \in \mathbb{R}^{3 \times 4}$ задана следующим образом:

$$ \mathbf{A} = \begin{bmatrix} 1 & 3 & 14 & -2 \\ 5 & 2 & -7 & 0 \\ 4 & -2 & 7 & 3 \end{bmatrix}. $$

Необходимо найти $L_2$-норму данной матрицы.
```

````{dropdown} Решение
Зададим матрицу $\mathbf{A}$ в виде двумерного массива:

```python
A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])
```

С учетом полученного выше выражения посчитаем и выведем $L_2$-норму данной матрицы (для расчета собственных значений матрицы можно использовать функцию [`numpy.linalg.eigvals`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html)):

```python
print(np.sqrt(np.linalg.eigvals(A.T @ A).max()))
```

```{glue:} norm9
```

Аналогичный результат может быть получен с использованием функции [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html):

```python
print(np.linalg.norm(A, ord=2))
```

```{glue:} norm10
```
````

```{code-cell} python
:tags: [remove-cell]

A = np.array([[1., 3., 14., -2.], [5., 2., -7., 0.], [4., -2., 7., 3.]])

glue('norm9', MultilineText(np.sqrt(np.linalg.eigvals(A.T @ A).max())))

glue('norm10', MultilineText(np.linalg.norm(A, ord=2)))
```

(lintran-transition)=
## Изменение базиса как линейное преобразование

```{admonition} Теорема
:class: theorem
Перевод вектора из одного базиса в другой может быть представлен в виде линейного преобразования.
```

```{admonition} Доказательство
:class: proof
Обозначим "старый" базис $\mathrm{E} = \begin{Bmatrix} \mathbf{e}_1, \, \mathbf{e}_2, \, \ldots, \, \mathbf{e}_n \end{Bmatrix}$. Поскольку $\mathrm{E}$ представляет собой базис, то любой вектор может быть представлен в виде линейной комбинации базисных векторов и его координат:

$$ \mathbf{b}_j = b_{1j} \mathbf{e}_1 + b_{2j} \mathbf{e}_2 + \ldots + b_{nj} \mathbf{e}_n = \sum_{i=1}^n b_{ij} \mathbf{e}_i, \; j = 1 \, \ldots \, n, $$

где $b_{1j}, \, b_{2j}, \, \ldots, \, b_{nj}, j = 1 \, \ldots \, n,$ представляют собой координаты вектора $\mathbf{b}_j, \, j = 1 \, \ldots \, n,$ в базисе $\mathrm{E}$.

Пусть матрица $\mathbf{B} \in \mathbb{C}^{n \times n}$ составлена из координат векторов $\mathbf{b}_j, \, j = 1 \, \ldots \, n$:

$$ \mathbf{B} = \begin{bmatrix} \vert & \vert & & \vert \\ \mathbf{b_1} & \mathbf{b_2} & \ldots & \mathbf{b_n} \\ \vert & \vert & & \vert \end{bmatrix}. $$

Создание "нового" базиса $\mathrm{B} = \begin{Bmatrix} \mathbf{b_1}, \, \mathbf{b_2}, \, \ldots, \, \mathbf{b_n} \end{Bmatrix}$ возможно тогда и только тогда, когда векторы $\mathbf{b}_j, \, j = 1 \, \ldots \, n,$ [линейно независимы](LAB-4-LinearTransformations.md#lintran-linindep), что равносильно тому, что [определитель](LAB-2-Matrices.md#matrix-det) матрицы $\mathbf{B}$ отличен от нуля, а сама матрица $\mathbf{B}$ [обратима](LAB-2-Matrices.md#matrix-inv). Тогда матрица $\mathbf{B}$ будет представлять собой *матрицу перехода* между базисами $\mathrm{E}$ и $\mathrm{B}$.

Пусть имеется произвольный вектор $\mathbf{z} \in \mathbb{C}^{n}$, координаты которого относительно базиса $\mathrm{E}$: $\begin{bmatrix} \mathbf{z} \end{bmatrix}_\mathrm{E} = \mathbf{x} = \begin{bmatrix} x_1, \, x_2 , \, \ldots, \, x_n \end{bmatrix}^\top_\mathrm{E}$ и относительно базиса $\mathrm{B}$: $\begin{bmatrix} \mathbf{z} \end{bmatrix}_\mathrm{B} = \mathbf{y} = \begin{bmatrix} y_1, \, y_2, \, \ldots, \, y_n \end{bmatrix}^\top_\mathrm{B}$.

При этом, поскольку вектор является одним и тем же, то, учитывая возможность разложения вектора по базису, справедливо следующее соотношение:

$$ \mathbf{z} = \sum_{i=1}^n x_i \mathbf{e}_i = \sum_{j=1}^n y_j \mathbf{b}_j. $$

Выполним следующие преобразования:

$$ \mathbf{z} = \sum_{j=1}^n y_j \mathbf{b}_j = \sum_{j=1}^n y_j \sum_{i=1}^n b_{ij} \mathbf{e}_i = \sum_{j=1}^n \sum_{i=1}^n b_{ij} y_j \mathbf{e}_i = \sum_{i=1}^n \left( \sum_{j=1}^n b_{ij} y_j \right) \mathbf{e}_i. $$

Поскольку $\mathbf{z} = \sum_{i=1}^n x_i \mathbf{e}_i$, то:

$$ x_i = \sum_{j=1}^n b_{ij} y_j, \; i = 1 \, \ldots \, n. $$

Таким образом, переход от базиса $\mathrm{B}$ к базису $\mathrm{E}$ может быть записан в виде матричного произведения:

$$ \mathbf{x} = \mathbf{B} \mathbf{y} \implies \begin{bmatrix} \mathbf{z} \end{bmatrix}_\mathrm{E} = \mathbf{B} \begin{bmatrix} \mathbf{z} \end{bmatrix}_\mathrm{B}. $$

В свою очередь, переход от базиса $\mathrm{E}$ к базису $\mathrm{B}$:

$$ \mathbf{y} = \mathbf{B}^{-1} \mathbf{x} \implies \begin{bmatrix} \mathbf{z} \end{bmatrix}_\mathrm{B} = \mathbf{B}^{-1} \begin{bmatrix} \mathbf{z} \end{bmatrix}_\mathrm{E}. $$
```

Таким образом, переход от одного базиса к другому является примером линейного преобразования – произведения квадратной матрицы и вектора. При этом такую матрицу принято называть ***матрицей перехода*** или ***матрицей замены***.

Рассмотрим пример.

```{admonition} Пример
:class: exercise
Пусть базис $\mathrm{B}$ задан векторами $\mathbf{b_1} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ и $\mathbf{b_2} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$:

$$ \mathrm{B} = \begin{Bmatrix} \mathbf{b_1}, \mathbf{b_2} \end{Bmatrix}. $$

Необходимо определить координаты вектора $\mathbf{x} = \begin{bmatrix} 8 \\ 7 \end{bmatrix}$ в базисе $\mathrm{B}$.
```

````{dropdown} Решение
Составим матрицу перехода между стандартным базисом и базизом $\mathrm{B}$:

$$ \mathbf{B} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}. $$

С учетом доказанных выше соотношений, координаты вектора $\mathbf{x}$ в базисе $\mathrm{B}$:

$$ \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathrm{B} = \mathbf{B}^{-1} \mathbf{x}. $$

Вычислим координаты вектора $\mathbf{x}$ в базисе $\mathrm{B}$. Для этого будем использовать библиотеку [numpy](https://numpy.org):

```python
import numpy as np
```

Создадим вектор $\mathbf{x}$ в виде одномерного массива и матрицу $\mathbf{B}$, составленную из координат базисных векторов, в виде двумерного массива:

```python
x = np.array([8., 7.])
B = np.array([[2., 1.], [1., 2.]])
```

Найдем обратную матрицу матрице $\mathbf{B}$:

```python
B_inv = np.linalg.inv(B)
```

Определим координаты вектора $\mathbf{x}$ в базисе $\mathrm{B}$:

```python
x_B = B_inv.dot(x)
print(x_B)
```

```{glue:} glued_out1
```

Проиллюстрируем данное решение следующим рисунком:

```{glue:} glued_fig1
```
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

x = np.array([8., 7.])
B = np.array([[2., 1.], [1., 2.]])
B_inv = np.linalg.inv(B)
x_B = B_inv.dot(x)

glue('glued_out1', MultilineText(x_B))

fig1, ax1 = plt.subplots(figsize=(6., 4.), tight_layout=True)
fig1.canvas.header_visible = False

x0 = y0 = [0., 0., 0.]

U = [[2., 1., 8.]]
V = [[1., 2., 7.]]

ax1.plot([0., 8.], [0., 4.], color='k', ls='--', zorder=1)
ax1.plot([1., 2., 3.], [2., 4., 6.], lw=0., marker='o', color='k', ms=4.)
ax1.plot([0., 4.], [0., 8.], color='k', ls='--', zorder=1)
ax1.plot([2., 4., 6.], [1., 2., 3.], lw=0., marker='o', color='k', ms=4.)
ax1.plot([6., 8.], [3., 7.], color='r', ls='--', zorder=1)
ax1.plot([2., 8.], [4., 7.], color='r', ls='--', zorder=1)
ax1.quiver(x0, y0, U, V, scale=1, angles='xy', scale_units='xy', color=['k', 'k', 'r'], zorder=2)

ax1.text(2., 0.5, r'$\mathbf{v_1}$')
ax1.text(4., 1.5, r'$2 \cdot \mathbf{v_1}$')
ax1.text(6., 2.5, r'$3 \cdot \mathbf{v_1}$')
ax1.text(0.5, 2, r'$\mathbf{v_2}$')
ax1.text(1., 4., r'$2 \cdot \mathbf{v_2}$')
ax1.text(2., 6., r'$3 \cdot \mathbf{v_2}$')
ax1.text(8., 7., r'$\mathbf{x}$', c='r')

ax1.set_xlim(-1., 10.)
ax1.set_ylim(-1., 10.)
ax1.set_axisbelow(True)
ax1.grid()
ax1.set_xticks(range(0, 10, 1))
ax1.set_yticks(range(0, 10, 1))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

glue('glued_fig1', fig1)
```

Итак, выше было рассмотрено изменение базиса, которое может быть выражено в виде линейного преобразования. Рассмотрим более сложную задачу о представлении линейного преобразования в различных базисах.

```{admonition} Лемма
:class: caution
Линейное преобразование в новом базисе также является линейным преобразованием.
```

```{admonition} Доказательство
:class: proof
Необходимо доказать, что линейное преобразование, выполненное в другом базисе, также является линейным. Поскольку переход от одного базиса к другому представляет собой линейное преобразование, то рассматриваемая задача сводится к доказательству линейности преобразования, представляющего собой *композицию линейных преобразований*. То есть если $\mathcal{T} \left( \cdot \right)$ и $\mathcal{S} \left( \cdot \right)$ являются линейными преобразованиями (для которых выполняются *свойства линейности*), то необходимо доказать, что их композиция $\mathcal{T} \left( \mathcal{S} \left( \cdot \right) \right)$ также является линейным преобразованием.

Докажем первое свойство линейности.

$$ \mathcal{T} \left( \mathcal{S} \left( \mathbf{a} + \mathbf{b} \right) \right) = \mathcal{T} \left( \mathcal{S} \left( \mathbf{a} \right) + \mathcal{S} \left( \mathbf{b} \right) \right) = \mathcal{T} \left( \mathcal{S} \left( \mathbf{a} \right) \right) + \mathcal{T} \left( \mathcal{S} \left( \mathbf{b} \right) \right). $$

Докажем второе свойство линейности.

$$ \mathcal{T} \left( \mathcal{S} \left( \lambda \cdot \mathbf{a} \right) \right) = \mathcal{T} \left( \lambda \cdot \mathcal{S} \left( \mathbf{a} \right) \right) = \lambda \cdot \mathcal{T} \left( \mathcal{S} \left( \mathbf{a} \right) \right). $$

Таким образом, доказав корректность свойств линейности, можно сделать вывод о том, что композиция линейных преобразований также является линейным преобразованием. Следовательно, линейное преобразование, выполненное в новом базисе также будет являться линейным преобразованием.
```

```{admonition} Теорема
:class: theorem
Матрицы одного и того же линейного преобразования в разных базисах ***подобны***.

Иными словами, пусть матрица $\mathbf{B} \in \mathbb{C}^{n \times n}$ составлена из базисных векторов базиса $\mathrm{B}$ и представляет собой матрицу перехода между базисами $\mathrm{B}$ и $\mathrm{E}$. То есть координаты произвольного вектора $\mathbf{v} \in \mathbb{C}^n$ в этих базисах соотносятся $\begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{E} = \mathbf{B} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B}$. Пусть в базисе $\mathrm{E}$ линейное преобразование $\mathcal{T} \left( \cdot \right)$ задано матрицей $\mathbf{A} \in \mathbb{C}^{n \times n}$. Тогда в базисе $\mathrm{B}$ это же линейное преобразование будет представлено матрицей $\mathbf{D} = \mathbf{B}^{-1} \mathbf{A} \mathbf{B}$, такой что $\begin{bmatrix} \mathbf{A} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{D} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B}$.
```

```{admonition} Доказательство
:class: proof
С учетом того, что композиция линейных преобразований также является линейным преобразованием, можно сделать вывод о том, что линейное преобразование $\mathcal{T} \left( \cdot \right)$, выполненное в базисе $\mathrm{B}$, можно представить в виде произведения матрицы линейного преобразования в базисе $\mathrm{B}$ и вектора, координаты которого соответствуют базису $\mathrm{B}$:

$$ \begin{bmatrix} \mathcal{T} \left( \mathbf{v} \right) \end{bmatrix}_\mathrm{B} = \begin{bmatrix} \mathbf{A} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{D} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B}. $$

Нашей задачей является получение выражения для матрицы линейного преобразования $\mathbf{D}$ в базисе $\mathrm{B}$ через матрицу этого линейного преобразования в стандартном базисе $\mathbf{A}$ и матрицу перехода $\mathbf{B}$ между базисами $\mathrm{B}$ и $\mathrm{E}$. Здесь и далее под векторами и матрицами, записанными без указания базиса, будет пониматься их определенность по базису $\mathrm{E}$.

Переход от базиса $\mathrm{E}$ к базису $\mathrm{B}$ может быть записан в виде матричного произведения:

$$ \begin{bmatrix} \mathbf{A} \mathbf{v} \end{bmatrix}_\mathbf{B} = \mathbf{B}^{-1} \mathbf{A} \mathbf{v}. $$

Подставляя это равенство в выражение выше, получим:

$$ \begin{bmatrix} \mathcal{T} \left( \mathbf{v} \right) \end{bmatrix}_\mathrm{B} = \begin{bmatrix} \mathbf{A} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{D} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{B}^{-1} \mathbf{A} \mathbf{v}. $$

В свою очередь, координаты вектора $\mathbf{v}$ в базисе $\mathrm{E}$ могут быть представлены следующим образом:

$$ \mathbf{v} = \mathbf{B} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B}. $$

Тогда:

$$ \begin{bmatrix} \mathcal{T} \left( \mathbf{v} \right) \end{bmatrix}_\mathrm{B} = \begin{bmatrix} \mathbf{A} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{D} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B} = \mathbf{B}^{-1} \mathbf{A} \mathbf{v} = \mathbf{B}^{-1} \mathbf{A} \mathbf{B} \begin{bmatrix} \mathbf{v} \end{bmatrix}_\mathrm{B}. $$

Итак, матрица $\mathbf{D}$ линейного преобразования $\mathcal{T} \left( \cdot \right)$ по отношению к базису $\mathrm{B}$:

$$ \mathbf{D} = \mathbf{B}^{-1} \mathbf{A} \mathbf{B}, \; \mathbf{A} = \mathbf{B} \mathbf{D} \mathbf{B}^{-1}, $$

где: $\mathbf{B}$ – матрица перехода между базисами $\mathrm{B}$ и $\mathrm{E}$, $\mathbf{A}$ – матрица линейного преобразования вектора $\mathbf{v}$ в базисе $\mathrm{E}$. То есть линейный оператор (матрицу линейного преобразования) можно перевести в новый базис с использованием *матрицы перехода* между этими базисами.
```

Доказанная теорема на практике может быть полезна, когда необходимо изменить "вид" матрицы линейного преобразования, например, сделать ее диагональной в некотором базисе для упрощения вычислений.

(lintran-rotation)=
## Вращение векторов

В качестве примера ортогонального линейного преобразования рассмотрим *вращение* вектора.

````{margin}
```{admonition} Дополнительно
:class: note
При рассмотрении задачи о вращении вектора важно обозначить используемую [систему координат](https://en.wikipedia.org/wiki/Cartesian_coordinate_system) (правую или левую), от которой зависит положительное направление вращения вектора. Здесь и далее будет рассматриваться правая система координат. Интересно отметить, что популярные графические интерфейсы, например, OpenGL и DirectX [используют разные системы координат](https://softwareengineering.stackexchange.com/questions/17519/why-does-directx-use-a-left-handed-coordinate-system): правую и левую соответственно.
```
````

Начнем с вращения вектора в плоскости $Oxy$, то есть вокруг оси $Oz$. Обозначим вектором $\mathbf{v}$ начальное положение вектора, а вектором $\mathbf{u}$ – конечное, как показано на следующем рисунке.

```{code-cell} python
:tags: [remove-cell]

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, Arc
from mpl_toolkits.mplot3d import proj3d

def rotate(v, a, b, g):
  sing = np.sin(g)
  cosg = np.cos(g)
  Rg = np.array([
    [cosg, -sing, 0.],
    [sing, cosg, 0.],
    [0., 0., 1.],
  ])
  sinb = np.sin(b)
  cosb = np.cos(b)
  Rb = np.array([
    [cosb, 0., sinb],
    [0., 1., 0.],
    [-sinb, 0., cosb],
  ])
  sina = np.sin(a)
  cosa = np.cos(a)
  Ra = np.array([
    [1., 0., 0.],
    [0., cosa, -sina],
    [0., sina, cosa],
  ])
  return Rg.dot(Rb).dot(Ra).dot(v)

class Arrow3D(FancyArrowPatch):
  def __init__(self, xe, ye, ze, xb=0., yb=0., zb=0., c='k', *args, **kwargs):
    FancyArrowPatch.__init__(
      self, (0, 0), (0, 0), *args, mutation_scale=10, arrowstyle='-|>',
      shrinkA=0., shrinkB=0., color=c, **kwargs
    )
    self._verts3d = (xb, xe), (yb, ye), (zb, ze)
    pass

  def do_3d_projection(self, renderer=None):
    xs, ys, zs = proj3d.proj_transform(*self._verts3d, self.axes.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    return np.min(zs)
```

```{code-cell} python
:tags: [remove-input]

fig1 = plt.figure(figsize=(8., 4.), tight_layout=True)
fig1.canvas.header_visible = False
ax11 = fig1.add_subplot(121)
ax12 = fig1.add_subplot(122, projection='3d')

w = np.array([1., 0., 0.])

da0, db0, dg0 = 0., 0., 30.

v = rotate(w, da0 * np.pi / 180., db0 * np.pi / 180., dg0 * np.pi / 180.)

arrv = Arrow3D(*v, c='b')
ax12.add_artist(arrv)
ax12.text(*(1.1*v), r'$\mathbf{v}$', color='b')

da, db, dg = 0., 0., 40.

a = da * np.pi / 180.
b = db * np.pi / 180.
g = dg * np.pi / 180.

u = rotate(v, a, b, g)

arru = Arrow3D(*u, c='r')
ax12.add_artist(arru)
ax12.text(*(1.02*u), r'$\mathbf{u}$', color='r')

ax12.set_xlim([0., 1.])
ax12.set_ylim([0., 1.])
ax12.set_zlim([0., 1.])
ax12.text(1.05, 0., 0., 'x', color='k')
ax12.text(0., 1., 0., 'y', color='k')
ax12.text(0., 0., 1., 'z', color='k')
ax12.text(0., 0., 0., 'O', color='k')
ax12.add_artist(Arrow3D(1., 0., 0., c='k'))
ax12.add_artist(Arrow3D(0., 1., 0., c='k'))
ax12.add_artist(Arrow3D(0., 0., 1., c='k'))
ax12.view_init(elev=30., azim=55.)
ax12.set_axis_off()

z = (0., 0.)
idx = np.array([0, 1])

ax11.arrow(*z, *v[idx], length_includes_head=True, head_width=0.02, color='b')
ax11.arrow(*z, *u[idx], length_includes_head=True, head_width=0.02, color='r')
ax11.text(*v[idx], r'$\mathbf{v}$', color='b')
ax11.text(*u[idx], r'$\mathbf{u}$', color='r')

pg0 = np.array([0.28, 0.07, 0.])

arcv = Arc((0., 0.), 0.5, 0.5, angle=0., theta1=0., theta2=dg0, color='b')
ax11.add_patch(arcv)
ax11.text(*pg0[idx], r'$\gamma_0$', color='b')

pg1 = rotate(pg0, a, b, g)

arcu1 = Arc(z, 0.48, 0.48, angle=0., theta1=dg0, theta2=dg0+dg, color='r')
arcu2 = Arc(z, 0.52, 0.52, angle=0., theta1=dg0, theta2=dg0+dg, color='r')
ax11.add_patch(arcu1)
ax11.add_patch(arcu2)
ax11.text(*pg1[idx], r'$\gamma$', color='r')

ax11.set_xlim(-0.05, 1.05)
ax11.set_ylim(-0.05, 1.05)
ax11.grid(False)
ax11.set_axis_off()
ax11.arrow(*z, 1., 0., length_includes_head=True, head_width=0.02, color='k')
ax11.arrow(*z, 0., 1., length_includes_head=True, head_width=0.02, color='k')
ax11.text(1., 0., 'x', color='k')
ax11.text(0., 1., 'y', color='k')

fig1.tight_layout()
```

Важно отметить, что в процессе вращения вектора не происходит изменение его длины. Следовательно, если обозначить $v$ [длину](LAB-1-Vectors.md#vector-length) вектора $\mathbf{v}$ и $u$ – длину вектора $\mathbf{u}$, то в результате вращения будет справедливо равенство $v = u$. Кроме того, в процессе вращения вектора вокруг оси $Oz$ не происходит изменение его соответствующей координаты: $v_z = u_z$. Таким образом, зная длину вектора, можно записать выражения для его координат с использованием [направляющих косинусов](LAB-1-Vectors.md#vector-angles):

$$ \begin{cases}
v_x = v \cdot \cos \gamma_0, \\
v_y = v \cdot \sin \gamma_0, \\
v_z = v_z.
\end{cases} $$

Аналогично могут быть записаны координаты для вектора $\mathbf{u}$:

$$ \begin{cases}
u_x = u \cdot \cos \left( \gamma + \gamma_0 \right), \\
u_y = u \cdot \sin \left( \gamma + \gamma_0 \right), \\
u_z = u_z.
\end{cases} $$

Преобразуем данные выражения, применив [формулы](https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities) для разложения синуса и косинуса суммы углов:

$$ \begin{cases}
u_x = u \cdot \cos \left( \gamma + \gamma_0 \right) = v \cdot \cos \gamma_0 \cdot \cos \gamma - v \cdot \sin \gamma_0 \cdot \sin \gamma = v_x \cdot \cos \gamma - v_y \cdot \sin \gamma , \\
u_y = u \cdot \sin \left( \gamma + \gamma_0 \right) = v \cdot \sin \gamma_0 \cdot \cos \gamma + v \cdot \cos \gamma_0 \cdot \sin \gamma = v_y \cdot \cos \gamma + v_x \cdot \sin \gamma , \\
u_z = v_z.
\end{cases} $$

Данную систему можно записать в матричном виде:

$$ \begin{bmatrix} u_x \\ u_y \\ u_z \end{bmatrix} = \begin{bmatrix} \cos \gamma & - \sin \gamma & 0 \\ \sin \gamma & \cos \gamma & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}. $$

Или:

$$ \mathbf{u} = \mathbf{R}_z \mathbf{v}, $$

где матрица $\mathbf{R}_z \in \mathbb{R}^{3 \times 3}$ представляет собой матрицу вращения вектора вокруг оси $Oz$.

Таким образом, вращение вектора может быть представлено в виде линейного преобразования исходного вектора.

Рассмотрим вращение вектора вокруг оси $Oy$, при котором остается без изменения соответствующая координата: $v_y = u_y$.

```{code-cell} python
:tags: [remove-input]

fig2 = plt.figure(figsize=(8., 4.), tight_layout=True)
fig2.canvas.header_visible = False
ax21 = fig2.add_subplot(121)
ax22 = fig2.add_subplot(122, projection='3d')

w = np.array([0., 0., 1.])

da0, db0, dg0 = 0., 30., 0.

v = rotate(w, da0 * np.pi / 180., db0 * np.pi / 180., dg0 * np.pi / 180.)

arrv = Arrow3D(*v, c='b')
ax22.add_artist(arrv)
ax22.text(*(1.1*v), r'$\mathbf{v}$', color='b')

da, db, dg = 0., 40., 0.

a = da * np.pi / 180.
b = db * np.pi / 180.
g = dg * np.pi / 180.

u = rotate(v, a, b, g)

arru = Arrow3D(*u, c='r')
ax22.add_artist(arru)
ax22.text(*(1.02*u), r'$\mathbf{u}$', color='r')

ax22.set_xlim([0., 1.])
ax22.set_ylim([0., 1.])
ax22.set_zlim([0., 1.])
ax22.text(1.05, 0., 0., 'x', color='k')
ax22.text(0., 1., 0., 'y', color='k')
ax22.text(0., 0., 1., 'z', color='k')
ax22.text(0., 0., 0., 'O', color='k')
ax22.add_artist(Arrow3D(1., 0., 0., c='k'))
ax22.add_artist(Arrow3D(0., 1., 0., c='k'))
ax22.add_artist(Arrow3D(0., 0., 1., c='k'))
ax22.view_init(elev=30., azim=55.)
ax22.set_axis_off()

z = (0., 0.)
idx = np.array([0, 2])

ax21.arrow(*z, *v[idx], length_includes_head=True, head_width=0.02, color='b')
ax21.arrow(*z, *u[idx], length_includes_head=True, head_width=0.02, color='r')
ax21.text(*v[idx], r'$\mathbf{v}$', color='b')
ax21.text(*u[idx], r'$\mathbf{u}$', color='r')

pg0 = np.array([0.09, 0., 0.28])

arcv = Arc(z, 0.5, 0.5, angle=90.-db0, theta1=0., theta2=db0, color='b')
ax21.add_patch(arcv)
ax21.text(*pg0[idx], r'$\beta_0$', color='b')

pg1 = rotate(pg0, a, b, g)

arcu1 = Arc((0., 0.), 0.48, 0.48, angle=90.-db0-db, theta1=0., theta2=db, color='r')
arcu2 = Arc((0., 0.), 0.52, 0.52, angle=90.-db0-db, theta1=0., theta2=db, color='r')
ax21.add_patch(arcu1)
ax21.add_patch(arcu2)
ax21.text(*pg1[idx], r'$\beta$', color='r')

ax21.set_xlim(-0.05, 1.05)
ax21.set_ylim(-0.05, 1.05)
ax21.grid(False)
ax21.set_axis_off()
ax21.arrow(*z, 1., 0., length_includes_head=True, head_width=0.02, color='k')
ax21.arrow(*z, 0., 1., length_includes_head=True, head_width=0.02, color='k')
ax21.text(1., 0., 'x', color='k')
ax21.text(0., 1., 'z', color='k')
ax21.invert_xaxis()

fig2.tight_layout()
```

Запишем координаты исходного вектора $\mathbf{v}$:

$$ \begin{cases}
v_x = v \cdot \sin \beta_0, \\
v_y = v_y, \\
v_z = v \cdot \cos \beta_0.
\end{cases} $$

Аналогично получим координаты для вектора $\mathbf{u}$:

$$ \begin{cases}
u_x = u \cdot \sin \left( \beta + \beta_0 \right), \\
u_y = u_y, \\
u_z = u \cdot \cos \left( \beta + \beta_0 \right).
\end{cases} $$

Применяя [формулы](https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities) для разложения синуса и косинуса суммы углов, преобразуем данные выражения следующим образом:

$$ \begin{cases}
u_x = u \cdot \sin \left( \beta + \beta_0 \right) = v \cdot \sin \beta_0 \cdot \cos \beta + v \cdot \cos \beta_0 \cdot \sin \beta = v_x \cdot \cos \beta + v_z \cdot \sin \beta , \\
u_y = v_y, \\
u_z = u \cdot \cos \left( \beta + \beta_0 \right) = v \cdot \cos \beta_0 \cdot \cos \beta - v \cdot \sin \beta_0 \cdot \sin \beta = v_z \cdot \cos \beta - v_x \cdot \sin \beta.
\end{cases} $$

Данную систему можно записать в матричном виде:

$$ \begin{bmatrix} u_x \\ u_y \\ u_z \end{bmatrix} = \begin{bmatrix} \cos \beta & 0 & \sin \beta \\ 0 & 1 & 0 \\ -\sin \beta & 0 & \cos \beta \end{bmatrix} \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}. $$

Или:

$$ \mathbf{u} = \mathbf{R}_y \mathbf{v}, $$

где матрица $\mathbf{R}_y \in \mathbb{R}^{3 \times 3}$ представляет собой матрицу вращения вектора вокруг оси $Oy$.

Рассмотрим вращение вокруг оси $Ox$, при котором остается без изменения соответствующая координата: $v_x = u_x$.

```{code-cell} python
:tags: [remove-input]

fig3 = plt.figure(figsize=(8., 4.), tight_layout=True)
fig3.canvas.header_visible = False
ax31 = fig3.add_subplot(121)
ax32 = fig3.add_subplot(122, projection='3d')

w = np.array([0., 1., 0.])

da0, db0, dg0 = 30., 0., 0.

v = rotate(w, da0 * np.pi / 180., db0 * np.pi / 180., dg0 * np.pi / 180.)

arrv = Arrow3D(*v, c='b')
ax32.add_artist(arrv)
ax32.text(*v, r'$\mathbf{v}$', color='b')

da, db, dg = 40., 0., 0.

a = da * np.pi / 180.
b = db * np.pi / 180.
g = dg * np.pi / 180.

u = rotate(v, a, b, g)

arru = Arrow3D(*u, c='r')
ax32.add_artist(arru)
ax32.text(*(1.02*u), r'$\mathbf{u}$', color='r')

ax32.set_xlim([0., 1.])
ax32.set_ylim([0., 1.])
ax32.set_zlim([0., 1.])
ax32.text(1.1, 0., 0., 'x', color='k')
ax32.text(0., 1., 0., 'y', color='k')
ax32.text(0., 0., 1., 'z', color='k')
ax32.text(0., -0.1, 0., 'O', color='k')
ax32.add_artist(Arrow3D(1., 0., 0., c='k'))
ax32.add_artist(Arrow3D(0., 1., 0., c='k'))
ax32.add_artist(Arrow3D(0., 0., 1., c='k'))
ax32.view_init(elev=30., azim=25.)
ax32.set_axis_off()

z = (0., 0.)
idx = np.array([1, 2])

ax31.arrow(*z, *v[idx], length_includes_head=True, head_width=0.02, color='b')
ax31.arrow(*z, *u[idx], length_includes_head=True, head_width=0.02, color='r')
ax31.text(*v[idx], r'$\mathbf{v}$', color='b')
ax31.text(*u[idx], r'$\mathbf{u}$', color='r')

pg0 = np.array([0., 0.28, 0.07])

arcv = Arc(z, 0.5, 0.5, angle=0., theta1=0., theta2=da0, color='b')
ax31.add_patch(arcv)
ax31.text(*pg0[idx], r'$\alpha_0$', color='b')

pg1 = rotate(pg0, a, b, g)

arcu1 = Arc((0., 0.), 0.48, 0.48, angle=0., theta1=da0, theta2=da0+da, color='r')
arcu2 = Arc((0., 0.), 0.52, 0.52, angle=0., theta1=da0, theta2=da0+da, color='r')
ax31.add_patch(arcu1)
ax31.add_patch(arcu2)
ax31.text(*pg1[idx], r'$\alpha$', color='r')

ax31.set_xlim(-0.05, 1.05)
ax31.set_ylim(-0.05, 1.05)
ax31.grid(False)
ax31.set_axis_off()
ax31.arrow(*z, 1., 0., length_includes_head=True, head_width=0.02, color='k')
ax31.arrow(*z, 0., 1., length_includes_head=True, head_width=0.02, color='k')
ax31.text(1., 0., 'y', color='k')
ax31.text(0., 1., 'z', color='k')

fig3.tight_layout()
```

Запишем координаты исходного вектора $\mathbf{v}$:

$$ \begin{cases}
v_x = v_x, \\
v_y = v \cdot \cos \alpha_0, \\
v_z = v \cdot \sin \alpha_0.
\end{cases} $$

Аналогично получим координаты для вектора $\mathbf{u}$:

$$ \begin{cases}
u_x = u_x, \\
u_y = u \cdot \cos \left( \alpha + \alpha_0 \right), \\
u_z = u \cdot \sin \left( \alpha + \alpha_0 \right).
\end{cases} $$

Применяя [формулы](https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities) для разложения синуса и косинуса суммы углов, преобразуем данные выражения следующим образом:

$$ \begin{cases}
u_x = v_x \\
u_y = u \cdot \cos \left( \alpha + \alpha_0 \right) = v \cdot \cos \alpha_0 \cdot \cos \alpha - v \cdot \sin \alpha_0 \cdot \sin \alpha = v_y \cdot \cos \alpha - v_z \cdot \sin \alpha, \\
u_z = u \cdot \sin \left( \alpha + \alpha_0 \right) = v \cdot \sin \alpha_0 \cdot \cos \alpha + v \cdot \cos \alpha_0 \cdot \sin \alpha = v_z \cdot \cos \alpha + v_y \cdot \sin \alpha .
\end{cases} $$

Данную систему можно записать в матричном виде:

$$ \begin{bmatrix} u_x \\ u_y \\ u_z \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha \\ 0 & \sin \alpha & \cos \alpha \end{bmatrix} \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}. $$

Или:

$$ \mathbf{u} = \mathbf{R}_x \mathbf{v}, $$

где матрица $\mathbf{R}_x \in \mathbb{R}^{3 \times 3}$ представляет собой матрицу вращения вектора вокруг оси $Ox$.

В общем виде, когда необходимо повернуть вектор вокруг нескольких осей, матрица линейного преобразования выражается в виде произведения трех матриц:

$$ \mathbf{u} = \mathbf{R} \mathbf{v} = \mathbf{R}_z \mathbf{R}_y \mathbf{R}_x \mathbf{v}. $$

Выражения для элементов матрицы вращения $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ можно получить аналитически:

$$ \begin{align}
\mathbf{R}
&= \begin{bmatrix} \cos \gamma & - \sin \gamma & 0 \\ \sin \gamma & \cos \gamma & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} \cos \beta & 0 & \sin \beta \\ 0 & 1 & 0 \\ -\sin \beta & 0 & \cos \beta \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha \\ 0 & \sin \alpha & \cos \alpha \end{bmatrix} \\
&= \begin{bmatrix}
\cos \beta \cos \gamma & \sin \alpha \sin \beta \cos \gamma - \cos \alpha \sin \gamma & \cos \alpha \sin \beta \cos \gamma + \sin \alpha \sin \gamma \\
\cos \beta \sin \gamma & \sin \alpha \sin \beta \sin \gamma + \cos \alpha \cos \gamma & \cos \alpha \sin \beta \sin \gamma - \sin \alpha \cos \gamma \\
-\sin \beta & \sin \alpha \cos \beta & \cos \alpha \cos \beta
\end{bmatrix}
\end{align} $$

```{code-cell} python
from sympy import symbols, Matrix, cos, sin

a = symbols('alpha')
b = symbols('beta')
g = symbols('gamma')

Rz = Matrix([
  [cos(g), -sin(g), 0],
  [sin(g), cos(g), 0],
  [0, 0, 1],
])

Ry = Matrix([
  [cos(b), 0, sin(b)],
  [0, 1, 0],
  [-sin(b), 0, cos(b)],
])

Rx = Matrix([
  [1, 0, 0],
  [0, cos(a), -sin(a)],
  [0, sin(a), cos(a)],
])

Rz @ Ry @ Rx
```

Представленное выше вращение, когда вращается сам вектор, называется *активным*. Существует, однако, и *пассивное* вращение вектора, когда происходит вращение базиса. Аналитические выражения матриц линейного преобразования для такого вида вращения могут быть получены аналогичным образом.

(lintran-hauseholder)=
## Преобразование Хаусхолдера

Другим примером ортогонального линейного преобразования является преобразование Хаусхолдера, решающего задачу о зеркальном отражении вектора.

Рассмотрим следующую задачу. Пусть имеется вектор $\mathbf{v} \in \mathbb{R}^n$, координаты которого представлены в базисе $\mathrm{B} = \begin{Bmatrix} \mathbf{e}_1, \, \mathbf{e}_2, \, \ldots, \, \mathbf{e}_n \end{Bmatrix}$. Необходимо определить матрицу линейного преобразования $\mathbf{H}_1 \in \mathbb{R}^n$, позволяющую представить вектор $\mathbf{v}$ в виде произведения единичного базисного вектора $\mathbf{e}_1$ и скалярной величины $\lambda = \lVert \mathbf{v} \rVert_2$, то есть:

$$ \mathbf{H}_1 \mathbf{v} = \lambda \mathbf{e}_1. $$

Сопроводим данную задачу следующим рисунком:

```{code-cell} python
:tags: [remove-input]

fig4, ax4 = plt.subplots(figsize=(5., 5.), tight_layout=True)
fig4.canvas.header_visible = False

ax4.arrow(0., 0., 0.6, 0.8, length_includes_head=True, head_width=0.02, color='b', zorder=4)
ax4.text(0.3, 0.45, r'$\mathbf{v}$', color='b')

ax4.arrow(0., 0., 1., 0., length_includes_head=True, head_width=0.02, color='g', zorder=4)
ax4.text(0.5, -0.04, r'$\mathbf{w}$', color='g')

ax4.arrow(0., 0., 0.1, 0., lw=2., length_includes_head=True, head_width=0.02, color='g', zorder=3)
ax4.text(0.1, -0.04, r'$\mathbf{e}_1$', color='g')

ax4.arrow(0.6, 0.8, 0.2, -0.4, length_includes_head=True, head_width=0.02, color='olive', zorder=4)
ax4.text(0.7, 0.6, r'$\mathbf{x} = -\left( \mathbf{v}^\top \mathbf{u} \right) \mathbf{u}$', color='olive')

ax4.arrow(0.8, 0.4, 0.2, -0.4, length_includes_head=True, head_width=0.02, color='olive', zorder=4)
ax4.text(0.9, 0.2, r'$\mathbf{x} = -\left( \mathbf{v}^\top \mathbf{u} \right) \mathbf{u}$', color='olive')

ax4.plot([-0.2, 1.2], [-0.1, 0.6], lw=1., ls='--', c='r', zorder=2)
ax4.plot([0.1, -0.2], [-0.2, 0.4], lw=1., ls='--', c='r', zorder=2)

ax4.arrow(0., 0., -0.3 / np.sqrt(0.45) * 0.1, 0.6 / np.sqrt(0.45) * 0.1, lw=2., length_includes_head=True, head_width=0.02, color='r', zorder=3)
ax4.text(-0.1, 0.08, r'$\mathbf{u}$', color='r')

alpha = np.arctan(0.5)

R = np.array([
  [np.cos(alpha), -np.sin(alpha)],
  [np.sin(alpha), np.cos(alpha)],
])

x1 = R.dot(np.array([0., 0.05]))
x2 = R.dot(np.array([-0.05, 0.]))
x3 = R.dot(np.array([-0.05, 0.05]))

ax4.plot([0., x1[0]], [0., x1[1]], lw=0.5, c='r', zorder=2)
ax4.plot([x2[0], x3[0]], [x2[1], x3[1]], lw=0.5, c='r', zorder=2)
ax4.plot([x3[0], x1[0]], [x3[1], x1[1]], lw=0.5, c='r', zorder=2)

h = np.array([0.6 + 0.2, 0.8 - 0.4])
a1 = x1 + h
a2 = x2 + h
a3 = x3 + h

ax4.plot([h[0], a1[0]], [h[1], a1[1]], lw=0.5, c='r', zorder=2)
ax4.plot([a2[0], a3[0]], [a2[1], a3[1]], lw=0.5, c='r', zorder=2)
ax4.plot([a3[0], a1[0]], [a3[1], a1[1]], lw=0.5, c='r', zorder=2)

ax4.plot([0.], [0.], 'o', lw=0., ms=4., c='k', zorder=5)
ax4.text(-0.04, -0.05, 'A')

ax4.plot([0.6], [0.8], 'o', lw=0., ms=4., c='k', zorder=5)
ax4.text(0.62, 0.8, 'B')

ax4.plot([1.], [0.], 'o', lw=0., ms=4., c='k', zorder=5)
ax4.text(1.01, 0.01, 'C')

ax4.plot([0.8], [0.4], 'o', lw=0., ms=4., c='k', zorder=5)
ax4.text(0.765, 0.35, 'M')

beta = 2. * alpha / np.pi * 180.

an1 = Arc([0.6, 0.8], 0.1, 0.1, angle=180.+beta, theta1=0., theta2=(180.-beta)*.5, color='k')
ax4.add_patch(an1)
an2 = Arc([0.0, 0.0], 0.1, 0.1, angle=beta, theta1=0., theta2=(180.-beta)*.5, color='k')
ax4.add_patch(an2)
ax4.text(0., 0.06, r'$\alpha$')
ax4.text(0.58, 0.71, r'$\alpha$')

ax4.set_xlim(-0.2, 1.3)
ax4.set_ylim(-0.2, 1.3)
ax4.grid(False)
ax4.set_axis_off()
ax4.arrow(-0.2, 0., 1.4, 0., lw=0.0001, length_includes_head=True, head_width=0.02, color='k', zorder=1)
ax4.arrow(0., -0.2, 0., 1.4, lw=0.0001, length_includes_head=True, head_width=0.02, color='k', zorder=1)
ax4.text(1.2, 0., 'x', color='k')
ax4.text(0., 1.2, 'y', color='k')

fig4.tight_layout()
```

На данном рисунке представлен исходный вектор $\textcolor{blue}{\mathbf{v}}$, единичный вектор $\textcolor{green}{\mathbf{e}_1}$, произведение которого на скаляр $\lambda = \lVert \textcolor{blue}{\mathbf{v}} \rVert_2$ дает вектор $\textcolor{green}{\mathbf{w}}$, причем $\lVert \textcolor{blue}{\mathbf{v}} \rVert_2 = \lVert \textcolor{green}{\mathbf{w}} \rVert_2$ по условию. Обозначим разность между вектором $\textcolor{green}{\mathbf{w}}$ и вектором $\textcolor{blue}{\mathbf{v}}$ как $\textcolor{olive}{2\mathbf{x}}$, то есть $\textcolor{blue}{\mathbf{v}} + \textcolor{olive}{2\mathbf{x}} = \textcolor{green}{\mathbf{w}}$. Следовательно, отрезок $\mathrm{AM}$ является медианой для треугольника $\mathrm{ABC}$. Поскольку $\lVert \textcolor{blue}{\mathbf{v}} \rVert_2 = \lVert \textcolor{green}{\mathbf{w}} \rVert_2$, то $\mathrm{AB} = \mathrm{AC}$, то есть треугольник $\mathrm{ABC}$ является равнобедренным, а медиана $\mathrm{AM}$ – также еще биссектрисой и высотой. Для угла $\alpha$ можно записать следующее отношение:

$$ \cos \alpha = \frac{\mathrm{BM}}{\mathrm{AB}} = \frac{\lVert \mathbf{x} \rVert_2}{\lVert \mathbf{v} \rVert_2} \implies
\lVert \mathbf{v} \rVert_2 \cos \alpha = \lVert \mathbf{x} \rVert_2 \implies
\lVert \mathbf{v} \rVert_2 \lVert \mathbf{u} \rVert_2 \cos \alpha = \lVert \mathbf{x} \rVert_2 \implies
\mathbf{v}^\top \mathbf{u} = \lVert \mathbf{x} \rVert_2. $$

То есть длина вектора $\mathbf{x}$ определяется скалярным произведением векторов $\mathbf{v}$ и $\mathbf{u}$. Тогда сам вектор $\mathbf{x}$, противоположно направленный единичному вектору $\mathbf{u}$, будет равен $\mathbf{x} = - \left( \mathbf{v}^\top \mathbf{u} \right) \mathbf{u}$. Подставим данное выражение в соотношение между векторами $\mathbf{v}$, $\mathbf{w}$ и $2 \mathbf{x}$:

$$ \begin{align}
\mathbf{v} + 2 \mathbf{x} &= \mathbf{w}, \\
\mathbf{v} - 2 \left( \mathbf{v}^\top \mathbf{u} \right) \mathbf{u} &= \mathbf{w}, \\
\mathbf{v} - 2 \mathbf{u} \left( \mathbf{u}^\top \mathbf{v} \right) &= \mathbf{w}, \\
\mathbf{v} - 2 \mathbf{u} \mathbf{u}^\top \mathbf{v} &= \mathbf{w}, \\
\left( \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top \right) \mathbf{v} &= \mathbf{w}, \\
\mathbf{H}_1 \mathbf{v} &= \lVert \mathbf{v} \rVert_2 \mathbf{e}_1.
\end{align} $$

Таким образом, матрица $\mathbf{H}_1 = \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top$ позволяет зеркально отразить вектор $\mathbf{v}$ относительно плоскости, заданной единичной нормалью $\mathbf{u}$, и получить вектор $\mathbf{w}$ той же длины, что и вектор $\mathbf{v}$, и сонаправленный с базисным вектором $\mathbf{e}_1$. Вектор $\mathbf{y} = 2 \mathbf{x}$ называется вектором отражения.

Отметим несколько особенностей линейного оператора Хаусхолдера $\mathbf{H} = \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top$.

```{admonition} Свойство
:class: note
Линейный оператор Хаусхолдера является [симметричной матрицей](LAB-2-Matrices.md#def-matrix-symmetric).
```

```{admonition} Доказательство
:class: proof
Симметричная матрица $\mathbf{A} \in \mathbb{R}^{m \times n}$, по определению, характеризуется следующим соотношением $\mathbf{A} = \mathbf{A}^\top$. Докажем справедливость данного соотношения для матрицы Хаусхолдера:

$$ \mathbf{H}^\top
= \left( \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top \right)^\top
= \mathbf{I}^\top - 2 \left( \mathbf{u} \mathbf{u}^\top \right)^\top
= \mathbf{I} - 2 \left( \mathbf{u}^\top \right)^\top \mathbf{u}^\top
= \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top
= \mathbf{H}. $$

В процессе преобразования использовался доказанный ранее [закон обратного порядка для транспонирования](LAB-2-Matrices.md#theorem-matrix-transp-dot).
```

```{admonition} Свойство
:class: note
Линейный оператор Хаусхолдера является [ортогональной матрицей](LAB-2-Matrices.md#matrix-orthogonal).
```

```{admonition} Доказательство
:class: proof
Для того чтобы доказать, что матрица Хаусхолдера является ортогональной, необходимо и достаточно доказать справедливость следующего соотношения $\mathbf{H}^\top = \mathbf{H}^{-1}$. Для этого рассмотрим и преобразуем следующее произведение:

$$ \begin{align}
\mathbf{H}^\top \mathbf{H}
&= \mathbf{H} \mathbf{H} \\
&= \left( \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top \right) \left( \mathbf{I} - 2 \mathbf{u} \mathbf{u}^\top \right) \\
&= \mathbf{I} \mathbf{I} - 2 \mathbf{I} \mathbf{u} \mathbf{u}^\top - 2 \mathbf{u} \mathbf{u}^\top \mathbf{I} + 4 \left( \mathbf{u} \mathbf{u}^\top \right) \left( \mathbf{u} \mathbf{u}^\top \right) \\
&= \mathbf{I} - 4 \mathbf{u} \mathbf{u}^\top + 4 \mathbf{u} \mathbf{u}^\top \mathbf{u} \mathbf{u}^\top \\
&= \mathbf{I} - 4 \mathbf{u} \mathbf{u}^\top + 4 \mathbf{u} \mathbf{u}^\top \\
&= \mathbf{I}.
\end{align} $$

Поскольку $\mathbf{H}^\top \mathbf{H} = \mathbf{I}$ и $\mathbf{H}^{-1} \mathbf{H} = \mathbf{I}$, то $\mathbf{H}^\top = \mathbf{H}^{-1}$, следовательно, линейный оператор Хаусхолдера является ортогональной матрицей.
```

Важно отметить, что матрица линейного преобразования Хаусхолдера содержит в себе единичный вектор, коллинеарный заданному вектору. Это можно показать следующим образом:

$$ \begin{align}
\mathbf{H}_1 \mathbf{v} &= \lVert \mathbf{v} \rVert_2 \mathbf{e}_1, \\
\mathbf{H}_1^{-1} \mathbf{H}_1 \mathbf{v} &= \mathbf{H}_1^{-1} \lVert \mathbf{v} \rVert_2 \mathbf{e}_1, \\
\mathbf{v} &= \lVert \mathbf{v} \rVert_2 \mathbf{H}_1^{-1} \mathbf{e}_1, \\
\frac{\mathbf{v}}{\lVert \mathbf{v} \rVert_2} &= \mathbf{H}_1 \mathbf{e}_1.
\end{align} $$

То есть в результате произведение матрицы линейного преобразования Хаусхолдера $\mathbf{H}_1$ и единичного базисного вектора $\mathbf{e}_1$ получается один из векторов-столбцов из матрицы $\mathbf{H}_1$, равный нормированному заданному вектору $\mathbf{v}$.

Рассмотрим преобразование Хаусхолдера на следующем примере.

```{admonition} Пример
:class: exercise
Необходимо построить ортонормированный базис на произвольном векторе $\mathbf{v} \in \mathbb{R}^3$ с использованием преобразования Хаусхолдера.
```

````{dropdown} Решение
С использованием [`numpy.random.uniform`](https://numpy.org/doc/stable/reference/generated/numpy.random.uniform.html) создадим произвольный вектор в $\mathbb{R}^3$:

```python
v = np.random.uniform(size=(3,)) - 0.5
print(v)
```

```{glue:} glued_out16
```

Вычислим его длину:

```python
lmbd = np.linalg.norm(v)
print(v / lmbd)
```

```{glue:} glued_out17
```

Зададим базисный вектор $\mathbf{e} \in \mathbb{R}^3$:

```python
e = np.array([1., 0., 0.])
```

Вычислим вектор отражения $\mathbf{y} \in \mathbb{R}^3$:

```python
y = v - lmbd * e
```

Тогда единичный вектор нормали к плоскости $\mathbf{u} \in \mathbb{R}^3$:

```python
u = - y / np.linalg.norm(y)
```

Построим матрицу Хаусхолдера, содержащую нормированный исходный вектор и ортонормированные ему векторы:

```
H = np.eye(3) - 2. * u[:, None] * u
print(H)
```

```{glue:} glued_out18
```

Длина каждого из трех векторов равна единице:

```python
print(np.linalg.norm(H[:, 0]), np.linalg.norm(H[:, 1]), np.linalg.norm(H[:, 2]))
```

```{glue:} glued_out19
```

Попарные [скалярные произведения](LAB-1-Vectors.md#vector-dot) векторов равны нулю:

```python
print(H[:, 0].dot(H[:, 1]), H[:, 0].dot(H[:, 2]), H[:, 1].dot(H[:, 2]))
```

```{glue:} glued_out20
```

Таким образом, матрица `H` составлена из нормированного исходного вектора и ему ортонормальных векторов. Следовательно, векторы-столбцы матрицы линейного преобразования Хаусхолдера образуют ортонормированный базис, базисный вектор которого коллинеарен исходному.
````

```{code-cell} python
:tags: [remove-cell]

v = np.random.uniform(size=(3,)) - 0.5

glue('glued_out16', MultilineText(v))

lmbd = np.linalg.norm(v)

glue('glued_out17', MultilineText(v / lmbd))

e = np.array([1., 0., 0.])

y = v - lmbd * e

u = - y / np.linalg.norm(y)

H = np.eye(3) - 2. * u[:, None] * u

glue('glued_out18', MultilineText(H))

glue('glued_out19', MultilineText(np.linalg.norm(H[:, 0]), np.linalg.norm(H[:, 1]), np.linalg.norm(H[:, 2])))

glue('glued_out20', MultilineText(H[:, 0].dot(H[:, 1]), H[:, 0].dot(H[:, 2]), H[:, 1].dot(H[:, 2])))
```
