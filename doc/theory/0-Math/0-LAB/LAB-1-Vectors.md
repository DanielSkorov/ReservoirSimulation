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

# Линейное пространство

Прежде чем перейти к изложению основ линейной алгебры необходимо ввести множество чисел, которые будут являться составляющими для элементов линейного (векторного) пространства. Такую алгебраическую структуру, для которой определены операции сложения, взятия противоположного значения, умножения и деления (кроме деления на ноль), принято называть [полем](https://en.wikipedia.org/wiki/Field_(mathematics)). Примерами полей являются пространства действительных или комплексных чисел, с которыми в рамках данного курса предстоит работать. Элементом поля является скаляр (число).

<a id='complex-numbers'></a>
````{dropdown} Арифметика комплексных чисел
С целью соблюдения математической строгости теорем и их доказательств в разделе, посвященном основам линейной алгебры, будут рассматриваться комплексные числа. В связи с этим следует напомнить их определение, свойства, а также некоторые операции над ними.

**Определение**

Компексными числами называют числа вида $u = a + b i$, где $a$ и $b$ – действительные числа, а $i$ – мнимая единица: $i^2 = -1$. Величина $\Re \left( u \right) = a$ комплексного числа называется его вещественной частью, а $\Im \left( u \right) = b$ мнимой. Причем если $a = 0$, то такое число $u = b i$ называют чисто мнимым, а если $b = 0$, то вещественным (или действительным) числом (то есть множество действительных чисел входит в множество комплексных и характеризуется нулевой мнимой частью). Комплексным нулем (или просто нулем) называют число вида $0 = 0 + 0 i$.

В [python](https://www.python.org/) комплексные числа задаются и выводятся в следующем виде:

```python
u = 1 + 2j
print(u)
```

```{glue:} complex1
```

Переменная `u` имеет тип [`complex`](https://docs.python.org/3/library/functions.html#complex):

```python
print(type(u))
```

```{glue:} complex2
```

К действительной и мнимой части комплексного числа можно отдельно обратиться с использованием атрибутов `real` и `imag`:

```python
print(u.real, u.imag)
```

```{glue:} complex3
```

Каждый из этих объектов имеет тип [`float`](https://docs.python.org/3/library/functions.html#float):

```python
print(type(u.real), type(u.imag))
```

```{glue:} complex4
```

**Сравнение**

Между комплексными числами нельзя ставить знаки "больше" или "меньше", однако два комплексных числа равны друг другу, если равны их вещественные и мнимые части соответственно.

При сравнении двух комплексных чисел в [python](https://www.python.org/) появится ошибка [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError):

```python
v = 3 + 4j
print(v > u)
```

```{glue:} complex5
```

Однако мы можем сравнить два комплексных числа с использованием [оператора сравнения](https://docs.python.org/3/reference/expressions.html#comparisons) `==`:

```python
print(u == v, v == 3 + 4j)
```

```{glue:} complex6
```

**Сложение и вычитание**

Для того чтобы сложить два комплексных числа, необходимо отдельно сложить их вещественные и мнимые части:

$$ \left( a + b i \right) + \left( c + d i \right) = \left( a + c \right) + \left( b + d \right) i. $$

Аналогичное правило используется для записи вычитания одного комплексного числа из другого:

$$ \left( a + b i \right) - \left( c + d i \right) = \left( a - c \right) + \left( b - d \right)  i. $$

Операции сложения и вычитания характеризуются следующими свойствами:

* *коммутативностью*: $u + v = v + u$,
* *ассоциативностью*: $\left( u + v \right) + w = u + \left(v + w \right)$,
* *свойство нуля*: $u + 0 = u$.

Например, сложим числа $u = 1 + 2 i$ и $v = 3 + 4 i$:

$$ u + v = \left( 1 + 2 i \right) + \left( 3 + 4 i \right) = \left( 1 + 3 \right) + \left( 2 + 4 \right) i = 4 + 6 i. $$

Аналогично вычтем число $u$ из числа $v$:

$$ v - u = \left( 3 + 4 i \right) - \left( 1 + 2 i \right) = \left( 3 - 1 \right) + \left( 4 - 2 \right) i = 2 + 2 i. $$

```python
print(u + v, v - u)
```

```{glue:} complex7
```

**Умножение**

Операция произведения двух комплексных чисел выполняется следующим образом:

$$ \left( a + b i \right) \cdot \left( c + d i \right) = ac + ad i + bc i + bd i^2 = \left( ac - bd \right) + \left( ad + bc \right) i. $$

Свойства умножения комплексных чисел:

* *коммутативность*: $u \cdot v = v \cdot u$,
* *ассоциативность*: $\left( u \cdot v \right) \cdot w = u \cdot \left(v \cdot w \right)$,
* *свойство нуля*: $u \cdot 0 = 0$,
* *свойство единицы*: $u \cdot 1 = u$,
* *дистрибутивность относительно сложения и вычитания*: $u \cdot \left( v + w \right) = uv + uw$.

Например, умножим числа $u = 1 + 2 i$ и $v = 3 + 4 i$:

$$ u \cdot v = \left( 1 + 2 i \right) \cdot \left( 3 + 4 i \right) = 3 + 4 i + 6 i + 8 i^2 = -5 + 10 i. $$

```python
print(u * v)
```

```{glue:} complex8
```

**Деление**

При делении одного комплексного числа на другое, не равное нулю, в результате также получается комплексное число:

$$ \frac{a + b i}{c + d i} = \frac{\left( a + b i \right) \left( c - d i \right)}{\left( c + d i \right) \left( c - d i \right)} = \frac{ac - ad i + bc i - bd i^2}{c^2 - d^2 i^2} = \frac{ac + bd}{c^2 + d^2} + \frac{bc - ad}{c^2 + d^2} i. $$

Рассмотрим операцию деления на примере нахождения частного от чисел $v = 3 + 4 i$ и $u = 1 + 2 i$:

$$ \frac{v}{u} = \frac{3 + 4 i}{1 + 2 i} = \frac{\left( 3 + 4 i \right) \left(1 - 2 i \right)}{\left(1 + 2 i \right) \left(1 - 2 i \right)} = \frac{3 - 6 i + 4 i - 8 i^2}{1 - 4 i^2} = \frac{11 - 2 i}{5} = \frac{11}{5} - \frac{2}{5} i. $$

```python
print(v / u)
```

```{glue:} complex9
```

**Сопряженность**

Сопряженным числом к некоторому комплексному числу $u = a + b i$ называется такое число $\bar{u} = a - b i$, вещественная часть которого равна вещественной части числа $u$, а мнимая взята с обратным знаком.

Данная операция имеет следующие свойства:

* $u = \bar{u}$ тогда и только тогда, когда $u$ – действительное число;
* $\bar{\bar{u}} = u$,
* $u \cdot \bar{u} = a^2 + b^2$ – действительное число;
* $u + \bar{u} = 2a$ – действительное число;
* $\overline{u + v} = \bar{u} + \bar{v}$,
* $\overline{u \cdot v} = \bar{u} \cdot \bar{v}$,
* $\overline{u - v} = \bar{u} - \bar{v}$,
* $\overline{\left( \frac{u}{v} \right)} = \frac{\bar{u}}{\bar{v}}$.

**Модуль числа**

Модуль комплексного числа $u = a + b i$ определяется следующим образом:

$$ \left| u \right| = \sqrt{a^2 + b^2}. $$

Аналогичный результат может быть получен, если под корнем умножим число на его сопряженное:

$$ \sqrt{u \cdot \bar{u}} = \sqrt{\left( a + b i \right) \cdot \left( a - b i \right)} = \sqrt{a^2 - b^2 i^2} = \sqrt{a^2 + b^2}. $$

Например, для числа $v = 3 + 4 i$ модуль числа:

$$ \left| v \right| = \left| 3 + 4 i \right| = \sqrt{3^2 + 4^2} = 5. $$

В [python](https://www.python.org/) встроенная функция [`abs`](https://docs.python.org/3/library/functions.html#abs) позволяет вычислить модуль комплексного числа:

```python
print(abs(v))
```

```{glue:} complex10
```

Выше были рассмотрены лишь базовые свойства комплексных чисел. С другими, в том числе с геометрическим представлением комплексных чисел, операциями возведения в степень и извлечения корня при желании можно ознакомиться, например, [здесь](https://en.wikipedia.org/wiki/Complex_number). Для работы с комплексными числами в [python](https://www.python.org/) существует встроенная библиотека [cmath](https://docs.python.org/3/library/cmath.html).
````

```{code-cell} python
:tags: [remove-cell]

u = 1 + 2j

glue('complex1', MultilineText(u))

glue('complex2', MultilineText(type(u)))

glue('complex3', MultilineText(u.real, u.imag))

glue('complex4', MultilineText(type(u.real), type(u.imag)))

v = 3 + 4j

try:
    print(v > u)
except TypeError as e:
    glue('complex5', MultilineText('TypeError:', e, split=True))

glue('complex6', MultilineText(u == v, v == 3 + 4j))

glue('complex7', MultilineText(u + v, v - u))

glue('complex8', MultilineText(u * v))

glue('complex9', MultilineText(v / u))

glue('complex10', MultilineText(abs(v)))
```

(vector)=
## Объявление и определение вектора

Для обозначения векторов здесь и далее используются малые латинские буквы, выделенные жирным: $\mathbf{v}$. Например, следующая запись

$$ \exists ~ \mathbf{v} \in \mathbb{C}^n $$

читается так: *существует* $\left( \exists \right)$ *вектор "v"* $\left( \mathbf{v} \right)$*, принадлежащий* $\left( \in \right)$ *пространству комплексных чисел* $\left( \mathbb{C} \right)$ *размерности* $n$.

Следует отметить, что здесь и далее под вектором, например $\mathbf{v}$, понимается именно вектор-столбец:

$$ \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}. $$

Его преобразование в вектор-строку будет осуществляться с использованием операции транспонирования, обозначаемой $\mathbf{v}^\top$.

```{admonition} Определение
:class: definition
Линейным (векторным) пространством над полем комплексных чисел будем называть множество элементов упорядоченной структуры, называемых ***векторами***, для которых определены операции сложения и умножения на скаляр.
```

Далее рассмотрим векторные операции сложения и умножения на скаляр. К этим и другим преобразованиям над элементами векторного пространства следует относиться, как к неким абстракциям: необходимо понимать их смысл, представлять результат и знать свойства. При этом в рамках данного учебного пособия для осуществления этих действий будем использовать библиотеку [numpy](https://numpy.org).

<a id='numpy-array'></a>
````{dropdown} Особенности работы с массивами [numpy](https://numpy.org)
Если по каким-то причинам не открывается страница с [документацией](https://numpy.org/doc/stable/index.html) к [numpy](https://numpy.org), статическую версию можно найти и скачать на [github](https://github.com/numpy/doc/tree/main).

**Создание массивов с использованием [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html)**

Одномерный массив с использованием [numpy](https://numpy.org) может быть создан следующим образом. Импортируется библиотека:

```python
import numpy as np
```

В качестве аргумента функции [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) может быть передан [список](https://docs.python.org/3/library/stdtypes.html#typesseq-list), [кортеж](https://docs.python.org/3/library/stdtypes.html#tuples) (или другая [последовательность](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)) из чисел (в данном случае – список действительных чисел типа [`float`](https://docs.python.org/3/tutorial/floatingpoint.html)):

```python
a = np.array([1., 2., 3.])
```

Объект `a` имеет тип [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html):

```python
print(type(a))
```

```{glue:} array1
```

Здесь и далее любые объекты, имеющие такой тип, будем именовать массивами.

**Тип данных**

У массивов есть ряд важных атрибутов (свойств). Одним из них является тип данных [`dtype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dtype.html#numpy-ndarray-dtype):

```python
print(a.dtype)
```

```{glue:} array2
```

Типом данных массива `a` является [`numpy.float64`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float64), наследуемый от стандартного типа [`float`](https://docs.python.org/3/tutorial/floatingpoint.html). При этом в качестве типа данных массивов могут быть:

* логический [`numpy.bool`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool);
* целые числа различного размера [`numpy.integer`](https://numpy.org/doc/stable/reference/arrays.scalars.html#integer-types);
* числа с плавающей точкой различного размера [`numpy.floating`](https://numpy.org/doc/stable/reference/arrays.scalars.html#floating-point-types);
* комплексные числа различного размера [`numpy.complexfloating`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.complexfloating);
* строки [`numpy.character`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.character);
* и др.

Тип данных можно явно задать при вызове функции [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html), указав необходимый тип в качестве параметра `dtype` данной функции. В рамках данного учебного пособия преимущественно будут использоваться массивы с [комплексными числами двойной точности](https://en.wikipedia.org/wiki/Complex_data_type) [`numpy.complex128`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.complex128), [числами с плавающей точкой двойной точности](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) [`numpy.float64`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float64), [двойными длинными целыми числами](https://en.wikipedia.org/wiki/Integer_(computer_science)#Common_integral_data_types) [`numpy.int_`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int_), а также с [логическим типом данных](https://en.wikipedia.org/wiki/Boolean_data_type) [`numpy.bool`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool).

**Измерение**

Другим важным атрибутом объекта [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) является [`ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html), хранящий данные о количестве осей массива или, другими словами, о его измерении:

```python
print(a.ndim)
```

```{glue:} array3
```

Мы видим, что созданный массив `a` является одномерным.

Создадим двумерный массив `b`, передав в качестве аргумента функции [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) список из списков с числами с плавающей точкой:

```python
b = np.array([[2., 3., 4.], [3., 4., 5.]])
```

Такой массив является двумерным:

```python
print(b.ndim)
```

```{glue:} array4
```

**Размерность**

Для отображения размерности массива (количества строк, столбцов и т.д.) используется атрибут [`shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html):

```python
print(b.shape)
```

```{glue:} array5
```

Данный атрибут представляет собой кортеж, количество элементов в котором соответствует измерению массива, при этом сам элемент обозначает "размер" массива по определенной оси. Например, созданный двумерный массив `b` состоит из двух строк и трех столбцов:

```python
print(b)
```

```{glue:} array6
```

Первая ось данного массива – это строки (две строки), вторая – столбцы (три столбца).

Одномерный массив `a` имеет всего одну ось, а "размер" массива по этой оси равен трем.

```python
print(a.shape)
```

```{glue:} array7
```

При создании многомерных массивов необходимо следить за "размером" массива по каждой оси. Например, при создании следующего массива появится ошибка `ValueError`, поскольку "размер" массива по последней оси (то есть по столбцам) не одинаков:

```python
e = np.array([[2., 3., 4.], [3., 4.]])
```

```{glue:} array8
```

**Смежность**

Отличительной особенностью массивов библиотеки [numpy](https://numpy.org) от встроенных в [python](https://www.python.org/) последовательностей является их ***смежность*** (*contiguity*). При использовании такой структуры хранения данных каждый ее следующий элемент [расположен в соседней ячейке памяти](https://stackoverflow.com/a/26999092), что позволяет гораздо быстрее обращаться к ним в ходе реализации циклов `for` при выполнении различных операций линейной алгебры и приводит к существенному ускорению вычислений. При этом в библиотеке [реализовано](https://numpy.org/doc/stable/dev/internals.html) два способа [хранения многомерных массивов](https://en.wikipedia.org/wiki/Row-_and_column-major_order): ***C-contiguous*** (построчный способ используется по умолчанию в библиотеках для языков прогаммирования, наследуемых от `C`, например, в [`numpy`](https://numpy.org/doc/stable/glossary.html#term-row-major) или стандартной библиотеке для [`C++`](https://en.cppreference.com/w/cpp/container/array.html) и др.) и ***F-contiguous*** (столбцовый порядок используется по умолчанию в языке программирования [`Fortran`](https://fortran-lang.org/learn/best_practices/multidim_arrays/)). Понимание способа хранения данных важно для низкоуровневой оптимизации алгоритмов и численных методов, используемых в гидродинамическом моделировании.

Для отображения способа хранения данных массива используется атрибут [`flags`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html):

```python
print(b.flags['C_CONTIGUOUS'], b.flags['F_CONTIGUOUS'])
```

```{glue:} array9
```

Таким образом, данные для двумерного массива `b` сохранены построчно. При этом одномерные массивы являются смежными как для построчного хранения данных, так и для столбцового:

```python
print(a.flags['C_CONTIGUOUS'], a.flags['F_CONTIGUOUS'])
```

```{glue:} array10
```

Для перевода из одного способа хранения данных в другой используются функции [`numpy.asfortranarray`](https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html#numpy-asfortranarray) и [`numpy.ascontiguousarray`](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray), которые возвращают тот же объект, что и передан на вход данным функциям, если целевой тип хранения данных совпадает с входным, и копию объекта, если целевой тип хранения не совпадает с входным.

Выведем рассмотренные выше атрибуты для следующих массивов:

```python
r = np.array([[1., 2., 3.]])
print(r.ndim, r.shape, r.flags['C_CONTIGUOUS'], r.flags['F_CONTIGUOUS'])
```

```{glue:} array11
```

Данный массив (*вектор-строка*) является:

* двумерным,
* состоящим из одной строки и трех столбцов,
* смежным как для построчного хранения данных, так и для столбцового.

```python
c = np.array([[1.], [2.], [3.]])
print(c.ndim, c.shape, c.flags['C_CONTIGUOUS'], c.flags['F_CONTIGUOUS'])
```

```{glue:} array12
```

Данный массив (*вектор-столбец*) является:

* двумерным,
* состоящим из трех строк и одного столбца,
* смежным как для построчного хранения данных, так и для столбцового.

**Шаг памяти**

Шаг памяти для каждой оси массива хранится в атрибуте [`strides`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html):

```python
print(b.strides)
```

```{glue:} array13
```

Шаг памяти говорит о количестве байтов, через которые находится следующий элемент по соответствующей оси массива. Например, для массива `b`, который сохранен построчно, каждый следующий элемент в одной строке находится через 8 байтов (64 бита). Ровно столько памяти необходимо для хранения одного элемента в формате числа с плавающей точкой двойной точности. При этом, поскольку в строке только три элемента, то каждый следующий элемент в одном столбце находится через 24 байта. Если же создать копию данного массива, сохранив данные в столбец, то получим следующий результат:

```python
f = np.asfortranarray(b)
print(f.strides)
```

```{glue:} array14
```

Теперь каждый следующий элемент в одном столбце расположен в соседних 8 байтах, в то время как соседний элемент в одной строке – в 16 байтах (количество строк в массиве `f`, как и в массиве `b`, равняется двум). Таким образом, чем через меньшее количество памяти находится элемент смежного массива, тем быстрее к нему осуществляется доступ при индексировании в цикле `for`, что вызвано особенностями иерархии памяти современных процессоров. Когда процессор запрашивает элемент смежного массива из оперативной памяти, то в быстрый кэш (L1/L2) подгружается сразу несколько элементов (в зависимости от размера кэша). Если следующий элемент массива находится в быстром кэше, то обращение к нему будет в десятки раз быстрее обращения к оперативной памяти.

**Вид**

Таким образом, меняя шаги памяти можно по-разному обращаться к одному и тому же набору данных. Например, мы можем создать *одномерный вид (1d-view)* на многомерный массив с использованием метода [`ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ravel.html):

```python
s = b.ravel()
print(s, s.strides, np.may_share_memory(b, s), sep='\n')
```

```{glue:} array15
```

Массив `s` является одномерным, каждый элемент которого расположен в 8 байтах от другого, причем его данные и данные массива `b` лежат в одних и тех же ячейках памяти. Это может быть полезно в тех случаях, когда на один и тот же набор данных необходимо *смотреть ([указывать](https://en.wikipedia.org/wiki/Pointer_(computer_programming)))* по-разному: для одного действия нам необходимо, чтобы эти данные выглядели, как двумерный массив, а для другой – как одномерный. Важно отметить, что операция создания нового вида для уже существующих данных выполняется гораздо быстрее, чем копирование этих данных с учетом требуемой структуры их хранения.

**Индексирование**

Создадим произвольный одномерный массив `a`:

```python
a = np.array([9., 2., 4., 6., 1., 0., 8., 3.])
```

Одномерные массивы индексируются аналигично встроенным в [python](https://www.python.org/) спискам. Для обращения к элементу массива по индексу необходимо передать его порядковый номер (начиная с нуля) в квадратных скобках:

```python
print(a[1])
```

```{glue:} array16
```

Поддерживаются и отрицательные индексы:

```python
print(a[-1])
```

```{glue:} array17
```

В качестве индекса можно передать [срез](https://docs.python.org/3/glossary.html#term-slice):

```python
print(a[1:4], a[:3], a[5:])
```

```{glue:} array18
```

Покажем, как индексируются двумерные массивы на следующем примере:

```python
b = np.array([
    [0., 1., 2., 3., 4.,  5.],
    [1., 2., 3., 4., 5.,  6.],
    [2., 3., 4., 5., 6.,  7.],
    [3., 4., 5., 6., 7.,  8.],
    [4., 5., 6., 7., 8.,  9.],
    [5., 6., 7., 8., 9., 10.],
])
```

При индексировании многомерных массивов для каждой оси указывается свой индекс через запятую в квадратных скобках:

```python
print(b[2, 4])
```

```{glue:} array19
```

При этом сначала индексируются строки, потом столбцы. Для того чтобы получить строку или столбец целиком, необходимо указать индекс для данной оси, а для остальных – `:`. Поскольку двумерный массив можно интерпретировать как массив одномерных массивов (строк), то обратиться к строке целиком можно по ее индексу:

```python
print(b[1, :], b[:, 2], b[3])
```

```{glue:} array20
```

В целом, при индексировании многомерных массивов необходимо помнить о том, что порядок индексов должен соответствовать порядку осей многомерного массива. Срезы также можно использовать для индексирования многомерных массивов. Следует отметить, что третьим аргументом среза может быть шаг:

```python
t = b[0:3:2, 0:3:2]
print(t)
```

```{glue:} array21
```

Однако если шаг больше единицы, то обращение к элементам массива непоследовательно, то есть такой вид (указатель) на данные не является смежным:

```python
print(np.may_share_memory(b, t), t.flags['C_CONTIGUOUS'], t.flags['F_CONTIGUOUS'], t.strides)
```

```{glue:} array22
```

Массивы `b` и `t` обращаются к одним и тем же данным в памяти, при этом данные массива `t` не являются смежными, поскольку следующий элемент в одной строке расположен в 16 байтах от предыдущего, а в одном столбце – в 80 байтах.

Изменять элементы массива можно посредством индексирования:

```python
print(a)
a[2] = 1.
a[3] -= 6.
a[4:7] = [8., 1., 0.]
print(a)
```

```{glue:} array23
```

При изменении элемента вида (указателя) на некоторый массив происходит измение самих данных, что приводит к изменению и исходного массива:

```python
r = b.ravel()
r[14] = -1.
print(b)
```

```{glue:} array24
```

**Изменение измерения и размерности**

Выше уже был рассмотрен пример изменения измерения и размерности массива посредством создания одномерного вида на многомерный массив с использованием метода [`ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ravel.html). Рассмотрим и другие способы.

К массиву можно добавлять новую ось с использованием [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). Например, можно создать двумерный вид "вектор-строка" или "вектор-столбец" на одномерный массив:

```python
a = np.array([1., 2., 3.])
v = a[:, None]
print(v, v.shape, np.may_share_memory(v, a), sep='\n')
```

```{glue:} array25
```

```python
s = a[None, :]
print(s, s.shape, np.may_share_memory(s, a), sep='\n')
```

```{glue:} array26
```

В первом случае был создан вид размерностью `(3, 1)` (то есть вектор-столбец) на данные одномерного массива `a`. Во втором случае был создан вид размерностью `(1, 3)` (то есть вектор-строка) на данные одномерного массива `a`. В представленных выше примерах `None` указывает на необходимость создания и положение новой оси массива. Вместо `None` для обозначения новой оси массива можно использовать [`numpy.newaxis`](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis). Символы двоеточия `:` указывают положение существующих осей массива `a`.

Другим подходом к изменению размерности массива является метод [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html#numpy-ndarray-reshape), принимающий на вход размерность нового массива и возвращающий, по возможности, вид на уже имеющиеся данные.

```python
b = np.array([1., 2., 3., 4., 5., 6.])
m = b.reshape((2, 3))
print(m, m.shape, np.may_share_memory(m, b), sep='\n')
```

```{glue:} array27
```

В данном случае был создан двумерный вид `m` размерностью `(2, 3)` на одномерный массив `b`.

Следует отметить, что на практике в качестве векторных величин гораздо удобнее и быстрее использовать одномерные массивы (размерность которых `(n,)`), чем двумерные вектор-столбец с размерностью `(n, 1)` или вектор-строку с размерностью `(1, n)`. При этом если имеется необходимость представления данных в виде многомерного массива, то рекомендуется создать соответствующий вид с использованием [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), метода [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html#numpy-ndarray-reshape) или функции [`numpy.lib.stride_tricks.as_strided`](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html), применяемой для создания пользовательских видов (указателей) с учетом заданных размерности и шагов памяти. Существуют и другие способы изменения измерения и размерности массива, например, [`numpy.swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html), [`numpy.rollaxis`](https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html), [`numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html) и другие.

С массивами можно выполнять различные операции: сложение, вычитание, умножение, деление, возведение в степень и т.д., которые будут рассмотрены по мере их изложения в главе, посвященной основам линейной алгебры.
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

a = np.array([1., 2., 3.])

glue('array1', MultilineText(type(a)))

glue('array2', MultilineText(a.dtype))

glue('array3', MultilineText(a.ndim))

b = np.array([[2., 3., 4.], [3., 4., 5.]])

glue('array4', MultilineText(b.ndim))

glue('array5', MultilineText(b.shape))

glue('array6', MultilineText(b))

glue('array7', MultilineText(a.shape))

try:
    e = np.array([[2., 3., 4.], [3., 4.]])
except ValueError as e:
    glue('array8', MultilineText('ValueError:', e, split=True))

glue('array9', MultilineText(b.flags['C_CONTIGUOUS'], b.flags['F_CONTIGUOUS']))

glue('array10', MultilineText(a.flags['C_CONTIGUOUS'], a.flags['F_CONTIGUOUS']))

r = np.array([[1., 2., 3.]])

glue('array11', MultilineText(r.ndim, r.shape, r.flags['C_CONTIGUOUS'], r.flags['F_CONTIGUOUS']))

c = np.array([[1.], [2.], [3.]])

glue('array12', MultilineText(c.ndim, c.shape, c.flags['C_CONTIGUOUS'], c.flags['F_CONTIGUOUS']))

glue('array13', MultilineText(b.strides))

f = np.asfortranarray(b)

glue('array14', MultilineText(f.strides))

s = b.ravel()

glue('array15', MultilineText(s, s.strides, np.may_share_memory(b, s), sep='\n'))

a = np.array([9., 2., 4., 6., 1., 0., 8., 3.])

glue('array16', MultilineText(a[1]))

glue('array17', MultilineText(a[-1]))

glue('array18', MultilineText(a[1:4], a[:3], a[5:]))

b = np.array([
    [0., 1., 2., 3., 4.,  5.],
    [1., 2., 3., 4., 5.,  6.],
    [2., 3., 4., 5., 6.,  7.],
    [3., 4., 5., 6., 7.,  8.],
    [4., 5., 6., 7., 8.,  9.],
    [5., 6., 7., 8., 9., 10.],
])

glue('array19', MultilineText(b[2, 4]))

glue('array20', MultilineText(b[1, :], b[:, 2], b[3]))

t = b[0:3:2, 0:3:2]

glue('array21', MultilineText(t))

glue('array22', MultilineText(np.may_share_memory(b, t), t.flags['C_CONTIGUOUS'], t.flags['F_CONTIGUOUS'], t.strides))

acopy = a.copy()
a[2] = 1.
a[3] -= 6.
a[4:7] = [8., 1., 0.]

glue('array23', MultilineText(acopy, a, sep='\n'))

r = b.ravel()
r[14] = -1.

glue('array24', MultilineText(b))

a = np.array([1., 2., 3.])
v = a[:, None]

glue('array25', MultilineText(v, v.shape, np.may_share_memory(v, a), sep='\n'))

s = a[None, :]

glue('array26', MultilineText(s, s.shape, np.may_share_memory(s, a), sep='\n'))

b = np.array([1., 2., 3., 4., 5., 6.])
m = b.reshape((2, 3))

glue('array27', MultilineText(m, m.shape, np.may_share_memory(m, b), sep='\n'))
```

***Сложением*** векторов $\mathbf{a}, \, \mathbf{b} \in \mathbb{C}^n$ называется операция, результатом которой является новый вектор $\mathbf{c} \in \mathbb{C}^n$, каждый элемент которого равна сумме соответствующих элементов векторов-слагаемых:

$$ \mathbf{c} = \mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{bmatrix}. $$

Данная операция характеризуется следующими аксиомами:

* *коммутативность сложения:* $\mathbf{a} + \mathbf{b} = \mathbf{b} + \mathbf{a}$;
* *ассоциативность сложения:* $\mathbf{a} + \left( \mathbf{b} + \mathbf{c} \right) = \left( \mathbf{a} + \mathbf{b} \right) + \mathbf{c}$;
* существует *нулевой вектор* $\mathbf{0}$ такой, что $\mathbf{a} + \mathbf{0} = \mathbf{a}$;
* *противоположным вектором* вектору $\mathbf{a}$ будем называть вектор $\left( -\mathbf{a} \right)$ такой, что $\mathbf{a} + \left( -\mathbf{a} \right) = \mathbf{a} - \mathbf{a} = \mathbf{0}$;

Рассмотрим данную операцию на следующем примере.

```{admonition} Пример
:class: exercise

Пусть векторы $\mathbf{a}, \, \mathbf{b} \in \mathbb{C}^2$ заданы следующим образом:

$$ \mathbf{a} = \begin{bmatrix} 2 + 3 i \\ 1 - 2 i \end{bmatrix}, \; \mathbf{b} = \begin{bmatrix} 1 + 2 i \\ 3 + 3 i \end{bmatrix}. $$

Необходимо найти вектор $\mathbf{c} \in \mathbb{C}^2$, удовлетворяющий условию $\mathbf{c} = \mathbf{a} + \mathbf{b}$.
```

````{dropdown} Решение
Сначала выполним операцию сложения векторов $\mathbf{a}$ и $\mathbf{b}$ с использованием библиотеки [numpy](https://numpy.org). Для этого импортируем библиотеку:

```python
import numpy as np
```

Зададим координаты векторов $\mathbf{a}$ и $\mathbf{b}$ в виде одномерных массивов:

```python
a = np.array([2 + 3j, 1 - 2j])
b = np.array([1 + 2j, 3 + 3j])
```

Операция сложения векторов осуществляется следующим образом:

```python
c = a + b
```

Выведем полученный результат:

```python
print(c)
```

```{glue:} add1
```

Аналогичным образом осуществляется операция вычитания векторов:

```python
d = c - b
print(d)
```

```{glue:} add2
```
````

```{code-cell} python
:tags: [remove-cell]

import numpy as np

a = np.array([2 + 3j, 1 - 2j])
b = np.array([1 + 2j, 3 + 3j])

c = a + b

glue('add1', MultilineText(c))

d = c - b

glue('add2', MultilineText(d))
```

***Умножением вектора*** $\mathbf{a} \in \mathbb{C}^n$ ***на скаляр*** $\lambda \in \mathbb{C}$ называется операция, результатом которой является новый вектор $\mathbf{b} \in \mathbb{C}^n$, элементы которого получаются умножением каждого элемента вектора $\mathbf{a}$ на число $\lambda$:

$$ \mathbf{b} = \lambda \cdot \mathbf{a} = \lambda \mathbf{a} = \lambda \cdot \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} = \begin{bmatrix} \lambda \cdot a_1 \\ \lambda \cdot a_2 \\ \vdots \\ \lambda \cdot a_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}. $$

Данная операция характеризуются следующими аксиомами:

* *ассоциативность умножения на скаляр:* $\alpha \left( \beta \mathbf{a} \right) = \left( \alpha \beta \right) \mathbf{a}$;
* существует *унитарный скаляр* (единица) такой, что $\mathbf{a} \cdot 1 = \mathbf{a}$;
* *дистрибутивность умножения вектора на скаляр относительно сложения скаляров:* $\left( \alpha + \beta \right) \mathbf{a} = \alpha \mathbf{a} + \beta \mathbf{a}$;
* *дистрибутивность умножения вектора на скаляр относительно сложения векторов:* $\alpha \left( \mathbf{a} + \mathbf{b} \right) = \alpha \mathbf{a} + \alpha \mathbf{b}$.

Рассмотрим произведение вектора и числа на следующем примере.

```{admonition} Пример
:class: exercise
Пусть, имеется вектор $\mathbf{a} = \begin{bmatrix} 3 + 2 i \\ 4 - 3 i \end{bmatrix}$. Необходимо найти векторы $\mathbf{b} = \left( 2 + 2 i \right) \cdot \mathbf{a}$ и $\mathbf{c} = -1 \cdot \mathbf{a}$.
```

````{dropdown} Решение
Зададим координаты вектора $\mathbf{a}$ в виде одномерного массива:

```python
a = np.array([3 + 2j, 4 - 3j])
```

Вычислим координаты вектора $\mathbf{b}$:

```python
b = (2 + 2j) * a
print(b)
```

```{glue:} mult1
```

Повторим данное действие для вектора $\mathbf{c}$:

```python
c = -1. * a
print(c)
```

```{glue:} mult2
```

Аналогичным образом можно осуществить и другие операции между числом и массивом, в том числе сложение, вычитание, деление.
````

```{code-cell} python
:tags: [remove-cell]

a = np.array([3 + 2j, 4 - 3j])

b = (2 + 2j) * a

glue('mult1', MultilineText(b))

c = -1. * a

glue('mult2', MultilineText(c))
```

(linear-dependence)=
## Линейная зависимость векторов

```{admonition} Определение
:class: definition
Выражение вида

$$ \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \ldots + \alpha_n \mathbf{v}_n = \sum_{i=1}^n \alpha_i \mathbf{v}_i $$

будем называть ***линейной комбинацией*** элементов векторного пространства $\mathbf{v}_i \in \mathbb{C}^n, \, i = 1 \, \ldots \, n,$ с коэффициентами $\alpha_i \in \mathbb{C}, \, i = 1 \, \ldots \, n$.
```

Очевидно, что результатом линейной комбинации векторов с различными коэффициентами является вектор. Совокупность всех возможных результатов линейной комбинации выбранных векторов, принадлежащих единому векторному пространству, называется *линейной оболочкой (span или linear hull)*. По сути, линейная оболочка представляет собой замкнутое неограниченное подпространство в выбранном векторном пространстве, что обуславливается фиксированием векторов в их линейной комбинации. То есть какие бы коэффициенты ни использовались для расчета вектора с использованием заданной линейной комбинации, он всегда будет находиться внутри линейной оболочки.

Линейная оболочка одного ненулевого вектора представляет множество всех векторов вида $\alpha \mathbf{v}$ и геометрически описывается прямой, содержащей данный вектор. Аналогично, линейной оболочкой двух ненулевых векторов, не лежащих на одной прямой (неколлинеарных), является плоскость. А все трехмерное пространство является линейной оболочкой трех ненулевых векторов, не принадлежащих одной плоскости. Данные частные случаи иллюстрируют неограниченность линейной оболочки.

Определив линейную оболочку как совокупность всех линейных комбинаций векторов, важно исследовать внутреннюю структуру самого набора векторов. Приведенные выше частные случаи линейных оболочек даны с определенными условиями, например, "векторы, не лежащие на одной прямой", или "векторы, не принадлежащие одной плоскости". В связи с этим возникает вопрос: являются ли все элементы произвольного набора векторов уникальными с точки зрения формирования подпространства, или некоторые из них могут быть выражены через остальные? Ответ на этот вопрос дает анализ линейной зависимости векторов.

```{admonition} Определение
:class: tip
Совокупность векторов $\mathbf{v}_i \in \mathbb{C}^n, \, i = 1 \, \ldots \, n,$ называется ***линейно независимой***, если их линейная комбинация равна нулевому вектору только в том случае, когда равны нулю все коэффициенты $\alpha_i, \, i = 1 \, \ldots \, n$:

$$ \sum_{i=1}^n \alpha_i \mathbf{v}_i = 0, \; \mathrm{iff} \; \alpha_i = 0, \; i = 1 \, \ldots \, n. $$
```

Иными словами, векторы являются линейно независимыми тогда и только тогда, когда уравнение (относительно коэффициентов):

$$ \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \ldots + \alpha_n \mathbf{v}_n = \mathbf{0} $$

имеет только одно решение – все $\alpha_i, \, i = 1 \, \ldots \, n,$ равны нулю (такое решение называется *тривиальным*). И наоборот, если существует хотя бы один набор коэффициентов $\alpha_i, \, i = 1 \, \ldots \, n,$ отличных от нуля, при котором выражение выше обращается в ноль, то как минимум один вектор является "избыточным", а вся совокупность векторов – линейно зависимой.

Из данного определения следует, что если хотя бы один из рассматриваемых векторов является нулевым, то такая система является линейно зависимой. Действительно, допустим, $\mathbf{v}_1 = \mathbf{0}$, а все остальные векторы $\mathbf{v}_i \neq \mathbf{0}, \, i = 1 \, \ldots \, n$. Тогда равенство $\sum_{i=1}^n \alpha_i \mathbf{v}_i = 0$ может быть справедливо и при ненулевом значении $\alpha_1$. Следовательно, отсутствие в совокупности нулевого вектора является необходимым условием, для того чтобы считать эту совокупность векторов линейно независимой.

Другим важным следствием из данного определения является утверждение о том, что если количество векторов, формирующих линейную комбинацию, больше размерности пространства, то эти векторы линейно зависимы. Действительно, приведенное выше векторное уравнение можно представить в виде системы линейных уравнений, в которой в качестве известных переменных участвуют элементы векторов:

$$ \begin{cases}
v_{11} \alpha_1 + v_{12} \alpha_2 + \ldots + v_{1m} \alpha_m = 0, \\
v_{21} \alpha_1 + v_{22} \alpha_2 + \ldots + v_{2m} \alpha_m = 0, \\
\vdots \\
v_{n1} \alpha_1 + v_{n2} \alpha_2 + \ldots + v_{nm} \alpha_m = 0,
\end{cases} $$

где $v_{ij}, \, i = 1 \, \ldots \, n, j = 1 \, \ldots \, m$ представляет собой $i$-й элемент $j$-го вектора. Такая система линейных уравнений является [однородной](https://en.wikipedia.org/wiki/System_of_linear_equations#Homogeneous_systems), то есть она всегда имеет минимум одно решение. Соответственно, если в однородной системе линейных уравнений количество неизвестных $m$ (количество векторов) больше количества уравнений $n$ (размерности векторного пространства), то система имеет бесконечно много решений, в том числе отличное от нулевого, что обуславливает линейную зависимость векторов.

В контексте векторных пространств важно не только отсутствие линейной зависимости, но и способность набора векторов описывать все пространство целиком. Объединяя свойство линейной независимости с понятием линейной оболочки, мы приходим к фундаментальному определению базиса – минимального порождающего множества, которое служит своего рода "каркасом" или системой координат для всего линейного пространства.

(basis)=
## Базис

```{admonition} Определение
:class: definition

Базис конечномерного векторного пространства – это упорядоченная совокупность линейно независимых векторов, такая что любой вектор этого пространства может быть представлен в виде их линейной комбинации.
```

По определению, любой базис выражается через $n$ базисных линейно независимых векторов:

$$ \mathrm{B} = \begin{Bmatrix} \mathbf{v}_1, \, \mathbf{v}_2, \, \ldots, \, \mathbf{v}_n \end{Bmatrix}. $$

Пусть имеется вектор $\mathbf{x}$, координаты которого выражены через базис $\mathrm{B}$. Это может быть записано так: $\begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathrm{B}$.

Если вектор $\begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathrm{B}$ характеризуется координатами $\begin{bmatrix} x_1, \,  x_2, \,  \ldots, \, x_n \end{bmatrix}^\top_\mathrm{B}$ в базисе $\mathrm{B}$, то справедливо следующее соотношение:

$$ \begin{bmatrix} \mathbf{x} \end{bmatrix}_\mathrm{B} = x_1 \cdot \begin{bmatrix} \mathbf{v_1} \end{bmatrix}_\mathrm{B} + x_2 \cdot \begin{bmatrix} \mathbf{v_2} \end{bmatrix}_\mathrm{B} + \ldots + x_n \cdot \begin{bmatrix} \mathbf{v_n} \end{bmatrix}_\mathrm{B}. $$

Следует отметить, что условие о том, что любой вектор можно представить в виде линейной комбинации базисных векторов, в данном определении дается аксиоматически. То есть мы договариваемся называть базисом только такое количество линейно независимых векторов, которое равняется размерности пространства. Однако условие единственности разложения произвольного вектора в базисе требует отдельного доказательства.

<a id='basis-theorem'></a>
```{admonition} Теорема
:class: theorem
Любой вектор может быть *уникально* представлен в виде линейной комбинации базисных векторов и координат данного вектора в этом базисе.
```

```{admonition} Доказательство
:class: proof

Для доказательства данной теоремы воспользуемся методом от противного. Обозначим базис $\mathrm{B} = \begin{Bmatrix} \mathbf{e}_1, \, \mathbf{e}_2, \, \ldots, \, \mathbf{e}_n \end{Bmatrix}$. Предположим, что произвольный вектор $\mathbf{v}$ можно разложить по этому базису двумя способами:

$$ \begin{align}
\mathbf{v} &= a_1 \mathbf{e}_1 + a_2 \mathbf{e}_2 + \ldots + a_n \mathbf{e}_n, \\
\mathbf{v} &= b_1 \mathbf{e}_1 + b_2 \mathbf{e}_2 + \ldots + b_n \mathbf{e}_n.
\end{align} $$

Вычтем одно выражение из другого:

$$ \mathbf{v} - \mathbf{v} = a_1 \mathbf{e}_1 + a_2 \mathbf{e}_2 + \ldots + a_n \mathbf{e}_n - b_1 \mathbf{e}_1 - b_2 \mathbf{e}_2 - \ldots - b_n \mathbf{e}_n. $$

Поскольку вектор один и тот же, то $\mathbf{v} - \mathbf{v} = 0$. В другой части вынесем общие множители за скобку:

$$ \left( a_1 - b_1 \right) \mathbf{e}_1 + \left( a_2 - b_2 \right) \mathbf{e}_2 + \ldots + \left( a_n - b_n \right) \mathbf{e}_n = 0. $$

Векторы $\mathbf{e}_i, \, i = 1 \, \ldots \, n,$ являются линейно независимые. Следовательно, по определению, их линейная комбинация равна нулю тогда и только тогда, когда равны нулю коэффициенты этой линейной комбинации:

$$ \begin{array}{c}
a_1 - b_1 = 0, \\
a_2 - b_2 = 0, \\
\vdots \\
a_n - b_n = 0.
\end{array} $$

Таким образом, координаты векторов совпадают. Следовательно, разложение произвольного вектора по базису единственно.
```

Определив понятие базиса, мы получили возможность описывать любой вектор через набор его координат. Однако одной лишь линейной структуры недостаточно для полноценного изучения пространства: нам необходимо научиться измерять длины векторов и определять взаимное расположение между ними. Для этих целей используется эрмитово скалярное произведение.

(inner-product)=
## Эрмитово скалярное произведение векторов

```{admonition} Определение
:class: definition
***Эрмитовым скалярным произведением*** двух векторов $\mathbf{a}, \, \mathbf{b} \in \mathbb{C}^n$ будем называть операцию, результатом которой является комплексное число, и удовлетворяющую следующим аксиомам:

* *эрмитова (комплексная) симметричность:* $\langle \mathbf{a}, \, \mathbf{b} \rangle = \overline{\langle \mathbf{b}, \, \mathbf{a} \rangle}$;
* *линейность по второму аргументу:* $\langle \mathbf{a}, \, \mathbf{b} + \mathbf{c} \rangle = \langle \mathbf{a}, \, \mathbf{b} \rangle + \langle \mathbf{a}, \, \mathbf{c} \rangle$ и  $\langle \mathbf{a}, \, \alpha \mathbf{b} \rangle = \alpha \langle \mathbf{a}, \, \mathbf{b} \rangle$;
* *положительная полуопределенность:* $\langle \mathbf{a}, \, \mathbf{a} \rangle \geq 0, \; \langle \mathbf{a}, \, \mathbf{a} \rangle = 0 \Leftrightarrow \mathbf{a} = 0$.
```

Следствием из данного определения является *полулинейность по первому аргументу* эрмитового скалярного произведения:

$$ \langle \mathbf{a} + \mathbf{b}, \, \mathbf{c} \rangle = \langle \mathbf{a}, \, \mathbf{c} \rangle + \langle \mathbf{b}, \, \mathbf{c} \rangle, \\ \langle \alpha \mathbf{a}, \, \mathbf{b} \rangle = \bar{\alpha} \langle \mathbf{a}, \, \mathbf{b} \rangle. $$

Данные равенства получаются из последовательного применения первых двух аксиом, а также свойств сопряжения:

$$ \langle \mathbf{a} + \mathbf{b}, \, \mathbf{c} \rangle = \overline{ \langle \mathbf{c}, \, \mathbf{a} + \mathbf{b} \rangle } = \overline{ \langle \mathbf{c}, \, \mathbf{a} \rangle + \langle \mathbf{c}, \, \mathbf{b} \rangle } = \overline{ \langle \mathbf{c}, \, \mathbf{a} \rangle } + \overline{ \langle \mathbf{c}, \, \mathbf{b} \rangle } = \langle \mathbf{a}, \, \mathbf{c} \rangle + \langle \mathbf{b}, \, \mathbf{c} \rangle, \\
\langle \alpha \mathbf{a}, \, \mathbf{b} \rangle = \overline{\langle \mathbf{b}, \, \alpha \mathbf{a} \rangle} = \overline{ \alpha \langle \mathbf{b}, \, \mathbf{a} \rangle } = \bar{\alpha} \overline{ \langle \mathbf{b}, \, \mathbf{a} \rangle } = \bar{\alpha} \langle \mathbf{a}, \, \mathbf{b} \rangle. $$

Теперь подробнее остановимся на том, как можно вычислить эрмитово скалярное произведение двух векторов с элементами $\mathbf{x} = \begin{bmatrix} x_1, \, x_2, \, \ldots, \, x_n \end{bmatrix}^\top$ и $\mathbf{y} = \begin{bmatrix} y_1, \, y_2, \, \ldots, \, y_n \end{bmatrix}^\top$, заданных в базисе $\mathrm{B} = \begin{Bmatrix} \mathbf{v}_1, \, \mathbf{v}_2, \, \ldots, \, \mathbf{v}_m \end{Bmatrix}$.

$$ \begin{align}
\langle \mathbf{x}, \, \mathbf{y} \rangle
&= \langle \sum_{i=1}^n x_i \mathbf{v}_i, \, \sum_{j=1}^n y_j \mathbf{v}_j \rangle && \text{разложение векторов по базису } \mathrm{B} \\
&= \sum_{j=1}^n \langle \sum_{i=1}^n x_i \mathbf{v}_i, \, y_j \mathbf{v}_j \rangle && \text{первое свойство аксиомы о линейности по второму аргументу} \\
&= \sum_{j=1}^n y_j \langle \sum_{i=1}^n x_i \mathbf{v}_i, \, \mathbf{v}_j \rangle && \text{второе свойство аксиомы о линейности по второму аргументу} \\
&= \sum_{j=1}^n y_j \left( \sum_{i=1}^n \langle x_i \mathbf{v}_i, \, \mathbf{v}_j \rangle \right) && \text{первое свойство следствия о полулинейности по первому аргументу} \\
&= \sum_{j=1}^n y_j \left( \sum_{i=1}^n \bar{x}_i \langle \mathbf{v}_i, \, \mathbf{v}_j \rangle \right) && \text{второе свойство следствия о полулинейности по первому аргументу} \\
&= \sum_{i=1}^n \sum_{j=1}^n \bar{x}_i \langle \mathbf{v}_i, \, \mathbf{v}_j \rangle y_j. && \text{перегруппировка}
\end{align} $$

Обозначим результат эрмитового скалярного произведения $i$-го и $j$-го базисных векторов $g_{ij} = \langle \mathbf{v}_i, \, \mathbf{v}_j \rangle$ (соответствующий элементу [матрицы Грама](https://en.wikipedia.org/wiki/Gram_matrix), находящегося на пересечении $i$-й строчки и $j$-го столбца). Тогда:

$$ \langle \mathbf{x}, \, \mathbf{y} \rangle = \sum_{i=1}^n \sum_{j=1}^n \bar{x}_i g_{ij} y_j. $$

Введем понятие *стандартного эрмитового скалярного произведения*, удовлетворяющего тем же самым аксиомам, что и эрмитово скалярное произведение.

```{admonition} Определение
:class: definition
***Стандартным эрмитовым скалярным произведением*** двух векторов $\mathbf{x}, \, \mathbf{y} \in \mathbb{C}^n$ будем называть число, равное сумме произведений сопряженных элементов первого вектора на соответствующие элементы второго вектора:

$$ \mathbf{x}^\dagger \mathbf{y} = \sum_{i=1}^n \bar{x}_i y_i. $$
```

Операция комплексно-сопряженного транспонирования вектора $\mathbf{x}$, обозначаемая, как $\mathbf{x}^\dagger$ (также в научной литературе встречаются обозначения $\mathbf{x}^*$ или $\mathbf{x}^H$), по сути, состоит из двух действий: сопряжения всех элементов вектора $\mathbf{x}$, то есть замены всех элементов на сопряженные, и его транспонирования (преобразования из вектора-столбца в вектор-строку).

То есть стандартное эрмитово скалярное произведение является частным случаем эрмитового скалярного произведения, для которого элементы $g_{ij} = \delta_{ij}, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n,$ где $\delta_{ij} = 1 \, i = j,$ и $\delta_{ij} = 0, \, i \neq j,$ представляет собой [дельту Кронекера](https://en.wikipedia.org/wiki/Kronecker_delta). Важно отметить, что если оба вектора принадлежат пространству действительных чисел, то стандратное эрмитово скалярное произведение сводится к *стандартному скалярному произведению* этих векторов вследствие вещественности всех элементов $\left( \bar{x}_i = x_i \Leftrightarrow x_i \in \mathbb{R}, \, i = 1 \, \ldots \, n \right)$:

$$ \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i. $$

<a id='bilinear-form'></a>
Данную алгебраическую операцию, $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i$, также можно выполнить и с комплексными векторами. В этом случае, это действие будем называть *стандартной симметричной билинейной формой*, чтобы отличать его от стандартного эрмитового скалярного произведения. Данная операция характеризуется симметричностью, линейностью по обоим аргументам (билинейностью), но не обладает свойством положительной полуопределенности.

Введем определение *стандартного (канонического) базиса* в $n$-мерном числовом пространстве.

<a id='standard-basis'></a>
```{admonition} Определение
:class: definition
Стандартным (каноническим) базисом $\mathrm{E}$ в $n$-мерном пространстве будем называть упорядоченную совокупность векторов $\begin{Bmatrix} \mathbf{e}_1, \, \mathbf{e}_2, \, \ldots, \, \mathbf{e}_n \end{Bmatrix}$, у которых один из элементов равен единице, а остальные — нулю:

$$ \begin{array}{c}
\mathbf{e}_1 = \begin{bmatrix} 1, \, 0, \, \ldots, \, 0 \end{bmatrix}^\top, \\
\mathbf{e}_2 = \begin{bmatrix} 0, \, 1, \, \ldots, \, 0 \end{bmatrix}^\top, \\
\vdots \\
\mathbf{e}_n = \begin{bmatrix} 0, \, 0, \, \ldots, \, 1 \end{bmatrix}^\top.
\end{array} $$

При этом в пространстве, определяемым данным базисом, эрмитово скалярное произведение аксиоматически соответствует стандартному эрмитовому скалярному произведению (данное пространство наделяется стандартным эрмитовым скалярным произведением).
```

Итак, после введения стандартного эрмитового скалярного произведения и стандартного (канонического) базиса вернемся к определению эрмитового скалярного произведения для произвольного базиса, зависящего от результата эрмитового скалярного произведения базисных векторов:

$$ \langle \mathbf{x}, \, \mathbf{y} \rangle = \sum_{i=1}^n \sum_{j=1}^n \bar{x}_i g_{ij} y_j. $$

Существует два способа задания значений $g_{ij}, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n$:

* *конструктивный (координатный, внешний):* координаты базисных векторов даны в стандартном базисе, который аксиоматически наделен стандартным эрмитовым скалярным произведением, то есть $g_{ij} = \langle \mathbf{v}_i, \, \mathbf{v}_j \rangle = \mathbf{v}_i^\dagger \mathbf{v}_j, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n$;
* *аксиоматический (бескоординатный, внутренний):* элементы $g_{ij}, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n$ даны напрямую, то есть координаты базисных векторов в "удобном" базисе неизвестны, но зато известно само правило эрмитового скалярного произведения.

Рассмотрим эрмитово скалярное произведение векторов на следующем примере.

```{admonition} Пример
:class: exercise
Пусть координаты векторов $\mathbf{a}, \, \mathbf{b} \in \mathbb{C}^2$ определены в стандартном базисе следующим образом:

$$ \mathbf{a} = \begin{bmatrix} 3 + 2 i \\ 2 - i \end{bmatrix}, \; \mathbf{b} = \begin{bmatrix} 4 - 3 i \\ 2 + 2 i \end{bmatrix}. $$

Необходимо вычислить эрмитово скалярное произведение.
```

````{dropdown} Решение
Зададим координаты векторов $\mathbf{a}$ и $\mathbf{b}$ в виде одномерных массивов:

```python
a = np.array([3 + 2j, 2 - 1j])
b = np.array([4 - 3j, 2 + 2j])
```

Поскольку координаты векторов даны в стандартном базисе, то эрмитового скалярное произведение сводится к стандартному эрмитовому скалярному произведению: $\langle \mathbf{a}, \, \mathbf{b} \rangle = \mathbf{a}^\dagger \mathbf{b}$.

Вычисление стандартного эрмитового скалярного произведения векторов можно осуществить несколькими способами. Первый из них основан на преобразовании первого вектора в его сопряженный вектор с использованием метода [`conjugate`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conjugate.html) и вычислении суммы произведений их элементов методом [`dot`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html):

```python
r = a.conjugate().dot(b)
print(r)
```

```{glue:} dot1
```

Второй способ основан на использовании функции [`numpy.vdot`](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html), которая автоматически преобразует первый вектор в его сопряженный:

```python
t = np.vdot(a, b)
print(t)
```

```{glue:} dot2
```

Если векторы принадлежат пространству действительных чисел, то операция преобразования первого вектора в его сопряженный опускается, и стандартное скалярное произведение вычисляется с использованием метода [`dot`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html):

```python
a = np.array([3., 2.])
b = np.array([4., 2.])
p = a.dot(b)
print(p)
```

```{glue:} dot3
```
````

```{code-cell} python
:tags: [remove-cell]

a = np.array([3 + 2j, 2 - 1j])
b = np.array([4 - 3j, 2 + 2j])

r = a.conjugate().dot(b)

glue('dot1', MultilineText(r))

t = np.vdot(a, b)

glue('dot2', MultilineText(t))

a = np.array([3., 2.])
b = np.array([4., 2.])
p = a.dot(b)

glue('dot3', MultilineText(p))
```

Введя в рассмотрение эрмитово скалярное произведение, мы можем определить понятие ортогональных векторов.

<a id='orthogonal'></a>
```{admonition} Определение
:class: definition
***Ортогональными векторами*** $\mathbf{a}, \, \mathbf{b} \in \mathbb{C}^n$ будем называть векторы, эрмитово скалярное произведение которых равняется нулю:

$$ \langle \mathbf{a}, \, \mathbf{b} \rangle = 0. $$
```

Аналогично для векторов над пространством действительных чисел ортогональность определяется из условия равенства нулю скалярного произведения этих векторов: $\mathbf{a} \perp \mathbf{b} \Leftrightarrow \mathbf{a}^\top \mathbf{b} = 0, \, \mathbf{a}, \, \mathbf{b} \in \mathbb{R}^n$.

Следует отличать эрмитово скалярное произведение от поэлементного, обозначение которого в рамках данного курса будет осуществляться следующим образом:

```{admonition} Определение
:class: definition
***Поэлементным произведением*** двух векторов $\mathbf{a}, \, \mathbf{b} \in \mathbb{C}^n$ называется вектор $\mathbf{c} \in \mathbb{C}^n$, элементы которого вычисляются путем умножения соответствующих элементов векторов $\mathbf{a}$ и $\mathbf{b}$:

$$ \mathbf{c} = \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} \cdot \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} = \begin{bmatrix} a_1 \cdot b_1 \\ a_2 \cdot b_2 \\ \vdots \\ a_n \cdot b_n \end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{bmatrix}. $$
```

Поэлементное произведение векторов также иногда называют [произведением Адамара](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)). Оно обладает следующими свойствами:

* *коммутативностью*: $\mathbf{a} \mathbf{b} = \mathbf{b} \mathbf{a}$,
* *дистрибутивностью*: $\mathbf{a} \left( \mathbf{b} + \mathbf{c} \right) = \mathbf{a} \mathbf{b} + \mathbf{a} \mathbf{c}$,
* *ассоциативностью*: $\left( \mathbf{a} \mathbf{b} \right) \mathbf{c} = \mathbf{a} \left( \mathbf{b} \mathbf{c} \right)$.

```{admonition} Пример
:class: exercise
Пусть векторы $\mathbf{a}, \, \mathbf{b} \in \mathbb{R}^2$ определены следующим образом:

$$ \mathbf{a} = \begin{bmatrix} 1 + 2 i \\ 2 - i \end{bmatrix}, \; \mathbf{b} = \begin{bmatrix} 4 - 3 i \\ 2 + 2 i \end{bmatrix}. $$

Необходимо вычислить поэлементное произведение $\mathbf{a} \cdot \mathbf{b}$.
```

````{dropdown} Решение
Зададим координаты векторов $\mathbf{a}$ и $\mathbf{b}$ в виде одномерных массивов:

```python
a = np.array([1 + 2j, 2 - 1j])
b = np.array([4 - 3j, 2 + 2j])
```

Выполним поэлементное произведение:

```python
c = a * b
print(c)
```

```{glue:} prod1
```

Аналогично можно вычислить и поэлементное деление векторов:

```python
d = c / a
print(d)
```

```{glue:} prod2
```

А также поэлементное возведение в степень:

```python
p = a ** 2
print(p)
```

```{glue:} prod3
```
````

```{code-cell} python
:tags: [remove-cell]

a = np.array([1 + 2j, 2 - 1j])
b = np.array([4 - 3j, 2 + 2j])

c = a * b

glue('prod1', MultilineText(c))

d = c / a

glue('prod2', MultilineText(d))

p = a ** 2

glue('prod3', MultilineText(p))
```

(vector-length)=
## Длина вектора

```{admonition} Определение
:class: definition
Под ***длиной вектора*** $\mathbf{a} \in \mathbb{C}^n$ понимается число $\lVert \mathbf{a} \rVert_2 \in \mathbb{R}$, равное:

$$ \lVert \mathbf{a} \rVert_2 = \sqrt{\langle \mathbf{a}, \, \mathbf{a} \rangle} = \sum_{i=1}^n \sum_{j=1}^n a_i g_{ij} a_j. $$
```

Если координаты вектора в стандартном базисе состоят из действительных чисел, то его длина равняется квадратному корню из стандартного скалярного произведения вектора на самого себя:

$$ \lVert \mathbf{a} \rVert_2 = \sqrt{\mathbf{a}^\top \mathbf{a}} = \sqrt{\sum_{i=1}^n a_i^2}, \; \mathbf{a} = \begin{bmatrix} \mathbf{a} \end{bmatrix}_\mathrm{E} \in \mathbb{R}^n. $$

По аналогии, длину вектора, комплексные координаты которого даны в стандартном базисе, можно вычислить путем извлечения корня из результата стандартного эрмитового скалярного произведения вектора на самого себя:

$$ \lVert \mathbf{a} \rVert_2 = \sqrt{\mathbf{a}^\dagger \mathbf{a}} = \sqrt{\sum_{i=1}^n \bar{a}_i a_i} = \sqrt{\sum_{i=1}^n \left| a_i \right|^2}, \; \mathbf{a} = \begin{bmatrix} \mathbf{a} \end{bmatrix}_\mathrm{E} \in \mathbb{C}^n. $$

Именно для того чтобы длина комплексного вектора являлась вещественным числом, эрмитово скалярное произведение формулируется с учетом сопряжения первого аргумента.

<!-- Отметим, что выражение $\lVert \mathbf{a} \rVert_2 = \sum_{i=1}^n \sum_{j=1}^n a_i g_{ij} a_j$ является *инвариантным* по отношению к смене базиса. То есть изменение базиса будет приводить как к изменению координат вектора $a_i, \, i = 1 \, \ldots \, n,$ так и к изменению элементов $g_{ij}, \, i = 1 \, \ldots \, n, \, j = 1 \, \ldots \, n$. Причем длина вектора не будет меняться. Данное утверждение будет доказано в разделе, посвященном [линейным преобразованиям](LAB-3-LinearTransformations.md). -->

Рассмотрим нахождение длины вектора на следующем примере.

```{admonition} Пример
:class: exercise

Пусть координаты вектора $\mathbf{a} \in \mathbb{C}^2$ заданы следующим образом:

$$ \mathbf{a} = \begin{bmatrix} 2 + i \\ 2 i \end{bmatrix}. $$

Необходимо найти длину данного вектора.
```

````{dropdown} Решение
Зададим координаты вектора в виде одномерного массива:

```python
a = np.array([2 + 1j, 2j])
```

Вычислим длину вектора через эрмитово скалярное произведение вектора на самого себя:

```python
l = np.sqrt(np.vdot(a, a))
print(l)
```

```{glue:} glued_text30
```

Для расчета квадратного корня использовалась функция [`numpy.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html). Определение длины вектора можно осуществить с использованием [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html):

```python
l = np.linalg.norm(a)
print(l)
```

```{glue:} glued_text31
```
````

```{code-cell} python
:tags: [remove-cell]

a = np.array([2 + 1j, 2j])
l = np.sqrt(np.vdot(a, a))

glue('glued_text30', MultilineText(l))

l = np.linalg.norm(a)

glue('glued_text31', MultilineText(l))
```

Следует отметить, что если длина каждого из множества ортогональных векторов равна единице, то есть векторы являются единичными, то базис, построенный на таких вектораз называется *ортонормированным*. Примером ортонормированного базиса является стандартный (канонический) базис, введенный ранее в этом разделе.

<a id='vector-norm'></a>
````{dropdown} Норма вектора
***Норма вектора*** – это алгебраическое обобщение понятия длины вектора, координаты которого заданы в ортонормированном базисе, например, в стандартном: $\mathbf{a} = \begin{bmatrix} \mathbf{a} \end{bmatrix}_\mathrm{E} \in \mathbb{C}^n$, и определяемое следующим выражением:

$$ \lVert \mathbf{a} \rVert_p = \left( \sum_{i=1}^n \left| a_i \right|^p \right)^{\frac{1}{p}}. $$

Соответственно длину вектора в ортонормированном базисе зачастую называют $L_2$-нормой (или Евклидовой нормой):

$$ \lVert \mathbf{a} \rVert_2 = \left( \sum_{i=1}^n \left| a_i \right|^2 \right)^{\frac{1}{2}} = \sqrt{ \sum_{i=1}^n \left| a_i \right|^2}, \; \mathbf{a} = \begin{bmatrix} \mathbf{a} \end{bmatrix}_\mathrm{E} \in \mathbb{C}^n. $$

Сушествует также $L_1$-норма (или Манхэттенская норма), численно равная сумме модулей всех элементов вектора:

$$ \lVert \mathbf{a} \rVert_1 = \sum_{i=1}^n \left| a_i \right|. $$

Кроме того, необходимо отметить норму Чебышёва или $L_\infty$-норму:

$$ \lVert \mathbf{a} \rVert_\infty = \lim_{p \rightarrow \infty} \left( \sum_{i=1}^n \left| a_i \right|^p \right)^{\frac{1}{p}} = \lim_{p \rightarrow \infty} \left( \left| a_{m} \right|^p \sum_{i=1}^n \left| \frac{a_i}{a_m} \right|^p \right)^{\frac{1}{p}}, $$

где $a_m$ обозначен максимальный по модулю элемент вектора $\mathbf{a}$: $a_m = \max_i \left| a_i \right|$. Пусть существует $1 \leq k \leq n$ элементов вектора $\mathbf{a}$, соответствующих его максимальному по модулю значению. Тогда

$$ \begin{align}
\lVert \mathbf{a} \rVert_\infty
&= \lim_{p \rightarrow \infty} \left( \left| a_{m} \right|^p \sum_{i=1}^n \left| \frac{a_i}{a_m} \right|^p \right)^{\frac{1}{p}} \\
&= \lim_{p \rightarrow \infty} \left( \left| a_{m} \right|^p \sum_{i=1}^k \left| 1 \right|^p \right)^{\frac{1}{p}} \\
&= \lim_{p \rightarrow \infty} \left( \left| a_{m} \right|^p k \right)^{\frac{1}{p}} \\
&= \left| a_{m} \right| \lim_{p \rightarrow \infty} k^{\frac{1}{p}} \\
&= \left| a_{m} \right| \\
&= \max_i \left| a_i \right|.
\end{align} $$

Таким образом, $L_\infty$-норма вектора $\mathbf{a}$ равняется наибольшему по модулю элементу вектора:

$$ \lVert \mathbf{a} \rVert_\infty = \max_i \left| a_i \right|. $$

Отметим следующие свойства нормы вектора:

* *неотрицательность*:

$$ \begin{align}
& \lVert \mathbf{a} \rVert_p \geq 0, \\
& \lVert \mathbf{a} \rVert_p = 0 \Leftrightarrow \mathbf{a} = 0, \; \forall \; p \in \mathbb{N},
\end{align} $$

* *однородность*:

$$ \lVert \alpha \mathbf{a} \rVert_p = \left| \alpha \right| \lVert \mathbf{a} \rVert_p, \; \forall \; p \in \mathbb{N}, $$

* *правило треугольника*:

$$ \lVert \mathbf{a} + \mathbf{b} \rVert_p \leq \lVert \mathbf{a} \rVert_p + \lVert \mathbf{b} \rVert_p, \; \forall \; p \in \mathbb{N}. $$

Для вычисления различных норм массивов можно использовать функцию [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html), в которой параметр нормы выбирается с использованием дополнительного аргумента `ord`.
````

