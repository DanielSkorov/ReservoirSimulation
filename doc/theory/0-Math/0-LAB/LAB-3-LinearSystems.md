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

# Решение систем линейных уравнений

Допустим, имеется следующая система линейных уравнений:

$$ \left\{\begin{array} \\ 3x + 2y + z = 5 \\ 2x + 3y + z = -1 \\ 2x + y + 3z = 3 \end{array}\right. $$

Левую часть уравнений в данной системе можно представить в виде произведения матрицы и вектора искомых переменных:

$$ \mathbf{A} \mathbf{x}, $$

где:

$$ \mathbf{A} = \begin{bmatrix} 3 & 2 & 1 \\ 2 & 3 & 1 \\ 2 & 1 & 3 \end{bmatrix}, \; \mathbf{x} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}. $$

Тогда в результате произведения матрицы $\mathbf{A}$ и вектора $\mathbf{x}$ получится вектор:

$$ \mathbf{A} \mathbf{x} = \begin{bmatrix} 3x + 2y + z \\ 2x + 3y + z \\ 2x + y + 3z \end{bmatrix}. $$

Правые части уравнений также можно представить в виде матрицы-столбца $\mathbf{b}$:

$$ \mathbf{b} = \begin{bmatrix} 5 \\ -1 \\ 3 \end{bmatrix}. $$

Тогда исходную систему линейных уравнений можно представить в виде матричного уравнения:

$$ \mathbf{A} \mathbf{x} = \mathbf{b}. $$

Домножим обе части уравнения на $\mathbf{A}^{-1}$:

$$ \mathbf{A}^{-1} \mathbf{A} \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}. $$

Левую часть уравнения можно упростить, применив определение обратной матрицы:

$$ \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}. $$

```{code-cell} python
import numpy as np
A = np.array([[3., 2., 1.], [2., 3., 1.], [2., 1., 3.]])
b = np.array([5., -1., 3.])
x = np.linalg.inv(A).dot(b)
print(x)
```

Проверка:

```{code-cell} python
print(A.dot(x))
```

Однако в практике численного моделирования для небольших по размерности систем линейных уравнений вместо нахождения обратных матриц используют метод Гаусса, заключающийся в последовательном исключении переменных, что, в конечном счете, позволяет преобразовать матрицу к степенчатому виду.

(ls-gauss)=
## Метод Гаусса

(ls-lu)=
## LU-разложение

(ls-cholesky)=
## Разложение Шолески

(ls-mcholesky)=
## Модифицированное разложение Шолески
