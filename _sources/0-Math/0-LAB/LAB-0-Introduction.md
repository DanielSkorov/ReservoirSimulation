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

(math-lab)=
# Основы линейной алгебры

```{epigraph}
The main thing that I know from my experience of teaching [this class](https://youtube.com/playlist?list=PLXsmhnDvpjORzPelSDs0LSDrfJcqyLlZc) many times to many students from different departments is that linear algebra is the sticking point, and it is a thing that we never get enough of.

-- Dr. Constantine Caramanis
```

Понимание операций над векторами, матрицами является необходимым, поскольку основными смысловыми единицами в моделировании могут выступать величины, представляющие собой векторы (как, например, *компонентный состав* в PVT-моделировании, *скорость течения* в гидродинамике или *вектор напряжения* в геомеханике). В данном разделе собрана основная вспомогательная информация об основах линейной алгебры. Данный раздел не претендует на полноценный курс о линейной алгебре, а является кратким изложением избранных понятий, свойств и теорем, необходимых для дальнейшего изучения основ моделирования пластовых систем.

В данном разделе будут использоваться стандартные библиотеки [*numpy*](https://numpy.org) для проведения операций над векторами и матрицами, а также [*matplotlib*](https://matplotlib.org) для построения графиков и отображения информации.

При составлении данного раздела использовалась информация, изложенная в курсе линейной алгебры от [Khan Academy](https://www.youtube.com/playlist?list=PLFD0EB975BA0CC1E0).

**Содержание:**
* [Вектор. Координаты вектора](LAB-1-Vectors.md)
* [Операции с векторами](LAB-2-VectorOperations.md)
* [Направляющие косинусы](LAB-3-RotationAngles.md)
* [Матрицы](LAB-4-Matrices.md)
* [Определитель матрицы. Обратная матрица](LAB-5-Determinant-InverseMatrix.md)
* [Матричные уравнения](LAB-6-MatrixEquation.md)
* [Собственные векторы и значения матриц](LAB-7-Eigenvalues-Eigenvectors.md)
* [Линейные преобразования](LAB-8-LinearTransformations.md)
