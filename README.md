# Basics of Reservoir Simulation / Основы моделирования пластовых систем

This repository contains the source files for the course of reservoir simulation basics. This course studies the process of developing the customised multidimensional, equation-of-state (EOS), compositional reservoir simulator: from basics of the linear algebra and thermodynamics to the solution of volume balance, heat and mass transfer equations in the discretized space and time. This course is mostly suited for studying the physics under the hood of any compositional reservoir simulator rather than learning high-performance computing. Nevertheless, this course does contain examples of numerical algorithms written in Python and Fortran.

Данная библиотека содержит исходники для курса по основам моделирования пластовых систем, посвященного изучению этапов создания композиционного гидродинамического симулятора: от основ линейной алгебры и термодинамики до разрешения системы нелинейных уравнений тепло- и массопереноса, а также баланса объемов для дискретизированных порового пространства и времени. Данный курс в большей степени подходит для изучения физико-химических основ любого композиционного гидродинамического симулятора, чем для рассмотрения высокоэффективных алгоритмов. Тем не менее, курс содержит примеры реализации различных численных методов, написанных на языках программирования Python и Fortran.

## Course materials / Материалы курса

Course materials are available at the following [link](https://danielskorov.github.io/ReservoirSimulation/). For now, the course materials have been written in russian, but are going to be translated. At this moment, the course is under development. Contributions are welcome in the form of *Pull requests*.

Материалы практикума доступны по [ссылке](https://danielskorov.github.io/ReservoirSimulation/). В данный момент курс находится в разработке. Конструктивные предложения и доработки приветствуются в форме *Pull requests*.

## Contributing / Участие

You are welcome to contribute. You are also welcome to open an issue if you find a typo or mistake. Besides, you are welcome to ask a question or start a discussion using the *GitHub's Discussions*.

For making changes to the text contents or to the code examples, please create a separete branch and then edit the `.md` files using either the GitHub interface or a text editor. Then you will need to install some extra dependencies for rendering edited files to `.html`-pages. They are listed in the `requirements.txt` file. The course materials are built with [Jupyter Book](https://github.com/jupyter-book/jupyter-book) with some custom modifications (in the `_static/` folder). The `gh-pages` brach contains rendered `.md` files as `.html`-pages. The source code of algorithms is collected in the `_src/` folder.

In order to call Fortran *subroutines* in Python files, one can use [`numpy.f2py`](https://numpy.org/doc/stable/f2py/). This `numpy` extension enables the automatic construction and compilation of an extension module that can be imported into `py`-code. The [`meson`](https://github.com/mesonbuild/meson) and [`ninja`](https://github.com/ninja-build/ninja) libraries are necessary to run `numpy.f2py`. To compile a Fortran code, [`gfortran`](https://gcc.gnu.org/fortran/) can be used. Installation guides for Linux are comparatively easy and were described on the [fortran-lang.org](https://fortran-lang.org/ru/learn/os_setup/install_gfortran/#linux). For Windows everything is a little more complicated. To install `gfortran` for Windows, one can use the [`mingw-w64`](https://www.mingw-w64.org/) installation guide as it was recommended by the official manual of [`numpy.f2py`](https://numpy.org/devdocs/f2py/windows/index.html). Another approach is based on the libraries provided by [`winlibs`](https://www.winlibs.com/). To resolve an error related to the absence of `dll`-libraries when importing compiled Fortran code with `numpy.f2py`, one can refer to [this post](https://github.com/numpy/numpy/issues/28151#issuecomment-2720506610).

Внесение конструктивных (и не только) изменений приветствуется. Если Вы нашли ошибку или опечатку, можете создать *issue*. Кроме того, Вы можете задать вопрос или начать обсуждение с использованием *GitHub's Discussions*.

Для внесения изменений в материалы курса, пожалуйста, создайте отдельную ветку. После этого Вы можете изменить `.md`-файлы с использованием интерфейса GitHub или текстового редактора. Для того чтобы преобразовать Ваши изменения в страницы сайта, необходимо установить ряд библиотек. Они перечислены в файле `requirements.txt`. Сайт с материалами курса создан с использованием [Jupyter Book](https://github.com/jupyter-book/jupyter-book) с небольшими модификациями, находящимися в папке `_static/`. Ветка `gh-pages` содержит преобразованные `.md`-файлы в виде `.html`-страниц. Исходный код алгоритмов находится в папке `_src/`.

Для вызова подпрограмм (*subroutine*), написанных на Fortran, внутри исполняемых Python файлов можно использовать [`numpy.f2py`](https://numpy.org/doc/stable/f2py/). Данное расширение `numpy` позволяет автоматически компилировать и создавать расширение, которое можно импортировать в `py`-код. Для использования `f2py` также потребуется установить библиотеки [`meson`](https://github.com/mesonbuild/meson) и [`ninja`](https://github.com/ninja-build/ninja). Для компилирования кода, написанного на Fortran, можно использовать, например, [`gfortran`](https://gcc.gnu.org/fortran/). Процесс его установки в Linux достаточно прост и изложен на [fortran-lang.org](https://fortran-lang.org/ru/learn/os_setup/install_gfortran/#linux). Порядок действий для Windows несколько сложнее и зачастую основан на [`mingw-w64`](https://www.mingw-w64.org/). Для установки `gfortran` на Windows можно следовать порядку действий, представленному на официальном ресурсе [`numpy.f2py`](https://numpy.org/devdocs/f2py/windows/index.html). С другой стороны, необходимые библиотеки для компилирования можно скачать на странице [`winlibs`](https://www.winlibs.com/). Для устанения ошибки, связанной с отсутствием `dll`-библиотек, при импорте скомпилированного расширения с использованием `numpy.f2py` рекомендуется обратиться к [данным рекомендациям](https://github.com/numpy/numpy/issues/28151#issuecomment-2720506610).

## Examples / Примеры

This library allows to run the stability test, flash calculations, determine the saturation pressure or temperature, cricondenbar or cricondentherm, and construct the phase envelope for a multicomponent mixture based on the PT-thermodynamics. It also provides the function to calculate the critical point for the VT-thermodynamics.

Using the library, one can simulate some laboratory experiments such as the Constant Volume Depletion (CVD) and the Constant Composition Expansion (CCE).

For now, only the modified Peng-Robinson equation of state is supported. In the future, the [CPA](https://doi.org/10.1021/ie9600203) and [ePC-SAFT](https://doi.org/10.1016/j.fluid.2021.112967) equations of state will be added.

The usage examples of the code can be found in the `_tests` folder.

Библиотека содержит реализации (для PT-термодинамики) теста стабильности фазового состояния, расчета равновесного состояния, определения давления и температуры насыщения, криконденбары и крикондентермы, построения границы двухфазной области многокомпонентной системы, а также для VT-термодинамики – расчет критической точки.

С использованием данной библиотеки также можно моделировать некоторые лабораторные эксперименты, например, контактную (CCE) и контактно-дифференциальную (CVD) конденсации.

На данный момент поддержано только модифицированное уравнение состояние Пенга-Робинсона. В дальнейшие планы входит добавление уравнений состояния [CPA](https://doi.org/10.1021/ie9600203) и [ePC-SAFT](https://doi.org/10.1016/j.fluid.2021.112967).

Применение имеющегося функционала проиллюстрировано примерами, представленными в папке `_tests`.

## License / Лицензия

The code portions of this material are licensed under the [BSD 3-Clause License](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE.txt) (described in LICENSE.txt) while the non-code portions of the material are licensed under the Creative Commons Attribution License ([CC-BY-4.0](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE-text.txt), described in LICENSE-text.txt).

Программный код данной работы лицензирован по лицензии [BSD 3-Clause License](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE.txt) (файл LICENSE.txt), а остальная часть работы – по лицензии Creative Commons Attribution License ([CC-BY-4.0](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE-text.txt), файл LICENSE-text.txt).
