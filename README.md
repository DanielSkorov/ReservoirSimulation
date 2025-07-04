# Basics of Reservoir Simulation / Основы моделирования пластовых систем

This repository contains the source files for the course of reservoir simulation basics. This course studies the process of developing the customised multidimensional, equation-of-state (EOS), compositional reservoir simulator: from basics of the linear algebra and thermodynamics to the solution of volume balance, heat and mass transfer equations in the discretized space and time. This course is mostly suited for studying the physics under the hood of any compositional reservoir simulator rather than learning high-performance computing. Nevertheless, this course does contain examples of numerical algorithms written in Python and Fortran.

Данная библиотека содержит исходники для курса по основам моделирования пластовых систем, посвященного изучению этапов создания композиционного гидродинамического симулятора: от основ линейной алгебры и термодинамики до разрешения системы нелинейных уравнений тепло- и массопереноса, а также баланса объемов для дискретизированных порового пространства и времени. Данный курс в большей степени подходит для изучения физико-химических основ любого композиционного гидродинамического симулятора, чем для рассмотрения высокоэффективных алгоритмов. Тем не менее, курс содержит примеры реализации различных численных методов, написанных на языках программирования Python и Fortran.

## Course materials / Материалы курса

Course materials are available at the following [link](https://danielskorov.github.io/ReservoirSimulation/). For now, the course materials have been written in russian, but are going to be translated. At this moment, the course is under development. Contributions are welcome in the form of *Pull requests*.

Материалы практикума доступны по [ссылке](https://danielskorov.github.io/ReservoirSimulation/). В данный момент курс находится в разработке. Конструктивные предложения и доработки приветствуются в форме *Pull requests*.

## Contributing / Участие

You are welcome to contribute. You are also welcome to open an issue if you find a typo or mistake. Besides, you are welcome to ask a question or start a discussion using the GitHub's Discussions.

For making changes to the text contents or to the code examples, please create a separete branch and then edit the `.md` files using either the GitHub interface or a text editor. Then you will need to install some extra dependencies for rendering edited files to `.html`-pages. They are listed in the `dependencies.txt` file. The course materials are built with [Jupyter Book](https://github.com/jupyter-book/jupyter-book) with some custom modifications (in the `_static/` folder). The `gh-pages` brach contains rendered `.md` files as `.html`-pages. The source code of algorithms is collected in the `_src/` folder.

Внесение конструктивных (и не только) изменений приветствуется. Если Вы нашли ошибку или опечатку, можете создать *issue*. Кроме того, Вы можете задать вопрос или начать обсуждение с использованием GitHub's Discussions.

Для внесения изменений в материалы курса, пожалуйста, создайте отдельную ветку. После этого Вы можете изменить `.md`-файлы с использованием интерфейса GitHub или текстового редактора. Для того чтобы преобразовать Ваши изменения в страницы сайта, необходимо установить ряд библиотек. Они перечислены в файле `dependencies.txt`. Сайт с материалами курса создан с использованием [Jupyter Book](https://github.com/jupyter-book/jupyter-book) с небольшими модификациями, находящимися в папке `_static/`. Ветка `gh-pages` содержит преобразованные `.md`-файлы в виде `.html`-страниц. Исходный код алгоритмов находится в папке `_src/`.

## License / Лицензия

The code portions of this material are licensed under the [BSD 3-Clause License](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE.txt) (described in LICENSE.txt) while the non-code portions of the material are licensed under the Creative Commons Attribution License ([CC-BY-4.0](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE-text.txt), described in LICENSE-text.txt).

Программный код данной работы лицензирован по лицензии [BSD 3-Clause License](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE.txt) (файл LICENSE.txt), а остальная часть работы – по лицензии Creative Commons Attribution License ([CC-BY-4.0](https://github.com/DanielSkorov/ReservoirSimulation/blob/main/LICENSE-text.txt), файл LICENSE-text.txt).
