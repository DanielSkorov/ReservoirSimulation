---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<a id='pvt'></a>
# PVT-моделирование

+++

Изучение вопросов PVT-моделирования позволяет понять, как постоянно изменяющиеся условия (термобарические, химические и т.д.) в процессе разработки залежей углеводородов влияют на свойства флюидов. Понимание взаимовлияния параметров многокомпонентных систем необходимо для моделирования их течения в поровом (и не только) пространстве. Следовательно, рассмотрение вопросов гидродинамического моделирования рекомендуется начать с изучения PVT-моделирования.

+++

**Содержание:**
* ***[Основы термодинамики](./1-TD/TD-0-Introduction.md)***
    * [Введение в термодинамику](./1-TD/TD-1-Basics.md)
    * [Термическое равновесие](./1-TD/TD-2-ThermalEquilibrium.md)
    * [Количество теплоты и работа](./1-TD/TD-3-Heat-Work.md)
    * [Теплоемкость](./1-TD/TD-4-HeatCapacity.md)
    * [Энтальпия](./1-TD/TD-5-Enthalpy.md)
    * [Энтропия](./1-TD/TD-6-Entropy.md)
    * [Химический потенциал](./1-TD/TD-7-ChemicalPotential.md)
    * [Энергия Гельмгольца. Энергия Гиббса](./1-TD/TD-8-Helmholtz-Gibbs.md)
    * [Интенсивные и экстенсивные параметры](./1-TD/TD-9-Observables.md)
    * [Энергия Гиббса многокомпонентной системы](./1-TD/TD-10-MixtureGibbsEnergy.md)
    * [Правило фаз Гиббса](./1-TD/TD-11-GibbsPhaseRule.md)
    * [Уравнение Гиббса-Дюгема](./1-TD/TD-12-GibbsDuhemEquation.md)
    * [Соотношения Максвелла](./1-TD/TD-13-MaxwellRelations.md)
    * [Свободная энергия и фазовое равновесие](./1-TD/TD-14-PhaseEquilibrium.md)
    * [Летучесть](./1-TD/TD-15-Fugacity.md)
* ***[Уравнения состояния](./2-EOS/EOS-0-Introduction.md)***
    * [Уравнение состояния Ван-дер-Ваальса](./2-EOS/EOS-1-VanDerWaals.md)
    * [Уравнения состояния Суаве-Редлиха-Квонга и Пенга-Робинсона](./2-EOS/EOS-2-SRK-PR.md)
    * [Уравнение состояния Сорейде-Уитсона](./2-EOS/EOS-3-SW.md)
    * [Теория SAFT](./2-EOS/EOS-4-SAFT.md)
    * [Уравнение состояния PC-SAFT](./2-EOS/EOS-5-PCSAFT.md)
    * [Кубическое уравнение состояния с учетом ассоциации молекул](./2-EOS/EOS-6-CPA.md)
    * [Приложение A. Частные производные летучести компонентов](./2-EOS/EOS-Appendix-A-PD.md)
    * [Приложение B. Универсальный расчет коэффициентов попарного взаимодействия](./2-EOS/EOS-Appendix-B-BIP.md)
* ***[Определение параметров систем с использованием уравнений состояния](./3-Parameters/Parameters-0-Introduction.md)***
    * [Внутренняя энергия](./3-Parameters/Parameters-1-InternalEnergy.md)
    * [Энтальпия](./3-Parameters/Parameters-2-Enthalpy.md)
    * [Энтропия](./3-Parameters/Parameters-3-Entropy.md)
    * [Теплоемкость](./3-Parameters/Parameters-4-HeatCapacity.md)
    * [Химический потенциал](./3-Parameters/Parameters-5-ChemicalPotential.md)
    * [Энергия Гиббса](./3-Parameters/Parameters-6-GibbsEnergy.md)
    * [Энергия Гельмгольца](./3-Parameters/Parameters-7-HelmholtzEnergy.md)
    * [Коэффициент сжимаемости. Коэффициент термического расширения](./3-Parameters/Parameters-8-Compressibility-ThermalExpansion.md)
    * [Коэффициент Джоуля-Томсона](./3-Parameters/Parameters-9-JouleThomsonCoefficient.md)
    * [Приложение A. Частные производные термодинамических параметров](./3-Parameters/Parameters-Appendix-A-PD.md)
* ***[Равновесие. Стабильность. Критическое состояние](./4-ESC/ESC-0-Introduction.md)***
    * [Определение равновесного состояния системы](./4-ESC/ESC-1-Equilibrium.md)
    * [Определение стабильности фазового состояния системы](./4-ESC/ESC-2-Stability.md)
    * [Определение критического состояния системы](./4-ESC/ESC-3-Criticality.md)
