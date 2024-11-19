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

(pvt)=
# PVT-моделирование

Изучение вопросов PVT-моделирования позволяет понять, как постоянно изменяющиеся условия (термобарические, химические и т.д.) в процессе разработки залежей углеводородов влияют на свойства флюидов. Понимание взаимовлияния параметров многокомпонентных систем необходимо для моделирования их течения в поровом (и не только) пространстве. Следовательно, рассмотрение вопросов гидродинамического моделирования рекомендуется начать с изучения PVT-моделирования.

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
    * [Кубическое уравнение состояния с учетом ассоциации молекул](./2-EOS/EOS-3-CPA.md)
    * [Приложение A. Частные производные летучести компонентов](./2-EOS/EOS-Appendix-A-PD.md)
    * [Приложение B. Универсальный расчет коэффициентов попарного взаимодействия](./2-EOS/EOS-Appendix-B-BIP.md)
* ***[Стабильность. Равновесие. Критическое состояние](./3-SEC/SEC-0-Introduction.md)***
    * [Определение стабильности фазового состояния системы](./3-SEC/SEC-1-Stability.md)
    * [Уравнение Речфорда-Райса](./3-SEC/SEC-2-RR.md)
    * [Уравнение Речфорда-Райса для двухфазных систем](./3-SEC/SEC-3-RR-2P.md)
    * [Уравнение Речфорда-Райса для многофазных систем](./3-SEC/SEC-4-RR-NP.md)
    * [Определение равновесного состояния системы](./3-SEC/SEC-5-Equilibrium.md)
    * [Определение насыщенного состояния системы](./3-SEC/SEC-6-Saturation.md)
    * [Определение критического состояния системы](./3-SEC/SEC-7-Criticality.md)
* ***[Моделирование лабораторных исследований](./4-LAB/LAB-0-Introduction.md)***