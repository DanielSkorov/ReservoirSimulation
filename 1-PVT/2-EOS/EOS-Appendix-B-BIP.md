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

(pvt-eos-appendix-bip)=
# Универсальный расчет коэффициентов попарного взаимодействия
В данном приложении изложены ряд подходов, в том числе универсальный (погрупповой \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\]), для расчета коэффициентов попарного взаимодействия при использовании [уравнений состояния Суаве-Редлиха-Квонга, Пенга-Робинсона](EOS-2-SRK-PR.md#pvt-eos-srkpr) и [правил смешивания Ван-дер-Ваальса](EOS-1-VanDerWaals.md#pvt-eos-vdw-mixrules).

Параметр $\delta_{jk}$, коэффициент попарного взаимодействия компонентов, играет важную роль при расчете PVT-свойств флюидов с использованием уравнений состояния. [Ранее](EOS-1-VanDerWaals.md#pvt-eos-vdw-bip) было показано, что коэффициент попарного взаимодействия между углеводородными компонентами можно оценить, используя их критические объемы $V_c$:

$$\delta_{jk} = 1 - \left( \frac{2 {V_c}_j^{\frac{1}{6}} {V_c}_k^{\frac{1}{6}}}{{V_c}_j^{\frac{1}{3}} + {V_c}_k^{\frac{1}{3}}} \right)^c.$$

В данном выражении параметр $c$ подбирается при адаптации модели на лабораторные исследования. В качестве начального значения можно использовать $c = 1.2$, согласно \[[Oellrich et al, 1981](https://api.semanticscholar.org/CorpusID:94056056)\]. Расчет коэффициентов попарного взаимодействия между углеводородными компонентами (и углеводородными гетероатомными, например, с включениями серы – меркаптаны) также рассматривается в работах \[[Slot-Petersen, 1983](https://doi.org/10.2118/16941-PA); [Gao et al, 1992](https://doi.org/10.1016/0378-3812(92)85054-C) [Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059); [Jaubert et al, 2005](https://doi.org/10.1016/j.fluid.2005.09.003); [Vitu et al, 2006](https://doi.org/10.1016/j.fluid.2006.02.004); [Privat et al, 2008](https://doi.org/10.1016/j.jct.2008.05.013); [Privat and Jaubert, 2012](https://doi.org/10.1016/j.fluid.2012.08.007); [Qian et al, 2013](https://doi.org/10.1016/j.fluid.2013.06.040); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\]. Кроме того, коэффициенты попарного взаимодействия исследовались и для уравнения состояния Суаве-Редлиха-Квонга: \[[Mathias, 1983](https://doi.org/10.1021/i200022a008); [Coutinho et al, 1994](https://doi.org/10.1016/0378-3812(94)87090-X)\].

<a id='pvt-eos-appendix-bip-kato'></a>
Для неуглеводородных компонентов коэффициенты попарного взаимодействия можно оценить с использованием корреляций. Например, в работе \[[Kato et al, 1981](https://doi.org/10.1016/0378-3812(81)80009-X)\] приводится корреляция для расчета коэффициента попарного взаимодействия между диоксидом углерода и углеводородными компонентами для уравнения состояния Пенга-Робинсона:

$$ \delta_{ij} = a \cdot \left( T - b \right)^2 + c, $$

где коэффициенты $a, \; b, \; c$ рассчитываются по функции от ацентрического фактора углеводородного компонента:

$$ \begin{align} a &= -0.70421 \cdot 10^{-5} \lg \omega - 0.132 \cdot 10^{-7}; \\ b &= 301.58 \omega + 226.57; \\ c &= -0.0470356 \left( \lg \omega + 1.08884 \right)^2 + 0.13040. \end{align} $$

Данная корреляция получена регрессией для алканового ряда до нормального декана. Также, зависимости коэффициента попарного взаимодействия между диоксидом углерода и углеводородными компонентами исследуются в работах \[[Mulliken and Sandler, 1980](https://doi.org/10.1021/i260076a033); [Turek et al, 1984](https://doi.org/10.2118/9231-PA); [Nishiumi et al, 1988](https://doi.org/10.1016/0378-3812(88)80049-9); [Pedersen et al, 2001](https://doi.org/10.1016/S0378-3812(01)00562-3); [Mutelet et al, 2005](https://doi.org/10.1016/j.fluid.2005.10.001); [Vitu et al, 2008](https://doi.org/10.1016/j.supflu.2007.11.015); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\].

<a id='pvt-eos-appendix-bip-nishumi'></a>
В работе \[[Nishiumi et al, 1988](https://doi.org/10.1016/0378-3812(88)80049-9)\] приводится обобщенная зависимость коэффициентов попарного взаимодействия для углеводородных (алканов до $C_{20}$, алкенов, циклоалканов, ароматических углеводородов) и неуглеводородных компонентов (диоксида углерода, азота и сероводорода) для уравнения состояния Пенга-Робинсона в следующем виде:

$$ m_{ij} = 1 - \delta_{ij} = C + D \frac{{V_c}_i}{{V_c}_j} + E \left( \frac{{V_c}_i}{{V_c}_j} \right)^2. $$

В данном уравнении коэффициенты $C, \; D, \; E$ зависят от ацентрических факторов, а также типа компонента:

$$ \begin{align} C &= c_1 + c_2 \left| \omega_i - \omega_j \right|; \\ D &= d_1 + d_2 \left| \omega_i - \omega_j \right|. \end{align}$$

Также зависимости коэффициента попарного взаимодействия между азотом и углеводородными компонентами исследуются в работах \[[Katz and Firoozabadi, 1978](https://doi.org/10.2118/6721-PA); [Mehra, 1981](http://dx.doi.org/10.11575/PRISM/13997); [Pedersen et al, 2001](https://doi.org/10.1016/S0378-3812(01)00562-3); [Privat et al, 2008](https://doi.org/10.1021/ie800636h); [Privat et al, 2008](https://doi.org/10.1021/ie071524b); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\].

Зависимости коэффициента попарного взаимодействия между кислородом, углеводородными и неуглеводородными компонентами рассматриваются в работе \[[Xu et al, 2015](https://doi.org/10.1021/acs.iecr.5b02639)\]. Зависимости коэффициента попарного взаимодействия между водородом, сероводородом и другими углеводородоными и неуглеводородными компонентами приводятся в работах \[[Valderama et al, 1990](https://doi.org/10.1016/0009-2509(90)87079-8); [Qian et al, 2013](https://doi.org/10.1021/ie402541h); [Fateen et al, 2013](https://doi.org/10.1016/j.jare.2012.03.004)\]. Зависимости коэффициента попарного взаимодействия между водой, углеводородными и неуглеводородными компонентами для уравнения состояния Пенга-Робинсона изучаются в работах \[[Qian et al, 2013](https://doi.org/10.1021/ie402541h)\].

Таким образом, учитывая колоссальное количество корреляций для коэффициентов попарного взаимодействия, возникает задача разработки универсального подхода к расчету коэффициентов попарного взаимодействия. Данная задача решается в рамках работы \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\]. Авторами предложен погрупповой, по аналогии с рассмотренным [ранее](#pvt-eos-appendix-bip-nishumi), подход к определению коэффициентов попарного взаимодействия для [уравнения состояния Пенга-Робинсона](EOS-2-SRK-PR.md#pvt-eos-srk_pr) и [правил смешивания Ван-дер-Ваальса](EOS-1-VanDerWaals.md#pvt-eos-vdw-mixrules) с тем отличием, что в рамках универсального подхода учитывается температурная зависимость коэффициентов попарного взаимодействия. При этом в качестве исходных данных для расчета коэффициентов попарного взаимодействия необходимы критические давления компонентов ${P_c}_i$, критические температуры компонентов ${T_c}_i$, а также ацентрические факторы компонентов $\omega_i$, то есть те же самые параметры, необходимые для использования [уравнения состояния Пенга-Робинсона](EOS-2-SRK-PR.md#pvt-eos-srk_pr).

<a id='pvt-eos-appendix-bip-gcm'></a>
Согласно \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\], температурная зависимость коэффициента попарного взаимодействия определяется следующим выражением:

$$ k_{ij} \left( T \right) = \frac{-\frac{1}{2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 1} - \left( \delta_i - \delta_j \right)^2}{2 \delta_i \delta_j},$$

где $\delta_i = \frac{\sqrt{\alpha_i}}{b_i}$ (параметры $\alpha_i, \; b_i$ определяются в соответствии с [уравнением состояния Пенга-Робинсона](EOS-2-SRK-PR.md#pvt-eos-srk_pr)), $N_g$ – количество групп (соответственно $k, \; l$ – индексы групп), $\alpha_{ik}$ – доля молекулы $i$, занятая группой $k$ (то есть количество появлений группы $k$ в молекуле $i$, деленное на общее количество групп в молекуле $i$), параметры $A_{kl} = A_{lk}, \; B_{kl} = B_{lk}$ – коэффициенты, подобранные авторами работы \[[Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)\] для различных групп ($A_{kk} = B_{kk} = 0$). Следовательно, с учетом данной формулы для расчета коэффициента попарного взаимодействия молекул необходимо знать критические свойства компонентов, их ацентрические факторы и структуру молекулы.

Приведенное выше выражение берет свое начало из описания термодинамической системы решеточной моделью ([lattice model](https://en.wikipedia.org/wiki/Lattice_model_(physics))). Фундамент для вывода данного выражения приводится в работах \[[Redlich et al, 1959](https://doi.org/10.1021/ja01519a001); [Kehiaian et al, 1971](https://doi.org/10.1051/jcp/1971680922); [Kehiaian, 1985](https://doi.org/10.1351/pac198557010015); [Tine and Kehiaian, 1987](https://doi.org/10.1016/0378-3812(87)85056-2); [Abdoul et al, 1991](https://doi.org/10.1016/0378-3812(91)85010-R)\].

Коэффициенты $A_{kl} = A_{lk}, \; B_{kl} = B_{lk}$ в МПа представлены в работе \[[Qian et al, 2013](https://doi.org/10.1021/ie402541h)\] и следующих таблицах соответственно:

<table class="table">
  <caption>Коэффициенты Aₖₗ (МПа)</caption>
  <thead>
    <tr>
      <th colspan="2" rowspan="2">&nbsp;</th>
      <th colspan="13">Группы</th>
    </tr>
    <tr>
      <th>CH₃</th>
      <th>CH₂</th>
      <th>CH</th>
      <th>C</th>
      <th>CH₄</th>
      <th>C₂H₆</th>
      <th>CO₂</th>
      <th>N₂</th>
      <th>H₂S</th>
      <th>SH</th>
      <th>H₂</th>
      <th>C₂H₄</th>
      <th>H₂O</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="14" class="rotate">Группы</th>
    </tr>
    <tr>
      <th>CH₃</th>
      <td>0</td>
      <td>75</td>
      <td>262</td>
      <td>397</td>
      <td>33</td>
      <td>9</td>
      <td>164</td>
      <td>53</td>
      <td>158</td>
      <td>800</td>
      <td>203</td>
      <td>7</td>
      <td>3557</td>
    </tr>
    <tr>
      <th>CH₂</th>
      <td>75</td>
      <td>0</td>
      <td>51</td>
      <td>89</td>
      <td>37</td>
      <td>31</td>
      <td>137</td>
      <td>82</td>
      <td>135</td>
      <td>460</td>
      <td>132</td>
      <td>60</td>
      <td>4324</td>
    </tr>
    <tr>
      <th>CH</th>
      <td>262</td>
      <td>51</td>
      <td>0</td>
      <td>-306</td>
      <td>145</td>
      <td>174</td>
      <td>184</td>
      <td>365</td>
      <td>194</td>
      <td>426</td>
      <td>415</td>
      <td>177</td>
      <td>971</td>
    </tr>
    <tr>
      <th>C</th>
      <td>397</td>
      <td>89</td>
      <td>-306</td>
      <td>0</td>
      <td>264</td>
      <td>333</td>
      <td>288</td>
      <td>264</td>
      <td>305</td>
      <td>683</td>
      <td>226</td>
      <td>320</td>
      <td>–</td>
    </tr>
    <tr>
      <th>CH₄</th>
      <td>33</td>
      <td>37</td>
      <td>145</td>
      <td>264</td>
      <td>0</td>
      <td>13</td>
      <td>137</td>
      <td>38</td>
      <td>181</td>
      <td>704</td>
      <td>156</td>
      <td>15</td>
      <td>2265</td>
    </tr>
    <tr>
      <th>C₂H₆</th>
      <td>9</td>
      <td>31</td>
      <td>174</td>
      <td>333</td>
      <td>13</td>
      <td>0</td>
      <td>136</td>
      <td>62</td>
      <td>157</td>
      <td>–</td>
      <td>138</td>
      <td>8</td>
      <td>2333</td>
    </tr>
    <tr>
      <th>CO₂</th>
      <td>164</td>
      <td>137</td>
      <td>184</td>
      <td>288</td>
      <td>137</td>
      <td>136</td>
      <td>0</td>
      <td>98</td>
      <td>135</td>
      <td>470</td>
      <td>266</td>
      <td>73</td>
      <td>559</td>
    </tr>
    <tr>
      <th>N₂</th>
      <td>53</td>
      <td>82</td>
      <td>365</td>
      <td>264</td>
      <td>38</td>
      <td>62</td>
      <td>98</td>
      <td>0</td>
      <td>320</td>
      <td>1044</td>
      <td>65</td>
      <td>89</td>
      <td>2574</td>
    </tr>
    <tr>
      <th>H₂S</th>
      <td>158</td>
      <td>135</td>
      <td>194</td>
      <td>305</td>
      <td>181</td>
      <td>157</td>
      <td>135</td>
      <td>320</td>
      <td>0</td>
      <td>-77</td>
      <td>146</td>
      <td>–</td>
      <td>604</td>
    </tr>
    <tr>
      <th>SH</th>
      <td>800</td>
      <td>460</td>
      <td>426</td>
      <td>683</td>
      <td>704</td>
      <td>–</td>
      <td>470</td>
      <td>1044</td>
      <td>-77</td>
      <td>0</td>
      <td>–</td>
      <td>–</td>
      <td>31</td>
    </tr>
    <tr>
      <th>H₂</th>
      <td>203</td>
      <td>132</td>
      <td>415</td>
      <td>226</td>
      <td>156</td>
      <td>138</td>
      <td>266</td>
      <td>65</td>
      <td>146</td>
      <td>–</td>
      <td>0</td>
      <td>151</td>
      <td>831</td>
    </tr>
    <tr>
      <th>C₂H₄</th>
      <td>7</td>
      <td>60</td>
      <td>177</td>
      <td>320</td>
      <td>15</td>
      <td>8</td>
      <td>73</td>
      <td>89</td>
      <td>–</td>
      <td>–</td>
      <td>151</td>
      <td>0</td>
      <td>1632</td>
    </tr>
    <tr>
      <th>H₂O</th>
      <td>3557</td>
      <td>4324</td>
      <td>971</td>
      <td>–</td>
      <td>2265</td>
      <td>2333</td>
      <td>559</td>
      <td>2574</td>
      <td>604</td>
      <td>31</td>
      <td>831</td>
      <td>1632</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<br>

<table class="table">
  <caption>Коэффициенты Bₖₗ (МПа)</caption>
  <thead>
    <tr>
      <th colspan="2" rowspan="2">&nbsp;</th>
      <th colspan="13">Группы</th>
    </tr>
    <tr>
      <th>CH₃</th>
      <th>CH₂</th>
      <th>CH</th>
      <th>C</th>
      <th>CH₄</th>
      <th>C₂H₆</th>
      <th>CO₂</th>
      <th>N₂</th>
      <th>H₂S</th>
      <th>SH</th>
      <th>H₂</th>
      <th>C₂H₄</th>
      <th>H₂O</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="14" class="rotate">Группы</th>
    </tr>
    <tr>
      <th>CH₃</th>
      <td>0</td>
      <td>166</td>
      <td>389</td>
      <td>804</td>
      <td>-35</td>
      <td>-30</td>
      <td>269</td>
      <td>87</td>
      <td>241</td>
      <td>2109</td>
      <td>317</td>
      <td>39</td>
      <td>11195</td>
    </tr>
    <tr>
      <th>CH₂</th>
      <td>166</td>
      <td>0</td>
      <td>80</td>
      <td>315</td>
      <td>108</td>
      <td>85</td>
      <td>255</td>
      <td>203</td>
      <td>138</td>
      <td>627</td>
      <td>147</td>
      <td>79</td>
      <td>12126</td>
    </tr>
    <tr>
      <th>CH</th>
      <td>389</td>
      <td>80</td>
      <td>0</td>
      <td>-251</td>
      <td>302</td>
      <td>352</td>
      <td>762</td>
      <td>522</td>
      <td>308</td>
      <td>515</td>
      <td>726</td>
      <td>118</td>
      <td>568</td>
    </tr>
    <tr>
      <th>C</th>
      <td>804</td>
      <td>315</td>
      <td>-251</td>
      <td>0</td>
      <td>532</td>
      <td>204</td>
      <td>346</td>
      <td>773</td>
      <td>-143</td>
      <td>1544</td>
      <td>1812</td>
      <td>-248</td>
      <td>–</td>
    </tr>
    <tr>
      <th>CH₄</th>
      <td>-35</td>
      <td>108</td>
      <td>302</td>
      <td>532</td>
      <td>0</td>
      <td>7</td>
      <td>194</td>
      <td>37</td>
      <td>289</td>
      <td>1496</td>
      <td>93</td>
      <td>30</td>
      <td>4722</td>
    </tr>
    <tr>
      <th>C₂H₆</th>
      <td>-30</td>
      <td>85</td>
      <td>352</td>
      <td>204</td>
      <td>7</td>
      <td>0</td>
      <td>240</td>
      <td>85</td>
      <td>217</td>
      <td>–</td>
      <td>150</td>
      <td>19</td>
      <td>5147</td>
    </tr>
    <tr>
      <th>CO₂</th>
      <td>269</td>
      <td>255</td>
      <td>762</td>
      <td>346</td>
      <td>194</td>
      <td>240</td>
      <td>0</td>
      <td>221</td>
      <td>201</td>
      <td>900</td>
      <td>268</td>
      <td>115</td>
      <td>278</td>
    </tr>
    <tr>
      <th>N₂</th>
      <td>87</td>
      <td>203</td>
      <td>522</td>
      <td>773</td>
      <td>37</td>
      <td>85</td>
      <td>221</td>
      <td>0</td>
      <td>550</td>
      <td>1872</td>
      <td>70</td>
      <td>109</td>
      <td>5490</td>
    </tr>
    <tr>
      <th>H₂S</th>
      <td>241</td>
      <td>138</td>
      <td>308</td>
      <td>-143</td>
      <td>289</td>
      <td>217</td>
      <td>201</td>
      <td>550</td>
      <td>0</td>
      <td>156</td>
      <td>824</td>
      <td>–</td>
      <td>599</td>
    </tr>
    <tr>
      <th>SH</th>
      <td>2109</td>
      <td>627</td>
      <td>515</td>
      <td>1544</td>
      <td>1496</td>
      <td>–</td>
      <td>900</td>
      <td>1872</td>
      <td>156</td>
      <td>0</td>
      <td>–</td>
      <td>–</td>
      <td>-114</td>
    </tr>
    <tr>
      <th>H₂</th>
      <td>317</td>
      <td>147</td>
      <td>726</td>
      <td>1812</td>
      <td>93</td>
      <td>150</td>
      <td>268</td>
      <td>70</td>
      <td>824</td>
      <td>–</td>
      <td>0</td>
      <td>165</td>
      <td>-138</td>
    </tr>
    <tr>
      <th>C₂H₄</th>
      <td>39</td>
      <td>79</td>
      <td>118</td>
      <td>-248</td>
      <td>30</td>
      <td>19</td>
      <td>115</td>
      <td>109</td>
      <td>–</td>
      <td>–</td>
      <td>165</td>
      <td>0</td>
      <td>1612</td>
    </tr>
    <tr>
      <th>H₂O</th>
      <td>11195</td>
      <td>12126</td>
      <td>568</td>
      <td>–</td>
      <td>4722</td>
      <td>5147</td>
      <td>278</td>
      <td>5490</td>
      <td>599</td>
      <td>-114</td>
      <td>-138</td>
      <td>1612</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<br>

Рассмотрим пример расчета коэффициента попарного взаимодействия с использованием данного выражения. В качестве примера возьмем смесь нормального бутана $\left( CH_3 - CH_2 - CH_2 - CH_3 \right)$ и пропана $\left( CH_3 - CH_2 - CH_3 \right)$ при температуре $T = 303.15 \; K$.

```{code-cell} python
R = 8.314472 # Universal gas constant [J/mol/K]
T = 303.15 # Temperature [K]
```

Пропан состоит из двух групп $\#1:CH_3$ и одной группы $\#2:CH_2$, следовательно, общее количество групп ${N_g}_1 = 3$. Нормальный бутан состоит из двух групп $\#1:CH_3$ и двух групп $\#2:CH_2$, следовательно, общее количество групп ${N_g}_2 = 4$. Поскольку в данном примере используются всего две группы, то матрицы $A_{kl}$ и $B_{kl}$ можно упростить до:

```{code-cell} python
import numpy as np
Akl = np.array([[0., 74.81], [74.81, 0.]]) * 1e6 # Group interaction parameters (Akl) [Pa]
Bkl = np.array([[0., 165.7], [165.7, 0.]]) * 1e6 # Group interaction parameters (Bkl) [Pa]
```

Доля группы $\#1:CH_3$ в составе молекулы пропана:

$$ \alpha_{11} = \frac{2}{3}. $$

Аналогично доля группы $\#2:CH_2$ в составе молекулы пропана:

$$ \alpha_{12} = \frac{1}{3}. $$

Для остальных групп $k = 3 \ldots 21$ доля в составе молекулы пропана:

$$ \alpha_{1k} = 0, \; k = 3 \dots 21. $$

Для молекулы нормального бутана:

$$ \begin{align} \alpha_{21} &= \frac{2}{4} = \frac{1}{2}; \\ \alpha_{22} &= \frac{2}{4} = \frac{1}{2}; \\ \alpha_{2k} &= 0, \; k = 3 \dots 2. \end{align} $$

Тогда сформируем два вектора следующим образом:

```{code-cell} python
alpha1 = np.array([[2/3], [1/3]]) # Fractions occupied by considered groups in the propane molecule
alpha2 = np.array([[1/2], [1/2]]) # Fractions occupied by considered groups in the butane molecule
```

Определим значение параметра $DS$ *(double sum)*:

$$DS = -\frac{1}{2} \sum_k^{N_g} \sum_l^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \left( \frac{298.15}{T} \right)^{\frac{B_{kl}}{A_{kl}} - 1}.$$

```{code-cell} python
DS = np.sum(np.outer(alpha1 - alpha2, alpha1 - alpha2) * Akl * (298.15 / T) ** (np.divide(Bkl, Akl, out=np.ones_like(Akl), where=Akl!=0.) - 1.)) / (-2.) # Double sum parameter
DS
```

Определим значения параметров $\alpha_i$ и $b_i$ для компонентов с учетом их критических свойств и ацентрического фактора:

```{code-cell} python
Pci = np.array([4.248, 3.796]) * 1e6 # Critical pressures of components [Pa]
Tci = np.array([369.83, 425.12]) # Critical temperatures of components [K]
wi = np.array([.152, .2]) # Acentric factors of components
```

```{code-cell} python
ai = .45724 * (R * Tci)**2 / Pci
alphai = ai * (1. + (.37464 + 1.54226 * wi - .26992 * wi**2) * (1 - np.sqrt(T / Tci)))**2
bi = .07780 * R * Tci / Pci
```

Определим значение $\delta_i$ для каждого компонента:

```{code-cell} python
deltai = alphai**0.5 / bi
deltai
```

Тогда коэффициент попарного взаимодействия:

```{code-cell} python
kij = (DS - (deltai[0] - deltai[1])**2) / (2 * deltai[0] * deltai[1])
kij
```

```{admonition} NB
Очевидно, что изложенный выше подход к расчету коэффициентов попарного взаимодействия применим для компонентов с известной молекулярной структурой. Для псевдокомпонентов, характеризующих, скорее фракцию, чем конкретное вещество, данный подход может быть использован для определеления начальных значений коэффициентов попарного взаимодействия с целью последующего уточнения значений в процессе адаптации модели.
```
