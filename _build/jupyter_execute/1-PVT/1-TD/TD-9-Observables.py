#!/usr/bin/env python
# coding: utf-8

# <a id='pvt-td-observables'></a>
# # Интенсивные и экстенсивные параметры
# На данный момент было введено множество параметров термодинамической системы, характеризующие ее с разных сторон:
# 
# $$T, P, V, N, \mu, U, H, S, F, G.$$
# 
# Было рассмотрено множество уравнений, взаимосвязывающие данные параметры между собой. При использовании данных уравнений, безусловно, необходимо следить за соответствием единиц измерения, Однако есть еще один критерий – интенсивность и экстенсивность параметров.

# <a id='pvt-td-observables-intensive'></a>
# ```{prf:определение}
# :nonumber:
# ***Интенсивным*** называется параметр, величина которого не зависит от размеров системы.
# ```

# <a id='pvt-td-observables-extensive'></a>
# ```{prf:определение}
# :nonumber:
# ***Экстенсивным*** называется параметр, значение которого равно сумме значений величин, соответствующих его частям.
# ```
# 
# Интенсивными параметрами из приведенного выше списка являются температура $T$, давление $P$ и химический потенциал $\mu$. Экстенсивными параметрами – все остальные: объем $V$, количество молекул $N$, внутренняя энергия $U$, энтальпия $H$, энергия Гельмгольца $F$ и энергия Гиббса $G$.

# Также необходимо отметить, что в результате произведения экстенсивного и интенсивного параметров получается экстенсивная величина. Отношение экстенсивных параметров – интенсивная величина. Сумма экстенсивных параметров – экстенсивная величина. Сумма интенсивных параметров – интенсивная величина. Из этого следует, что в дополнении к проверке уравнения по единицам измерения параметров можно использовать проверку на экстенсивность / интенсивность параметров. Например, выражение для [thermodynamic identity](TD-7-ChemicalPotential.html#pvt-td-chemical_potential-thermodynamic_identity):
# 
# $$ dU = T dS - P dV + \mu dN. $$
# 
# В левой части уравнения – экстенсивный параметр. А в правой части – сумма произведений интенсивного и экстенсивного параметров, что является экстенсивной величиной.
