# import numpy as np

# Pci = np.array([7.37646, 4.600155,]) * 1e6 # Pa
# Tci = np.array([304.2, 190.6,]) # K
# dij = np.array([.025,])

# Pci = np.array([7.37646, 4.600155, 3.5,]) * 1e6 # Pa
# Tci = np.array([304.2, 190.6, 120.1]) # K
# dij = np.array([.025, 0.01, 0.2])
# Pci = np.array([2., np.inf])

# from eos import vdw

# dij = np.array([np.inf,])
# dij = [0.]
# Pci[1] = -1.
# dij = np.array([1,], dtype=int)
# dij = np.array([1., 2.])

# vdw_eos = vdw(Pci, Tci, dij)

# P = np.float64(20e5)
# T = np.float64(40. + 273.15)
# yi = np.array([.15, .85])

# yi = [.15, .85]
# P = 20e5
# yi = np.array([1, 2], dtype=int)
# P = np.array([20e5])
# yi = np.array([.15, .85, .2])
# yi = np.array([.15, np.nan])
# P = np.float64(-20e5)
# yi = np.array([.15, -.85])

# print(vdw_eos.get_Z(P, T, yi))
# print(vdw_eos.get_lnfi(P, T, yi))

# quit()


# from eos import pr78

# Pci = np.array([7.37646, 4.600155]) * 1e6
# Tci = np.array([304.2, 190.6])
# wi = np.array([.225, .008])
# vsi = np.array([0., 0.])
# dij = np.array([.025,])

# pr = pr78(Pci, Tci, wi, vsi, dij)

# P = np.float64(6e6)
# T = np.float64(10. + 273.15)
# yji = np.array([[.9, .1], [.9, .1]])

# print(pr.get_Z(P, T, yji[0]))
# print(pr.get_lnfi(P, T, yji[0]))

# lnphiji, Zj = pr.get_lnphiji_Zj(P, T, yji)
# lnfji = lnphiji + np.log(P * yji)
# print(lnfji)

# print(eos_pr.get_lnfi(P, T, yi))

# print(help(eos_pr.get_Z))

# from myst_nb import glue


# out = 'a\na'

# class MultilineText(object):
#     def __init__(self, text):
#         self.text = text

#     def _repr_html_(self):
#         # print(self.text.replace('\n', '<br>'))
#         return self.text.replace('\n', '<br>')

# glue('glued_out1', MultilineText(out))



import numpy as np
from rr import solve2p_FGH, solve2p_GH
yi = np.array([0.770, 0.200, 0.010, 0.010, 0.005, 0.005])
kvi = np.array([1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997])

print(solve2p_FGH(kvi, yi))

