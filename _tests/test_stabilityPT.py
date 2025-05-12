import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('stab')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
  '%(process)d:%(name)s:%(levelname)s:\n\t%(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np

from eos import (
  pr78,
)

from stability import (
  stabilityPT,
)


class stabPT(unittest.TestCase):

  def test_1(self):
    P = np.float64(2e6)
    T = np.float64(40. + 273.15)
    yi = np.array([.15, .85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, method='qnss')
    res = stab.run(P, T, yi)
    self.assertTrue(res.stable & res.success)
    pass

  def test_2(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, method='qnss')
    res = stab.run(P, T, yi)
    self.assertFalse(res.stable & res.success)
    pass

  def test_3(self):
    P = np.float64(101325.)
    T = np.float64(20. + 273.15)
    yi = np.array([.1, .6, .3])
    Pci = np.array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np.array([190.6, 507.5, 647.3])
    wi = np.array([.008, .27504, .344])
    mwi = np.array([0.016043, 0.086, 0.018015])
    vsi = np.array([0., 0., 0.])
    dij = np.array([.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, level=1, method='qnss')
    res = stab.run(P, T, yi)
    self.assertFalse(res.stable & res.success)
    pass

  def test_4(self):
    P = np.float64(2e6)
    T = np.float64(40. + 273.15)
    yi = np.array([.15, .85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, method='ss')
    res = stab.run(P, T, yi)
    self.assertTrue(res.stable & res.success)
    pass

  def test_5(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, method='ss')
    res = stab.run(P, T, yi)
    self.assertFalse(res.stable & res.success)
    pass

  def test_6(self):
    P = np.float64(101325.)
    T = np.float64(20. + 273.15)
    yi = np.array([.1, .6, .3])
    Pci = np.array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np.array([190.6, 507.5, 647.3])
    wi = np.array([.008, .27504, .344])
    mwi = np.array([0.016043, 0.086, 0.018015])
    vsi = np.array([0., 0., 0.])
    dij = np.array([.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, level=1, method='ss')
    res = stab.run(P, T, yi)
    self.assertFalse(res.stable & res.success)
    pass

  def test_7(self):
    P = np.float64(17e6)
    T = np.float64(68. + 273.15)
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np.array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, level=0, method='ss', maxiter=100)
    res = stab.run(P, T, yi)
    self.assertFalse(res.stable & res.success)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
