import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('flash')
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

from flash import (
  flash2pPT,
)


class flash(unittest.TestCase):

  def test_01(self):
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
    tol = 1e-5
    flash = flash2pPT(pr, flashmethod='ss', stabmethod='ss', tol=tol,
                      maxiter=7)
    res = flash.run(P, T, yi)
    self.assertTrue((res.gnorm < tol) & (res.success))
    pass

  def test_02(self):
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
    tol = 1e-5
    flash = flash2pPT(pr, flashmethod='qnss', stabmethod='qnss', tol=tol,
                      maxiter=6)
    res = flash.run(P, T, yi)
    self.assertTrue((res.gnorm < tol) & (res.success))
    pass

  def test_03(self):
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
    tol = 1e-6
    flash = flash2pPT(pr, flashmethod='qnss', stabmethod='qnss', tol=tol,
                      maxiter=12)
    res = flash.run(P, T, yi)
    self.assertTrue((res.gnorm < tol) & (res.success))
    pass

  def test_04(self):
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
    tol = 1e-5
    flash = flash2pPT(pr, flashmethod='newton', stabmethod='ss', tol=tol)
    res = flash.run(P, T, yi)
    self.assertTrue((res.gnorm < tol) & (res.success))
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
