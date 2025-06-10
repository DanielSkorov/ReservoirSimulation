import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
  '%(process)d:%(name)s:%(levelname)s:\n\t%(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np
np.set_printoptions(linewidth=np.inf)

from eos import (
  pr78,
)

from boundary import (
  TsatPT,
)


class tsat(unittest.TestCase):

  # def test_01(self):
  #   P = np.float64(16e6)
  #   T0 = np.float64(68. + 273.15)
  #   yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
  #   Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
  #   Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
  #   mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
  #   wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
  #   vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
  #   dij = np.array([
  #     0.0027,
  #     0.0085, 0.0017,
  #     0.0147, 0.0049, 0.0009,
  #     0.0393, 0.0219, 0.0117, 0.0062,
  #   ])
  #   pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
  #   Tsat = TsatPT(pr, method='ss', upper=True,
  #                 stabkwargs=dict(method='qnss'), maxiter=39)
  #   res = Tsat.run(P, yi, T0)
  #   self.assertTrue(res.success)
  #   pass

  def test_02(self):
    P = np.float64(16e6)
    T0 = np.float64(100. + 273.15)
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
    vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np.array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    Tsat = TsatPT(pr, method='ss', upper=True,
                  stabkwargs=dict(method='qnss'), maxiter=1)
    res = Tsat.run(P, yi, T0)
    self.assertTrue(res.success)
    pass

  # def test_11(self):
  #   P0 = np.float64(1e6)
  #   T = np.float64(68. + 273.15)
  #   yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
  #   Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
  #   Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
  #   mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
  #   wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
  #   vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
  #   dij = np.array([
  #     0.0027,
  #     0.0085, 0.0017,
  #     0.0147, 0.0049, 0.0009,
  #     0.0393, 0.0219, 0.0117, 0.0062,
  #   ])
  #   pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
  #   Psat = PsatPT(pr, method='ss', upper=False,
  #                 stabkwargs=dict(method='qnss'), maxiter=3)
  #   res = Psat.run(T, yi, P0)
  #   self.assertTrue(res.success)
  #   pass

  # def test_16(self):
  #   P0 = np.float64(1e5)
  #   T = np.float64(68. + 273.15)
  #   yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
  #   Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
  #   Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
  #   mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
  #   wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
  #   vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
  #   dij = np.array([
  #     0.0027,
  #     0.0085, 0.0017,
  #     0.0147, 0.0049, 0.0009,
  #     0.0393, 0.0219, 0.0117, 0.0062,
  #   ])
  #   pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
  #   Psat = PsatPT(pr, method='ss', upper=False,
  #                 stabkwargs=dict(method='qnss'), maxiter=3)
  #   res = Psat.run(T, yi, P0)
  #   self.assertTrue(res.success)
  #   pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
