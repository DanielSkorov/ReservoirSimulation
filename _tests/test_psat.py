import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
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

from boundary import (
  PsatPT,
)

class stabPT(unittest.TestCase):

  def test_1(self):
    T = np.float64(68. + 273.15)
    P0 = np.float64(15e6)
    yi = np.array([.15, .85])
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
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    Psat = PsatPT(pr, method='qnss', improve_P0=True, stabgrid=False,
                  stabkwargs=dict(method='qnss'))
    res = Psat.run(T, yi, P0=P0, Npoint=10)
    self.assertTrue(res.success)
    pass

  def test_2(self):
    T = np.float64(68. + 273.15)
    P0 = np.float64(20e6)
    yi = np.array([.15, .85])
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
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    Psat = PsatPT(pr, method='qnss', improve_P0=True, stabgrid=True,
                  stabkwargs=dict(method='qnss'))
    res = Psat.run(T, yi, P0=P0, Npoint=10)
    self.assertTrue(res.success)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
