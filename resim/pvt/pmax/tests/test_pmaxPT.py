import sys
import logging
import unittest

from functools import (
  partial,
)

from numpy import (
  array as np_array,
)

from resim.pvt.eos import (
  pr78,
)

from resim.pvt.pmax import (
  _pmaxPT_newt,
  _pmaxPT_qnss,
  _pmaxPT_ss,
  pmax,
)


logger = logging.getLogger('pmax')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class pmaxPT(unittest.TestCase):

  def test_01(self):
    logger.info('\nTest #01.\nComponents: C1 C2 C3 NC4 C5+')
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i)
    solver = partial(_pmaxPT_ss, maxiter=1047)
    state = pmax.runPT(pr, yi, solver=solver)
    pass

  def test_02(self):
    logger.info('\nTest #02.\nComponents: C1 C2 C3 NC4 C5+')
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i)
    solver = partial(_pmaxPT_qnss, maxiter=59)
    state = pmax.runPT(pr, yi, solver=solver)
    pass

  def test_03(self):
    logger.info('\nTest #03.\nComponents: C1 C2 C3 NC4 C5+')
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i)
    solver = partial(_pmaxPT_newt, maxiter=20)
    state = pmax.runPT(pr, yi, solver=solver)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
