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

from resim.pvt.tsat import (
  _tsatPT_newtA,
  _tsatPT_newtB,
  _tsatPT_newtC,
  _tsatPT_qnss,
  _tsatPT_ss,
  tsat,
)


logger = logging.getLogger('tsat')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class tsatPT(unittest.TestCase):

  def test_01(self):
    logger.info('\nTest #01.\nComponents: C1 C2 C3 NC4 C5+')
    P = 16e6
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
    solver = partial(_tsatPT_ss, maxiter=103)
    state = tsat.runPT(pr, P, yi, upper=True, solver=solver)
    pass

  def test_02(self):
    logger.info('\nTest #02.\nComponents: C1 C2 C3 NC4 C5+')
    P = 4e6
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
    solver = partial(_tsatPT_ss, maxiter=8)
    state = tsat.runPT(pr, P, yi, upper=False, solver=solver)
    pass

  def test_03(self):
    logger.info('\nTest #03.\nComponents: C1 C2 C3 NC4 C5+')
    P = 16e6
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
    solver = partial(_tsatPT_qnss, maxiter=12)
    state = tsat.runPT(pr, P, yi, upper=True, solver=solver)
    pass

  def test_04(self):
    logger.info('\nTest #04.\nComponents: C1 C2 C3 NC4 C5+')
    P = 4e6
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
    solver = partial(_tsatPT_qnss, maxiter=5)
    state = tsat.runPT(pr, P, yi, upper=False, solver=solver)
    pass

  def test_05(self):
    logger.info('\nTest #05.\nComponents: C1 C2 C3 NC4 C5+')
    P = 16e6
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
    solver = partial(_tsatPT_newtA, maxiter=4)
    state = tsat.runPT(pr, P, yi, upper=True, solver=solver)
    pass

  def test_06(self):
    logger.info('\nTest #06.\nComponents: C1 C2 C3 NC4 C5+')
    P = 4e6
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
    solver = partial(_tsatPT_newtA, maxiter=4)
    state = tsat.runPT(pr, P, yi, upper=False, solver=solver)
    pass

  def test_07(self):
    logger.info('\nTest #07.\nComponents: C1 C2 C3 NC4 C5+')
    P = 16e6
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
    solver = partial(_tsatPT_newtB, maxiter=4)
    state = tsat.runPT(pr, P, yi, upper=True, solver=solver)
    pass

  def test_08(self):
    logger.info('\nTest #08.\nComponents: C1 C2 C3 NC4 C5+')
    P = 4e6
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
    solver = partial(_tsatPT_newtB, maxiter=3)
    state = tsat.runPT(pr, P, yi, upper=False, solver=solver)
    pass

  def test_09(self):
    logger.info('\nTest #09.\nComponents: C1 C2 C3 NC4 C5+')
    P = 16e6
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
    solver = partial(_tsatPT_newtC, maxiter=4)
    state = tsat.runPT(pr, P, yi, upper=True, solver=solver)
    pass

  def test_10(self):
    logger.info('\nTest #10.\nComponents: C1 C2 C3 NC4 C5+')
    P = 4e6
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
    solver = partial(_tsatPT_newtC, maxiter=4)
    state = tsat.runPT(pr, P, yi, upper=False, solver=solver)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
