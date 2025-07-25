import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
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

  def test_01(self):
    P = np.float64(16e6)
    T0 = np.float64(68. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=38, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

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
    tol = 1e-5
    Tsat = TsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=33, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_03(self):
    P = np.float64(4e6)
    T0 = np.float64(-60. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=4, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_04(self):
    P = np.float64(4e6)
    T0 = np.float64(-90. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=4, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_05(self):
    P = np.float64(16e6)
    T0 = np.float64(68. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=7, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_06(self):
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=6, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_07(self):
    P = np.float64(4e6)
    T0 = np.float64(-60. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=3, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_08(self):
    P = np.float64(4e6)
    T0 = np.float64(-90. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=3, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_09(self):
    P = np.float64(16e6)
    T0 = np.float64(68. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=3, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_10(self):
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_11(self):
    P = np.float64(4e6)
    T0 = np.float64(-60. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_12(self):
    P = np.float64(4e6)
    T0 = np.float64(-90. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_13(self):
    P = np.float64(16e6)
    T0 = np.float64(68. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-b', stabkwargs=dict(method='qnss-newton'),
                  maxiter=3, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_14(self):
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_15(self):
    P = np.float64(4e6)
    T0 = np.float64(-60. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_16(self):
    P = np.float64(4e6)
    T0 = np.float64(-90. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_17(self):
    P = np.float64(16e6)
    T0 = np.float64(68. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=3, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_18(self):
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_19(self):
    P = np.float64(4e6)
    T0 = np.float64(-60. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_20(self):
    P = np.float64(4e6)
    T0 = np.float64(-90. + 273.15)
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
    tol = 1e-5
    Tsat = TsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=2, tol=tol)
    res = Tsat.run(P, T0, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
