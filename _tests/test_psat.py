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
  PsatPT,
)


class psat(unittest.TestCase):

  def test_01(self):
    P0 = 15e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=118, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_02(self):
    P0 = 20e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=118, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_03(self):
    P0 = 1e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_04(self):
    P0 = 1e5
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='ss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_05(self):
    P0 = 15e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=11, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_06(self):
    P0 = 20e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=11, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_07(self):
    P0 = 1e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_08(self):
    P0 = 1e5
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='qnss', stabkwargs=dict(method='qnss-newton'),
                  maxiter=2, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_09(self):
    P0 = 15e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=5, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_10(self):
    P0 = 20e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=6, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_11(self):
    P0 = 1e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=1, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_12(self):
    P0 = 1e5
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton', stabkwargs=dict(method='qnss-newton'),
                  maxiter=1, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_13(self):
    P0 = 15e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=4, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_14(self):
    P0 = 20e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=4, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_15(self):
    P0 = 1e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=1, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_16(self):
    P0 = 1e5
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=1, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_17(self):
    P0 = 15e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=5, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_18(self):
    P0 = 20e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=6, tol=tol)
    res = Psat.run(P0, T, yi, upper=True)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_19(self):
    P0 = 1e6
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=1, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_20(self):
    P0 = 1e5
    T = 68. + 273.15
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
    Psat = PsatPT(pr, method='newton-c',
                  stabkwargs=dict(method='qnss-newton'), maxiter=1, tol=tol)
    res = Psat.run(P0, T, yi, upper=False)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_21(self):
    P0 = 25e6
    T = 373.15
    yi = np.array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np.array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np.array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np.array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np.array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-8
    Psat = PsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=18,
                  tol=tol, tol_tpd=1e-8)
    res = Psat.run(P0, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_22(self):
    P0 = 23.3e6
    T = 373.15
    yi = np.array([0.55, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
                   0.0370, 0.0238, 0.0124, 0.0208])
    Pci = np.array([73.82, 45.40, 48.20, 41.90, 37.50, 28.82, 23.74, 18.59,
                    14.80, 11.95, 8.52]) * 1e5
    Tci = np.array([304.21, 190.60, 305.40, 369.80, 425.20, 516.67, 590.00,
                    668.61, 745.78, 812.67, 914.89])
    mwi = np.array([44.0, 16.0, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3,
                    336.1, 536.7]) / 1e3
    wi = np.array([0.225, 0.008, 0.098, 0.152, 0.193, 0.265, 0.364, 0.499,
                   0.661, 0.877, 1.279])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.105,
      0.115, 0.000,
      0.115, 0.000, 0.002,
      0.115, 0.000, 0.005, 0.001,
      0.115, 0.045, 0.016, 0.008, 0.003,
      0.115, 0.055, 0.027, 0.016, 0.009, 0.001,
      0.115, 0.055, 0.042, 0.027, 0.019, 0.006, 0.002,
      0.115, 0.060, 0.057, 0.040, 0.030, 0.013, 0.006, 0.001,
      0.115, 0.080, 0.070, 0.051, 0.040, 0.020, 0.011, 0.004, 0.001,
      0.115, 0.280, 0.089, 0.068, 0.055, 0.032, 0.020, 0.010, 0.004, 0.001,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-8
    Psat = PsatPT(pr, method='newton-b',
                  stabkwargs=dict(method='qnss-newton'), maxiter=5,
                  tol=tol, tol_tpd=1e-8)
    res = Psat.run(P0, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
