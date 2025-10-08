import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('eos')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np

np.set_printoptions(linewidth=np.inf)

from jax import (
  config,
)

config.update('jax_platforms', 'cpu')
config.update('jax_enable_x64', True)

from jax import (
  numpy as jnp,
  jacfwd,
)

from jax.lax import (
  cond,
  select,
)

from eos import (
  pr78,
)


class pr(unittest.TestCase):

  def test_01(self):
    P = 2e6
    T = 40. + 273.15
    yi = np.array([0.15, 0.85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([0.225, 0.008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    Z, lnfi = pr.getPT_Z_lnfi(P, T, yi)
    _Z = 0.95664027
    _lnfi = np.array([12.52680951, 14.30933176])
    self.assertTrue(np.isclose(Z, _Z) and np.allclose(lnfi, _lnfi))
    pass

  def test_02(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.71747271, 0.08947668, 0.09156880, 0.04467569, 0.05680612],
      [0.60172443, 0.09297005, 0.11122143, 0.06329671, 0.13078738],
    ])
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
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij, phaseid='wilson')
    sj = pr.getPT_PIDj(P, T, yji)
    self.assertTrue(sj[0] == 0 and sj[1] == 1)
    pass

  def test_03(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.71747271, 0.08947668, 0.09156880, 0.04467569, 0.05680612],
      [0.60172443, 0.09297005, 0.11122143, 0.06329671, 0.13078738],
    ])
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
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij, phaseid='vcTc')
    sj = pr.getPT_PIDj(P, T, yji)
    self.assertTrue(sj[0] == 0 and sj[1] == 1)
    pass

  def test_04(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.71747271, 0.08947668, 0.09156880, 0.04467569, 0.05680612],
      [0.60172443, 0.09297005, 0.11122143, 0.06329671, 0.13078738],
    ])
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
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij, phaseid='dadT')
    sj = pr.getPT_PIDj(P, T, yji)
    self.assertTrue(sj[0] == 0 and sj[1] == 1)
    pass

  def test_05(self):
    P = 17340e3
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
    lnphii = pr.getPT_lnphii(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii))
    pass

  def test_06(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidP = pr.getPT_Z_lnphii_dP(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidP = jacfwd(jlnphii, argnums=0)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidP, _dlnphiidP))
    pass

  def test_07(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidT = pr.getPT_Z_lnphii_dT(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidT = jacfwd(jlnphii, argnums=1)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidT, _dlnphiidT))
    pass

  def test_08(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidnj = pr.getPT_Z_lnphii_dnj(P, T, yi, 0.3)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(0.3 * yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidnj = jacfwd(jlnphii, argnums=2)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidnj, _dlnphiidnj))
    pass

  def test_09(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidP, dlnphiidT = pr.getPT_Z_lnphii_dP_dT(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidP = jacfwd(jlnphii, argnums=0)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidT = jacfwd(jlnphii, argnums=1)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidP, _dlnphiidP) and
                    np.allclose(dlnphiidT, _dlnphiidT))
    pass

  def test_10(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidP, dlnphiidnj = pr.getPT_Z_lnphii_dP_dnj(P, T, yi, 0.3)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(0.3 * yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidP = jacfwd(jlnphii, argnums=0)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidnj = jacfwd(jlnphii, argnums=2)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidP, _dlnphiidP) and
                    np.allclose(dlnphiidnj, _dlnphiidnj))
    pass

  def test_11(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidT, dlnphiidnj = pr.getPT_Z_lnphii_dT_dnj(P, T, yi, 0.3)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(0.3 * yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidT = jacfwd(jlnphii, argnums=1)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidnj = jacfwd(jlnphii, argnums=2)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidT, _dlnphiidT) and
                    np.allclose(dlnphiidnj, _dlnphiidnj))
    pass

  def test_12(self):
    P = 17340e3
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
    (Z, lnphii,
     dlnphiidP, dlnphiidT, dlnphiidyj) = pr.getPT_Z_lnphii_dP_dT_dyj(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidP = jacfwd(jlnphii, argnums=0)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidT = jacfwd(jlnphii, argnums=1)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidyj = jacfwd(jlnphii, argnums=2)
    _dlnphiidyj = jdlnphiidyj(P, T, ni, Pci, Tci, wi, vsi, pr.D, False)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidP, _dlnphiidP) and
                    np.allclose(dlnphiidT, _dlnphiidT) and
                    np.allclose(dlnphiidyj, _dlnphiidyj))
    pass

  def test_13(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
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
    Zj, lnphiji, dlnphijidnk = pr.getPT_Zj_lnphiji_dnk(P, T, yji, 0.3)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(0.3 * yji[0])
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidnj = jacfwd(jlnphii, argnums=2)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphiji[0], _lnphii) and
                    np.allclose(lnphiji[1], _lnphii) and
                    np.allclose(dlnphijidnk[0], _dlnphiidnj) and
                    np.allclose(dlnphijidnk[1], _dlnphiidnj))
    pass

  def test_14(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidP, d2lnphiidP2 = pr.getPT_Z_lnphii_dP_dP2(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidP = jacfwd(jlnphii, argnums=0)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jd2lnphiidP2 = jacfwd(jdlnphiidP, argnums=0)
    _d2lnphiidP2 = jd2lnphiidP2(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidP, _dlnphiidP) and
                    np.allclose(d2lnphiidP2, _d2lnphiidP2))
    pass

  def test_15(self):
    P = 17340e3
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
    Z, lnphii, dlnphiidT, d2lnphiidT2 = pr.getPT_Z_lnphii_dT_dT2(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdlnphiidT = jacfwd(jlnphii, argnums=1)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jd2lnphiidT2 = jacfwd(jdlnphiidT, argnums=1)
    _d2lnphiidT2 = jd2lnphiidT2(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(lnphii, _lnphii) and
                    np.allclose(dlnphiidT, _dlnphiidT) and
                    np.allclose(d2lnphiidT2, _d2lnphiidT2))
    pass

  def test_16(self):
    P = 17340e3
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
    Z, dZdT = pr.getPT_Z_dT(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdT = jacfwd(jZ, argnums=1)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Z, _Z) and
                    np.allclose(dZdT, _dZdT))
    pass

  def test_17(self):
    P = 17340e3
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
    Z, dZdT, d2ZdT2 = pr.getPT_Z_dT_dT2(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdT = jacfwd(jZ, argnums=1)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jd2ZdT2 = jacfwd(jdZdT, argnums=1)
    _d2ZdT2 = jd2ZdT2(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Z, _Z) and
                    np.allclose(dZdT, _dZdT) and
                    np.allclose(d2ZdT2, _d2ZdT2))
    pass

  def test_18(self):
    P = 17340e3
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
    Z, dZdP = pr.getPT_Z_dP(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdP = jacfwd(jZ, argnums=0)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Z, _Z) and
                    np.allclose(dZdP, _dZdP))
    pass

  def test_19(self):
    P = 17340e3
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
    Z, dZdP, d2ZdP2 = pr.getPT_Z_dP_dP2(P, T, yi)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdP = jacfwd(jZ, argnums=0)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jd2ZdP2 = jacfwd(jdZdP, argnums=0)
    _d2ZdP2 = jd2ZdP2(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Z, _Z) and
                    np.allclose(dZdP, _dZdP) and
                    np.allclose(d2ZdP2, _d2ZdP2))
    pass

  def test_20(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
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
    Zj, dZjdPj = pr.getPT_Zj_dP(P, T, yji)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdP = jacfwd(jZ, argnums=0)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Zj[0], _Z) and
                    np.allclose(Zj[1], _Z) and
                    np.allclose(dZjdPj[0], _dZdP) and
                    np.allclose(dZjdPj[1], _dZdP))
    pass

  def test_21(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
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
    Zj, dZjdTj = pr.getPT_Zj_dT(P, T, yji)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdT = jacfwd(jZ, argnums=1)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Zj[0], _Z) and
                    np.allclose(Zj[1], _Z) and
                    np.allclose(dZjdTj[0], _dZdT) and
                    np.allclose(dZjdTj[1], _dZdT))
    pass

  def test_22(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
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
    Zj, dZjdPj, d2ZjdP2j = pr.getPT_Zj_dP_dP2(P, T, yji)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdP = jacfwd(jZ, argnums=0)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jd2ZdP2 = jacfwd(jdZdP, argnums=0)
    _d2ZdP2 = jd2ZdP2(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Zj[0], _Z) and
                    np.allclose(Zj[1], _Z) and
                    np.allclose(dZjdPj[0], _dZdP) and
                    np.allclose(dZjdPj[1], _dZdP) and
                    np.allclose(d2ZjdP2j[0], _d2ZdP2) and
                    np.allclose(d2ZjdP2j[1], _d2ZdP2))
    pass

  def test_23(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np.array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
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
    Zj, dZjdTj, d2ZjdT2j = pr.getPT_Zj_dT_dT2(P, T, yji)
    P = jnp.array(P)
    T = jnp.array(T)
    ni = jnp.asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jdZdT = jacfwd(jZ, argnums=1)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    jd2ZdT2 = jacfwd(jdZdT, argnums=1)
    _d2ZdT2 = jd2ZdT2(P, T, ni, Pci, Tci, wi, vsi, pr.D)
    self.assertTrue(np.allclose(Zj[0], _Z) and
                    np.allclose(Zj[1], _Z) and
                    np.allclose(dZjdTj[0], _dZdT) and
                    np.allclose(dZjdTj[1], _dZdT) and
                    np.allclose(d2ZjdT2j[0], _d2ZdT2) and
                    np.allclose(d2ZjdT2j[1], _d2ZdT2))
    pass


def _cardano(b, c, d):
  p = (3. * c - b * b) / 3.
  q = (2. * b * b * b - 9. * b * c + 27. * d) / 27.
  s = .25 * q * q + p * p * p / 27.
  return cond(s >= 0., _cardano_br1, _cardano_br3, p, q, s, b, d)

def _cardano_br1(p, q, s, b, d):
  s_ = jnp.sqrt(s)
  u1 = jnp.cbrt(-q / 2. + s_)
  u2 = jnp.cbrt(-q / 2. - s_)
  return (u1 + u2 - b / 3., 0., 0.), True

def _cardano_br3(p, q, s, b, d):
  t0 = (2. * jnp.sqrt(-p / 3.)
        * jnp.cos(jnp.arccos(1.5 * q * jnp.sqrt(-3. / p) / p) / 3.))
  x0 = t0 - b / 3.
  r = b + x0
  k = -d / x0
  D = jnp.sqrt(r * r - 4. * k)
  x1 = .5 * (-r + D)
  x2 = .5 * (-r - D)
  return (x0, x1, x2), False

def _take_Z(Zs, *args):
  return Zs[0]

def _select_Z(Zs, A, B, d1, d2):
  Z1 = Zs[0]
  Z2 = select(Zs[2] > 0., Zs[2], Zs[1])
  dG = (jnp.log((Z2 - B) / (Z1 - B))
        - (Z2 - Z1)
        + A / B / (d2 - d1) * jnp.log((Z2 + d2 * B) * (Z1 + d1 * B)
                                      / ((Z2 + d1 * B) * (Z1 + d2 * B))))
  return select(dG > 0., Z2, Z1)

def jZ(P, T, ni, Pci, Tci, wi, vsi, dij, asmoles=True):
  R = 8.3144598
  if asmoles:
    yi = ni / ni.sum()
  else:
    yi = ni
  d1 = -0.414213562373095
  d2 = 2.414213562373095
  RT = R * T
  PRT = P / RT
  wi2 = wi * wi
  wi3 = wi2 * wi
  kappai = jnp.where(
    wi <= 0.491,
    0.37464 + 1.54226 * wi - 0.26992 * wi2,
    0.379642 + 1.48503 * wi - 0.164423 * wi2 + 0.016666 * wi3,
  )
  sqrtai = 0.6761919320144113 * R * Tci / jnp.sqrt(Pci)
  bi = 0.07779607390388851 * R * Tci / Pci
  multi = 1. + kappai * (1. - jnp.sqrt(T / Tci))
  sqrtalphai = sqrtai * multi
  Si = sqrtalphai * jnp.dot(dij, yi * sqrtalphai)
  alpham = yi.dot(Si)
  bm = yi.dot(bi)
  A = alpham * PRT / RT
  B = bm * PRT
  Zs, unique = _cardano(
    B - 1., A - 2. * B - 3. * B * B, -A * B + B * B * (1. + B)
  )
  Z = cond(unique, _take_Z, _select_Z, Zs, A, B, d1, d2)
  return Z - yi.dot(vsi * bi) * PRT

def jlnphii(P, T, ni, Pci, Tci, wi, vsi, dij, asmoles=True):
  R = 8.3144598
  if asmoles:
    yi = ni / ni.sum()
  else:
    yi = ni
  d1 = -0.414213562373095
  d2 = 2.414213562373095
  RT = R * T
  PRT = P / RT
  wi2 = wi * wi
  wi3 = wi2 * wi
  kappai = jnp.where(
    wi <= 0.491,
    0.37464 + 1.54226 * wi - 0.26992 * wi2,
    0.379642 + 1.48503 * wi - 0.164423 * wi2 + 0.016666 * wi3,
  )
  sqrtai = 0.6761919320144113 * R * Tci / jnp.sqrt(Pci)
  bi = 0.07779607390388851 * R * Tci / Pci
  multi = 1. + kappai * (1. - jnp.sqrt(T / Tci))
  sqrtalphai = sqrtai * multi
  Si = sqrtalphai * jnp.dot(dij, yi * sqrtalphai)
  alpham = yi.dot(Si)
  bm = yi.dot(bi)
  A = alpham * PRT / RT
  B = bm * PRT
  Zs, unique = _cardano(
    B - 1., A - 2. * B - 3. * B * B, -A * B + B * B * (1. + B)
  )
  Z = cond(unique, _take_Z, _select_Z, Zs, A, B, d1, d2)
  gphii = A / B * (2. / alpham * Si - bi / bm)
  return ((Z - 1.) / bm * bi
          - jnp.log(Z - B)
          + gphii * (jnp.log((Z + B * d1) / (Z + B * d2)) / (d2 - d1))
          - (vsi * bi) * PRT)


if __name__ == '__main__':
  unittest.main(verbosity=0)
