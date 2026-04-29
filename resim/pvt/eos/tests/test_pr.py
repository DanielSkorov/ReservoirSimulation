import sys
import logging
import unittest

from numpy import (
  allclose as np_allclose,
  array as np_array,
)

from jax import (
  config,
  jacfwd,
)

from jax.numpy import (
  asarray as jnp_asarray,
  arccos as jnp_arccos,
  array as jnp_array,
  cbrt as jnp_cbrt,
  cos as jnp_cos,
  dot as jnp_dot,
  log as jnp_log,
  sqrt as jnp_sqrt,
  where as jnp_where,
)

from resim.pvt.eos import (
  pr78,
)


logger = logging.getLogger('eos')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


config.update('jax_platforms', 'cpu')
config.update('jax_enable_x64', True)


class pr(unittest.TestCase):

  def test_01(self):
    P = 2e6
    T = 40. + 273.15
    yi = np_array([0.15, 0.85])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    lnfi = pr.getPT_lnfi(P, T, yi)
    _lnfi = np_array([12.52680951, 14.30933176])
    self.assertTrue(np_allclose(lnfi, _lnfi))
    pass

  def test_02(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.71747271, 0.08947668, 0.09156880, 0.04467569, 0.05680612],
      [0.60172443, 0.09297005, 0.11122143, 0.06329671, 0.13078738],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi, phaseid=0)
    sj = pr.getPT_PIDj(P, T, yji)
    self.assertTrue(sj[0] == 0)
    self.assertTrue(sj[1] == 1)
    pass

  def test_03(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.71747271, 0.08947668, 0.09156880, 0.04467569, 0.05680612],
      [0.60172443, 0.09297005, 0.11122143, 0.06329671, 0.13078738],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi, phaseid=1)
    sj = pr.getPT_PIDj(P, T, yji)
    self.assertTrue(sj[0] == 0)
    self.assertTrue(sj[1] == 1)
    pass

  def test_04(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.71747271, 0.08947668, 0.09156880, 0.04467569, 0.05680612],
      [0.60172443, 0.09297005, 0.11122143, 0.06329671, 0.13078738],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi, phaseid=2)
    sj = pr.getPT_PIDj(P, T, yji)
    self.assertTrue(sj[0] == 0)
    self.assertTrue(sj[1] == 1)
    pass

  def test_05(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii = pr.getPT_lnphii(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    pass

  def test_06(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidP = pr.getPT_lnphii_dP(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidP, _dlnphiidP))
    pass

  def test_07(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidT = pr.getPT_lnphii_dT(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidT, _dlnphiidT))
    pass

  def test_08(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidnj = pr.getPT_lnphii_dnj(P, T, yi, 0.3)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(0.3 * yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidnj, _dlnphiidnj))
    pass

  def test_09(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidP, dlnphiidT = pr.getPT_lnphii_dP_dT(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidP, _dlnphiidP))
    self.assertTrue(np_allclose(dlnphiidT, _dlnphiidT))
    pass

  def test_10(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidP, dlnphiidnj = pr.getPT_lnphii_dP_dnj(P, T, yi, 0.3)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(0.3 * yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidP, _dlnphiidP))
    self.assertTrue(np_allclose(dlnphiidnj, _dlnphiidnj))
    pass

  def test_11(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidT, dlnphiidnj = pr.getPT_lnphii_dT_dnj(P, T, yi, 0.3)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(0.3 * yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidT, _dlnphiidT))
    self.assertTrue(np_allclose(dlnphiidnj, _dlnphiidnj))
    pass

  def test_12(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidP, dlnphiidT, dlnphiidyj = pr.getPT_lnphii_dP_dT_dyj(
      P, T, yi,
    )
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidyj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi,
                              False)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidP, _dlnphiidP))
    self.assertTrue(np_allclose(dlnphiidT, _dlnphiidT))
    self.assertTrue(np_allclose(dlnphiidyj, _dlnphiidyj))
    pass

  def test_13(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphiji, dlnphijidnk = pr.getPT_lnphiji_dnk(P, T, yji, 0.3)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(0.3 * yji[0])
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidnj = jdlnphiidnj(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphiji[0], _lnphii))
    self.assertTrue(np_allclose(lnphiji[1], _lnphii))
    self.assertTrue(np_allclose(dlnphijidnk[0], _dlnphiidnj))
    self.assertTrue(np_allclose(dlnphijidnk[1], _dlnphiidnj))
    pass

  def test_14(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidP, d2lnphiidP2 = pr.getPT_lnphii_dP_dP2(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidP = jdlnphiidP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2lnphiidP2 = jd2lnphiidP2(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidP, _dlnphiidP))
    self.assertTrue(np_allclose(d2lnphiidP2, _d2lnphiidP2))
    pass

  def test_15(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    lnphii, dlnphiidT, d2lnphiidT2 = pr.getPT_lnphii_dT_dT2(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _lnphii = jlnphii(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dlnphiidT = jdlnphiidT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2lnphiidT2 = jd2lnphiidT2(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(lnphii, _lnphii))
    self.assertTrue(np_allclose(dlnphiidT, _dlnphiidT))
    self.assertTrue(np_allclose(d2lnphiidT2, _d2lnphiidT2))
    pass

  def test_16(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Z, dZdT = pr.getPT_Z_dT(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Z, _Z))
    self.assertTrue(np_allclose(dZdT, _dZdT))
    pass

  def test_17(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Z, dZdT, d2ZdT2 = pr.getPT_Z_dT_dT2(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2ZdT2 = jd2ZdT2(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Z, _Z))
    self.assertTrue(np_allclose(dZdT, _dZdT))
    self.assertTrue(np_allclose(d2ZdT2, _d2ZdT2))
    pass

  def test_18(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Z, dZdP = pr.getPT_Z_dP(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Z, _Z))
    self.assertTrue(np_allclose(dZdP, _dZdP))
    pass

  def test_19(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Z, dZdP, d2ZdP2 = pr.getPT_Z_dP_dP2(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2ZdP2 = jd2ZdP2(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Z, _Z))
    self.assertTrue(np_allclose(dZdP, _dZdP))
    self.assertTrue(np_allclose(d2ZdP2, _d2ZdP2))
    pass

  def test_20(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Zj, dZjdPj = pr.getPT_Zj_dP(P, T, yji)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Zj[0], _Z))
    self.assertTrue(np_allclose(Zj[1], _Z))
    self.assertTrue(np_allclose(dZjdPj[0], _dZdP))
    self.assertTrue(np_allclose(dZjdPj[1], _dZdP))
    pass

  def test_21(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Zj, dZjdTj = pr.getPT_Zj_dT(P, T, yji)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Zj[0], _Z))
    self.assertTrue(np_allclose(Zj[1], _Z))
    self.assertTrue(np_allclose(dZjdTj[0], _dZdT))
    self.assertTrue(np_allclose(dZjdTj[1], _dZdT))
    pass

  def test_22(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Zj, dZjdPj, d2ZjdP2j = pr.getPT_Zj_dP_dP2(P, T, yji)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2ZdP2 = jd2ZdP2(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Zj[0], _Z))
    self.assertTrue(np_allclose(Zj[1], _Z))
    self.assertTrue(np_allclose(dZjdPj[0], _dZdP))
    self.assertTrue(np_allclose(dZjdPj[1], _dZdP))
    self.assertTrue(np_allclose(d2ZjdP2j[0], _d2ZdP2))
    self.assertTrue(np_allclose(d2ZjdP2j[1], _d2ZdP2))
    pass

  def test_23(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Zj, dZjdTj, d2ZjdT2j = pr.getPT_Zj_dT_dT2(P, T, yji)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2ZdT2 = jd2ZdT2(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Zj[0], _Z))
    self.assertTrue(np_allclose(Zj[1], _Z))
    self.assertTrue(np_allclose(dZjdTj[0], _dZdT))
    self.assertTrue(np_allclose(dZjdTj[1], _dZdT))
    self.assertTrue(np_allclose(d2ZjdT2j[0], _d2ZdT2))
    self.assertTrue(np_allclose(d2ZjdT2j[1], _d2ZdT2))
    pass

  def test_24(self):
    P = 17340e3
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Z, dZdP, dZdT, d2ZdPdT = pr.getPT_Z_dP_dT_dPdT(P, T, yi)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2ZdPdT = jd2ZdPdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Z, _Z))
    self.assertTrue(np_allclose(dZdP, _dZdP))
    self.assertTrue(np_allclose(dZdT, _dZdT))
    self.assertTrue(np_allclose(d2ZdPdT, _d2ZdPdT))
    pass

  def test_25(self):
    P = 17340e3
    T = 68. + 273.15
    yji = np_array([
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
      [0.7167, 0.0895, 0.0917, 0.0448, 0.0573],
    ])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    Zj, dZjdPj, dZjdTj, d2ZjdPjdTj = pr.getPT_Zj_dP_dT_dPdT(P, T, yji)
    P = jnp_array(P)
    T = jnp_array(T)
    ni = jnp_asarray(yji[0])
    _Z = jZ(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdP = jdZdP(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dZdT = jdZdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2ZdPdT = jd2ZdPdT(P, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(Zj[0], _Z))
    self.assertTrue(np_allclose(Zj[1], _Z))
    self.assertTrue(np_allclose(dZjdPj[0], _dZdP))
    self.assertTrue(np_allclose(dZjdPj[1], _dZdP))
    self.assertTrue(np_allclose(dZjdTj[0], _dZdT))
    self.assertTrue(np_allclose(dZjdTj[1], _dZdT))
    self.assertTrue(np_allclose(d2ZjdPjdTj[0], _d2ZdPdT))
    self.assertTrue(np_allclose(d2ZjdPjdTj[1], _d2ZdPdT))
    pass

  def test_26(self):
    v = 0.00011196163999952803
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    P = pr.getVT_P(v, T, yi)
    self.assertTrue(np_allclose(P, 17340e3))
    pass

  def test_27(self):
    v = 0.00011196163999952803
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    P, dPdv = pr.getVT_P_dv(v, T, yi)
    v = jnp_array(v)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _P = jP(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dPdv = jdPdv(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(P, _P))
    self.assertTrue(np_allclose(dPdv, _dPdv))
    pass

  def test_28(self):
    v = 0.00011196163999952803
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    P, dPdT = pr.getVT_P_dT(v, T, yi)
    v = jnp_array(v)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _P = jP(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dPdT = jdPdT(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(P, _P))
    self.assertTrue(np_allclose(dPdT, _dPdT))
    pass

  def test_29(self):
    v = 0.00011196163999952803
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    P, dPdv, d2Pdv2 = pr.getVT_P_dv_dv2(v, T, yi)
    v = jnp_array(v)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _P = jP(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dPdv = jdPdv(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2Pdv2 = jd2Pdv2(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(P, _P))
    self.assertTrue(np_allclose(dPdv, _dPdv))
    self.assertTrue(np_allclose(d2Pdv2, _d2Pdv2))
    pass

  def test_30(self):
    v = 0.00011196163999952803
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    P, dPdT, d2PdT2 = pr.getVT_P_dT_dT2(v, T, yi)
    v = jnp_array(v)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _P = jP(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dPdT = jdPdT(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2PdT2 = jd2PdT2(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(P, _P))
    self.assertTrue(np_allclose(dPdT, _dPdT))
    self.assertTrue(np_allclose(d2PdT2, _d2PdT2))
    pass

  def test_31(self):
    v = 0.00011196163999952803
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np_array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np_array([0.012, 0.1, 0.152, 0.2, 0.414])
    s0i = np_array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    s1i = np_array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Trsi = np_array([293.15, 293.15, 293.15, 293.15, 293.15])
    dij = np_array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i, s1i, Trsi)
    P, dPdv, dPdT, d2Pdv2, d2PdT2, d2PdvdT = pr.getVT_P_dv_dT_dv2_dT2_dvdT(
      v, T, yi,
    )
    v = jnp_array(v)
    T = jnp_array(T)
    ni = jnp_asarray(yi)
    _P = jP(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dPdv = jdPdv(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2Pdv2 = jd2Pdv2(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _dPdT = jdPdT(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2PdT2 = jd2PdT2(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    _d2PdvdT = jd2PdvdT(v, T, ni, Pci, Tci, wi, pr.D, s0i, s1i, Trsi)
    self.assertTrue(np_allclose(P, _P))
    self.assertTrue(np_allclose(dPdv, _dPdv))
    self.assertTrue(np_allclose(d2Pdv2, _d2Pdv2))
    self.assertTrue(np_allclose(dPdT, _dPdT))
    self.assertTrue(np_allclose(d2PdT2, _d2PdT2))
    self.assertTrue(np_allclose(d2PdvdT, _d2PdvdT))
    pass


def jlnphii(P, T, ni, Pci, Tci, wi, dij, s0i, s1i, Tri, asmoles=True):
  R = 8.31446261815324
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
  kappai = jnp_where(
    wi <= 0.491,
    0.37464 + 1.54226 * wi - 0.26992 * wi2,
    0.379642 + 1.48503 * wi - 0.164423 * wi2 + 0.016666 * wi3,
  )
  sqrtai = 0.6761919320144113 * R * Tci / jnp_sqrt(Pci)
  bi = 0.07779607390388851 * R * Tci / Pci
  multi = 1. + kappai * (1. - jnp_sqrt(T / Tci))
  sqrtalphai = sqrtai * multi
  Si = sqrtalphai * jnp_dot(dij, yi * sqrtalphai)
  alpham = yi.dot(Si)
  bm = yi.dot(bi)
  A = alpham * PRT / RT
  B = bm * PRT
  Z = jsolve(A, B)
  gphii = A / B * (2. / alpham * Si - bi / bm) / (d2 - d1)
  vsi = bi * (s0i + s1i * (T - Tri))
  return ((Z - 1.) / bm * bi
          - jnp_log(Z - B)
          + gphii * jnp_log((Z + B * d1) / (Z + B * d2))
          - vsi * PRT)

def jZ(P, T, ni, Pci, Tci, wi, dij, s0i, s1i, Tri, asmoles=True):
  R = 8.31446261815324
  if asmoles:
    yi = ni / ni.sum()
  else:
    yi = ni
  RT = R * T
  PRT = P / RT
  wi2 = wi * wi
  wi3 = wi2 * wi
  kappai = jnp_where(
    wi <= 0.491,
    0.37464 + 1.54226 * wi - 0.26992 * wi2,
    0.379642 + 1.48503 * wi - 0.164423 * wi2 + 0.016666 * wi3,
  )
  sqrtai = 0.6761919320144113 * R * Tci / jnp_sqrt(Pci)
  bi = 0.07779607390388851 * R * Tci / Pci
  multi = 1. + kappai * (1. - jnp_sqrt(T / Tci))
  sqrtalphai = sqrtai * multi
  Si = sqrtalphai * jnp_dot(dij, yi * sqrtalphai)
  alpham = yi.dot(Si)
  bm = yi.dot(bi)
  A = alpham * PRT / RT
  B = bm * PRT
  Z = jsolve(A, B)
  vsi = bi * (s0i + s1i * (T - Tri))
  return Z - yi.dot(vsi) * PRT

def jdG(Z1, Z2, A, B):
  d1 = -0.414213562373095
  d2 = 2.414213562373095
  return (jnp_log((Z2 - B) / (Z1 - B))
          + (Z1 - Z2)
          + A / (B * (d2 - d1)) * jnp_log((Z1 + d1 * B) * (Z2 + d2 * B)
                                          / ((Z1 + d2 * B) * (Z2 + d1 * B))))

def jsolve(A, B):
  b = B - 1.
  c = A - B * (2. + 3. * B)
  d = B * (B * (1. + B) - A)
  r = b * b
  p = c + r / -3.
  q = d + b * (2. * r - 9. * c) / 27.
  s = q * q * .25 + p * p * p / 27.
  if s >= 0.:
    s = jnp_sqrt(s)
    x = jnp_cbrt(-.5 * q + s) + jnp_cbrt(-.5 * q - s) - b / 3.
    y = d + x * (c + x * (b + x))
    if y > 1e-12 or y < -1e-12:
      x -= y / (c + x * (2. * b + 3. * x))
    return x
  else:
    x0 = (2. * jnp_sqrt(-p / 3.)
          * jnp_cos(jnp_arccos(1.5 * q * jnp_sqrt(-3. / p) / p) / 3.)
          - b / 3.)
    y0 = d + x0 * (c + x0 * (b + x0))
    if y0 > 1e-12 or y0 < -1e-12:
      x0 -= y0 / (c + x0 * (2. * b + 3. * x0))
    r = b + x0
    D = jnp_sqrt(r * r + 4. * d / x0)
    x1 = (-r + D) * .5
    x2 = (-r - D) * .5
    if x2 > B:
      dG = jdG(x0, x2, A, B)
      if dG < 0.:
        return x0
      else:
        return x2
    elif x1 > B:
      dG = jdG(x0, x1, A, B)
      if dG < 0.:
        return x0
      else:
        return x1
    else:
      return x0

def jP(v, T, ni, Pci, Tci, wi, dij, s0i, s1i, Tri, asmoles=True):
  R = 8.31446261815324
  if asmoles:
    yi = ni / ni.sum()
  else:
    yi = ni
  wi2 = wi * wi
  wi3 = wi2 * wi
  kappai = jnp_where(
    wi <= 0.491,
    0.37464 + 1.54226 * wi - 0.26992 * wi2,
    0.379642 + 1.48503 * wi - 0.164423 * wi2 + 0.016666 * wi3,
  )
  sqrtai = 0.6761919320144113 * R * Tci / jnp_sqrt(Pci)
  bi = 0.07779607390388851 * R * Tci / Pci
  multi = 1. + kappai * (1. - jnp_sqrt(T / Tci))
  sqrtalphai = sqrtai * multi
  Si = sqrtalphai * jnp_dot(dij, yi * sqrtalphai)
  alpham = yi.dot(Si)
  bm = yi.dot(bi)
  vsi = bi * (s0i + s1i * (T -  Tri))
  v2p = v + yi.dot(vsi)
  return R * T / (v2p - bm) - alpham / (v2p * v2p + 2. * bm * v2p - bm * bm)


jdZdP = jacfwd(jZ, argnums=0)

jdZdT = jacfwd(jZ, argnums=1)

jd2ZdP2 = jacfwd(jdZdP, argnums=0)

jd2ZdT2 = jacfwd(jdZdT, argnums=1)

jd2ZdPdT = jacfwd(jdZdP, argnums=1)


jdlnphiidP = jacfwd(jlnphii, argnums=0)

jdlnphiidT = jacfwd(jlnphii, argnums=1)

jdlnphiidnj = jacfwd(jlnphii, argnums=2)

jd2lnphiidP2 = jacfwd(jdlnphiidP, argnums=0)

jd2lnphiidT2 = jacfwd(jdlnphiidT, argnums=1)


jdPdv = jacfwd(jP, argnums=0)

jdPdT = jacfwd(jP, argnums=1)

jd2Pdv2 = jacfwd(jdPdv, argnums=0)

jd2PdT2 = jacfwd(jdPdT, argnums=1)

jd2PdvdT = jacfwd(jdPdv, argnums=1)


if __name__ == '__main__':
  unittest.main(verbosity=0)
