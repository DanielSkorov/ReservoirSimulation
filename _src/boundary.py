import logging

import numpy as np

from utils import (
  mineig_rayquot,
)

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EOSVTType,
)


logger = logging.getLogger('bound')


def getVT_Tspinodal(
  V: ScalarType,
  yi: VectorType,
  eos: EOSVTType,
  T0: None | ScalarType = None,
  zeta0i : None | VectorType = None,
  multdT0: ScalarType = np.float64(1e-5),
  tol: ScalarType = np.float64(1e-5),
  Niter: int = 25,
) -> tuple[ScalarType, VectorType]:
  """Calculates the spinodal temperature for a given volume
  and composition of a mixture.

  Arguments:
  ----------
    V : numpy.float64
      Volume of a mixture [m3].

    yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Mole fractions of `(Nc,)` components.

    eos : EOSVTType
      An initialized instance of a VT-based equation of state.

    T0 : None | numpy.float64
      Initial guess of a spinodal temperature [K]. If it equals `None`
      a pseudocritical temperature will used.

    zeta0i : None | numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Initial guess for a eigenvector corresponding to the lowest
      eigenvalue of the matrix of second partial derivatives of the
      Helmholtz energy function with respect to component mole
      numbers. If it equals `None` then `yi` will be used instead.

    multdT0 : numpy.float64
      Multiplier used to compute the temperature shift to estimate
      the partial derivative of the lowest eigenvalue with respect to
      temperature in the first iteration. The default is
      `multdT0 = 1e-5`.

    tol : numpy.float64
      Tolerance for the lowest eigenvalue equation. The default is
      `tol = 1e-5`.

    Niter : int
      Maximum number of solver iterations. The default is
      `Niter = 25`.

  Returns a tuple of the spinodal temperature and the eigenvector.
  """
  logger.debug(
    'The spinodal temperature calculation procedure\n\tV = %s m3\n\tyi = %s',
    V, yi,
  )
  k: int = 0
  if T0 is None:
    Tk = 1.3 * yi.dot(eos.Tci)
  else:
    Tk = T0
  if zeta0i is None:
    zeta0i = yi
  Q = eos.getVT_lnfi_dnj(V, Tk, yi)[1]
  Nitrq, zetai, lmbdk = mineig_rayquot(Q, zeta0i)
  dT = multdT0 * Tk
  Qs = eos.getVT_lnfi_dnj(V, Tk + dT, yi)[1]
  lmbdkdT = np.linalg.eigvals(Qs).min()
  dlmbddT = (lmbdkdT - lmbdk) / dT
  dT = -lmbdk / dlmbddT
  logger.debug(
    'Iteration #%s:\n\tlmbd = %s\n\tzetai = %s\n\tT = %s\n\tdT = %s',
    k, lmbdk, zetai, Tk, dT,
  )
  k += 1
  while (np.abs(lmbdk) > tol) & (k < Niter):
    Tkp1 = Tk + dT
    Q = pr.getVT_lnfi_dnj(V, Tkp1, yi)[1]
    Nitrq, zetai, lmbdkp1 = mineig_rayquot(Q, zetai)
    dlmbddT = (lmbdkp1 - lmbdk) / dT
    dT = -lmbdkp1 / dlmbddT
    Tk = Tkp1
    lmbdk = lmbdkp1
    logger.debug(
      'Iteration #%s:\n\tlmbd = %s\n\tzetai = %s\n\t\n\tT = %s\n\tdT = %s',
      k, lmbdk, zetai, Tk, dT,
    )
    k += 1
  return Tk, zetai


def getVT_PcTc(
  self,
  yi: VectorType,
  eos: EOSVTType,
  v0: None | ScalarType = None,
  T0: None | ScalarType = None,
  kappa0: ScalarType = np.float64(3.5),
  multdV0: ScalarType = np.float64(1e-5),
  krange: tuple[ScalarType, ScalarType] (np.float64(1.1), np.float64(5.)),
  tol: ScalarType = np.float64(1e-5),
  Niter: int = 25,
) -> tuple[ScalarType, ScalarType]:
  """
  """
  logger.debug('The critical point calculation procedure\n\tyi = %s', yi)
  k: int = 0
  if T0 is None:
    T = 1.3 * yi.dot(eos.Tci)
  else:
    T = T0
  if v0 is None:
    if kappa0 is None:
      raise NotImplementedError(
        'The gridding procedure is not implemented yet.'
      )
    else:
      vmin = eos.getVT_vmin(T, yi)
      vk = kappa0 * vmin
  else:
    vk = v0
  T, zetaik = getVT_Tspinodal(vk, yi, eos, T, yi)
  vmin = eos.getVT_vmin(T, yi)
  kappak = vk / vmin
  Ck = (kappak - 1.)**2 * eos.getVT_d3F(vk, T, yi, zetaik)
  dv = multdV0 * vk
  vks = vk + dv
  kappaks = vks / vmin
  Cks = (kappaks - 1.)**2 * eos.getVT_d3F(vks, T, yi, zetaik)
  dCdkappa = (Cks - Ck) / (kappaks - kappak)
  dkappa = -Ck / dCdkappa
  logger.debug(
    'Iteration #%s:\n\tkappa = %s\n\tT = %s\n\tC = %s\n\tdkappa = %s',
    k, kappak, T, Ck, dkappa,
  )
  k += 1
  while (np.abs(dkappa) > tol) & (k < Niter):
    kappakp1 = kappak + dkappa
    vkp1 = kappakp1 * vmin
    if vkp1 < vmin:
      vkp1 = (vk + vmin) / 2.
      kappakp1 = vkp1 / vmin
      dkappa = kappakp1 - kappak
    T, zetaikp1 = getVT_Tspinodal(vkp1, yi, eos, T, zetaik)
    if zetaikp1[0] * zetaik[0] < 0.:
      zetaikp1 *= -1.
    Ckp1 = (kappakp1 - 1.)**2 * eos.getVT_d3F(vkp1, T, yi, zetaikp1)
    dCdkappa = (Ckp1 - Ck) / dkappa
    dkappa = -Cjp1 / dCdkappa
    vmin = eos.getVT_vmin(T, yi)
    vk = vkp1
    kappak = kappakp1
    Ck = Ckp1
    zetaik = zetaikp1
    logger.debug(
      'Iteration #%s:\n\tkappa = %s\n\tT = %s\n\tC = %s\n\tdkappa = %s',
      k, kappak, T, Ck, dkappa,
    )
    k += 1
  return eos.getVT_P(vk, T, yi), T
