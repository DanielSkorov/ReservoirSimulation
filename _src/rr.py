import logging

from functools import (
  partial,
)

import numpy as np

from custom_types import (
  ScalarType,
  VectorType,
)


logger = logging.getLogger('rr')


def fG(
  a: ScalarType,
  yi: VectorType,
  di: VectorType,
  y0: ScalarType,
  yN: ScalarType,
) -> tuple[ScalarType, ScalarType]:
  denom = 1. / (di * (a + 1.) + a)
  return (
    (a + 1.) * (y0 / a + yi.dot(denom) - yN),
    -y0 / (a * a) - yi.dot(denom * denom) - yN,
  )


def fH(
  a: ScalarType,
  yi: VectorType,
  di: VectorType,
  y0: ScalarType,
  yN: ScalarType,
) -> tuple[ScalarType, ScalarType]:
  denom = 1. / (di * (a + 1.) + a)
  G = (1. + a) * (y0 / a + yi.dot(denom) - yN)
  dGda = -y0 / (a * a) - yi.dot(denom * denom) - yN
  return -a * G, -G - a * dGda


def fD(
  a: ScalarType,
  yi: VectorType,
  di: VectorType,
  yidi: VectorType,
  y0: ScalarType,
  yN: ScalarType,
) -> tuple[ScalarType, ScalarType]:
  denom = 1. / (di * (a + 1.) + a)
  return y0 + a * yi.dot(denom) - yN * a, yidi.dot(denom * denom) - yN


def solve2p_FGH(
  kvi: VectorType,
  yi: VectorType,
  tol: ScalarType = np.float64(1e-10),
  Niter: int = 50,
) -> ScalarType:
  """FGH-method for solving the Rachford-Rice equation.

  Solves the Rachford-rice equation for two-phase systems using
  the FGH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Arguments
  ---------
    kvi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      K-values of `(Nc,)` components.

    yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Mole fractions of `(Nc,)` components.

    tol : numpy.float64
      Tolerance.

    Niter : int
      Maximum number of iterations.

  Returns a mole fraction of the non-reference phase in a system.
  """
  logger.debug(
    'Solving RR-equation using FGH-method\n\tkvi = %s\n\tyi = %s', kvi, yi,
  )
  idx = kvi.argsort()[::-1]
  yi = yi[idx]
  kvi = kvi[idx]
  ci = 1. / (1. - kvi)
  di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
  y0 = yi[0]
  yN = yi[-1]
  yi = yi[1:-1]
  pD = partial(fD, yi=yi, di=di, yidi=yi*di, y0=y0, yN=yN)
  k = 0
  ak = y0 / yN
  D, dDda = pD(ak)
  hk = D / dDda
  logger.debug('Iteration #%s:\n\ta = %s\n\tD = %s', 0, ak, D)
  while (np.abs(D) > tol) & (k < Niter):
    akp1 = ak - hk
    if akp1 < 0.:
      if D > 0.:
        akp1 += hk * hk / (hk - ak * (ak + 1.))
      else:
        akp1 += hk * hk / (hk + ak + 1.)
    k += 1
    ak = akp1
    D, dDda = pD(ak)
    logger.debug('Iteration #%s:\n\ta = %s\n\tD = %s', k, ak, D)
    hk = D / dDda
  F = (ci[0] + ak * ci[-1]) / (1. + ak)
  logger.debug('Solution:\n\tF = %s\n', F)
  return F


def solve2p_GH(
  kvi: VectorType,
  yi: VectorType,
  tol: ScalarType = np.float64(1e-10),
  Niter: int = 50,
) -> ScalarType:
  """GH-method for solving the Rachford-Rice equation.

  Solves the Rachford-rice equation for two-phase systems using
  the GH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Arguments
  ---------

    kvi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      K-values of `(Nc,)` components.

    yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Mole fractions of `(Nc,)` components.

    tol : numpy.float64
      Tolerance.

    Niter : int
      Maximum number of iterations.

  Returns a mole fraction of the non-reference phase in a system.
  """
  logger.debug(
    'Solving RR-equation using GH-method\n\tkvi = %s\n\tyi = %s', kvi, yi,
  )
  idx = kvi.argsort()[::-1]
  yi = yi[idx]
  kvi = kvi[idx]
  ci = 1. / (1. - kvi)
  di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
  y0 = yi[0]
  yN = yi[-1]
  yi = yi[1:-1]
  k = 0
  ak = y0 / yN
  denom = 1. / (di * (ak + 1.) + ak)
  eq = (ak + 1.) * (y0 / ak + yi.dot(denom) - yN)
  deqda = -y0 / (ak * ak) - yi.dot(denom * denom) - yN
  if eq > 0.:
    logger.debug('Use G-formulation')
    peq = partial(fG, yi=yi, di=di, y0=y0, yN=yN)
  else:
    logger.debug('Use H-formulation')
    peq = partial(fH, yi=yi, di=di, y0=y0, yN=yN)
    deqda = -eq - ak * deqda
    eq *= -ak
  hk = eq / deqda
  logger.debug('Iteration #%s:\n\ta = %s\n\teq = %s', 0, ak, eq)
  while (eq > tol) & (k < Niter):
    akp1 = ak - hk
    k +=1
    ak = akp1
    eq, deqda = peq(ak)
    logger.debug('Iteration #%s:\n\ta = %s\n\teq = %s', k, ak, eq)
    hk = eq / deqda
  F = (ci[0] + ak * ci[-1]) / (1. + ak)
  logger.debug('Solution:\n\tF = %s\n', F)
  return F

