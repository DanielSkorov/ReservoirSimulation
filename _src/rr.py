import logging

from functools import (
  partial,
)

import numpy as np

from utils import (
  linsolver2d,
)

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
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
  tol: ScalarType = np.float64(1e-8),
  Niter: int = 50,
) -> ScalarType:
  """FGH-method for solving the Rachford-Rice equation.

  Solves the Rachford-Rice equation for two-phase systems using
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
    'Solving the RR-equation using the FGH-method\n\tkvi = %s\n\tyi = %s',
    kvi, yi,
  )
  idx = kvi.argsort()[::-1]
  ysi = yi[idx]
  kvsi = kvi[idx]
  ci = 1. / (1. - kvsi)
  di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
  y0 = ysi[0]
  yN = ysi[-1]
  ysi = ysi[1:-1]
  pD = partial(fD, yi=ysi, di=di, yidi=ysi*di, y0=y0, yN=yN)
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
  tol: ScalarType = np.float64(1e-8),
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
    'Solving the RR-equation using the GH-method\n\tkvi = %s\n\tyi = %s',
    kvi, yi,
  )
  idx = kvi.argsort()[::-1]
  ysi = yi[idx]
  kvsi = kvi[idx]
  ci = 1. / (1. - kvsi)
  di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
  y0 = ysi[0]
  yN = ysi[-1]
  ysi = ysi[1:-1]
  k = 0
  ak = y0 / yN
  denom = 1. / (di * (ak + 1.) + ak)
  eq = (ak + 1.) * (y0 / ak + ysi.dot(denom) - yN)
  deqda = -y0 / (ak * ak) - ysi.dot(denom * denom) - yN
  if eq > 0.:
    logger.debug('Use G-formulation')
    peq = partial(fG, yi=ysi, di=di, y0=y0, yN=yN)
  else:
    logger.debug('Use H-formulation')
    peq = partial(fH, yi=ysi, di=di, y0=y0, yN=yN)
    deqda = -eq - ak * deqda
    eq *= -ak
  hk = eq / deqda
  logger.debug('Iteration #%s:\n\ta = %s\n\teq = %s', 0, ak, eq)
  while (eq > tol) & (k < Niter):
    k +=1
    ak -= hk
    eq, deqda = peq(ak)
    logger.debug('Iteration #%s:\n\ta = %s\n\teq = %s', k, ak, eq)
    hk = eq / deqda
  F = (ci[0] + ak * ci[-1]) / (1. + ak)
  logger.debug('Solution:\n\tF = %s\n', F)
  return F


def solveNp(
  Kji: MatrixType,
  yi: VectorType,
  fj0: VectorType,
  tol: ScalarType = np.float64(1e-6),
  Niter: int = 30,
  beta: ScalarType = np.float64(0.8),
  c: ScalarType = np.float64(0.3),
):
  """Solves the system of Rachford-Rice equations

  Implementation of Okuno's method for solving systems of Rachford-Rice
  equations. This method is based on Newton's method for optimization
  and the backtracking line search technique to prevent leaving
  the feasible region. For the details see 10.2118/117752-PA.

  Arguments
  ---------

    Kji : numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]
      K-values of `Nc` components in `Np-1` phases.

    yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Mole fractions of `(Nc,)` components.

    fj0 : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Initial guess for phase mole fractions.

    tol : numpy.float64
      Tolerance.

    Niter : int
      Maximum number of iterations.

    beta : numpy.float64
      Coefficient used to update step size in the backtracking line
      search procedure.

    c : numpy.float64
      Coefficient used to calculate the Goldstein's condition for the
      backtracking line search procedure.

  Returns a vector of mole fractions of non-reference phases in a system.
  """
  logger.debug(
    "Solving the system of RR-equations using Okuno's method"
    "\n\tkvji = %s\n\tyi = %s",
    Kji, yi,
  )
  assert Kji.shape[0] > 1
  if Kji.shape[0] == 2:
    linsolver = linsolver2d
  else:
    linsolver = np.linalg.solve
  Aji = 1. - Kji
  Bji = np.sqrt(yi) * Aji
  bi = np.vstack([Kji * yi, yi]).max(axis=0)
  fjk = fj0
  ti = 1. - fjk.dot(Aji)
  F = - np.log(np.abs(ti)).dot(yi)
  gj = Aji.dot(yi / ti)
  gnorm = np.linalg.norm(gj)
  if gnorm < tol:
    return fjk
  logger.debug(
    'Iteration #0:\n\tFj = %s\n\tgnorm = %s', fjk, gnorm,
  )
  k: int = 1
  while (gnorm > tol) & (k < Niter):
    Pji = Bji / ti
    Hjl = Pji.dot(Pji.T)
    dfj = -linsolver(Hjl, gj)
    denom = dfj.dot(Aji)
    where = denom > 0.
    lmbdi = ((ti - bi) / denom)[where]
    idx = np.argmin(lmbdi)
    lmbdmax = lmbdi[idx]
    if lmbdmax < 1.:
      logger.debug('Run LS:\n\t\tFk = %s\n\t\tlmbd_max = %s', F, lmbdmax)
      gdf = gj.dot(dfj)
      lmbdn = beta * lmbdmax
      fjkp1 = fjk + lmbdn * dfj
      ti = 1. - fjkp1.dot(Aji)
      Fkp1 = - np.log(np.abs(ti)).dot(yi)
      n: int = 1
      logger.debug(
        '\tLS-Iteration #%s:\n\t\tlmbd = %s\n\t\tFkp1 = %s',
        n, lmbdn, Fkp1,
      )
      while Fkp1 > F + c * lmbdn * gdf:
        lmbdn *= beta
        fjkp1 = fjk + lmbdn * dfj
        ti = 1. - fjkp1.dot(Aji)
        Fkp1 = - np.log(np.abs(ti)).dot(yi)
        n += 1
        logger.debug(
          '\tLS-Iteration #%s:\n\t\tlmbd = %s\n\t\tFkp1 = %s',
          n, lmbdn, Fkp1,
        )
      fjk = fjkp1
      F = Fkp1
      gj = Aji.dot(yi / ti)
      gnorm = np.linalg.norm(gj)
    else:
      fjk += dfj
      ti = 1. - fjk.dot(Aji)
      F = - np.log(np.abs(ti)).dot(yi)
      gj = Aji.dot(yi / ti)
      gnorm = np.linalg.norm(gj)
    logger.debug('Iteration #%s:\n\tFj = %s\n\tgnorm = %s', k, fjk, gnorm)
    k += 1
  return fjk
