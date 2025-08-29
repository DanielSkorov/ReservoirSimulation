import logging

from functools import (
  partial,
)

import numpy as np

from typing import (
  Callable,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  SolutionNotFoundError,
)


logger = logging.getLogger('rr')


def fG(
  a: Scalar,
  yi: Vector,
  di: Vector,
  y0: Scalar,
  yN: Scalar,
) -> tuple[Scalar, Scalar]:
  denom = 1. / (di * (a + 1.) + a)
  return (
    (a + 1.) * (y0 / a + yi.dot(denom) - yN),
    -y0 / (a * a) - yi.dot(denom * denom) - yN,
  )


def fH(
  a: Scalar,
  yi: Vector,
  di: Vector,
  y0: Scalar,
  yN: Scalar,
) -> tuple[Scalar, Scalar]:
  denom = 1. / (di * (a + 1.) + a)
  G = (1. + a) * (y0 / a + yi.dot(denom) - yN)
  dGda = -y0 / (a * a) - yi.dot(denom * denom) - yN
  return -a * G, -G - a * dGda


def fD(
  a: Scalar,
  yi: Vector,
  di: Vector,
  yidi: Vector,
  y0: Scalar,
  yN: Scalar,
) -> tuple[Scalar, Scalar]:
  denom = 1. / (di * (a + 1.) + a)
  return y0 + a * yi.dot(denom) - yN * a, yidi.dot(denom * denom) - yN


def solve2p_FGH(
  kvi: Vector,
  yi: Vector,
  tol: Scalar = 1e-12,
  maxiter: int = 50,
) -> Scalar:
  """FGH-method for solving the Rachford-Rice equation.

  Solves the Rachford-Rice equation for two-phase systems using
  the FGH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Parameters
  ----------
  kvi: Vector, shape (Nc,)
    K-values of `Nc` components.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  tol: Scalar
    Terminate successfully if the absolute value of the D-function
    is less than `tol`. Default is `1e-12`.

  maxiter: int
    Maximum number of iterations. Default is `50`.

  Returns
  -------
  A mole fraction of the non-reference phase in a system.
  """
  logger.info('Solving the two-phase Rachford-Rice equation (FGH-method).')
  logger.debug('%3s%12s%11s', 'Nit', 'a', 'eq')
  tmpl = '%3s%12.3e%11.2e'
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
  repeat = D < -tol or D > tol
  logger.debug(tmpl, k, ak, D)
  while repeat and k < maxiter:
    hk = D / dDda
    akp1 = ak - hk
    if akp1 < 0.:
      if D > 0.:
        akp1 += hk * hk / (hk - ak * (ak + 1.))
      else:
        akp1 += hk * hk / (hk + ak + 1.)
    k += 1
    ak = akp1
    D, dDda = pD(ak)
    repeat = D < -tol or D > tol
    logger.debug(tmpl, k, ak, D)
  if not repeat:
    F = (ci[0] + ak * ci[-1]) / (1. + ak)
    logger.info('Solution is: F = %.2f.', F)
    return F
  logger.warning(
    'FGH-method for solving the RR-equation terminates unsuccessfully.\n'
    'kvi = %s\nyi = %s', kvi, yi,
  )
  raise SolutionNotFoundError(
    'FGH-method for solving the RR-equation terminates unsuccessfully.\n'
    'Try to increase the maximum number of iterations or check your data.'
  )


def solve2p_GH(
  kvi: Vector,
  yi: Vector,
  tol: Scalar = 1e-12,
  maxiter: int = 50,
) -> Scalar:
  """GH-method for solving the Rachford-Rice equation.

  Solves the Rachford-rice equation for two-phase systems using
  the GH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Parameters
  ----------
  kvi: Vector, shape (Nc,)
    K-values of `Nc` components.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  tol: Scalar
    Terminate successfully if the absolute value of the D-function
    is less than `tol`. Default is `1e-12`.

  maxiter: int
    Maximum number of iterations. Default is `50`.

  Returns
  -------
  A mole fraction of the non-reference phase in a system.
  """
  logger.info('Solving the two-phase Rachford-Rice equation (GH-method).')
  logger.debug('%3s%12s%11s', 'Nit', 'a', 'eq')
  tmpl = '%3s%12.3e%11.2e'
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
    peq = partial(fG, yi=ysi, di=di, y0=y0, yN=yN)
  else:
    peq = partial(fH, yi=ysi, di=di, y0=y0, yN=yN)
    deqda = -eq - ak * deqda
    eq *= -ak
  logger.debug(tmpl, k, ak, eq)
  while eq > tol and k < maxiter:
    hk = eq / deqda
    k +=1
    ak -= hk
    eq, deqda = peq(ak)
    logger.debug(tmpl, k, ak, eq)
  if eq < tol:
    F = (ci[0] + ak * ci[-1]) / (1. + ak)
    logger.info('Solution is: F = %.2f.', F)
    return F
  logger.warning(
    'GH-method for solving the RR-equation terminates unsuccessfully.\n'
    'kvi = %s\nyi = %s', kvi, yi,
  )
  raise SolutionNotFoundError(
    'GH-method for solving the RR-equation terminates unsuccessfully.\n'
    'Try to increase the maximum number of iterations or check your data.'
  )


def solveNp(
  Kji: Matrix,
  yi: Vector,
  fj0: Vector,
  tol: Scalar = 1e-20,
  maxiter: int = 30,
  beta: Scalar = 0.8,
  c: Scalar = 0.3,
  maxiter_ls: int = 10,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> Vector:
  """Solves the system of Rachford-Rice equations

  Implementation of Okuno's method for solving systems of Rachford-Rice
  equations. This method is based on Newton's method for optimization
  and the backtracking line search technique to prevent leaving
  the feasible region. For the details see 10.2118/117752-PA.

  Parameters
  ----------
  Kji: Matrix, shape (Np - 1, Nc)
    K-values of `Nc` components in `Np-1` phases.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  fj0: Vector, shape (Np - 1,)
    Initial guess for phase mole fractions.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    gradient is less than `tol`. Default is `1e-20`.

  maxiter: int
    Maximum number of iterations. Default is `30`.

  beta: Scalar
    Coefficient used to update step size in the backtracking line
    search procedure. Default is `0.8`.

  c: Scalar
    Coefficient used to calculate the Goldstein's condition for the
    backtracking line search procedure. Default is `0.3`.

  maxiter_ls: int
    Maximum number of linesearch iterations. Default is `10`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  A vector of mole fractions of non-reference phases in a system.
  """
  logger.info("Solving the system of Rachford-Rice equations.")
  Npm1 = Kji.shape[0]
  assert Npm1 > 1
  logger.debug(
    '%3s%5s' + Npm1 * '%11s' + '%11s%11s',
    'Nit', 'Nls', *['f%s' % s for s in range(Npm1)], 'F', 'g2',
  )
  tmpl = '%3s%5s' + Npm1 * '%11.2e' + '%11.2e%11.2e'
  Aji = 1. - Kji
  Bji = np.sqrt(yi) * Aji
  bi = np.vstack([Kji * yi, yi]).max(axis=0)
  k = 0
  n = 0
  fjk = fj0
  ti = 1. - fjk.dot(Aji)
  F = - np.log(np.abs(ti)).dot(yi)
  gj = Aji.dot(yi / ti)
  g2 = gj.dot(gj)
  logger.debug(tmpl, k, n, *fjk, F, g2)
  while g2 > tol and k < maxiter:
    Pji = Bji / ti
    Hjl = Pji.dot(Pji.T)
    dfj = -linsolver(Hjl, gj)
    denom = dfj.dot(Aji)
    where = denom > 0.
    # if where.any():
    lmbdi = ((ti - bi) / denom)[where]
    idx = np.argmin(lmbdi)
    lmbdmax = lmbdi[idx]
    # else:
    #   lmbdmax = 1.
    if lmbdmax < 1.:
      gdf = gj.dot(dfj)
      lmbdn = beta * lmbdmax
      fjkp1 = fjk + lmbdn * dfj
      ti = 1. - fjkp1.dot(Aji)
      Fkp1 = - np.log(np.abs(ti)).dot(yi)
      n = 1
      logger.debug(tmpl, k, n, *fjkp1, Fkp1, g2)
      while Fkp1 > F + c * lmbdn * gdf and n < maxiter_ls:
        lmbdn *= beta
        fjkp1 = fjk + lmbdn * dfj
        ti = 1. - fjkp1.dot(Aji)
        Fkp1 = - np.log(np.abs(ti)).dot(yi)
        n += 1
        logger.debug(tmpl, k, n, *fjkp1, Fkp1, g2)
      fjk = fjkp1
      F = Fkp1
      gj = Aji.dot(yi / ti)
      g2 = gj.dot(gj)
      n = 0
    else:
      fjk += dfj
      ti = 1. - fjk.dot(Aji)
      F = - np.log(np.abs(ti)).dot(yi)
      gj = Aji.dot(yi / ti)
      g2 = gj.dot(gj)
    k += 1
    logger.debug(tmpl, k, n, *fjk, F, g2)
  if g2 < tol:
    logger.info('Solution is: Fj = [' + Npm1 * ' %.4f' + '].', *fjk)
    return fjk
  logger.warning(
    'Solving the system of RR-equations was completed unsuccessfully.\n'
    'kvji:\n%s\nyi = %s\nfj0 = %s', Kji, yi, fj0,
  )
  raise SolutionNotFoundError(
    'Solving the system of RR-equations was completed unsuccessfully.\n'
    'Try to increase the maximum number of iterations or check your data.'
  )
