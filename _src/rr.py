import logging

from functools import (
  partial,
)

import numpy as np

from customtypes import (
  Double,
  Vector,
  Matrix,
  Linsolver,
  SolutionNotFoundError,
)


logger = logging.getLogger('rr')


def fG(
  a: float,
  yi: Vector[Double],
  di: Vector[Double],
) -> tuple[float, float]:
  denom = 1. / (di * (a + 1.) + a)
  return (a + 1.) * yi.dot(denom), -yi.dot(denom * denom)


def fH(
  a: float,
  yi: Vector[Double],
  di: Vector[Double],
) -> tuple[float, float]:
  denom = 1. / (di * (a + 1.) + a)
  G = (a + 1.) * yi.dot(denom)
  dGda = -yi.dot(denom * denom)
  return -a * G, -G - a * dGda


def fD(
  a: float,
  yi: Vector[Double],
  di: Vector[Double],
  yidi: Vector[Double],
) -> tuple[float, float]:
  denom = 1. / (di * (a + 1.) + a)
  return a * yi.dot(denom), yidi.dot(denom * denom)


def solve2p_FGH(
  kvi: Vector[Double],
  yi: Vector[Double],
  f0: float | None = None,
  tol: float = 1e-12,
  maxiter: int = 50,
  miniter: int = 1,
) -> float:
  """FGH-method for solving the Rachford-Rice equation.

  Solves the Rachford-Rice equation for a two-phase system using
  the FGH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Parameters
  ----------
  kvi: Vector[Double], shape (Nc,)
    K-values of `Nc` components.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  f0: float | None
    The initial guess for the mole fraction of the non-reference phase.
    Default is `None`, which means using an internal formula based on
    the paper 10.1016/j.fluid.2017.08.020.

  tol: float
    Terminate successfully if the absolute value of the D-function
    is less than `tol`. Default is `1e-12`.

  maxiter: int
    The maximum number of iterations. Default is `50`.

  miniter: int
    The minimum number of iterations. Default is `1`.

  Returns
  -------
  The mole fraction of the non-reference phase in the system.
  """
  logger.info('Solving the two-phase Rachford-Rice equation (FGH-method).')
  logger.debug('%3s%12s%11s', 'Nit', 'a', 'eq')
  tmpl = '%3s%12.3e%11.2e'
  idxmin = kvi.argmin()
  idxmax = kvi.argmax()
  ci = 1. / (1. - kvi)
  cmin = ci[idxmin]
  cmax = ci[idxmax]
  di = (cmax - ci) / (cmin - cmax)
  pD = partial(fD, yi=yi, di=di, yidi=yi*di)
  k = 0
  ak = yi[idxmax] / yi[idxmin]
  if f0 is None:
    ak = yi[idxmax] / yi[idxmin]
  else:
    ak = (f0 - cmax) / (cmin - f0)
    if ak < 0.:
      ak = yi[idxmax] / yi[idxmin]
  D, dDda = pD(ak)
  repeat = D < -tol or D > tol
  logger.debug(tmpl, k, ak, D)
  while (repeat or k < miniter) and k < maxiter:
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
    f = (cmax + ak * cmin) / (1. + ak)
    logger.info('Solution is: f = %.4f.', f)
    return f
  logger.warning(
    'FGH-method for solving the RR-equation terminates unsuccessfully.\n'
    'kvi = %s\nyi = %s', kvi.tolist(), yi.tolist(),
  )
  raise SolutionNotFoundError(
    'FGH-method for solving the RR-equation terminates unsuccessfully.\n'
    'Try to increase the maximum number of iterations or check your data.'
  )


def solve2p_GH(
  kvi: Vector[Double],
  yi: Vector[Double],
  f0: float | None = None,
  tol: float = 1e-12,
  maxiter: int = 50,
) -> float:
  """GH-method for solving the Rachford-Rice equation.

  Solves the Rachford-rice equation for a two-phase system using
  the GH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Parameters
  ----------
  kvi: Vector[Double], shape (Nc,)
    K-values of `Nc` components.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  f0: float | None
    The initial guess for the mole fraction of the non-reference phase.
    Default is `None`, which means using an internal formula based on
    the paper 10.1016/j.fluid.2017.08.020.

  tol: float
    Terminate successfully if the absolute value of the D-function
    is less than `tol`. Default is `1e-12`.

  maxiter: int
    The maximum number of iterations. Default is `50`.

  Returns
  -------
  The mole fraction of the non-reference phase in the system.
  """
  logger.info('Solving the two-phase Rachford-Rice equation (GH-method).')
  logger.debug('%3s%12s%11s', 'Nit', 'a', 'eq')
  tmpl = '%3s%12.3e%11.2e'
  idxmin = kvi.argmin()
  idxmax = kvi.argmax()
  ci = 1. / (1. - kvi)
  cmin = ci[idxmin]
  cmax = ci[idxmax]
  di = (cmax - ci) / (cmin - cmax)
  k = 0
  if f0 is None:
    ak = yi[idxmax] / yi[idxmin]
  else:
    ak = (f0 - cmax) / (cmin - f0)
    if ak < 0.:
      ak = yi[idxmax] / yi[idxmin]
  denom = 1. / (di * (ak + 1.) + ak)
  eq = (ak + 1.) * yi.dot(denom)
  deqda = -yi.dot(denom * denom)
  if eq > 0.:
    peq = partial(fG, yi=yi, di=di)
  else:
    peq = partial(fH, yi=yi, di=di)
    deqda = -eq - ak * deqda
    eq *= -ak
  logger.debug(tmpl, k, ak, eq)
  while (eq > tol or eq < -tol) and k < maxiter:
    hk = eq / deqda
    k +=1
    ak -= hk
    eq, deqda = peq(ak)
    logger.debug(tmpl, k, ak, eq)
  if eq < tol:
    f = (cmax + ak * cmin) / (1. + ak)
    logger.info('Solution is: f = %.4f.', f)
    return f
  logger.warning(
    'GH-method for solving the RR-equation terminates unsuccessfully.\n'
    'kvi = %s\nyi = %s', kvi.tolist(), yi.tolist(),
  )
  raise SolutionNotFoundError(
    'GH-method for solving the RR-equation terminates unsuccessfully.\n'
    'Try to increase the maximum number of iterations or check your data.'
  )


def solveNp(
  Kji: Matrix[Double],
  yi: Vector[Double],
  fj0: Vector[Double],
  tol: float = 1e-20,
  maxiter: int = 30,
  beta: float = 0.8,
  c: float = 0.3,
  maxiter_ls: int = 10,
  linsolver: Linsolver = np.linalg.solve,
) -> Vector[Double]:
  """Solves the system of Rachford-Rice equations.

  Implementation of Okuno's method for solving systems of Rachford-Rice
  equations. This method is based on Newton's method for optimization
  and the backtracking line search technique to prevent leaving
  the feasible region. For the details see 10.2118/117752-PA.

  Parameters
  ----------
  Kji: Matrix[Double], shape (Np - 1, Nc)
    K-values of `Nc` components in `Np-1` phases.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  fj0: Vector[Double], shape (Np - 1,)
    Initial guess for phase mole fractions.

  tol: float
    Terminate successfully if the sum of squared elements of the
    gradient is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations. Default is `30`.

  beta: float
    This parameter is used to update step size in the backtracking line
    search procedure. Default is `0.8`.

  c: float
    This parameter is used to calculate the Goldstein's condition for
    the backtracking line search procedure. Default is `0.3`.

  maxiter_ls: int
    The maximum number of linesearch iterations. Default is `10`.

  linsolver: Callable[[Matrix[Double], Vector[Double]], Vector[Double]]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  A vector of mole fractions of non-reference phases in the system.
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
  fjk = fj0.flatten()
  ti = 1. - fjk.dot(Aji)
  if (ti < 0.).any():
    fjk = np.full_like(fjk, 1 / (Npm1 + 1))
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
    lmbdi = ((ti - bi) / denom)[where]
    if (lmbdi < 0.).any():
      lmbdi = (ti / denom)[where]
    lmbdmax = lmbdi[np.argmin(lmbdi)]
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
