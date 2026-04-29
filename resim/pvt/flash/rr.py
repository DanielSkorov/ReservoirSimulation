from logging import (
  getLogger,
)

from functools import (
  partial,
)

from math import (
  isfinite,
)

from numpy import (
  abs as np_abs,
  argmin as np_argmin,
  full as np_full,
  full_like as np_full_like,
  log as np_log,
  sqrt as np_sqrt,
  vstack as np_vstack,
)

from resim.pvt.datatypes import (
  Float,
  Vector,
  Matrix,
)

from resim.pvt.utils import (
  LinearSolver,
  lusolver,
)


logger = getLogger('rr')


class RRConvergenceError(Exception):
  """An exception that should be raised if a phase split solver does not
  converge.
  """
  pass


def fG(
  a: float,
  yi: Vector[Float],
  di: Vector[Float],
) -> tuple[float, float]:
  denom = 1. / (di * (a + 1.) + a)
  return (a + 1.) * yi.dot(denom), -yi.dot(denom * denom)


def fH(
  a: float,
  yi: Vector[Float],
  di: Vector[Float],
) -> tuple[float, float]:
  denom = 1. / (di * (a + 1.) + a)
  G = (a + 1.) * yi.dot(denom)
  dGda = -yi.dot(denom * denom)
  return -a * G, -G - a * dGda


def fD(
  a: float,
  yi: Vector[Float],
  di: Vector[Float],
  yidi: Vector[Float],
) -> tuple[float, float]:
  denom = 1. / (di * (a + 1.) + a)
  return a * yi.dot(denom), yidi.dot(denom * denom)


def rr2p_fgh(
  kvi: Vector[Float],
  yi: Vector[Float],
  f0: float | None = None,
  tol: float = 1e-10,
  maxiter: int = 50,
  miniter: int = 1,
) -> float:
  """Solve the Rachford-Rice equation using the FGH-method.

  For details of the FGH-method see the paper of D.V. Nichita and
  C.F. Leibovici, 2013 (doi: 10.1016/j.fluid.2013.05.030).

  Parameters
  ----------
  kvi: Vector[Float], shape (Nc,)
    K-values of `Nc` components. At least one of them should be greater
    than one, and at least one should be lower than one.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  f0: float | None
    An initial guess for the mole fraction of the non-reference phase.
    Default is `None`, which means using an internal formula based on
    the paper of D.V. Nichita and C.F. Leibovici, 2017 (doi:
    10.1016/j.fluid.2017.08.020).

  tol: float
    Terminate successfully if the absolute value of the D-function
    is less than `tol`. Default is `1e-10`.

  maxiter: int
    The maximum number of iterations. Default is `50`.

  miniter: int
    The minimum number of iterations. Default is `1`.

  Returns
  -------
  The mole fraction of the non-reference phase.

  Raises
  ------
  `RRConvergenceError` if the solver does not converge.
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
    logger.info('Mole fraction of the non-reference phase: %.4f.', f)
    return f
  logger.warning(
    'FGH-method for solving the RR-equation terminates unsuccessfully.\n'
    'kvi = %s\nyi = %s\nf0 = %s', kvi.tolist(), yi.tolist(), f0,
  )
  raise RRConvergenceError(
    'The solution of the Rachford-Rice equation corresponding to the '
    'negative-flash window was not found using the FGH-method. Try to '
    'increase the maximum number of iterations or check the input data.'
  )


def rr2p_gh(
  kvi: Vector[Float],
  yi: Vector[Float],
  f0: float | None = None,
  tol: float = 1e-10,
  maxiter: int = 50,
) -> float:
  """Solve the Rachford-Rice equation using the GH-method.

  For details of the GH-method see the paper of D.V. Nichita and
  C.F. Leibovici, 2013 (doi: 10.1016/j.fluid.2013.05.030).

  Parameters
  ----------
  kvi: Vector[Float], shape (Nc,)
    K-values of `Nc` components. At least one of them should be greater
    than one, and at least one should be lower than one.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  f0: float | None
    An initial guess for the mole fraction of the non-reference phase.
    Default is `None`, which means using an internal formula based on
    the paper of D.V. Nichita and C.F. Leibovici, 2017 (doi:
    10.1016/j.fluid.2017.08.020).

  tol: float
    Terminate successfully if the absolute value of the D-function
    is less than `tol`. Default is `1e-10`.

  maxiter: int
    The maximum number of iterations. Default is `50`.

  Returns
  -------
  The mole fraction of the non-reference phase.

  Raises
  ------
  `RRConvergenceError` if the solver does not converge.
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
    logger.info('Mole fraction of the non-reference phase: %.4f.', f)
    return f
  logger.warning(
    'GH-method for solving the RR-equation terminates unsuccessfully.\n'
    'kvi = %s\nyi = %s\nf0 = %s', kvi.tolist(), yi.tolist(), f0,
  )
  raise RRConvergenceError(
    'The solution of the Rachford-Rice equation corresponding to the '
    'negative-flash window was not found using the GH-method. Try to '
    'increase the maximum number of iterations or check the input data.'
  )


def rrNp(
  Kji: Matrix[Float],
  yi: Vector[Float],
  fj0: Vector[Float] | None,
  tol: float = 1e-20,
  maxiter: int = 30,
  beta: float = 0.8,
  c: float = 0.3,
  maxiter_ls: int = 10,
  linsolver: LinearSolver = lusolver,
) -> Vector[Float]:
  r"""Solve the system of Rachford-Rice equations.

  This function implements Okuno's method for solving systems of
  Rachford-Rice equations. It is based on optimization using Newton's
  method with the backtracking line search technique to prevent leaving
  the feasible region. For the details, see the paper of R. Okuno et al,
  2010 (doi: 10.2118/117752-PA).

  Parameters
  ----------
  Kji: Matrix[Float], shape (Np - 1, Nc)
    K-values of `Nc` components in `Np - 1` phases.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  fj0: Vector[Float], shape (Np - 1,) | None
    An nitial guess for phase mole fractions. If it is `None`, each mole
    fraction will be set to `1.0 / Np`.

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

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `(Np - 1, Np - 1)` and a `b: Vector[Float]` of shape `(Np - 1,)` and
    finds `x: Vector[Float]` of shape `(Np - 1,)`, which is the solution
    to the linear system :math:`\mathbf{A}^\top \mathbf{x}= \mathbf{b}`.
    The matrix `A` is a symmetric and positive definite matrix within
    the negative flash window. Default is `lusolver`.

  Returns
  -------
  A `Vector[Float]` of shape `(Np - 1,)` of mole fractions of non-
  reference phases.

  Raises
  ------
  `RRConvergenceError` if the solver does not converge.
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
  Bji = np_sqrt(yi) * Aji
  bi = np_vstack([Kji * yi, yi]).max(axis=0)
  k = 0
  n = 0
  if fj0 is None:
    fjk = np_full((Npm1,), 1 / (Npm1 + 1))
    ti = 1. - fjk.dot(Aji)
  else:
    fjk = fj0.flatten()
    ti = 1. - fjk.dot(Aji)
    if (ti < 0.).any():
      # TODO: Find and implement an initialization scheme which
      #       ensures that the starting point is inside the negative
      #       flash window.
      fjk = np_full_like(fjk, 1 / (Npm1 + 1))
      ti = 1. - fjk.dot(Aji)
  F = - np_log(np_abs(ti)).dot(yi)
  gj = Aji.dot(yi / ti)
  g2 = gj.dot(gj)
  logger.debug(tmpl, k, n, *fjk, F, g2)
  while g2 > tol and k < maxiter:
    Pji = Bji / ti
    Hjl = Pji.dot(Pji.T)
    # TODO: Replace the LU-solver with the custom implementation of the
    #       modified Cholesky decomposition solver. Can it be used to
    #       correct the initial guess if it is not in the negative flash
    #       window?
    dfj = -linsolver(Hjl, gj)
    denom = dfj.dot(Aji)
    where = denom > 0.
    lmbdi = ((ti - bi) / denom)[where]
    if (lmbdi < 0.).any():
      lmbdi = (ti / denom)[where]
    if lmbdi.size:
      lmbdmax = lmbdi[np_argmin(lmbdi)]
    else:
      logger.debug(
        'The initial guess lies outside the negative flash window.\n'
        'kvji:\n%s\nyi = %s\nfj0 = %s', Kji, yi, fj0,
      )
      raise RRConvergenceError(
        'The initial guess lies outside the negative flash window.'
      )
    if lmbdmax < 1.:
      gdf = gj.dot(dfj)
      lmbdn = beta * lmbdmax
      fjkp1 = fjk + lmbdn * dfj
      ti = 1. - fjkp1.dot(Aji)
      Fkp1 = - np_log(np_abs(ti)).dot(yi)
      n = 1
      logger.debug(tmpl, k, n, *fjkp1, Fkp1, g2)
      while Fkp1 > F + c * lmbdn * gdf and n < maxiter_ls:
        lmbdn *= beta
        fjkp1 = fjk + lmbdn * dfj
        ti = 1. - fjkp1.dot(Aji)
        Fkp1 = - np_log(np_abs(ti)).dot(yi)
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
      F = - np_log(np_abs(ti)).dot(yi)
      gj = Aji.dot(yi / ti)
      g2 = gj.dot(gj)
    k += 1
    logger.debug(tmpl, k, n, *fjk, F, g2)
  if g2 < tol and isfinite(g2):
    logger.info(
      'Mole fractions of non-reference phases: [' + Npm1 * ' %.4f' + '].',
      *fjk,
    )
    return fjk
  logger.warning(
    'Solving the system of RR-equations was completed unsuccessfully.\n'
    'kvji:\n%s\nyi = %s\nfj0 = %s', Kji, yi, fj0,
  )
  raise RRConvergenceError(
    'Solving the system of RR-equations was completed unsuccessfully.\n'
    'Try to increase the maximum number of iterations or check your data.'
  )
