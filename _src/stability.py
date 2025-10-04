import logging

from dataclasses import (
  dataclass,
)

from functools import (
  partial,
)

import numpy as np

from typing import (
  Protocol,
  Sequence,
)

from customtypes import (
  Logical,
  Double,
  Vector,
  Matrix,
  Linsolver,
)

from eos import (
  Eos,
)


logger = logging.getLogger('stab')


class EosStabPT(Eos, Protocol):

  def getPT_kvguess(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> Sequence[Vector[Double]]: ...

  def getPT_Z_lnphii(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double]]: ...

  def getPT_Z_lnphii_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    n: float,
  ) -> tuple[float, Vector[Double], Matrix[Double]]: ...


@dataclass(eq=False, slots=True)
class StabResult(object):
  """Container for stability test outputs with pretty-printing.

  Attributes
  ----------
  stable: bool | Logical
    A boolean flag indicating if the one-phase state is stable.

  Z: float
    The compressibility factor of the tested mixture at a given
    pressure and temperature.

  kvji: tuple[Vector[Double], ...] | None
    K-values of `Nc` components as a tuple of `Vector[Double]`s of shape
    `(Nc,)`. They may be further used as an initial guess for flash
    calculations. This attribute is `None` if the system will be
    considered stable.
  """
  stable: bool | Logical
  Z: float
  kvji: tuple[Vector[Double], ...] | None

  def __repr__(self) -> str:
    return f"The one-phase state is stable: {self.stable}."


class stabilityPT(object):
  """Stability test based on the Gibbs energy analysis.

  Checks the tangent-plane distance (TPD) at the local minimum of
  the Gibbs energy function.

  Parameters
  ----------
  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: float,T: float, yi: Vector[Double])
       -> Sequence[Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must generate
      initial guesses of k-values as a sequence of `Vector[Double]` of
      shape `(Nc,)`.

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_Z_lnphii_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`) and phase mole number [mol],
      this method must return a tuple of:

      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  method: str
    Type of a solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution),
    - `'newton'` (Newton's method),
    - `'ss-bfgs'` (BFGS method with preceding successive substitution
      iterations for initial guess improvement),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'qnss-newton'`.

  useprev: bool
    Allows to preseve previous calculation results (if the solution
    is non-trivial) and to use them as the first initial guess in next
    run. Default is `False`.

  **kwargs: dict
    Other arguments for a stability test solver. It may contain such
    arguments as `tol`, `maxiter` and others appropriate for the
    selected stability test solver.
  """
  def __init__(
    self,
    eos: EosStabPT,
    method: str = 'qnss-newton',
    useprev: bool = False,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.useprev = useprev
    self.prevkvji: tuple[Vector[Double], ...] | None
    if method == 'ss':
      self.solver = partial(_stabPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_stabPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_stabPT_newt, eos=eos, **kwargs)
    elif method == 'ss-bfgs':
      self.solver = partial(_stabPT_ssbfgs, eos=eos, **kwargs)
    elif method == 'ss-newton':
      self.solver = partial(_stabPT_ssnewt, eos=eos, **kwargs)
    elif method == 'qnss-newton':
      self.solver = partial(_stabPT_qnssnewt, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    kvji0: Sequence[Vector[Double]] | None = None,
  ) -> StabResult:
    """Performs the stability test for a given pressure, temperature
    and composition.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    kvji0: Sequence[Vector[Double]] | None
      A sequence containing 1d-arrays of initial k-value guesses. Each
      array's shape should be `(Nc,)`. Default is `None` which means
      to use initial guesses from the method `getPT_kvguess` of the
      initialized instance of an EOS.

    Returns
    -------
    Stability test results as an instance of `StabResult`. Important
    attributes are:
    - `stab` a boolean flag indicating if a one-phase state is stable,
    - `kvji` a tuple containing vectors of k-values that can be used
      as an initial guess for flash calculations.

    Raises
    ------
    `SolutionNotFoundError` if none of the local minima of the Gibbs
    energy function was found.
    """
    if kvji0 is None:
      kvji = self.eos.getPT_kvguess(P, T, yi)
    else:
      kvji = kvji0
    if self.useprev and self.prevkvji is not None:
      kvji = *self.prevkvji, *kvji
    stab = self.solver(P, T, yi, kvji)
    if self.useprev:
      self.prevkvji = stab.kvji
    return stab


def _stabPT_ss(
  P: float,
  T: float,
  yi: Vector[Double],
  kvji0: Sequence[Vector[Double]],
  eos: EosStabPT,
  tol: float = 1e-20,
  maxiter: int = 500,
  eps: float = -1e-8,
  checktrivial: bool = True,
  breakunstab: bool = False,
) -> StabResult:
  """Successive Substitution (SS) method to perform the stability test
  using a PT-based equation of state.

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[Vector[Double], ...]
    A tuple containing arrays of initial k-value guesses. Each array's
    shape should be `(Nc,)`.

  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate successfully if the sum of squared elements of the
    gradient of the TPD-function is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `500`.

  eps: float
    System will be considered stable when `TPD >= eps`.
    Default is `-1e-8`.

  checktrivial: bool
    A flag indicating whether it is necessary to perform a check for
    early detection of convergence to the trivial solution. It is based
    on the paper of M.L. Michelsen (doi: 10.1016/0378-3812(82)85001-2).
    Default is `True`.

  breakunstab: bool
    A boolean flag indicating whether it is allowed to break a loop
    of various initial guesses checking if the one-phase state was
    identified as unstable. This option should be activated to prevent
    finding the global minimum of the TPD function if a local minimum
    is characterised with the negative value of the TPD function.
    Default is `False`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `kvji` a tuple containing vectors of k-values that can be used as
    an initial guess for flash calculations.
  """
  logger.info('Stability Test (SS-method).')
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%9s' + '%11s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'g2',
  )
  tmpl = '%3s%5s' + Nc * '%9.4f' + '%11.2e'
  Z, lnphiyi = eos.getPT_Z_lnphii(P, T, yi)
  TPDo = eps
  yti = yi
  kvji = None
  for j, ki in enumerate(kvji0):
    k = 0
    trivial = False
    lnki = np.log(ki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    logger.debug(tmpl, j, k, *lnki, g2)
    while g2 > tol and k < maxiter:
      k += 1
      lnki -= gi
      ki = np.exp(lnki)
      ni = ki * yi
      n = ni.sum()
      xi = ni / n
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gi = lnki + lnphixi - lnphiyi
      g2 = gi.dot(gi)
      logger.debug(tmpl, j, k, *lnki, g2)
      if checktrivial:
        if g2 < tol:
          break
        ng = ni.dot(gi)
        tpds = 1. + ng - n
        r = 2. * tpds / (ng - yi.dot(gi))
        if tpds < 1e-3 and r > 0.8 and r < 1.2:
          trivial = True
          break
    if g2 < tol and np.isfinite(g2) and not trivial:
      TPD = -np.log(n)
      if TPD < TPDo:
        TPDo = TPD
        yti = xi
        if breakunstab:
          logger.info('The system is stable: False. TPD = %.3e.', TPDo)
          return StabResult(False, Z, (yti / yi,))
  stable = TPDo >= eps
  logger.info('The system is stable: %s. TPD = %.3e.', stable, TPDo)
  if not stable:
    kvji = (yti / yi,)
  return StabResult(stable, Z, kvji)


def _stabPT_qnss(
  P: float,
  T: float,
  yi: Vector[Double],
  kvji0: Sequence[Vector[Double]],
  eos: EosStabPT,
  tol: float = 1e-20,
  maxiter: int = 200,
  lmbdmax: float = 30.,
  eps: float = -1e-8,
  breakunstab: bool = False,
) -> StabResult:
  """QNSS-method to perform the stability test using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find local minimums of the Gibbs energy function from different
  initial guesses. Checks the TPD-value at a found local minima.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate successfully if the sum of squared elements of the
    gradient of the TPD-function is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `200`.

  lmbdmax: float
    The maximum step length. Default is `30.0`.

  eps: float
    System will be considered stable when `TPD >= eps`.
    Default is `-1e-8`.

  breakunstab: bool
    A boolean flag indicating whether it is allowed to break a loop
    of various initial guesses checking if the one-phase state was
    identified as unstable. This option should be activated to prevent
    finding the global minimum of the TPD function if a local minimum
    is characterised with the negative value of the TPD function.
    Default is `False`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `kvji` a tuple containing vectors of k-values that can be used as
    an initial guess for flash calculations.
  """
  logger.info('Stability Test (QNSS-method).')
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%9s' + '%11s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'g2',
  )
  tmpl = '%3s%5s' + Nc * '%9.4f' + '%11.2e'
  Z, lnphiyi = eos.getPT_Z_lnphii(P, T, yi)
  TPDo = eps
  yti = yi
  kvji = None
  for j, ki in enumerate(kvji0):
    k = 0
    lnki = np.log(ki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    logger.debug(tmpl, j, k, *lnki, g2)
    lmbd = 1.
    while g2 > tol and k < maxiter:
      dlnki = -lmbd * gi
      max_dlnki = np.abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
      k += 1
      tkm1 = dlnki.dot(gi)
      lnki += dlnki
      ki = np.exp(lnki)
      ni = ki * yi
      n = ni.sum()
      xi = ni / n
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gi = lnki + lnphixi - lnphiyi
      g2 = gi.dot(gi)
      logger.debug(tmpl, j, k, *lnki, g2)
      if g2 < tol:
        break
      if k % Nc == 0:
        lmbd = 1.
      else:
        lmbd *= tkm1 / (dlnki.dot(gi) - tkm1)
        if lmbd < 0.:
          lmbd = -lmbd
        if lmbd > lmbdmax:
          lmbd = lmbdmax
    if g2 < tol and np.isfinite(g2):
      TPD = -np.log(n)
      if TPD < TPDo:
        TPDo = TPD
        yti = xi
        if breakunstab:
          logger.info('The system is stable: False. TPD = %.3e.', TPDo)
          return StabResult(False, Z, (yti / yi,))
  stable = TPDo >= eps
  logger.info('The system is stable: %s. TPD = %.3e.', stable, TPDo)
  if not stable:
    kvji = (yti / yi,)
  return StabResult(stable, Z, kvji)


def _stabPT_newt(
  P: float,
  T: float,
  yi: Vector[Double],
  kvji0: Sequence[Vector[Double]],
  eos: EosStabPT,
  tol: float = 1e-20,
  maxiter: int = 50,
  eps: float = -1e-8,
  forcenewton: bool = False,
  breakunstab: bool = False,
  linsolver: Linsolver = np.linalg.solve,
) -> StabResult:
  """Performs minimization of the Michelsen's modified tangent-plane
  distance function using Newton's method and a PT-based equation
  of state. A switch to the successive substitution iteration is
  implemented if Newton's method does not decrease the norm of the
  gradient.

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Z_lnphii_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], mole composition
      (`Vector[Double]` of shape `(Nc,)`), and phase mole number [mol],
      this method must return a tuple of:

      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate Newton's method successfully if the sum of squared
    elements of the gradient of Michelsen's modified tangent-plane
    distance function is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of Newton's method iterations. Default is `50`.

  eps: float
    System will be considered stable when `TPD >= eps`.
    Default is `-1e-8`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  breakunstab: bool
    A boolean flag indicating whether it is allowed to break a loop
    of various initial guesses checking if the one-phase state was
    identified as unstable. This option should be activated to prevent
    finding the global minimum of the TPD function if a local minimum
    is characterised with the negative value of the TPD function.
    Default is `False`.

  linsolver: Linsolver
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `kvji` a tuple containing vectors of k-values that can be used as
    an initial guess for flash calculations.
  """
  logger.info("Stability Test (Newton's method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%9s' + '%11s%9s',
    'Nkv', 'Nit', *['alpha%s' % s for s in range(Nc)], 'g2', 'method',
  )
  tmpl = '%3s%5s' + Nc * '%9.4f' + '%11.2e%9s'
  Z, lnphiyi = eos.getPT_Z_lnphii(P, T, yi)
  hi = lnphiyi + np.log(yi)
  TPDo = eps
  yti = yi
  kvji = None
  for j, ki in enumerate(kvji0):
    k = 0
    ni = ki * yi
    sqrtni = np.sqrt(ni)
    alphai = 2. * sqrtni
    n = ni.sum()
    xi = ni / n
    Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
    gi = np.log(ni) + lnphixi - hi
    g2 = gi.dot(gi)
    logger.debug(tmpl, j, k, *alphai, g2, 'newt')
    while g2 > tol and k < maxiter:
      H = np.diagflat(.5 * gi + 1.) + (sqrtni[:,None] * sqrtni) * dlnphixidnj
      dalphai = linsolver(H, -sqrtni * gi)
      k += 1
      alphai += dalphai
      sqrtni = alphai * .5
      ni = sqrtni * sqrtni
      n = ni.sum()
      xi = ni / n
      Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
      gi = np.log(ni) + lnphixi - hi
      g2kp1 = gi.dot(gi)
      if g2kp1 < g2 or forcenewton:
        g2 = g2kp1
        method = 'newt'
      else:
        ni *= np.exp(-gi)
        sqrtni = np.sqrt(ni)
        alphai = 2. * sqrtni
        n = ni.sum()
        xi = ni / n
        Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
        gi = np.log(ni) + lnphixi - hi
        g2 = gi.dot(gi)
        method = 'ss'
      logger.debug(tmpl, j, k, *alphai, g2, method)
    if g2 < tol and np.isfinite(g2):
      TPD = -np.log(n)
      if TPD < TPDo:
        TPDo = TPD
        yti = xi
        if breakunstab:
          logger.info('The system is stable: False. TPD = %.3e.', TPDo)
          return StabResult(False, Z, (yti / yi,))
  stable = TPDo >= eps
  logger.info('The system is stable: %s. TPD = %.3e.', stable, TPDo)
  if not stable:
    kvji = (yti / yi,)
  return StabResult(stable, Z, kvji)


def _stabPT_ssnewt(
  P: float,
  T: float,
  yi: Vector[Double],
  kvji0: Sequence[Vector[Double]],
  eos: EosStabPT,
  tol: float = 1e-20,
  maxiter: int = 50,
  switchers: tuple[float, float, float] = (0.1, 1e-12, 1e-4),
  eps: float = -1e-8,
  forcenewton: bool = False,
  checktrivial: bool = True,
  breakunstab: bool = False,
  linsolver: Linsolver = np.linalg.solve,
) -> StabResult:
  r"""Performs minimization of the Michelsen's modified tangent-plane
  distance function using Newton's method and a PT-based equation
  of state. A switch to the successive substitution iteration is
  implemented if Newton's method does not decrease the norm of the
  gradient. Preceding successive substitution iterations are implemented
  to improve the initial guess of k-values.

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Z_lnphii_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], mole composition
      (`Vector[Double]` of shape `(Nc,)`), and phase mole number [mol],
      this method must return a tuple of:

      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the gradient of the tangent-plane distance function is less than
    `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `50`.

  switchers: tuple[float, float, float]
    Allows to modify the conditions of switching from the successive
    substitution method to Newton's method. The parameter must be
    represented as a tuple containing three values: :math:`\eps_r`,
    :math:`\eps_l`, :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
      \end{cases}

    where :math:`\mathbf{g}` is a vector of the gradient of the TPD
    function, :math:`k` is the iteration number. Analytical expressions
    of the switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.1, 1e-12, 1e-4)`.

  eps: float
    System will be considered stable when `TPD >= eps`.
    Default is `-1e-8`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  checktrivial: bool
    A flag indicating whether it is necessary to perform a check for
    early detection of convergence to the trivial solution. It is based
    on the paper of M.L. Michelsen (doi: 10.1016/0378-3812(82)85001-2).
    Default is `True`.

  breakunstab: bool
    A boolean flag indicating whether it is allowed to break a loop
    of various initial guesses checking if the one-phase state was
    identified as unstable. This option should be activated to prevent
    finding the global minimum of the TPD function if a local minimum
    is characterised with the negative value of the TPD function.
    Default is `False`.

  linsolver: Linsolver
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `kvji` a tuple containing vectors of k-values that can be used as
    an initial guess for flash calculations.
  """
  logger.info("Stability Test (SS-Newton method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  tmpl = '%3s%5s' + Nc * '%9s' + '%11s%9s'
  rangeNc = range(Nc)
  h1 = tmpl % ('Nkv', 'Nit', *['lnkv%s' % s for s in rangeNc], 'g2', 'method')
  h2 = tmpl % ('Nkv', 'Nit', *['alph%s' % s for s in rangeNc], 'g2', 'method')
  tmpl = '%3s%5s' + Nc * '%9.4f' + '%11.2e%9s'
  Z, lnphiyi = eos.getPT_Z_lnphii(P, T, yi)
  TPDo = eps
  yti = yi
  kvji = None
  epsr, epsl, epsu = switchers
  for j, ki in enumerate(kvji0):
    k = 0
    trivial = False
    lnki = np.log(ki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    switch = g2 < tol
    logger.debug(h1)
    logger.debug(tmpl, j, k, *lnki, g2, 'ss')
    while not switch and g2 > tol and k < maxiter:
      k += 1
      lnki -= gi
      ki = np.exp(lnki)
      ni = ki * yi
      n = ni.sum()
      xi = ni / n
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gi = lnki + lnphixi - lnphiyi
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = g2 / g2km1 > epsr and g2 > epsl and g2 < epsu
      logger.debug(tmpl, j, k, *lnki, g2, 'ss')
      if checktrivial:
        if g2 < tol:
          break
        ng = ni.dot(gi)
        tpds = 1. + ng - n
        r = 2. * tpds / (ng - yi.dot(gi))
        if tpds < 1e-3 and r > 0.8 and r < 1.2:
          trivial = True
          break
    if np.isfinite(g2) and not trivial:
      if g2 < tol:
        TPD = -np.log(n)
        if TPD < TPDo:
          TPDo = TPD
          yti = xi
          if breakunstab:
            logger.info('The system is stable: False. TPD = %.3e.', TPDo)
            return StabResult(False, Z, (yti / yi,))
      elif k < maxiter:
        hi = lnphiyi + np.log(yi)
        sqrtni = np.sqrt(ni)
        alphai = 2. * sqrtni
        Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
        logger.debug(h2)
        logger.debug(tmpl, j, k, *alphai, g2, 'newt')
        while g2 > tol and k < maxiter:
          H = (np.diagflat(.5 * gi + 1.)
               + (sqrtni[:,None] * sqrtni) * dlnphixidnj)
          dalphai = linsolver(H, -sqrtni * gi)
          k += 1
          alphai += dalphai
          sqrtni = alphai * .5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
          gi = np.log(ni) + lnphixi - hi
          g2kp1 = gi.dot(gi)
          if g2kp1 < g2 or forcenewton:
            g2 = g2kp1
            method = 'newt'
          else:
            ni *= np.exp(-gi)
            sqrtni = np.sqrt(ni)
            alphai = 2. * sqrtni
            n = ni.sum()
            xi = ni / n
            Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
            gi = np.log(ni) + lnphixi - hi
            g2 = gi.dot(gi)
            method = 'ss'
          logger.debug(tmpl, j, k, *alphai, g2, method)
        if g2 < tol and np.isfinite(g2):
          TPD = -np.log(n)
          if TPD < TPDo:
            TPDo = TPD
            yti = xi
            if breakunstab:
              logger.info('The system is stable: False. TPD = %.3e.', TPDo)
              return StabResult(False, Z, (yti / yi,))
  stable = TPDo >= eps
  logger.info('The system is stable: %s. TPD = %.3e.', stable, TPDo)
  if not stable:
    kvji = (yti / yi,)
  return StabResult(stable, Z, kvji)


def _stabPT_qnssnewt(
  P: float,
  T: float,
  yi: Vector[Double],
  kvji0: Sequence[Vector[Double]],
  eos: EosStabPT,
  tol: float = 1e-20,
  maxiter: int = 50,
  switchers: tuple[float, float, float] = (0.1, 1e-12, 1e-4),
  lmbdmax: float = 30.,
  eps: float = -1e-8,
  forcenewton: bool = False,
  breakunstab: bool = False,
  linsolver: Linsolver = np.linalg.solve,
) -> StabResult:
  r"""Performs minimization of the Michelsen's modified tangent-plane
  distance function using Newton's method and a PT-based equation of
  state. A switch to the successive substitution iteration is
  implemented if Newton's method does not decrease the norm of the
  gradient. Preceding successive substitution iterations are
  implemented to improve the initial guess of k-values.

  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Z_lnphii_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], mole composition
      (`Vector[Double]` of shape `(Nc,)`), and phase mole number [mol],
      this method must return a tuple of:

      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the gradient of the tangent-plane distance function is less than
    `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `50`.

  switchers: tuple[float, float, float]
    Allows to modify the conditions of switching from the successive
    substitution method to Newton's method. The parameter must be
    represented as a tuple containing three values: :math:`\eps_r`,
    :math:`\eps_l`, :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
      \end{cases}

    where :math:`\mathbf{g}` is a vector of the gradient of the TPD
    function, :math:`k` is the iteration number. Analytical expressions
    of the switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.1, 1e-12, 1e-4)`.

  lmbdmax: float
    The maximum step length. Default is `30.0`.

  eps: float
    System will be considered stable when `TPD >= eps`.
    Default is `-1e-8`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  breakunstab: bool
    A boolean flag indicating whether it is allowed to break a loop
    of various initial guesses checking if the one-phase state was
    identified as unstable. This option should be activated to prevent
    finding the global minimum of the TPD function if a local minimum
    is characterised with the negative value of the TPD function.
    Default is `False`.

  linsolver: Linsolver
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `kvji` a tuple containing vectors of k-values that can be used as
    an initial guess for flash calculations.
  """
  logger.info("Stability Test (QNSS-Newton method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  tmpl = '%3s%5s' + Nc * '%9s' + '%11s%9s'
  rangeNc = range(Nc)
  h1 = tmpl % ('Nkv', 'Nit', *['lnkv%s' % s for s in rangeNc], 'g2', 'method')
  h2 = tmpl % ('Nkv', 'Nit', *['alph%s' % s for s in rangeNc], 'g2', 'method')
  tmpl = '%3s%5s' + Nc * '%9.4f' + '%11.2e%9s'
  Z, lnphiyi = eos.getPT_Z_lnphii(P, T, yi)
  TPDo = eps
  yti = yi
  kvji = None
  epsr, epsl, epsu = switchers
  for j, ki in enumerate(kvji0):
    k = 0
    lnki = np.log(ki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    switch = g2 < tol
    lmbd = 1.
    logger.debug(h1)
    logger.debug(tmpl, j, k, *lnki, g2, 'qnss')
    while not switch and g2 > tol and k < maxiter:
      dlnki = -lmbd * gi
      max_dlnki = np.abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
      k += 1
      tkm1 = dlnki.dot(gi)
      lnki += dlnki
      ki = np.exp(lnki)
      ni = ki * yi
      n = ni.sum()
      xi = ni / n
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gi = lnki + lnphixi - lnphiyi
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = g2 / g2km1 > epsr and g2 > epsl and g2 < epsu
      logger.debug(tmpl, j, k, *lnki, g2, 'qnss')
      if g2 < tol or switch:
        break
      if k % Nc == 0:
        lmbd = 1.
      else:
        lmbd *= tkm1 / (dlnki.dot(gi) - tkm1)
        if lmbd < 0.:
          lmbd = -lmbd
        if lmbd > lmbdmax:
          lmbd = lmbdmax
    if np.isfinite(g2):
      if g2 < tol:
        TPD = -np.log(n)
        if TPD < TPDo:
          TPDo = TPD
          yti = xi
          if breakunstab:
            logger.info('The system is stable: False. TPD = %.3e.', TPDo)
            return StabResult(False, Z, (yti / yi,))
      elif k < maxiter:
        hi = lnphiyi + np.log(yi)
        sqrtni = np.sqrt(ni)
        alphai = 2. * sqrtni
        Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
        logger.debug(h2)
        logger.debug(tmpl, j, k, *alphai, g2, 'newt')
        while g2 > tol and k < maxiter:
          H = (np.diagflat(.5 * gi + 1.)
               + (sqrtni[:,None] * sqrtni) * dlnphixidnj)
          dalphai = linsolver(H, -sqrtni * gi)
          k += 1
          alphai += dalphai
          sqrtni = alphai * .5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
          gi = np.log(ni) + lnphixi - hi
          g2kp1 = gi.dot(gi)
          if g2kp1 < g2 or forcenewton:
            g2 = g2kp1
            method = 'newt'
          else:
            ni *= np.exp(-gi)
            sqrtni = np.sqrt(ni)
            alphai = 2. * sqrtni
            n = ni.sum()
            xi = ni / n
            Zx, lnphixi, dlnphixidnj = eos.getPT_Z_lnphii_dnj(P, T, xi, n)
            gi = np.log(ni) + lnphixi - hi
            g2 = gi.dot(gi)
            method = 'ss'
          logger.debug(tmpl, j, k, *alphai, g2, method)
        if g2 < tol and np.isfinite(g2):
          TPD = -np.log(n)
          if TPD < TPDo:
            TPDo = TPD
            yti = xi
            if breakunstab:
              stable = TPDo >= eps
              logger.info('The system is stable: False. TPD = %.3e.', TPDo)
              return StabResult(False, Z, (yti / yi,))
  stable = TPDo >= eps
  logger.info('The system is stable: %s. TPD = %.3e.', stable, TPDo)
  if not stable:
    kvji = (yti / yi,)
  return StabResult(stable, Z, kvji)


def _stabPT_ssbfgs(
  P: float,
  T: float,
  yi: Vector[Double],
  kvji0: Sequence[Vector[Double]],
  eos: EosStabPT,
  tol: float = 1e-20,
  maxiter: int = 50,
  switchers: tuple[float, float, float] = (0.1, 1e-12, 1e-4),
  eps: float = -1e-8,
  checktrivial: bool = True,
  breakunstab: bool = False,
) -> StabResult:
  r"""Performs minimization of the Michelsen's modified tangent-plane
  distance function using BFGS method and a PT-based equation of state.
  Preceding successive substitution iterations are implemented to
  improve the initial guess of k-values. The reference for the
  implementation of the BFGS method for the stability test is the paper
  of H. Hoteit and A. Firoozabadi (doi: 10.1002/aic.10908).

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosStabPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the gradient of the tangent-plane distance function is less
    than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `50`.

  switchers: tuple[float, float, float]
    Allows to modify the conditions of switching from the successive
    substitution method to Newton's method. The parameter must be
    represented as a tuple containing three values: :math:`\eps_r`,
    :math:`\eps_l`, :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
      \end{cases}

    where :math:`\mathbf{g}` is a vector of the gradient of the TPD
    function, :math:`k` is the iteration number. Analytical expressions
    of the switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.1, 1e-12, 1e-4)`.

  eps: float
    System will be considered stable when `TPD >= eps`.
    Default is `-1e-8`.

  checktrivial: bool
    A flag indicating whether it is necessary to perform a check for
    early detection of convergence to the trivial solution. It is based
    on the paper of M.L. Michelsen (doi: 10.1016/0378-3812(82)85001-2).
    Default is `True`.

  breakunstab: bool
    A boolean flag indicating whether it is allowed to break a loop
    of various initial guesses checking if the one-phase state was
    identified as unstable. This option should be activated to prevent
    finding the global minimum of the TPD function if a local minimum
    is characterised with the negative value of the TPD function.
    Default is `False`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `kvji` a tuple containing vectors of k-values that can be used as
    an initial guess for flash calculations.
  """
  logger.info("Stability Test (SS-BFGS method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  tmpl = '%3s%5s' + Nc * '%9s' + '%11s%9s'
  rangeNc = range(Nc)
  h1 = tmpl % ('Nkv', 'Nit', *['lnkv%s' % s for s in rangeNc], 'g2', 'method')
  h2 = tmpl % ('Nkv', 'Nit', *['alph%s' % s for s in rangeNc], 'g2', 'method')
  tmpl = '%3s%5s' + Nc * '%9.4f' + '%11.2e%9s'
  Z, lnphiyi = eos.getPT_Z_lnphii(P, T, yi)
  hi = lnphiyi + np.log(yi)
  TPDo = eps
  yti = yi
  kvji = None
  epsr, epsl, epsu = switchers
  for j, ki in enumerate(kvji0):
    k = 0
    trivial = False
    lnki = np.log(ki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    switch = g2 < tol
    logger.debug(h1)
    logger.debug(tmpl, j, k, *lnki, g2, 'ss')
    while not switch and g2 > tol and k < maxiter:
      k += 1
      lnki -= gi
      ki = np.exp(lnki)
      ni = ki * yi
      n = ni.sum()
      xi = ni / n
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gi = lnki + lnphixi - lnphiyi
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = g2 / g2km1 > epsr and g2 > epsl and g2 < epsu
      logger.debug(tmpl, j, k, *lnki, g2, 'ss')
      if checktrivial:
        if g2 < tol:
          break
        ng = ni.dot(gi)
        tpds = 1. + ng - n
        r = 2. * tpds / (ng - yi.dot(gi))
        if tpds < 1e-3 and r > 0.8 and r < 1.2:
          trivial = True
          break
    if np.isfinite(g2) and not trivial:
      if g2 < tol:
        TPD = -np.log(n)
        if TPD < TPDo:
          TPDo = TPD
          yti = xi
          if breakunstab:
            logger.info('The system is stable: False. TPD = %.3e.', TPDo)
            return StabResult(False, Z, (yti / yi,))
      elif k < maxiter:
        hi = lnphiyi + np.log(yi)
        sqrtni = np.sqrt(ni)
        alphai = 2. * sqrtni
        gik = sqrtni * gi
        logger.debug(h2)
        logger.debug(tmpl, j, k, *alphai, g2, 'bfgs')
        si = -gik
        k += 1
        alphai += si
        sqrtni = alphai * 0.5
        ni = sqrtni * sqrtni
        n = ni.sum()
        xi = ni / n
        Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
        gi = np.log(ni) + lnphixi - hi
        gikm1 = gik
        gik = sqrtni * gi
        g2 = gi.dot(gi)
        logger.debug(tmpl, j, k, *alphai, g2, 'bfgs')
        if g2 < tol and np.isfinite(g2):
          TPD = -np.log(n)
          if TPD < TPDo:
            TPDo = TPD
            yti = xi
            if breakunstab:
              logger.info('The system is stable: False. TPD = %.3e.', TPDo)
              return StabResult(False, Z, (yti / yi,))
            else:
              continue
        if checktrivial:
          tpds = 1. + ni.dot(gi) - n
          r = 2. * tpds / gik.dot(ni - yi)
          if tpds < 1e-3 and r > 0.8 and r < 1.2:
            trivial = True
            continue
        while g2 > tol and k < maxiter:
          qi = gik - gikm1
          sq = si.dot(qi)
          sg = si.dot(gik)
          qq = qi.dot(qi)
          qg = qi.dot(gik)
          si = -(gik + (sg - qg + qq * sg / sq) / sq * si - sg / sq * qi)
          k += 1
          alphai += si
          sqrtni = alphai * 0.5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
          gi = np.log(ni) + lnphixi - hi
          gikm1 = gik
          gik = sqrtni * gi
          g2 = gi.dot(gi)
          logger.debug(tmpl, j, k, *alphai, g2, 'bfgs')
          if checktrivial:
            if g2 < tol:
              break
            tpds = 1. + ni.dot(gi) - n
            r = 2. * tpds / gik.dot(ni - yi)
            if tpds < 1e-3 and r > 0.8 and r < 1.2:
              trivial = True
              break
        if g2 < tol and np.isfinite(g2) and not trivial:
          TPD = -np.log(n)
          if TPD < TPDo:
            TPDo = TPD
            yti = xi
            if breakunstab:
              logger.info('The system is stable: False. TPD = %.3e.', TPDo)
              return StabResult(False, Z, (yti / yi,))
  stable = TPDo >= eps
  logger.info('The system is stable: %s. TPD = %.3e.', stable, TPDo)
  if not stable:
    kvji = (yti / yi,)
  return StabResult(stable, Z, kvji)
