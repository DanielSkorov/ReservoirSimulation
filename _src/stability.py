import logging

from functools import (
  partial,
)

import numpy as np

from typing import Callable

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Eos,
)


logger = logging.getLogger('stab')


class StabEosPT(Eos):

  def getPT_kvguess(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    level: int,
    idx: int,
    eps: Scalar,
  ) -> tuple[Vector, ...]: ...

  def getPT_lnphii_Z(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar]: ...

  def getPT_lnphii_Z_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[Vector, Scalar, Matrix]: ...


class StabResult(dict):
  """Container for stability test outputs with pretty-printing.

  Attributes
  ----------
  stable: bool
    A boolean flag indicating if the one-phase state is stable.

  TPD: Scalar
    The tangent-plane distance at a local minimum of the potential
    energy function.

  Z: Scalar
    The compressibility factor of the tested mixture at a given
    pressure and temperature.

  yti: Vector, shape (Nc,)
    The trial-phase composition if it was found. `Nc` is the number
    of components.

  kvji: tuple[Vector, ...]
    K-values of `Nc` components in the trial phase and given mixture.
    They may be further used as an initial guess for flash calculations.

  gnorm: Scalar
    The norm of the vector of equilibrium equations.

  success: bool
    A boolean flag indicating whether or not the procedure exited
    successfully. For the stability test algorithms, it is always
    `True` because the divergence of an algorithm is usually caused
    by the vicinity to the stability test limit locus outside of
    which the tangent plane distance function has only a trivial
    solution. Between STLL and the phase boundary, there is a
    non-trivial positive solution that leads to the stability of the
    one-phase state.
  """
  def __getattr__(self, name: str) -> object:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      s = (f"The one-phase state is stable:\n{self.stable}\n"
           f"Tangent-plane distance:\n{self.TPD}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class stabilityPT(object):
  """Stability test based on the Gibbs energy analysis.

  Checks the tangent-plane distance (TPD) at the local minimum of
  the Gibbs energy function.

  Parameters
  ----------
  eos: StabEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector, level: int) -> tuple[Vector, ...]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a tuple of `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of a solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'ss'`.

  level: int
    Regulates a set of initial k-values obtained by the method
    `getPT_kvguess` of the initialized instance of an EOS.
    Default is `0`.

  useprev: bool
    Allows to preseve previous calculation results (if the solution
    is non-trivial) and to use them as the first initial guess in next
    run. Default is `False`.

  kwargs: dict
    Other arguments for a stability test solver. It may contain such
    arguments as `tol`, `maxiter` and others appropriate for the selected
    stability test solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> StabResult
    This method performs stability test procedure for a given pressure
    `P` in [Pa], temperature `T` in [K], mole composition `yi` of shape
    `(Nc,)`, and returns stability test results as an instance of
    `StabResult`.
  """
  def __init__(
    self,
    eos: StabEosPT,
    method: str = 'ss',
    level: int = 0,
    useprev: bool = False,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.level = level
    self.useprev = useprev
    self.prevkvji: None | tuple[Vector, ...] = None
    self.preserved = False
    if method == 'ss':
      self.solver = partial(_stabPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_stabPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_stabPT_newt, eos=eos, **kwargs)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the stability test is not implemented yet.'
      )
    elif method == 'ss-newton':
      self.solver = partial(_stabPT_ssnewt, eos=eos, **kwargs)
    elif method == 'qnss-newton':
      self.solver = partial(_stabPT_qnssnewt, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    kvji0: tuple[Vector, ...] | None = None,
  ) -> StabResult:
    """Performs the stability test for a given pressure, temperature
    and composition.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    kvji0: tuple[Vector, ...] | None
      A tuple containing arrays of initial k-value guesses. Each array's
      shape should be `(Nc,)`. Default is `None` which means to use
      initial guesses from the method `getPT_kvguess` of the initialized
      instance of an EOS.

    Returns
    -------
    Stability test results as an instance of `StabResult`. Important
    attributes are:
    - `stab` a boolean flag indicating if a one-phase state is stable,
    - `TPD` the tangent-plane distance at a local minimum of the Gibbs
      energy function,
    - `success` a boolean flag indicating if the calculation completed
      successfully.
    """
    if kvji0 is None:
      kvji0 = self.eos.getPT_kvguess(P, T, yi, self.level)
    if self.useprev and self.preserved:
      kvji0 = *self.prevkvji, *kvji0
    stab = self.solver(P, T, yi, kvji0)
    if stab.success and self.useprev:
      self.prevkvji = stab.kvji
      self.preserved = True
    return stab


def _stabPT_ss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: StabEosPT,
  eps: Scalar = -1e-4,
  tol: Scalar = 1e-6,
  maxiter: int = 50,
) -> StabResult:
  """Successive Substitution (SS) method to perform the stability test
  using a PT-based equation of state.

  Parameters
  ----------
  P: Scalar
    Pressure of the mixture [Pa].

  T: Scalar
    Temperature of the mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[Vector, ...]
    A tuple containing arrays of initial k-value guesses. Each array's
    shape should be `(Nc,)`.

  eos: StabEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  eps: Scalar
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: Scalar
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of solver iterations. Default is `50`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `TPD` the tangent-plane distance at a local minimum of the Gibbs
    energy function,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.info('Stability Test (SS-method).')
  Nc = eos.Nc
  logger.debug(
    '%3s%5s' + Nc * '%9s' + '%11s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'gnorm',
  )
  tmpl = '%3s%5s' + Nc * ' %8.4f' + ' %10.2e'
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  for j, kvi0 in enumerate(kvji0):
    k = 0
    lnkik = np.log(kvi0)
    ni = kvi0 * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, j, k, *lnkik, gnorm)
    while gnorm > tol and k < maxiter:
      k += 1
      lnkik -= gi
      ni = np.exp(lnkik) * yi
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = lnkik + lnphixi - lnphiyi
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, j, k, *lnkik, gnorm)
    if gnorm < tol and np.isfinite(gnorm):
      TPD = -np.log(ni.sum())
      if TPD < eps:
        logger.info('The system is unstable. TPD = %.3e.', TPD)
        n = ni.sum()
        xi = ni / n
        kvji = xi / yi, yi / xi
        return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                          TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                          lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info('The system is stable. TPD = %.3e.', TPD)
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_qnss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: StabEosPT,
  eps: Scalar = -1e-4,
  tol: Scalar = 1e-6,
  maxiter: int = 50,
) -> StabResult:
  """QNSS-method to perform the stability test using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find local minimums of the Gibbs energy function from different
  initial guesses. Checks the TPD-value at a found local minima.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P: Scalar
    Pressure of the mixture [Pa].

  T: Scalar
    Temperature of the mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[Vector, ...]
    A tuple containing arrays of initial k-value guesses. Each array's
    shape should be `(Nc,)`.

  eos: StabEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  eps: Scalar
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: Scalar
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of solver iterations. Default is `50`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `TPD` the tangent-plane distance at a local minimum of the Gibbs
    energy function,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.info('Stability Test (QNSS-method).')
  Nc = eos.Nc
  logger.debug(
    '%3s%5s' + Nc * '%9s' + '%11s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'gnorm',
  )
  tmpl = '%3s%5s' + Nc * ' %8.4f' + ' %10.2e'
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  for j, kvi0 in enumerate(kvji0):
    k = 0
    lnkik = np.log(kvi0)
    ni = kvi0 * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, j, k, *lnkik, gnorm)
    lmbd = 1.
    while gnorm > tol and k < maxiter:
      dlnki = -lmbd * gi
      max_dlnki = np.abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
      k += 1
      tkm1 = dlnki.dot(gi)
      lnkik += dlnki
      ni = np.exp(lnkik) * yi
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = lnkik + lnphixi - lnphiyi
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, j, k, *lnkik, gnorm)
      if gnorm < tol:
        break
      lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if gnorm < tol and np.isfinite(gnorm):
      TPD = -np.log(ni.sum())
      if TPD < eps:
        logger.info('The system is unstable. TPD = %.3e.', TPD)
        n = ni.sum()
        xi = ni / n
        kvji = xi / yi, yi / xi
        return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                          TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                          lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info('The system is stable. TPD = %.3e.', TPD)
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_newt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: StabEosPT,
  eps: Scalar = -1e-4,
  tol: Scalar = 1e-6,
  maxiter: int = 20,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> StabResult:
  """Performs minimization of the Michelsen's modified tangent-plane
  distance function using Newton's method and a PT-based equation
  of state. A switch to the successive substitution iteration is
  implemented if Newton's method does not decrease the norm of the
  gradient.

  Parameters
  ----------
  P: Scalar
    Pressure of the mixture [Pa].

  T: Scalar
    Temperature of the mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[Vector, ...]
    A tuple containing arrays of initial k-value guesses. Each array's
    shape should be `(Nc,)`.

  eos: StabEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  eps: Scalar
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: Scalar
    Terminate Newton's method successfully if the norm of the gradient
    of Michelsen's modified tangent-plane distance function is less than
    `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of Newton's method iterations. Default is `20`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `TPD` the tangent-plane distance at a local minimum of the Gibbs
    energy function,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.info("Stability Test (Newton's method).")
  Nc = eos.Nc
  logger.debug(
    '%3s%5s' + Nc * '%9s' + '%11s%9s',
    'Nkv', 'Nit', *map(lambda s: 'alpha' + s, map(str, range(Nc))),
    'gnorm', 'method',
  )
  tmpl = '%3s%5s' + Nc * ' %8.4f' + ' %10.2e %8s'
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  for j, kvi0 in enumerate(kvji0):
    k = 0
    ni = kvi0 * yi
    sqrtni = np.sqrt(ni)
    alphaik = 2. * sqrtni
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
    gi = sqrtni * (np.log(ni) + lnphixi - hi)
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, j, k, *alphaik, gnorm, 'newt')
    while gnorm > tol and k < maxiter:
      H = np.diagflat(.5 * gi + 1.) + (sqrtni[:,None] * sqrtni) * dlnphixidnj
      dalphai = linsolver(H, -gi)
      k += 1
      alphaik += dalphai
      sqrtni = alphaik * .5
      ni = sqrtni * sqrtni
      n = ni.sum()
      xi = ni / n
      lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
      gi = sqrtni * (np.log(ni) + lnphixi - hi)
      gnormkp1 = np.linalg.norm(gi)
      if gnormkp1 < gnorm or forcenewton:
        gnorm = gnormkp1
        logger.debug(tmpl, j, k, *alphaik, gnorm, 'newt')
      else:
        ni *= np.exp(-gi / sqrtni)
        sqrtni = np.sqrt(ni)
        alphaik = 2. * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
        gi = sqrtni * (np.log(ni) + lnphixi - hi)
        gnorm = np.linalg.norm(gi)
        logger.debug(tmpl, j, k, *alphaik, gnorm, 'ss')
    if gnorm < tol and np.isfinite(gnorm):
      TPD = -np.log(ni.sum())
      if TPD < eps:
        logger.info('The system is unstable. TPD = %.3e.', TPD)
        n = ni.sum()
        xi = ni / n
        kvji = xi / yi, yi / xi
        return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                          TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                          lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info('The system is stable. TPD = %.3e.', TPD)
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_ssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: StabEosPT,
  eps: Scalar = -1e-4,
  tol: Scalar = 1e-6,
  maxiter: int = 30,
  tol_ss: Scalar = 1e-2,
  maxiter_ss: int = 10,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> StabResult:
  """Performs minimization of the Michelsen's modified tangent-plane
  distance function using Newton's method and a PT-based equation
  of state. A switch to the successive substitution iteration is
  implemented if Newton's method does not decrease the norm of the
  gradient. Preceding successive substitution iterations are implemented
  to improve the initial guess of k-values.

  Parameters
  ----------
  P: Scalar
    Pressure of the mixture [Pa].

  T: Scalar
    Temperature of the mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[Vector, ...]
    A tuple containing arrays of initial k-value guesses. Each array's
    shape should be `(Nc,)`.

  eos: StabEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  eps: Scalar
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: Scalar
    Terminate Newton's method successfully if the norm of the gradient
    of Michelsen's modified tangent-plane distance function is less than
    `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `30`.

  tol_ss: Scalar
    Switch to Newton's method if the norm of the vector of equilibrium
    equations is less than `tol_ss`. Default is `1e-2`.

  maxiter_ss: int
    The maximum number of successive substitution iterations.
    Default is `10`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:
  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `TPD` the tangent-plane distance at a local minimum of the Gibbs
    energy function,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.info("Stability Test (SS-Newton method).")
  Nc = eos.Nc
  tmpl = '%3s%5s' + Nc * ' %8.4f' + ' %10.2e %8s'
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug(
      '%3s%5s' + Nc * '%9s' + '%11s%9s',
      'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
      'gnorm', 'method',
    )
    k = 0
    lnkik = np.log(kvi0)
    ni = kvi0 * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, j, k, *lnkik, gnorm, 'ss')
    while gnorm > tol_ss and k < maxiter_ss:
      k += 1
      lnkik -= gi
      ni = np.exp(lnkik) * yi
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = lnkik + lnphixi - lnphiyi
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, j, k, *lnkik, gnorm, 'ss')
    if np.isfinite(gnorm):
      if gnorm < tol:
        TPD = -np.log(ni.sum())
        if TPD < eps:
          logger.info('The system is unstable. TPD = %.3e.', TPD)
          n = ni.sum()
          xi = ni / n
          kvji = xi / yi, yi / xi
          return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                            TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                            lnphiyti=lnphixi, success=True)
      else:
        hi = lnphiyi + np.log(yi)
        sqrtni = np.sqrt(ni)
        alphaik = 2. * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
        gi = sqrtni * (np.log(ni) + lnphixi - hi)
        gnorm = np.linalg.norm(gi)
        logger.debug(
          '%3s%5s' + Nc * '%9s' + '%11s%9s',
          'Nkv', 'Nit', *map(lambda s: 'alpha' + s, map(str, range(Nc))),
          'gnorm', 'method',
        )
        logger.debug(tmpl, j, k, *alphaik, gnorm, 'newt')
        while gnorm > tol and k < maxiter:
          H = (np.diagflat(.5 * gi + 1.)
               + (sqrtni[:,None] * sqrtni) * dlnphixidnj)
          dalphai = linsolver(H, -gi)
          k += 1
          alphaik += dalphai
          sqrtni = alphaik * .5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
          gi = sqrtni * (np.log(ni) + lnphixi - hi)
          gnormkp1 = np.linalg.norm(gi)
          if gnormkp1 < gnorm or forcenewton:
            gnorm = gnormkp1
            logger.debug(tmpl, j, k, *alphaik, gnorm, 'newt')
          else:
            ni *= np.exp(-gi / sqrtni)
            sqrtni = np.sqrt(ni)
            alphaik = 2. * sqrtni
            n = ni.sum()
            xi = ni / n
            lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
            gi = sqrtni * (np.log(ni) + lnphixi - hi)
            gnorm = np.linalg.norm(gi)
            logger.debug(tmpl, j, k, *alphaik, gnorm, 'ss')
        if gnorm < tol and np.isfinite(gnorm):
          TPD = -np.log(ni.sum())
          if TPD < eps:
            logger.info('The system is unstable. TPD = %.3e.', TPD)
            n = ni.sum()
            xi = ni / n
            kvji = xi / yi, yi / xi
            return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                              TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                              lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info('The system is stable. TPD = %.3e.', TPD)
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_qnssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: StabEosPT,
  eps: Scalar = -1e-4,
  tol: Scalar = 1e-6,
  maxiter: int = 30,
  tol_qnss: Scalar = 1e-2,
  maxiter_qnss: int = 10,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> StabResult:
  """Performs minimization of the Michelsen's modified tangent-plane
  distance function using Newton's method and a PT-based equation of
  state. A switch to the quasi-newton successive substitution (QNSS)
  iteration is implemented if Newton's method does not decrease the
  norm of the gradient. Preceding successive substitution iterations
  are implemented to improve the initial guess of k-values.

  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P: Scalar
    Pressure of the mixture [Pa].

  T: Scalar
    Temperature of the mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[Vector, ...]
    A tuple containing arrays of initial k-value guesses. Each array's
    shape should be `(Nc,)`.

  eos: StabEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  eps: Scalar
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: Scalar
    Terminate Newton's method successfully if the norm of the gradient
    of Michelsen's modified tangent-plane distance function is less than
    `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `30`.

  tol_qnss: Scalar
    Switch to Newton's method if the norm of the vector of equilibrium
    equations is less than `tol_qnss`. Default is `1e-2`.

  maxiter_qnss: int
    The maximum number of quasi-newton successive substitution
    iterations. Default is `10`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  attributes are:

  - `stab` a boolean flag indicating if a one-phase state is stable,
  - `TPD` the tangent-plane distance at a local minimum of the Gibbs
    energy function,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.info("Stability Test (QNSS-Newton method).")
  Nc = eos.Nc
  tmpl = '%3s%5s' + Nc * ' %8.4f' + ' %10.2e %8s'
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug(
      '%3s%5s' + Nc * '%9s' + '%11s%9s',
      'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
      'gnorm', 'method',
    )
    k = 0
    lnkik = np.log(kvi0)
    ni = kvi0 * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    lmbd = 1.
    logger.debug(tmpl, j, k, *lnkik, gnorm, 'qnss')
    while gnorm > tol_qnss and k < maxiter_qnss:
      dlnki = -lmbd * gi
      max_dlnki = np.abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
      k += 1
      tkm1 = dlnki.dot(gi)
      lnkik += dlnki
      ni = np.exp(lnkik) * yi
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = lnkik + lnphixi - lnphiyi
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, j, k, *lnkik, gnorm, 'qnss')
      if gnorm < tol_qnss:
        break
      lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if np.isfinite(gnorm):
      if gnorm < tol:
        TPD = -np.log(ni.sum())
        if TPD < eps:
          logger.info('The system is unstable. TPD = %.3e.', TPD)
          n = ni.sum()
          xi = ni / n
          kvji = xi / yi, yi / xi
          return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                            TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                            lnphiyti=lnphixi, success=True)
      else:
        hi = lnphiyi + np.log(yi)
        sqrtni = np.sqrt(ni)
        alphaik = 2. * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
        gi = sqrtni * (np.log(ni) + lnphixi - hi)
        gnorm = np.linalg.norm(gi)
        logger.debug(
          '%3s%5s' + Nc * '%9s' + '%11s%9s',
          'Nkv', 'Nit', *map(lambda s: 'alpha' + s, map(str, range(Nc))),
          'gnorm', 'method',
        )
        logger.debug(tmpl, j, k, *alphaik, gnorm, 'newt')
        while gnorm > tol and k < maxiter:
          H = (np.diagflat(.5 * gi + 1.)
               + (sqrtni[:,None] * sqrtni) * dlnphixidnj)
          dalphai = linsolver(H, -gi)
          k += 1
          alphaik += dalphai
          sqrtni = alphaik * .5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
          gi = sqrtni * (np.log(ni) + lnphixi - hi)
          gnormkp1 = np.linalg.norm(gi)
          if gnormkp1 < gnorm or forcenewton:
            gnorm = gnormkp1
            logger.debug(tmpl, j, k, *alphaik, gnorm, 'newt')
          else:
            ni *= np.exp(-gi / sqrtni)
            sqrtni = np.sqrt(ni)
            alphaik = 2. * sqrtni
            n = ni.sum()
            xi = ni / n
            lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
            gi = sqrtni * (np.log(ni) + lnphixi - hi)
            gnorm = np.linalg.norm(gi)
            logger.debug(tmpl, j, k, *alphaik, gnorm, 'ss')
        if gnorm < tol and np.isfinite(alphaik).all():
          TPD = -np.log(ni.sum())
          if TPD < eps:
            logger.info('The system is unstable. TPD = %.3e.', TPD)
            n = ni.sum()
            xi = ni / n
            kvji = xi / yi, yi / xi
            return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                              TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                              lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info('The system is stable. TPD = %.3e.', TPD)
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)

