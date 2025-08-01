import logging

from functools import (
  partial,
)

import numpy as np

from stability import (
  stabilityPT,
)

from rr import (
  solve2p_FGH,
)

from typing import (
  Callable,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Eos,
  SolutionNotFoundError,
)


logger = logging.getLogger('flash')


class Flash2pEosPT(Eos):

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


class FlashResult(dict):
  """Container for flash calculation outputs with pretty-printing.

  Attributes
  ----------
  yji: Matrix, shape (Np, Nc)
    Mole fractions of components in each phase. Two-dimensional
    array of real elements of size `(Np, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  Fj: Vector, shape (Np,)
    Phase mole fractions. Vector of real elements of size `(Np,)`,
    where `Np` is the number of phases.

  Zj: Vector, shape (Np,)
    Compressibility factors of each phase. Array of real elements of
    size `(Np,)`, where `Np` is the number of phases.

  kvji: Matrix, shape (Np - 1, Nc)
    K-values of components in non-reference phases. Two-dimensional
    array of real elements of size `(Np - 1, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  gnorm: Scalar
    The norm of the vector of equilibrium equations.

  Niter: int
    The number of iterations.
  """
  def __getattr__(self, name: str) -> object:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      s = (f"Phase composition:\n{self.yji}\n"
           f"Phase mole fractions:\n{self.Fj}\n"
           f"Phase compressibility factors:\n{self.Zj}\n")
    return s


class flash2pPT(object):
  """Two-phase flash calculations.

  Performs two-phase flash calculations for a given pressure [Pa],
  temperature [K] and composition of the mixture.

  Parameters
  ----------
  eos: FlashFlash2pEosPT
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

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  flashmethod: str
    Type of flash calculations solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'ss'`.

  stabmethod: str
    Type of stability tests sovler. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement),
    - `'full-newton'` (Newton's method without solving the Rachford-
      Rice equation in the inner loop, 10.1021/acs.iecr.3c00550).

    Default is `'ss'`.

  level: int
    Regulates a set of initial k-values obtained by the method
    `eos.getPT_kvguess(P: Scalar, T: Scalar, yi: Vector, level: int = 0)`.
    Default is `0`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`. The value of
    this flag can be changed to `False` if the one-phase state will
    be unstable.

  runstab: bool
    If `True` then the algorithm will perform the stability test, for
    which initial guesses of k-values will be calculated by the method
    of an eos instance and taken from previous flash calculations if
    the flag `useprev` was set `True`. Initial guesses of k-values for
    flash calculations will be taken from the stability test results.
    Default is `True`.

  useprev: bool
    Allows to preseve previous calculation results (if the solution was
    found) and to use them as the first initial guess for the next run.
    Default is `False`.

  stabkwargs: dict
    Dictionary that used to regulate the stability test procedure.
    Default is an empty dictionary.

  kwargs: dict
    Other arguments for a phase split solver. It may contain such
    arguments as `tol`, `maxiter` and others, depending on the selected
    phase split solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> FlashResult
    This method performs two-phase flash calculation for a given
    pressure in [Pa], temperature in [K], mole composition `yi` of shape
    `(Nc,)`, and returns flash calculation results as an instance of
    `FlashResult`.
  """
  def __init__(
    self,
    eos: Flash2pEosPT,
    method: str = 'ss',
    level: int = 0,
    negflash: bool = True,
    runstab: bool = True,
    useprev: bool = False,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.level = level
    self.negflash = negflash
    self.runstab = runstab
    self.useprev = useprev
    self.stabsolver = stabilityPT(eos, level=level, **stabkwargs)
    self.preserved = False
    self.prevkvji: None | Matrix = None
    if method == 'ss':
      self.solver = partial(_flash2pPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_flash2pPT_qnss, eos=eos, **kwargs)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for flash calculations is not implemented yet.'
      )
    elif method == 'newton':
      self.solver = partial(_flash2pPT_newt, eos=eos, **kwargs)
    elif method == 'ss-newton':
      self.solver = partial(_flash2pPT_ssnewt, eos=eos, **kwargs)
    elif method == 'qnss-newton':
      self.solver = partial(_flash2pPT_qnssnewt, eos=eos, **kwargs)
    elif method == 'newton-full':
      raise NotImplementedError(
        "Full Newton's method for flash calculations is not implemented yet."
      )
    else:
      raise ValueError(f'The unknown flash-method: {method}.')
    self._1pstab_yi = np.zeros_like(eos.mwi)
    self._1pstab_Fj = np.array([1., 0.])
    pass

  def run(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    kvji0: tuple[Vector, ...] | None = None,
  ) -> FlashResult:
    """Performs flash calculations for given pressure, temperature and
    composition.

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
    Flash calculation results as an instance of `FlashResult` object.
    Important attributes are:
    - `yji` the component mole fractions in each phase,
    - `Fj` the phase mole fractions,
    - `Zj` the compressibility factors of each phase.

    Raises
    ------
    `SolutionNotFoundError` if the flash calculation procedure terminates
    unsuccessfully.
    """
    if kvji0 is None:
      kvji0 = self.eos.getPT_kvguess(P, T, yi, self.level)
    if self.useprev and self.preserved:
      kvji0 = *self.prevkvji, *kvji0
    if self.runstab:
      stab = self.stabsolver.run(P, T, yi, kvji0)
      if stab.stable:
        return FlashResult(yji=np.vstack([yi, self._1pstab_yi]),
                           Fj=self._1pstab_Fj, Zj=np.array([stab.Z, 0.]),
                           gnorm=0.)
      else:
        self.negflash = False
      if self.useprev and self.preserved:
        kvji0 = *self.prevkvji, *stab.kvji
      else:
        kvji0 = stab.kvji
    flash = self.solver(P, T, yi, kvji0, negflash=self.negflash)
    if self.useprev:
      self.prevkvji = flash.kvji
      self.preserved = True
    return flash


def _flash2pPT_ss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: Flash2pEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 30,
  negflash: bool = True,
) -> FlashResult:
  """Successive substitution method for two-phase flash calculations
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

  eos: Flash2pEosPT
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

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Two-phase flash calculation (SS method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Fv', 'gnorm',
  )
  tmpl = '%3s%5s' + Nc * ' %9.4f' + ' %8.4f %10.2e'
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm)
    while gnorm > tol and k < maxiter:
      k += 1
      lnkvik -= gi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm)
    if gnorm < tol and np.isfinite(gnorm) and (0. < Fv < 1. or negflash):
      rhol = yli.dot(eos.mwi) / Zl
      rhov = yvi.dot(eos.mwi) / Zv
      kvji = np.atleast_2d(kvik)
      if rhov < rhol:
        yji = np.vstack([yvi, yli])
        Fj = np.array([Fv, 1. - Fv])
        Zj = np.array([Zv, Zl])
      else:
        yji = np.vstack([yli, yvi])
        Fj = np.array([1. - Fv, Fv])
        Zj = np.array([Zl, Zv])
      logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                         gnorm=gnorm, Niter=k)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully.\n"
      "The solution method was SS, EOS: %s.\nParameters:\nP = %s Pa"
      "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
      eos.name, P, T, yi, *kvji0,
    )
    raise SolutionNotFoundError(
      'The flash calculation procedure\nterminates unsuccessfully. Try '
      'to increase the maximum number of\nsolver iterations. It also may be '
      'advisable to improve the initial\nguesses of k-values.'
    )


def _flash2pPT_qnss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: Flash2pEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 30,
  negflash: bool = True,
) -> FlashResult:
  """QNSS-method for two-phase flash calculations using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
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

  eos: Flash2pEosPT
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

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Two-phase flash calculation (QNSS method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Fv', 'gnorm',
  )
  tmpl = '%3s%5s' + Nc * ' %9.4f' + ' %8.4f %10.2e'
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    lmbd = 1.
    logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm)
    while gnorm > tol and k < maxiter:
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvik += dlnkvi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm)
      if gnorm < tol:
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if gnorm < tol and np.isfinite(gnorm) and (0. < Fv < 1. or negflash):
      rhol = yli.dot(eos.mwi) / Zl
      rhov = yvi.dot(eos.mwi) / Zv
      kvji = np.atleast_2d(kvik)
      if rhov < rhol:
        yji = np.vstack([yvi, yli])
        Fj = np.array([Fv, 1. - Fv])
        Zj = np.array([Zv, Zl])
      else:
        yji = np.vstack([yli, yvi])
        Fj = np.array([1. - Fv, Fv])
        Zj = np.array([Zl, Zv])
      logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                         gnorm=gnorm, Niter=k)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully.\n"
      "The solution method was QNSS, EOS: %s.\nParameters:\nP = %s Pa"
      "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
      eos.name, P, T, yi, *kvji0,
    )
    raise SolutionNotFoundError(
      'The flash calculation procedure\nterminates unsuccessfully. Try '
      'to increase the maximum number of\nsolver iterations. It also may be '
      'advisable to improve the initial\nguesses of k-values.'
    )


def _flash2pPT_newt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: Flash2pEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 30,
  negflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient.

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

  eos: Flash2pEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

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

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate Newton's method successfully if the norm of the gradient
    of Gibbs energy function is less than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Two-phase flash calculation (Newton's method).")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s%8s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Fv', 'gnorm', 'method',
  )
  tmpl = '%3s%5s' + Nc * ' %9.4f' + ' %8.4f %10.2e %7s'
  U = np.full(shape=(Nc, Nc), fill_value=-1.)
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    if np.isclose(Fv, 0.):
      Fv = 1e-8
    elif np.isclose(Fv, 1.):
      Fv = 0.99999999
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
    lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm, 'Newt')
    while gnorm > tol and k < maxiter:
      ui = yi / (yli * yvi) - 1.
      FvFl = 1. / (Fv * (1. - Fv))
      np.fill_diagonal(U, ui)
      H = U * FvFl + (dlnphividnj + dlnphilidnj)
      dnvi = linsolver(H, -gi)
      dlnkvi = U.dot(dnvi) * FvFl
      k += 1
      lnkvik += dlnkvi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
      lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
      gi = lnkvik + lnphivi - lnphili
      gnormkp1 = np.linalg.norm(gi)
      if gnormkp1 < gnorm or forcenewton:
        gnorm = gnormkp1
        logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm, 'Newt')
      else:
        lnkvik -= gi
        kvik = np.exp(lnkvik)
        Fv = solve2p_FGH(kvik, yi)
        yli = yi / ((kvik - 1.) * Fv + 1.)
        yvi = yli * kvik
        lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
        lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
        gi = lnkvik + lnphivi - lnphili
        gnorm = np.linalg.norm(gi)
        logger.debug(tmpl, j, k, *lnkvik, Fv, gnorm, 'SS')
    if gnorm < tol and np.isfinite(gnorm) and (0. < Fv < 1. or negflash):
      rhol = yli.dot(eos.mwi) / Zl
      rhov = yvi.dot(eos.mwi) / Zv
      kvji = np.atleast_2d(kvik)
      if rhov < rhol:
        yji = np.vstack([yvi, yli])
        Fj = np.array([Fv, 1. - Fv])
        Zj = np.array([Zv, Zl])
      else:
        yji = np.vstack([yli, yvi])
        Fj = np.array([1. - Fv, Fv])
        Zj = np.array([Zl, Zv])
      logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                         gnorm=gnorm, Niter=k)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully.\n"
      "The solution method was Newton, EOS: %s.\nParameters:\nP = %s Pa"
      "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
      eos.name, P, T, yi, *kvji0,
    )
    raise SolutionNotFoundError(
      'The flash calculation procedure\nterminates unsuccessfully. Try '
      'to increase the maximum number of\nsolver iterations. It also may be '
      'advisable to improve the initial\nguesses of k-values.'
    )


def _flash2pPT_ssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: Flash2pEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 30,
  tol_ss: Scalar = 1e-2,
  maxiter_ss: int = 10,
  negflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient. Successive substitution iterations
  precede Newton's method to improve initial guesses of k-values.

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

  eos: Flash2pEosPT
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

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  tol_ss: Scalar
    Switch to Newton's method if the norm of the vector of equilibrium
    equations is less than `tol_ss`. Default is `1e-2`.

  maxiter_ss: int
    The maximum number of the successive substitution iterations.
    Default is `10`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info('Two-phase flash calculation (SS-Newton method).')
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s%8s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Fv', 'gnorm', 'method',
  )
  tmpl = '%3s%5s' + Nc * ' %9.4f' + ' %8.4f %10.2e %7s'
  for i, kvi0 in enumerate(kvji0):
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'SS')
    while gnorm > tol_ss and k < maxiter_ss:
      k += 1
      lnkvik -= gi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'SS')
    if np.isfinite(gnorm):
      if gnorm < tol:
        if 0. < Fv < 1. or negflash:
          rhol = yli.dot(eos.mwi) / Zl
          rhov = yvi.dot(eos.mwi) / Zv
          kvji = np.atleast_2d(kvik)
          if rhov < rhol:
            yji = np.vstack([yvi, yli])
            Fj = np.array([Fv, 1. - Fv])
            Zj = np.array([Zv, Zl])
          else:
            yji = np.vstack([yli, yvi])
            Fj = np.array([1. - Fv, Fv])
            Zj = np.array([Zl, Zv])
          logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                             gnorm=gnorm, Niter=k)
      else:
        U = np.full(shape=(Nc, Nc), fill_value=-1.)
        lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
        lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
        logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'Newt')
        while gnorm > tol and k < maxiter:
          ui = yi / (yli * yvi) - 1.
          FvFl = 1. / (Fv * (1. - Fv))
          np.fill_diagonal(U, ui)
          H = U * FvFl + (dlnphividnj + dlnphilidnj)
          dnvi = linsolver(H, -gi)
          dlnkvi = U.dot(dnvi) * FvFl
          k += 1
          lnkvik += dlnkvi
          kvik = np.exp(lnkvik)
          Fv = solve2p_FGH(kvik, yi)
          yli = yi / ((kvik - 1.) * Fv + 1.)
          yvi = yli * kvik
          lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1.-Fv)
          lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
          gi = lnkvik + lnphivi - lnphili
          gnormkp1 = np.linalg.norm(gi)
          if gnormkp1 < gnorm or forcenewton:
            gnorm = gnormkp1
            logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'Newt')
          else:
            lnkvik -= gi
            kvik = np.exp(lnkvik)
            Fv = solve2p_FGH(kvik, yi)
            yli = yi / ((kvik - 1.) * Fv + 1.)
            yvi = yli * kvik
            lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P,T, yli, 1.-Fv)
            lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P,T, yvi, Fv)
            gi = lnkvik + lnphivi - lnphili
            gnorm = np.linalg.norm(gi)
            logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'SS')
        if gnorm < tol and np.isfinite(gnorm) and (0. < Fv < 1. or negflash):
          rhol = yli.dot(eos.mwi) / Zl
          rhov = yvi.dot(eos.mwi) / Zv
          kvji = np.atleast_2d(kvik)
          if rhov < rhol:
            yji = np.vstack([yvi, yli])
            Fj = np.array([Fv, 1. - Fv])
            Zj = np.array([Zv, Zl])
          else:
            yji = np.vstack([yli, yvi])
            Fj = np.array([1. - Fv, Fv])
            Zj = np.array([Zl, Zv])
          logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                             gnorm=gnorm, Niter=k)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully.\n"
      "The solution method was SS+Newton, EOS: %s.\nParameters:\nP = %s Pa"
      "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
      eos.name, P, T, yi, *kvji0,
    )
    raise SolutionNotFoundError(
      'The flash calculation procedure\nterminates unsuccessfully. Try '
      'to increase the maximum number of\nsolver iterations. It also may be '
      'advisable to improve the initial\nguesses of k-values.'
    )


def _flash2pPT_qnssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: tuple[Vector, ...],
  eos: Flash2pEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 30,
  tol_qnss: Scalar = 1e-2,
  maxiter_qnss: int = 10,
  negflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient. Quasi-Newton Successive Substitution
  (QNSS) iterations precede Newton's method to improve initial guesses
  of k-values.

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

  eos: Flash2pEosPT
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

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  tol_qnss: Scalar
    Switch to Newton's method if the norm of the vector of equilibrium
    equations is less than `tol_qnss`. Default is `1e-2`.

  maxiter_qnss: int
    The maximum number of quasi-newton successive substitution
    iterations. Default is `10`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info('Two-phase flash calculation (QNSS-Newton method).')
  Nc = eos.Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * ' %6.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s%8s',
    'Nkv', 'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Fv', 'gnorm', 'method',
  )
  tmpl = '%3s%5s' + Nc * ' %9.4f' + ' %8.4f %10.2e %7s'
  for i, kvi0 in enumerate(kvji0):
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    lmbd = 1.
    logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'QNSS')
    while gnorm > tol_qnss and k < maxiter_qnss:
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvik += dlnkvi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'QNSS')
      if gnorm < tol:
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if np.isfinite(gnorm):
      if gnorm < tol:
        if 0. < Fv < 1. or negflash:
          rhol = yli.dot(eos.mwi) / Zl
          rhov = yvi.dot(eos.mwi) / Zv
          kvji = np.atleast_2d(kvik)
          if rhov < rhol:
            yji = np.vstack([yvi, yli])
            Fj = np.array([Fv, 1. - Fv])
            Zj = np.array([Zv, Zl])
          else:
            yji = np.vstack([yli, yvi])
            Fj = np.array([1. - Fv, Fv])
            Zj = np.array([Zl, Zv])
          logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                             gnorm=gnorm, Niter=k)
      else:
        U = np.full(shape=(Nc, Nc), fill_value=-1.)
        lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
        lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
        logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'Newt')
        while gnorm > tol and k < maxiter:
          ui = yi / (yli * yvi) - 1.
          FvFl = 1. / (Fv * (1. - Fv))
          np.fill_diagonal(U, ui)
          H = U * FvFl + (dlnphividnj + dlnphilidnj)
          dnvi = linsolver(H, -gi)
          dlnkvi = U.dot(dnvi) * FvFl
          k += 1
          lnkvik += dlnkvi
          kvik = np.exp(lnkvik)
          Fv = solve2p_FGH(kvik, yi)
          yli = yi / ((kvik - 1.) * Fv + 1.)
          yvi = yli * kvik
          lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1.-Fv)
          lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
          gi = lnkvik + lnphivi - lnphili
          gnormkp1 = np.linalg.norm(gi)
          if gnormkp1 < gnorm or forcenewton:
            gnorm = gnormkp1
            logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'Newt')
          else:
            lnkvik -= gi
            kvik = np.exp(lnkvik)
            Fv = solve2p_FGH(kvik, yi)
            yli = yi / ((kvik - 1.) * Fv + 1.)
            yvi = yli * kvik
            lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P,T, yli, 1.-Fv)
            lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P,T, yvi, Fv)
            gi = lnkvik + lnphivi - lnphili
            gnorm = np.linalg.norm(gi)
            logger.debug(tmpl, i, k, *lnkvik, Fv, gnorm, 'SS')
        if gnorm < tol and np.isfinite(gnorm) and (0. < Fv < 1. or negflash):
          rhol = yli.dot(eos.mwi) / Zl
          rhov = yvi.dot(eos.mwi) / Zv
          kvji = np.atleast_2d(kvik)
          if rhov < rhol:
            yji = np.vstack([yvi, yli])
            Fj = np.array([Fv, 1. - Fv])
            Zj = np.array([Zv, Zl])
          else:
            yji = np.vstack([yli, yvi])
            Fj = np.array([1. - Fv, Fv])
            Zj = np.array([Zl, Zv])
          logger.info('Vapour mole fraction: Fv = %.4f.', Fj[0])
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji,
                             gnorm=gnorm, Niter=k)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully.\n"
      "The solution method was SS+Newton, EOS: %s.\nParameters:\nP = %s Pa"
      "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
      eos.name, P, T, yi, *kvji0,
    )
    raise SolutionNotFoundError(
      'The flash calculation procedure\nterminates unsuccessfully. Try '
      'to increase the maximum number of\nsolver iterations. It also may be '
      'advisable to improve the initial\nguesses of k-values.'
    )

