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
  solveNp,
)

from typing import (
  Callable,
  Iterable,
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


class FlashnpEosPT(Flash2pEosPT):

  def getPT_lnphiji_Zj(
    self,
    P: Scalar,
    T: Scalar,
    yji: Matrix,
  ) -> tuple[Matrix, Vector]: ...


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
  eos: Flash2pEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
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
      For a given pressure [Pa], temperature [K], mole composition
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

  method: str
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

  **kwargs: dict
    Other arguments for a flash solver. It may contain such arguments
    as `tol`, `maxiter` and others, depending on the selected solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> FlashResult
    This method performs two-phase flash calculations for a given
    pressure in [Pa], temperature in [K], mole composition `yi` of shape
    `(Nc,)`, and returns flash calculation results as an instance of
    `FlashResult`.
  """
  def __init__(
    self,
    eos: Flash2pEosPT,
    method: str = 'ss',
    negflash: bool = True,
    runstab: bool = True,
    useprev: bool = False,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.negflash = negflash
    self.runstab = runstab
    self.useprev = useprev
    self.stabsolver = stabilityPT(eos, **stabkwargs)
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
    pass

  def run(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    kvji0: Iterable[Vector] | None = None,
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

    kvji0: Iterable[Vector] | None
      An iterable object containing arrays of initial k-value guesses.
      Each array's shape should be `(Nc,)`. Default is `None` which
      means to use initial guesses from the method `getPT_kvguess` of
      the initialized instance of an EOS.

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
      kvji0 = self.eos.getPT_kvguess(P, T, yi)
    if self.useprev and self.preserved:
      kvji0 = *self.prevkvji, *kvji0
    if self.runstab:
      stab = self.stabsolver.run(P, T, yi, kvji0)
      if stab.stable:
        return FlashResult(yji=np.atleast_2d(yi), Fj=np.array([1.]),
                           Zj=np.array([stab.Z]), gnorm=0., Niter=0)
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
  kvji0: Iterable[Vector],
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

  kvji0: Iterable[Vector]
    An iterable object containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

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
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'Fv', 'gnorm',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e'
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
    if (gnorm < tol and np.isfinite(gnorm) and
        (Fv > 0. and Fv < 1. or negflash)):
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
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
    eos.name, P, T, yi, *kvji0,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flash2pPT_qnss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: Iterable[Vector],
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

  kvji0: Iterable[Vector]
    An iterable object containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

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
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'Fv', 'gnorm',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e'
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
    if (gnorm < tol and np.isfinite(gnorm) and
        (Fv > 0. and Fv < 1. or negflash)):
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
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was QNSS, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
    eos.name, P, T, yi, *kvji0,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flash2pPT_newt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: Iterable[Vector],
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

  kvji0: Iterable[Vector]
    An iterable object containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

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
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s%8s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'Fv', 'gnorm',
    'method',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
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
    if (gnorm < tol and np.isfinite(gnorm) and
        (Fv > 0. and Fv < 1. or negflash)):
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
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
    eos.name, P, T, yi, *kvji0,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flash2pPT_ssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: Iterable[Vector],
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

  kvji0: Iterable[Vector]
    An iterable object containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

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
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s%8s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'Fv', 'gnorm',
    'method',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
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
        if Fv > 0. and Fv < 1. or negflash:
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
        if (gnorm < tol and np.isfinite(gnorm) and
            (Fv > 0. and Fv < 1. or negflash)):
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
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS+Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
    eos.name, P, T, yi, *kvji0,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flash2pPT_qnssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  kvji0: Iterable[Vector],
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

  kvji0: Iterable[Vector]
    An iterable object containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

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
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Nc * '%10s' + '%9s%11s%8s',
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'Fv', 'gnorm',
    'method',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
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
        if Fv > 0. and Fv < 1. or negflash:
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
        if (gnorm < tol and np.isfinite(gnorm) and
            (Fv > 0. and Fv < 1. or negflash)):
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
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS+Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * '\n%s',
    eos.name, P, T, yi, *kvji0,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


class flashnpPT(object):
  """Multiphase flash calculations.

  Performs multiphase flash calculations for a given pressure [Pa],
  temperature [K] and composition of the mixture.

  Parameters
  ----------
  eos: FlashnpEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
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

    If the solution method for two-phase flash calculations would be
    one of `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also
    must have:

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K], mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    To perform multiphase flash calculations, this instance of an
    equation of state class must have:

    - `getPT_Z(P: Scalar, T: Scalar, yi: Vector) -> Scalar`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return the
      compressibility factor of the mixture.

    - `getPT_lnphiji_Zj(P: Scalar, T: Scalar,
                        yji: Matrix) -> tuple[Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of

      - logarithms of the fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`,
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Selects two- and multiphase flash solvers. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Currently raises `NotImplementedError`),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Currently raises `NotImplementedError`),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `False`.

  stabkwargs: dict
    Dictionary that used to regulate the stability test procedure.
    Default is an empty dictionary.

  flash2pkwargs: dict
    Dictionary that used to regulate the two-phase flash calculation
    procedure. Default is an empty dictionary.

  **kwargs: dict
    Other arguments for a flash calculation solver. It may contain such
    arguments as `tol`, `maxiter` and others, depending on the selected
    flash calculation solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector, maxNp: int= 3) -> FlashResult
    This method performs multiphase flash calculation for a given
    pressure in [Pa], temperature in [K], mole composition `yi` of shape
    `(Nc,)` and the maximum number of phases. It returns flash
    calculation results as an instance of `FlashResult`.

  initialize(P: Scalar, T: Scalar,
             flash: FlashResult) -> tuple[list[Matrix], list[Vector]]`
    This method generates initial guesses of k-values and phase mole
    fractions for multiphase flash calculation by performing the
    stability test procedure for each phase determined a previous state.
    This method is also used to check the stability of a previous state.
  """
  def __init__(
    self,
    eos: FlashnpEosPT,
    method: str = 'ss',
    negflash: bool = False,
    flash2pkwargs: dict = {},
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.negflash = negflash
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    self.solver2p = flash2pPT(eos, method=method, stabkwargs=stabkwargs,
                              negflash=negflash, **flash2pkwargs, **kwargs)
    if method == 'ss':
      self.solver = partial(_flashnpPT_ss, eos=eos, negflash=negflash,
                            **kwargs)
    elif method == 'qnss':
      raise NotImplementedError(
        'QNSS-method for flash calculations is not implemented yet.'
      )
    elif method == 'bfgs':
      raise NotImplementedError(
        'BFGS-method for flash calculations is not implemented yet.'
      )
    elif method == 'newton':
      raise NotImplementedError(
        "Newton's method for flash calculations is not implemented yet."
      )
    elif method == 'ss-newton':
      raise NotImplementedError(
        'SS-Newton method for flash calculations is not implemented yet.'
      )
    elif method == 'qnss-newton':
      raise NotImplementedError(
        'QNSS-Newton method for flash calculations is not implemented yet.'
      )
    elif method == 'newton-full':
      raise NotImplementedError(
        "Full Newton's method for flash calculations is not implemented yet."
      )
    else:
      raise ValueError(f'The unknown flash-method: {method}.')
    pass

  def run(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    maxNp: int = 3,
    kvji0: Iterable[Vector] | None = None,
  ) -> FlashResult:
    """Performs multiphase flash calculations for given pressure,
    temperature and composition.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    maxNp: int
      The maximum number of phases. Default is `3`.

    kvji0: Iterable[Vector] | None
      An iterable object containing arrays of initial k-value guesses
      for two-phase flash calculations. Each array's shape should be
      `(Nc,)`. Default is `None` which means to use initial guesses
      from the method `getPT_kvguess` of the initialized instance of
      an EOS.

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
    if maxNp < 2:
      Z = self.eos.getPT_Z(P, T, yi)
      return FlashResult(yji=np.atleast_2d(yi), Fj=np.array([1.]),
                         Zj=np.array([Z]), gnorm=0., Niter=0)
    res = self.solver2p.run(P, T, yi, kvji0)
    if res.yji.shape[0] == 1:
      return res
    for k in range(3, maxNp + 1):
      kvsji, fsj = self.initialize(P, T, res)
      if kvsji:
        logger.debug('Running %sp-flash...', k)
        res = self.solver(P, T, yi, fsj, kvsji)
      else:
        return res
    return res

  def initialize(
    self,
    P: Scalar,
    T: Scalar,
    flash: FlashResult,
  ) -> tuple[list[Matrix], list[Vector]]:
    """Generates initial guesses of k-values and phase mole fractions
    for `Np`-phase (`Np >= 3`) flash calculation.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    flash: FlashResult
      `(Np-1)`-flash calculation results as an instance of
      `FlashResult`.

    Returns
    -------
    A tuple of initial guesses of k-values (list of matrices) and phase
    mole fractions (list of 1d-arrays).
    """
    yji = flash.yji
    Fj = flash.Fj
    kvsji = []
    fsj = []
    for r in range(yji.shape[0]):
      xi = yji[r]
      stab = self.stabsolver.run(P, T, xi)
      if not stab.stable:
        kvji = yji / xi
        kvji[r] = stab.kvji[0]
        kvsji.append(kvji)
        fj = Fj.copy()
        fj[r] = 0.
        fsj.append(fj)
    return kvsji, fsj


def _flashnpPT_ss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  fsj0: Iterable[Vector],
  kvsji0: Iterable[Matrix],
  eos: FlashnpEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 30,
  negflash: bool = True,
) -> FlashResult:
  """Successive substitution method for multiphase flash calculations
  using a PT-based equation of state.

  Parameters
  ----------
  P: Scalar
    Pressure of the mixture [Pa].

  T: Scalar
    Temperature of the mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  fsj0: Iterable[Vector]
    An iterable object containing 1d-arrays of initial guesses of
    phase mole fraction. Each array's shape should be `(Np - 1,)`.

  kvji0: Iterable[Matrix]
    An iterable object containing 2d-arrays of initial k-value guesses.
    Each array's shape should be `(Np - 1, Nc)`.

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

    - `getPT_lnphiji_Zj(P: Scalar, T: Scalar,
                        yji: Matrix) -> tuple[Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of

      - logarithms of the fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`,
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases.

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
  logger.info("Multiphase flash calculation (SS method).")
  Nc = eos.Nc
  Npm1 = kvsji0[0].shape[0]
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + (Npm1 * Nc) * '%10s' + Npm1 * '%9s' + '%11s', 'Nkv', 'Nit',
    *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['F%s' % j for j in range(Npm1)], 'gnorm',
  )
  tmpl = '%3s%5s' + (Npm1 * Nc) * '%10.4f' + Npm1 * '%9.4f' + '%11.2e'
  for s, (fjk, kvjik) in enumerate(zip(fsj0, kvsji0)):
    k = 0
    lnkvjik = np.log(kvjik)
    fjk = solveNp(kvjik, yi, fjk)
    xi = yi / (fjk.dot(kvjik - 1.) + 1.)
    yji = kvjik * xi
    lnphiyji, Zyj = eos.getPT_lnphiji_Zj(P, T, yji)
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, xi)
    gji = lnkvjik + lnphiyji - lnphixi
    gnorm = np.linalg.norm(gji)
    logger.debug(tmpl, s, k, *lnkvjik.ravel(), *fjk, gnorm)
    while gnorm > tol and k < maxiter:
      k += 1
      lnkvjik -= gji
      kvjik = np.exp(lnkvjik)
      fjk = solveNp(kvjik, yi, fjk)
      xi = yi / (fjk.dot(kvjik - 1.) + 1.)
      yji = kvjik * xi
      lnphiyji, Zyj = eos.getPT_lnphiji_Zj(P, T, yji)
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, xi)
      gji = lnkvjik + lnphiyji - lnphixi
      gnorm = np.linalg.norm(gji)
      logger.debug(tmpl, s, k, *lnkvjik.ravel(), *fjk, gnorm)
    if (gnorm < tol and np.isfinite(gnorm) and
        ((fjk < 1.).all() and (fjk > 0.).all() or negflash)):
      yji = np.vstack([yji, xi])
      Fj = np.hstack([fjk, 1. - fjk.sum()])
      Zj = np.hstack([Zyj, Zx])
      rhoj = yji.dot(eos.mwi) / Zj
      idx = np.argsort(rhoj)
      yji = yji[idx]
      kvji = yji[:-1] / yji[-1]
      logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*Fj,))
      return FlashResult(yji=yji, Fj=Fj[idx], Zj=Zj[idx], kvji=kvji,
                         gnorm=gnorm, Niter=k)
  logger.warning(
    "Multiphase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s",
    eos.name, P, T, yi,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )

