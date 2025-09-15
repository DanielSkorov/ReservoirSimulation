import logging

from dataclasses import (
  dataclass,
)

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
  Optional,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Tensor,
  IVector,
  Eos,
  SolutionNotFoundError,
)


logger = logging.getLogger('flash')


class EosFlash2pPT(Eos):

  def getPT_kvguess(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> Iterable[Vector]: ...

  def getPT_Z_lnphii(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector]: ...

  def getPT_Z_lnphii_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[int, Scalar, Vector, Matrix]: ...


class EosFlashnpPT(EosFlash2pPT):

  def getPT_Zj_lnphiji(
    self,
    P: Scalar,
    T: Scalar,
    yji: Matrix,
  ) -> tuple[IVector, Vector, Matrix]: ...

  def getPT_Zj_lnphiji_dnk(
    self,
    P: Scalar,
    T: Scalar,
    yji: Matrix,
  ) -> tuple[IVector, Vector, Matrix, Tensor]: ...


@dataclass
class FlashResult(object):
  """Container for flash calculation outputs with pretty-printing.

  Attributes
  ----------
  yji: Matrix, shape (Np, Nc)
    Mole fractions of components in each phase as a `Matrix` of shape
    `(Np, Nc)`, where `Np` is the number of phases and `Nc` is the
    number of components.

  Fj: Vector, shape (Np,)
    Phase mole fractions. `Vector` of real elements of shape `(Np,)`,
    where `Np` is the number of phases.

  Zj: Vector, shape (Np,)
    Phase compressibility factors as a `Vector` of shape `(Np,)`,
    where `Np` is the number of phases.

  sj: IVector, shape (Np,)
    The designated phases (states) as a 1d-array of integers with the
    shape `(Np,)` (`0` = vapour, `1` = liquid, etc.).

  kvji: Matrix, shape (Np - 1, Nc) | None
    K-values of components in non-reference phases as a `Matrix` of
    shape `(Np - 1, Nc)`, where `Np` is the number of phases and `Nc`
    is the number of components. For a one phase state, this attribute
    is `None`.
  """
  yji: Matrix
  Fj: Vector
  Zj: Vector
  sj: IVector
  kvji: Optional[Matrix] = None

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      return (f"Phase compositions:\n{self.yji}\n"
              f"Phase mole fractions:\n{self.Fj}\n"
              f"Phase compressibility factors:\n{self.Zj}")


class flash2pPT(object):
  """Two-phase flash calculations.

  Performs two-phase flash calculations for a given pressure [Pa],
  temperature [K] and composition of the mixture.

  Parameters
  ----------
  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar, yi: Vector)
       -> Iterable[Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as an iterable object of `Vector` of shape
      `(Nc,)`.

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
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

    Default is `'qnss-newton'`.

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
  """
  def __init__(
    self,
    eos: EosFlash2pPT,
    method: str = 'qnss-newton',
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
    - `yji` component mole fractions in each phase,
    - `Fj` phase mole fractions,
    - `Zj` the compressibility factor of each phase.

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
        return FlashResult(np.atleast_2d(yi), np.array([1.]),
                           np.array([stab.Z]), np.array([stab.s]))
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
  eos: EosFlash2pPT,
  tol: Scalar = 1e-16,
  maxiter: int = 100,
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

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

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
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e'
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvi = kvi0
    lnkvi = np.log(kvi)
    f0 = solve2p_FGH(kvi, yi)
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
    gi = lnkvi + lnphi0i - lnphi1i
    g2 = gi.dot(gi)
    logger.debug(tmpl, j, k, *lnkvi, f0, g2)
    while g2 > tol and k < maxiter:
      k += 1
      lnkvi -= gi
      kvi = np.exp(lnkvi)
      f0 = solve2p_FGH(kvi, yi, f0)
      y1i = yi / ((kvi - 1.) * f0 + 1.)
      y0i = y1i * kvi
      s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2 = gi.dot(gi)
      logger.debug(tmpl, j, k, *lnkvi, f0, g2)
    if g2 < tol and np.isfinite(g2) and (f0 > 0. and f0 < 1. or negflash):
      if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
        yji = np.vstack([y0i, y1i])
        Fj = np.array([f0, 1. - f0])
        Zj = np.array([Z0, Z1])
        sj = np.array([s0, s1])
        kvji = np.atleast_2d(kvi)
      else:
        yji = np.vstack([y1i, y0i])
        Fj = np.array([1. - f0, f0])
        Zj = np.array([Z1, Z0])
        sj = np.array([s1, s0])
        kvji = np.atleast_2d(1. / kvi)
      logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
      return FlashResult(yji, Fj, Zj, sj, kvji)
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * "\n%s",
    eos.name, P, T, yi.tolist(), *map(lambda a: a.tolist(), kvji0),
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
  eos: EosFlash2pPT,
  tol: Scalar = 1e-16,
  maxiter: int = 50,
  lmbdmax: Scalar = 6.,
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

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `50`.

  lmbdmax: Scalar
    The maximum step length. Default is `6.0`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

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
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e'
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvi = kvi0
    lnkvi = np.log(kvi)
    f0 = solve2p_FGH(kvi, yi)
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
    gi = lnkvi + lnphi0i - lnphi1i
    g2 = gi.dot(gi)
    lmbd = 1.
    logger.debug(tmpl, j, k, *lnkvi, f0, g2)
    while g2 > tol and k < maxiter:
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvi += dlnkvi
      kvi = np.exp(lnkvi)
      f0 = solve2p_FGH(kvi, yi, f0)
      y1i = yi / ((kvi - 1.) * f0 + 1.)
      y0i = y1i * kvi
      s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2km1 = g2
      g2 = gi.dot(gi)
      logger.debug(tmpl, j, k, *lnkvi, f0, g2)
      if g2 < tol:
        break
      if k % Nc == 0 or g2 > g2km1:
        lmbd = 1.
      else:
        lmbd *= tkm1 / (dlnkvi.dot(gi) - tkm1)
        if lmbd < 0.:
          lmbd = -lmbd
        if lmbd > lmbdmax:
          lmbd = lmbdmax
    if g2 < tol and np.isfinite(g2) and (f0 > 0. and f0 < 1. or negflash):
      if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
        yji = np.vstack([y0i, y1i])
        Fj = np.array([f0, 1. - f0])
        Zj = np.array([Z0, Z1])
        sj = np.array([s0, s1])
        kvji = np.atleast_2d(kvi)
      else:
        yji = np.vstack([y1i, y0i])
        Fj = np.array([1. - f0, f0])
        Zj = np.array([Z1, Z0])
        sj = np.array([s1, s0])
        kvji = np.atleast_2d(1. / kvi)
      logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
      return FlashResult(yji, Fj, Zj, sj, kvji)
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was QNSS, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * "\n%s",
    eos.name, P, T, yi.tolist(), *map(lambda a: a.tolist(), kvji0),
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
  eos: EosFlash2pPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  negflash: bool = True,
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

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
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
    Terminate Newton's method successfully if the sum of squared
    elements of the gradient of Gibbs energy function is less than
    `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

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
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2', 'method',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
  U = np.full(shape=(Nc, Nc), fill_value=-1.)
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvi = kvi0
    lnkvi = np.log(kvi)
    f0 = solve2p_FGH(kvi, yi)
    if np.isclose(f0, 0.):
      f0 = 1e-8
    elif np.isclose(f0, 1.):
      f0 = 0.99999999
    f1 = 1. - f0
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
    s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
    gi = lnkvi + lnphi0i - lnphi1i
    g2 = gi.dot(gi)
    logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'Newt')
    while g2 > tol and k < maxiter:
      ui = yi / (y1i * y0i) - 1.
      F0F1 = 1. / (f0 * f1)
      np.fill_diagonal(U, ui)
      H = U * F0F1 + (dlnphi0idnj + dlnphi1idnj)
      dn0i = linsolver(H, -gi)
      dlnkvi = U.dot(dn0i) * F0F1
      k += 1
      lnkvi += dlnkvi
      kvi = np.exp(lnkvi)
      f0 = solve2p_FGH(kvi, yi, f0)
      f1 = 1. - f0
      y1i = yi / ((kvi - 1.) * f0 + 1.)
      y0i = y1i * kvi
      s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
      s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
      gi = lnkvi + lnphi0i - lnphi1i
      g2kp1 = gi.dot(gi)
      if g2kp1 < g2 or forcenewton:
        g2 = g2kp1
        logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'Newt')
      else:
        lnkvi -= gi
        kvi = np.exp(lnkvi)
        f0 = solve2p_FGH(kvi, yi, f0)
        f1 = 1. - f0
        y1i = yi / ((kvi - 1.) * f0 + 1.)
        y0i = y1i * kvi
        s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
        s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
        gi = lnkvi + lnphi0i - lnphi1i
        g2 = gi.dot(gi)
        logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
    if g2 < tol and np.isfinite(g2) and (f0 > 0. and f0 < 1. or negflash):
      if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
        yji = np.vstack([y0i, y1i])
        Fj = np.array([f0, f1])
        Zj = np.array([Z0, Z1])
        sj = np.array([s0, s1])
        kvji = np.atleast_2d(kvi)
      else:
        yji = np.vstack([y1i, y0i])
        Fj = np.array([f1, f0])
        Zj = np.array([Z1, Z0])
        sj = np.array([s1, s0])
        kvji = np.atleast_2d(1. / kvi)
      logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
      return FlashResult(yji, Fj, Zj, sj, kvji)
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * "\n%s",
    eos.name, P, T, yi.tolist(), *map(lambda a: a.tolist(), kvji0),
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
  eos: EosFlash2pPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  switchers: tuple[Scalar, Scalar, Scalar, Scalar] = (0.1, 1e-2, 1e-10, 1e-4),
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  negflash: bool = True,
) -> FlashResult:
  r"""Performs minimization of the Gibbs energy function using Newton's
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

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
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
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  switchers: tuple[Scalar, Scalar, Scalar, Scalar]
    Allows to modify the conditions of switching from the successive
    substitution method to Newton's method. The parameter must be
    represented as a tuple containing four values: :math:`\eps_r`,
    :math:`\eps_f`, :math:`\eps_l`, :math:`\eps_u`. The switching
    conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \left| F_1^k - F_1^{k-1} \right| < \eps_f, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
        0 < F_1^k < 1,
      \end{cases}

    where :math:`\mathbf{g}` is the equilibrium equations vector,
    :math:`k` is the iteration number, :math:`F_1` is the mole fraction
    of the non-reference phase. Analytical expressions of the switching
    conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.1, 1e-2, 1e-10, 1e-4)`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

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
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2', 'method',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvi = kvi0
    lnkvi = np.log(kvi)
    f0 = solve2p_FGH(kvi, yi)
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
    gi = lnkvi + lnphi0i - lnphi1i
    g2 = gi.dot(gi)
    switch = g2 < tol
    logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
    while not switch and g2 > tol and k < maxiter:
      k += 1
      lnkvi -= gi
      kvi = np.exp(lnkvi)
      f0km1 = f0
      f0 = solve2p_FGH(kvi, yi, f0)
      y1i = yi / ((kvi - 1.) * f0 + 1.)
      y0i = y1i * kvi
      s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = (g2 / g2km1 > epsr and
                (f0 - f0km1 < epsf and f0km1 - f0 < epsf) and
                g2 > epsl and g2 < epsu and
                (f0 > 0. and f0 < 1. or negflash))
      logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
    if np.isfinite(g2):
      if g2 < tol:
        if f0 > 0. and f0 < 1. or negflash:
          if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
            yji = np.vstack([y0i, y1i])
            Fj = np.array([f0, 1. - f0])
            Zj = np.array([Z0, Z1])
            sj = np.array([s0, s1])
            kvji = np.atleast_2d(kvi)
          else:
            yji = np.vstack([y1i, y0i])
            Fj = np.array([1. - f0, f0])
            Zj = np.array([Z1, Z0])
            sj = np.array([s1, s0])
            kvji = np.atleast_2d(1. / kvi)
          logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
          return FlashResult(yji, Fj, Zj, sj, kvji)
      elif k < maxiter:
        U = np.full(shape=(Nc, Nc), fill_value=-1.)
        f1 = 1. - f0
        s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
        s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
        logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'Newt')
        while g2 > tol and k < maxiter:
          ui = yi / (y1i * y0i) - 1.
          F0F1 = 1. / (f0 * f1)
          np.fill_diagonal(U, ui)
          H = U * F0F1 + (dlnphi0idnj + dlnphi1idnj)
          dn0i = linsolver(H, -gi)
          dlnkvi = U.dot(dn0i) * F0F1
          k += 1
          lnkvi += dlnkvi
          kvi = np.exp(lnkvi)
          f0 = solve2p_FGH(kvi, yi, f0)
          f1 = 1. - f0
          y1i = yi / ((kvi - 1.) * f0 + 1.)
          y0i = y1i * kvi
          s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
          s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
          gi = lnkvi + lnphi0i - lnphi1i
          g2kp1 = gi.dot(gi)
          if g2kp1 < g2 or forcenewton:
            g2 = g2kp1
            logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'Newt')
          else:
            lnkvi -= gi
            kvi = np.exp(lnkvi)
            f0 = solve2p_FGH(kvi, yi, f0)
            f1 = 1. - f0
            y1i = yi / ((kvi - 1.) * f0 + 1.)
            y0i = y1i * kvi
            s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P,T, y1i,f1)
            s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P,T, y0i,f0)
            gi = lnkvi + lnphi0i - lnphi1i
            g2 = gi.dot(gi)
            logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
        if g2 < tol and np.isfinite(g2) and (f0 > 0. and f0 < 1. or negflash):
          if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
            yji = np.vstack([y0i, y1i])
            Fj = np.array([f0, f1])
            Zj = np.array([Z0, Z1])
            sj = np.array([s0, s1])
            kvji = np.atleast_2d(kvi)
          else:
            yji = np.vstack([y1i, y0i])
            Fj = np.array([f1, f0])
            Zj = np.array([Z1, Z0])
            sj = np.array([s1, s0])
            kvji = np.atleast_2d(1. / kvi)
          logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
          return FlashResult(yji, Fj, Zj, sj, kvji)
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS-Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * "\n%s",
    eos.name, P, T, yi.tolist(), *map(lambda a: a.tolist(), kvji0),
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
  eos: EosFlash2pPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  lmbdmax: Scalar = 6.,
  switchers: tuple[Scalar, Scalar, Scalar, Scalar] = (0.1, 1e-2, 1e-10, 1e-4),
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  negflash: bool = True,
) -> FlashResult:
  r"""Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient. Quasi-Newton Successive
  Substitution (QNSS) iterations precede Newton's method to improve
  initial guesses of k-values.

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

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
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
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  lmbdmax: Scalar
    The maximum step length. Default is `6.0`.

  switchers: tuple[Scalar, Scalar, Scalar, Scalar]
    Allows to modify the conditions of switching from the QNSS to
    Newton's method. The parameter must be represented as a tuple
    containing four values: :math:`\eps_r`, :math:`\eps_f`,
    :math:`\eps_l`, :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \left| F_1^k - F_1^{k-1} \right| < \eps_f, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
        0 < F_1^k < 1,
      \end{cases}

    where :math:`\mathbf{g}` is the equilibrium equations vector,
    :math:`k` is the iteration number, :math:`F_1` is the mole fraction
    of the non-reference phase. Analytical expressions of the switching
    conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.1, 1e-2, 1e-10, 1e-4)`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

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
    'Nkv', 'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2', 'method',
  )
  tmpl = '%3s%5s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  for j, kvi0 in enumerate(kvji0):
    k = 0
    kvi = kvi0
    lnkvi = np.log(kvi)
    f0 = solve2p_FGH(kvi, yi)
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
    gi = lnkvi + lnphi0i - lnphi1i
    g2 = gi.dot(gi)
    lmbd = 1.
    switch = g2 < tol
    logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'QNSS')
    while not switch and g2 > tol and k < maxiter:
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvi += dlnkvi
      kvi = np.exp(lnkvi)
      f0km1 = f0
      f0 = solve2p_FGH(kvi, yi, f0)
      y1i = yi / ((kvi - 1.) * f0 + 1.)
      y0i = y1i * kvi
      s1, Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      s0, Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = (g2 / g2km1 > epsr and
                (f0 - f0km1 < epsf and f0km1 - f0 < epsf) and
                g2 > epsl and g2 < epsu and
                (f0 > 0. and f0 < 1. or negflash))
      logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'QNSS')
      if g2 < tol or switch:
        break
      if k % Nc == 0 or g2 > g2km1:
        lmbd = 1.
      else:
        lmbd *= tkm1 / (dlnkvi.dot(gi) - tkm1)
        if lmbd < 0.:
          lmbd = -lmbd
        if lmbd > lmbdmax:
          lmbd = lmbdmax
    if np.isfinite(g2):
      if g2 < tol:
        if f0 > 0. and f0 < 1. or negflash:
          if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
            yji = np.vstack([y0i, y1i])
            Fj = np.array([f0, 1. - f0])
            Zj = np.array([Z0, Z1])
            sj = np.array([s0, s1])
            kvji = np.atleast_2d(kvi)
          else:
            yji = np.vstack([y1i, y0i])
            Fj = np.array([1. - f0, f0])
            Zj = np.array([Z1, Z0])
            sj = np.array([s1, s0])
            kvji = np.atleast_2d(1. / kvi)
          logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
          return FlashResult(yji, Fj, Zj, sj, kvji)
      elif k < maxiter:
        U = np.full(shape=(Nc, Nc), fill_value=-1.)
        f1 = 1. - f0
        s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
        s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
        logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'Newt')
        while g2 > tol and k < maxiter:
          ui = yi / (y1i * y0i) - 1.
          F0F1 = 1. / (f0 * f1)
          np.fill_diagonal(U, ui)
          H = U * F0F1 + (dlnphi0idnj + dlnphi1idnj)
          dn0i = linsolver(H, -gi)
          dlnkvi = U.dot(dn0i) * F0F1
          k += 1
          lnkvi += dlnkvi
          kvi = np.exp(lnkvi)
          f0 = solve2p_FGH(kvi, yi, f0)
          f1 = 1. - f0
          y1i = yi / ((kvi - 1.) * f0 + 1.)
          y0i = y1i * kvi
          s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
          s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
          gi = lnkvi + lnphi0i - lnphi1i
          g2kp1 = gi.dot(gi)
          if g2kp1 < g2 or forcenewton:
            g2 = g2kp1
            logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'Newt')
          else:
            lnkvi -= gi
            kvi = np.exp(lnkvi)
            f0 = solve2p_FGH(kvi, yi, f0)
            f1 = 1. - f0
            y1i = yi / ((kvi - 1.) * f0 + 1.)
            y0i = y1i * kvi
            s1, Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P,T, y1i,f1)
            s0, Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P,T, y0i,f0)
            gi = lnkvi + lnphi0i - lnphi1i
            g2 = gi.dot(gi)
            logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
        if g2 < tol and np.isfinite(g2) and (f0 > 0. and f0 < 1. or negflash):
          if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
            yji = np.vstack([y0i, y1i])
            Fj = np.array([f0, f1])
            Zj = np.array([Z0, Z1])
            sj = np.array([s0, s1])
            kvji = np.atleast_2d(kvi)
          else:
            yji = np.vstack([y1i, y0i])
            Fj = np.array([f1, f0])
            Zj = np.array([Z1, Z0])
            sj = np.array([s1, s0])
            kvji = np.atleast_2d(1. / kvi)
          logger.info('Phase mole fractions: %.4f, %.4f', *Fj)
          return FlashResult(yji, Fj, Zj, sj, kvji)
  logger.warning(
    "Two-phase flash calculation terminates unsuccessfully.\n"
    "The solution method was QNSS-Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s\nInitial guesses of k-values:" + (j + 1) * "\n%s",
    eos.name, P, T, yi.tolist(), *map(lambda a: a.tolist(), kvji0),
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
  eos: EosFlashnpPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar, yi: Vector)
       -> Iterable[Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as an iterable object of `Vector` of shape
      `(Nc,)`.

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    To perform multiphase flash calculations, this instance of an
    equation of state class must have:

    - `getPT_Z(P: Scalar, T: Scalar, yi: Vector) -> tuple[int, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return the
      designated phase of the fluid and the compressibility factor.

    - `getPT_Zj_lnphiji(P: Scalar, T: Scalar, yji: Matrix)
       -> tuple[IVector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`.

    If the solution method would be one of `'newton'`, `'ss-newton'`
    or `'qnss-newton'` then it also must have:

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    - `getPT_Zj_lnphiji_dnk(P: Scalar, T: Scalar, yji: Matrix, nj: Vector)
       -> tuple[IVector, Vector, Matrix, Tensor]`
      For a given pressure `P` in [Pa], temperature `T` in [K], mole
      fractions of `Nc` components in `Np` phases `yji` of shape
      `(Np, Nc)` and mole numbers of phases in [mol], returns a tuple
      that contains:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a `Tensor`
        of shape `(Np, Nc, Nc)`.

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
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'qnss-newton'`.

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
    flash calculation solver. It also will be passed to a two-phase
    flash calculation solver.
  """
  def __init__(
    self,
    eos: EosFlashnpPT,
    method: str = 'qnss-newton',
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
      self.solver = partial(_flashnpPT_qnss, eos=eos, negflash=negflash,
                            **kwargs)
    elif method == 'newton':
      self.solver = partial(_flashnpPT_newt, eos=eos, negflash=negflash,
                            **kwargs)
    elif method == 'ss-newton':
      self.solver = partial(_flashnpPT_ssnewt, eos=eos, negflash=negflash,
                            **kwargs)
    elif method == 'qnss-newton':
      self.solver = partial(_flashnpPT_qnssnewt, eos=eos, negflash=negflash,
                            **kwargs)
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
    - `yji` component mole fractions in each phase,
    - `Fj` phase mole fractions,
    - `Zj` the compressibility factor of each phase.

    Raises
    ------
    `SolutionNotFoundError` if the flash calculation procedure terminates
    unsuccessfully.
    """
    if maxNp < 2:
      s, Z = self.eos.getPT_Z(P, T, yi)
      return FlashResult(np.atleast_2d(yi), np.array([1.]), np.array([Z]),
                         np.array([s]))
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
  eos: EosFlashnpPT,
  tol: Scalar = 1e-16,
  maxiter: int = 100,
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
    phase mole fractions. Each array's shape should be `(Np - 1,)`,
    where `Np` is the number of phases.

  kvji0: Iterable[Matrix]
    An iterable object containing 2d-arrays of initial guesses of.
    k-values. Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: Scalar, T: Scalar, yji: Matrix)
       -> tuple[IVector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

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
    *['F%s' % j for j in range(Npm1)], 'g2',
  )
  tmpl = '%3s%5s' + (Npm1 * Nc) * '%10.4f' + Npm1 * '%9.4f' + '%11.2e'
  for s, (fj, kvji) in enumerate(zip(fsj0, kvsji0)):
    k = 0
    lnkvji = np.log(kvji)
    lnkvi = lnkvji.ravel()
    fj = solveNp(kvji, yi, fj)
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2 = gi.dot(gi)
    logger.debug(tmpl, s, k, *lnkvi, *fj, g2)
    while g2 > tol and k < maxiter:
      k += 1
      lnkvji -= gji
      kvji = np.exp(lnkvji)
      fj = solveNp(kvji, yi, fj)
      xi = yi / (fj.dot(kvji - 1.) + 1.)
      yji = kvji * xi
      sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2 = gi.dot(gi)
      logger.debug(tmpl, s, k, *lnkvi, *fj, g2)
    if (g2 < tol and np.isfinite(g2) and
        ((fj < 1.).all() and (fj > 0.).all() or negflash)):
      yji = np.append(yji, np.atleast_2d(xi), 0)
      fj = np.append(fj, 1. - fj.sum())
      Zj = np.append(Zj, Zx)
      sj = np.append(sj, sx)
      idx = np.argsort(yji.dot(eos.mwi) / Zj)
      yji = yji[idx]
      kvji = yji[:-1] / yji[-1]
      logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
      return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
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


def _flashnpPT_qnss(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  fsj0: Iterable[Vector],
  kvsji0: Iterable[Matrix],
  eos: EosFlashnpPT,
  tol: Scalar = 1e-16,
  maxiter: int = 50,
  lmbdmax: Scalar = 6.,
  negflash: bool = True,
) -> FlashResult:
  """QNSS-method for multiphase flash calculations using a PT-based
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

  fsj0: Iterable[Vector]
    An iterable object containing 1d-arrays of initial guesses of
    phase mole fractions. Each array's shape should be `(Np - 1,)`,
    where `Np` is the number of phases.

  kvji0: Iterable[Matrix]
    An iterable object containing 2d-arrays of initial guesses of.
    k-values. Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: Scalar, T: Scalar, yji: Matrix)
       -> tuple[IVector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `50`.

  lmbdmax: Scalar
    The maximum step length. Default is `6.0`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Multiphase flash calculation (QNSS method).")
  Nc = eos.Nc
  Npm1 = kvsji0[0].shape[0]
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + (Npm1 * Nc) * '%10s' + Npm1 * '%9s' + '%11s', 'Nkv', 'Nit',
    *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['F%s' % j for j in range(Npm1)], 'g2',
  )
  tmpl = '%3s%5s' + (Npm1 * Nc) * '%10.4f' + Npm1 * '%9.4f' + '%11.2e'
  for s, (fj, kvji) in enumerate(zip(fsj0, kvsji0)):
    k = 0
    lnkvji = np.log(kvji)
    lnkvi = lnkvji.ravel()
    fj = solveNp(kvji, yi, fj)
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2 = gi.dot(gi)
    lmbd = 1.
    logger.debug(tmpl, s, k, *lnkvi, *fj, g2)
    while g2 > tol and k < maxiter:
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvi += dlnkvi
      kvji = np.exp(lnkvji)
      fj = solveNp(kvji, yi, fj)
      xi = yi / (fj.dot(kvji - 1.) + 1.)
      yji = kvji * xi
      sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2km1 = g2
      g2 = gi.dot(gi)
      logger.debug(tmpl, s, k, *lnkvi, *fj, g2)
      if g2 < tol:
        break
      if k % Nc == 0 or g2 > g2km1:
        lmbd = 1.
      else:
        lmbd *= tkm1 / (dlnkvi.dot(gi) - tkm1)
        if lmbd < 0.:
          lmbd = -lmbd
        if lmbd > lmbdmax:
          lmbd = lmbdmax
    if (g2 < tol and np.isfinite(g2) and
        ((fj < 1.).all() and (fj > 0.).all() or negflash)):
      yji = np.append(yji, np.atleast_2d(xi), 0)
      fj = np.append(fj, 1. - fj.sum())
      Zj = np.append(Zj, Zx)
      sj = np.append(sj, sx)
      idx = np.argsort(yji.dot(eos.mwi) / Zj)
      yji = yji[idx]
      kvji = yji[:-1] / yji[-1]
      logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
      return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
  logger.warning(
    "Multiphase flash calculation terminates unsuccessfully.\n"
    "The solution method was QNSS, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s",
    eos.name, P, T, yi,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flashnpPT_newt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  fsj0: Iterable[Vector],
  kvsji0: Iterable[Matrix],
  eos: EosFlashnpPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  negflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> FlashResult:
  """Newton's method for multiphase flash calculations using a PT-based
  equation of state. A switch to the successive substitution iteration
  is implemented if Newton's method does not decrease the norm of the
  gradient of Gibbs energy function.

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
    phase mole fractions. Each array's shape should be `(Np - 1,)`,
    where `Np` is the number of phases.

  kvji0: Iterable[Matrix]
    An iterable object containing 2d-arrays of initial guesses of.
    k-values. Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    - `getPT_Zj_lnphiji_dnk(P: Scalar, T: Scalar, yji: Matrix, nj: Vector)
       -> tuple[IVector, Vector, Matrix, Tensor]`
      For a given pressure `P` in [Pa], temperature `T` in [K], mole
      fractions of `Nc` components in `Np` phases `yji` of shape
      `(Np, Nc)` and mole numbers of phases in [mol], returns a tuple
      that contains:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a `Tensor`
        of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape
    `((Np - 1) * Nc, (Np - 1) * Nc)` and
    a vector `b` of shape `((Np - 1) * Nc,)` and finds a vector `x` of
    shape `((Np - 1) * Nc,)`, which is the solution of the system of
    linear equations `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Multiphase flash calculation (Newton's method).")
  Nc = eos.Nc
  Npm1 = kvsji0[0].shape[0]
  Npm1Nc = Npm1 * Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Npm1Nc * '%10s' + Npm1 * '%9s' + '%11s%8s', 'Nkv', 'Nit',
    *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['F%s' % j for j in range(Npm1)], 'g2', 'method',
  )
  tmpl = '%3s%5s' + Npm1Nc * '%10.4f' + Npm1 * '%9.4f' + '%11.2e%8s'
  H = np.empty(shape=(Npm1Nc, Npm1Nc))
  H_block = np.lib.stride_tricks.as_strided(
    H, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
  )
  H_blockdiag = np.lib.stride_tricks.as_strided(
    H, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
  )
  U = np.empty(shape=(Npm1Nc, Npm1Nc))
  U_block = np.lib.stride_tricks.as_strided(
    U, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
  )
  U_blockdiag = np.lib.stride_tricks.as_strided(
    U, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
  )
  Ux = np.full(shape=(Nc, Nc), fill_value=-1.)
  Ux_diag = np.lib.stride_tricks.as_strided(Ux, (Nc,), (8 * (Nc + 1),))
  Uy = np.full(shape=(Npm1, Nc, Nc), fill_value=-1.)
  Uy_diag = np.lib.stride_tricks.as_strided(
    Uy, (Npm1, Nc), (8 * Nc * Nc, 8 * (Nc + 1)),
  )
  for s, (fj, kvji) in enumerate(zip(fsj0, kvsji0)):
    k = 0
    lnkvji = np.log(kvji)
    lnkvi = lnkvji.ravel()
    fj = solveNp(kvji, yi, fj)
    fj = np.where(
      np.isclose(fj, 0.), 1e-8, np.where(np.isclose(fj, 1.), 0.99999999, fj)
    )
    fx = 1. - fj.sum()
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
    sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2 = gi.dot(gi)
    logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'Newt')
    while g2 > tol and k < maxiter:
      H_block[:] = dlnphixidnk
      H_blockdiag += dlnphijidnk
      Ux_diag[:] = 1. / xi - 1.
      Uy_diag[:] = 1. / yji - 1.
      U_block[:] = Ux / fx
      U_blockdiag += Uy / fj[:, None, None]
      H += U
      dni = np.linalg.solve(H, -gi)
      dlnkvi = U.dot(dni)
      lnkvi += dlnkvi
      k += 1
      kvji = np.exp(lnkvji)
      fj = solveNp(kvji, yi, fj)
      fx = 1. - fj.sum()
      xi = yi / (fj.dot(kvji - 1.) + 1.)
      yji = kvji * xi
      sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
      sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2km1 = g2
      g2 = gi.dot(gi)
      if g2 < g2km1 or forcenewton:
        logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'Newt')
      else:
        lnkvi -= gi
        kvji = np.exp(lnkvji)
        fj = solveNp(kvji, yi, fj)
        fx = 1. - fj.sum()
        xi = yi / (fj.dot(kvji - 1.) + 1.)
        yji = kvji * xi
        sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
        sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
        gji = lnkvji + lnphiji - lnphixi
        gi = gji.ravel()
        g2 = gi.dot(gi)
        logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
    if (g2 < tol and np.isfinite(g2) and
        ((fj < 1.).all() and (fj > 0.).all() or negflash)):
      yji = np.append(yji, np.atleast_2d(xi), 0)
      fj = np.append(fj, 1. - fj.sum())
      Zj = np.append(Zj, Zx)
      sj = np.append(sj, sx)
      idx = np.argsort(yji.dot(eos.mwi) / Zj)
      yji = yji[idx]
      kvji = yji[:-1] / yji[-1]
      logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
      return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
  logger.warning(
    "Multiphase flash calculation terminates unsuccessfully.\n"
    "The solution method was Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s",
    eos.name, P, T, yi,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flashnpPT_ssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  fsj0: Iterable[Vector],
  kvsji0: Iterable[Matrix],
  eos: EosFlashnpPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  switchers: tuple[Scalar, Scalar, Scalar, Scalar] = (0.6, 1e-2, 1e-10, 1e-4),
  negflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> FlashResult:
  r"""Newton's method for multiphase flash calculations using a PT-based
  equation of state. A switch to the successive substitution iteration
  is implemented if Newton's method does not decrease the norm of the
  gradient of Gibbs energy function. Successive substitution iterations
  precede Newton's method to improve initial guesses of k-values.

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
    phase mole fractions. Each array's shape should be `(Np - 1,)`,
    where `Np` is the number of phases.

  kvji0: Iterable[Matrix]
    An iterable object containing 2d-arrays of initial guesses of.
    k-values. Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: Scalar, T: Scalar, yji: Matrix)
       -> tuple[IVector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`.

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    - `getPT_Zj_lnphiji_dnk(P: Scalar, T: Scalar, yji: Matrix, nj: Vector)
       -> tuple[IVector, Vector, Matrix, Tensor]`
      For a given pressure `P` in [Pa], temperature `T` in [K], mole
      fractions of `Nc` components in `Np` phases `yji` of shape
      `(Np, Nc)` and mole numbers of phases in [mol], returns a tuple
      that contains:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a `Tensor`
        of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  switchers: tuple[Scalar, Scalar, Scalar, Scalar]
    Allows to modify the conditions of switching from the successive
    substitution method to Newton's method. The parameter must be
    represented as a tuple containing four values: :math:`\eps_r`,
    :math:`\eps_f`, :math:`\eps_l`, :math:`\eps_u`. The switching
    conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \max_j \left| F_j^k - F_j^{k-1} \right| < \eps_f, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
        0 < F_1^k < 1,
      \end{cases}

    where :math:`\mathbf{g}` is the equilibrium equations vector,
    :math:`k` is the iteration number, :math:`F_j` is the mole fraction
    of a non-reference phase :math:`j`. Analytical expressions of the
    switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.6, 1e-2, 1e-10, 1e-4)`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape
    `((Np - 1) * Nc, (Np - 1) * Nc)` and
    a vector `b` of shape `((Np - 1) * Nc,)` and finds a vector `x` of
    shape `((Np - 1) * Nc,)`, which is the solution of the system of
    linear equations `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Multiphase flash calculation (SS-Newton method).")
  Nc = eos.Nc
  Npm1 = kvsji0[0].shape[0]
  Npm1Nc = Npm1 * Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Npm1Nc * '%10s' + Npm1 * '%9s' + '%11s%8s', 'Nkv', 'Nit',
    *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['F%s' % j for j in range(Npm1)], 'g2', 'method',
  )
  tmpl = '%3s%5s' + Npm1Nc * '%10.4f' + Npm1 * '%9.4f' + '%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  for s, (fj, kvji) in enumerate(zip(fsj0, kvsji0)):
    k = 0
    lnkvji = np.log(kvji)
    lnkvi = lnkvji.ravel()
    fj = solveNp(kvji, yi, fj)
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2 = gi.dot(gi)
    switch = g2 < tol
    logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
    while not switch and g2 > tol and k < maxiter:
      k += 1
      lnkvji -= gji
      kvji = np.exp(lnkvji)
      fjkm1 = fj
      fj = solveNp(kvji, yi, fj)
      xi = yi / (fj.dot(kvji - 1.) + 1.)
      yji = kvji * xi
      sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2km1 = g2
      g2 = gi.dot(gi)
      physfj = (fj > 0.).all() and (fj < 1.).all() or negflash
      switch = (g2 / g2km1 > epsr and np.abs(fj - fjkm1).max() < epsf and
                g2 > epsl and g2 < epsu and physfj)
      logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
    if np.isfinite(g2):
      if g2 < tol:
        if physfj:
          yji = np.append(yji, np.atleast_2d(xi), 0)
          fj = np.append(fj, 1. - fj.sum())
          Zj = np.append(Zj, Zx)
          sj = np.append(sj, sx)
          idx = np.argsort(yji.dot(eos.mwi) / Zj)
          yji = yji[idx]
          kvji = yji[:-1] / yji[-1]
          logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
          return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
      elif k < maxiter:
        H = np.empty(shape=(Npm1Nc, Npm1Nc))
        H_block = np.lib.stride_tricks.as_strided(
          H, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
        )
        H_blockdiag = np.lib.stride_tricks.as_strided(
          H, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
        )
        U = np.empty(shape=(Npm1Nc, Npm1Nc))
        U_block = np.lib.stride_tricks.as_strided(
          U, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
        )
        U_blockdiag = np.lib.stride_tricks.as_strided(
          U, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
        )
        Ux = np.full(shape=(Nc, Nc), fill_value=-1.)
        Ux_diag = np.lib.stride_tricks.as_strided(Ux, (Nc,), (8 * (Nc + 1),))
        Uy = np.full(shape=(Npm1, Nc, Nc), fill_value=-1.)
        Uy_diag = np.lib.stride_tricks.as_strided(
          Uy, (Npm1, Nc), (8 * Nc * Nc, 8 * (Nc + 1)),
        )
        fx = 1. - fj.sum()
        sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
        sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
        logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'Newt')
        while g2 > tol and k < maxiter:
          H_block[:] = dlnphixidnk
          H_blockdiag += dlnphijidnk
          Ux_diag[:] = 1. / xi - 1.
          Uy_diag[:] = 1. / yji - 1.
          U_block[:] = Ux / fx
          U_blockdiag += Uy / fj[:, None, None]
          H += U
          dni = np.linalg.solve(H, -gi)
          dlnkvi = U.dot(dni)
          lnkvi += dlnkvi
          k += 1
          kvji = np.exp(lnkvji)
          fj = solveNp(kvji, yi, fj)
          fx = 1. - fj.sum()
          xi = yi / (fj.dot(kvji - 1.) + 1.)
          yji = kvji * xi
          sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T,
                                                                  yji, fj)
          sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
          gji = lnkvji + lnphiji - lnphixi
          gi = gji.ravel()
          g2km1 = g2
          g2 = gi.dot(gi)
          if g2 < g2km1 or forcenewton:
            logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'Newt')
          else:
            lnkvi -= gi
            kvji = np.exp(lnkvji)
            fj = solveNp(kvji, yi, fj)
            fx = 1. - fj.sum()
            xi = yi / (fj.dot(kvji - 1.) + 1.)
            yji = kvji * xi
            sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T,
                                                                    yji, fj)
            sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T,
                                                                  xi, fx)
            gji = lnkvji + lnphiji - lnphixi
            gi = gji.ravel()
            g2 = gi.dot(gi)
            logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
        if (g2 < tol and np.isfinite(g2) and
            ((fj < 1.).all() and (fj > 0.).all() or negflash)):
          yji = np.append(yji, np.atleast_2d(xi), 0)
          fj = np.append(fj, 1. - fj.sum())
          Zj = np.append(Zj, Zx)
          sj = np.append(sj, sx)
          idx = np.argsort(yji.dot(eos.mwi) / Zj)
          yji = yji[idx]
          kvji = yji[:-1] / yji[-1]
          logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
          return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
  logger.warning(
    "Multiphase flash calculation terminates unsuccessfully.\n"
    "The solution method was SS-Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s",
    eos.name, P, T, yi,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )


def _flashnpPT_qnssnewt(
  P: Scalar,
  T: Scalar,
  yi: Vector,
  fsj0: Iterable[Vector],
  kvsji0: Iterable[Matrix],
  eos: EosFlashnpPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  lmbdmax: Scalar = 6.,
  switchers: tuple[Scalar, Scalar, Scalar, Scalar] = (0.6, 1e-2, 1e-10, 1e-4),
  negflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> FlashResult:
  r"""Newton's method for multiphase flash calculations using a PT-based
  equation of state. A switch to the successive substitution iteration
  is implemented if Newton's method does not decrease the norm of the
  gradient of Gibbs energy function. Quasi-Newton Successive
  Substitution iterations precede Newton's method to improve initial
  guesses of k-values.

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
    phase mole fractions. Each array's shape should be `(Np - 1,)`,
    where `Np` is the number of phases.

  kvji0: Iterable[Matrix]
    An iterable object containing 2d-arrays of initial guesses of.
    k-values. Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z_lnphii(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: Scalar, T: Scalar, yji: Matrix)
       -> tuple[IVector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole compositions
      of `Np` phases (`Matrix` of shape `(Np, Nc)`), this method must
      return a tuple of:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`.

    - `getPT_Z_lnphii_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    - `getPT_Zj_lnphiji_dnk(P: Scalar, T: Scalar, yji: Matrix, nj: Vector)
       -> tuple[IVector, Vector, Matrix, Tensor]`
      For a given pressure `P` in [Pa], temperature `T` in [K], mole
      fractions of `Nc` components in `Np` phases `yji` of shape
      `(Np, Nc)` and mole numbers of phases in [mol], returns a tuple
      that contains:

      - a 1d-array of integers with the shape `(Np,)`, each of which
        corresponds to the designated phase (`0` = vapour,
        `1` = liquid, etc.),
      - a `Vector` of shape `(Np,)` of compressibility factors of
        phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a `Tensor`
        of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  lmbdmax: Scalar
    The maximum step length. Default is `6.0`.

  switchers: tuple[Scalar, Scalar, Scalar, Scalar]
    Allows to modify the conditions of switching from the QNSS to
    Newton's method. The parameter must be represented as a tuple
    containing four values: :math:`\eps_r`, :math:`\eps_f`,
    :math:`\eps_l`, :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \max_j \left| F_j^k - F_j^{k-1} \right| < \eps_f, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
        0 < F_1^k < 1,
      \end{cases}

    where :math:`\mathbf{g}` is the equilibrium equations vector,
    :math:`k` is the iteration number, :math:`F_j` is the mole fraction
    of a non-reference phase :math:`j`. Analytical expressions of the
    switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.6, 1e-2, 1e-10, 1e-4)`.

  negflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape
    `((Np - 1) * Nc, (Np - 1) * Nc)` and
    a vector `b` of shape `((Np - 1) * Nc,)` and finds a vector `x` of
    shape `((Np - 1) * Nc,)`, which is the solution of the system of
    linear equations `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `Fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Multiphase flash calculation (SS-Newton method).")
  Nc = eos.Nc
  Npm1 = kvsji0[0].shape[0]
  Npm1Nc = Npm1 * Nc
  logger.info('P = %.1f Pa, T = %.2f K, yi =' + Nc * '%7.4f', P, T, *yi)
  logger.debug(
    '%3s%5s' + Npm1Nc * '%10s' + Npm1 * '%9s' + '%11s%8s', 'Nkv', 'Nit',
    *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['F%s' % j for j in range(Npm1)], 'g2', 'method',
  )
  tmpl = '%3s%5s' + Npm1Nc * '%10.4f' + Npm1 * '%9.4f' + '%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  for s, (fj, kvji) in enumerate(zip(fsj0, kvsji0)):
    k = 0
    lnkvji = np.log(kvji)
    lnkvi = lnkvji.ravel()
    fj = solveNp(kvji, yi, fj)
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2 = gi.dot(gi)
    switch = g2 < tol
    lmbd = 1.
    logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'QNSS')
    while not switch and g2 > tol and k < maxiter:
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvi += dlnkvi
      kvji = np.exp(lnkvji)
      fjkm1 = fj
      fj = solveNp(kvji, yi, fj)
      xi = yi / (fj.dot(kvji - 1.) + 1.)
      yji = kvji * xi
      sj, Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      sx, Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2km1 = g2
      g2 = gi.dot(gi)
      physfj = (fj > 0.).all() and (fj < 1.).all() or negflash
      switch = (g2 / g2km1 > epsr and np.abs(fj - fjkm1).max() < epsf and
                g2 > epsl and g2 < epsu and physfj)
      logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'QNSS')
      if g2 < tol or switch:
        break
      if k % Nc == 0 or g2 > g2km1:
        lmbd = 1.
      else:
        lmbd *= tkm1 / (dlnkvi.dot(gi) - tkm1)
        if lmbd < 0.:
          lmbd = -lmbd
        if lmbd > lmbdmax:
          lmbd = lmbdmax
    if np.isfinite(g2):
      if g2 < tol:
        if physfj:
          yji = np.append(yji, np.atleast_2d(xi), 0)
          fj = np.append(fj, 1. - fj.sum())
          Zj = np.append(Zj, Zx)
          sj = np.append(sj, sx)
          idx = np.argsort(yji.dot(eos.mwi) / Zj)
          yji = yji[idx]
          kvji = yji[:-1] / yji[-1]
          logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
          return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
      elif k < maxiter:
        H = np.empty(shape=(Npm1Nc, Npm1Nc))
        H_block = np.lib.stride_tricks.as_strided(
          H, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
        )
        H_blockdiag = np.lib.stride_tricks.as_strided(
          H, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
        )
        U = np.empty(shape=(Npm1Nc, Npm1Nc))
        U_block = np.lib.stride_tricks.as_strided(
          U, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
        )
        U_blockdiag = np.lib.stride_tricks.as_strided(
          U, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
        )
        Ux = np.full(shape=(Nc, Nc), fill_value=-1.)
        Ux_diag = np.lib.stride_tricks.as_strided(Ux, (Nc,), (8 * (Nc + 1),))
        Uy = np.full(shape=(Npm1, Nc, Nc), fill_value=-1.)
        Uy_diag = np.lib.stride_tricks.as_strided(
          Uy, (Npm1, Nc), (8 * Nc * Nc, 8 * (Nc + 1)),
        )
        fx = 1. - fj.sum()
        sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
        sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
        logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'Newt')
        while g2 > tol and k < maxiter:
          H_block[:] = dlnphixidnk
          H_blockdiag += dlnphijidnk
          Ux_diag[:] = 1. / xi - 1.
          Uy_diag[:] = 1. / yji - 1.
          U_block[:] = Ux / fx
          U_blockdiag += Uy / fj[:, None, None]
          H += U
          dni = np.linalg.solve(H, -gi)
          dlnkvi = U.dot(dni)
          lnkvi += dlnkvi
          k += 1
          kvji = np.exp(lnkvji)
          fj = solveNp(kvji, yi, fj)
          fx = 1. - fj.sum()
          xi = yi / (fj.dot(kvji - 1.) + 1.)
          yji = kvji * xi
          sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T,
                                                                  yji, fj)
          sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
          gji = lnkvji + lnphiji - lnphixi
          gi = gji.ravel()
          g2km1 = g2
          g2 = gi.dot(gi)
          if g2 < g2km1 or forcenewton:
            logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'Newt')
          else:
            lnkvi -= gi
            kvji = np.exp(lnkvji)
            fj = solveNp(kvji, yi, fj)
            fx = 1. - fj.sum()
            xi = yi / (fj.dot(kvji - 1.) + 1.)
            yji = kvji * xi
            sj, Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T,
                                                                    yji, fj)
            sx, Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T,
                                                                  xi, fx)
            gji = lnkvji + lnphiji - lnphixi
            gi = gji.ravel()
            g2 = gi.dot(gi)
            logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
        if (g2 < tol and np.isfinite(g2) and
            ((fj < 1.).all() and (fj > 0.).all() or negflash)):
          yji = np.append(yji, np.atleast_2d(xi), 0)
          fj = np.append(fj, 1. - fj.sum())
          Zj = np.append(Zj, Zx)
          sj = np.append(sj, sx)
          idx = np.argsort(yji.dot(eos.mwi) / Zj)
          yji = yji[idx]
          kvji = yji[:-1] / yji[-1]
          logger.info('Phase mole fractions:' + (Npm1 + 1) * '%7.4f' % (*fj,))
          return FlashResult(yji, fj[idx], Zj[idx], sj[idx], kvji)
  logger.warning(
    "Multiphase flash calculation terminates unsuccessfully.\n"
    "The solution method was QNSS-Newton, EOS: %s.\nParameters:\nP = %s Pa"
    "\nT = %s K\nyi = %s",
    eos.name, P, T, yi,
  )
  raise SolutionNotFoundError(
    'The flash calculation procedure\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations. It also may be '
    'advisable to improve the initial\nguesses of k-values.'
  )
