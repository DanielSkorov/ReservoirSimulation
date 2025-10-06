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

from constants import (
  R,
)

from typing import (
  Protocol,
  Sequence,
  Callable,
)

from customtypes import (
  Integer,
  Double,
  Vector,
  Matrix,
  Tensor,
  Linsolver,
  SolutionNotFoundError,
)

from eos import (
  Eos,
)


logger = logging.getLogger('flash')


class EosFlash2pPT(Eos, Protocol):

  def getPT_kvguess(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> Sequence[Vector[Double]]: ...

  def getPT_PID(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> int: ...

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


class EosFlashNpPT(EosFlash2pPT, Protocol):

  def getPT_Z(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> float: ...

  def getPT_PIDj(
    self,
    P: float,
    T: float,
    yji: Matrix[Double],
  ) -> Vector[Integer]: ...

  def getPT_Zj_lnphiji(
    self,
    P: float,
    T: float,
    yji: Matrix[Double],
  ) -> tuple[Vector[Double], Matrix[Double]]: ...

  def getPT_Zj_lnphiji_dnk(
    self,
    P: float,
    T: float,
    yji: Matrix[Double],
    nj: Vector[Double],
  ) -> tuple[Vector[Double], Matrix[Double], Tensor[Double]]: ...


@dataclass(eq=False, slots=True)
class FlashResult(object):
  """Container for flash calculation outputs with pretty-printing.

  Attributes
  ----------
  Np: int
    The number of phases.

  P: float
    Pressure [Pa] of the equilibrium state.

  T: float
    Temperature [K] of the equilibrium state.

  V: float
    Volume [m3] of a mixture at the equilibrium state.

  n: float
    Number of moles [mol] of a mixture.

  ni: Vector[Double], shape (Nc,)
    Number of moles [mol] of `Nc` components.

  nj: Vector[Double]
    Phase mole numbers as a `Vector[Double]` of shape `(Np,)`, where
    `Np` is the number of phases.

  fj: Vector, shape (Np,)
    Phase mole fractions as a `Vector[Double]` of shape `(Np,)`, where
    `Np` is the number of phases.

  yji: Matrix[Double], shape (Np, Nc)
    Mole fractions of components in each phase as a `Matrix[Double]`
    of shape `(Np, Nc)`, where `Np` is the number of phases and `Nc`
    is the number of components.

  Zj: Vector, shape (Np,)
    Phase compressibility factors as a `Vector` of shape `(Np,)`,
    where `Np` is the number of phases.

  vj: Vector[Double], shape (Np,)
    Molar volumes [m3/mol] of phases as a `Vector[Double]` of shape
    `(Np,)`, where `Np` is the number of phases.

  sj: Vector[Integer], shape (Np,)
    The designated phases (states) as a `Vector[Integer]` of the shape
    `(Np,)`, where `Np` is the number of phases. (`0` = vapour,
    `1` = liquid, etc.).

  kvji: Matrix[Double], shape (Np - 1, Nc) | None
    K-values of components in non-reference phases as a `Matrix[Double]`
    of shape `(Np - 1, Nc)`, where `Np` is the number of phases and `Nc`
    is the number of components. For the one phase state, this attribute
    is `None`.

  Methods
  -------
  __str__(self) -> str
    Returns the string representation of this class.

  __repr__(self) -> str
    Returns the table representation of this class.
  """
  Np: int
  P: float
  T: float
  V: float
  n: float
  ni: Vector[Double]
  nj: Vector[Double]
  fj: Vector[Double]
  yji: Matrix[Double]
  Zj: Vector[Double]
  vj: Vector[Double]
  sj: Vector[Integer]
  kvji: Matrix[Double] | None = None

  def __str__(self) -> str:
    with np.printoptions(linewidth=1024):
      return (f'Pressure: {self.P / 1e6:.3f} [MPa]\n'
              f'Temperature: {self.T - 273.15:.2f} [°C]\n'
              f'Mole number of the system: {self.n:.4f} [mol]\n'
              f'Phase IDs: {self.sj}\n'
              f'Phase mole fractions: {self.fj}\n'
              f'Phase compressibility factors: {self.Zj}')

  def __repr__(self) -> str:
    return (f'{self.P/1e6:8.3f}{self.T-273.15:8.2f}{self.n:9.4f}'
            + ''.join(map(lambda t: '%4s%9.4f%9.4f' % t,
                          zip(self.sj, self.fj, self.Zj))))

  def __format__(self, fmt: str) -> str:
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')


class flash2pPT(object):
  """Two-phase flash calculations.

  Performs two-phase flash calculations for a given pressure [Pa],
  temperature [K] and composition of the mixture.

  Parameters
  ----------
  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: float, T: float, yi: Vector[Double])
       -> Sequence[Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must generate
      initial guesses of k-values as a sequence of `Vector[Double]` of
      shape `(Nc,)`.

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

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

    - `mwi: Vector[Double]`
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
    runstab: bool = True,
    useprev: bool = False,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.runstab = runstab
    self.useprev = useprev
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    self.prevkvji: None | Matrix[Double] = None
    self.solver: Callable[[float, float, Vector[Double], float,
                           Sequence[Vector[Double]]], FlashResult]
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
    P: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
    kvji0: Sequence[Vector[Double]] | None = None,
  ) -> FlashResult:
    """Performs flash calculations for given pressure, temperature and
    composition.

    Parameters
    ----------
    P: float
      Pressure [Pa] of the mixture.

    T: float
      Temperature [K] of the mixture.

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Number of moles [mol] of the mixture. Default is `1.0` [mol].

    kvji0: Sequence[Vector[Double]] | None
      A sequence containing 1d-arrays of initial k-value guesses. Each
      array's shape should be `(Nc,)`. Default is `None` which means
      to use initial guesses from the method `getPT_kvguess` of the
      initialized instance of an EOS.

    Returns
    -------
    Flash calculation results as an instance of `FlashResult` object.
    Important attributes are:
    - `yji` component mole fractions in each phase,
    - `fj` phase mole fractions,
    - `Zj` the compressibility factor of each phase.

    Raises
    ------
    `SolutionNotFoundError` if the flash calculation procedure
    terminates unsuccessfully.
    """
    if kvji0 is None:
      kvji = self.eos.getPT_kvguess(P, T, yi)
    else:
      kvji = kvji0
    if self.useprev and self.prevkvji is not None:
      kvji = self.prevkvji[0], *kvji
    if self.runstab:
      stab = self.stabsolver.run(P, T, yi, kvji)
      if stab.stable:
        Z = stab.Z
        v = Z * R * T / P
        V = n * v
        s = self.eos.getPT_PID(P, T, yi)
        return FlashResult(1, P, T, V, n, n * yi,
                           np.array([n]), np.array([1.]), np.atleast_2d(yi),
                           np.array([Z]), np.array([v]), np.array([s]))
      assert stab.kvji is not None
      if self.useprev and self.prevkvji is not None:
        kvji = self.prevkvji[0], *stab.kvji
      else:
        kvji = stab.kvji
    flash = self.solver(P, T, yi, n, kvji)
    if self.useprev:
      self.prevkvji = flash.kvji
    return flash


def _flash2pPT_ss(
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  kvji0: Sequence[Vector[Double]],
  eos: EosFlash2pPT,
  tol: float = 1e-16,
  maxiter: int = 100,
) -> FlashResult:
  """Successive substitution method for two-phase flash calculations
  using a PT-based equation of state.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
    Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
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
      Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2 = gi.dot(gi)
      logger.debug(tmpl, j, k, *lnkvi, f0, g2)
    if g2 < tol and np.isfinite(g2):
      s0 = eos.getPT_PID(P, T, y0i)
      s1 = eos.getPT_PID(P, T, y1i)
      if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
        yji = np.vstack([y0i, y1i])
        fj = np.array([f0, 1. - f0])
        Zj = np.array([Z0, Z1])
        sj = np.array([s0, s1])
        kvji = np.atleast_2d(kvi)
      else:
        yji = np.vstack([y1i, y0i])
        fj = np.array([1. - f0, f0])
        Zj = np.array([Z1, Z0])
        sj = np.array([s1, s0])
        kvji = np.atleast_2d(1. / kvi)
      logger.info('Phase mole fractions: %.4f, %.4f', *fj)
      vj = Zj * (R * T / P)
      nj = n * fj
      V = nj.dot(vj)
      return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj, kvji)
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
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  kvji0: Sequence[Vector[Double]],
  eos: EosFlash2pPT,
  tol: float = 1e-16,
  maxiter: int = 50,
  lmbdmax: float = 6.,
) -> FlashResult:
  """QNSS-method for two-phase flash calculations using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `50`.

  lmbdmax: float
    The maximum step length. Default is `6.0`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
    Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
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
      Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
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
    if g2 < tol and np.isfinite(g2):
      s0 = eos.getPT_PID(P, T, y0i)
      s1 = eos.getPT_PID(P, T, y1i)
      if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
        yji = np.vstack([y0i, y1i])
        fj = np.array([f0, 1. - f0])
        Zj = np.array([Z0, Z1])
        sj = np.array([s0, s1])
        kvji = np.atleast_2d(kvi)
      else:
        yji = np.vstack([y1i, y0i])
        fj = np.array([1. - f0, f0])
        Zj = np.array([Z1, Z0])
        sj = np.array([s1, s0])
        kvji = np.atleast_2d(1. / kvi)
      logger.info('Phase mole fractions: %.4f, %.4f', *fj)
      vj = Zj * (R * T / P)
      nj = n * fj
      V = nj.dot(vj)
      return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj, kvji)
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
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  kvji0: Sequence[Vector[Double]],
  eos: EosFlash2pPT,
  tol: float = 1e-16,
  maxiter: int = 30,
  forcenewton: bool = False,
  linsolver: Linsolver = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

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

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate Newton's method successfully if the sum of squared
    elements of the gradient of Gibbs energy function is less than
    `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Linsolver
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
      f0 = np.float64(1e-8)
    elif np.isclose(f0, 1.):
      f0 = np.float64(0.99999999)
    f1 = 1. - f0
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
    Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
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
      Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
      Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
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
        Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
        Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
        gi = lnkvi + lnphi0i - lnphi1i
        g2 = gi.dot(gi)
        logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
    if g2 < tol and np.isfinite(g2):
      s0 = eos.getPT_PID(P, T, y0i)
      s1 = eos.getPT_PID(P, T, y1i)
      if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
        yji = np.vstack([y0i, y1i])
        fj = np.array([f0, f1])
        Zj = np.array([Z0, Z1])
        sj = np.array([s0, s1])
        kvji = np.atleast_2d(kvi)
      else:
        yji = np.vstack([y1i, y0i])
        fj = np.array([f1, f0])
        Zj = np.array([Z1, Z0])
        sj = np.array([s1, s0])
        kvji = np.atleast_2d(1. / kvi)
      logger.info('Phase mole fractions: %.4f, %.4f', *fj)
      vj = Zj * (R * T / P)
      nj = n * fj
      V = nj.dot(vj)
      return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj, kvji)
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
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  kvji0: Sequence[Vector[Double]],
  eos: EosFlash2pPT,
  tol: float = 1e-16,
  maxiter: int = 30,
  forcenewton: bool = False,
  switchers: tuple[float, float, float, float] = (0.1, 1e-2, 1e-10, 1e-4),
  linsolver: Linsolver = np.linalg.solve,
) -> FlashResult:
  r"""Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient. Successive substitution iterations
  precede Newton's method to improve initial guesses of k-values.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

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

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  switchers: tuple[float, float, float, float]
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

  linsolver: Linsolver
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
    Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
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
      Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = (g2 / g2km1 > epsr and
                (f0 - f0km1 < epsf and f0km1 - f0 < epsf) and
                g2 > epsl and g2 < epsu and
                (f0 > 0. and f0 < 1.))
      logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
    if np.isfinite(g2):
      if g2 < tol:
        s0 = eos.getPT_PID(P, T, y0i)
        s1 = eos.getPT_PID(P, T, y1i)
        if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
          yji = np.vstack([y0i, y1i])
          fj = np.array([f0, 1. - f0])
          Zj = np.array([Z0, Z1])
          sj = np.array([s0, s1])
          kvji = np.atleast_2d(kvi)
        else:
          yji = np.vstack([y1i, y0i])
          fj = np.array([1. - f0, f0])
          Zj = np.array([Z1, Z0])
          sj = np.array([s1, s0])
          kvji = np.atleast_2d(1. / kvi)
        logger.info('Phase mole fractions: %.4f, %.4f', *fj)
        vj = Zj * (R * T / P)
        nj = n * fj
        V = nj.dot(vj)
        return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                           kvji)
      elif k < maxiter:
        U = np.full(shape=(Nc, Nc), fill_value=-1.)
        f1 = 1. - f0
        Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
        Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
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
          Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
          Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
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
            Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P,T, y1i, f1)
            Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P,T, y0i, f0)
            gi = lnkvi + lnphi0i - lnphi1i
            g2 = gi.dot(gi)
            logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
        if g2 < tol and np.isfinite(g2):
          s0 = eos.getPT_PID(P, T, y0i)
          s1 = eos.getPT_PID(P, T, y1i)
          if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
            yji = np.vstack([y0i, y1i])
            fj = np.array([f0, f1])
            Zj = np.array([Z0, Z1])
            sj = np.array([s0, s1])
            kvji = np.atleast_2d(kvi)
          else:
            yji = np.vstack([y1i, y0i])
            fj = np.array([f1, f0])
            Zj = np.array([Z1, Z0])
            sj = np.array([s1, s0])
            kvji = np.atleast_2d(1. / kvi)
          logger.info('Phase mole fractions: %.4f, %.4f', *fj)
          vj = Zj * (R * T / P)
          nj = n * fj
          V = nj.dot(vj)
          return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                             kvji)
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
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  kvji0: Sequence[Vector[Double]],
  eos: EosFlash2pPT,
  tol: float = 1e-16,
  maxiter: int = 30,
  lmbdmax: float = 6.,
  forcenewton: bool = False,
  switchers: tuple[float, float, float, float] = (0.1, 1e-2, 1e-10, 1e-4),
  linsolver: Linsolver = np.linalg.solve,
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
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  kvji0: Sequence[Vector[Double]]
    A sequence containing arrays of initial k-value guesses.
    Each array's shape should be `(Nc,)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

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

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  lmbdmax: float
    The maximum step length. Default is `6.0`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  switchers: tuple[float, float, float, float]
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

  linsolver: Linsolver
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
    Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
    Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
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
      Z1, lnphi1i = eos.getPT_Z_lnphii(P, T, y1i)
      Z0, lnphi0i = eos.getPT_Z_lnphii(P, T, y0i)
      gi = lnkvi + lnphi0i - lnphi1i
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = (g2 / g2km1 > epsr and
                (f0 - f0km1 < epsf and f0km1 - f0 < epsf) and
                g2 > epsl and g2 < epsu and
                (f0 > 0. and f0 < 1.))
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
        s0 = eos.getPT_PID(P, T, y0i)
        s1 = eos.getPT_PID(P, T, y1i)
        if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
          yji = np.vstack([y0i, y1i])
          fj = np.array([f0, 1. - f0])
          Zj = np.array([Z0, Z1])
          sj = np.array([s0, s1])
          kvji = np.atleast_2d(kvi)
        else:
          yji = np.vstack([y1i, y0i])
          fj = np.array([1. - f0, f0])
          Zj = np.array([Z1, Z0])
          sj = np.array([s1, s0])
          kvji = np.atleast_2d(1. / kvi)
        logger.info('Phase mole fractions: %.4f, %.4f', *fj)
        vj = Zj * (R * T / P)
        nj = n * fj
        V = nj.dot(vj)
        return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                           kvji)
      elif k < maxiter:
        U = np.full(shape=(Nc, Nc), fill_value=-1.)
        f1 = 1. - f0
        Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
        Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
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
          Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P, T, y1i, f1)
          Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P, T, y0i, f0)
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
            Z1, lnphi1i, dlnphi1idnj = eos.getPT_Z_lnphii_dnj(P,T, y1i, f1)
            Z0, lnphi0i, dlnphi0idnj = eos.getPT_Z_lnphii_dnj(P,T, y0i, f0)
            gi = lnkvi + lnphi0i - lnphi1i
            g2 = gi.dot(gi)
            logger.debug(tmpl, j, k, *lnkvi, f0, g2, 'SS')
        if g2 < tol and np.isfinite(g2):
          s0 = eos.getPT_PID(P, T, y0i)
          s1 = eos.getPT_PID(P, T, y1i)
          if y0i.dot(eos.mwi) / Z0 < y1i.dot(eos.mwi) / Z1:
            yji = np.vstack([y0i, y1i])
            fj = np.array([f0, f1])
            Zj = np.array([Z0, Z1])
            sj = np.array([s0, s1])
            kvji = np.atleast_2d(kvi)
          else:
            yji = np.vstack([y1i, y0i])
            fj = np.array([f1, f0])
            Zj = np.array([Z1, Z0])
            sj = np.array([s1, s0])
            kvji = np.atleast_2d(1. / kvi)
          logger.info('Phase mole fractions: %.4f, %.4f', *fj)
          vj = Zj * (R * T / P)
          nj = n * fj
          V = nj.dot(vj)
          return FlashResult(2, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                             kvji)
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


class flashNpPT(object):
  """Multiphase flash calculations.

  Performs multiphase flash calculations for a given pressure [Pa],
  temperature [K] and composition of the mixture.

  Parameters
  ----------
  eos: EosFlashNpPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: float, T: float, yi: Vector[Double])
       -> Sequence[Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must generate
      initial guesses of k-values as a sequence of `Vector[Double]` of
      shape `(Nc,)`.

    - `getPT_PID(P: float, T: float, yi: Vector[Double]) -> int`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return
      the phase identification number (`0` = vapour, `1` = liquid, etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    To perform multiphase flash calculations, this instance of an
    equation of state class must have:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z(P: float, T: float, yi: Vector[Double]) -> float`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return its
      compressibility factor.

    - `getPT_Zj_lnphiji(P: float, T: float, yji: Matrix[Double])
       -> tuple[Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return a tuple of:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`.

    If the solution method would be one of `'newton'`, `'ss-newton'`
    or `'qnss-newton'` then it also must have:

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

    - `getPT_Zj_lnphiji_dnk(P: float, T: float, yji: Matrix[Double],
       nj: Vector[Double]) -> tuple[Vector[Double], Matrix[Double],
       Tensor[Double]]`
      For a given pressure [Pa], temperature [K], mole fractions of `Nc`
      components in `Np` phases as a `Matrix[Double]` of shape
      `(Np, Nc)`, and mole numbers of phases as a `Vector[Double]`,
      this method must return a tuple that contains:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a
        `Tensor[Double]` of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
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

  maxNp: int
      The maximum number of phases. Default is `3`.

  stabkwargs: dict
    Dictionary that used to regulate the stability test procedure.
    Default is an empty dictionary.

  flash2pkwargs: dict
    Dictionary that used to regulate the two-phase flash calculation
    procedure. Default is an empty dictionary.

  **kwargs: dict
    Other arguments for a flash calculation solver. It may contain such
    arguments as `tol`, `maxiter` and others, depending on the selected
    flash calculation solver. They also will be passed to a two-phase
    flash calculation solver.
  """
  def __init__(
    self,
    eos: EosFlashNpPT,
    method: str = 'qnss-newton',
    maxNp: int = 3,
    flash2pkwargs: dict = {},
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.maxNp = maxNp
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    self.solver2p = flash2pPT(eos, method=method, stabkwargs=stabkwargs,
                              **flash2pkwargs, **kwargs)
    self.solver: Callable[[float, float, Vector[Double], float,
                           Sequence[Vector[Double]],
                           Sequence[Matrix[Double]]], FlashResult]
    if method == 'ss':
      self.solver = partial(_flashNpPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_flashNpPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_flashNpPT_newt, eos=eos, **kwargs)
    elif method == 'ss-newton':
      self.solver = partial(_flashNpPT_ssnewt, eos=eos, **kwargs)
    elif method == 'qnss-newton':
      self.solver = partial(_flashNpPT_qnssnewt, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown flash-method: {method}.')
    pass

  def run(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
    kvji0: Sequence[Vector[Double]] | None = None,
  ) -> FlashResult:
    """Performs multiphase flash calculations for given pressure,
    temperature and composition.

    Parameters
    ----------
    P: float
      Pressure [Pa] of the mixture.

    T: float
      Temperature [K] of the mixture.

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Number of moles [mol] of the mixture. Default is `1.0` [mol].

    kvji0: Sequence[Vector[Double]] | None
      A sequence containing 1d-arrays of initial k-value guesses for
      two-phase flash calculations. Each array's shape should be
      `(Nc,)`. Default is `None` which means to use initial guesses
      from the method `getPT_kvguess` of the initialized instance of
      an EOS.

    Returns
    -------
    Flash calculation results as an instance of `FlashResult` object.
    Important attributes are:
    - `yji` component mole fractions in each phase,
    - `fj` phase mole fractions,
    - `Zj` the compressibility factor of each phase.

    Raises
    ------
    `SolutionNotFoundError` if the flash calculation procedure
    terminates unsuccessfully.
    """
    if self.maxNp < 2:
      Z = self.eos.getPT_Z(P, T, yi)
      v = Z * R * T / P
      V = n * v
      s = self.eos.getPT_PID(P, T, yi)
      return FlashResult(1, P, T, V, n, n * yi,
                         np.array([n]), np.array([1.]), np.atleast_2d(yi),
                         np.array([Z]), np.array([v]), np.array([s]))
    res = self.solver2p.run(P, T, yi, n, kvji0)
    if res.Np == 1:
      return res
    for k in range(3, self.maxNp + 1):
      kvsji, fsj = self.initialize(P, T, res)
      if kvsji:
        logger.debug('Running %sp-flash...', k)
        res = self.solver(P, T, yi, n, fsj, kvsji)
      else:
        return res
    return res

  def initialize(
    self,
    P: float,
    T: float,
    flash: FlashResult,
  ) -> tuple[list[Matrix[Double]], list[Vector[Double]]]:
    """Generates initial guesses of k-values and phase mole fractions
    for `Np`-phase (`Np >= 3`) flash calculation.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
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
    fj = flash.fj
    kvsji = []
    fsj = []
    for r in range(flash.Np):
      xi = yji[r]
      stab = self.stabsolver.run(P, T, xi)
      if stab.kvji is not None:
        kvji = yji / xi
        kvji[r] = stab.kvji[0]
        kvsji.append(kvji)
        fj = fj.copy()
        fj[r] = 0.
        fsj.append(fj)
    return kvsji, fsj


def _flashNpPT_ss(
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  fsj0: Sequence[Vector[Double]],
  kvsji0: Sequence[Matrix[Double]],
  eos: EosFlashNpPT,
  tol: float = 1e-16,
  maxiter: int = 100,
) -> FlashResult:
  """Successive substitution method for multiphase flash calculations
  using a PT-based equation of state.

  Parameters
  ----------
  P: float
    Pressure of the mixture [Pa].

  T: float
    Temperature of the mixture [K].

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  fsj0: Sequence[Vector[Double]]
    A sequence containing 1d-arrays of initial guesses of phase mole
    fractions. Each array's shape should be `(Np - 1,)`, where `Np`
    is the number of phases.

  kvji0: Sequence[Matrix[Double]]
    A sequence containing 2d-arrays of initial guesses of k-values.
    Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: float, T: float, yji: Matrix[Double])
       -> tuple[Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return a tuple of:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
    Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
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
      Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2 = gi.dot(gi)
      logger.debug(tmpl, s, k, *lnkvi, *fj, g2)
    if g2 < tol and np.isfinite(g2):
      yji = np.append(yji, np.atleast_2d(xi), 0)
      fj = np.append(fj, 1. - fj.sum())
      Zj = np.append(Zj, Zx)
      idx = np.argsort(yji.dot(eos.mwi) / Zj)
      yji = yji[idx]
      fj = fj[idx]
      Zj = Zj[idx]
      vj = Zj * (R * T / P)
      nj = n * fj
      V = nj.dot(vj)
      kvji = yji[:-1] / yji[-1]
      sj = eos.getPT_PIDj(P, T, yji)
      Np = Npm1 + 1
      logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
      return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                         kvji)
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


def _flashNpPT_qnss(
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  fsj0: Sequence[Vector[Double]],
  kvsji0: Sequence[Matrix[Double]],
  eos: EosFlashNpPT,
  tol: float = 1e-16,
  maxiter: int = 50,
  lmbdmax: float = 6.,
) -> FlashResult:
  """QNSS-method for multiphase flash calculations using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  fsj0: Sequence[Vector[Double]]
    A sequence containing 1d-arrays of initial guesses of phase mole
    fractions. Each array's shape should be `(Np - 1,)`, where `Np`
    is the number of phases.

  kvji0: Sequence[Matrix[Double]]
    A sequence containing 2d-arrays of initial guesses of k-values.
    Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: float, T: float, yji: Matrix[Double])
       -> tuple[Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return a tuple of:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `50`.

  lmbdmax: float
    The maximum step length. Default is `6.0`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:
  - `yji` component mole fractions in each phase,
  - `fj` phase mole fractions,
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
    Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
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
      Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
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
    if g2 < tol and np.isfinite(g2):
      yji = np.append(yji, np.atleast_2d(xi), 0)
      fj = np.append(fj, 1. - fj.sum())
      Zj = np.append(Zj, Zx)
      idx = np.argsort(yji.dot(eos.mwi) / Zj)
      yji = yji[idx]
      fj = fj[idx]
      Zj = Zj[idx]
      vj = Zj * (R * T / P)
      nj = n * fj
      V = nj.dot(vj)
      kvji = yji[:-1] / yji[-1]
      sj = eos.getPT_PIDj(P, T, yji)
      Np = Npm1 + 1
      logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
      return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                         kvji)
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


def _flashNpPT_newt(
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  fsj0: Sequence[Vector[Double]],
  kvsji0: Sequence[Matrix[Double]],
  eos: EosFlashNpPT,
  tol: float = 1e-16,
  maxiter: int = 30,
  forcenewton: bool = False,
  linsolver: Linsolver = np.linalg.solve,
) -> FlashResult:
  """Newton's method for multiphase flash calculations using a PT-based
  equation of state. A switch to the successive substitution iteration
  is implemented if Newton's method does not decrease the norm of the
  gradient of Gibbs energy function.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  fsj0: Sequence[Vector[Double]]
    A sequence containing 1d-arrays of initial guesses of phase mole
    fractions. Each array's shape should be `(Np - 1,)`, where `Np`
    is the number of phases.

  kvji0: Sequence[Matrix[Double]]
    A sequence containing 2d-arrays of initial guesses of k-values.
    Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

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

    - `getPT_Zj_lnphiji_dnk(P: float, T: float, yji: Matrix[Double],
       nj: Vector[Double]) -> tuple[Vector[Double], Matrix[Double],
       Tensor[Double]]`
      For a given pressure [Pa], temperature [K], mole fractions of `Nc`
      components in `Np` phases as a `Matrix[Double]` of shape
      `(Np, Nc)`, and mole numbers of phases as a `Vector[Double]`,
      this method must return a tuple that contains:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a
        `Tensor[Double]` of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  linsolver: Linsolver
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
  - `fj` phase mole fractions,
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
    Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
    Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
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
      Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
      Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
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
        Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
        Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
        gji = lnkvji + lnphiji - lnphixi
        gi = gji.ravel()
        g2 = gi.dot(gi)
        logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
    if g2 < tol and np.isfinite(g2):
      yji = np.append(yji, np.atleast_2d(xi), 0)
      fj = np.append(fj, 1. - fj.sum())
      Zj = np.append(Zj, Zx)
      idx = np.argsort(yji.dot(eos.mwi) / Zj)
      yji = yji[idx]
      fj = fj[idx]
      Zj = Zj[idx]
      vj = Zj * (R * T / P)
      nj = n * fj
      V = nj.dot(vj)
      kvji = yji[:-1] / yji[-1]
      sj = eos.getPT_PIDj(P, T, yji)
      Np = Npm1 + 1
      logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
      return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                         kvji)
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


def _flashNpPT_ssnewt(
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  fsj0: Sequence[Vector[Double]],
  kvsji0: Sequence[Matrix[Double]],
  eos: EosFlashNpPT,
  tol: float = 1e-16,
  maxiter: int = 30,
  forcenewton: bool = False,
  switchers: tuple[float, float, float, float] = (0.6, 1e-2, 1e-10, 1e-4),
  linsolver: Linsolver = np.linalg.solve,
) -> FlashResult:
  r"""Newton's method for multiphase flash calculations using a PT-based
  equation of state. A switch to the successive substitution iteration
  is implemented if Newton's method does not decrease the norm of the
  gradient of Gibbs energy function. Successive substitution iterations
  precede Newton's method to improve initial guesses of k-values.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  fsj0: Sequence[Vector[Double]]
    A sequence containing 1d-arrays of initial guesses of phase mole
    fractions. Each array's shape should be `(Np - 1,)`, where `Np`
    is the number of phases.

  kvji0: Sequence[Matrix[Double]]
    A sequence containing 2d-arrays of initial guesses of k-values.
    Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: float, T: float, yji: Matrix[Double])
       -> tuple[Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return a tuple of:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`.

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

    - `getPT_Zj_lnphiji_dnk(P: float, T: float, yji: Matrix[Double],
       nj: Vector[Double]) -> tuple[Vector[Double], Matrix[Double],
       Tensor[Double]]`
      For a given pressure [Pa], temperature [K], mole fractions of `Nc`
      components in `Np` phases as a `Matrix[Double]` of shape
      `(Np, Nc)`, and mole numbers of phases as a `Vector[Double]`,
      this method must return a tuple that contains:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a
        `Tensor[Double]` of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  switchers: tuple[float, float, float, float]
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

  linsolver: Linsolver
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
  - `fj` phase mole fractions,
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
    Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
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
      Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = (g2 / g2km1 > epsr and np.abs(fj - fjkm1).max() < epsf and
                g2 > epsl and g2 < epsu and
                (fj > 0.).all() and (fj < 1.).all())
      logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
    if np.isfinite(g2):
      if g2 < tol:
        yji = np.append(yji, np.atleast_2d(xi), 0)
        fj = np.append(fj, 1. - fj.sum())
        Zj = np.append(Zj, Zx)
        idx = np.argsort(yji.dot(eos.mwi) / Zj)
        yji = yji[idx]
        fj = fj[idx]
        Zj = Zj[idx]
        vj = Zj * (R * T / P)
        nj = n * fj
        V = nj.dot(vj)
        kvji = yji[:-1] / yji[-1]
        sj = eos.getPT_PIDj(P, T, yji)
        Np = Npm1 + 1
        logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
        return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                           kvji)
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
        Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
        Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
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
          Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
          Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
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
            Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
            Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
            gji = lnkvji + lnphiji - lnphixi
            gi = gji.ravel()
            g2 = gi.dot(gi)
            logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
        if g2 < tol and np.isfinite(g2):
          yji = np.append(yji, np.atleast_2d(xi), 0)
          fj = np.append(fj, 1. - fj.sum())
          Zj = np.append(Zj, Zx)
          idx = np.argsort(yji.dot(eos.mwi) / Zj)
          yji = yji[idx]
          fj = fj[idx]
          Zj = Zj[idx]
          vj = Zj * (R * T / P)
          nj = n * fj
          V = nj.dot(vj)
          kvji = yji[:-1] / yji[-1]
          sj = eos.getPT_PIDj(P, T, yji)
          Np = Npm1 + 1
          logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
          return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                             kvji)
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


def _flashNpPT_qnssnewt(
  P: float,
  T: float,
  yi: Vector[Double],
  n: float,
  fsj0: Sequence[Vector[Double]],
  kvsji0: Sequence[Matrix[Double]],
  eos: EosFlashNpPT,
  tol: float = 1e-16,
  maxiter: int = 30,
  lmbdmax: float = 6.,
  forcenewton: bool = False,
  switchers: tuple[float, float, float, float] = (0.6, 1e-2, 1e-10, 1e-4),
  linsolver: Linsolver = np.linalg.solve,
) -> FlashResult:
  r"""Newton's method for multiphase flash calculations using a PT-based
  equation of state. A switch to the successive substitution iteration
  is implemented if Newton's method does not decrease the norm of the
  gradient of Gibbs energy function. Quasi-Newton Successive
  Substitution iterations precede Newton's method to improve initial
  guesses of k-values.

  Parameters
  ----------
  P: float
    Pressure [Pa] of the mixture.

  T: float
    Temperature [K] of the mixture.

  yi: Vector[Double], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    Number of moles [mol] of the mixture.

  fsj0: Sequence[Vector[Double]]
    A sequence containing 1d-arrays of initial guesses of phase mole
    fractions. Each array's shape should be `(Np - 1,)`, where `Np`
    is the number of phases.

  kvji0: Sequence[Matrix[Double]]
    A sequence containing 2d-arrays of initial guesses of k-values.
    Each array's shape should be `(Np - 1, Nc)`.

  eos: EosFlash2pPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z_lnphii(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple that contains:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a 
        `Vector[Double]` of shape `(Nc,)`.

    - `getPT_Zj_lnphiji(P: float, T: float, yji: Matrix[Double])
       -> tuple[Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return a tuple of:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`.

    - `getPT_Z_lnphii_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], mole composition
      (`Vector[Double]` of shape `(Nc,)`), and phase mole number [mol],
      this method must return a tuple of:

      - the mixture compressibility factor,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to components mole numbers as a `Matrix[Double]` of
        shape `(Nc, Nc)`.

    - `getPT_Zj_lnphiji_dnk(P: float, T: float, yji: Matrix[Double],
       nj: Vector[Double]) -> tuple[Vector[Double], Matrix[Double],
       Tensor[Double]]`
      For a given pressure [Pa], temperature [K], mole fractions of `Nc`
      components in `Np` phases as a `Matrix[Double]` of shape
      `(Np, Nc)`, and mole numbers of phases as a `Vector[Double]`,
      this method must return a tuple that contains:

      - a `Vector[Double]` of shape `(Np,)` of compressibility factors
        of phases,
      - logarithms of fugacity coefficients of components in each
        phase as a `Matrix[Double]` of shape `(Np, Nc)`,
      - partial derivatives of logarithms of fugacity coefficients with
        respect to component mole numbers for each phase as a
        `Tensor[Double]` of shape `(Np, Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the sum of squared elements of the
    vector of equilibrium equations is less than `tol`.
    Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  lmbdmax: float
    The maximum step length. Default is `6.0`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the switch from
    Newton's method to successive substitution iterations if Newton's
    method doesn't decrese the sum of squared elements of the vector of
    equilibrium equations. Default is `False`.

  switchers: tuple[float, float, float, float]
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

  linsolver: Linsolver
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
  - `fj` phase mole fractions,
  - `Zj` the compressibility factor of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the flash calculation procedure terminates
  unsuccessfully.
  """
  logger.info("Multiphase flash calculation (QNSS-Newton method).")
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
    Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
    Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
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
      Zj, lnphiji = eos.getPT_Zj_lnphiji(P, T, yji)
      Zx, lnphixi = eos.getPT_Z_lnphii(P, T, xi)
      gji = lnkvji + lnphiji - lnphixi
      gi = gji.ravel()
      g2km1 = g2
      g2 = gi.dot(gi)
      switch = (g2 / g2km1 > epsr and np.abs(fj - fjkm1).max() < epsf and
                g2 > epsl and g2 < epsu and
                (fj > 0.).all() and (fj < 1.).all())
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
        yji = np.append(yji, np.atleast_2d(xi), 0)
        fj = np.append(fj, 1. - fj.sum())
        Zj = np.append(Zj, Zx)
        idx = np.argsort(yji.dot(eos.mwi) / Zj)
        yji = yji[idx]
        fj = fj[idx]
        Zj = Zj[idx]
        vj = Zj * (R * T / P)
        nj = n * fj
        V = nj.dot(vj)
        kvji = yji[:-1] / yji[-1]
        sj = eos.getPT_PIDj(P, T, yji)
        Np = Npm1 + 1
        logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
        return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                           kvji)
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
        Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
        Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
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
          Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
          Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
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
            Zj, lnphiji, dlnphijidnk = eos.getPT_Zj_lnphiji_dnk(P, T, yji, fj)
            Zx, lnphixi, dlnphixidnk = eos.getPT_Z_lnphii_dnj(P, T, xi, fx)
            gji = lnkvji + lnphiji - lnphixi
            gi = gji.ravel()
            g2 = gi.dot(gi)
            logger.debug(tmpl, s, k, *lnkvi, *fj, g2, 'SS')
        if g2 < tol and np.isfinite(g2):
          yji = np.append(yji, np.atleast_2d(xi), 0)
          fj = np.append(fj, 1. - fj.sum())
          Zj = np.append(Zj, Zx)
          idx = np.argsort(yji.dot(eos.mwi) / Zj)
          yji = yji[idx]
          fj = fj[idx]
          Zj = Zj[idx]
          vj = Zj * (R * T / P)
          nj = n * fj
          V = nj.dot(vj)
          kvji = yji[:-1] / yji[-1]
          sj = eos.getPT_PIDj(P, T, yji)
          Np = Npm1 + 1
          logger.info('Phase mole fractions:' + Np * '%7.4f' % (*fj,))
          return FlashResult(Np, P, T, V, n, n * yi, nj, fj, yji, Zj, vj, sj,
                             kvji)
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
