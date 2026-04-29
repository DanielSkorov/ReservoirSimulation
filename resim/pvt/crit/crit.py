from logging import (
  getLogger,
)

from typing import (
  cast,
)

from math import (
  isfinite,
)

from numpy import (
  array as np_array,
  atleast_2d as np_atleast_2d,
  linspace as np_linspace,
)

from resim.pvt.datatypes import (
  Float,
  State,
  OnePhaseState,
  Vector,
)

from resim.pvt.constants import (
  R,
)

from resim.pvt.eos import (
  State2pVTEos,
)

from resim.pvt.utils import (
  EigenSolver,
  rqi,
)

from resim.pvt.crit.protocols import (
  CritSolver,
  CritSolverVTEos,
  CritVTEos,
  SpinSolver,
  SpinVTEos,
)


logger = getLogger('crit')


class SpinodalConvergenceError(Exception):
  """An exception that will be raised if a spinodal temperature solver
  does not converge.
  """
  def __init__(
    self,
    msg: str = ('A spinodal temperature solution procedure does not '
                'converge. Try to increase the number of iterations '
                'or improve the initial guess.'),
  ) -> None:
    super().__init__(msg)
    pass


class CritConvergenceError(Exception):
  """An exception that will be raised if a critical point solver does
  not converge.
  """
  def _init__(
    self,
    msg: str = ('A critical point solution procedure does not converge. '
                'Try to increase the number of iterations or improve '
                'the initial guess.'),
  ) -> None:
    super().__init__(msg)
    pass


def _spinVT_newt(
  eos: SpinVTEos,
  v: float,
  yi: Vector[Float],
  T0: float,
  zetai0: Vector[Float],
  mdT: float = 1e-5,
  tol: float = 1e-10,
  maxiter: int = 10,
  eigensolver: EigenSolver = rqi,
) -> tuple[float, float, Vector[Float]]:
  """Calculate the spinodal temperature of a mixture for a given volume
  and composition.

  Parameters
  ----------
  eos: SpinVTEos
    An initialized instance of a VT-based equation of state.

  v: float
    Molar volume [m³/mol].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  T0: float
    An initial guess for the spinodal temperature [K].

  zetai0: Vector[Float], shape (Nc,)
    An eigenvector initial guess corresponding to the minimum eigenvalue
    of a matrix of second partial derivatives of the Helmholtz energy
    function with respect to mole numbers of components.

  mdT: float
    A multiplier used to compute the temperature shift to estimate
    the partial derivative of the lowest eigenvalue with respect
    to temperature at the zeroth iteration. Default is `1e-5`.

  tol: float
    Terminate successfully if the absolute value of the lowest
    eigenvalue is less then `tol`. Default is `1e-10`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  eigensolver: EigenSolver
    A callable object that for a given `Q: Matrix[Float]` of shape
    `(Nc, Nc)` finds a pair of the eigenvector of shape `(Nc,)` and
    the corresponding eigenvalue. Default is the implementation of
    the Rayleigh quotient iteration `rqi`.

  Returns
  -------
  A tuple containing:
  - spinodal temperature [K],
  - eigenvalue,
  - eigenvector as a `Vector[Float]` of shape `(Nc,)`.

  Raises
  ------
  SpinodalConvergenceError
    This exceptioon is raised if the solver does not converge.
  """
  k = 0
  T = T0
  _, Q = eos.getVT_lnfi_dnj(v, T, yi, 1.)
  zetai, lmbd = eigensolver(Q, zetai0, None)
  dT = mdT * T
  _, Qkp1 = eos.getVT_lnfi_dnj(v, T + dT, yi, 1.)
  _, lmbdkp1 = eigensolver(Qkp1, zetai, lmbd)
  dlmbddT = (lmbdkp1 - lmbd) / dT
  dT = -lmbd / dlmbddT
  repeat = lmbd < -tol or lmbd > tol
  while repeat and k < maxiter:
    k += 1
    T += dT
    _, Q = eos.getVT_lnfi_dnj(v, T, yi, 1.)
    zetai, lmbdkp1 = eigensolver(Q, zetai, lmbd)
    dlmbddT = (lmbdkp1 - lmbd) / dT
    lmbd = lmbdkp1
    dT = -lmbd / dlmbddT
    repeat = lmbd < -tol or lmbd > tol
  if not repeat:
    return T, lmbd, zetai
  logger.warning(
    'The spinodal temperature was not found.\nEOS: "%s".\nParameters:\n'
    'v = %s [m³/mol]\nyi = %s\nT0 = %s [K]\nzetai0 = %s\nmdT = %s',
    eos.name, v, yi.tolist(), T0, zetai0.tolist(), mdT,
  )
  raise SpinodalConvergenceError()


def _critVT_newt(
  eos: CritSolverVTEos,
  yi: Vector[Float],
  v0: float,
  T0: float,
  spinsolver: SpinSolver[SpinVTEos] = _spinVT_newt,
  mdv: float = 1e-5,
  tol: float = 1e-10,
  maxiter: int = 20,
) -> tuple[float, float]:
  """Solve the critical state problem for molar volume and temperature.

  Parameters
  ----------
  eos: CritSolverVTEos
    An initialized instance of a VT-based equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  v0: float
    An initial guess for the critical molar volume [m³/mol].

  T0: float
    An initial guess for the critical temperaure [K].

  spinsolver: SpinSolver[SpinVTEos]
    A callable object that can be used to find the spinodal temperature
    for a given volume and composition of a mixture.

  mdv: float
    A multiplier used to compute the partial derivative of the second
    critical point equation with respect to molar volume at the zeroth
    iteration. Default is `1e-5`.

  tol: float
    Terminate successfully if the absolute value of the primary
    variable change is less than `tol`. Default is `1e-10`.

  maxiter: int
    The maximum number of iterations. Default is `20`.

  Returns
  -------
  A tuple containing:
  - critical molar volume [m³/mol],
  - critical temperature [K].

  Raises
  ------
  CritConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The critical state problem for VT-based equations of state is
  represented by two equations.

  The first sets the minimum eigenvalue of the matrix of partial
  derivatives of natural logarithms of fugacities (with respect to mole
  numbers of components) to zero. This equation defines the spinodal
  condition and is linearized with the temperature in the inner loop
  using the `spinsolver`.

  The second sets the cubic form of the Taylor series decomposition of
  the Helmholtz energy function to zero. This condition describes the
  intrinsic stability of a thermodynamic state taking into account
  possible parameter variations (fluctuations). It is linearized in the
  outer loop with respect to relative molar volume (relative to minimum
  molar volume for a given temperature and composition of a mixture).

  Both equations are solved using Newton's method, for which all
  derivatives are calculated numerically. For the details of the
  algorithm implementation, see the following papers:

  1. R.A. Heidemann and A.M. Khalil, 1980 (doi: 10.1002/aic.690260510).
  2. B.E. Eaton, 1988 (doi: 10.6028/NBS.TN.1313).
  """
  logger.info('Calculating the critical point.')
  Nc = eos.Nc
  logger.info('yi =' + Nc * ' %6.4f', *yi)
  s = 0
  v = v0
  T, lmbd, zi = spinsolver(eos, v, yi, T0, yi)
  vmin = eos.getVT_vmin(T, yi)
  k = v / vmin
  m = k - 1.
  C = m * m * eos.getVT_d3F(v, T, yi, zi, 1.)
  vsp1 = v * (1. + mdv)
  ksp1 = vsp1 / vmin
  m = ksp1 - 1.
  Csp1 = m * m * eos.getVT_d3F(vsp1, T, yi, zi, 1.)
  dCdk = (Csp1 - C) / (ksp1 - k)
  dk = -C / dCdk
  repeat = dk < -tol or dk > tol
  logger.debug('%3s%9s%9s%11s%11s%11s', 'Nit', 'ϰ', 'T [K]', 'λ', 'C*', 'Δϰ')
  tmpl = '%3s%9.4f%9.2f%11.2e%11.2e%11.2e'
  logger.debug(tmpl, s, k, T, lmbd, C, dk)
  s += 1
  while repeat and s < maxiter:
    ksp1 = k + dk
    vsp1 = ksp1 * vmin
    if vsp1 < vmin:
      vsp1 = (v + vmin) * .5
      ksp1 = vsp1 / vmin
      dk = ksp1 - k
    T, lmbd, zisp1 = spinsolver(eos, vsp1, yi, T, zi)
    if zisp1[0] * zi[0] < 0.:
      zisp1 *= -1.
    m = ksp1 - 1.
    Csp1 = m * m * eos.getVT_d3F(vsp1, T, yi, zisp1, 1.)
    dCdk = (Csp1 - C) / dk
    dk = -Csp1 / dCdk
    vmin = eos.getVT_vmin(T, yi)
    v = vsp1
    k = ksp1
    C = Csp1
    zi = zisp1
    repeat = dk < -tol or dk > tol
    logger.debug(tmpl, s, k, T, lmbd, C, dk)
    s += 1
  if not repeat and isfinite(C):
    return v, T
  logger.warning(
    'The critical point calculation terminates unsuccessfully.\n'
    'The solver is "_critsolverVT".\nEOS: "%s".\nParameters:\n'
    'yi = %s\nv0 = %s [m³/mol]\nT0 = %s [K]\nmultdV0 = %s\n',
    eos.name, yi.tolist(), v0, T0, mdv,
  )
  raise CritConvergenceError()


class crit(object):
  def __init__(self, state: State | None = None) -> None:
    """Specify settings for a critical state calculation procedure.

    Parameters
    ----------
    state: State | None
      A thermodynamic state of a mixture, parameters of which can be
      used to initialize a critical state calculation procedure. If it
      is `None`, the option to use previously calculated results is not
      available until such a state is passed directly to the callable
      instance of this class. Default is `None`.

    Notes
    -----
    Above settings influence the calculation only if the corresponding
    parameters of the `__call__` method of this class are set to `None`.
    So, the priority of these "global" settings is lower than of those
    that will be given manually when the initialized instance is called.
    """
    self.state = state
    pass

  def __call__(
    self,
    eos: CritVTEos,
    yi: Vector[Float],
    n: float = 1.,
    init: tuple[float, float] | State | None = None,
    solver: CritSolver | str = 'default',
    **kwargs,
  ) -> OnePhaseState:
    """Calculate the critical state of a mixture.

    Parameters
    ----------
    eos: CritVTEos
      An initialized instance of an equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture. Default is `1.0` [mol].

    init: tuple[float, float] | State | None
      This parameter is used to initialize a critical state calculation
      procedure. The detailed explanation of the logic behind different
      types of `init` is given in the following table:

      +---------------------+------------------------------------------+
      | Type                |     Description                          |
      +=====================+==========================================+
      | tuple[float, float] | A tuple containing initial guesses for   |
      |                     | critical thermodynamic parameters.       |
      +---------------------+------------------------------------------+
      | State               | A state, parameters of which can be used |
      |                     | to initialize a procedure.               |
      +---------------------+------------------------------------------+
      | None                | An internal initialization routine is    |
      |                     | used to obtain initial guesses.          |
      +---------------------+------------------------------------------+

      Default is `None`.

    solver: CritSolver | str
      A callable object that can be used to solve the critical state
      problem formulated for a given equation of state. It also can
      be a string defining the name of an internal solver.

      For the VT-thermodynamics, the following internal solvers are
      available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'newton'`      | Newton's method is implemented to solve the  |
      |                 | problem (the spinodal equation is solved in  |
      |                 | the inner loop).                             |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      formulation of the two-phase flash problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | V, T            | Newton (`'newton'`).                         |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    **kwargs
      Other keyword arguments for an intenral initialization procedure.

    Returns
    -------
    The critical state of a mixture.

    Raises
    ------
    CritConvergenceError
      This exception is raised if the solver does not converge.
    """
    if init is None:
      init = self.state
    if eos.form == 'VT':
      eos = cast(CritVTEos, eos)
      if callable(solver):
        solverVT = cast(CritSolver[CritSolverVTEos], solver)
      elif solver == 'default' or solver == 'newton':
        solverVT = _critVT_newt
      else:
        raise ValueError(
          f'Unknown solver for the VT-thermodynamics: "{solver}".'
        )
      return self.runVT(eos, yi, n, init, solverVT, **kwargs)
    else:
      raise NotImplementedError(
        f'The {eos.form}-formulation of the critical state calculation '
        'procedure is not implemented yet.'
      )

  @classmethod
  def runVT(
    cls,
    eos: CritVTEos,
    yi: Vector[Float],
    n: float = 1.,
    init: tuple[float, float] | State | None = None,
    solver: CritSolver[CritSolverVTEos] = _critVT_newt,
    **kwargs
  ) -> OnePhaseState:
    """Calculate the critical state of a mixture using a VT-based
    equation of state.

    Parameters
    ----------
    eos: CritVTEos
      An initialized instance of a VT-based equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture. Default is `1.0` [mol].

    init: tuple[float, float] | State | None
      This parameter is used to initialize the critical state
      calculation procedure. The detailed explanation of the logic
      behind different types of `init` is given in the following table:

      +---------------------+------------------------------------------+
      | Type                |     Description                          |
      +=====================+==========================================+
      | tuple[float, float] | A tuple containing initial guesses for   |
      |                     | molar volume [m³/mol] and temperature    |
      |                     | [K].                                     |
      +---------------------+------------------------------------------+
      | State               | A state of a mixture, molar volume and   |
      |                     | temperature of which can be used to      |
      |                     | initialize the procedure.                |
      +---------------------+------------------------------------------+
      | None                | The Li-factor correlation is implemented |
      |                     | to estimate the initial guess of the     |
      |                     | critical temperature. After that, the    |
      |                     | gridding procedure is used to obtain the |
      |                     | initial guess of the critical molar      |
      |                     | volume.                                  |
      +---------------------+------------------------------------------+

      Default is `None`.

    solver: CritSolver[CritSolverVTEos]
      A callable object that can solve the critical state problem
      formulated for a VT-based equation of state. Default is
      `_critVT_newt`.

    **kwargs
      Other keyword arguments for the intenral gridding procedure.

    Returns
    -------
    The critical state of a mixture.

    Raises
    ------
    CritConvergenceError
      This exception is raised if the solver does not converge.
    """
    if init is None:
      v0, T0 = cls.gridding(eos, yi, **kwargs)
    elif isinstance(init, State):
      v0 = init.V / init.n
      T0 = init.T
    else:
      v0, T0 = init
    vc, Tc = solver(eos, yi, v0, T0)
    return cls.outputVT(eos, vc, Tc, yi, n)

  @staticmethod
  def gridding(
    eos: CritVTEos,
    yi: Vector[Float],
    factli: float = 1.,
    krange: tuple[float, float] = (1.1, 5.),
    Nnodes: int = 50,
    eigensolver: EigenSolver = rqi
  ) -> tuple[float, float]:
    r"""Internal initialization procedure for the VT-based calculation
    of the critical state of a mixture.

    Parameters
    ----------
    eos: CritVTEos
      An initialized instance of an equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    factli: float
      A coefficient for the Li-correlation used to estimate the
      critical temperature of a mixture.

      .. math::

        T_c = C \cdot frac{\sum_{i=1}^{N_c} y_i {v_c}_i {T_c}_i
                          {\sum_{i=1}^{N_c} y_i {v_c}_i},

      where :math:`T_c` is the initial guess of the critical temperature
      :math:`y_i, \, i = 1 \, \ldots \, N_c,` is the mole fraction of
      :math:`i`-th component in a mixture, :math:`{v_c}_i` and
      :math:`{T_c}_i` are critical mole volume and temperature of
      :math:`i`-th component respectively. :math:`C` is the Li-factor
      defined be the `factli` parameter. Default is `1.0`.

    krange: tuple[float, float]
      A range of possible values of the relation of the molar volume
      to the minimal possible volume provided by an equation of state
      for the gridding procedure. Default is `(1.1, 5.0)`.

    Nnodes: int
      The number of grid nodes. Default is `50`.

    eigensolver: EigenSolver
      A callable object that for a given `Q: Matrix[Float]` of shape
      `(Nc, Nc)` finds a pair of the eigenvector of shape `(Nc,)` and
      the corresponding eigenvalue. Default is the implementation of
      the Rayleigh quotient iteration `rqi`.

    Returns
    -------
    A tuple containing initial guesses for critical molar volume and
    temperature.

    Raises
    ------
    ValueError
      This exception is raised if for a given range of the molar volume,
      the change in a sign of the minimum eigenvalue of the partial
      derivatives matrix is not detected.

    Notes
    -----
    The initialization procedure consists of the following steps.

    First, the initial guess for the critical temperature is estimated
    using the Li-correlation. The `factli` parameter can be used to
    correct this temperature.

    Next, the gridding procedure identifies the molar volume where the
    minimum eigenvalue of the fugacity-logarithm derivative matrix
    udergoes a sign change.

    Once this range is identified, the second gridding is implemented to
    refine the initial guess for the critical molar volume.
    """
    logger.debug('Initializing with the gridding procedure.')
    wi = yi * eos.vci
    T0 = factli * wi.dot(eos.Tci) / wi.sum()
    logger.debug('Initial guess for temperature: %.2f [K].', T0)
    vmin = eos.getVT_vmin(T0, yi)
    vv = np_linspace(*krange, Nnodes, endpoint=True) * vmin
    vsm1 = vv[0]
    Q = eos.getVT_lnfi_dnj(vsm1, T0, yi, 1.)[1]
    zism1, lmbdsm1 = eigensolver(Q, yi, None)
    logger.debug('For v = %.3e [m³/mol]: λ = %.3f.', vsm1, lmbdsm1)
    for v in vv[1:]:
      Q = eos.getVT_lnfi_dnj(v, T0, yi, 1.)[1]
      zi, lmbd = eigensolver(Q, zism1, lmbdsm1)
      logger.debug('For v = %.3e [m³/mol]: λ = %.3f.', v, lmbd)
      if lmbd * lmbdsm1 < 0.:
        break
      else:
        vsm1 = v
        zism1 = zi
        lmbdsm1 = lmbd
    else:
      raise ValueError(
        'For a given range of the molar volume, the change in a sign of '
        'the minimum eigenvalue of the partial derivatives matrix was '
        'not detected. Try to change `krange` or `factli` parameters of '
        'the gridding procedure.'
      )
    vv = np_linspace(vsm1, v, Nnodes, endpoint=True)
    for v in vv[1:]:
      Q = eos.getVT_lnfi_dnj(v, T0, yi, 1.)[1]
      zi, lmbd = eigensolver(Q, zism1, lmbdsm1)
      logger.debug('For v = %.3e [m³/mol]: λ = %.3f.', v, lmbd)
      if lmbd * lmbdsm1 < 0.:
        break
      else:
        vsm1 = v
        zism1 = zi
        lmbdsm1 = lmbd
    return (vsm1 + v) * .5, T0

  @staticmethod
  def outputVT(
    eos: State2pVTEos,
    v: float,
    T: float,
    yi: Vector[Float],
    n: float,
  ) -> OnePhaseState:
    P = eos.getVT_P(v, T, yi)
    V = v * n
    ni = n * yi
    nj = np_array([n])
    nji = np_atleast_2d(ni)
    fj = np_array([1.])
    yji = np_atleast_2d(yi)
    Z = P * v / (R * T)
    Zj = np_array([Z])
    vj = np_array([v])
    Vj = np_array([V])
    sj = np_array([1.])
    d = yi.dot(eos.mwi) / v
    dj = np_array([d])
    pid = eos.getVT_PID(V, T, yi)
    pidj = np_array([pid])
    return OnePhaseState(
      eos.Nc, 1, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
      None,
    )
