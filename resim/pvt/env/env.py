from logging import (
  getLogger,
)

from typing import (
  Callable,
  cast,
)

from math import (
  copysign,
  exp,
  isfinite,
  log,
)

from numpy import (
  abs as np_abs,
  append as np_append,
  argmax as np_argmax,
  argmin as np_argmin,
  array as np_array,
  column_stack as np_column_stack,
  empty as np_empty,
  exp as np_exp,
  eye as np_eye,
  diff as np_diff,
  full as np_full,
  full_like as np_full_like,
  linspace as np_linspace,
  log as np_log,
  sign as np_sign,
  stack as np_stack,
  vstack as np_vstack,
  where as np_where,
  zeros as np_zeros,
)

from resim.pvt.datatypes import (
  Envelope,
  Float,
  Integer,
  Matrix,
  State,
  Vector,
)

from resim.pvt.constants import (
  R,
)

from resim.pvt.utils import (
  LinearSolver,
  lusolver,
)

from resim.pvt.eos import (
  StateNpPTEos,
)

from resim.pvt.flash import (
  Flash2pPTEos,
  FlashConvergenceError,
  FlashRoutine,
  RRConvergenceError,
  flash,
)

from resim.pvt.psat import (
  PsatPTEos,
  PsatRoutine,
  psat,
)

from resim.pvt.env.protocols import (
  Env2pPTEos,
  Env2pSolver,
  Env2pSolverPTEos,
  EnvNpSolverPTEos,
)


logger = getLogger('env')


class EnvConvergenceError(Exception):
  """An exception that will be raised if a solver for envelope
  problems does not converge.
  """
  def __init__(
    self,
    msg: str = ('The solver for the envelope problem does not converge. '
                'Try to improve the initial guess or change fixed variable'
                'index or value. It may also be advisable to increase '
                'the number of iterations.'),
  ) -> None:
    super().__init__(msg)
    pass


def _env2pPT_newt(
  eos: Env2pSolverPTEos,
  x0: Vector[Float],
  sidx: int | Integer,
  sval: float,
  yi: Vector[Float],
  phf: float,
  tolres: float = 1e-24,
  tolvar: float = 1e-14,
  maxiter: int = 5,
  miniter: int = 0,
  dxmax: float = 0.1,
  linsolver: LinearSolver = lusolver,
) -> tuple[Vector[Float], Vector[Float], Vector[Float], Matrix[Float], int]:
  r"""Solve the two-phase envelope problem using Newton's method.

  Parameters
  ----------
  eos: Env2pSolverPTEos
    An inizialized instance of a PT-based equation of state.

  x0: Vector[Float], shape (Nc + 2,)
    An initial guess of primary variables.

  sidx: int | Integer
    The specified variable index. The specified variable is considered
    known and fixed for the algorithm.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of components in the mixture.

  phf: float
    Phase mole fraction for which the phase envelope point should be
    calculated.

  tolres: float
    Terminate successfully if the sum of squared elements of the vector
    of the phase envelope equations is less than `tolres`. Default is
    `1e-24`.

  tolvar: float
    Terminate successfully if the sum of squared elements of the
    direction vector is less than `tolvar`. Default is `1e-14`.

  maxiter: int
    The maximum number of iterations. Default is `5`.

  miniter: int
    The minimum number of iterations. Default is `0`.

  dxmax: float
    The maximum absolute change of a variable during iteration.
    Defult is `0.1`.

  linsolver: LinearSolver
    A callable object that accepts an `A: Matrix[Float]` of shape
    `(Nc + 2, Nc + 2)` and `b: Vector[Float]` of shape `(Nc + 2,)` and
    finds `x: Vector[Float]` of shape `(Nc + 2,)`, which is the solution
    to the linear system :math:`\mathbf{A}^\top \mathbf{x}= \mathbf{b}`.
    The matrix `A` is a non-symmetric matrix. Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - the solution of the two-phase envelope problem as a `Vector[Float]`
    of shape `(Nc + 2,)`,
  - mole fractions of components in the non-reference phase as a
    `Vector[Float]` of shape `(Nc,)`
  - mole fractions of components in the reference phase as a
    `Vector[Float]` of shape `(Nc,)`
  - the Jacobian (at the solution) as a `Matrix[Float]` of shape
    `(Nc + 2, Nc + 2)`,
  - the number of iterations to converge.

  Raises
  ------
  EnvConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The implemented algorithm is based on the paper of M.L. Michelsen,
  1980 (doi: 10.1016/0378-3812(80)80001-X). The two-phase envelope
  problem constists of `(Nc + 2)` equations:

  - `Nc` phase equilibrium equations represented by equalities of
    fugacities of corresponding components in phases^

    .. math::

      \mathbf{f} \left(P, \, T, \, \mathbf{y} \right) -
      \mathbf{f} \left(P, \, T, \, \mathbf{x} \right) = 0,

    where :math:`\mathbf{f}` is a vector of fugacities that depends
    on pressure :math:`P`, temperature :math:`T`, and phase composition:
    :math:`\mathbf{y}` (vapour) or :math:`\mathbf{x}` (liquid);

  - the constraint equation that can be formulated as the following:

    .. math::

      \sum_{i=1}^{N_c} y_i - \sum_{i=1}^{N_c} x_i = 0;

  - the specification equation:

    ..math::

      \alpfa - S = 0,

    where :math:`\alpha` is an item of the vector of primary variables
    (specified or specification variable) and :math:`S` is the value of
    that variable.

  The above system of non-linear equations is resolved with respect to
  `(Nc + 2)` primary variables:
  - `Nc` natural logarithms of k-values of components,
  - the natural logarithm of pressure,
  - the natural logarithm of temperature.
  """
  logger.debug(
    'Solving the system of two-phase boundary equations for: '
    'phf = %.3f, sidx = %s, sval = %.4f', phf, sidx, sval,
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%8s%10s%10s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'lnP', 'lnT', 'g2', 'dx2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.4f%8.4f%10.2e%10.2e'
  J = np_zeros(shape=(Nc + 2, Nc + 2))
  J[-1, sidx] = 1.
  g = np_empty(shape=(Nc + 2,))
  I = np_eye(Nc)
  k = 0
  xk = x0.flatten()
  xk[sidx] = sval
  ex = np_exp(xk)
  P = ex[-2]
  T = ex[-1]
  lnkvi = xk[:Nc]
  kvi = ex[:Nc]
  di = 1. + phf * (kvi - 1.)
  yli = yi / di
  yvi = kvi * yli
  lnphivi, dlnphividP, dlnphividT, dlnphividyvj = eos.getPT_lnphii_dP_dT_dyj(
    P, T, yvi,
  )
  lnphili, dlnphilidP, dlnphilidT, dlnphilidylj = eos.getPT_lnphii_dP_dT_dyj(
    P, T, yli,
  )
  g[:Nc] = lnkvi + lnphivi - lnphili
  g[Nc] = (yvi - yli).sum()
  g[Nc+1] = xk[sidx] - sval
  g2 = g.dot(g)
  dylidlnkvi = -phf * yvi / di
  dyvidlnkvi = yvi + kvi * dylidlnkvi
  J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
  J[-2,:Nc] = yvi / di
  J[:Nc,-2] = P * (dlnphividP - dlnphilidP)
  J[:Nc,-1] = T * (dlnphividT - dlnphilidT)
  dx = linsolver(J, -g)
  dx2 = dx.dot(dx)
  repeat = isfinite(dx2) and (dx2 > tolvar and g2 > tolres or k < miniter)
  logger.debug(tmpl, k, *xk, g2, dx2)
  while repeat and k < maxiter:
    k += 1
    dx = np_where(np_abs(dx) > dxmax, np_sign(dx) * dxmax, dx)
    xk += dx
    ex = np_exp(xk)
    P = ex[-2]
    T = ex[-1]
    lnkvi = xk[:Nc]
    kvi = ex[:Nc]
    di = 1. + phf * (kvi - 1.)
    yli = yi / di
    yvi = kvi * yli
    lnphivi, dlnphividP,dlnphividT, dlnphividyvj = eos.getPT_lnphii_dP_dT_dyj(
      P, T, yvi,
    )
    lnphili, dlnphilidP,dlnphilidT, dlnphilidylj = eos.getPT_lnphii_dP_dT_dyj(
      P, T, yli,
    )
    g[:Nc] = lnkvi + lnphivi - lnphili
    g[Nc] = (yvi - yli).sum()
    g[Nc+1] = xk[sidx] - sval
    g2 = g.dot(g)
    dylidlnkvi = -phf * yvi / di
    dyvidlnkvi = yvi + kvi * dylidlnkvi
    J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
    J[-2,:Nc] = yvi / di
    J[:Nc,-2] = P * (dlnphividP - dlnphilidP)
    J[:Nc,-1] = T * (dlnphividT - dlnphilidT)
    dx = linsolver(J, -g)
    dx2 = dx.dot(dx)
    repeat = isfinite(dx2) and (dx2 > tolvar and g2 > tolres or k < miniter)
    logger.debug(tmpl, k, *xk, g2, dx2)
  if (g2 < tolres or dx2 < tolvar) and isfinite(dx2):
    return xk, yvi, yli, J, k
  raise EnvConvergenceError()


def _envNpPT_newt(
  eos: EnvNpSolverPTEos,
  x0: Vector[Float],
  sidx: int | Integer,
  sval: float,
  fidx: int | Integer,
  fval: float,
  yi: Vector[Float],
  tolres: float = 1e-24,
  tolvar: float = 1e-14,
  maxiter: int = 20,
  miniter: int = 0,
  dxmax: float = 0.1,
  linsolver: LinearSolver = lusolver,
) -> tuple[Vector[Float], Matrix[Float], Vector[Float], Matrix[Float], int]:
  logger.debug(
    'Solving the system of multiphase boundary equations for: '
    'fidx = %s, fval = %.3f, sidx = %s, sval = %.4f', fidx, fval, sidx, sval,
  )
  Nc = eos.Nc
  Neq = x0.shape[0]
  Npm1 = (Neq - 2) // (Nc + 1)
  Npm1Nc = Npm1 * Nc
  Npm2 = Npm1 - 1
  logger.debug(
    '%3s' + Npm1 * '%9s' + Npm1Nc * '%10s' + '%10s%10s%11s%11s',
    'Nit',
    *['f%s' % j for j in range(Npm1)],
    *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    'lnP',
    'lnT',
    'g2',
    'x2',
  )
  tmpl = '%3s' + Npm1*'%9.4f' + Npm1Nc*'%10.4f' + '%10.4f%10.4f%11.2e%11.2e'
  g = np_empty(shape=(Neq,))
  q = g[:Npm2]
  r = g[Npm1:-2].reshape((Npm1, Nc))
  J = np_zeros(shape=(Neq, Neq))
  if sidx > 0:
    sidx = Npm1 + sidx
  J[-2, sidx] = 1.
  J[-1, fidx] = 1.
  dqdf = J[:Npm2, :Npm1]
  dhdf = J[Npm2, :Npm1]
  drdf = J[Npm1:-2, :Npm1].reshape(Npm1, Nc, Npm1)
  dqdk = J[:Npm2, Npm1:-2].reshape(Npm2, Npm1, Nc)
  dhdk = J[Npm2, Npm1:-2].reshape(Npm1, Nc)
  drdk = J[Npm1:-2, Npm1:-2].reshape(Npm1, Nc, Npm1, Nc)
  drdp = J[Npm1:-2, -2]
  drdt = J[Npm1:-2, -1]
  Ijk = np_eye(Npm1, Npm1)
  dIjk = np_eye(Npm2, Npm1, 1) - np_eye(Npm2, Npm1)
  Ijikl = np_eye(Npm1 * Nc).reshape(Npm1, Nc, Npm1, Nc)
  k = 0
  xk = x0.flatten()
  xk[sidx] = sval
  fj = xk[:Npm1]
  fj[fidx] = fval
  lnkvi = xk[Npm1:-2]
  lnkvji = lnkvi.reshape(Npm1, Nc)
  P = exp(xk[-2])
  T = exp(xk[-1])
  kvji = np_exp(lnkvji)
  Aji = 1. - kvji
  ti = 1. - fj.dot(Aji)
  xi = yi / ti
  yji = kvji * xi
  lnphiji, dlnphijidP, dlnphijidT, dlnphijidyjk = eos.getPT_lnphiji_dP_dT_dyk(
    P, T, yji,
  )
  lnphixi, dlnphixidP, dlnphixidT, dlnphixidxk = eos.getPT_lnphii_dP_dT_dyj(
    P, T, xi,
  )
  dkvji = np_diff(kvji, axis=0)
  q[:] = dkvji.dot(xi)
  g[Npm2] = xi.sum() - 1.
  r[:] = lnkvji + lnphiji - lnphixi
  g[-2] = xk[sidx] - sval
  g[-1] = fj[fidx] - fval
  g2 = g.dot(g)
  ui = xi / ti
  dqdf[:] = (dkvji * ui).dot(Aji.T)
  dhdf[:] = Aji.dot(ui)
  drdf[:] = (dlnphijidyjk * kvji[:,None,:] - dlnphixidxk).dot((ui * Aji).T)
  dqdk[:] = (
    kvji * ui * (ti * dIjk[:,:,None] - fj[None,:,None] * dkvji[:,None,:])
  )
  dhdk[:] = -fj[:,None] * kvji * ui
  drdk[:] = Ijikl + (
    yji
    * (
        dlnphijidyjk[:,:,None,:]
        * (
            Ijk[:,None,:,None]
            - kvji[:,None,None,:] * fj[None,None,:,None] / ti
        )
        + dlnphixidxk[None,:,None,:] * fj[None,None,:,None] / ti
    )
  )
  drdp[:] = P * (dlnphijidP - dlnphixidP).ravel()
  drdt[:] = T * (dlnphijidT - dlnphixidT).ravel()
  dx = linsolver(J, -g)
  dx2 = dx.dot(dx)
  logger.debug(tmpl, k, *xk, g2, dx2)
  repeat = isfinite(dx2) and (dx2 > tolvar and g2 > tolres or k < miniter)
  while repeat and k < maxiter:
    k += 1
    dx = np_where(np_abs(dx) > dxmax, np_sign(dx) * dxmax, dx)
    xk += dx
    kvji = np_exp(lnkvji)
    P = exp(xk[-2])
    T = exp(xk[-1])
    Aji = 1. - kvji
    ti = 1. - fj.dot(Aji)
    xi = yi / ti
    yji = kvji * xi
    lnphiji, dlnphijidP, dlnphijidT, dlnphijidyjk = (
      eos.getPT_lnphiji_dP_dT_dyk(
        P, T, yji,
      )
    )
    lnphixi, dlnphixidP, dlnphixidT, dlnphixidxk = (
      eos.getPT_lnphii_dP_dT_dyj(
        P, T, xi,
      )
    )
    dkvji = np_diff(kvji, axis=0)
    q[:] = dkvji.dot(xi)
    g[Npm2] = xi.sum() - 1.
    r[:] = lnkvji + lnphiji - lnphixi
    g[-2] = xk[sidx] - sval
    g[-1] = fj[fidx] - fval
    g2 = g.dot(g)
    ui = xi / ti
    dqdf[:] = (dkvji * ui).dot(Aji.T)
    dhdf[:] = Aji.dot(ui)
    drdf[:] = (dlnphijidyjk * kvji[:,None,:] - dlnphixidxk).dot((ui * Aji).T)
    dqdk[:] = (
      kvji * ui * (ti * dIjk[:,:,None] - fj[None,:,None] * dkvji[:,None,:])
    )
    dhdk[:] = -fj[:,None] * kvji * ui
    drdk[:] = Ijikl + (
      yji
      * (
          dlnphijidyjk[:,:,None,:]
          * (
              Ijk[:,None,:,None]
              - kvji[:,None,None,:] * fj[None,None,:,None] / ti
          )
          + dlnphixidxk[None,:,None,:] * fj[None,None,:,None] / ti
      )
    )
    drdp[:] = P * (dlnphijidP - dlnphixidP).ravel()
    drdt[:] = T * (dlnphijidT - dlnphixidT).ravel()
    dx = linsolver(J, -g)
    dx2 = dx.dot(dx)
    repeat = isfinite(dx2) and (dx2 > tolvar and g2 > tolres or k < miniter)
    logger.debug(tmpl, k, *xk, g2, dx2)
  if (g2 < tolres or dx2 < tolvar) and isfinite(dx2):
    return xk, yji, xi, J, k
  raise EnvConvergenceError()


def _env2pPT_step(Niter: int, dx: Vector[Float]) -> float:
  """A function that can be used to control the step size during the
  phase envelope calculation routine. The step size miltipliers were
  taken from the paper of L. Xu and H. Li, 2023 (doi:
  10.1016/j.geoen.2023.212058).

  Parameters
  ----------
  Niter: int
    The number of iterations the solver required to converge for the
    previous phase envelope point.

  dx: Vector[Float], shape (Nc + 2,)
    Changes of primary variables for the previous phase envelope point.

  Returns
  -------
  The step multiplier used to calculate the next value of the specified
  variable.

  Notes
  -----
  This function does not use the changes of primary variables to select
  the multiplier for the step size. However, one can create a function
  that takes into account the departure of actual changes of primary
  variables from the expected ones to obtain the step size multiplier.
  """
  if Niter < 2:
    return 2.
  elif Niter == 2:
    return 1.5
  elif Niter == 3:
    return 1.2
  elif Niter == 4:
    return 0.9
  else:
    return 0.5


class env2p(object):
  def __init__(
    self,
    sidx0: int | None = None,
    step0: float = 0.01,
    fstep: Callable[[int, Vector[Float]], float] = _env2pPT_step,
    maxstep: float = 0.25,
    switchmult: float = 0.5,
    unconvmult: float = 0.75,
    maxrepeats: int = 8,
    maxpoints: int = 200,
    Pmin: float = 1.,
    Pmax: float = 1e8,
    Tmin: float = 173.15,
    Tmax: float = 973.15,
  ) -> None:
    """Specify settings for the phase envelope calculation.

    Parameters
    ----------
    sidx0: int | None
      This parameter specifies an index of the fixed item of a vector of
      the primary variables for calculation of the first point of the
      phase envelope. If it is `None`, the internal procedure is used
      to determine the index of the specification variable. Default is
      `None`.

    step0: float
      A change in the primary variable (step size) after the first point
      is calculted. Default is `0.01`.

    fstep: Callable[[int, Vector[Float]], float]
      A callable object that can be used to control the step size for
      the phase envelope calculation procedure. It must accept the
      number of iterations and a `Vector[Float]` of shape `(Nc + 2,)` of
      primary variables changes. Default is `_env2pPT_step`.

    maxstep: float
      The maximum step size for the phase envelope calculation routine.
      Default is `0.25`.

    switchmult: float
      The multiplier which is used to reduce the step size if the
      specified variable is switched. Default is `0.5`.

    unconvmult: float
      The multiplier which is used to reduce the step size if the solver
      for the phase envelope problem does not converge. Default is
      `0.75`.

    maxrepeats: int
      The maximum number of step size cuts that are used to reduce the
      step size if a solver for the phase envelope problem does not
      converge for any point of the phase envelope. Default is `8`.

    maxpoints: int
      The maximum number of points of the phase envelope. Default is
      `200`.

    Pmin: float
      The minimum pressure [Pa] for the phase envelope calculation
      routine. Default is `1.0` [Pa].

    Pmax: float
      The maximum pressure [Pa] for the phase envelope calculation
      routine. Default is `1e8` [Pa].

    Tmin: float
      The minimum temperature [K] for the phase envelope calculation
      routine. Default is `173.15` [K].

    Tmax: float
      The maximum temperature [K] for the phase envelope calculation
      routine. Default is `937.15` [K].

    Notes
    -----
    These settings influence the calculation only if the corresponding
    parameters of the `__call__` method of this class are set to `None`.
    """
    self.sidx0 = sidx0
    self.step0 = step0
    self.fstep = fstep
    self.maxstep = maxstep
    self.switchmult = switchmult
    self.unconvmult = unconvmult
    self.maxrepeats = maxrepeats
    self.maxpoints = maxpoints
    self.Pmin = Pmin
    self.Pmax = Pmax
    self.Tmin = Tmin
    self.Tmax = Tmax
    pass

  def __call__(
    self,
    eos: Env2pPTEos,
    yi: Vector[Float],
    phf: float,
    init: tuple[float, float, Vector[Float]] | State | float,
    solver: Env2pSolver | str = 'default',
    sidx0: int | None = None,
    step0: float | None = None,
    fstep: Callable[[int, Vector[Float]], float] | None = None,
    maxstep: float | None = None,
    switchmult: float | None = None,
    unconvmult: float | None = None,
    maxrepeats: int | None = None,
    maxpoints: int | None = None,
    Pmin: float | None = None,
    Pmax: float | None = None,
    Tmin: float | None = None,
    Tmax: float | None = None,
    **kwargs,
  ) -> Envelope:
    """Calculate the two-phase envelope of a mixture.

    Parameters
    ----------
    eos: Env2pPTEos
      An initialized instance of an equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    phf: float
      Phase mole fraction for which the envelope is calculated.

    init: tuple[float, float, Vector[Float]] | State | float
      This parameter is used to initialize the phase envelope routine.
      The detailed explanation of the logic behind different types of
      `init` is given in the following table:

      +------------------+---------------------------------------------+
      | Type             | Description                                 |
      +==================+=============================================+
      | tuple[           | A tuple containing an initial guess of:     |
      |   float,         | - presure [Pa],                             |
      |   float,         | - temperature [K],                          |
      |   Vector[Float], | - k-values of components as a vector of     |
      | ]                | shape `(Nc,)`.                              |
      +------------------+---------------------------------------------+
      | State            | A state of a mixture, k-values, pressure,   |
      |                  | and temperature of which can be used as an  |
      |                  | initial guess for calculation of the first  |
      |                  | point of the phase envelope.                |
      +------------------+---------------------------------------------+
      | float            | A temperature [K], from which the phase     |
      |                  | envelope calculation procedure starts. The  |
      |                  | initialization routine performs following   |
      |                  | steps. First, the saturation pressure is    |
      |                  | determined. Then, the gridding procedure is |
      |                  | implemented to find a phase mole fraction   |
      |                  | closed to a given one by employing several  |
      |                  | flash calculations along the pressure axis. |
      |                  | The resulting two-phase state is used to as |
      |                  | an initial guess for the first point of the |
      |                  | phase envelope.                             |
      +------------------+---------------------------------------------+

    solver: Env2pSolver[Env2pSolverPTEos]
      A callable object that can solve the phase envelope problem
      formulated for the given equation of state. It also can be a
      string defining the name of an internal solver.

      For the PT-thermodynamics, the following internal solvers are
      available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'newton'`      | Uses Newton's method to solve the system of  |
      |                 | non-linear equations.                        |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      formulation of the phase envelope problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | P, T            | Newton (`'newton'`).                         |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    sidx0: int | None
      This parameter specifies an index of the fixed item of a vector of
      the primary variables for calculation of the first point of the
      phase envelope. If it is `None`, the internal procedure is used
      to determine the index of the specification variable. Default is
      `None`.

    step0: float | None
      A change in the primary variable (step size) after the first point
      is calculted. Default is `None`.

    fstep: Callable[[int, Vector[Float]], float] | None
      A callable object that can be used to control the step size for
      the phase envelope calculation procedure. It must accept the
      number of iterations and a `Vector[Float]` of shape `(Nc + 2,)` of
      primary variables changes. Default is `None`.

    maxstep: float | None
      The maximum step size for the phase envelope calculation routine.
      Default is `None`.

    switchmult: float | None
      The multiplier which is used to reduce the step size if the
      specified variable is switched. Default is `None`.

    unconvmult: float | None
      The multiplier which is used to reduce the step size if the solver
      for the phase envelope problem does not converge. Default is
      `None`.

    maxrepeats: int | None
      The maximum number of step size cuts that are used to reduce the
      step size if a solver for the phase envelope problem does not
      converge for any point of the phase envelope. Default is `None`.

    maxpoints: int | None
      The maximum number of points of the phase envelope. Default is
      `None`.

    Pmin: float | None
      The minimum pressure [Pa] for the phase envelope calculation
      routine. Default is `None`.

    Pmax: float | None
      The maximum pressure [Pa] for the phase envelope calculation
      routine. Default is `None`.

    Tmin: float | None
      The minimum temperature [K] for the phase envelope calculation
      routine. Default is `None`.

    Tmax: float | None
      The maximum temperature [K] for the phase envelope calculation
      routine. Default is `None`.

    **kwargs
      Other parameters for an internal initialization procedure.

    Notes
    -----
    1. If the following parameters are set to `None`, a value specified
    for the corresponding parameter during the initialization of this
    callable object is used. The following table lists parameters and
    their default values.

    +---------------------------------+--------------------------------+
    | Parameter                       | Default                        |
    +=================================+================================+
    | `sidx0`                         | `0.01`                         |
    +---------------------------------+--------------------------------+
    | `fstep`                         | `_env2pPT_step`                |
    +---------------------------------+--------------------------------+
    | `maxstep`                       | `0.25`                         |
    +---------------------------------+--------------------------------+
    | `switchmult`                    | `0.5                           |
    +---------------------------------+--------------------------------+
    | `unconvmult`                    | `0.75`                         |
    +---------------------------------+--------------------------------+
    | `maxrepeats`                    | `8`                            |
    +---------------------------------+--------------------------------+
    | `maxpoints`                     | `200`                          |
    +---------------------------------+--------------------------------+
    | `Pmin`                          | `1.0` [Pa]                     |
    +---------------------------------+--------------------------------+
    | `Pmax`                          | `1e8` [Pa]                     |
    +---------------------------------+--------------------------------+
    | `Tmin`                          | `173.15` [K]                   |
    +---------------------------------+--------------------------------+
    | `Tmax`                          | `973.15` [K]                   |
    +---------------------------------+--------------------------------+

    2. The phase envelope calculation routine is based on the algorithms
    described by M.L. Michelsen with some custom modifications. For the
    math and other details behind the algorithms, see the following
    papers:
    - M.L. Michelsen, 1980 (doi: 10.1016/0378-3812(80)80001-X),
    - Y.-K. Li and L.X. Nghiem, 1982 (doi: 10.2118/11198-MS).
    """
    if sidx0 is None:
      sidx0 = self.sidx0
    if step0 is None:
      step0 = self.step0
    if fstep is None:
      fstep = self.fstep
    if maxstep is None:
      maxstep = self.maxstep
    if switchmult is None:
      switchmult = self.switchmult
    if unconvmult is None:
      unconvmult = self.unconvmult
    if maxrepeats is None:
      maxrepeats = self.maxrepeats
    if maxpoints is None:
      maxpoints = self.maxpoints
    if Pmin is None:
      Pmin = self.Pmin
    if Pmax is None:
      Pmax = self.Pmax
    if Tmin is None:
      Tmin = self.Tmin
    if Tmax is None:
      Tmax = self.Tmax
    if eos.form == 'PT':
      eos = cast(Env2pPTEos, eos)
      solverPT: Env2pSolver[Env2pSolverPTEos]
      if callable(solver):
        solverPT = cast(Env2pSolver[Env2pSolverPTEos], solver)
      elif solver == 'default' or solver == 'newton':
        solverPT = _env2pPT_newt
      elif solver == 'tr':
        raise NotImplementedError(
          'The trust region method for the phase envelope calculation '
          'is not implemented yet.'
        )
      else:
        raise ValueError(
          'Unknown solver of phase envelope problems based on the '
          f'PT-thermodynamics: "{solver}".'
        )
      return self.runPT(eos, yi, phf, init, sidx0, step0, fstep, maxstep,
                        switchmult, unconvmult, maxrepeats, maxpoints,
                        Pmin, Pmax, Tmin, Tmax, solverPT, **kwargs)
    else:
      raise NotImplementedError(
        f'The {eos.form}-formulation of the phase envelope calculation '
        'is not implemented yet.'
      )

  @classmethod
  def runPT(
    cls,
    eos: Env2pPTEos,
    yi: Vector[Float],
    phf: float,
    init: tuple[float, float, Vector[Float]] | State | float,
    sidx0: int | None = None,
    step0: float = 0.01,
    fstep: Callable[[int, Vector[Float]], float] = _env2pPT_step,
    maxstep: float = 0.25,
    switchmult: float = 0.5,
    unconvmult: float = 0.75,
    maxrepeats: int = 8,
    maxpoints: int = 200,
    Pmin: float = 1.,
    Pmax: float = 1e8,
    Tmin: float = 173.15,
    Tmax: float = 973.15,
    solver: Env2pSolver[Env2pSolverPTEos] = _env2pPT_newt,
    **kwargs
  ) -> Envelope:
    """Calculate the two-phase envelope of a mixture using a PT-based
    equation of state.

    Parameters
    ----------
    eos: Env2pPTEos
      An initialized instance of a PT-based equation of state (PTEos).

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    phf: float
      Phase mole fraction for which the envelope is calculated.

    init: tuple[float, float, Vector[Float]] | State | float
      This parameter is used to initialize the phase envelope routine.
      The detailed explanation of the logic behind different types of
      `init` is given in the following table:

      +------------------+---------------------------------------------+
      | Type             | Description                                 |
      +==================+=============================================+
      | tuple[           | A tuple containing an initial guess of:     |
      |   float,         | - presure [Pa],                             |
      |   float,         | - temperature [K],                          |
      |   Vector[Float], | - k-values of components as a vector of     |
      | ]                | shape `(Nc,)`.                              |
      +------------------+---------------------------------------------+
      | State            | A state of a mixture, k-values, pressure,   |
      |                  | and temperature of which can be used as an  |
      |                  | initial guess for calculation of the first  |
      |                  | point of the phase envelope.                |
      +------------------+---------------------------------------------+
      | float            | A temperature [K], from which the phase     |
      |                  | envelope calculation procedure starts. The  |
      |                  | initialization routine performs following   |
      |                  | steps. First, the saturation pressure is    |
      |                  | determined. Then, the gridding procedure is |
      |                  | implemented to find a phase mole fraction   |
      |                  | closed to a given one by employing several  |
      |                  | flash calculations along the pressure axis. |
      |                  | The resulting two-phase state is used as an |
      |                  | initial guess for the first point of the    |
      |                  | phase envelope.                             |
      +------------------+---------------------------------------------+

      For all types of initialization, it is not recommended to specify
      an initial guess close to the critical point of a mixture.

    solver: Env2pSolver[Env2pSolverPTEos]
      A callable object that can solve the phase envelope problem
      formulated for the PT-thermodynamics. Default is `_env2pPT_newt`.

    sidx0: int | None
      For the first point, this parameter allows to specify the index
      of the fixed item of a vector of the primary variables:

      - `Nc` natural logarithms of k-values of components,
      - the natural logarithm of pressure,
      - the natural logarithm of temperature.

      The specified (fixed) variable (or the specification variable) is
      considered known and fixed for the calculation of an envelope
      point. Therefore, changing this index may improve the algorithm
      converegence for the first and subsequent points. To initiate
      calculations, specification of pressure was recommended by M.L.
      Michelsen in his paper (doi: 10.1016/0378-3812(80)80001-X). The
      general rule for the specified variable selection was also given
      in this paper. It was recommended to select the specified variable
      based on the largest rate of change, which refers to the largest
      derivative of the primary variables with respect to the specified
      variable (the so-called sensitivity vector). When `sidx0` is set
      to `None`, it will correspond to the least volatile component on
      the bubble point line and the most volatile component on the dew
      point line. Default is `None`.

    step0: float
      The step size (the difference between two subsequent values of a
      specified variable) after the first point is calculted. It should
      be small enough to consider the result obtained from the linear
      extrapolation of the first phase envelope point as a good initial
      guess for the next point. Default is `0.01`.

    fstep: Callable[[int, Vector[Float]], float]
      A callable object that can be used to control the step size for
      the phase envelope calculation procedure. It must accept the
      number of iterations and a `Vector[Float]` of shape `(Nc + 2,)` of
      primary variables changes between previous and current points. The
      callable object must return a value of the step size multiplier.
      Default is `_env2pPT_step`.

    maxstep: float
      The maximum step size for the phase envelope calculation routine.
      Default is `0.25`.

      The step size calculation determined by the `fstep` and `maxstep`
      parameters is the most significant aspect of the succeed phase
      envelope calculation. If step sizes are too small, the algorithm
      may get stuck near the critical region, requiring larger step
      sizes to jump over it. In the opposite case, if step sizes are
      too big, then there is a risk that an initial guess generated by
      the cubic or linear extrapolation will not be good enough.

    switchmult: float
      The multiplier which is used to reduce the step size if the
      specified variable is switched. Default is `0.5`.

    unconvmult: float
      The multiplier which is used to reduce the step size if the solver
      for the phase envelope problem does not converge. Default is
      `0.75`.

    maxrepeats: int
      If the solver does not converge for any point of the phase
      envelope, the calculation can be repeated several times updating
      initial guesses of basic variables by reducing the step size.
      This parameter is used to restrict the number of repeats (step
      size cuts) for each point of the phase envelope. If the number
      of attempts exceeds the given limit, the calculation of the phase
      envelope will be stopped. Default is `8`.

    maxpoints: int
      The maximum number of points of the phase envelope. Default is
      `200`.

    Pmin: float
      The minimum pressure [Pa] for the phase envelope calculation
      routine. Default is `1.0` [Pa].

    Pmax: float
      The maximum pressure [Pa] for the phase envelope calculation
      routine. Default is `1e8` [Pa].

    Tmin: float
      The minimum temperature [K] for the phase envelope calculation
      routine. Default is `173.15` [K].

    Tmax: float
      The maximum temperature [K] for the phase envelope calculation
      routine. Default is `937.15` [K].

    **kwargs
      Other parameters for the internal initialization procedure.

    Returns
    -------
    The phase envelope as an instance of the `Envelope`.

    Raises
    ------
    Env2pConvergenceError
      This exception is raised if the solver of the phase envelope
      problem does not converge for the first point of the phase
      envelope.

    Notes
    -----
    The algorithm implemented by this method consists of the following
    steps:

    1. A vector of initial guesses for primary variables is obtained.
    For the available initialization options, see the description of
    the `init` parameter.

    2. To calculate the first point of the two-phase envelope, one
    of primary variables must be fixed. The index of this variable
    is defined either by the parameter `sidx0` or the internal rule
    (see the description of `sidx0`).

    3. The calculation of the first point of the two-phase envelope is
    performed. If the solver does not converge, the
    `Env2pConvergenceError` will be raised. To handle this exception
    at the first point, one should refine the initial guess. It can be
    done by increasing the number of nodes for the gridding procedure
    (see the parameter `init` and the docs of the `gridding` method).
    It may also be advisable to increase the maximum number of solver's
    iterations or change the starting temperature (especially if it
    close to the critical point).

    4. Once the first point of the phase envelope is calculated, the
    initial guess for the next one is obtained. For the second point
    of the phase envelope, the initialization procedure is based on the
    linear extrapolation taking into account:

      - the Jacobian that corresponds to the solution of the first
        point,
      - the sensitivity vector, which reveals the gradient information
        of the primary variables with respect to the specification
        variable of the first point,
      - the new specified variable corresponds to the maximum absolute
        item of the sensitivity vector,
      - the initial step size defined by the parameter `step0`, which is
        used to calculate the value of the new specification variable
        for the second point in the negative direction.

    It must be noted that, since the first point does not necessarily
    lie on the edge of the phase envelope, there are two directions
    relative to the first point for further phase envelope calculation.
    The phase envelope curve in the negative direction of the first
    specified variable will be produced first. Once the limit of
    pressure or temperature is met while drawing the "negative" curve
    of the phase envelope, if the condition on the maximum number of
    points allows the procedure to continue (see the parameter
    `maxpoints`), the phase envelope curve in the positive direction
    will be calculated.

    5. The second point of the phase envelope is calculated. If the
    solver does not converge, the step size is reduced using the
    `unconvmult` parameter until the convergence is achieved or the
    maximum number of step size cuts (parameter `maxrepeats`) is
    reached. If the calculation is completed successfully, the
    sensitivity vector and specification variable are both updated.

    6. The initialization procedure for subsequent points depends on the
    switch of the specification variable. If it does not change, the
    initial guess will be calculated using the third order extrapolation
    procedure and the new step size (obtained taking into account the
    number of solver iterations to converge and primary variable changes
    between two neighbour points of the phase envelope; for the details,
    see the parameters `fstep` and `maxstep`). Otherwise, the linear
    extrapolation is used with the reduced step size by applying the
    parameter `switchmult`.

    7. Critical points, cricondenbar, and cricondentherm of the phase
    envelope are calculated using the cubic interpolation taking into
    account conditions relevant for each special point.
    """
    if isinstance(init, tuple):
      P0, T0, kvi = init
    elif isinstance(init, State):
      T0 = init.T
      kvji = init.kvji
      if kvji is None:
        logger.info(
          'K-values obtained from the given state cannot be used to '
          'initialize the phase envelope routine because they are '
          '`None`.'
        )
        if phf > 0.5:
          phf = 1. - phf
        P0, kvi = cls.gridding(eos, yi, T0, phf, Pmin, Pmax, **kwargs)
      elif init.Np > 2:
        logger.info(
          'K-values obtained from the given state cannot be used to '
          'initialize the phase envelope routine because the number '
          'of phases is greater than two.'
        )
        if phf > 0.5:
          phf = 1. - phf
        P0, kvi = cls.gridding(eos, yi, T0, phf, Pmin, Pmax, **kwargs)
      else:
        P0 = init.P
        kvi = kvji.ravel()
    else:
      T0 = init
      if phf > 0.5:
        phf = 1. - phf
      P0, kvi = cls.gridding(eos, yi, T0, phf, Pmin, Pmax, **kwargs)
    xi = np_log(np_append(kvi, [P0, T0]))

    lnPmin = log(Pmin)
    lnPmax = log(Pmax)
    lnTmin = log(Tmin)
    lnTmax = log(Tmax)

    logger.info('Phase envelope for phase mole fraction = %.3f.', phf)
    Nc = eos.Nc
    Ncp2 = Nc + 2
    xki = np_zeros(shape=(maxpoints * 2, Ncp2))
    yvki = np_empty(shape=(maxpoints * 2, Nc))
    ylki = np_empty(shape=(maxpoints * 2, Nc))
    M = np_empty(shape=(4, 4))
    M[:, 0] = (1., 0., 1., 0.)
    M[1, 1] = 1.
    M[3, 1] = 1.
    B = np_empty(shape=(4, Ncp2))
    mdgds = np_zeros(shape=(Ncp2,))
    mdgds[-1] = 1.
    crits = []
    pmaxs: list[Vector[Float]] = []
    tmaxs: list[Vector[Float]] = []

    logger.info(
      '%4s%5s%6s%8s%5s%10s' + Nc * '%9s' + '%9s%8s',
      'Npnt', 'Ncut', 'Niter', 'Step', 'Sidx', 'Sval',
      *['lnkv%s' % s for s in range(Nc)], 'lnP', 'lnT',
    )
    tmpl = '%4s%5s%6s%8.4f%5s%10.4f' + Nc * '%9.4f' + '%9.4f%8.4f'

    c = 0
    cmax = maxpoints - 1
    k = cmax
    s0_idx: int | Integer
    if sidx0 is None:
      if phf > .5:
        s0_idx = np_argmax(kvi)
      else:
        s0_idx = np_argmin(kvi)
    else:
      s0_idx = sidx0
    s0 = xi[s0_idx]
    x0, yvi, yli, J0, nit = solver(eos, xi, s0_idx, s0, yi, phf)
    logger.info(tmpl, c, 0, nit, 0., s0_idx, s0, *x0)
    xki[k] = x0
    yvki[k] = yvi
    ylki[k] = yli
    dx0ds = lusolver(J0, mdgds)

    edgs = []
    for k_dir in [-1, 1]:
      k = cmax
      r = 0
      s_cnt = 1
      xk = x0
      Jk = J0
      dxkds = dx0ds
      sk_idx = s0_idx
      lnP = xk[-2]
      lnT = xk[-1]
      B[0] = xk
      B[1] = dxkds
      step = step0
      skp1_idx = np_argmax(np_abs(dxkds))
      sk = xk[skp1_idx]
      skp1 = sk + k_dir * step
      xi = xk + dxkds * (skp1 - sk)
      while (lnT >= lnTmin and lnT <= lnTmax and
             lnP >= lnPmin and lnP <= lnPmax and
             c < cmax):
        if r > maxrepeats:
          logger.warning('The maximum number of step cuts has been reached.')
          break
        try:
          xkp1, yvi, yli, Jkp1, nit = solver(eos, xi, skp1_idx, skp1, yi, phf)
          xkm1 = xk
          Jkm1 = Jk
          dxkm1ds = dxkds
          k += k_dir
          xk = xkp1
          Jk = Jkp1
          xki[k] = xk
          yvki[k] = yvi
          ylki[k] = yli
          c += 1
          lnP = xk[-2]
          lnT = xk[-1]
          logger.info(tmpl, c, r, nit, step, skp1_idx, skp1, *xk)
          r = 0
          dxkds = lusolver(Jk, mdgds)
          # dlnPdlnTk = dxkds[-2] / dxkds[-1]
          # if dlnPdlnTk * dlnPdlnTkm1 < 0.:
          #   ...
          if ((xk * xkm1)[:Nc] < 0.).all():
            if sk_idx == skp1_idx and skp1_idx < Nc:
              lnk_idx = sk_idx
              dlnPTkds = dxkds[Nc:]
              dlnPTkm1ds = dxkm1ds[Nc:]
            elif skp1_idx < Nc:
              lnk_idx = skp1_idx
              dlnPTkds = dxkds[Nc:]
              Jkm1[-1, sk_idx] = 0.
              Jkm1[-1, lnk_idx] = 1.
              dlnPTkm1ds = lusolver(Jkm1, mdgds)[Nc:]
            elif sk_idx < Nc:
              lnk_idx = sk_idx
              Jk[-1, skp1_idx] = 0.
              Jk[-1, lnk_idx] = 1.
              dlnPTkds = lusolver(Jk, mdgds)[Nc:]
              dlnPTkm1ds = dxkm1ds[Nc:]
            else:
              lnk_idx = 0
              Jk[-1, skp1_idx] = 0.
              Jk[-1, lnk_idx] = 1.
              dlnPTkds = lusolver(Jk, mdgds)[Nc:]
              Jkm1[-1, sk_idx] = 0.
              Jkm1[-1, lnk_idx] = 1.
              dlnPTkm1ds = lusolver(Jkm1, mdgds)[Nc:]
            lnkk_val = xk[lnk_idx]
            lnkkm1_val = xkm1[lnk_idx]
            cls._update_M(lnkk_val, lnkkm1_val, M)
            b = np_vstack([xk[Nc:], dlnPTkds, xkm1[Nc:], dlnPTkm1ds])
            crits.append(lusolver(M, b)[0])
          sk_idx = skp1_idx
          skp1_idx = np_argmax(np_abs(dxkds))
          if skp1_idx != sk_idx:
            s_cnt = 0
            step *= switchmult
          else:
            s_cnt += 1
            step *= fstep(nit, xk - xkm1)
            if abs(step) > maxstep:
              step = copysign(1., step) * maxstep
          skm1 = xkm1[skp1_idx]
          sk = xk[skp1_idx]
          cls._update_M(sk, skm1, M)
          cls._update_B(xk, dxkds, B)
          skp1 = sk + step * copysign(1., sk - skm1)
          if s_cnt > 1:
            C = lusolver(M, B)
            skp12 = skp1 * skp1
            skp13 = skp12 * skp1
            xi = np_array([1., skp1, skp12, skp13]).dot(C)
          else:
            xi = xk + (xk - xkm1) / (sk - skm1) * (skp1 - sk)
        except EnvConvergenceError:
          step *= unconvmult
          if s_cnt > 1:
            skp1 = sk + step * copysign(1., sk - skm1)
            skp12 = skp1 * skp1
            skp13 = skp12 * skp1
            xi = np_array([1., skp1, skp12, skp13]).dot(C)
          elif c > 0:
            skp1 = sk + step * copysign(1., sk - skm1)
            xi = xk + (xk - xkm1) / (sk - skm1) * (skp1 - sk)
          else:
            skp1 = sk + k_dir * step
            xi = xk + dxkds * (skp1 - sk)
          r += 1
      edgs.append(k)
    return cls.outputPT(
      eos, yi, phf, xki, yvki, ylki, crits, pmaxs, tmaxs, *edgs,
    )

  @staticmethod
  def _update_M(
    sk: float,
    skm1: float,
    M: Matrix[Float],
  ) -> None:
    skm12 = skm1 * skm1
    sk2 = sk * sk
    M[0, 1] = sk
    M[0, 2] = sk2
    M[0, 3] = sk2 * sk
    M[1, 2] = 2. * sk
    M[1, 3] = 3. * sk2
    M[2, 1] = skm1
    M[2, 2] = skm12
    M[2, 3] = skm12 * skm1
    M[3, 2] = 2. * skm1
    M[3, 3] = 3. * skm12
    pass

  @staticmethod
  def _update_B(
    xk: Vector[Float],
    dxkds: Vector[Float],
    B: Matrix[Float],
  ) -> None:
    B[2:] = B[:2]
    B[0] = xk
    B[1] = dxkds
    pass

  @staticmethod
  def gridding(
    eos: PsatPTEos,
    yi: Vector[Float],
    T: float,
    phf: float,
    Pmin: float = 1.,
    Pmax: float = 1e8,
    Nnodes: int = 100,
    psatroutine: PsatRoutine[PsatPTEos] = psat.runPT,
    flashroutine: FlashRoutine[Flash2pPTEos] = flash.run2pPT,
  ) -> tuple[float, Vector[Float]]:
    """Perform the gridding procedure to find a state, a phase mole
    fraction of which is close to the specified phase mole fraction.

    Parameters
    ----------
    eos: PsatPTEos
      An initialized instance of a PT-based equation of state that
      can be used to calculate saturation pressure of a mixture.

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of components.

    phf: float
      Phase mole fraction, close to which the initial guess is needed.
      It must be less than or equal to `0.5`.

    Pmin: float
      The lower bound of the pressure interval. Default is `1.0` [Pa].

    Pmax: float
      The upper bound of the pressure interval. Default is `1e8` [Pa].

    Nnodes: int
      The number of grid nodes. Default is `100`.

    psatroutine: PsatRoutine[PsatPTEos]
      A callable object that can be used to find a saturation state of
      a mixture for a given temperature and mole composition. Default
      is `psat.runPT`. `Pmin` and `Pmax` are passed to a saturation
      pressure calculation procedure.

    flashroutine: FlashRoutine[Flash2pPTEos]
      A callable object that can be used to perform two-phase flash
      routine for a given pressure, temperature, and mole composition
      of a mixture. Default is `flash.run2pPT`.

    Returns
    -------
    A tuple containing:
    - pressure [Pa] at which a given composition of a mixture separates
      into phases, resulting in a phase mole fraction that is close to
      the specified value,
    - corresponding k-values of components as a `Vector[Float]` of shape
      `(Nc,)`.
    """
    satstate = psatroutine(eos, T, yi, 1., True, None, Pmin=Pmin, Pmax=Pmax)
    Ps = satstate.P
    logger.debug('Saturation pressure (T = %.2f [K]): %.1f [Pa].', T, Ps)
    if phf == 0.:
      return Ps, satstate.kvji.ravel()
    PP = np_linspace(Ps, Pmin, Nnodes, endpoint=True)
    res: tuple[float, Vector[Float]]
    res = satstate.P, satstate.kvji.ravel()
    df = phf
    state: State = satstate
    for P in PP[1:]:
      try:
        state = flashroutine(eos, P, T, yi, 1., state)
        kvji = state.kvji
        if kvji is not None:
          f = state.fj[0]
          logger.debug(
            'Non-reference phase mole fraction (P = %.1f [Pa]): %.4f.', P, f,
          )
          df0 = abs(f - phf)
          if df0 < df:
            df = df0
            res = P, kvji.ravel()
        else:
          break
      except (FlashConvergenceError, RRConvergenceError):
        continue
    return res

  @staticmethod
  def outputPT(
    eos: StateNpPTEos,
    yi: Vector[Float],
    phf: float,
    xki: Matrix[Float],
    yvki: Matrix[Float],
    ylki: Matrix[Float],
    crits: list[Vector[Float]],
    pmaxs: list[Vector[Float]],
    tmaxs: list[Vector[Float]],
    idx0: int,
    idx1: int,
  ) -> Envelope:
    Nc = eos.Nc
    slc = slice(idx0, idx1 + 1)
    P = np_exp(xki[slc, -2])
    Ns = P.shape[0]
    T = np_exp(xki[slc, -1])
    kvji = np_exp(xki[slc, :Nc])[:,None,:]
    yvi = yvki[slc]
    yli = ylki[slc]
    yji = np_stack([yvi, yli], axis=1)
    Zv = eos.getPT_Zj(P, T, yvi)
    Zl = eos.getPT_Zj(P, T, yli)
    Zj = np_column_stack([Zv, Zl])
    vj = Zj * (R * T / P)[:,None]
    dj = yji.dot(eos.mwi) / vj
    pidv = eos.getPT_PIDj(P, T, yvi)
    pidl = eos.getPT_PIDj(P, T, yli)
    pidj = np_column_stack([pidv, pidl])
    n = np_full_like(P, 1.)
    ni = np_full((Ns, Nc), yi)
    _phf = np_full_like(P, phf)
    fj = np_column_stack([_phf, 1. - _phf])
    nj = fj
    nji = nj[:,:,None] * yji
    Vj = nj * vj
    V = Vj.sum(axis=1)
    sj = Vj / V[:,None]
    if crits:
      _crits = np_exp(crits)
      Pc = _crits[:, 0]
      Tc = _crits[:, 1]
    else:
      Pc = None
      Tc = None
    if pmaxs:
      _pmaxs = np_exp(pmaxs)
      Pcb = _pmaxs[:, 0]
      Tcb = _pmaxs[:, 1]
    else:
      Pcb = None
      Tcb = None
    if tmaxs:
      _tmaxs = np_exp(tmaxs)
      Pct = _tmaxs[:, 0]
      Tct = _tmaxs[:, 1]
    else:
      Pct = None
      Tct = None
    return Envelope(
      Ns, Nc, 2, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
      kvji, Pc, Tc, Pcb, Tcb, Pct, Tct,
    )
