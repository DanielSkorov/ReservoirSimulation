from logging import (
  getLogger,
)

from typing import (
  Iterable,
  cast,
)

from math import (
  isfinite,
)

from numpy import (
  abs as np_abs,
  append as np_append,
  argsort as np_argsort,
  array as np_array,
  atleast_2d as np_atleast_2d,
  empty as np_empty,
  exp as np_exp,
  fill_diagonal as np_fill_diagonal,
  full as np_full,
  full_like as np_full_like,
  log as np_log,
)

from numpy.lib.stride_tricks import (
  as_strided as np_as_strided,
)

from resim.pvt.datatypes import (
  Float,
  Matrix,
  MultiPhaseState,
  OnePhaseState,
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
  State2pPTEos,
  StateNpPTEos,
)

from resim.pvt.stab import (
  StabPTEos,
  StabRoutine,
  StabSolver,
  StabSolverPTEos,
  _stabPT_qnssnewt,
  stabtest,
)

from resim.pvt.flash.protocols import (
  Flash2pPTEos,
  Flash2pSolver,
  Flash2pSolverPTEos,
  FlashNpPTEos,
  FlashNpSolver,
  FlashNpSolverPTEos,
  RR2pSolver,
  RRNpSolver,
)

from resim.pvt.flash.rr import (
  RRConvergenceError,
  rr2p_fgh,
  rrNp,
)


logger = getLogger('flash')


class FlashConvergenceError(Exception):
  """An exception that will be raised if a solver for flash calculations
  does not converge.
  """
  def __init__(
    self,
    msg: str = ('The flash calculation procedure was\nterminated '
                'unsuccessfully. Try to improve initial guesses. '
                'It may\nalso be advisable to change the solver '
                'or increase the number of\niterations.'),
  ) -> None:
    super().__init__(msg)
    pass


class OnePhaseStateError(Exception):
  """Internal two-phase flash solvers perform the stability test
  during the solution of a system of non-linear equations. This
  exception will be raised if a one-phase state is determined to be
  stable at any iteration of those solvers.
  """
  def __init__(
    self,
    msg: str = ('During two-phase flash calculations, the stability test '
                'shows that the one-phase state is stable. To complete '
                'this flash, one should set `stabsolver` to `None`.'),
  ) -> None:
    super().__init__(msg)
    pass


class TrivialSolutionError(Exception):
  """This exception will be raised if an internal flash solver converges
  to the trivial solution, which is detected by monitoring for identical
  phase compositions.
  """
  def __init__(
    self,
    msg: str = ('During flash calculations, convergence to the trivial '
                'solution was detected. Try to change the `trivtol` '
                'keyword or revise the initial guess of k-values.')
  ) -> None:
    super().__init__(msg)
    pass


def _flash2pPT_ssnewt(
  eos: Flash2pSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 200,
  trivtol: float = 1e-4,
  stabeps: float = -1e-8,
  maxstabchecks: int = 1,
  switchers: tuple[float, float, float, float] = (0.6, 1e-2, 1e-12, 1e-6),
  rrsolver: RR2pSolver = rr2p_fgh,
  stabsolver: StabSolver[StabSolverPTEos] | None = _stabPT_qnssnewt,
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float], Vector[Float], Vector[Float]]:
  r"""Solve the two-phase flash problem formulated for the PT-
  thermodynamics using successive substitution iterations and Newton's
  method.

  Parameters
  ----------
  eos: Flash2pSolverPTEos
    An initialized instance of a PT-based equation of state.

  P: float
    Pressure [Pa].

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the vector of equilibrium equations (residuals) is less than
    `tolres`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `200`.

  trivtol: float
    If `stabsolver` is `None`, this parameter is used for an early
    check of the solver's convergence to the trivial solution. At any
    iteration of the successive substitution method, if absolute values
    of natural logarithms of k-values are all less than `trivtol`, the
    error will be raised. Default is `1e-4`.

  stabeps: float
    The one-phase state will be considered stable if the value of the
    tangent-plane distance function obtained from the `stabsolver` is
    greater than `stabeps`. Default is `-1e-8`.

  maxstabcheks: int
    The maximum number of stability tests performed during solving the
    two-phase flash problem. Default is `1`.

  switchers: tuple[float, float, float, float]
    Allows to modify the conditions of switching from successive
    substitution iterations to Newton's method. The parameter must be
    represented as a tuple containing four positive values:
    :math:`\eps_r`, :math:`\eps_f`, :math:`\eps_l`, :math:`\eps_u`.
    The switching conditions are:

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
    conditions were taken from the paper of L.X. Nghiem et al, 1983
    (doi: 10.2118/8285-PA). Default is `(0.6, 1e-2, 1e-12, 1e-6)`.

  rrsolver: RR2pSolver
    A callable object that can be used to solve the Rachford-Rice
    equation. Default is `rr2p_fgh`.

  stabsolver: StabPTSolver | None
    A callable object that can be used to solve one-phase stability
    problems based on the Gibbs energy analysis (PT-thermodynamics).
    If it is set to `None`, then the stability test will not be
    performed, which turns on the negative flash procedure. Default is
    `_stabPT_qnssnewt`.

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `(Nc, Nc)` and a `b: Vector[Float]` of shape `(Nc,)` and finds
    `x: Vector[Float]` of shape `(Nc,)`, which is the solution to the
    linear system :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`. The
    matrix `A` is a symmetric matrix and usually positive definite.
    Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - mole fraction of the non-reference phase,
  - a `Vector[Float]` of shape `(Nc,)` of k-values of components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the non-reference phase,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the reference phase.

  Raises
  ------
  FlashConvergenceError
    This exception is raised if the solver does not converge.

  RRConvergenceError
    This exception is raised if the solver of the Rachford-Rice
    equation does not converge.

  OnePhaseStateError
    This exception is raised when the stability test shows that the
    one-phase state is stable. To disable raising this error, one
    should turn on the negative flash by setting `stabsolver` to
    `None`.

  TrivialSolutionError
    This exception will be raised if the solver converges to the trivial
    solution, which is detected by monitoring for absolute values of
    natural logarithms of k-values.

  Notes
  -----
  Details of the implemented algorithm:

  - Successive substitution iterations precede Newton's method to
    improve the initial guess of k-values.

  - During successive substitution iterations, if an unphysical value of
    the non-reference phase mole fraction is obtained from the solver of
    the Rachford-Rice equation, the stability test will be conducted to
    assess the stability of the one-phase state. If the one-phase state
    is determined to be unstable, k-values from the stability test will
    be used to continue solving the flash problem. Otherwise, the error
    will be raised. This procedure enables the solver to converge to a
    solution in the positive flash window. However, one can set the
    `stabsolver` keyword to `None` to perform the negative flash. In
    this case, the convergence to the trivial solution is checked by
    monitoring absolute values of natural logarithms of k-values.

  - The Rachford-Rice equation is solved in the inner loop of Newton's
    method. This approach is faster because the symmetry of the Hessian
    matrix can be exploited with Cholesky decomposition when solving
    the system of linear equations. For the details, see the paper of
    M. Petitfrere, D.V. Nichita, 2016 (doi: 1016/j.fluid.2016.06.050).

  - In general, the Cholesky decomposition can be used to solve the
    system of linear equations. This is because Newton's method is
    applied only after the switching criteria, which contain the
    convergence metric, are satisfied. However, the use of the modified
    Cholesky decomposition is still recommended. The switching criteria
    do not guarantee the positive definiteness of the Hessian matrix.
    Because the modified Cholesky decomposition is not implemented in
    the `numpy` library, the `numpy.linalg.solve` is used instead by
    default.

  - The solver can be transformed to either the successive substitution
    method or Newton's method by changing the switching criteria.
  """
  logger.info(
    'Solving the two-phase flash problem using the SS-Newton solver.'
  )
  Nc = eos.Nc
  logger.info('P = %.1f [Pa], T = %.2f [K], yi =' + Nc * '%7.4f', P, T, *yi)
  tmpl = '%3s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  stabchecks = 0
  k = 0
  kvi = kvi0
  lnkvi = np_log(kvi)
  f0 = rrsolver(kvi, yi, None)
  outside_pfw = f0 <= 0. or f0 >= 1.
  if outside_pfw and stabchecks < maxstabchecks:
    if stabsolver is None:
      if (np_abs(lnkvi) < trivtol).all():
        logger.debug('The solver converged to the trivial solution.')
        raise TrivialSolutionError()
    else:
      logger.debug('Non-reference phase mole fraction: %.3e.', f0)
      TPD, kvi = stabsolver(eos, P, T, yi, kvi)
      logger.debug('TPD = %.3e.', TPD)
      if TPD > stabeps:
        logger.debug('The one-phase state is stable: True.')
        raise OnePhaseStateError()
      else:
        f0 = rrsolver(kvi, yi, None)
        outside_pfw = f0 <= 0. or f0 >= 1.
        stabchecks += 1
  y1i = yi / ((kvi - 1.) * f0 + 1.)
  y0i = y1i * kvi
  lnphi1i = eos.getPT_lnphii(P, T, y1i)
  lnphi0i = eos.getPT_lnphii(P, T, y0i)
  gi = lnkvi + lnphi0i - lnphi1i
  g2 = gi.dot(gi)
  use_ss = g2 > epsu or g2 < epsl or outside_pfw
  logger.debug(
    '%3s' + Nc * '%10s' + '%9s%11s%8s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2', 'method',
  )
  logger.debug(tmpl, k, *lnkvi, f0, g2, 'SS')
  while use_ss and g2 > tol and k < maxiter:
    k += 1
    lnkvi -= gi
    kvi = np_exp(lnkvi)
    f0km1 = f0
    f0 = rrsolver(kvi, yi, f0)
    outside_pfw = f0 <= 0. or f0 >= 1.
    if outside_pfw and stabchecks < maxstabchecks:
      if stabsolver is None:
        if (np_abs(lnkvi) < trivtol).all():
          logger.debug('The solver converged to the trivial solution.')
          raise TrivialSolutionError()
      else:
        logger.debug('Non-reference phase mole fraction: %.3e.', f0)
        TPD, kvi = stabsolver(eos, P, T, yi, kvi)
        logger.debug('TPD = %.3e.', TPD)
        if TPD > stabeps:
          logger.debug('The one-phase state is stable: True.')
          raise OnePhaseStateError()
        else:
          f0 = rrsolver(kvi, yi, None)
          lnkvi = np_log(kvi)
          outside_pfw = f0 <= 0. or f0 >= 1.
          stabchecks += 1
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    lnphi1i = eos.getPT_lnphii(P, T, y1i)
    lnphi0i = eos.getPT_lnphii(P, T, y0i)
    gi = lnkvi + lnphi0i - lnphi1i
    g2km1 = g2
    g2 = gi.dot(gi)
    use_ss = (outside_pfw
              or g2 < epsr * g2km1
              or g2 > epsu
              or g2 < epsl
              or f0 - f0km1 > epsf
              or f0km1 - f0 > epsf)
    logger.debug(tmpl, k, *lnkvi, f0, g2, 'SS')
  if isfinite(g2):
    if g2 > tol and k < maxiter:
      U = np_full(shape=(Nc, Nc), fill_value=-1.)
      f1 = 1. - f0
      lnphi1i, dlnphi1idnj = eos.getPT_lnphii_dnj(P, T, y1i, f1)
      lnphi0i, dlnphi0idnj = eos.getPT_lnphii_dnj(P, T, y0i, f0)
      logger.debug(tmpl, k, *lnkvi, f0, g2, 'Newt')
      while g2 > tol and k < maxiter:
        ui = yi / (y1i * y0i) - 1.
        F0F1 = 1. / (f0 * f1)
        np_fill_diagonal(U, ui)
        H = U * F0F1 + (dlnphi0idnj + dlnphi1idnj)
        # TODO: Replace the LU-solver with the custom implementation
        #       of the modified Cholesky decomposition solver.
        dn0i = linsolver(H, -gi)
        dlnkvi = U.dot(dn0i) * F0F1
        k += 1
        lnkvi += dlnkvi
        kvi = np_exp(lnkvi)
        f0 = rrsolver(kvi, yi, f0)
        f1 = 1. - f0
        y1i = yi / ((kvi - 1.) * f0 + 1.)
        y0i = y1i * kvi
        lnphi1i, dlnphi1idnj = eos.getPT_lnphii_dnj(P, T, y1i, f1)
        lnphi0i, dlnphi0idnj = eos.getPT_lnphii_dnj(P, T, y0i, f0)
        gi = lnkvi + lnphi0i - lnphi1i
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *lnkvi, f0, g2, 'Newt')
      if g2 < tol and isfinite(g2):
        return f0, kvi, y0i, y1i
    elif g2 < tol:
      return f0, kvi, y0i, y1i
  logger.warning(
    'Two-phase flash calculations completed unsuccessfully.\n'
    'The solver was "_flash2pPT_ssnewt".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvi0 = %s',
    eos.name, P, T, yi.tolist(), kvi0.tolist(),
  )
  raise FlashConvergenceError()


def _flash2pPT_qnssnewt(
  eos: Flash2pSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 100,
  lmbdmax: float = 6.,
  trivtol: float = 1e-4,
  stabeps: float = -1e-8,
  maxstabchecks: int = 1,
  switchers: tuple[float, float, float, float] = (0.6, 1e-2, 1e-12, 1e-6),
  rrsolver: RR2pSolver = rr2p_fgh,
  stabsolver: StabSolver[StabSolverPTEos] | None = _stabPT_qnssnewt,
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float], Vector[Float], Vector[Float]]:
  r"""Solve the two-phase flash problem formulated for the PT-
  thermodynamics using quasi-newton successive substitution iterations
  (QNSS) and Newton's method.

  Parameters
  ----------
  eos: Flash2pSolverPTEos
    An initialized instance of a PT-based equation of state.

  P: float
    Pressure [Pa].

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the vector of equilibrium equations (residuals) is less than
    `tolres`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  lmbdmax: float
    The maximum step length. Default is `6.0`.

  trivtol: float
    If `stabsolver` is `None`, this parameter is used for an early
    check of the solver's convergence to the trivial solution. At any
    iteration of the QNSS-method, if absolute values of natural
    logarithms of k-values are all less than `trivtol`, the error will
    be raised. Default is `1e-4`.

  stabeps: float
    The one-phase state will be considered stable if the value of the
    tangent-plane distance function obtained from the `stabsolver` is
    greater than `stabeps`. Default is `-1e-8`.

  maxstabcheks: int
    The maximum number of stability tests performed during solving the
    two-phase flash problem. Default is `1`.

  switchers: tuple[float, float, float, float]
    Allows to modify the conditions of switching from the QNSS to
    Newton's method. The parameter must be represented as a tuple
    containing four positive values: :math:`\eps_r`, :math:`\eps_f`,
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
    conditions were taken from the paper of L.X. Nghiem et al, 1983
    (doi: 10.2118/8285-PA). Default is `(0.6, 1e-2, 1e-12, 1e-6)`.

  rrsolver: RR2pSolver
    A callable object that can be used to solve the Rachford-Rice
    equation. Default is `rr2p_fgh`.

  stabsolver: StabPTSolver | None
    A callable object that can be used to solve one-phase stability
    problems based on the Gibbs energy analysis (PT-thermodynamics).
    If it is set to `None`, then the stability test will not be
    performed, which turns on the negative flash procedure. Default is
    `_stabPT_qnssnewt`.

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `(Nc, Nc)` and a `b: Vector[Float]` of shape `(Nc,)` and finds
    `x: Vector[Float]` of shape `(Nc,)`, which is the solution to the
    linear system :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`. The
    matrix `A` is a symmetric matrix and usually positive definite.
    Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - mole fraction of the non-reference phase,
  - a `Vector[Float]` of shape `(Nc,)` of k-values of components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the non-reference phase,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the reference phase.

  Raises
  ------
  FlashConvergenceError
    This exception is raised if the solver does not converge.

  RRConvergenceError
    This exception is raised if the solver of the Rachford-Rice
    equation does not converge.

  OnePhaseStateError
    This exception is raised when the stability test shows that the
    one-phase state is stable. To disable raising this error, one
    should turn on the negative flash by setting `stabsolver` to
    `None`.

  TrivialSolutionError
    This exception will be raised if the solver converges to the trivial
    solution, which is detected by monitoring for absolute values of
    natural logarithms of k-values.

  Notes
  -----
  Details of the implemented algorithm:

  - The QNSS-method precedes Newton's method to improve the initial
    guess of k-values.

  - During QNSS iterations, if an unphysical value of the non-reference
    phase mole fraction is obtained from the solver of the Rachford-Rice
    equation, the stability test will be conducted to assess the
    stability of the one-phase state. If the one-phase state is
    determined to be unstable, k-values from the stability test will be
    used to continue solving the flash problem. Otherwise, the error
    will be raised. This procedure enables the solver to converge to a
    solution in the positive flash window. However, one can set the
    `stabsolver` keyword to `None` to perform the negative flash. In
    this case, the convergence to the trivial solution is checked by
    monitoring absolute values of natural logarithms of k-values.

  - For other numerical details of the QNSS-method, see the paper of
    L.X. Nghiem and Y.-K. Li, 1984 (doi: 10.1016/0378-3812(84)80013-8).

  - The Rachford-Rice equation is solved in the inner loop of Newton's
    method. This approach is faster because the symmetry of the Hessian
    matrix can be exploited with Cholesky decomposition when solving
    the system of linear equations. For the details, see the paper of
    M. Petitfrere, D.V. Nichita, 2016 (doi: 1016/j.fluid.2016.06.050).

  - In general, the Cholesky decomposition can be used to solve the
    system of linear equations. This is because Newton's method is
    applied only after the switching criteria, which contain the
    convergence metric, are satisfied. However, the use of the modified
    Cholesky decomposition is still recommended. The switching criteria
    do not guarantee the positive definiteness of the Hessian matrix.
    Because the modified Cholesky decomposition is not implemented in
    the `numpy` library, the `numpy.linalg.solve` is used instead by
    default.

  - The solver can be transformed to either the quasi-newton successive
    substitution method (QNSS-method) or Newton's method by changing
    the switching criteria.
  """
  logger.info(
    'Solving the two-phase flash problem using the QNSS-Newton solver.'
  )
  Nc = eos.Nc
  logger.info('P = %.1f [Pa], T = %.2f [K], yi =' + Nc * '%7.4f', P, T, *yi)
  tmpl = '%3s' + Nc * '%10.4f' + '%9.4f%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  stabchecks = 0
  k = 0
  kvi = kvi0
  lnkvi = np_log(kvi)
  f0 = rrsolver(kvi, yi, None)
  outside_pfw = f0 <= 0. or f0 >= 1.
  if outside_pfw and stabchecks < maxstabchecks:
    if stabsolver is None:
      if (np_abs(lnkvi) < trivtol).all():
        logger.debug('The solver converged to the trivial solution.')
        raise TrivialSolutionError()
    else:
      logger.debug('Non-reference phase mole fraction: %.3e.', f0)
      TPD, kvi = stabsolver(eos, P, T, yi, kvi)
      logger.debug('TPD = %.3e.', TPD)
      if TPD > stabeps:
        logger.debug('The one-phase state is stable: True.')
        raise OnePhaseStateError()
      else:
        f0 = rrsolver(kvi, yi, None)
        outside_pfw = f0 <= 0. or f0 >= 1.
        stabchecks += 1
  y1i = yi / ((kvi - 1.) * f0 + 1.)
  y0i = y1i * kvi
  lnphi1i = eos.getPT_lnphii(P, T, y1i)
  lnphi0i = eos.getPT_lnphii(P, T, y0i)
  gi = lnkvi + lnphi0i - lnphi1i
  g2 = gi.dot(gi)
  lmbd = 1.
  dlnkvi = -gi
  use_qnss = g2 > epsu or g2 < epsl or outside_pfw
  logger.debug(
    '%3s' + Nc * '%10s' + '%9s%11s%8s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'f0', 'g2', 'method',
  )
  logger.debug(tmpl, k, *lnkvi, f0, g2, 'QNSS')
  while use_qnss and g2 > tol and k < maxiter:
    k += 1
    tkm1 = dlnkvi.dot(gi)
    lnkvi += dlnkvi
    kvi = np_exp(lnkvi)
    f0km1 = f0
    f0 = rrsolver(kvi, yi, f0)
    outside_pfw = f0 <= 0. or f0 >= 1.
    if outside_pfw and stabchecks < maxstabchecks:
      if stabsolver is None:
        if (np_abs(lnkvi) < trivtol).all():
          logger.debug('The solver converged to the trivial solution.')
          raise TrivialSolutionError()
      else:
        logger.debug('Non-reference phase mole fraction: %.3e.', f0)
        TPD, kvi = stabsolver(eos, P, T, yi, kvi)
        logger.debug('TPD = %.3e.', TPD)
        if TPD > stabeps:
          logger.debug('The one-phase state is stable: True.')
          raise OnePhaseStateError()
        else:
          f0 = rrsolver(kvi, yi, None)
          lnkvi = np_log(kvi)
          outside_pfw = f0 <= 0. or f0 >= 1.
    y1i = yi / ((kvi - 1.) * f0 + 1.)
    y0i = y1i * kvi
    lnphi1i = eos.getPT_lnphii(P, T, y1i)
    lnphi0i = eos.getPT_lnphii(P, T, y0i)
    gi = lnkvi + lnphi0i - lnphi1i
    g2km1 = g2
    g2 = gi.dot(gi)
    use_qnss = (outside_pfw
                or g2 < epsr * g2km1
                or g2 > epsu
                or g2 < epsl
                or f0 - f0km1 > epsf
                or f0km1 - f0 > epsf)
    if k % Nc == 0:
      lmbd = 1.
      dlnkvi = -gi
    else:
      lmbd *= tkm1 / (dlnkvi.dot(gi) - tkm1)
      if lmbd < 0.:
        lmbd = -lmbd
      if lmbd > lmbdmax:
        lmbd = lmbdmax
      dlnkvi = -lmbd * gi
      max_dlnkvi = np_abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
    logger.debug(tmpl, k, *lnkvi, f0, g2, 'QNSS')
  if isfinite(g2):
    if g2 > tol and k < maxiter:
      U = np_full(shape=(Nc, Nc), fill_value=-1.)
      f1 = 1. - f0
      lnphi1i, dlnphi1idnj = eos.getPT_lnphii_dnj(P, T, y1i, f1)
      lnphi0i, dlnphi0idnj = eos.getPT_lnphii_dnj(P, T, y0i, f0)
      logger.debug(tmpl, k, *lnkvi, f0, g2, 'Newt')
      while g2 > tol and k < maxiter:
        ui = yi / (y1i * y0i) - 1.
        F0F1 = 1. / (f0 * f1)
        np_fill_diagonal(U, ui)
        H = U * F0F1 + (dlnphi0idnj + dlnphi1idnj)
        # TODO: Replace the LU-solver with the custom implementation
        #       of the modified Cholesky decomposition solver.
        dn0i = linsolver(H, -gi)
        dlnkvi = U.dot(dn0i) * F0F1
        k += 1
        lnkvi += dlnkvi
        kvi = np_exp(lnkvi)
        f0 = rrsolver(kvi, yi, f0)
        f1 = 1. - f0
        y1i = yi / ((kvi - 1.) * f0 + 1.)
        y0i = y1i * kvi
        lnphi1i, dlnphi1idnj = eos.getPT_lnphii_dnj(P, T, y1i, f1)
        lnphi0i, dlnphi0idnj = eos.getPT_lnphii_dnj(P, T, y0i, f0)
        gi = lnkvi + lnphi0i - lnphi1i
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *lnkvi, f0, g2, 'Newt')
      if g2 < tol and isfinite(g2):
        return f0, kvi, y0i, y1i
    elif g2 < tol:
      return f0, kvi, y0i, y1i
  logger.warning(
    'Two-phase flash calculations completed unsuccessfully.\n'
    'The solver was "_flash2pPT_qnssnewt".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvi0 = %s',
    eos.name, P, T, yi.tolist(), kvi0.tolist(),
  )
  raise FlashConvergenceError()


def _flashNpPT_ssnewt(
  eos: FlashNpSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  fj0: Vector[Float],
  kvji0: Matrix[Float],
  tol: float = 1e-20,
  maxiter: int = 200,
  trivtol: float = 1e-4,
  switchers: tuple[float, float, float, float] = (0.6, 1e-2, 1e-12, 1e-6),
  rrsolver: RRNpSolver = rrNp,
  linsolver: LinearSolver = lusolver,
) -> tuple[Vector[Float], Matrix[Float], Matrix[Float], Vector[Float]]:
  r"""Solve the multiphase flash problem formulated for the PT-
  thermodynamics using successive substitution iterations and Newton's
  method.

  Parameters
  ----------
  eos: FlashNpSolverPTEos
    An initialized instance of a PT-based equation of state.

  P: float
    Pressure [Pa].

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  fj0: Vector[Float], shape (Np - 1,)
    An initial guess of mole fractions of non-reference phases.

  kvji0: Matrix[Float], shape (Np - 1, Nc)
    An initial guess of k-values of non-reference phases.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the vector of equilibrium equations is less than `tol`. Default
    is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `200`.

  trivtol: float
    This parameter is used for an early check of the solver's
    convergence to the trivial solution. At any iteration of the
    successive substitution method, the error will be raised if any of
    the following conditions is satisfied:

    - absolute values of natural logarithms of k-values of any non-
      reference phase are all less than `trivtol`,

    - absolute values of differences of natural logarithms of k-values
      between any non-reference phases are all less than `trivtol`.

    Default is `1e-4`.

  switchers: tuple[float, float, float, float]
    Allows to modify the conditions of switching from the successive
    substitution method to Newton's method. The parameter must be
    represented as a tuple containing four positive values:
    :math:`\eps_r`, :math:`\eps_f`, :math:`\eps_l`, :math:`\eps_u`.
    The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \max_j \left| f_j^k - f_j^{k-1} \right| < \eps_f, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
        0 < f_j^k < 1, \; j = 1 \, \ldots \, N_p - 1,
      \end{cases}

    where :math:`\mathbf{g}` is the equilibrium equations vector,
    :math:`k` is the iteration number, :math:`f_j` is the mole fraction
    of a non-reference phase :math:`j`. Analytical expressions of the
    switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.6, 1e-2, 1e-12, 1e-6)`.

  rrsolver: RRNpSolver
    A callable object that can be used to solve the system of
    Rachford-Rice equations. Default is `rrNp`.

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `((Np - 1) * Nc, (Np - 1) * Nc)` and a `b: Vector[Float]` of shape
    `((Np - 1) * Nc,)` and finds `x: Vector[Float]` of shape
    `((Np - 1) * Nc,)`, which is the solution to the linear system
    :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`. The matrix `A` is
    a symmetric matrix and usually positive definite. Default is
    `lusolver`.

  Returns
  -------
  A tuple containing:
  - a `Vector[Float]` of shape `(Np - 1,)` of mole fractions of non-
    reference phases,
  - a `Matrix[Float]` of shape `(Np - 1, Nc)` of k-values of components
    in non-reference phases,
  - a `Matrix[Float]` of shape `(Np - 1, Nc)` of mole fractions of
    components in non-reference phases,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the reference phase.

  Raises
  ------
  FlashConvergenceError
    This exception is raised if the solver does not converge.

  RRConvergenceError
    This exception is raised if the solver of the system of
    Rachford-Rice equations does not converge.

  TrivialSolutionError
    This exception will be raised if the solver converges to the trivial
    solution, which is detected by monitoring conditions described in
    the explanation for the `trivtol` parameter.

  Notes
  -----
  Details of the implemented algorithm

  - Successive substitution iterations precede Newton's method to
    improve the initial guess of k-values.

  - Because this solver does not conduct a stability test procedure
    when an unphysical value of mole fraction of any non-reference
    phase is yilded from the solver of Rachford-Rice equations, it is
    recommended to first perform a stability test and find the global
    minimum of the TPD-function. The stability test procedure can also
    provide initial guesses for the mole fractions of non-reference
    phases and k-values of components. For the details, see the paper of
    Z. Li and A. Firoozabadi, 2012 (doi: 10.1016/j.fluid.2012.06.021).

  - The solver's convergence to the trivial solution is checked by
    monitoring absolute values of natural logarithms of k-values and
    their differences between non-reference phases.

  - The system of Rachford-Rice equations is solved in the inner loop
    of Newton's method. This approach is faster because the symmetry
    of the Hessian matrix can be exploited with Cholesky decomposition
    when solving the system of linear equations. For the details, see
    the paper of M. Petitfrere and D.V. Nichita, 2016 (doi:
    1016/j.fluid.2016.06.050).

  - In general, the Cholesky decomposition can be used to solve the
    system of linear equations. This is because Newton's method is
    applied only after the switching criteria, which contain the
    convergence metric, are satisfied. However, the use of the modified
    Cholesky decomposition is still recommended. The switching criteria
    do not guarantee the positive definiteness of the Hessian matrix.
    Because the modified Cholesky decomposition is not implemented in
    the `numpy` library, the `numpy.linalg.solve` is used instead by
    default.

  - The solver can be transformed to either the successive substitution
    method or Newton's method by changing the switching criteria.
  """
  Nc = eos.Nc
  Npm1 = fj0.shape[0]
  Npm1Nc = Npm1 * Nc
  logger.info(
    'Solving the %s-phase flash problem using the SS-Newton solver.',
    Npm1 + 1,
  )
  logger.info('P = %.1f [Pa], T = %.2f [K], yi =' + Nc * '%7.4f', P, T, *yi)
  tmpl = '%3s' + Npm1Nc * '%10.4f' + Npm1 * '%9.4f' + '%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  k = 0
  kvji = kvji0
  fj = fj0
  lnkvji = np_log(kvji)
  lnkvi = lnkvji.ravel()
  fj = rrsolver(kvji, yi, fj)
  outside_pfw = (fj <= 0.).any() or (fj >= 1.).any()
  if outside_pfw:
    if (np_abs(lnkvji) < trivtol).all(axis=1).any():
      logger.debug('The solver converged to the trivial solution.')
      raise TrivialSolutionError()
    for r in range(Npm1):
      for s in range(r + 1, Npm1):
        if (np_abs(lnkvji[r] - lnkvji[s]) < trivtol).all():
          logger.debug('The solver converged to the trivial solution.')
          raise TrivialSolutionError()
  xi = yi / (fj.dot(kvji - 1.) + 1.)
  yji = kvji * xi
  lnphiji = eos.getPT_lnphiji(P, T, yji)
  lnphixi = eos.getPT_lnphii(P, T, xi)
  gji = lnkvji + lnphiji - lnphixi
  gi = gji.ravel()
  g2 = gi.dot(gi)
  use_ss = g2 > epsu or g2 < epsl or outside_pfw
  logger.debug(
    '%3s' + Npm1Nc * '%10s' + Npm1 * '%9s' + '%11s%8s',
    'Nit', *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['f%s' % j for j in range(Npm1)], 'g2', 'method',
  )
  logger.debug(tmpl, k, *lnkvi, *fj, g2, 'SS')
  while use_ss and g2 > tol and k < maxiter:
    k += 1
    lnkvji -= gji
    if outside_pfw:
      if (np_abs(lnkvji) < trivtol).all(axis=1).any():
        logger.debug('The solver converged to the trivial solution.')
        raise TrivialSolutionError()
      for r in range(Npm1):
        for s in range(r + 1, Npm1):
          if (np_abs(lnkvji[r] - lnkvji[s]) < trivtol).all():
            logger.debug('The solver converged to the trivial solution.')
            raise TrivialSolutionError()
    kvji = np_exp(lnkvji)
    fjkm1 = fj
    fj = rrsolver(kvji, yi, fj)
    outside_pfw = (fj <= 0.).any() or (fj >= 1.).any()
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    lnphiji = eos.getPT_lnphiji(P, T, yji)
    lnphixi = eos.getPT_lnphii(P, T, xi)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2km1 = g2
    g2 = gi.dot(gi)
    use_ss = (g2 < epsr * g2km1
              or g2 > epsu
              or g2 < epsl
              or np_abs(fj - fjkm1).max() > epsf
              or outside_pfw)
    logger.debug(tmpl, k, *lnkvi, *fj, g2, 'SS')
  if isfinite(g2):
    if g2 > tol and k < maxiter:
      H = np_empty(shape=(Npm1Nc, Npm1Nc))
      H_block = np_as_strided(
        H, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
      )
      H_blockdiag = np_as_strided(
        H, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
      )
      U = np_empty(shape=(Npm1Nc, Npm1Nc))
      U_block = np_as_strided(
        U, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
      )
      U_blockdiag = np_as_strided(
        U, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
      )
      Ux = np_full(shape=(Nc, Nc), fill_value=-1.)
      Ux_diag = np_as_strided(Ux, (Nc,), (8 * (Nc + 1),))
      Uy = np_full(shape=(Npm1, Nc, Nc), fill_value=-1.)
      Uy_diag = np_as_strided(
        Uy, (Npm1, Nc), (8 * Nc * Nc, 8 * (Nc + 1)),
      )
      fx = 1. - fj.sum()
      lnphiji, dlnphijidnk = eos.getPT_lnphiji_dnk(P, T, yji, fj)
      lnphixi, dlnphixidnk = eos.getPT_lnphii_dnj(P, T, xi, fx)
      logger.debug(tmpl, k, *lnkvi, *fj, g2, 'Newt')
      while g2 > tol and k < maxiter:
        H_block[:] = dlnphixidnk
        H_blockdiag += dlnphijidnk
        Ux_diag[:] = 1. / xi - 1.
        Uy_diag[:] = 1. / yji - 1.
        U_block[:] = Ux / fx
        U_blockdiag += Uy / fj[:, None, None]
        H += U
        # TODO: Replace the LU-solver with the custom implementation
        #       of the modified Cholesky decomposition solver.
        dni = linsolver(H, -gi)
        dlnkvi = U.dot(dni)
        lnkvi += dlnkvi
        k += 1
        kvji = np_exp(lnkvji)
        fj = rrsolver(kvji, yi, fj)
        fx = 1. - fj.sum()
        xi = yi / (fj.dot(kvji - 1.) + 1.)
        yji = kvji * xi
        lnphiji, dlnphijidnk = eos.getPT_lnphiji_dnk(P, T, yji, fj)
        lnphixi, dlnphixidnk = eos.getPT_lnphii_dnj(P, T, xi, fx)
        gji = lnkvji + lnphiji - lnphixi
        gi = gji.ravel()
        g2km1 = g2
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *lnkvi, *fj, g2, 'Newt')
      if g2 < tol and isfinite(g2):
        return fj, kvji, yji, xi
    elif g2 < tol:
      return fj, kvji, yji, xi
  logger.warning(
    'Multiphase flash calculation terminates unsuccessfully.\n'
    'The solver was "_flashNpPT_ssnewt".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvji0 = %s',
    eos.name, P, T, yi, kvji0.tolist()
  )
  raise FlashConvergenceError()


def _flashNpPT_qnssnewt(
  eos: FlashNpSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  fj0: Vector[Float],
  kvji0: Matrix[Float],
  tol: float = 1e-20,
  maxiter: int = 200,
  lmbdmax: float = 6.,
  trivtol: float = 1e-4,
  switchers: tuple[float, float, float, float] = (0.6, 1e-2, 1e-12, 1e-6),
  rrsolver: RRNpSolver = rrNp,
  linsolver: LinearSolver = lusolver,
) -> tuple[Vector[Float], Matrix[Float], Matrix[Float], Vector[Float]]:
  r"""Solve the multiphase flash problem formulated for the PT-
  thermodynamics using quasi-newton successive substitution iterations
  (QNSS) and Newton's method.

  Parameters
  ----------
  eos: FlashNpSolverPTEos
    An initialized instance of a PT-based equation of state.

  P: float
    Pressure [Pa].

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  fj0: Vector[Float], shape (Np - 1,)
    An initial guess of mole fractions of non-reference phases.

  kvji0: Matrix[Float], shape (Np - 1, Nc)
    An initial guess of k-values of non-reference phases.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the vector of equilibrium equations is less than `tol`. Default
    is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `200`.

  lmbdmax: float
    The maximum step length. Default is `6.0`.

  trivtol: float
    This parameter is used for an early check of the solver's
    convergence to the trivial solution. At any iteration of the
    QNSS-method, the error will be raised if any of the following
    conditions is satisfied:

    - absolute values of natural logarithms of k-values of any non-
      reference phase are all less than `trivtol`,

    - absolute values of differences of natural logarithms of k-values
      between any non-reference phases are all less than `trivtol`.

    Default is `1e-4`.

  switchers: tuple[float, float, float, float]
    Allows to modify the conditions of switching from the QNSS-method
    to Newton's method. The parameter must be represented as a tuple
    containing four positive values: :math:`\eps_r`, :math:`\eps_f`,
    :math:`\eps_l`, :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \max_j \left| f_j^k - f_j^{k-1} \right| < \eps_f, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
        0 < f_j^k < 1, \; j = 1 \, \ldots \, N_p - 1,
      \end{cases}

    where :math:`\mathbf{g}` is the equilibrium equations vector,
    :math:`k` is the iteration number, :math:`f_j` is the mole fraction
    of a non-reference phase :math:`j`. Analytical expressions of the
    switching conditions were taken from the paper of L.X. Nghiem
    (doi: 10.2118/8285-PA). Default is `(0.6, 1e-2, 1e-12, 1e-6)`.

  rrsolver: RRNpSolver
    A callable object that can be used to solve the system of
    Rachford-Rice equations. Default is `rrNp`.

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `((Np - 1) * Nc, (Np - 1) * Nc)` and a `b: Vector[Float]` of shape
    `((Np - 1) * Nc,)` and finds `x: Vector[Float]` of shape
    `((Np - 1) * Nc,)`, which is the solution to the linear system
    :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`. The matrix `A` is
    a symmetric matrix and usually positive definite. Default is
    `lusolver`.

  Returns
  -------
  A tuple containing:
  - a `Vector[Float]` of shape `(Np - 1,)` of mole fractions of non-
    reference phases,
  - a `Matrix[Float]` of shape `(Np - 1, Nc)` of k-values of components
    in non-reference phases,
  - a `Matrix[Float]` of shape `(Np - 1, Nc)` of mole fractions of
    components in non-reference phases,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the reference phase.

  Raises
  ------
  FlashConvergenceError
    This exception is raised if the solver does not converge.

  RRConvergenceError
    This exception is raised if the solver of the system of
    Rachford-Rice equations does not converge.

  TrivialSolutionError
    This exception will be raised if the solver converges to the trivial
    solution, which is detected by monitoring conditions described in
    the explanation for the `trivtol` parameter.

  Notes
  -----
  Details of the implemented algorithm:

  - Quasi-Newton successive substitution iterations precede Newton's
    method to improve the initial guess of k-values.

  - For other numerical details of the QNSS-method, see the paper of
    L.X. Nghiem and Y.-K. Li, 1984 (doi: 10.1016/0378-3812(84)80013-8).

  - Because this solver does not conduct a stability test procedure
    when an unphysical value of mole fraction of any non-reference
    phase is yilded from the solver of Rachford-Rice equations, it is
    recommended to first perform a stability test and find the global
    minimum of the TPD-function. The stability test procedure can also
    provide initial guesses for the mole fractions of non-reference
    phases and k-values of components. For the details, see the paper of
    Z. Li and A. Firoozabadi, 2012 (doi: 10.1016/j.fluid.2012.06.021).

  - The solver's convergence to the trivial solution is checked by
    monitoring absolute values of natural logarithms of k-values and
    their differences between non-reference phases.

  - The system of Rachford-Rice equations is solved in the inner loop
    of Newton's method. This approach is faster because the symmetry
    of the Hessian matrix can be exploited with Cholesky decomposition
    when solving the system of linear equations. For the details, see
    the paper of M. Petitfrere and D.V. Nichita, 2016 (doi:
    1016/j.fluid.2016.06.050).

  - In general, the Cholesky decomposition can be used to solve the
    system of linear equations. This is because Newton's method is
    applied only after the switching criteria, which contain the
    convergence metric, are satisfied. However, the use of the modified
    Cholesky decomposition is still recommended. The switching criteria
    do not guarantee the positive definiteness of the Hessian matrix.
    Because the modified Cholesky decomposition is not implemented in
    the `numpy` library, the `numpy.linalg.solve` is used instead by
    default.

  - The solver can be transformed to either the QNSS-method or Newton's
    method by changing the switching criteria.
  """
  Nc = eos.Nc
  Npm1 = fj0.shape[0]
  Npm1Nc = Npm1 * Nc
  logger.info(
    'Solving the %s-phase flash problem using the QNSS-Newton solver.',
    Npm1 + 1,
  )
  logger.info('P = %.1f [Pa], T = %.2f [K], yi =' + Nc * '%7.4f', P, T, *yi)
  tmpl = '%3s' + Npm1Nc * '%10.4f' + Npm1 * '%9.4f' + '%11.2e%8s'
  epsr, epsf, epsl, epsu = switchers
  k = 0
  kvji = kvji0
  fj = fj0
  lnkvji = np_log(kvji)
  lnkvi = lnkvji.ravel()
  fj = rrsolver(kvji, yi, fj)
  outside_pfw = (fj <= 0.).any() or (fj >= 1.).any()
  if outside_pfw:
    if (np_abs(lnkvji) < trivtol).all(axis=1).any():
      logger.debug('The solver converged to the trivial solution.')
      raise TrivialSolutionError()
    for r in range(Npm1):
      for s in range(r + 1, Npm1):
        if (np_abs(lnkvji[r] - lnkvji[s]) < trivtol).all():
          logger.debug('The solver converged to the trivial solution.')
          raise TrivialSolutionError()
  xi = yi / (fj.dot(kvji - 1.) + 1.)
  yji = kvji * xi
  lnphiji = eos.getPT_lnphiji(P, T, yji)
  lnphixi = eos.getPT_lnphii(P, T, xi)
  gji = lnkvji + lnphiji - lnphixi
  gi = gji.ravel()
  g2 = gi.dot(gi)
  lmbd = 1.
  dlnkvi = -gi
  use_qnss = g2 > epsu or g2 < epsl or outside_pfw
  logger.debug(
    '%3s' + Npm1Nc * '%10s' + Npm1 * '%9s' + '%11s%8s',
    'Nit', *['lnkv%s%s' % (j, i) for j in range(Npm1) for i in range(Nc)],
    *['f%s' % j for j in range(Npm1)], 'g2', 'method',
  )
  logger.debug(tmpl, k, *lnkvi, *fj, g2, 'QNSS')
  while use_qnss and g2 > tol and k < maxiter:
    k += 1
    tkm1 = dlnkvi.dot(gi)
    lnkvi += dlnkvi
    if outside_pfw:
      if (np_abs(lnkvji) < trivtol).all(axis=1).any():
        logger.debug('The solver converged to the trivial solution.')
        raise TrivialSolutionError()
      for r in range(Npm1):
        for s in range(r + 1, Npm1):
          if (np_abs(lnkvji[r] - lnkvji[s]) < trivtol).all():
            logger.debug('The solver converged to the trivial solution.')
            raise TrivialSolutionError()
    kvji = np_exp(lnkvji)
    fjkm1 = fj
    fj = rrsolver(kvji, yi, fj)
    outside_pfw = (fj <= 0.).any() or (fj >= 1.).any()
    xi = yi / (fj.dot(kvji - 1.) + 1.)
    yji = kvji * xi
    lnphiji = eos.getPT_lnphiji(P, T, yji)
    lnphixi = eos.getPT_lnphii(P, T, xi)
    gji = lnkvji + lnphiji - lnphixi
    gi = gji.ravel()
    g2km1 = g2
    g2 = gi.dot(gi)
    use_qnss = (g2 < epsr * g2km1
                or g2 > epsu
                or g2 < epsl
                or np_abs(fj - fjkm1).max() > epsf
                or outside_pfw)
    if k % Nc == 0:
      lmbd = 1.
      dlnkvi = -gi
    else:
      lmbd *= tkm1 / (dlnkvi.dot(gi) - tkm1)
      if lmbd < 0.:
        lmbd = -lmbd
      if lmbd > lmbdmax:
        lmbd = lmbdmax
      dlnkvi = -lmbd * gi
      max_dlnkvi = np_abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
    logger.debug(tmpl, k, *lnkvi, *fj, g2, 'QNSS')
  if isfinite(g2):
    if g2 > tol and k < maxiter:
      H = np_empty(shape=(Npm1Nc, Npm1Nc))
      H_block = np_as_strided(
        H, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
      )
      H_blockdiag = np_as_strided(
        H, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
      )
      U = np_empty(shape=(Npm1Nc, Npm1Nc))
      U_block = np_as_strided(
        U, (Npm1, Npm1, Nc, Nc), (8 * Npm1Nc * Nc, 8 * Nc, 8 * Npm1Nc, 8),
      )
      U_blockdiag = np_as_strided(
        U, (Npm1, Nc, Nc), (8 * (Npm1Nc + 1) * Nc, 8 * Npm1Nc, 8),
      )
      Ux = np_full(shape=(Nc, Nc), fill_value=-1.)
      Ux_diag = np_as_strided(Ux, (Nc,), (8 * (Nc + 1),))
      Uy = np_full(shape=(Npm1, Nc, Nc), fill_value=-1.)
      Uy_diag = np_as_strided(
        Uy, (Npm1, Nc), (8 * Nc * Nc, 8 * (Nc + 1)),
      )
      fx = 1. - fj.sum()
      lnphiji, dlnphijidnk = eos.getPT_lnphiji_dnk(P, T, yji, fj)
      lnphixi, dlnphixidnk = eos.getPT_lnphii_dnj(P, T, xi, fx)
      logger.debug(tmpl, k, *lnkvi, *fj, g2, 'Newt')
      while g2 > tol and k < maxiter:
        H_block[:] = dlnphixidnk
        H_blockdiag += dlnphijidnk
        Ux_diag[:] = 1. / xi - 1.
        Uy_diag[:] = 1. / yji - 1.
        U_block[:] = Ux / fx
        U_blockdiag += Uy / fj[:, None, None]
        H += U
        # TODO: Replace the LU-solver with the custom implementation
        #       of the modified Cholesky decomposition solver.
        dni = linsolver(H, -gi)
        dlnkvi = U.dot(dni)
        lnkvi += dlnkvi
        k += 1
        kvji = np_exp(lnkvji)
        fj = rrsolver(kvji, yi, fj)
        fx = 1. - fj.sum()
        xi = yi / (fj.dot(kvji - 1.) + 1.)
        yji = kvji * xi
        lnphiji, dlnphijidnk = eos.getPT_lnphiji_dnk(P, T, yji, fj)
        lnphixi, dlnphixidnk = eos.getPT_lnphii_dnj(P, T, xi, fx)
        gji = lnkvji + lnphiji - lnphixi
        gi = gji.ravel()
        g2km1 = g2
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *lnkvi, *fj, g2, 'Newt')
      if g2 < tol and isfinite(g2):
        return fj, kvji, yji, xi
    elif g2 < tol:
      return fj, kvji, yji, xi
  logger.warning(
    'Multiphase flash calculation terminates unsuccessfully.\n'
    'The solver was "_flashNpPT_qnssnewt".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvji0 = %s',
    eos.name, P, T, yi, kvji0.tolist()
  )
  raise FlashConvergenceError()


class flash(object):
  def __init__(
    self,
    maxNp: int = 3,
    state: State | None = None,
  ) -> None:
    """Specify settings for the multiphase flash calculation.

    Parameters
    ----------
    maxNp: int
      The maximum number of phases (`maxNp > 1`). Default is `3`.

    state: State | None
      A thermodynamic state of a mixture, k-values from which can be
      used as an initial guess for the flash calculation procedure.
      If it is `None`, the option to use previously calculated results
      is not available until such a state is passed directly to the
      callable instance of this class. Default is `None`.

    Notes
    -----
    Above settings influence the calculation only if the corresponding
    parameters of the `__call__` method of this class are set to `None`.
    """
    self.maxNp = maxNp
    self.state = state
    pass

  def __call__(
    self,
    eos: FlashNpPTEos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    n: float = 1.,
    init: Iterable[Vector[Float]] | State | None = None,
    maxNp: int | None = None,
    stabroutine: StabRoutine | None = stabtest(),
    solver2p: Flash2pSolver | str = 'default',
    solverNp: FlashNpSolver | str = 'default',
  ) -> State:
    """Multiphase flash calculation.

    Parameters
    ----------
    eos: FlashNpPTEos
      An initialized instance of an equation of state.

    P1: float
      The first thermodynamic parameter in SI units.

    P2: float
      The second thermodynamic parameter in SI units.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Default is `1.0` [mol].

    init: Iterable[Vector[Float]] | State | None
      This parameter is used to initialize the flash calculations. The
      detailed explanation of the logic behind different types of `init`
      is given in the following table:

      +-------------------------+--------------------------------------+
      | Type                    | Description                          |
      +=========================+======================================+
      | Iterable[Vector[Float]] | An iterable object of initial        |
      |                         | guesses of k-values (arrays of shape |
      |                         | `(Nc,)`), which are directly used    |
      |                         | to initialize the two-phase solver.  |
      +-------------------------+--------------------------------------+
      | State                   | A state of a mixture, k-values from  |
      |                         | which can be used to initialize the  |
      |                         | flash routine. In addition to these  |
      |                         | k-values, initial guesses are also   |
      |                         | obtained from the `eos` (if the      |
      |                         | number of phases in a given state    |
      |                         | is less than or equal to two).       |
      +-------------------------+--------------------------------------+
      | None                    | The `eos` is used to prepare initial |
      |                         | guesses of k-values.                 |
      +-------------------------+--------------------------------------+

      Default is `None`.

    maxNp: int | None
      The maximum number of phases (`maxNp > 1`). If it is `None` the
      maximum number of phases is equal to the value defined during the
      initialization of this callable object (which defaults to `3`).
      Default is `None`.

    stabroutine: StabRoutine | None
      A callable object that can be used to perform the one-phase
      stability test. This parameter is also used to regulate the
      initialization procedure for two- and multiphase flash routines.

      If `stabroutine is not None`:

      - the algorithm first determines the global minimum of the TPD-
        function by conducting the stability test procedure for all
        initial guesses of k-values given manually or obtained from a
        method of the initialized instance of an equation of state,

      - the two-phase flash procedure is only performed if a one-phase
        state is found to be unstable during the stability test,

      - the results from the stability test are used to initialize
        two- and multiphase flash routines; initialization of the
        multiphase flash routine is based on the paper of Z. Li and
        A. Firoozabadi, 2012 (doi: 10.1016/j.fluid.2012.06.021).

      If `stabroutine is None`:

      - the stability test is only performed if a two-phase split solver
        yields a non-physical value of the non-reference phase mole
        fraction when using an initial guess of k-values given manually
        or obtained from a method of the initialized instance of an
        equation of state,

      - if the above condition is met, the initial guess of k-values
        will be replaced with those from stability test if it shows that
        a one-phase state in unstable; otherwise, the next initial
        guess of k-values will be considered,

      - the above logic of performing the stability test and k-values
        replacement is implemented by internal two-phase flash solvers,

      - initialization of the multiphase flash procedure is based on
        k-values generated by a method of the initialized instance of
        an equation of state.

      The choice between these options depends on the reliability of
      initial guesses of k-values given manually or obtained from a
      method of the initialized instance of an equation of state:

      - if you are confident that your initial guesses are accurate, it
        is recommended to set this parameter to `None`.

      - if the initial guesses are less reliable, setting it to a
        stability test procedure may be beneficial to ensure a robust
        calculation.

      The first approach, i.e., the use of the k-values obtained from
      the stability test and characterized by the lower value of the
      TPD-function, was recommended by many authors. For example, one
      can find this recommendation in the following papers:

      - M.L. Michelsen, 1982 (doi: 10.1016/0378-3812(82)85002-4),

      - L. Nghiem and Y.K. Li, 1984 (doi: 10.1016/0378-3812(84)80013-8),

      - Z. Li and A. Firoozabadi, 2012 (doi: 10.2118/129844-PA).

      The second approach was described in the following paper:

      - C.P. Rasmussen et al, 2006 (doi: 10.2118/84181-PA).

      Default is `stabtest()`.

    solver2p: Flash2pSolver | str
      A callable object that can be used to solve two-phase flash
      problems formulated for the given thermodynamic parameters.
      It also can be a string defining the name of an internal solver.

      For the PT-thermodynamics, the following internal two-phase flash
      solvers are available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'ss-newton'`   | Newton's method with preceding successive    |
      |                 | substitution iterations.                     |
      +-----------------+----------------------------------------------+
      | `'qnss-newton'` | Newton's method with preceding quasi-newton  |
      |                 | successive substitution iterations.          |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      formulation of the two-phase flash problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | P, T            | QNSS-Newton (`'qnss-newton'`).               |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    solverNp: FlashNpSolver | str
      A callable object that can be used to solve multiphase flash
      problems formulated for the given thermodynamic parameters.
      It also can be a string defining the name of an internal solver.

      For the PT-thermodynamics, the following internal multiphase flash
      solvers are available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'ss-newton'`   | Newton's method with preceding successive    |
      |                 | substitution iterations.                     |
      +-----------------+----------------------------------------------+
      | `'qnss-newton'` | Newton's method with preceding quasi-newton  |
      |                 | successive substitution iterations.          |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      formulation of the multiphase flash problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | P, T            | QNSS-Newton (`'qnss-newton'`).               |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    Returns
    -------
    Flash calculation results as an instance of `State`.

    Raises
    ------
    FlashConvergenceError
      This exception is raised if a flash solver does not converge.

    RRConvergenceError
      This exception is raised if a solver of the (system of)
      Rachford-Rice equation(s) does not converge.

    Notes
    -----
    1. Internal solvers for two- and multiphase flash problems use
    solvers for the Rachford-Rice equation and the system of Rachford-
    Rice equations correspondingly. The following table lists available
    solvers for the Rachford-Rice equation:

    +-----------------+------------------------------------------------+
    | Internal solver | Description                                    |
    +=================+================================================+
    | `rr2p_gh`       | Uses convex transformations of the Rachford-   |
    |                 | Rice equation to find the solution inside the  |
    |                 | negative flash window robustly.                |
    +-----------------+------------------------------------------------+
    | `rr2p_fgh`      | Uses convex and quasi-linear transformations   |
    |                 | of the Rachford-Rice equation to find the      |
    |                 | solution inside the negative flash window      |
    |                 | rapidly and robustly.                          |
    +-----------------+------------------------------------------------+

    Default is `rr2p_fgh`. For the details, see the following papers:

    - D.V. Nichita and C.F. Leibovici, 2013 (doi:
      10.1016/j.fluid.2013.05.030),

    - D.V. Nichita and C.F. Leibovici, 2014 (doi:
      10.1016/j.compchemeng.2014.10.006),

    - D.V. Nichita and C.F. Leibovici, 2017 (doi:
      10.1016/j.fluid.2017.08.020).

    The following table lists available solvers for the system of
    Rachford-Rice equations:

    +-----------------+------------------------------------------------+
    | Internal solver | Description                                    |
    +=================+================================================+
    | `rrNp`          | Uses convex transformations of the system of   |
    |                 | Rachford-Rice equations and Newton's method    |
    |                 | for minimization as well as the line-search    |
    |                 | technique to find the solution inside the      |
    |                 | negative flash window (NF-window) rapidly      |
    |                 | and robustly. An initial guess within the NF-  |
    |                 | window is crucial for the algorithm.           |
    +-----------------+------------------------------------------------+

    Default is `rrNp`. For the details, see the following papers:

    - R. Okuno et al, 2010 (doi: 10.2118/117752-PA),

    - Z. Li and A. Firoozabadi, 2012 (doi: 10.1016/j.fluid.2012.06.021).

    2. To perform the negative flash, set the `stabroutine` parameter of
    this function and the `stabsolver` parameter of an internal two-
    phase flash solver to `None`. Then, pass the modified two-phase
    flash solver to this function using the `solver2p` parameter. To
    change additional arguments of a function, use the `partial`
    function from Python's built-in `functools` module.

    3. Selection between different formulations of the multiphase flash
    calculation is based on the `form` attribute of the `eos`.
    """
    if maxNp is None:
      maxNp = self.maxNp
    if init is None:
      init = self.state
    if eos.form == 'PT':
      eos = cast(FlashNpPTEos, eos)
      solver2pPT: Flash2pSolver[Flash2pSolverPTEos]
      if callable(solver2p):
        solver2pPT = cast(Flash2pSolver[Flash2pSolverPTEos], solver2p)
      elif solver2p == 'default' or solver2p == 'qnss-newton':
        solver2pPT = _flash2pPT_qnssnewt
      elif solver2p == 'ss-newton':
        solver2pPT = _flash2pPT_ssnewt
      else:
        raise ValueError(
          f'Unknown two-phase flash-solver for a PT-based EOS: "{solver2p}".'
        )
      solverNpPT: FlashNpSolver[FlashNpSolverPTEos]
      if callable(solverNp):
        solverNpPT = cast(FlashNpSolver[FlashNpSolverPTEos], solverNp)
      elif solverNp == 'default' or solverNp == 'qnss-newton':
        solverNpPT = _flashNpPT_qnssnewt
      elif solverNp == 'ss-newton':
        solverNpPT = _flashNpPT_ssnewt
      else:
        raise ValueError(
          f'Unknown multiphase flash-solver for a PT-based EOS: "{solverNp}".'
        )
      stabroutine = cast(StabRoutine[StabPTEos], stabroutine)
      return self.runNpPT(
        eos, P1, P2, yi, n, init, maxNp, solver2pPT, solverNpPT, stabroutine,
      )
    else:
      raise NotImplementedError(
        f'The {eos.form}-formulation of the flash calculation routine '
        'is not implemented yet.'
      )

  @classmethod
  def runNpPT(
    cls,
    eos: FlashNpPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    init: Iterable[Vector[Float]] | State | None = None,
    maxNp: int = 3,
    solver2p: Flash2pSolver[Flash2pSolverPTEos] = _flash2pPT_qnssnewt,
    solverNp: FlashNpSolver[FlashNpSolverPTEos] = _flashNpPT_qnssnewt,
    stabroutine: StabRoutine[StabPTEos] | None = stabtest.runPT,
  ) -> State:
    """Perform the flash routine for a given pressure, temperature,
    and mole composition of a mixture to determine the multiphase
    state, which is characterized by the minimum Gibbs energy. The
    stability test procedure is used to determine whether a found
    local minimum of the Gibbs energy function is the global minimum.

    Parameters
    ----------
    eos: FlashNpPTEos
      An initialized instance of a PT-based equation of state.

    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Defalt is `1.0` [mol].

    init: Iterable[Vector[Float]] | State | None
      This parameter is used to initialize the flash calculations. The
      detailed explanation of the logic behind different types of `init`
      is given in the following table:

      +-------------------------+--------------------------------------+
      | Type                    | Description                          |
      +=========================+======================================+
      | Iterable[Vector[Float]] | An iterable object of initial        |
      |                         | guesses of k-values (arrays of shape |
      |                         | `(Nc,)`), which are directly used    |
      |                         | to initialize the two-phase solver.  |
      +-------------------------+--------------------------------------+
      | State                   | A state of a mixture, k-values from  |
      |                         | which can be used to initialize the  |
      |                         | flash routine. In addition to these  |
      |                         | k-values, initial guesses are also   |
      |                         | obtained from the `eos` (if the      |
      |                         | number of phases in a given state    |
      |                         | is less than or equal to two).       |
      +-------------------------+--------------------------------------+
      | None                    | The `eos` is used to prepare initial |
      |                         | guesses of k-values.                 |
      +-------------------------+--------------------------------------+

      Default is `None`.

    maxNp: int
      The maximum number of phases (`maxNp > 1`). Default is `3`.

    solver2p: Flash2pSolver[Flash2pSolverPTEos]
      A callable object that can be used to solve two-phase flash
      problems formulated for the PT-thermodynamics. Default is
      `_flash2pPT_qnssnewt`.

    solverNp: FlashNpSolver[FlashNpSolverPTEos] | str
      A callable object that can be used to solve multiphase flash
      problems formulated for the PT-thermodynamics. Default is
      `_flashNpPT_qnssnewt`.

    stabroutine: StabRoutine | None
      A callable object that can be used to perform the one-phase
      stability test formulated for the PT-thermodynamics. This
      parameter is also used to regulate initialization procedures
      for two- and multiphase flash routines. Default is
      `stabtest.runPT`.

    Returns
    -------
    Flash calculation results as an instance of `State`.

    Raises
    ------
    FlashConvergenceError
      This exception is raised if a flash solver does not converge.

    RRConvergenceError
      This exception is raised if a solver of the system of
      Rachford-Rice equations does not converge.
    """
    state: State
    if (isinstance(init, State)
        and init.Np > 2
        and init.Np <= maxNp
        and init.kvji is not None):
      kvji = init.kvji
      try:
        fj, kvji, yji, xi = solverNp(eos, P, T, yi, init.fj[:-1], kvji)
        state = cls.outputNpPT(eos, P, T, yi, n, fj, kvji, yji, xi)
      except (FlashConvergenceError,
              RRConvergenceError,
              TrivialSolutionError):
        state = cls.run2pPT(eos, P, T, yi, n, (*kvji,), solver2p, stabroutine)
        if state.Np == 1:
          return state
    else:
      state = cls.run2pPT(eos, P, T, yi, n, init, solver2p, stabroutine)
      if state.Np == 1:
        return state
    exception: Exception | None = None
    for k in range(state.Np + 1, maxNp + 1):
      if exception is None:
        fsj0, kvsji0 = cls.initNpPT(eos, P, T, state, stabroutine)
      else:
        raise exception
      if kvsji0:
        for s, (fj, kvji) in enumerate(zip(fsj0, kvsji0)):
          logger.debug('Initial guess of k-values #%s.', s)
          try:
            fj, kvji, yji, xi = solverNp(eos, P, T, yi, fj, kvji)
            state = cls.outputNpPT(eos, P, T, yi, n, fj, kvji, yji, xi)
            exception = None
            break
          except (FlashConvergenceError, RRConvergenceError) as e:
            exception = e
            continue
          except (TrivialSolutionError) as e:
            if stabroutine is not None:
              exception = e
            continue
      else:
        return state
    if exception is None:
      return state
    else:
      raise exception

  @classmethod
  def run2pPT(
    cls,
    eos: Flash2pPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    init: Iterable[Vector[Float]] | State | None = None,
    solver: Flash2pSolver[Flash2pSolverPTEos] = _flash2pPT_qnssnewt,
    stabroutine: StabRoutine[StabPTEos] | None = None,
  ) -> State:
    """Perform the flash routine for a given pressure, temperature,
    and mole composition of a mixture to determine a two-phase state,
    which is characterized by the equal figacities of components in
    both phases. If a one-phase state is determined to be stable, the
    routine may instead return it and its properties as an instance
    of `State`.

    Parameters
    ----------
    eos: Flash2pPTEos
      An initialized instance of a PT-based equation of state.

    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Default is `1.0` [mol].

    init: Iterable[Vector[Float]] | State | None
      This parameter is used to initialize the flash calculations. The
      detailed explanation of the logic behind different types of `init`
      is given in the following table:

      +-------------------------+--------------------------------------+
      | Type                    | Description                          |
      +=========================+======================================+
      | Iterable[Vector[Float]] | An iterable object of initial        |
      |                         | guesses of k-values (arrays of shape |
      |                         | `(Nc,)`), which are directly used    |
      |                         | to initialize the solver.            |
      +-------------------------+--------------------------------------+
      | State                   | A state of a mixture, k-values from  |
      |                         | which can be used to initialize the  |
      |                         | flash routine. In addition to these  |
      |                         | k-values, initial guesses are also   |
      |                         | obtained from the `eos`.             |
      +-------------------------+--------------------------------------+
      | None                    | The `eos` is used to prepare initial |
      |                         | guesses of k-values.                 |
      +-------------------------+--------------------------------------+

      Default is `None`.

    solver: Flash2pSolver[Flash2pSolverPTEos]
      A callable object that can solve two-phase flash problems
      formulated for the PT-thermodynamics. Default is
      `_flash2pPT_qnssnewt`.

    stabroutine: StabRoutine[StabPTEos] | None
      A callable object that can be used to perform the one-phase
      stability test, results of which are further exploited as the
      initial guess for the two-phase flash routine. If it is `None`,
      k-values given manually or generated by the method `getPT_kvguess`
      of the `eos` are passed directly to the `solver`. Default is
      `None`.

    Returns
    -------
    Flash calculation results as an instance of `State`.

    Raises
    ------
    FlashConvergenceError
      This exception is raised if a flash solver does not converge.

    RRConvergenceError
      This exception is raised if a solver of the Rachford-Rice equation
      does not converge.
    """
    if isinstance(init, Iterable):
      kvji0 = init
    else:
      kvji0 = eos.getPT_kvguess(P, T, yi)
      if isinstance(init, State):
        kvjim1 = init.kvji
        if kvjim1 is not None and init.Np < 3:
          # TODO: Use the extrapolation procedure to improve the initial
          #       guess of k-values. For the details, see the paper of
          #       L.X. Nghiem and Y.K. Li, 1990 (doi: 10.2118/13517-PA).
          kvji0 = (kvjim1.ravel(), *kvji0)
    exception: Exception | None
    if stabroutine is None:
      exception = None
    else:
      stabstate = stabroutine(eos, P, T, yi, n, kvji0)
      if stabstate.kvji is None:
        logger.debug('The one phase is stable: True.')
        return stabstate
      logger.debug('The one phase is stable: False.')
      stabkvi = stabstate.kvji.ravel()
      if isinstance(init, State):
        kvjim1 = init.kvji
        if kvjim1 is not None and init.Np < 3:
          kvji0 = (kvjim1.ravel(), stabkvi, *kvji0)
        else:
          kvji0 = (stabkvi, *kvji0)
      else:
        kvji0 = (stabkvi, *kvji0)
      exception = FlashConvergenceError()
    for j, kvi0 in enumerate(kvji0):
      logger.debug('Initial guess of k-values #%s.', j)
      try:
        f0, kvi, y0i, y1i = solver(eos, P, T, yi, kvi0)
        return cls.output2pPT(eos, P, T, yi, n, f0, kvi, y0i, y1i)
      except (FlashConvergenceError, RRConvergenceError) as e:
        exception = e
        continue
      except (OnePhaseStateError, TrivialSolutionError):
        continue
    if exception is None:
      return cls.output1pPT(eos, P, T, yi, n)
    else:
      raise exception

  @staticmethod
  def initNpPT(
    eos: StabPTEos,
    P: float,
    T: float,
    state: State,
    stabroutine: StabRoutine[StabPTEos] | None,
  ) -> tuple[list[Vector[Float]], list[Matrix[Float]]]:
    """Generate initial guesses of k-values and phase mole fractions
    for `Np`-phase flash calculation.

    Parameters
    ----------
    eos: StabPTEos
      An initialized instance of a PT-based equation of state that can
      be used to perform the one-phase stability test.

    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    state: State
      `(Np - 1)`-flash results.

    stabroutine: StabRoutine | None
      A callable object that can be used to perform the one-phase
      stability test. If it is not `None`, then the initialization
      procedure is based on the paper of Z. Li and A. Firoozabadi, 2012
      (doi: 10.1016/j.fluid.2012.06.021). Otherwise, initialization of
      the multiphase flash procedure is based on k-values generated by
      the method `getPT_kvguess` of the `eos`.

    Returns
    -------
    A tuple containig:
    - initial guesses of mole fractions of non-reference phases as a
      `list[Vector[Float]]`; the shape of each vector is `(Np - 1,)`,
    - initial guesses of k-values as a `list[Matrix[Float]]`; the shape
      of each matrix is `(Np - 1, Nc)`, where `Nc` is the number of
      components.

    Notes
    -----
    Pairs of initial guesses are sorted in descending order based on
    the minimum value of mole fractions in a reference phase. This
    avoids selecting a phase with very small mole fractions of
    components in it as a reference, which, in turn, leads to more
    smoothly solving the system of non-linear equations.
    """
    Np = state.Np
    yji = state.yji
    fj = state.fj
    phases = np_argsort(yji.min(axis=1))[::-1]
    kvsji = []
    fsj = []
    if stabroutine is None:
      fj_ = np_full_like(fj, 1 / (Np + 1))
      for r in phases:
        xi = yji[r]
        kvsi = eos.getPT_kvguess(P, T, xi)
        kvji_ = yji / xi
        for kvi in kvsi:
          kvji = kvji_.copy()
          kvji[r] = kvi
          kvsji.append(kvji)
          fsj.append(fj_)
    else:
      logger.debug('Checking stability of the %s-phase state.', Np)
      for r in phases:
        xi = yji[r]
        stab = stabroutine(eos, P, T, xi, 1., None)
        stabkvji = stab.kvji
        logger.debug('The phase #%s is stable: %s.', r, stabkvji is None)
        if stabkvji is not None:
          kvji = yji / xi
          kvji[r] = stabkvji.ravel()
          kvsji.append(kvji)
          fj0 = fj.copy()
          fj0[r] = 0.
          fsj.append(fj0)
    return fsj, kvsji

  @staticmethod
  def output1pPT(
    eos: State2pPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
  ) -> OnePhaseState:
    Z = eos.getPT_Z(P, T, yi)
    pid = eos.getPT_PID(P, T, yi)
    v = Z * R * T / P
    d = yi.dot(eos.mwi) / v
    V = n * v
    ni = n * yi
    nj = np_array([n])
    nji = np_atleast_2d(ni)
    fj = np_array([1.])
    yji = np_atleast_2d(yi)
    Zj = np_array([Z])
    vj = np_array([v])
    Vj = np_array([V])
    sj = np_array([1.])
    dj = np_array([d])
    pidj = np_array([pid])
    return OnePhaseState(
      eos.Nc, 1, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
      None,
    )

  @staticmethod
  def output2pPT(
    eos: State2pPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
    f0: float,
    kvi: Vector[Float],
    y0i: Vector[Float],
    y1i: Vector[Float],
  ) -> MultiPhaseState:
    f1 = 1. - f0
    Z0 = eos.getPT_Z(P, T, y0i)
    Z1 = eos.getPT_Z(P, T, y1i)
    pid0 = eos.getPT_PID(P, T, y0i)
    pid1 = eos.getPT_PID(P, T, y1i)
    RTP = R * T / P
    v0 = Z0 * RTP
    v1 = Z1 * RTP
    n0 = n * f0
    n1 = n * f1
    V0 = n0 * v0
    V1 = n1 * v1
    V = V0 + V1
    s0 = V0 / V
    s1 = 1. - s0
    d0 = y0i.dot(eos.mwi) / v0
    d1 = y1i.dot(eos.mwi) / v1
    ni = n * yi
    nj = np_array([n0, n1])
    nji = np_array([n0 * y0i, n1 * y1i])
    fj = np_array([f0, f1])
    yji = np_array([y0i, y1i])
    Zj = np_array([Z0, Z1])
    vj = np_array([v0, v1])
    Vj = np_array([V0, V1])
    sj = np_array([s0, s1])
    dj = np_array([d0, d1])
    pidj = np_array([pid0, pid1])
    kvji = np_atleast_2d(kvi)
    return MultiPhaseState(
      eos.Nc, 2, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
      kvji,
    )

  @staticmethod
  def outputNpPT(
    eos: StateNpPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
    fj: Vector[Float],
    kvji: Matrix[Float],
    yji: Matrix[Float],
    xi: Vector[Float],
  ) -> MultiPhaseState:
    yji = np_append(yji, [xi], 0)
    fj = np_append(fj, [1. - fj.sum()], 0)
    Np = fj.shape[0]
    Zj = eos.getPT_Zj(P, T, yji)
    vj = Zj * (R * T / P)
    ni = n * yi
    nj = n * fj
    nji = nj[:,None] * yji
    Vj = nj * vj
    V = Vj.sum()
    sj = Vj / V
    dj = yji.dot(eos.mwi) / vj
    pidj = eos.getPT_PIDj(P, T, yji)
    return MultiPhaseState(
      eos.Nc, Np, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
      kvji,
    )
