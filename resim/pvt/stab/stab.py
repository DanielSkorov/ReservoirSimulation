from logging import (
  getLogger,
)

from typing import (
  cast,
  Iterable,
)

from math import (
  isfinite,
  log,
)

from numpy import (
  abs as np_abs,
  array as np_array,
  atleast_2d as np_atleast_2d,
  exp as np_exp,
  fill_diagonal as np_fill_diagonal,
  log as np_log,
  sqrt as np_sqrt,
)

from resim.pvt.datatypes import (
  Float,
  MultiPhaseState,
  OnePhaseState,
  Vector,
  State,
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
)

from resim.pvt.stab.protocols import (
  StabSolverPTEos,
  StabPTEos,
  StabSolver,
)


logger = getLogger('stab')


class StabConvergenceError(Exception):
  """An exception that will be raised if a stability solver does not
  converge.
  """
  def __init__(
    self,
    msg: str = ('The stability test completed unsuccessfully. Try to '
                'increase the maximum number of iterations or improve '
                'the initial guess. It may also be advisable to change '
                'the solver.'),
  ) -> None:
    super().__init__(msg)
    pass


def _stabPT_ssnewt(
  eos: StabSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 150,
  checktrivial: bool = True,
  switchers: tuple[float, float, float] = (0.1, 1e-12, 1e-4),
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float]]:
  r"""Solve the one-phase stability problem formulated for the PT-
  thermodynamics using successive substitution iterations and Newton's
  method.

  Parameters
  ----------
  eos: StabSolverPTEos
    An initialized instance of a PT-based equation of state of type
    `StabSolverPTEos`.

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
    of the gradient of the tangent-plane distance function is less
    than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `150`.

  checktrivial: bool
    A flag indicating whether it is necessary to perform the
    check for early detection of convergence to the trivial
    solution. It is based on the paper of M.L. Michelsen, 1982
    (doi: 10.1016/0378-3812(82)85001-2). Default is `True`.

  switchers: tuple[float, float, float]
    Allows to modify the conditions of switching from successive
    substitution iterations to Newton's method. The parameter must be
    represented as a tuple containing three positive values:
    :math:`\eps_r`, :math:`\eps_l`, :math:`\eps_u`. The switching
    conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
      \end{cases}

    where :math:`\mathbf{g}` is a vector of the gradient of the TPD
    function, :math:`k` is the iteration number. Analytical expressions
    of the switching conditions were taken from the paper of L.X. Nghiem
    et al, 1983 (doi: 10.2118/8285-PA). Default is `(0.1, 1e-12, 1e-4)`.

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
  - the TPD-function value at a found solution (a local minimum of the
    TPD-function),
  - k-values of components (which are the solution to the system of
    nonlinear equations) as a `Vector[Float]` of shape `(Nc,)`.

  Raises
  ------
  StabConvergenceError
    This exception is raised if the algorithm does not converge.

  Notes
  -----
  Details of the implemented algorithm:

  - The initial part of the algorithm is successive substitution
    iterations, which are used to improve an initial guess.

  - The subsequent application of Newton's method minimizes the
    Michelsen's modified TPD-function.

  Such the combination of successive substitution iterations and
  Newton's method makes the algorithm robust and rapid. The solver
  can be transformed to either the successive substitution method
  or Newton's method by changing the switching criteria.
  """
  logger.info('Stability Test (SS-Newton method).')
  Nc = eos.Nc
  logger.info(
    'P = %.1f [Pa], T = %.2f [K], yi =' + Nc * ' %6.4f', P, T, *yi,
  )
  tmpl = '%3s' + Nc * '%9s' + '%11s%9s'
  rangeNc = range(Nc)
  h1 = tmpl % ('Nit', *['lnkv%s' % s for s in rangeNc], 'g2', 'method')
  h2 = tmpl % ('Nit', *['alph%s' % s for s in rangeNc], 'g2', 'method')
  tmpl = '%3s' + Nc * '%9.4f' + '%11.2e%9s'
  epsr, epsl, epsu = switchers
  lnphiyi = eos.getPT_lnphii(P, T, yi)
  k = 0
  trivial = False
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  lnphixi = eos.getPT_lnphii(P, T, xi)
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  switch = g2 > epsl and g2 < epsu
  logger.debug(h1)
  logger.debug(tmpl, k, *lnki, g2, 'ss')
  while not switch and g2 > tol and k < maxiter:
    k += 1
    lnki -= gi
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    lnphixi = eos.getPT_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2km1 = g2
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnki, g2, 'ss')
    if checktrivial:
      if g2 < tol:
        break
      ng = ni.dot(gi)
      tpds = 1. + ng - n
      r = 2. * tpds / (ng - yi.dot(gi))
      if tpds < 1e-3 and r > 0.8 and r < 1.2:
        trivial = True
        break
    switch = g2 / g2km1 > epsr and g2 > epsl and g2 < epsu
  if not trivial and isfinite(g2):
    if g2 < tol:
      TPD = -log(n)
      return TPD, ki
    elif k < maxiter:
      hi = lnphiyi + np_log(yi)
      sqrtni = np_sqrt(ni)
      alphai = 2. * sqrtni
      lnphixi, dlnphixidnj = eos.getPT_lnphii_dnj(P, T, xi, n)
      logger.debug(h2)
      logger.debug(tmpl, k, *alphai, g2, 'newt')
      while g2 > tol and k < maxiter:
        H = sqrtni[:,None] * sqrtni * dlnphixidnj
        np_fill_diagonal(H, H.diagonal() + .5 * gi + 1.)
        # TODO: Replace the LU-solver with the custom implementation
        #       of the modified Cholesky decomposition solver.
        dalphai = linsolver(H, -sqrtni * gi)
        k += 1
        alphai += dalphai
        sqrtni = alphai * .5
        ni = sqrtni * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, dlnphixidnj = eos.getPT_lnphii_dnj(P, T, xi, n)
        gi = np_log(ni) + lnphixi - hi
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *alphai, g2, 'newt')
      if g2 < tol and isfinite(g2):
        TPD = -log(n)
        return TPD, ni / yi
  if trivial:
    TPD = -log(n)
    return TPD, ki
  logger.warning(
    'The stability test completed unsuccessfully.\n'
    'The solver was "_stabPT_ssnewt".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvi0 = %s',
    eos.name, P, T, yi.tolist(), kvi0.tolist(),
  )
  raise StabConvergenceError()


def _stabPT_qnssnewt(
  eos: StabSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 100,
  lmbdmax: float = 30.,
  checktrivial: bool = True,
  switchers: tuple[float, float, float] = (0.1, 1e-12, 1e-4),
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float]]:
  r"""Solve the one-phase stability problem formulated for the PT-
  thermodynamics using quasi-newton successive substitution iterations
  (QNSS) and Newton's method.

  Parameters
  ----------
  eos: StabSolverPTEos
    An initialized instance of a PT-based equation of state of type
    `StabSolverPTEos`.

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
    of the gradient of the tangent-plane distance function is less
    than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `100`.

  lmbdmax: float
    The maximum step length of the QNSS-method. Default is `30.0`.

  checktrivial: bool
    A flag indicating whether it is necessary to perform the
    check for early detection of convergence to the trivial
    solution. It is based on the paper of M.L. Michelsen, 1982
    (doi: 10.1016/0378-3812(82)85001-2), but differs in conditions
    due to the QNSS-method. Default is `True`.

  switchers: tuple[float, float, float]
    Allows to modify the conditions of switching from the QNSS-method
    to Newton's method. The parameter must be represented as a tuple
    containing three positive values: :math:`\eps_r`, :math:`\eps_l`,
    :math:`\eps_u`. The switching conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
      \end{cases}

    where :math:`\mathbf{g}` is a vector of the gradient of the TPD
    function, :math:`k` is the iteration number. Analytical expressions
    of the switching conditions were taken from the paper of L.X. Nghiem
    et al, 1983 (doi: 10.2118/8285-PA). Default is `(0.1, 1e-12, 1e-4)`.

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
  - the TPD-function value at a found solution (a local minimum of the
    TPD-function),
  - k-values of components (which are the solution to the system of
    nonlinear equations) as a `Vector[Float]` of shape `(Nc,)`.

  Raises
  ------
  StabConvergenceError
    This exception is raised if the algorithm does not converge.

  Notes
  -----
  Details of the implemented algorithm:

  - The initial part of the algorithm is quasi-newton successive
    substitution iterations (the QNSS-method), which are used to
    improve an initial guess.

  - The subsequent application of Newton's method minimizes the
    Michelsen's modified TPD-function.

  Such the combination of the QNSS-method and Newton's method makes the
  algorithm robust and rapid. The solver can be transformed to either
  the successive substitution method or Newton's method by changing the
  switching criteria. For the details of the QNSS-method, see the paper
  of L.X. Nghiem and Y.-K. Li, 1984 (doi: 10.1016/0378-3812(84)80013-8).
  """
  logger.info('Stability Test (QNSS-Newton method).')
  Nc = eos.Nc
  logger.info('P = %.1f [Pa], T = %.2f [K], yi =' + Nc * ' %6.4f', P, T, *yi)
  tmpl = '%3s' + Nc * '%9s' + '%11s%9s'
  rangeNc = range(Nc)
  h1 = tmpl % ('Nit', *['lnkv%s' % s for s in rangeNc], 'g2', 'method')
  h2 = tmpl % ('Nit', *['alph%s' % s for s in rangeNc], 'g2', 'method')
  tmpl = '%3s' + Nc * '%9.4f' + '%11.2e%9s'
  epsr, epsl, epsu = switchers
  lnphiyi = eos.getPT_lnphii(P, T, yi)
  k = 0
  trivial = False
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  lnphixi = eos.getPT_lnphii(P, T, xi)
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  switch = g2 > epsl and g2 < epsu
  lmbd = 1.
  dlnki = -gi
  logger.debug(h1)
  logger.debug(tmpl, k, *lnki, g2, 'qnss')
  while not switch and g2 > tol and k < maxiter:
    k += 1
    tkm1 = dlnki.dot(gi)
    lnki += dlnki
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    lnphixi = eos.getPT_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2km1 = g2
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnki, g2, 'qnss')
    if g2 < tol:
      break
    if checktrivial:
      tpd = -log(n)
      if tpd > 0. and tpd < 1e-3:
        ng = ni.dot(gi)
        r = 2. * (1. + ng - n) / (ng - yi.dot(gi))
        if r > 0.8 and r < 1.2:
          trivial = True
          break
    switch = g2 / g2km1 > epsr and g2 > epsl and g2 < epsu
    if switch:
      break
    if k % Nc == 0:
      lmbd = 1.
      dlnki = -gi
    else:
      lmbd *= tkm1 / (dlnki.dot(gi) - tkm1)
      if lmbd < 0.:
        lmbd = -lmbd
      if lmbd > lmbdmax:
        lmbd = lmbdmax
      dlnki = -lmbd * gi
      max_dlnki = np_abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
  if not trivial and isfinite(g2):
    if g2 < tol:
      TPD = -log(n)
      return TPD, ki
    elif k < maxiter:
      hi = lnphiyi + np_log(yi)
      sqrtni = np_sqrt(ni)
      alphai = 2. * sqrtni
      lnphixi, dlnphixidnj = eos.getPT_lnphii_dnj(P, T, xi, n)
      logger.debug(h2)
      logger.debug(tmpl, k, *alphai, g2, 'newt')
      while g2 > tol and k < maxiter:
        H = sqrtni[:,None] * sqrtni * dlnphixidnj
        np_fill_diagonal(H, H.diagonal() + .5 * gi + 1.)
        # TODO: Replace the LU-solver with the custom implementation
        #       of the modified Cholesky decomposition solver.
        dalphai = linsolver(H, -sqrtni * gi)
        k += 1
        alphai += dalphai
        sqrtni = alphai * .5
        ni = sqrtni * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, dlnphixidnj = eos.getPT_lnphii_dnj(P, T, xi, n)
        gi = np_log(ni) + lnphixi - hi
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *alphai, g2, 'newt')
      if g2 < tol and isfinite(g2):
        TPD = -log(n)
        return TPD, ni / yi
  if trivial:
    TPD = -log(n)
    return TPD, ki
  logger.warning(
    'The stability test completed unsuccessfully.\n'
    'The solver was "_stabPT_qnssnewt".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvi0 = %s',
    eos.name, P, T, yi.tolist(), kvi0.tolist(),
  )
  raise StabConvergenceError()


def _stabPT_ssbfgs(
  eos: StabSolverPTEos,
  P: float,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 200,
  checktrivial: bool = True,
  switchers: tuple[float, float, float] = (0.1, 1e-12, 1e-4),
) -> tuple[float, Vector[Float]]:
  r"""Solve the one-phase stability problem formulated for the PT-
  thermodynamics using successive substitution iterations and the BFGS
  method.

  Parameters
  ----------
  eos: StabSolverPTEos
    An initialized instance of a PT-based equation of state of type
    `StabSolverPTEos`.

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
    of the gradient of the tangent-plane distance function is less
    than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of iterations (total number, for both methods).
    Default is `200`.

  checktrivial: bool
    A flag indicating whether it is necessary to perform the
    check for early detection of convergence to the trivial
    solution. It is based on the paper of M.L. Michelsen, 1982
    (doi: 10.1016/0378-3812(82)85001-2). Default is `True`.

  switchers: tuple[float, float, float]
    Allows to modify the conditions of switching from the successive
    substitution iterations to the BFGS method. The parameter must be
    represented as a tuple containing three positive values:
    :math:`\eps_r`, :math:`\eps_l`, :math:`\eps_u`. The switching
    conditions are:

    .. math::

      \begin{cases}
        \frac{\left(\mathbf{g}^\top\mathbf{g}\right)^{k  }}
             {\left(\mathbf{g}^\top\mathbf{g}\right)^{k-1}} > \eps_r, \\
        \eps_l < \left(\mathbf{g}^\top\mathbf{g}\right)^k < \eps_u, \\
      \end{cases}

    where :math:`\mathbf{g}` is a vector of the gradient of the TPD
    function, :math:`k` is the iteration number. Analytical expressions
    of the switching conditions were taken from the paper of L.X. Nghiem
    et al, 1983 (doi: 10.2118/8285-PA). Default is `(0.1, 1e-12, 1e-4)`.

  Returns
  -------
  A tuple containing:
  - the TPD-function value at a found solution (a local minimum of the
    TPD-function),
  - k-values of components (which are the solution to the system of
    nonlinear equations) as a `Vector[Float]` of shape `(Nc,)`.

  Raises
  ------
  StabConvergenceError
    This exception is raised if the algorithm does not converge.

  Notes
  -----
  Details of the implemented algorithm:

  - The initial part of the algorithm is successive substitution
    iterations, which are used to improve an initial guess.

  - The subsequent application of the BFGS method minimizes the
    Michelsen's modified TPD-function.

  Such the combination of successive substitution iterations and
  the BFGS-method makes the algorithm robust and rapid. The solver can
  be transformed to either the successive substitution method or the
  BFGS-method by changing the switching criteria. For the details of
  the algorithm, see the paper of H. Hoteit and A. Firoozabadi, 2006
  (doi: 10.1002/aic.10908).
  """
  logger.info('Stability Test (SS-BFGS method).')
  Nc = eos.Nc
  logger.info('P = %.1f [Pa], T = %.2f [K], yi =' + Nc * ' %6.4f', P, T, *yi)
  tmpl = '%3s' + Nc * '%9s' + '%11s%9s'
  rangeNc = range(Nc)
  h1 = tmpl % ('Nit', *['lnkv%s' % s for s in rangeNc], 'g2', 'method')
  h2 = tmpl % ('Nit', *['alph%s' % s for s in rangeNc], 'g2', 'method')
  tmpl = '%3s' + Nc * '%9.4f' + '%11.2e%9s'
  epsr, epsl, epsu = switchers
  lnphiyi = eos.getPT_lnphii(P, T, yi)
  hi = lnphiyi + np_log(yi)
  k = 0
  trivial = False
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  lnphixi = eos.getPT_lnphii(P, T, xi)
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  switch = g2 > epsl and g2 < epsu
  logger.debug(h1)
  logger.debug(tmpl, k, *lnki, g2, 'ss')
  while not switch and g2 > tol and k < maxiter:
    k += 1
    lnki -= gi
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    lnphixi = eos.getPT_lnphii(P, T, xi)
    gi = lnki + lnphixi - lnphiyi
    g2km1 = g2
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnki, g2, 'ss')
    if checktrivial:
      if g2 < tol:
        break
      ng = ni.dot(gi)
      tpds = 1. + ng - n
      r = 2. * tpds / (ng - yi.dot(gi))
      if tpds < 1e-3 and r > 0.8 and r < 1.2:
        trivial = True
        break
    switch = g2 / g2km1 > epsr and g2 > epsl and g2 < epsu
  if not trivial and isfinite(g2):
    if g2 < tol:
      TPD = -log(n)
      return TPD, ki
    elif k < maxiter:
      hi = lnphiyi + np_log(yi)
      sqrtni = np_sqrt(ni)
      alphai = 2. * sqrtni
      gik = sqrtni * gi
      logger.debug(h2)
      logger.debug(tmpl, k, *alphai, g2, 'bfgs')
      si = -gik
      k += 1
      alphai += si
      sqrtni = alphai * 0.5
      ni = sqrtni * sqrtni
      n = ni.sum()
      xi = ni / n
      lnphixi = eos.getPT_lnphii(P, T, xi)
      gi = np_log(ni) + lnphixi - hi
      gikm1 = gik
      gik = sqrtni * gi
      g2 = gi.dot(gi)
      logger.debug(tmpl, k, *alphai, g2, 'bfgs')
      if g2 < tol and isfinite(g2):
        TPD = -log(n)
        return TPD, ni / yi
      if checktrivial:
        ng = ni.dot(gi)
        tpds = 1. + ng - n
        r = 2. * tpds / (ng - yi.dot(gi))
        if tpds < 1e-3 and r > 0.8 and r < 1.2:
          trivial = True
          TPD = -log(n)
          return TPD, ni / yi
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
        lnphixi = eos.getPT_lnphii(P, T, xi)
        gi = np_log(ni) + lnphixi - hi
        gikm1 = gik
        gik = sqrtni * gi
        g2 = gi.dot(gi)
        logger.debug(tmpl, k, *alphai, g2, 'bfgs')
        if checktrivial:
          if g2 < tol:
            break
          ng = ni.dot(gi)
          tpds = 1. + ng - n
          r = 2. * tpds / (ng - yi.dot(gi))
          if tpds < 1e-3 and r > 0.8 and r < 1.2:
            trivial = True
            break
      if not trivial and g2 < tol and isfinite(g2):
        TPD = -log(n)
        return TPD, ni / yi
  if trivial:
    TPD = -log(n)
    return TPD, ni / yi
  logger.warning(
    'The stability test completed unsuccessfully.\n'
    'The solver was "_stabPT_ssbfgs".\nEOS: "%s".\nParameters:\n'
    'P = %s [Pa]\nT = %s [K]\nyi = %s\nkvi0 = %s',
    eos.name, P, T, yi.tolist(), kvi0.tolist(),
  )
  raise StabConvergenceError()


class stabtest(object):
  def __init__(
    self,
    eps: float = -1e-8,
    breakunstab: bool = False,
    state: State | None = None,
  ) -> None:
    """Specify settings for the stability test.

    Parameters
    ----------
    eps: float
      The one-phase state of a mixture will be considered unstable if
      a value at a local minimum of the tangent-plane distance function
      (TPD), calculated by the selected solver, is less than `eps`.
      Default is `-1e-8`.

    breakunstab: bool | None
      A boolean flag indicating whether it is allowed to break the
      loop of various initial guesses checking if a one-phase state
      was identified as unstable. This option should be activated to
      terminate the search for the global minimum of the TPD function
      if a local minimum with a negative value is found. Default is
      `False`.

    state: State | None
      A thermodynamic state of a mixture, k-values from which can be
      used as an initial guess for the stability test procedure. If it
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
    self.eps = eps
    self.breakunstab = breakunstab
    self.state = state
    pass

  def __call__(
    self,
    eos: StabPTEos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    n: float = 1.,
    init: Iterable[Vector[Float]] | State | None = None,
    solver: StabSolver | str = 'default',
    eps: float | None = None,
    breakunstab: bool | None = None,
    useprev: bool | None = None,
  ) -> State:
    """Perform the one-phase stability test of a mixture for a given
    composition and two thermodynamic parameters.

    Parameters
    ----------
    eos: StabPTEos
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
      This parameter is used to initialize the stability test. The
      detailed explanation of the logic behind different types of
      `init` is given in the following table:

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
      |                         | stability test. In addition to these |
      |                         | k-values, initial guesses are also   |
      |                         | obtained from the `eos`.             |
      +-------------------------+--------------------------------------+
      | None                    | The `eos` is used to prepare initial |
      |                         | guesses of k-values.                 |
      +-------------------------+--------------------------------------+

      Default is `None`.

    solver: StabSolver | str
      A callable object that can solve the one-phase stability problem
      formulated for the given thermodynamic parameters. It also can be
      a string defining the name of an internal solver.

      For the PT-thermodynamics, the following internal solvers are
      available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'ss-bfgs'`     | BFGS method with preceding successive        |
      |                 | substitution iterations.                     |
      +-----------------+----------------------------------------------+
      | `'ss-newton'`   | Newton's method with preceding successive    |
      |                 | substitution iterations.                     |
      +-----------------+----------------------------------------------+
      | `'qnss-newton'` | Newton's method with preceding quasi-newton  |
      |                 | successive substitution iterations.          |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      pair of basic variables of the one-phase stability problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | P, T            | QNSS-Newton (`'qnss-newton'`).               |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    eps: float | None
      The one-phase state of a mixture will be considered unstable if
      a value at a local minimum of the tangent-plane distance function
      (TPD), calculated by the selected solver, is less than `eps`.
      If it is `None` this parameter is equal to the value defined
      during the initialization of this callable object (which defaults
      to `-1e-8`). Default is `None`.

    breakunstab: bool | None
      A boolean flag indicating whether it is allowed to break the
      loop of various initial guesses checking if a one-phase state
      was identified as unstable. This option should be activated to
      terminate the search for the global minimum of the TPD function
      if a local minimum with a negative value is found. If it is `None`
      this flag is equal to the value defined during the initialization
      of this callable object (which defaults to `False`). Default is
      `None`.

    useprev: bool | None
      Allows to preserve previously calculated results (if the solution
      is non-trivial) and to use them as the first initial guess in the
      next run. If it is `None` this flag is equal to the value defined
      during the initialization of this callable object (which defaults
      to `False`). Default is `None`.

    Notes
    -----
    The following aspects of the procedure should be noted:

    1. The unconvergence of a given solver will be considered as an
    indicator of the one-phase state stability.

    2. Selection between different formulations of the one-phase
    stability problem is based on the `form` attribute of the `eos`.

    Returns
    -------
    Stability test results as an instance of `State`.
    """
    if eps is None:
      eps = self.eps
    if breakunstab is None:
      breakunstab = self.breakunstab
    if init is None:
      init = self.state
    if eos.form == 'PT':
      eos = cast(StabPTEos, eos)
      solverPT: StabSolver[StabSolverPTEos]
      if callable(solver):
        solverPT = cast(StabSolver[StabSolverPTEos], solver)
      elif solver == 'default' or solver == 'qnss-newton':
        solverPT = _stabPT_qnssnewt
      elif solver == 'ss-bfgs':
        solverPT = _stabPT_ssbfgs
      elif solver == 'ss-newton':
        solverPT = _stabPT_ssnewt
      else:
        raise ValueError(
          'Unknown stability test solver for the PT-thermodynamics: '
          f'"{solver}".'
        )
      return self.runPT(eos, P1, P2, yi, n, init, solverPT, eps, breakunstab)
    else:
      raise NotImplementedError(
        f'The {eos.form}-formulation of the stability test routine is '
        'not implemented yet.'
      )

  @classmethod
  def runPT(
    cls,
    eos: StabPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    init: Iterable[Vector[Float]] | State | None = None,
    solver: StabSolver[StabSolverPTEos] = _stabPT_qnssnewt,
    eps: float = -1e-8,
    breakunstab: bool = False,
  ) -> State:
    """Perform the one-phase stability test for a given pressure,
    temperature, and composition of a mixture based on the analysis
    of the Gibbs energy function.

    Parameters
    ----------
    eos: StabPTEos
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
      This parameter is used to initialize the stability test. The
      detailed explanation of the logic behind different types of
      `init` is given in the following table:

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
      |                         | stability test. In addition to these |
      |                         | k-values, initial guesses are also   |
      |                         | obtained from the `eos`.             |
      +-------------------------+--------------------------------------+
      | None                    | The `eos` is used to prepare initial |
      |                         | guesses of k-values.                 |
      +-------------------------+--------------------------------------+

      Default is `None`.

    solver: StabSolver[StabSolverPTEos]
      A callable object that can solve the one-phase stability problem
      formulated for the PT-thermodynamics. Default is
      `_stabPT_qnssnewt`.

    eps: float
      The one-phase state of a mixture will be considered unstable if
      a value at a local minimum of the tangent-plane distance function
      (TPD), calculated by the selected solver, is less than `eps`.
      Default is `-1e-8`.

    breakunstab: bool
      A boolean flag indicating whether it is allowed to break the
      loop of various initial guesses checking if a one-phase state
      was identified as unstable. This option should be activated to
      terminate the search for the global minimum of the TPD function
      if a local minimum with a negative value is found. Default is
      `False`.

    useprev: bool
      Allows to preserve previously calculated results (if the solution
      is non-trivial) and to use them as the first initial guess in the
      next run. Default is `False`.

    Notes
    -----
    The unconvergence of the selected solver will be considered as an
    indicator of the one-phase state stability.

    Returns
    -------
    Stability test results as an instance of `State`.
    """
    if isinstance(init, Iterable):
      kvji0 = init
    else:
      kvji0 = eos.getPT_kvguess(P, T, yi)
      if init is not None:
        kvjim1 = init.kvji
        if kvjim1 is not None:
          # TODO: Use extrapolation to improve the initial guess of
          #       k-values. For the details, see the paper of L.X.
          #       Nghiem and Y.-K. Li, 1990 (doi: 10.2118/13517-PA).
          kvji0 = (kvjim1.ravel(), *kvji0)
    TPDo = eps
    kvio: Vector[Float] | None = None
    for j, kvi in enumerate(kvji0):
      logger.debug('Initial guess of k-values #%s.', j)
      try:
        TPD, kvi = solver(eos, P, T, yi, kvi)
        logger.debug('TPD: %.3e.', TPD)
        if TPD < TPDo:
          TPDo = TPD
          kvio = kvi
          if breakunstab:
            logger.info('Local minimum of TPD: %.3e.', TPDo)
            return cls.outputPT(eos, P, T, yi, n, kvio)
      except StabConvergenceError:
        continue
    logger.info('Global minimum of TPD: %.3e.', TPDo)
    return cls.outputPT(eos, P, T, yi, n, kvio)

  @staticmethod
  def outputPT(
    eos: State2pPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
    kvi: Vector[Float] | None,
  ) -> State:
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
    if kvi is None:
      return OnePhaseState(
        eos.Nc, 1, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
        None,
      )
    else:
      kvji = np_atleast_2d(kvi)
      return MultiPhaseState(
        eos.Nc, 1, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
        kvji,
      )
