import logging

import numpy as np

from custom_types import (
  Scalar,
  Vector,
)

from constants import (
  R,
)

from flash import (
  Flash2pEosPT,
  flash2pPT,
)

from boundary import (
  PsatEosPT,
  PsatPT,
)


logger = logging.getLogger('lab')


class LabResult(dict):
  """Container for experiment simulation outputs with pretty-printing.

  Attributes
  ----------
  Ps: Vector, shape (Ns,)
    A 1d-array with pressures corresponding to the stages of an
    experiment, where `Ns` is the number of stages.

  ysji: Tensor, shape (Ns, Np, Nc)
    Mole fractions of components in each phase at each stage of an
    experiment as a `Tensor` of shape `(Ns, Np, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  Fsj: Matrix, shape (Ns, Np)
    Phase mole fractions at each stage of an experiment as a `Matrix`
    of shape `(Ns, Np)`.

  Zsj: Matrix, shape (Ns, Np)
    Phase compressibility factors at each stage of an experiment as
    a `Matrix` of shape `(Ns, Np)`.

  ns: Vector, shape (Ns,)
    Mole number of a system at each stage of an experiment.
  """
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      s = (f"Pressure steps:\n{self.Ps} Pa\n"
           f"Phase mole fractions:\n{self.Fsj.T}\n"
           f"Phase compressibility factors:\n{self.Zsj.T}")
    return s


def cvdPT(
  PP: Vector,
  T: Scalar,
  yi: Vector,
  Psat0: Scalar,
  eos: PsatEosPT,
  n0: Scalar = 1.,
  flashkwargs: dict = {},
  psatkwargs: dict = {},
) -> LabResult:
  """Constant volume depletion (CVD) experiment.

  The CVD experiment is usually performed for natural gas to simulate the
  processes encountered during the field development. The experiment is
  conducted according to the following procedure:

  1) The gas sample is prepared to ensure that its state corresponds to
     the dew point.
  2) Pressure is reduced by increasing the cell volume. Some amount of
     the condensate appears. The volume of that liquid phase is
     registered.
  3) Part of the gas is expelled from the cell until the volume of the
     cell equals the volume at the dew point.
  4) The gas collected is sent to a multistage separator to study its
     properties and composition.
  5) The process is repeated for several pressure steps.

  Parameters
  ----------
  PP: Vector, shape (Ns,)
    A 1d-array with pressures corresponding to the stages of an
    experiment, where `Ns` is the number of stages. All pressures above
    the saturation pressure will be ignored.

  T: Scalar
    The temperature held constant during the experiment.

  yi: Vector, shape (Nc,)
    The global mole fractions of the gas at the beginning of the
    experiment.

  Psat0: Scalar
    An initial guess of the saturation pressure. It may be derived from
    existing experimental data, if available. The initial guess of the
    saturation pressure would be refined by internal procedures.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> Iterable[Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a an iterable object of `Vector` of shape
      `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    If solution methods for the saturation pressure determination and
    flash calculations are based on Newton's method, then it also must
    have:

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

    - `getPT_lnphii_Z_dnj_dP(P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  n0: Scalar
    The initial amount of the fluid in a cell. Default is `1.0` mol.

  flashkwargs: dict
    Settings for the flash calculation procedure. Default is an empty
    dictionary.

  psatkwargs: dict
    Setting for the saturation pressure calculation procedure. Default
    is an empty dictionary.

  Returns
  -------
  Constant volume depletion simulation results as an instance of the
  `LabResult`.
  """
  logger.info('Constant volume depletion (CVD).')
  Nc = eos.Nc
  zi = yi.copy()
  logger.info('T = %.2f K, zi =' + Nc * '%7.4f', T, *zi)
  flash = flash2pPT(eos, **flashkwargs)
  Psatres = PsatPT(eos, **psatkwargs).run(Psat0, T, zi, True)
  Psat = Psatres.P
  Zj = Psatres.Zj
  logger.debug('Saturation pressure: %.1f Pa', Psat)
  n = n0
  V0 = n * Zj[0] * R * T / Psat
  PP_filtered = PP[PP < Psat]
  PP_out = np.hstack([Psat, PP_filtered])
  Npoints = PP_out.shape[0]
  yji_out = np.zeros(shape=(Npoints, 2, Nc))
  yji_out[0] = Psatres.yji
  Zj_out = np.zeros(shape=(Npoints, 2))
  Zj_out[0] = Zj
  Fj_out = np.zeros(shape=(Npoints, 2))
  Fj_out[0, 0] = 1.
  Fj_out[0, 1] = 0.
  n_out = np.zeros_like(PP_out)
  n_out[0] = n
  logger.debug(
    '%3s%12s' + (Nc + 3) * '%9s',
    'Nst', 'P, Pa', 'F0', 'F1', *['z%s' % s for s in range(Nc)], 'n, mol',
  )
  tmpl = '%3s%12.1f' + (Nc + 3) * '%9.4f'
  logger.debug(tmpl, 0, Psat, 1., 0., *zi, n)
  for i, P in enumerate(PP_filtered, 1):
    flashres = flash.run(P, T, zi)
    yji = flashres.yji
    Zj = flashres.Zj
    Fj = flashres.Fj
    yji_out[i] = yji
    Zj_out[i] = Zj
    Fj_out[i] = Fj
    n_out[i] = n
    Np = Fj.shape[0]
    nj = Fj * n
    V = R * T * nj.dot(Zj) / P
    rhog = P / (Zj[0] * R * T)
    dV = V - V0
    dng = dV * rhog
    dngi = yji[0] * dng
    ngi = yji[0] * nj[0]
    nginew = ngi - dngi
    ngnew = nj[0] - dng
    if Np == 2:
      logger.debug(tmpl, i, P, *Fj, *zi, n)
      n = nj[1] + ngnew
      zi = (nginew + nj[1] * yji[1]) / n
    else:
      logger.debug(tmpl, i, P, 1., 0., *zi, n)
      n = ngnew
  logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
  return LabResult(Ps=PP_out, ysji=yji_out, Zsj=Zj_out, Fsj=Fj_out, ns=n_out)


def ccePT(
  PP: Vector,
  T: Scalar,
  yi: Vector,
  eos: Flash2pEosPT,
  n0: Scalar = 1.,
  flashkwargs: dict = {},
) -> LabResult:
  """Constant composition (mass) expansion (CCE) experiment.

  A sample of the reservoir fluid (natural gas or oil) is placed in a
  cell. Pressure is adjusted to a value equal to or greater than the
  initial reservoir pressure. The experiment is conducted at the
  constant temperature equal to the reservoir temperature. Pressure is
  reduced by increasing the volume of the cell. No gas or liquid is
  removed from the cell.

  At each stage, the pressure and total volume of the reservoir fluid
  (oil and gas) are measured. Additional phase properties that can be
  determined include the liquid phase volume, oil and gas densities,
  viscosities, compressibility factors, etc.

  Parameters
  ----------
  PP: Vector, shape (Ns,)
    A 1d-array with pressures corresponding to the stages of an
    experiment, where `Ns` is the number of stages.

  T: Scalar
    The temperature held constant during the experiment.

  yi: Vector, shape (Nc,)
    The global mole fractions of the reservoir fluid.

  eos: Flash2pEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> Iterable[Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as an iterable object of `Vector` of shape
      `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    If the solution method for flash calculations would be one of
    `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also must have:

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

  n0: Scalar
    The initial amount of the fluid in a cell. Default is `1.0` mol.

  flashkwargs: dict
    Settings for the flash calculation procedure. Default is an empty
    dictionary.

  Returns
  -------
  Constant composition expansion simulation results as an instance of
  the `LabResult`.
  """
  logger.info('Constant composition expansion (CCE).')
  Nc = eos.Nc
  Npoints = PP.shape[0]
  logger.info('T = %.2f K, zi =' + Nc * '%7.4f', T, *yi)
  yji_out = np.zeros(shape=(Npoints, 2, Nc))
  Zj_out = np.zeros(shape=(Npoints, 2))
  Fj_out = np.zeros(shape=(Npoints, 2))
  n_out = np.zeros_like(PP)
  flash = flash2pPT(eos, **flashkwargs)
  logger.debug('%3s%12s%9s%9s%9s%9s', 'Nst', 'P, Pa', 'F0', 'F1', 'Z0', 'Z1')
  tmpl = '%3s%12.1f%9.4f%9.4f%9.4f%9.4f'
  for i, P in enumerate(PP):
    flashres = flash.run(P, T, yi)
    yji = flashres.yji
    Zj = flashres.Zj
    Fj = flashres.Fj
    yji_out[i] = yji
    Zj_out[i] = Zj
    Fj_out[i] = Fj
    n_out[i] = n0
    Np = Fj.shape[0]
    if Np == 2:
      logger.debug(tmpl, i, P, *Fj, *Zj)
    else:
      logger.debug(tmpl, i, P, 1., 0., Zj[0], 0.)
  return LabResult(Ps=PP, ysji=yji_out, Zsj=Zj_out, Fsj=Fj_out, ns=n_out)
