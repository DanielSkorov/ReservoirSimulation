import logging

from dataclasses import (
  dataclass,
)

import numpy as np

from flash import (
  Flash2pEosPT,
  flash2pPT,
)

from envelope import (
  PsatEosPT,
  PsatPT,
)

from constants import (
  R,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Tensor,
)


logger = logging.getLogger('lab')


@dataclass
class LabResult(object):
  """Container for experiment simulation outputs with pretty-printing.

  Attributes
  ----------
  Ps: Vector, shape (Ns,)
    A 1d-array with pressures [Pa] corresponding to the stages of an
    experiment, where `Ns` is the number of stages.

  ns: Vector, shape (Ns,)
    Mole number [mol] of a system at each stage of an experiment.

  Fsj: Matrix, shape (Ns, Np)
    Phase mole fractions at each stage of an experiment as a `Matrix`
    of shape `(Ns, Np)`.

  ysji: Tensor, shape (Ns, Np, Nc)
    Mole fractions of components in each phase at each stage of an
    experiment as a `Tensor` of shape `(Ns, Np, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  Zsj: Matrix, shape (Ns, Np)
    Phase compressibility factors at each stage of an experiment as
    a `Matrix` of shape `(Ns, Np)`.

  props: Matrix, shape (Ns, 11)
    A 2d-array containing values of properties. It includes (in the
    iteration order):

    0: formation volume factor [rm3/sm3] of the gas phase at each stage
       of an experiment,

    1: viscosity [cP] of the gas phase at each stage of an experiment,

    2: density [kg/sm3] of the gas phase at standard conditions,

    3: condensate-gas ratio [kg/sm3] of the gas phase at standard
       conditions,

    4: potential condensate-gas ratio [kg/sm3] of the gas phase at
       standard conditions (it is calculated as the ratio of the sum of
       masses of C5+ components in the gas phase at the pressure and
       temperature of the stage to the volume of the gas phase at
       standard conditions),

    5: density [kg/sm3] of the condensate at standard conditions,

    6: formation volume factor [rm3/sm3] of the oil phase at each stage
       of an experiment,

    7: viscosity [cP] of the oil phase at each stage of an experiment,

    8: density [kg/sm3] of the dead oil phase at standard conditions,

    9: gas-oil ratio [sm3/sm3] of the oil phase at standard conditions,

    10: density [kg/sm3] of the dissolved gas at standard conditions.
  """
  Ps: Vector
  ns: Vector
  Fsj: Matrix
  ysji: Tensor
  Zsj: Matrix
  props: Matrix

  def __repr__(cls):
    tmpl = ('%3s%9s%9s%9s%9s'
            '%12s%10s%11s%11s%11s%11s'
            '%12s%8s%11s%12s%12s\n')
    s = tmpl % (
      'Nst', 'P', 'Fg', 'Fo', 'n',
      'Bg', 'μg', 'Dg', 'Cc', 'C₅₊', 'Dc',
      'Bo', 'μo', 'Ddo', 'GOR', 'Dsg',
    )
    s += tmpl % (
      '', '[MPa]', '[fr]', '[fr]', '[mol]',
      '[rm3/sm3]', '[cP]', '[kg/sm3]', '[kg/sm3]', '[kg/sm3]', '[kg/sm3]',
      '[rm3/sm3]', '[cP]', '[kg/sm3]', '[sm3/sm3]', '[kg/sm3]',
    )
    tmpl = ('%3s%9.3f%9.4f%9.4f%9.4f'
            '%12.5f%10.5f%11.4f%11.4f%11.4f%11.1f'
            '%12.3f%8.3f%11.1f%12.1f%12.4f\n')
    for i in range(cls.Ps.shape[0]):
      s += tmpl % (i, cls.Ps[i] / 1e6, *cls.Fsj[i], cls.ns[i], *cls.props[i])
    return s


class cvdPT(object):
  """Constant volume depletion (CVD) experiment.

  The CVD experiment is usually performed with natural gas to simulate
  the processes encountered during the field development. The experiment
  is conducted according to the following procedure:

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
  eos: PsatEosPT
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

    - `getPT_Z_lnphii_dP(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector` of shape `(Nc,)`.

    If solution methods for the saturation pressure determination and
    flash calculations are based on Newton's method, then it also must
    have:

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

    - `getPT_Z_lnphii_dP_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to mole numbers of components as a `Matrix` of
        shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  Psc: Scalar
    Pressure at standard conditions used to calculate properties of
    phases formed during the experiment. Default is `101325.0` [Pa].

  Tsc: Scalar
    Temperature at standard conditions used to calculate properties of
    phases formed during the experiment. Default is `293.15` [K].

  flashkwargs: dict
    Settings for the flash calculation procedure. Default is an empty
    dictionary.

  psatkwargs: dict
    Settings for the saturation pressure calculation procedure. Default
    is an empty dictionary.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].
  """
  def __init__(
    self,
    eos: PsatEosPT,
    Psc: Scalar = 101325.,
    Tsc: Scalar = 293.15,
    flashkwargs: dict = {},
    psatkwargs: dict = {},
    mwc5: Scalar = 0.07215,
  ) -> None:
    self.eos = eos
    self.Psc = Psc
    self.Tsc = Tsc
    self.solver = flash2pPT(eos, **flashkwargs)
    self.solversc = flash2pPT(eos, **flashkwargs)
    self.psatsolver = PsatPT(eos, **psatkwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    pass

  def run(
    self,
    PP: Vector,
    T: Scalar,
    yi: Vector,
    Psat0: Scalar,
    n0: Scalar = 1.,
  ) -> LabResult:
    """Performs the CVD experiment for a given range of pressures, fixed
    temperature, composition of the mixture and initial guess of the
    saturation pressure.

    Parameters
    ----------
    PP: Vector, shape (Ns,)
      A 1d-array with pressures corresponding to the stages of an
      experiment, where `Ns` is the number of stages. All pressures
      above the saturation pressure will be ignored.

    T: Scalar
      The temperature held constant during the experiment.

    yi: Vector, shape (Nc,)
      The global mole fractions of a mixture at the beginning of the
      experiment.

    Psat0: Scalar
      An initial guess of the saturation pressure. It may be obtained
      from existing experimental data, if available. The initial guess
      of the saturation pressure would be refined by internal
      procedures.

    n0: Scalar
      The initial amount of the fluid in a cell. Default is `1.0` [mol].

    Returns
    -------
    Constant volume depletion simulation results as an instance of the
    `LabResult`.
    """
    logger.info('Constant volume depletion (CVD).')
    zi = yi.copy()
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *zi)
    psatres = self.psatsolver.run(Psat0, T, zi, True)
    Psat = psatres.P
    logger.info('Saturation pressure: %.1f Pa', Psat)
    vrg, vro = psatres.Zj * R * T / Psat
    V0 = n0 * vrg
    PP_filtered = PP[PP < Psat]
    Ps = np.hstack([Psat, PP_filtered])
    Ns = Ps.shape[0]
    ysji = np.zeros(shape=(Ns, 2, self.eos.Nc))
    Zsj = np.zeros(shape=(Ns, 2))
    Fsj = np.zeros_like(Zsj)
    ns = np.zeros_like(Ps)
    props = np.empty(shape=(Ns, 11))
    n = n0
    ysji[0] = psatres.yji
    Zsj[0] = psatres.Zj
    Fsj[0, 0] = 1.
    ns[0] = n
    bg, mug, deng, cc, c5p, denc = self._gasprops(psatres.yji[0], vrg)
    bo, muo, dendo, gor, densg = self._oilprops(psatres.yji[1], vro)
    props[0] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
    for i, P in enumerate(PP_filtered, 1):
      mix = self.solver.run(P, T, zi)
      ysji[i] = mix.yji
      Zsj[i] = mix.Zj
      Fsj[i] = mix.Fj
      ns[i] = n
      if mix.sj.shape[0] == 2:
        yrgi, yroi = mix.yji
        nrg, nro = mix.Fj * n
        vrg, vro = mix.Zj * R * T / P
        Vrg = nrg * vrg
        Vro = nro * vro
        Vrm = Vrg + Vro
        Srg = Vrg / Vrm
        Sro = Vro / Vrm
        dVrg = Vrm - V0
        dnrg = dVrg / vrg
        dnrgi = yrgi * dnrg
        nrgi = yrgi * nrg
        nrgi_sp1 = nrgi - dnrgi
        nrg_sp1 = nrg - dnrg
        bg, mug, deng, cc, c5p, denc = self._gasprops(yrgi, vrg)
        bo, muo, dendo, gor, densg = self._oilprops(yroi, vro)
        props[i] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
        if nrg_sp1 < 0.:
          logger.warning('There is no gas to remove. '
                         'The CVD experiment can not be completed.')
          break
        n = nro + nrg_sp1
        zi = (nrgi_sp1 + nro * yroi) / n
      elif mix.sj[0] == 0:
        yrgi = zi
        nrg = n
        vrg = R * T * mix.Zj[0] / P
        Vrm = nrg * vrg
        dVrg = Vrm - V0
        dnrg = dVrg / vrg
        dnrgi = yrgi * dnrg
        nrgi = yrgi * nrg
        nrgi_sp1 = nrgi - dnrgi
        nrg_sp1 = nrg - dnrg
        bg, mug, deng, cc, c5p, denc = self._gasprops(yrgi, vrg)
        props[i] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
        if nrg_sp1 < 0.:
          logger.warning('There is no gas to remove. '
                         'The CVD experiment can not be completed.')
          break
        n = nrg_sp1
      else:
        yroi = mix.yji[0]
        vro = R * T * mix.Zj[0] / P
        bo, muo, dendo, gor, densg = self._oilprops(yroi, vro)
        props[i] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
        logger.warning('There is no gas to remove.'
                       'The CVD experiment can not be completed.')
        break
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    res = LabResult(Ps, ns, Fsj, ysji, Zsj, props)
    logger.info(res)
    return res

  def _gasprops(
    self,
    yrgi: Vector,
    vrg: Scalar,
  ) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """Calculates properties of the gas phase at standard conditions.
    The flash calculation procedure is performed to compute fluid
    properties at standard pressure and temperature.

    Parameters
    ----------
    yrgi: Vector
      The composition of the gas phase at pressure and temperature
      corresponding to a certain stage of the experiment.

    vrg: Scalar
      Molar volume [m3/mol] of the gas phase at pressure and
      temperature corresponding to a certain stage of the experiment.

    Returns
    -------
    A tuple of:
    - gas formation volume factor [rm3/sm3],
    - gas viscosity [cP],
    - gas density [kg/sm3],
    - condensate-gas ratio (condensate solubility in gas) [kg/sm3],
    - potential condensate-gas ratio (potential condensate solubility
      in gas) [kg/sm3],
    - condensate density [kg/sm3].
    """
    mix = self.solversc.run(self.Psc, self.Tsc, yrgi)
    if mix.sj.shape[0] == 2:
      vj = mix.Zj * (R * self.Tsc / self.Psc)
      Vj = mix.Fj * vj
      bg = vrg / Vj[0]
      mug = -1.
      mwj = mix.yji.dot(self.eos.mwi)
      deng, denc = mwj / vj
      cc = mix.Fj[1] * mwj[1] / Vj[0]
      c5p = yrgi[self.maskc5pi].dot(self.mwc5pi) / Vj[0]
      return bg, mug, deng, cc, c5p, denc
    elif mix.sj[0] == 0:
      vg = mix.Zj[0] * R * self.Tsc / self.Psc
      bg = vrg / vg
      mug = -1.
      deng = mix.yji[0].dot(self.eos.mwi) / vg
      cc = 0.
      c5p = yrgi[self.maskc5pi].dot(self.mwc5pi) / vg
      return bg, mug, deng, 0., c5p, -1.
    else:
      vc = mix.Zj[0] * R * self.Tsc / self.Psc
      denc = mix.yji[0].dot(self.eos.mwi) / vc
      return -1., -1., -1., -1., -1., denc

  def _oilprops(
    self,
    yroi: Vector,
    vro: Scalar,
  ) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar]:
    """Calculates properties of the oil phase at standard conditions.
    The flash calculation procedure is performed to compute fluid
    properties at standard pressure and temperature.

    Parameters
    ----------
    yroi: Vector
      The composition of the oil phase at pressure and temperature
      corresponding to a certain stage of the experiment.

    vro: Scalar
      Molar volume [m3/mol] of the oil phase at pressure and
      temperature corresponding to a certain stage of the experiment.

    Returns
    -------
    A tuple of:
    - oil formation volume factor [rm3/sm3],
    - oil viscosity [cP],
    - dead oil density [kg/sm3],
    - gas-oil ratio (gas solubility in oil) [sm3/sm3],
    - dissolved gas density [kg/sm3].
    """
    mix = self.solversc.run(self.Psc, self.Tsc, yroi)
    if mix.sj.shape[0] == 2:
      vj = mix.Zj * (R * self.Tsc / self.Psc)
      Vj = mix.Fj * vj
      bo = vro / Vj[1]
      muo = -1.
      deng, deno = mix.yji.dot(self.eos.mwi) / vj
      gor = mix.Fj[0] * mix.Zj[0] / (mix.Fj[1] * mix.Zj[1])
      return bo, muo, deno, gor, deng
    elif mix.sj[0] == 1:
      vo = mix.Zj[0] * R * self.Tsc / self.Psc
      bo = vro / vo
      muo = -1.
      deno = mix.yji[0].dot(self.eos.mwi) / vo
      gor = 0.
      return bo, muo, deno, gor, -1.
    else:
      vg = mix.Zj[0] * R * self.Tsc / self.Psc
      deng = mix.yji[0].dot(self.eos.mwi) / vg
      return -1., -1., -1., -1., deng


class ccePT(cvdPT):
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
  eos: Flash2pEosPT
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

    If the solution method for flash calculations would be one of
    `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also must have:

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

  Psc: Scalar
    Pressure at surface conditions used to calculate properties of
    phases formed during the experiment. Default is `101325.0` [Pa].

  Tsc: Scalar
    Temperature at surface conditions used to calculate properties of
    phases formed during the experiment. Default is `273.15` [K].

  flashkwargs: dict
    Settings for the flash calculation procedure. Default is an empty
    dictionary.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].

  Methods
  -------
  run(PP: Vector, T: Scalar, yi: Vector) -> LabResult
  This method performs the CCE experiment for a given range of
  pressures [Pa], fixed temperature [K] and composition of the mixture.
  """
  def __init__(
    self,
    eos: PsatEosPT,
    Psc: Scalar = 101325.,
    Tsc: Scalar = 293.15,
    flashkwargs: dict = {},
    mwc5: Scalar = 0.07215,
  ) -> None:
    self.eos = eos
    self.Psc = Psc
    self.Tsc = Tsc
    self.solver = flash2pPT(eos, **flashkwargs)
    self.solversc = flash2pPT(eos, **flashkwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    pass

  def run(
    self,
    PP: Vector,
    T: Scalar,
    yi: Vector,
    n0: Scalar = 1.,
  ) -> LabResult:
    """Performs the CCE experiment for a given range of pressures, fixed
    temperature and composition of the mixturee.

    Parameters
    ----------
    PP: Vector, shape (Ns,)
      A 1d-array with pressures corresponding to the stages of an
      experiment, where `Ns` is the number of stages.

    T: Scalar
      The temperature held constant during the experiment.

    yi: Vector, shape (Nc,)
      The global mole fractions of the reservoir fluid.

    n0: Scalar
      The initial amount of the fluid in a cell. Default is `1.0` [mol].

    Returns
    -------
    Constant composition expansion simulation results as an instance of
    the `LabResult`.
    """
    logger.info('Constant composition expansion (CCE).')
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *yi)
    Ns = PP.shape[0]
    ysji = np.zeros(shape=(Ns, 2, self.eos.Nc))
    Zsj = np.zeros(shape=(Ns, 2))
    Fsj = np.zeros_like(Zsj)
    ns = np.zeros_like(PP)
    props = np.empty(shape=(Ns, 11))
    for i, P in enumerate(PP):
      mix = self.solver.run(P, T, yi)
      yji = mix.yji
      Zj = mix.Zj
      Fj = mix.Fj
      ysji[i] = yji
      Zsj[i] = Zj
      Fsj[i] = Fj
      ns[i] = n0
      if mix.sj.shape[0] == 2:
        vj = Zj * R * T / P
        Vj = Fj * vj
        Sj = Vj / Vj.sum()
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[0], vj[0])
        bo, muo, dendo, gor, densg = self._oilprops(yji[1], vj[1])
        props[i] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
      elif mix.sj[0] == 0:
        bg, mug, deng, cc, c5p, denc = self._gasprops(yi, Zj[0] * R * T / P)
        props[i] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
      else:
        bo, muo, dendo, gor, densg = self._oilprops(yi, Zj[0] * R * T / P)
        props[i] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
    res = LabResult(PP, ns, Fsj, ysji, Zsj, props)
    logger.info(res)
    return res


class dlPT(cvdPT):
  """Differential liberation (DL) experiment.

  The DL experiment is usually performed with an oil phase to simulate
  the processes encountered during the field development. The experiment
  is conducted according to the following procedure:

  1) The oil sample is prepared to ensure that its state corresponds to
     the dew point.
  2) Pressure is reduced by increasing the cell volume. Some amount of
     the gas appears. The volume of that liquid phase is registered.
  3) All gas is expelled from the cell.
  4) The gas collected is sent to a multistage separator to study its
     properties and composition.
  5) The process is repeated for several pressure steps.

  Parameters
  ----------
  eos: PsatEosPT
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

    - `getPT_Z_lnphii_dP(P: Scalar, T: Scalar, yi: Vector)
       -> tuple[int, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector` of shape `(Nc,)`.

    If solution methods for the saturation pressure determination and
    flash calculations are based on Newton's method, then it also must
    have:

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

    - `getPT_Z_lnphii_dP_dnj(P: Scalar, T: Scalar, yi: Vector, n: Scalar)
       -> tuple[int, Scalar, Vector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - the designated phase (`0` = vapour, `1` = liquid, etc.),
      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a `Vector`
        of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to mole numbers of components as a `Matrix` of
        shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  Psc: Scalar
    Pressure at standard conditions used to calculate properties of
    phases formed during the experiment. Default is `101325.0` [Pa].

  Tsc: Scalar
    Temperature at standard conditions used to calculate properties of
    phases formed during the experiment. Default is `293.15` [K].

  flashkwargs: dict
    Settings for the flash calculation procedure. Default is an empty
    dictionary.

  psatkwargs: dict
    Settings for the saturation pressure calculation procedure. Default
    is an empty dictionary.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].
  """
  def __init__(
    self,
    eos: PsatEosPT,
    Psc: Scalar = 101325.,
    Tsc: Scalar = 293.15,
    flashkwargs: dict = {},
    psatkwargs: dict = {},
    mwc5: Scalar = 0.07215,
  ) -> None:
    self.eos = eos
    self.Psc = Psc
    self.Tsc = Tsc
    self.solver = flash2pPT(eos, **flashkwargs)
    self.solversc = flash2pPT(eos, **flashkwargs)
    self.psatsolver = PsatPT(eos, **psatkwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    pass

  def run(
    self,
    PP: Vector,
    T: Scalar,
    yi: Vector,
    Psat0: Scalar,
    n0: Scalar = 1.,
  ) -> LabResult:
    """Performs the DL experiment for a given range of pressures, fixed
    temperature, composition of the mixture and initial guess of the
    saturation pressure.

    Parameters
    ----------
    PP: Vector, shape (Ns,)
      A 1d-array with pressures corresponding to the stages of an
      experiment, where `Ns` is the number of stages. All pressures
      above the saturation pressure will be ignored.

    T: Scalar
      The temperature held constant during the experiment.

    yi: Vector, shape (Nc,)
      The global mole fractions of a mixture at the beginning of the
      experiment.

    Psat0: Scalar
      An initial guess of the saturation pressure. It may be obtained
      from existing experimental data, if available. The initial guess
      of the saturation pressure would be refined by internal
      procedures.

    n0: Scalar
      The initial amount of the fluid in a cell. Default is `1.0` [mol].

    Returns
    -------
    Differential liberation simulation results as an instance of the
    `LabResult`.
    """
    logger.info('Differential liberation (DL).')
    zi = yi.copy()
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *zi)
    psatres = self.psatsolver.run(Psat0, T, zi, True)
    Psat = psatres.P
    logger.info('Saturation pressure: %.1f Pa', Psat)
    PP_filtered = PP[PP < Psat]
    Ps = np.hstack([Psat, PP_filtered])
    Ns = Ps.shape[0]
    ysji = np.zeros(shape=(Ns, 2, self.eos.Nc))
    Zsj = np.zeros(shape=(Ns, 2))
    Fsj = np.zeros_like(Zsj)
    ns = np.zeros_like(Ps)
    props = np.empty(shape=(Ns, 11))
    n = n0
    ysji[0] = psatres.yji
    Zsj[0] = psatres.Zj
    Fsj[0, 1] = 1.
    ns[0] = n
    vrg, vro = psatres.Zj * R * T / Psat
    bg, mug, deng, cc, c5p, denc = self._gasprops(psatres.yji[0], vrg)
    bo, muo, dendo, gor, densg = self._oilprops(psatres.yji[1], vro)
    props[0] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
    for i, P in enumerate(PP_filtered, 1):
      mix = self.solver.run(P, T, zi)
      yji = mix.yji
      Zj = mix.Zj
      Fj = mix.Fj
      ysji[i] = yji
      Zsj[i] = Zj
      Fsj[i] = Fj
      ns[i] = n
      if mix.sj.shape[0] == 2:
        vj = Zj * R * T / P
        Vj = Fj * vj
        Sj = Vj / Vj.sum()
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[0], vj[0])
        bo, muo, dendo, gor, densg = self._oilprops(yji[1], vj[1])
        props[i] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
        zi = yji[1]
        n *= Fj[1]
      elif mix.sj[0] == 1:
        bo, muo, dendo, gor, densg = self._oilprops(zi, Zj[0] * R * T / P)
        props[i] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
      else:
        bg, mug, deng, cc, c5p, denc = self._gasprops(zi, Zj[0] * R * T / P)
        props[i] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
        logger.warning('There is no oil in a cell.'
                       'The DL experiment can not be completed.')
        break
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    res = LabResult(PP, ns, Fsj, ysji, Zsj, props)
    logger.info(res)
    return res
