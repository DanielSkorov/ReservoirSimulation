import logging

from dataclasses import (
  dataclass,
)

import numpy as np

from flash import (
  EosFlash2pPT,
  FlashResult,
  flash2pPT,
)

from envelope import (
  EosPsatPT,
  PsatPT,
)

from constants import (
  R,
)

from typing import (
  Protocol,
  Iterable,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Tensor,
  IVector,
)


logger = logging.getLogger('lab')


class EosSeparator(EosFlash2pPT):

  def getPT_Z(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector
  ) -> tuple[int, Scalar]: ...


class Separator(Protocol):
  Ps: Iterable[Scalar] | Scalar
  Ts: Iterable[Scalar] | Scalar

  def run(self, yi: Vector) -> FlashResult: ...


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


class SimpleSeparator(object):
  """Two-phase simple separator.

  Performs flash calculation for specific pressure and temperature, and
  returns its outputs as an instance of `FlashResult`.

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

  Ps: Scalar
    Pressure [Pa] for which flash calculations should be performed.
    Default is `101325.0` Pa.

  Ts: Scalar
    Temperatures [K] for which flash calculations should be performed.
    Default is `293.15` K.

  **kwargs: dict
    Other arguments for a flash solver. It may contain such arguments
    as `method`, `tol`, `maxiter` and others, depending on the selected
    solver.
  """
  def __init__(
    self,
    eos: EosFlash2pPT,
    Ps: Scalar = 101325.,
    Ts: Scalar = 293.15,
    **kwargs,
  ) -> None:
    self.Ps = Ps
    self.Ts = Ts
    self.solver = flash2pPT(eos, **kwargs)
    pass

  def run(self, yi: Vector) -> FlashResult:
    """Performs two-phase flash calculations to obtain compositions
    of phases at separator conditions.

    Parameters
    ----------
    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Separator calculation results as an instance of `FlashResult`.
    """
    return self.solver.run(self.Ps, self.Ts, yi)


class GasSeparator(object):
  """Two-phase gas separator that can be used to simulate low-
  temperature separation. At each stage of the separation process the
  liquid phase is removed into the stock tank conditions. The gas phase
  collected from one stage is transferred to the subsequent stage. The
  gas obtained from the liquid phase under stock tank conditions is
  combined with the primary gas flow. The stock tank conditions are the
  pressure and temperature at the last stage.

  Parameters
  ----------
  eos: EosSeparator
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_Z(P: Scalar, T: Scalar, yi: Vector) -> tuple[int, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return the
      designated phase of the fluid and the compressibility factor.

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

  Ps: Scalar
    Pressure [Pa] at stages of the separation process.
    Default is `[7e6, 4.5e6, 2e6, 101325.0]` Pa.

  Ts: Scalar
    Temperatures [K] at stages of the separation process.
    Default is `[223.15, 238.15, 258.15, 293.15]` K.

  **kwargs: dict
    Other arguments for a flash solver. It can contain such arguments
    as `method`, `tol`, `maxiter` and others, depending on the selected
    solver.
  """
  def __init__(
    self,
    eos: EosSeparator,
    Ps: Iterable[Scalar] = [7e6, 4.5e6, 2e6, 101325.],
    Ts: Iterable[Scalar] = [223.15, 238.15, 258.15, 293.15],
    **kwargs,
  ) -> None:
    self.Ps = Ps
    self.Ts = Ts
    self.eos = eos
    self.solver = flash2pPT(eos, **kwargs)
    pass

  def run(self, yi: Vector) -> FlashResult:
    """Performs a series of flash calculations, collecting the liquid
    phase in stock tank conditions.

    Parameters
    ----------
    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Separator calculation results as an instance of `FlashResult`.
    """
    ng = 1.
    ygi = yi
    noi = np.zeros(shape=(self.eos.Nc,))
    for P, T in zip(self.Ps, self.Ts):
      flash = self.solver.run(P, T, ygi)
      ygi = flash.yji[0]
      nj = ng * flash.Fj
      if flash.sj.shape[0] > 1:
        noi += flash.yji[1] * nj[1]
      ng = nj[0]
    no = noi.sum()
    if no > 0.:
      Zg = flash.Zj[0]
      sg = flash.sj[0]
      yoi = noi / no
      flash = self.solver.run(P, T, yoi)
      if flash.sj.shape[0] > 1:
        nsg = no * flash.Fj[0]
        ngi = ng * ygi
        ngi += nsg * flash.yji[0]
        ng += nsg
        ygi = ngi / ng
        sg, Zg = self.eos.getPT_Z(P, T, ygi)
        nj = np.array([ng, no - nsg])
        Fj = nj / nj.sum()
        yji = np.vstack([ygi, flash.yji[1]])
        Zj = np.array([Zg, flash.Zj[1]])
        sj = np.array([sg, flash.sj[1]])
        return FlashResult(yji, Fj, Zj, sj)
      elif flash.sj[0] == 0:
        nsg = no
        ngi = ng * ygi
        ngi += nsg * flash.yji[0]
        ng += nsg
        ygi = ngi / ng
        sg, Zg = self.eos.getPT_Z(P, T, ygi)
        Fj = np.array([1.])
        yji = np.atleast_2d(ygi)
        Zj = np.array([Zg])
        sj = np.array([sg])
        return FlashResult(yji, Fj, Zj, sj)
      else:
        nj = np.array([ng, no])
        Fj = nj / nj.sum()
        yji = np.vstack([ygi, flash.yji[0]])
        Zj = np.array([Zg, flash.Zj[0]])
        sj = np.array([sg, flash.sj[0]])
        return FlashResult(yji, Fj, Zj, sj)
    else:
      return flash


class cvdPT(object):
  """Two-phase constant volume depletion (CVD) experiment.

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
  eos: EosPsatPT
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

  gassep: Separator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) at the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the gas mixture collected from the
    cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  oilsep: Separator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) at the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the oil mixture at the current stage
    of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].

  psatkwargs: dict
    Settings for the saturation pressure calculation procedure. Default
    is an empty dictionary.

  **kwargs: dict
    Other arguments for a flash solver. It can contain such arguments
    as `method`, `tol`, `maxiter` and others, depending on the selected
    solver.
  """
  def __init__(
    self,
    eos: EosPsatPT,
    gassep: Separator,
    oilsep: Separator,
    mwc5: Scalar = 0.07215,
    psatkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.gassep = gassep
    self.oilsep = oilsep
    self.solver = flash2pPT(eos, **kwargs)
    self.psatsolver = PsatPT(eos, **psatkwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    if isinstance(gassep.Ts, Iterable) and isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps[-1]
    elif isinstance(gassep.Ts, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps
    elif isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts / gassep.Ps[-1]
    else:
      self.TPgsc = gassep.Ts / gassep.Ps
    if isinstance(oilsep.Ts, Iterable) and isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps[-1]
    elif isinstance(oilsep.Ts, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps
    elif isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts / oilsep.Ps[-1]
    else:
      self.TPosc = oilsep.Ts / oilsep.Ps
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
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *yi)
    psatres = self.psatsolver.run(Psat0, T, yi, True)
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
      mix = self.solver.run(P, T, yi)
      ysji[i] = mix.yji
      Zsj[i] = mix.Zj
      Fsj[i] = mix.Fj
      if mix.sj.shape[0] == 2:
        yrgi, yroi = mix.yji
        nrg, nro = mix.Fj * n
        vrg, vro = mix.Zj * R * T / P
        Vrg = nrg * vrg
        Vro = nro * vro
        Vrm = Vrg + Vro
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
        yi = (nrgi_sp1 + nro * yroi) / n
      elif mix.sj[0] == 0:
        yrgi = yi
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
      ns[i] = n
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
    mix = self.gassep.run(yrgi)
    if mix.sj.shape[0] > 1:
      vj = mix.Zj * (R * self.TPgsc)
      Vj = mix.Fj * vj
      bg = vrg / Vj[0]
      mug = -1.
      mwj = mix.yji.dot(self.eos.mwi)
      deng, denc = mwj / vj
      cc = mix.Fj[1] * mwj[1] / Vj[0]
      c5p = yrgi[self.maskc5pi].dot(self.mwc5pi) / Vj[0]
      return bg, mug, deng, cc, c5p, denc
    elif mix.sj[0] == 0:
      vg = mix.Zj[0] * R * self.TPgsc
      bg = vrg / vg
      mug = -1.
      deng = mix.yji[0].dot(self.eos.mwi) / vg
      cc = 0.
      c5p = yrgi[self.maskc5pi].dot(self.mwc5pi) / vg
      return bg, mug, deng, 0., c5p, -1.
    else:
      vc = mix.Zj[0] * R * self.TPgsc
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
    mix = self.oilsep.run(yroi)
    if mix.sj.shape[0] > 1:
      vj = mix.Zj * (R * self.TPosc)
      Vj = mix.Fj * vj
      bo = vro / Vj[1]
      muo = -1.
      deng, deno = mix.yji.dot(self.eos.mwi) / vj
      gor = mix.Fj[0] * mix.Zj[0] / (mix.Fj[1] * mix.Zj[1])
      return bo, muo, deno, gor, deng
    elif mix.sj[0] == 1:
      vo = mix.Zj[0] * R * self.TPosc
      bo = vro / vo
      muo = -1.
      deno = mix.yji[0].dot(self.eos.mwi) / vo
      gor = 0.
      return bo, muo, deno, gor, -1.
    else:
      vg = mix.Zj[0] * R * self.TPosc
      deng = mix.yji[0].dot(self.eos.mwi) / vg
      return -1., -1., -1., -1., deng


class ccePT(cvdPT):
  """Two phase constant composition (mass) expansion (CCE) experiment.

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

  gassep: Separator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) at the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the gas mixture collected from the
    cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  oilsep: Separator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) at the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the oil mixture at the current stage
    of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].

  **kwargs: dict
    Other arguments for a flash solver. It can contain such arguments
    as `method`, `tol`, `maxiter` and others, depending on the selected
    solver.

  Methods
  -------
  run(PP: Vector, T: Scalar, yi: Vector) -> LabResult
  This method performs the CCE experiment for a given range of
  pressures [Pa], fixed temperature [K] and composition of the mixture.
  """
  def __init__(
    self,
    eos: EosPsatPT,
    gassep: Separator,
    oilsep: Separator,
    mwc5: Scalar = 0.07215,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.gassep = gassep
    self.oilsep = oilsep
    self.solver = flash2pPT(eos, **kwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    if isinstance(gassep.Ts, Iterable) and isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps[-1]
    elif isinstance(gassep.Ts, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps
    elif isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts / gassep.Ps[-1]
    else:
      self.TPgsc = gassep.Ts / gassep.Ps
    if isinstance(oilsep.Ts, Iterable) and isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps[-1]
    elif isinstance(oilsep.Ts, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps
    elif isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts / oilsep.Ps[-1]
    else:
      self.TPosc = oilsep.Ts / oilsep.Ps
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
  """Two phase differential liberation (DL) experiment.

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
  eos: EosPsatPT
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

  gassep: Separator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) at the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the gas mixture collected from the
    cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  oilsep: Separator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) at the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the oil mixture at the current stage
    of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].

  psatkwargs: dict
    Settings for the saturation pressure calculation procedure. Default
    is an empty dictionary.

  **kwargs: dict
    Other arguments for a flash solver. It can contain such arguments
    as `method`, `tol`, `maxiter` and others, depending on the selected
    solver.
  """
  def __init__(
    self,
    eos: EosPsatPT,
    gassep: Separator,
    oilsep: Separator,
    mwc5: Scalar = 0.07215,
    psatkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.gassep = gassep
    self.oilsep = oilsep
    self.solver = flash2pPT(eos, **kwargs)
    self.psatsolver = PsatPT(eos, **psatkwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    if isinstance(gassep.Ts, Iterable) and isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps[-1]
    elif isinstance(gassep.Ts, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps
    elif isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts / gassep.Ps[-1]
    else:
      self.TPgsc = gassep.Ts / gassep.Ps
    if isinstance(oilsep.Ts, Iterable) and isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps[-1]
    elif isinstance(oilsep.Ts, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps
    elif isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts / oilsep.Ps[-1]
    else:
      self.TPosc = oilsep.Ts / oilsep.Ps
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
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *yi)
    psatres = self.psatsolver.run(Psat0, T, yi, True)
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
      mix = self.solver.run(P, T, yi)
      yji = mix.yji
      Zj = mix.Zj
      Fj = mix.Fj
      ysji[i] = yji
      Zsj[i] = Zj
      Fsj[i] = Fj
      ns[i] = n
      if mix.sj.shape[0] == 2:
        vj = Zj * R * T / P
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[0], vj[0])
        bo, muo, dendo, gor, densg = self._oilprops(yji[1], vj[1])
        props[i] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
        yi = yji[1]
        n *= Fj[1]
      elif mix.sj[0] == 1:
        bo, muo, dendo, gor, densg = self._oilprops(yi, Zj[0] * R * T / P)
        props[i] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
      else:
        bg, mug, deng, cc, c5p, denc = self._gasprops(yi, Zj[0] * R * T / P)
        props[i] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
        logger.warning('There is no oil in a cell.'
                       'The DL experiment can not be completed.')
        break
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    res = LabResult(PP, ns, Fsj, ysji, Zsj, props)
    logger.info(res)
    return res


class swellPT(cvdPT):
  """Two-phase swelling experiment.

  This experiment is usually conducted with oil to determine:
  1) how much gas can be dissolved in oil at specific pressures,
  2) the change in the saturation pressure with gas dissolution,
  3) the swelling factor, which is the relative increase of the volume
     of the fluid.
  4) the first-contact miscibility pressure, which is the maximum point
     of the saturation pressure vs. injection gas mole fraction curve.

  The laboratory procedure can be briefly described as follows.
  The reservoir oil is loaded in a cell, and the temperature is set at
  the reservoir temperature. The bubble point of the oil and the
  corresponding volume are measured. A small amount of gas is
  transferred into the cell. A new saturation pressure is determined
  and a new saturation volume recorded. This process is repeated until
  the upper bound of injection-gas concentration is reached.

  Parameters
  ----------
  eos: EosPsatPT
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

    If solution method for the saturation pressure determination is
    based on Newton's method, then it also must have:

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

  gassep: Separator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) at the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the gas mixture collected from the
    cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  oilsep: Separator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) at the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the final separation step will be
    used to calculate properties of the oil mixture at the current stage
    of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector) -> FlashResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `FlashResult`.

    Also, this instance must have attributes:

    - `Ps: Iterable[Scalar] | Scalar`
      An iterable object of pressures [Pa] representing a sequence of
      pressures at separation steps or specific pressure for the
      separation process.

    - `Ts: Iterable[Scalar] | Scalar`
      An iterable object of temperatures [K] representing a sequence of
      temperatures at separation steps or specific temperature for the
      separation process.

  mwc5: Scalar
    Molar mass used to filter components to calculate potential oil-gas
    ratio. Default is `0.07215` [kg/mol].

  **kwargs: dict
    Other arguments for the saturation pressure solver. It can contain
    such arguments as `method`, `tol`, `maxiter` and others, depending
    on the selected solver.
  """
  def __init__(
    self,
    eos: EosPsatPT,
    gassep: Separator,
    oilsep: Separator,
    mwc5: Scalar = 0.07215,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.gassep = gassep
    self.oilsep = oilsep
    self.solver = PsatPT(eos, densort=False, **kwargs)
    self.maskc5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.maskc5pi]
    if isinstance(gassep.Ts, Iterable) and isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps[-1]
    elif isinstance(gassep.Ts, Iterable):
      self.TPgsc = gassep.Ts[-1] / gassep.Ps
    elif isinstance(gassep.Ps, Iterable):
      self.TPgsc = gassep.Ts / gassep.Ps[-1]
    else:
      self.TPgsc = gassep.Ts / gassep.Ps
    if isinstance(oilsep.Ts, Iterable) and isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps[-1]
    elif isinstance(oilsep.Ts, Iterable):
      self.TPosc = oilsep.Ts[-1] / oilsep.Ps
    elif isinstance(oilsep.Ps, Iterable):
      self.TPosc = oilsep.Ts / oilsep.Ps[-1]
    else:
      self.TPosc = oilsep.Ts / oilsep.Ps
    pass

  def run(
    self,
    P0: Scalar,
    T: Scalar,
    zi: Vector,
    yi: Vector,
    Fj: Vector,
  ) -> LabResult:
    """Performs the swelling experiment.

    Parameters
    ----------
    P0: Scalar
      Initial guess of the saturation pressure [Pa] of the reservoir
      fluid.

    T: Scalar
      Experiment temperature [K].

    zi: Vector, shape (Nc,)
      Mole fractions of components of the reservoir fluid as a `Vector`
      of shape `(Nc,)`, where `Nc` is the number of components.

    yi: Vector, shape (Nc,)
      Mole fractions of components of the injected fluid as a `Vector`
      of shape `(Nc,)`.

    Fj: Vector, shape (Ns,)
      Mole fractions of the injected fluid that need to be dissolved in
      the reservoir fluid. It should be represented as a `Vector` of
      shape `(Ns,)`, where `Ns` is the number of stages of the
      experiment.

    Returns
    -------
    Swelling simulation results as an instance of the `LabResult`.
    """
    logger.info('Swelling test.')
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *zi)
    xi = zi.copy()
    Ns = Fj.shape[0]
    Ps = np.zeros(shape=(Ns,))
    ysji = np.zeros(shape=(Ns, 2, self.eos.Nc))
    Zsj = np.zeros(shape=(Ns, 2))
    Fsj = np.zeros_like(Zsj)
    ns = np.zeros_like(Ps)
    props = np.empty(shape=(Ns, 11))
    for i, F in enumerate(Fj):
      xi = (1. - F) * zi + F * yi
      res = self.solver.run(P0, T, xi, True)
      P0 = res.P
      yji = res.yji
      Zj = res.Zj
      Ps[i] = P0
      ysji[i] = yji
      Zsj[i] = Zj
      ns[i] = 1. / (1. - F)
      vj = Zj * R * T / P0
      denj = yji.dot(self.eos.mwi) / vj
      if denj[0] < denj[1]:
        Fsj[i, 0] = 1.
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[0], vj[0])
        bo, muo, dendo, gor, densg = self._oilprops(yji[1], vj[1])
      else:
        Fsj[i, 1] = 1.
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[1], vj[1])
        bo, muo, dendo, gor, densg = self._oilprops(yji[0], vj[0])
      props[i] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
    res = LabResult(Ps, ns, Fsj, ysji, Zsj, props)
    logger.info(res)
    return res
