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

  bgs: Vector, shape (Ns,)
    Formation volume factors [m3/m3] of the gas phase at each stage of
    an experiment as a `Vector` of shape `(Ns,)`.

  ccs: Vector, shape (Ns,)
    Condensate-gas ratios [kg/m3] at standard conditions of the gas
    phase at each stage of an experiment as a `Vector` of shape `(Ns,)`.

  c5ps: Vector, shape (Ns,)
    Potential condensate-gas ratios [kg/m3] at standard conditions of
    the gas phase at each stage of an experiment as a `Vector` of shape
    `(Ns,)`. It is calculated as the ratio of the sum of masses of C5+
    components in the gas phase at the pressure and temperature of
    the stage to the volume of the gas phase at standard conditions.

  bos: Vector, shape (Ns,)
    Formation volume factors [m3/m3] of the oil phase at each stage of
    an experiment as a `Vector` of shape `(Ns,)`.

  gors: Vector, shape (Ns,)
    Gas-oil ratios [m3/m3] at standard conditions of the oil phase
    at each stage of an experiment as a `Vector` of shape `(Ns,)`.
  """
  Ps: Vector
  ns: Vector
  Fsj: Matrix
  ysji: Tensor
  Zsj: Matrix
  bgs: Vector
  ccs: Vector
  c5ps: Vector
  bos: Vector
  gors: Vector

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      return (f"Pressure steps:\n{self.Ps} Pa\n"
              f"Phase mole fractions:\n{self.Fsj.T}\n"
              f"Phase compressibility factors:\n{self.Zsj.T}")


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
    bgs = np.full_like(Ps, -1.)
    ccs = np.full_like(Ps, -1.)
    c5ps = np.full_like(Ps, -1.)
    mugs = np.full_like(Ps, -1.)
    bos = np.full_like(Ps, -1.)
    gors = np.full_like(Ps, -1.)
    muos = np.full_like(Ps, -1.)
    logger.info(
      '%3s%13s%9s%9s%9s%12s%11s%12s%10s%12s%13s%9s',
      'Nst', 'P, Pa', 'Sg, fr', 'So, fr', 'n, mol', 'Bg, m3/m3', 'Co, g/m3',
      'C5+, g/m3', 'μg, cP', 'Bo, m3/m3', 'GOR, m3/m3', 'μo, cP',
    )
    tmpl = '%3s%13.1f%9.4f%9.4f%9.4f%12.5f%11.3f%12.3f%10.5f%12.3f%13.2f%9.3f'
    n = n0
    ysji[0] = psatres.yji
    Zsj[0] = psatres.Zj
    Fsj[0, 0] = 1.
    ns[0] = n
    bg, cc, c5p, mug = self._gasprops(psatres.yji[0], vrg)
    bo, gor, muo = self._oilprops(psatres.yji[1], vro)
    bgs[0] = bg
    ccs[0] = cc
    c5ps[0] = c5p
    mugs[0] = mug
    bos[0] = bo
    gors[0] = gor
    muos[0] = muo
    logger.info(
      tmpl, 0, Psat, 1., 0., n, bg, cc * 1e3, c5p * 1e3, mug, bo, gor, muo,
    )
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
        bg, cc, c5p, mug = self._gasprops(yrgi, vrg)
        bo, gor, muo = self._oilprops(yroi, vro)
        bgs[i] = bg
        ccs[i] = cc
        c5ps[i] = c5p
        mugs[i] = mug
        bos[i] = bo
        gors[i] = gor
        muos[i] = muo
        logger.info(
          tmpl, i, P, Srg, Sro, n, bg, cc * 1e3, c5p * 1e3, mug, bo, gor, muo,
        )
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
        bg, cc, c5p, mug = self._gasprops(yrgi, vrg)
        bgs[i] = bg
        ccs[i] = cc
        c5ps[i] = c5p
        mugs[i] = mug
        logger.info(
          tmpl, i, P, 1., 0., n, bg, cc * 1e3, c5p * 1e3, mug, -1., -1., -1.,
        )
        if nrg_sp1 < 0.:
          logger.warning('There is no gas to remove. '
                         'The CVD experiment can not be completed.')
          break
        n = nrg_sp1
      else:
        yroi = mix.yji[0]
        vro = R * T * mix.Zj[0] / P
        bo, gor, muo = self._oilprops(yroi, vro)
        bos[i] = bo
        gors[i] = gor
        muos[i] = muo
        logger.info(tmpl, i, P, 0., 1., n, -1., -1., -1., -1., bo, gor, muo)
        logger.warning('There is no gas to remove.'
                     'The CVD experiment can not be completed.')
        break
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    return LabResult(Ps, ns, Fsj, ysji, Zsj, bgs, ccs, c5ps, bos, gors)

  def _gasprops(
    self,
    yrgi: Vector,
    vrg: Scalar,
  ) -> tuple[Scalar, Scalar, Scalar, Scalar]:
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
    - gas formation volume factor [m3/m3],
    - oil-gas ratio (oil solubility in gas) [kg/m3],
    - potential oil-gas ratio [kg/m3],
    - gas viscosity [cP].
    """
    gsc = self.solversc.run(self.Psc, self.Tsc, yrgi)
    if gsc.sj.shape[0] == 2:
      vsg = gsc.Fj[0] * gsc.Zj[0] * R * self.Tsc / self.Psc
      bg = vrg / vsg
      cc = gsc.Fj[1] * gsc.yji[1].dot(self.eos.mwi) / vsg
      c5p = yrgi[self.maskc5pi].dot(self.mwc5pi) / vsg
      mug = -1.
      return bg, cc, c5p, mug
    elif gsc.sj[0] == 0:
      vsg = gsc.Zj[0] * R * self.Tsc / self.Psc
      bg = vrg / vsg
      cc = 0.
      c5p = yrgi[self.maskc5pi].dot(self.mwc5pi) / vsg
      mug = -1.
      return bg, 0., c5p, mug
    else:
      return -1., -1., -1., -1.

  def _oilprops(
    self,
    yroi: Vector,
    vro: Scalar,
  ) -> tuple[Scalar, Scalar, Scalar]:
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
    - formation volume factor [m3/m3],
    - gas-oil ratio (gas solubility in oil) [m3/m3],
    - oil viscosity [cP].
    """
    osc = self.solversc.run(self.Psc, self.Tsc, yroi)
    if osc.sj.shape[0] == 2:
      bo = vro * self.Psc / (osc.Fj[1] * osc.Zj[1] * R * self.Tsc)
      gor = osc.Fj[0] * osc.Zj[0] / (osc.Fj[1] * osc.Zj[1])
      muo = -1.
      return bo, gor, muo
    elif osc.sj[0] == 1:
      bo = vro * self.Psc / (osc.Zj[0] * R * self.Tsc)
      gor = 0.
      muo = -1.
      return bo, gor, muo
    else:
      return -1., -1., -1.


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
    bgs = np.full_like(PP, -1.)
    ccs = np.full_like(PP, -1.)
    c5ps = np.full_like(PP, -1.)
    mugs = np.full_like(PP, -1.)
    bos = np.full_like(PP, -1.)
    gors = np.full_like(PP, -1.)
    muos = np.full_like(PP, -1.)
    logger.info(
      '%3s%13s%9s%9s%9s%12s%11s%12s%10s%12s%13s%9s',
      'Nst', 'P, Pa', 'Sg, fr', 'So, fr', 'n, mol', 'Bg, m3/m3', 'Co, g/m3',
      'C5+, g/m3', 'μg, cP', 'Bo, m3/m3', 'GOR, m3/m3', 'μo, cP',
    )
    tmpl = '%3s%13.1f%9.4f%9.4f%9.4f%12.5f%11.3f%12.3f%10.5f%12.3f%13.2f%9.3f'
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
        bg, cc, c5p, mug = self._gasprops(yji[0], vj[0])
        bo, gor, muo = self._oilprops(yji[1], vj[1])
        bgs[i] = bg
        ccs[i] = cc
        c5ps[i] = c5p
        mugs[i] = mug
        bos[i] = bo
        gors[i] = gor
        muos[i] = muo
        logger.info(
          tmpl, i, P, *Sj, n0, bg, cc * 1e3, c5p * 1e3, mug, bo, gor, muo,
        )
      elif mix.sj[0] == 0:
        bg, cc, c5p, mug = self._gasprops(yi, Zj[0] * R * T / P)
        bgs[i] = bg
        ccs[i] = cc
        c5ps[i] = c5p
        mugs[i] = mug
        logger.info(
          tmpl, i, P, 1., 0., n0, bg, cc * 1e3, c5p * 1e3, mug, -1., -1., -1.,
        )
      else:
        bo, gor, muo = self._oilprops(yi, Zj[0] * R * T / P)
        bos[i] = bo
        gors[i] = gor
        muos[i] = muo
        logger.info(tmpl, i, P, 0., 1., n0, -1., -1., -1., -1., bo, gor, muo)
    return LabResult(PP, ns, Fsj, ysji, Zsj, bgs, ccs, c5ps, bos, gors)


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
    Constant volume depletion simulation results as an instance of the
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
    bgs = np.zeros_like(Ps)
    ccs = np.zeros_like(Ps)
    c5ps = np.zeros_like(Ps)
    mugs = np.zeros_like(Ps)
    bos = np.zeros_like(Ps)
    gors = np.zeros_like(Ps)
    muos = np.zeros_like(Ps)
    logger.info(
      '%3s%13s%9s%9s%9s%12s%11s%12s%10s%12s%13s%9s',
      'Nst', 'P, Pa', 'Sg, fr', 'So, fr', 'n, mol', 'Bg, m3/m3', 'Co, g/m3',
      'C5+, g/m3', 'μg, cP', 'Bo, m3/m3', 'GOR, m3/m3', 'μo, cP',
    )
    tmpl = '%3s%13.1f%9.4f%9.4f%9.4f%12.5f%11.3f%12.3f%10.5f%12.3f%13.2f%9.3f'
    n = n0
    ysji[0] = psatres.yji
    Zsj[0] = psatres.Zj
    Fsj[0, 1] = 1.
    ns[0] = n
    vrg, vro = psatres.Zj * R * T / Psat
    bg, cc, c5p, mug = self._gasprops(psatres.yji[0], vrg)
    bo, gor, muo = self._oilprops(psatres.yji[1], vro)
    bgs[0] = bg
    ccs[0] = cc
    c5ps[0] = c5p
    mugs[0] = mug
    bos[0] = bo
    gors[0] = gor
    muos[0] = muo
    logger.info(
      tmpl, 0, Psat, 0., 1., n, bg, cc * 1e3, c5p * 1e3, mug, bo, gor, muo,
    )
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
        bg, cc, c5p, mug = self._gasprops(yji[0], vj[0])
        bo, gor, muo = self._oilprops(yji[1], vj[1])
        bgs[i] = bg
        ccs[i] = cc
        c5ps[i] = c5p
        mugs[i] = mug
        bos[i] = bo
        gors[i] = gor
        muos[i] = muo
        logger.info(
          tmpl, i, P, *Sj, n, bg, cc * 1e3, c5p * 1e3, mug, bo, gor, muo,
        )
        zi = yji[1]
        n *= Fj[1]
      elif mix.sj[0] == 1:
        bo, gor, muo = self._oilprops(zi, Zj[0] * R * T / P)
        bos[i] = bo
        gors[i] = gor
        muos[i] = muo
        logger.info(tmpl, i, P, 0., 1., n, -1., -1., -1., -1., bo, gor, muo)
      else:
        bg, cc, c5p, mug = self._gasprops(zi, Zj[0] * R * T / P)
        bgs[i] = bg
        ccs[i] = cc
        c5ps[i] = c5p
        mugs[i] = mug
        logger.info(
          tmpl, i, P, 1., 0., n, bg, cc * 1e3, c5p * 1e3, mug, -1., -1., -1.,
        )
        logger.warning('There is no oil in a cell.'
                       'The DL experiment can not be completed.')
        break
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    return LabResult(Ps, ns, Fsj, ysji, Zsj, bgs, ccs, c5ps, bos, gors)
