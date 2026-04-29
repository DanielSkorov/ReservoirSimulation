from logging import (
  getLogger,
)

from typing import (
  Type,
)

from resim.pvt.datatypes import (
  Float,
  ReportState,
  State,
  States,
  Vector,
)

from resim.pvt.utils import (
  PhaseFinder,
  denmax,
  findex,
)

from resim.pvt.flash import (
  FlashRoutine,
  flash,
)

from resim.pvt.psat import (
  PsatRoutine,
  psat,
)

from resim.pvt.lab.protocols import (
  CvdPTEos,
  CcePTEos,
  DlePTEos,
  SwlPTEos,
)


logger = getLogger('lab')


def cvd(
  eos: CvdPTEos,
  PP: Vector[Float],
  T: float,
  yi: Vector[Float],
  n0: float = 1.,
  psatroutine: PsatRoutine = psat(),
  flashroutine: FlashRoutine = flash(),
  gasfinder: PhaseFinder = findex,
  repstate: Type[ReportState] | None = None,
) -> States[ReportState]:
  """Multiphase constant volume depletion (CVD) experiment.

  Parameters
  ----------
  eos: CvdPTEos
    An initialized instance of an equation of state that can be used
    to perform the CVD experiment.

  PP: Vector[Float], shape (Ns,)
    A 1d-array of pressures [Pa] corresponding to the experiment stages,
    where `Ns` is the number of stages. All values above the saturation
    pressure will be ignored.

  T: float
    A thermodynamic parameter in SI units held constant during the
    experiment.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n0: float
    The initial mole amount of the fluid in a cell. Default is `1.0`
    [mol].

  psatroutine: PsatRoutine
    A callable object that can be used to find a saturation state of a
    mixture. Default is `psat()`.

  flashroutine: FlashRoutine
    A callable object that can be used to perform multiphase flash
    procedure. Default is `flash()`.

  gasfinder: PhaseFinder
    A callable object that can be used to obtain the index of a phase
    removed from the cell to adjust its volume. Default is `findex`.

  repstate: Type[ReportState] | None
    A dataclass used to calculate, store, and print custom parameters
    of a mixture state. If it is `None`, calculated data of the CVD
    experiment will be represented as `States[State]`. Default is
    `None`.

  Returns
  -------
  Constant volume depletion simulation results as an instance of the
  `States`.

  Notes
  -----
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

  Before calculating properties of gas and oil phases (formation volume
  factors, densities, viscosities, etc.) for the current experiment
  stage, both phases may undergo a series of separation steps, similar
  to what would be performed in a field. To simulate such a separation
  procedure for gas and oil phases `gassep` and `oilsep`
  should be used correspondingly.
  """
  logger.debug('Constant volume depletion (CVD).')
  logger.debug('T = %.2f [K], zi =' + eos.Nc * '%7.4f', T, *yi)
  n = n0
  state: State = psatroutine(eos, T, yi, n, True, None)
  Psat = state.P
  V0 = state.V
  logger.debug('Saturation pressure: %.1f [Pa].', Psat)
  PP_filtered = PP[PP < Psat]
  states: States[ReportState] = States()
  if repstate is None:
    states.append(state)
  else:
    states.append(repstate.from_state(state, eos))
  for P in PP_filtered:
    state = flashroutine(eos, P, T, yi, n, state)
    if repstate is None:
      states.append(state)
    else:
      states.append(repstate.from_state(state, eos))
    try:
      j = gasfinder(state, 0)
      dVg = state.V - V0
      dng = dVg / state.vj[j]
      dngi = state.yji[j] * dng
      n = n - dng
      yi = (state.ni - dngi) / n
    except IndexError:
      logger.warning(
        'There is no gas to remove. The CVD experiment can not be completed.'
      )
      logger.debug('%r', states)
      return states
  logger.debug('Total produced amount of gas: %.4f [mol].', n0 - n)
  logger.debug('%r', states)
  return states


def cce(
  eos: CcePTEos,
  PP: Vector[Float],
  T: float,
  yi: Vector[Float],
  n0: float = 1.,
  flashroutine: FlashRoutine = flash(),
  repstate: Type[ReportState] | None = None,
) -> States[ReportState]:
  """Multiphase constant composition (mass) expansion (CCE) experiment.

  Parameters
  ----------
  eos: CcePTEos
    An initialized instance of an equation of state that can be used
    to perform the CCE experiment.

  PP: Vector[Float], shape (Ns,)
    A 1d-array of pressures [Pa] corresponding to the experiment stages,
    where `Ns` is the number of stages.

  T: float
    A thermodynamic parameter in SI units held constant during the
    experiment.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n0: float
    The initial mole amount of the fluid in a cell. Default is `1.0`
    [mol].

  flashroutine: FlashRoutine
    A callable object that can be used to perform multiphase flash
    procedure. Default is `flash()`.

  repstate: Type[ReportState] | None
    A dataclass used to calculate, store, and print custom parameters
    of a mixture state. If it is `None`, calculated data of the CVD
    experiment will be represented as `States[State]`. Default is
    `None`.

  Returns
  -------
  Constant composition expansion simulation results as an instance of
  the `States`.

  Notes
  -----
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

  Before calculating properties of gas and oil phases (formation volume
  factors, densities, viscosities, etc.) for the current experiment
  stage, both phases may undergo a series of separation steps, similar
  to what would be performed in a field. To simulate such a separation
  procedure for gas and oil phases `gassep` and `oilsep`
  should be used correspondingly.
  """
  logger.debug('Constant composition expansion (CCE).')
  logger.debug('T = %.2f [K], zi =' + eos.Nc * '%7.4f', T, *yi)
  state: State | None = None
  states: States[ReportState] = States()
  for P in PP:
    state = flashroutine(eos, P, T, yi, n0, state)
    if repstate is None:
      states.append(state)
    else:
      states.append(repstate.from_state(state, eos))
  logger.debug('%r', states)
  return states


def dle(
  eos: DlePTEos,
  PP: Vector[Float],
  T: float,
  yi: Vector[Float],
  n0: float = 1.,
  psatroutine: PsatRoutine = psat(),
  flashroutine: FlashRoutine = flash(),
  oilfinder: PhaseFinder = denmax,
  repstate: Type[ReportState] | None = None,
) -> States[ReportState]:
  """Multiphase differential liberation experiment (DLE).

  Parameters
  ----------
  eos: DlePTEos
    An initialized instance of an equation of state that can be used
    to perform the DL experiment.

  PP: Vector[Float], shape (Ns,)
    A 1d-array of pressures [Pa] corresponding to the experiment stages,
    where `Ns` is the number of stages. All values above the saturation
    pressure will be ignored.

  T: float
    A thermodynamic parameter in SI units held constant during the
    experiment.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n0: float
    The initial mole amount of the fluid in a cell. Default is `1.0`
    [mol].

  psatroutine: PsatRoutine
    A callable object that can be used to find a saturation state of a
    mixture. Default is `psat()`.

  flashroutine: FlashRoutine
    A callable object that can be used to perform multiphase flash
    procedure. Default is `flash()`.

  oilfinder: PhaseFinder
    A callable object that can be used to obtain the index of a phase
    remaining in the cell after each stage of the experiment. Default
    is `denmax`.

  repstate: Type[ReportState] | None
    A dataclass used to calculate, store, and print custom parameters
    of a mixture state. If it is `None`, calculated data of the DLE
    experiment will be represented as `States[State]`. Default is
    `None`.

  Returns
  -------
  Differential liberation simulation results as an instance of the
  `States`.

  Notes
  -----
  The DL experiment is usually performed with an oil phase to simulate
  the processes encountered during the field development. The experiment
  is conducted according to the following procedure:

  1) The oil sample is prepared to ensure that its state corresponds to
     the bubble point.
  2) Pressure is reduced by increasing the cell volume. Some amount of
     the gas appears. The volume of that liquid phase is registered.
  3) All gas is expelled from the cell.
  4) The gas collected is sent to a multistage separator to study its
     properties and composition.
  5) The process is repeated for several pressure steps.

  Before calculating properties of gas and oil phases (formation volume
  factors, densities, viscosities, etc.) for the current experiment
  stage, both phases may undergo a series of separation steps, similar
  to what would be performed in a field. To simulate such a separation
  procedure for gas and oil phases `gassep` and `oilsep`
  should be used correspondingly.
  """
  logger.info('Differential liberation expreiment (DLE).')
  logger.info('T = %.2f [K], zi =' + eos.Nc * '%7.4f', T, *yi)
  n = n0
  state: State = psatroutine(eos, T, yi, n, True, None)
  Psat = state.P
  logger.info('Saturation pressure: %.1f [Pa].', Psat)
  PP_filtered = PP[PP < Psat]
  states: States[ReportState] = States()
  if repstate is None:
    states.append(state)
  else:
    states.append(repstate.from_state(state, eos))
  for P in PP_filtered:
    state = flashroutine(eos, P, T, yi, n, state)
    if repstate is None:
      states.append(state)
    else:
      states.append(repstate.from_state(state, eos))
    try:
      j = oilfinder(state, 1)
      yi = state.yji[j]
      n = state.nj[j]
    except IndexError:
      logger.warning(
        'There is no oil in the cell. The DL experiment can not be completed.'
      )
      logger.debug('%r', states)
      return states
  logger.info('Total produced amount of gas: %.4f [mol].', n0 - n)
  logger.debug('%r', states)
  return states


def swl(
  eos: SwlPTEos,
  phfs: Vector[Float],
  T: float,
  zi: Vector[Float],
  yi: Vector[Float],
  n0: float = 1.,
  psatroutine: PsatRoutine = psat(),
  repstate: Type[ReportState] | None = None,
) -> States[ReportState]:
  """Two-phase swelling experiment.

  Parameters
  ----------
  eos: SwlPTEos
    An initialized instance of an equation of state that can be used
    to perform the swelling test.

  phfs: Vector[Float], shape (Ns,)
    Mole fractions of the injected fluid that need to be dissolved in
    the reservoir fluid. `Ns` is the number of stages of the experiment.

  T: float
    A thermodynamic parameter in SI units held constant during the
    experiment.

  zi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components characterising the reservoir
    fluid.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components characterising the injection
    fluid.

  n0: float
    The initial mole amount of the reservoir fluid in a cell. Default
    is `1.0` [mol].

  psatroutine: PsatRoutine
    A callable object that can be used to find a saturation state of a
    mixture. Default is `psat()`.

  repstate: Type[ReportState] | None
    A dataclass used to calculate, store, and print custom parameters of
    a mixture state. If it is `None`, calculated data of the swelling
    experiment will be represented as `States[State]`. Default is
    `None`.

  Returns
  -------
  Swelling test simulation results as an instance of the `States`.

  Notes
  -----
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

  Before calculating properties of gas and oil phases (formation volume
  factors, densities, viscosities, etc.) for the current experiment
  stage, both phases may undergo a series of separation steps, similar
  to what would be performed in a field. To simulate such a separation
  procedure for gas and oil phases `gassep` and `oilsep`
  should be used correspondingly.
  """
  logger.info('Swelling test.')
  logger.info('T = %.2f [K], zi =' + eos.Nc * '%7.4f', T, *zi)
  states: States[ReportState] = States()
  init: State | None = None
  xi = zi.copy()
  for phf in phfs:
    n = n0 / (1. - phf)
    xi = (1. - phf) * zi + phf * yi
    state = psatroutine(eos, T, xi, n, True, init)
    if repstate is None:
      states.append(state)
    else:
      states.append(repstate.from_state(state, eos))
    init = state
  logger.debug('%r', states)
  return states
