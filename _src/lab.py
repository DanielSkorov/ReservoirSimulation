import logging

from dataclasses import (
  dataclass,
)

import numpy as np

from flash import (
  EosFlash2pPT,
  EosFlashNpPT,
  FlashResult,
  flash2pPT,
  flashNpPT,
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
  Sequence,
  Iterator,
  Literal,
  ValuesView,
  KeysView,
)

from customtypes import (
  Logical,
  Integer,
  Double,
  Vector,
  Matrix,
  Tensor,
)

from eos import (
  Eos,
)


logger = logging.getLogger('lab')


class StreamResult(object):
  """Container for separator stream outputs with pretty-printing.

  Attributes
  ----------
  _data: list[FlashResult]
    A list of calculation results at each stage that this stream is
    going through.

  Methods
  -------
  __getitem__(self, stage: int) -> FlashResult
    Returns stage results as an instance of `FlashResult` for the
    given number (index) of a stage.

  __setitem__(self, stage: int, res: FlashResult) -> None
    Replaces stage results with the given for the specific number
    (index) of a stage.

  __iter__(self) -> Iterator[FlashResult]
    Returns the `Iterator` of `FlashResult`.

  __str__(self) -> str
    Returns the string representation of this class.

  __repr__(self) -> str
    Returns the table representation of this class.

  append(self, res: FlashResult) -> None
    Adds the given stage results to the end of the `_data` list.
  """
  __slots__ = ('_data',)
  _data: list[FlashResult]

  def __init__(
    self,
    flash: Sequence[FlashResult] | FlashResult | None = None,
  ) -> None:
    if isinstance(flash, FlashResult):
      self._data = [flash]
    elif isinstance(flash, Sequence):
      self._data = list(flash)
    else:
      self._data = []
    pass

  def __getitem__(self, stage: int) -> FlashResult:
    return self._data[stage]

  def __setitem__(self, stage: int, flash: FlashResult) -> None:
    self._data[stage] = flash
    pass

  def __iter__(self) -> Iterator[FlashResult]:
    return iter(self._data)

  def __contains__(self, stage: int) -> bool:
    return stage < len(self._data)

  def __str__(self) -> str:
    return '\n'.join([f'Stage #{s}:\n{flash:s}'
                      for (s, flash) in enumerate(self._data)])

  def __repr__(self) -> str:
    return '\n'.join(['%4s%r' % (s, flash) if s == 0 else '%8s%r' % (s, flash)
                      for (s, flash) in enumerate(self._data)])

  def __format__(self, fmt: str):
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')

  def append(self, flash: FlashResult) -> None:
    self._data.append(flash)
    pass


class SeparatorResult(object):
  """Container for separator outputs with pretty-printing.

  Attributes
  ----------
  _data: dict[str, StreamResult]
    A dictionary of separator calculation results. Keys are strings
    (names) of streams. Values are instances of the `StreamResult`.

  Methods
  -------
  __getitem__(self, stream: str) -> StreamResult
    Returns stream results as an instance of `StreamResult` for the
    given name of a stream.

  __setitem__(self, stream: str, res: StreamResult) -> None
    Replaces (or adds) stream results with the given one for specific
    name of a stream.

  __iter__(self) -> Iterator[str]
    Returns the `Iterator` of keys of `_data` (names of streams as
    `str`).

  __contains__(self, key: str) -> bool
    Returns a boolean flag indicating if the given name of a stream is
    in the `_data` dictionary.

  __str__(self) -> str
    Returns the string representation of this class.

  __repr__(self) -> str
    Returns the table representation of this class.

  values(self) -> ValuesView
    Returns the `ValuesView` of the `_data` dictionary.

  keys(self) -> KeysView
    Returns the `KeysView` of the `_data` dictionary.

  new_stream(self, stream: str) -> None
    Creates a new stream in the `_data` dictionary with the empty
    stream results as an instance of `StreamResult`.
  """
  __slots__ = ('_data',)
  _data: dict[str, StreamResult]

  def __init__(
    self,
    stream: Sequence[str] | str | None = None,
    result: Sequence[StreamResult] | StreamResult | None = None,
  ) -> None:
    if isinstance(stream, Sequence) and isinstance(result, Sequence):
      self._data = dict(zip(stream, result))
    elif isinstance(stream, str) and isinstance(result, StreamResult):
      self._data = {stream: result}
    elif isinstance(stream, Sequence):
      self._data = dict(zip(stream, map(lambda _: StreamResult(), stream)))
    elif isinstance(stream, str):
      self._data = {stream: StreamResult()}
    elif stream is None and result is None:
      self._data = {}
    else:
      logger.error('Pair each stream with its result.')
      raise ValueError('Pair each stream with its result.')
    pass

  def __getitem__(self, stream: str) -> StreamResult:
    return self._data[stream]

  def __setitem__(self, stream: str, res: StreamResult) -> None:
    self._data[stream] = res
    pass

  def __iter__(self) -> Iterator[str]:
    return iter(self._data.keys())

  def __contains__(self, key: str) -> bool:
    return key in self._data

  def __str__(self) -> str:
    return '\n'.join([f'Stream "{stream}":\n{self._data[stream]:s}'
                      for stream in self._data])

  def __repr__(self) -> str:
    Np = 0
    for stream in self._data:
      for stage in self._data[stream]:
        _Np = stage.Np
        if _Np > Np:
          Np = _Np
    tmpl = '%4s%4s%8s%8s%9s' + Np * '%4s%9s%9s' + '\n'
    out = tmpl % ('Strm', 'Stg', 'P', 'T', 'n',
                  *[l + str(i) for i in range(Np) for l in ('s', 'f', 'Z')])
    out += tmpl % (('', '', '[MPa]', '[°C]', '[mol]') + Np * ('','[fr.]',''))
    out += '\n'.join(['%4s%r' % (s[:4], self._data[s]) for s in self._data])
    return out

  def __format__(self, fmt: str) -> str:
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')

  def values(self) -> ValuesView:
    return self._data.values()

  def keys(self) -> KeysView:
    return self._data.keys()

  def new_stream(self, stream: str) -> None:
    self._data[stream] = StreamResult()
    pass


class Separator(Protocol):
  def run(self, yi: Vector[Double], n: float = 1.) -> SeparatorResult: ...


class LabSeparator(Separator, Protocol):
  gasstream: str
  oilstream: str
  gasstage: int
  oilstage: int


class SimpleSeparator(object):
  """Two-phase simple separator.

  Performs flash calculation for specific pressure and temperature, and
  returns its outputs as an instance of `SeparatorResult`.

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

  Psc: float
    Pressure [Pa] for which flash calculations should be performed.
    Default is `101325.0` Pa.

  Tsc: float
    Temperatures [K] for which flash calculations should be performed.
    Default is `293.15` K.

  **kwargs: dict
    Other arguments for the two-phase flash solver. It may contain such
    arguments as `method`, `tol`, `maxiter` and others, depending on the
    selected solver.
  """
  def __init__(
    self,
    eos: EosFlash2pPT,
    Psc: float = 101325.,
    Tsc: float = 293.15,
    **kwargs,
  ) -> None:
    self.Psc = Psc
    self.Tsc = Tsc
    self.gasstream = 'main'
    self.oilstream = 'main'
    self.gasstage = 0
    self.oilstage = 0
    self.solver = flash2pPT(eos, **kwargs)
    pass

  def run(self, yi: Vector[Double], n: float = 1.) -> SeparatorResult:
    """Performs two-phase flash calculations to obtain compositions
    of phases at standard conditions.

    Parameters
    ----------
    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture. Default is `1.0` mol.

    Returns
    -------
    Separator calculation results as an instance of `SeparatorResult`.
    """
    flash = self.solver.run(self.Psc, self.Tsc, yi, n)
    res = SeparatorResult('main', StreamResult(flash))
    logger.debug('%r', res)
    return res


class GasSeparator(object):
  """Multiphase gas separator that can be used to simulate low-
  temperature separation. At each stage of the separation process,
  liquid phases are removed into the stock tank conditions. The gas
  phase collected from previous stage is transferred to the subsequent
  stage. The gas obtained from the liquid phase under stock tank
  conditions is combined with the main gas stream. The stock tank
  conditions are the pressure and temperature at the last stage.

  Parameters
  ----------
  eos: EosFlash2pPT | EosFlashNpPT
    An initialized instance of a PT-based equation of state. For
    two-phase flash calculations, it must have the following methods:

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

    If the solution method for two-phase flash calculations would be one
    of `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also must
    have:

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

    If the maximum number of phases would be greater than two, then
    this istance also must have:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z(P: float, T: float, yi: Vector[Double]) -> float`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return the
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

    If the solution method for multiphase flash calculations would be
    one of `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also
    must have:

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

  Ps: Sequence[float]
    Pressure [Pa] at stages of the separation process.
    Default is `[7e6, 4.5e6, 2e6, 101325.0]` Pa.

  Ts: Sequence[float]
    Temperatures [K] at stages of the separation process.
    Default is `[223.15, 238.15, 258.15, 293.15]` K.

  maxNp: int
    The maximum number of phases. Default is `2`.

  **kwargs: dict
    Other arguments for the two-phase flash solver. It can contain such
    arguments as `method`, `tol`, `maxiter` and others, depending on the
    selected solver.
  """
  def __init__(
    self,
    eos: EosFlash2pPT | EosFlashNpPT,
    Ps: Sequence[float] = [7e6, 4.5e6, 2e6, 101325.],
    Ts: Sequence[float] = [223.15, 238.15, 258.15, 293.15],
    maxNp: int = 2,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.Ps = Ps
    self.Ts = Ts
    if maxNp == 2:
      self.solver = flash2pPT(eos, **kwargs)
    elif maxNp > 2:
      self.solver = flashNpPT(eos, maxNp=maxNp, **kwargs)
    else:
      raise ValueError(
        'The maximum number of phases must be greater than or equal to 2.'
      )
    self.gasstream = 'main'
    self.oilstream = 'liquid'
    self.gasstage = len(Ps) - 1
    self.oilstage = 0
    pass

  def run(self, yi: Vector[Double], n: float = 1.) -> SeparatorResult:
    """Performs a series of flash calculations, collecting the liquid
    phase in stock tank conditions. Gas obtained from the oil will be
    mixed with the main gas stream.

    Parameters
    ----------
    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture. Default is `1.0` mol.

    Returns
    -------
    Separator calculation results as an instance of `SeparatorResult`.
    """
    streamg = StreamResult()
    res = SeparatorResult('main', streamg)
    ng = n
    ygi = yi
    ngi = ng * ygi
    nli = np.zeros(shape=(self.eos.Nc,))
    for s, (P, T) in enumerate(zip(self.Ps, self.Ts)):
      flash = self.solver.run(P, T, ygi, ng)
      streamg.append(flash)
      if 0 in flash.sj:
        j = np.argwhere(flash.sj == 0).flatten()[0]
        ng = flash.nj[j]
        ygi = flash.yji[j]
        ngi = ng * ygi
        nli += flash.ni - ngi
      else:
        nli += flash.ni
        ng = 0.
        ngi = ng * ygi
      if s == self.gasstage:
        nl = nli.sum()
        if nl > 0.:
          yli = nli / nl
          flash = self.solver.run(P, T, yli, nl)
          if 0 in flash.sj:
            j = np.argwhere(flash.sj == 0).flatten()[0]
            nsg = flash.nj[j]
            nsgi = nsg * flash.yji[j]
            ng += nsg
            ngi += nsgi
            ygi = ngi / ng
            streamg[s] = self.solver.run(P, T, ygi, ng)
            nl -= nsg
            nli -= nsgi
            if nl > 0.:
              yli = nli / nl
              res['liquid'] = StreamResult(self.solver.run(P, T, yli, nl))
          else:
            res['liquid'] = StreamResult(flash)
      if ng == 0.:
        logger.warning(
          'There is no gas at the stage #%s of the GasSeparator. The '
          'separation procedure will be stopped.', s,
        )
        break
    logger.debug('%r', res)
    return res


class MultiSeparator(object):
  r"""Multiphase separator.

  The main idea behind the `MultiSeparator` is division of a separator
  logic into *streams*, each of which may undergo several separation
  *stages*. To provide the flexibility of the `MultiSeparator`, its
  program flow is governed by a user-defined code. There are three
  main keywords:

  - **def** *stream_name number_of_stages*
    This keywords allows to define the stream `stream_name: str` and
    its number of stages `number_of_stages: int`. The program starts by
    placing the given amount of the mixture of known component
    composition into the stage `0` of the stream `'main'`. Therefore,
    the `'main'` stream should always be defined. After placing some
    amount of the fluid into the stage `0` of the stream `'main'`, one
    should perform the flash procedure to split the mixture into the
    phases.

  - **flash** *stream_name stage_number prs tem*
    This keyword specifies the pressure [Pa] `prs: float` and
    temperature [K] `tmp: float`, at which the flash calculation must
    be performed for the stage `stage_number: int` of the stream
    `stream_name: str`. If there is no fluid present at the specified
    stage, the flash procedure will be skipped. Therefore, before
    running the flash, one should place some amount of the fluid into
    a stage of a stream.

  - **move** *phase_id stream_src stage_src stream_trg stage_trg*
    By using this keyword one can move the full amount the phase
    `phase_id: int` from the source stage `stage_src: int` of the
    source stream `stream_src: str` to the target stage `stage_trg: int`
    of the target stream `stream_trg: str`. If there is no amount of
    the phase, then the movement procedure will be skipped.

  Also, there are keywords that can be used to define the name of the
  gas phase stream and its stage, the name of the oil phase stream and
  its stage from which flash results will be taken to calculate phase
  properties in laboratory experiments (such as CVD, CCE, DL etc.):

  - **gasstream** *stream_name*
    This keyword specifies the `stream_name: str` of a stream that
    should be considered as the gas phase stream to calculate properties
    in laboratory experiments.

  - **gasstage** *stage_number*
    This keyword specifies the `stage_number: int` of the `gasstream`
    from which the flash results will be taken to calculate properties
    in laboratory experiments.

  - **oilstream** *stream_name*
    This keyword specifies the `stream_name: str` of a stream that
    should be considered as the oil phase stream to calculate properties
    in laboratory experiments.

  - **oilstage** *stage_number*
    This keyword specifies the `stage_number: int` of the `oilstream`
    from which the flash results will be taken to calculate properties
    in laboratory experiments.

  All these keywords are necessary only if the `labmode` is set to
  `True`.

  Parameters
  ----------
  eos: EosFlashNpPT
    An initialized instance of a PT-based equation of state that will
    be used in multiphase flash calculation procedure. It must have
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
      (`Vector[Double]` of shape `(Nc,)`), this method must return the
      compressibility factor.

    - `getPT_Zj_lnphiji(P: float, T: float, yji: Matrix[Double])
       -> tuple[Vector[Double], Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this
      method must return a tuple of:

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

  code: str
    The code that describes the logic of the separator. It must be a
    string of commands, each of which should be separated from another
    by `'\n'` or `';''.

  maxNp: int
    The maximum number of phases for the flash calculation solver.
    Default is `3`.

  labmode: bool
    Activates the laboratory mode of the separator. It means that the
    keywords *gasstream*, *gasstage*, *oilstream*, *oilstage* must be
    present in the code. Default is `False`.

  **kwargs: dict
    Other arguments for the flash calculation solver. It may contain
    such arguments as `method`, `tol`, `maxiter` and others.

  Examples
  --------
  For example, let's assume that we need to repeat the separation logic
  the same as for `GasSeparator` (with default pressures and
  temperatures at the stages). Therefore, the `code` parameter should
  be:
  ```
  code = (
    # define the stream 'main' of 4 stages
    'def main 4;'
    # define the stream 'oil' of 1 stage
    'def oil 1;'
    # flash the fluid at the stage 0 of the stream 'main'
    'flash main 0 7e6 223.15;'
    # move the phase 1 from the stage 0 of the stream 'main'
    # to the stage 0 of the stream 'oil'
    'move 1 main 0 oil 0;'
    # move the phase 0 from the stage 0 of the stream 'main'
    # to the stage 1 of the stream 'main'
    'move 0 main 0 main 1;'
    # flash the fluid at the stage 1 of the stream 'main'
    'flash main 1 4.5e6 238.15;'
    # move the phase 1 from the stage 1 of the stream 'main'
    # to the stage 0 of the stream 'oil'
    'move 1 main 1 oil 0;'
    # move the phase 0 from the stage 1 of the stream 'main'
    # to the stage 2 of the stream 'main'
    'move 0 main 1 main 2;'
    # flash the fluid at the stage 2 of the stream 'main'
    'flash main 2 2e6 258.15;'
    # move the phase 1 from the stage 2 of the stream 'main'
    # to the stage 0 of the stream 'oil'
    'move 1 main 2 oil 0;'
    # move the phase 0 from the stage 2 of the stream 'main'
    # to the stage 3 of the stream 'main'
    'move 0 main 2 main 3;'
    # flash the fluid at the stage 3 of the stream 'main'
    'flash main 3 101325. 293.15;'
    # move the phase 1 from the stage 3 of the stream 'main'
    # to the stage 0 of the stream 'oil'
    'move 1 main 3 oil 0;'
    # flash the fluid at the stage 0 of the stream 'oil'
    'flash oil 0 101325. 293.15;'
    # move the phase 0 from the stage 0 of the stream 'oil'
    # to the stage 3 of the stream 'main'
    'move 0 oil 0 main 3;'
    # flash the fluid at the stage 3 of the stream 'main'
    'flash main 3 101325. 293.15;'
    # flash the fluid at the stage 0 of the stream 'oil'
    'flash oil 0 101325. 293.15'
  )
  ```
  """
  def __init__(
    self,
    eos: EosFlashNpPT,
    code: str,
    maxNp: int = 3,
    labmode: bool = False,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.solver = flashNpPT(eos, maxNp=maxNp, **kwargs)
    self.maxNp = maxNp
    self._flow: list[tuple[Literal[True], str, int, float, float] |
                     tuple[Literal[False], int, str, int, str, int]] = []
    self._design = {}
    gasstream: str | None = None
    oilstream: str | None = None
    gasstage: int | None = None
    oilstage: int | None = None
    lines = code.replace('\n', ';').split(';')
    for line in lines:
      line = line.strip().lower()
      if line:
        keyword, *args = line.split()
        if keyword == 'flash':
          stream = args[0]
          stage = int(args[1])
          self._check(stream, stage)
          P = float(args[2])
          T = float(args[3])
          self._flow.append((True, stream, stage, P, T))
        elif keyword == 'move':
          phase = int(args[0])
          source_stream = args[1]
          source_stage = int(args[2])
          self._check(source_stream, source_stage)
          target_stream = args[3]
          target_stage = int(args[4])
          self._check(target_stream, target_stage)
          self._flow.append((False, phase, source_stream, source_stage,
                                    target_stream, target_stage))
        elif keyword == 'gasstream' and labmode:
          gasstream = args[0]
          gasstage = int(args[1])
        elif keyword == 'oilstream' and labmode:
          oilstream = args[0]
          oilstage = int(args[1])
        elif keyword == 'def':
          stream = args[0]
          stages = int(args[1])
          self._design[stream] = stages
        else:
          raise ValueError(f'Unknown keyword: "{keyword}".')
    if gasstream is not None and gasstage is not None:
      self.gasstream = gasstream
      self.gasstage = gasstage
    elif labmode:
      logger.error('The keyword "gasstream" must be specified.')
      raise ValueError('The keyword "gasstream" must be specified.')
    if oilstream is not None and oilstage is not None:
      self.oilstream = oilstream
      self.oilstage = oilstage
    elif labmode:
      logger.error('The keyword "oilstream" must be specified.')
      raise ValueError('The keyword "oilstream" must be specified.')
    if 'main' not in self._design:
      raise ValueError('The stream named "main" must be present.')
    pass

  def _check(self, stream: str, stage: int) -> None:
    if stream not in self._design:
      raise ValueError(
        f'The stream "{stream}" must be defined using keyword "def".'
      )
    elif stage >= self._design[stream]:
      raise ValueError(
        f'The stage #{stage} is greater than the number of stages '
        f'for the stream "{stream}".'
      )
    pass

  def _empty_flash(self) -> FlashResult:
    Np = self.maxNp
    Nc = self.eos.Nc
    ni = np.zeros((Nc,))
    nj = np.zeros((Np,))
    fj = np.zeros((Np,))
    yji = np.zeros((Np, Nc))
    Zj = np.full((Np,), -1.)
    vj = np.full((Np,), -1.)
    sj = np.full((Np,), -1)
    return FlashResult(Np, -1., -1., -1., 0., ni, nj, fj, yji, Zj, vj, sj)

  def run(self, yi: Vector[Double], n: float = 1.) -> SeparatorResult:
    """Performs the user defined separator logic for a given mixture
    composition and mole number.

    Parameters
    ----------
    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture. Default is `1.0` mol.

    Returns
    -------
    Separator calculation results as an instance of `SeparatorResult`.
    """
    Nc = self.eos.Nc
    res = SeparatorResult()
    for s in self._design:
      res.new_stream(s)
      for _ in range(self._design[s]):
        res[s].append(self._empty_flash())
    res['main'][0].n += n
    res['main'][0].ni += n * yi
    for cmd in self._flow:
      if cmd[0]:
        stream, stage, P, T = cmd[1:]
        flash = res[stream][stage]
        n = flash.n
        if n > 0.:
          # TODO: pass the empty flash result as the `out` argument to
          # make inplace updates (for fortran backend at least)
          res[stream][stage] = self.solver.run(P, T, flash.ni/n, n)
        else:
          logger.warning(
            'The "flash" command for the stage #%s of the stream "%s" '
            'was skipped due to abscence of the fluid.', stage, stream,
          )
      else:
        phase, stream_src, stage_src, stream_trg, stage_trg = cmd[1:]
        flash_src = res[stream_src][stage_src]
        flash_trg = res[stream_trg][stage_trg]
        if phase in flash_src.sj:
          j = np.argwhere(flash_src.sj == phase).flatten()[0]
          n = flash_src.nj[j]
          ni = n * flash_src.yji[j]
          flash_trg.n += n
          flash_trg.ni += ni
          flash_src.n -= n
          flash_src.ni -= ni
        else:
          logger.warning(
            'The "move" command from the stage #%s of the source stream "%s" '
            'to the stage #%s of the target stream "%s" was skipped due to '
            'the abscence of the phase "%s" in the flash results of the '
            'source stream.',
            stage_src, stream_src, stage_trg, stream_trg, phase,
          )
    logger.debug('%r', res)
    return res


@dataclass(eq=False, slots=True)
class LabResult(object):
  """Container for experiment simulation outputs with pretty-printing.

  Attributes
  ----------
  Ps: Vector[Double], shape (Ns,)
    A 1d-array with pressures [Pa] corresponding to the stages of an
    experiment, where `Ns` is the number of stages.

  ns: Vector[Double], shape (Ns,)
    Mole number [mol] of a system at each stage of an experiment.

  fsj: Matrix[Double], shape (Ns, Np)
    Phase mole fractions at each stage of an experiment as a
    `Matrix[Double]` of shape `(Ns, Np)`.

  ysji: Tensor[Double], shape (Ns, Np, Nc)
    Mole fractions of components in each phase at each stage of an
    experiment as a `Tensor[Double]` of shape `(Ns, Np, Nc)`, where
    `Np` is the number of phases and `Nc` is the number of components.

  Zsj: Matrix[Double], shape (Ns, Np)
    Phase compressibility factors at each stage of an experiment as
    a `Matrix[Double]` of shape `(Ns, Np)`.

  props: Matrix[Double], shape (Ns, 11)
    A 2d-array containing values of properties. It includes (in the
    iteration order):

    0: formation volume factor [rm3/sm3] of the gas phase at each stage
       of an experiment,

    1: viscosity [cP] of the gas phase at each stage of an experiment,

    2: density [kg/sm3] of the gas phase at standard conditions,

    3: condensate-gas ratio [kg/sm3] of the gas phase at standard
       conditions,

    4: pseudo condensate-gas ratio [kg/sm3] of the gas phase at standard
       conditions (it is calculated as the ratio of the sum of masses of
       C5+ components in the gas phase at the pressure and temperature
       of the stage to the volume of the gas phase at standard
       conditions),

    5: density [kg/sm3] of the condensate at standard conditions,

    6: formation volume factor [rm3/sm3] of the oil phase at each stage
       of an experiment,

    7: viscosity [cP] of the oil phase at each stage of an experiment,

    8: density [kg/sm3] of the dead oil phase at standard conditions,

    9: gas-oil ratio [sm3/sm3] of the oil phase at standard conditions,

    10: density [kg/sm3] of the dissolved gas at standard conditions.
  """
  Ps: Vector[Double]
  ns: Vector[Double]
  fsj: Matrix[Double]
  ysji: Tensor[Double]
  Zsj: Matrix[Double]
  props: Matrix[Double]

  def __repr__(cls):
    Np = cls.fsj.shape[1]
    rng = range(Np)
    tmpl = ('%3s%9s' + Np * '%9s' + '%9s'
            '%12s%10s%11s%11s%11s%11s'
            '%12s%8s%11s%12s%12s\n')
    s = tmpl % (
      'Nst', 'P', *[f'f{i}' for i in rng], 'n',
      'Bg', 'μg', 'Dg', 'Cc', 'C₅₊', 'Dc',
      'Bo', 'μo', 'Ddo', 'GOR', 'Dsg',
    )
    s += tmpl % (
      '', '[MPa]', *['[fr]' for _ in rng], '[mol]',
      '[rm3/sm3]', '[cP]', '[kg/sm3]', '[kg/sm3]', '[kg/sm3]', '[kg/sm3]',
      '[rm3/sm3]', '[cP]', '[kg/sm3]', '[sm3/sm3]', '[kg/sm3]',
    )
    tmpl = ('%3s%9.3f' + Np * '%9.4f' + '%9.4f'
            '%12.5f%10.5f%11.4f%11.4f%11.4f%11.1f'
            '%12.3f%8.3f%11.1f%12.1f%12.4f\n')
    for i in range(cls.Ps.shape[0]):
      s += tmpl % (i, cls.Ps[i] / 1e6, *cls.fsj[i], cls.ns[i], *cls.props[i])
    return s


class _labprops(object):
  eos: Eos
  sepg: LabSeparator
  sepo: LabSeparator
  c5pi: Vector[Logical]
  mwc5pi: Vector[Double]

  def _gasprops(
    self,
    yrgi: Vector[Double],
    Vrg: float,
  ) -> tuple[float, float, float, float, float, float]:
    """Calculates properties of the gas phase at standard conditions.
    The flash calculation procedure is performed to compute fluid
    properties at standard pressure and temperature.

    Parameters
    ----------
    yrgi: Vector[Double]
      The composition of the gas phase at pressure and temperature
      corresponding to a certain stage of the experiment.

    Vrg: float
      Molar volume [m3/mol] of the gas phase at pressure and
      temperature corresponding to a certain stage of the experiment.

    Returns
    -------
    A tuple of:
    - gas formation volume factor [rm3/sm3],
    - gas viscosity [cP],
    - gas density [kg/sm3],
    - condensate-gas ratio (condensate solubility in gas) [kg/sm3],
    - pseudo condensate-gas ratio (pseudo condensate solubility
      in gas) [kg/sm3],
    - condensate density [kg/sm3].
    """
    sep = self.sepg.run(yrgi)
    if self.sepg.gasstream in sep:
      streamg = sep[self.sepg.gasstream]
      if self.sepg.gasstage in streamg:
        flashg = streamg[self.sepg.gasstage]
        gasexists = 0 in flashg.sj
      else:
        gasexists = False
    else:
      gasexists = False
    if self.sepg.oilstream in sep:
      streamo = sep[self.sepg.oilstream]
      if self.sepg.oilstage in streamo:
        flasho = streamo[self.sepg.oilstage]
        oilexists = 1 in flasho.sj
      else:
        oilexists = False
    else:
      oilexists = False
    if gasexists and oilexists:
      j = np.argwhere(flashg.sj == 0).flatten()[0]
      vsg = flashg.vj[j]
      Vsg = flashg.nj[j] * vsg
      bg = Vrg / Vsg
      mwg = flashg.yji[j].dot(self.eos.mwi)
      deng = mwg / vsg
      mug = -1.
      j = np.argwhere(flasho.sj == 1).flatten()[0]
      vso = flasho.vj[j]
      mwo = flasho.yji[j].dot(self.eos.mwi)
      deno = mwo / vso
      co = flasho.nj[j] * mwo / Vsg
      c5p = yrgi[self.c5pi].dot(self.mwc5pi) / Vsg
      return bg, mug, deng, co, c5p, deno
    elif gasexists:
      j = np.argwhere(flashg.sj == 0).flatten()[0]
      vsg = flashg.vj[j]
      Vsg = flashg.nj[j] * vsg
      bg = Vrg / Vsg
      deng = flashg.yji[j].dot(self.eos.mwi) / vsg
      mug = -1.
      c5p = yrgi[self.c5pi].dot(self.mwc5pi) / Vsg
      return bg, mug, deng, 0., c5p, -1.
    elif oilexists:
      j = np.argwhere(flasho.sj == 1).flatten()[0]
      deno = flasho.yji[j].dot(self.eos.mwi) / flasho.vj[j]
      return -1., -1., -1., -1., -1., deno
    else:
      return -1., -1., -1., -1., -1., -1.

  def _oilprops(
    self,
    yroi: Vector[Double],
    Vro: float,
  ) -> tuple[float, float, float, float, float]:
    """Calculates properties of the oil phase at standard conditions.
    The flash calculation procedure is performed to compute fluid
    properties at standard pressure and temperature.

    Parameters
    ----------
    yroi: Vector[Double]
      The composition of the oil phase at pressure and temperature
      corresponding to a certain stage of the experiment.

    Vro: float
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
    sep = self.sepo.run(yroi)
    if self.sepo.gasstream in sep:
      streamg = sep[self.sepo.gasstream]
      if self.sepo.gasstage in streamg:
        flashg = streamg[self.sepo.gasstage]
        gasexists = 0 in flashg.sj
      else:
        gasexists = False
    else:
      gasexists = False
    if self.sepo.oilstream in sep:
      streamo = sep[self.sepo.oilstream]
      if self.sepo.oilstage in streamo:
        flasho = streamo[self.sepo.oilstage]
        oilexists = 1 in flasho.sj
      else:
        oilexists = False
    else:
      oilexists = False
    if gasexists and oilexists:
      j = np.argwhere(flasho.sj == 1).flatten()[0]
      vso = flasho.vj[j]
      Vso = flasho.nj[j] * vso
      bo = Vro / Vso
      deno = flasho.yji[j].dot(self.eos.mwi) / vso
      muo = -1.
      j = np.argwhere(flashg.sj == 0).flatten()[0]
      vsg = flashg.vj[j]
      Vsg = flashg.nj[j] * vsg
      deng = flashg.yji[j].dot(self.eos.mwi) / vsg
      gor = Vsg / Vso
      return bo, muo, deno, gor, deng
    elif gasexists:
      j = np.argwhere(flashg.sj == 0).flatten()[0]
      deng = flashg.yji[j].dot(self.eos.mwi) / flashg.vj[j]
      return -1., -1., -1., -1., deng
    elif oilexists:
      j = np.argwhere(flasho.sj == 1).flatten()[0]
      vso = flasho.vj[j]
      Vso = flasho.nj[j] * vso
      bo = Vro / Vso
      deno = flasho.yji[j].dot(self.eos.mwi) / vso
      muo = -1.
      return bo, muo, deno, 0., -1.
    else:
      return -1., -1., -1., -1., -1.


class cvdPT(_labprops):
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

    - `getPT_Z_lnphii_dP(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double], Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple of:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector[Double]` of shape `(Nc,)`.

    If solution methods for the saturation pressure determination and
    flash calculations are based on Newton's method, then it also must
    have:

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

    - `getPT_Z_lnphii_dP_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Vector[Double],
       Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple of:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to mole numbers of components as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  sepg: LabSeparator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) for the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the gas mixture collected from
    the cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  sepo: LabSeparator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) for the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the oil mixture at the current
    stage of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  mwc5: float
    Molar mass used to filter components for calculating pseudo oil-gas
    ratio. Default is `0.07215` [kg/mol].

  maxNp: int
    The maximum number of phases. Default is `2`. If the maximum number
    of phases would be greater than `2`, then the `NotImplementedError`
    will be raised due to the absence of the multiphase saturation
    pressure solver.

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
    sepg: LabSeparator,
    sepo: LabSeparator,
    mwc5: float = 0.07215,
    maxNp: int = 2,
    psatkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.sepg = sepg
    self.sepo = sepo
    self.maxNp = maxNp
    if maxNp == 2:
      self.solver = flash2pPT(eos, **kwargs)
      self.psatsolver = PsatPT(eos, **psatkwargs)
    elif maxNp > 2:
      raise NotImplementedError(
        'The multiphase saturation pressure solver is not implemented yet.'
      )
    else:
      raise ValueError(
        'The maximum number of phases must be greater than or equal to 2.'
      )
    self.c5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.c5pi]
    pass

  def run(
    self,
    PP: Vector[Double],
    T: float,
    yi: Vector[Double],
    Psat0: float,
    n0: float = 1.,
  ) -> LabResult:
    """Performs the CVD experiment for a given range of pressures, fixed
    temperature, composition of the mixture and initial guess of the
    saturation pressure.

    Parameters
    ----------
    PP: Vector[Double], shape (Ns,)
      A 1d-array with pressures corresponding to the stages of an
      experiment, where `Ns` is the number of stages. All pressures
      above the saturation pressure will be ignored.

    T: float
      The temperature held constant during the experiment.

    yi: Vector[Double], shape (Nc,)
      The global mole fractions of a mixture at the beginning of the
      experiment.

    Psat0: float
      An initial guess of the saturation pressure. It may be obtained
      from existing experimental data, if available. The initial guess
      of the saturation pressure would be refined by internal
      procedures.

    n0: float
      The initial amount of the fluid in a cell. Default is `1.0` [mol].

    Returns
    -------
    Constant volume depletion simulation results as an instance of the
    `LabResult`.
    """
    logger.info('Constant volume depletion (CVD).')
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *yi)
    res = self.psatsolver.run(Psat0, T, yi, n0, True)
    Psat = res.P
    logger.info('Saturation pressure: %.1f Pa', Psat)
    PP_filtered = PP[PP < Psat]
    Pk = np.hstack([Psat, PP_filtered])
    Nk = Pk.shape[0]
    ykji = np.zeros(shape=(Nk, self.maxNp, self.eos.Nc))
    Zkj = np.zeros(shape=(Nk, self.maxNp))
    fkj = np.zeros_like(Zkj)
    nk = np.zeros_like(Pk)
    props = np.empty(shape=(Nk, 11))
    Np = res.Np
    ykji[0, :Np] = res.yji
    Zkj[0, :Np] = res.Zj
    fkj[0, :Np] = res.fj
    nk[0] = n0
    gasexists = 0 in res.sj
    oilexists = 1 in res.sj
    if gasexists and oilexists:
      j = np.argwhere(res.sj == 0).flatten()[0]
      V0 = res.V
      n = n0
      bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
      j = np.argwhere(res.sj == 1).flatten()[0]
      bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
      props[0] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
    elif gasexists:
      j = np.argwhere(res.sj == 0).flatten()[0]
      V0 = res.V
      n = n0
      bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
      props[0] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
    elif oilexists:
      j = np.argwhere(res.sj == 1).flatten()[0]
      bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
      props[0] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
      logger.warning('There is no gas to remove. '
                     'The CVD experiment can not be completed.')
      out = LabResult(Pk, nk, fkj, ykji, Zkj, props)
      logger.info('%s', out)
      return out
    else:
      props[0] = -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.
      logger.warning('There is no gas to remove. '
                     'The CVD experiment can not be completed.')
      out = LabResult(Pk, nk, fkj, ykji, Zkj, props)
      logger.info('%s', out)
      return out
    for k, P in enumerate(PP_filtered, 1):
      res = self.solver.run(P, T, yi, n)
      Np = res.Np
      ykji[k, :Np] = res.yji
      Zkj[k, :Np] = res.Zj
      fkj[k, :Np] = res.fj
      gasexists = 0 in res.sj
      oilexists = 1 in res.sj
      if gasexists:
        j = np.argwhere(res.sj == 0).flatten()[0]
        ygi = res.yji[j]
        ng = res.nj[j]
        vg = res.vj[j]
        dVg = res.V - V0
        dng = dVg / vg
        dngi = ygi * dng
        bg, mug, deng, cc, c5p, denc = self._gasprops(ygi, vg)
        if oilexists:
          j = np.argwhere(res.sj == 1).flatten()[0]
          bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
          props[k] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
        else:
          props[k] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
        n -= dng
        if n < 0.:
          logger.warning('There is no gas to remove. '
                         'The CVD experiment can not be completed.')
          break
        yi = (res.ni - dngi) / n
      elif oilexists:
        j = np.argwhere(res.sj == 1).flatten()[0]
        bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
        props[k] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
        logger.warning('There is no gas to remove.'
                       'The CVD experiment can not be completed.')
        break
      else:
        props[k] = -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.
        logger.warning('There is no gas to remove.'
                       'The CVD experiment can not be completed.')
        break
      nk[k] = n
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    out = LabResult(Pk, nk, fkj, ykji, Zkj, props)
    logger.info('%s', out)
    return out


class ccePT(_labprops):
  """Multiphase constant composition (mass) expansion (CCE) experiment.

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
  eos: EosFlash2pPT | EosFlashNpPT
    An initialized instance of a PT-based equation of state. For
    two-phase flash calculations, it must have the following methods:

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

    If the solution method for two-phase flash calculations would be one
    of `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also must
    have:

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

    If the maximum number of phases would be greater than two, then
    this istance also must have:

    - `getPT_PIDj(P: float, T: float, yji: Matrix[Double])
       -> Vector[Integer]`
      For a given pressure [Pa], temperature [K], and mole compositions
      of `Np` phases (`Matrix[Double]` of shape `(Np, Nc)`), this method
      must return the phase identification number for each phase as a
      `Vector[Integer]` of shape `(Np,)` (`0` = vapour, `1` = liquid,
      etc).

    - `getPT_Z(P: float, T: float, yi: Vector[Double]) -> float`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return the
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

    If the solution method for multiphase flash calculations would be
    one of `'newton'`, `'ss-newton'` or `'qnss-newton'` then it also
    must have:

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

  sepg: LabSeparator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) for the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the gas mixture collected from
    the cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  sepo: LabSeparator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) for the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the oil mixture at the current
    stage of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  For the multiphase case, the oil phase will be identified as the
  first phase, which has zero as the designated phase ID.

  mwc5: float
    Molar mass used to filter components for calculating pseudo oil-gas
    ratio. Default is `0.07215` [kg/mol].

  maxNp: int
    The maximum number of phases. Default is `2`.

  **kwargs: dict
    Other arguments for a flash solver. It can contain such arguments
    as `method`, `tol`, `maxiter` and others, depending on the selected
    solver.

  Methods
  -------
  run(PP: Vector[Double], T: float, yi: Vector[Double]) -> LabResult
  This method performs the CCE experiment for a given range of
  pressures [Pa], fixed temperature [K] and composition of the mixture.
  """
  def __init__(
    self,
    eos: EosFlash2pPT | EosFlashNpPT,
    sepg: LabSeparator,
    sepo: LabSeparator,
    mwc5: float = 0.07215,
    maxNp: int = 2,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.sepg = sepg
    self.sepo = sepo
    self.Nc = eos.Nc
    self.maxNp = maxNp
    if maxNp == 2:
      self.solver = flash2pPT(eos, **kwargs)
    elif maxNp > 2:
      self.solver = flashNpPT(eos, maxNp=maxNp, **kwargs)
    else:
      raise ValueError(
        'The maximum number of phases must be greater than or equal to 2.'
      )
    self.c5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.c5pi]
    pass

  def run(
    self,
    PP: Vector[Double],
    T: float,
    yi: Vector[Double],
    n0: float = 1.,
  ) -> LabResult:
    """Performs the CCE experiment for a given range of pressures, fixed
    temperature and composition of the mixture.

    Parameters
    ----------
    PP: Vector[Double], shape (Ns,)
      A 1d-array with pressures corresponding to the stages of an
      experiment, where `Ns` is the number of stages.

    T: float
      The temperature held constant during the experiment.

    yi: Vector[Double], shape (Nc,)
      The global mole fractions of the reservoir fluid.

    n0: float
      The initial amount of the fluid in a cell. Default is `1.0` [mol].

    Returns
    -------
    Constant composition expansion simulation results as an instance of
    the `LabResult`.
    """
    logger.info('Constant composition expansion (CCE).')
    logger.info('T = %.2f K, zi =' + self.Nc * '%7.4f', T, *yi)
    Ns = PP.shape[0]
    ysji = np.zeros(shape=(Ns, self.maxNp, self.Nc))
    Zsj = np.zeros(shape=(Ns, self.maxNp))
    fsj = np.zeros_like(Zsj)
    ns = np.zeros_like(PP)
    props = np.empty(shape=(Ns, 11))
    for s, P in enumerate(PP):
      res = self.solver.run(P, T, yi)
      Np = res.Np
      ysji[s, :Np] = res.yji
      Zsj[s, :Np] = res.Zj
      fsj[s, :Np] = res.fj
      ns[s] = n0
      gasexists = 0 in res.sj
      oilexists = 1 in res.sj
      if gasexists and oilexists:
        j = np.argwhere(res.sj == 0).flatten()[0]
        bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
        j = np.argwhere(res.sj == 1).flatten()[0]
        bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
        props[s] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
      elif gasexists:
        j = np.argwhere(res.sj == 0).flatten()[0]
        bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
        props[s] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
      elif oilexists:
        j = np.argwhere(res.sj == 1).flatten()[0]
        bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
        props[s] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
      else:
        props[s] = -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.
    out = LabResult(PP, ns, fsj, ysji, Zsj, props)
    logger.info('%s', out)
    return out


class dlPT(_labprops):
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

    - `getPT_Z_lnphii_dP(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double], Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple of:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector[Double]` of shape `(Nc,)`.

    If solution methods for the saturation pressure determination and
    flash calculations are based on Newton's method, then it also must
    have:

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

    - `getPT_Z_lnphii_dP_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Vector[Double],
       Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple of:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to mole numbers of components as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  sepg: LabSeparator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) for the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the gas mixture collected from
    the cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  sepo: LabSeparator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) for the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the oil mixture at the current
    stage of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  mwc5: float
    Molar mass used to filter components for calculating pseudo oil-gas
    ratio. Default is `0.07215` [kg/mol].

  maxNp: int
    The maximum number of phases. Default is `2`. If the maximum number
    of phases would be greater than `2`, then the `NotImplementedError`
    will be raised due to the absence of the multiphase saturation
    pressure solver.

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
    sepg: LabSeparator,
    sepo: LabSeparator,
    mwc5: float = 0.07215,
    maxNp: int = 2,
    psatkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.sepg = sepg
    self.sepo = sepo
    self.maxNp = maxNp
    if maxNp == 2:
      self.solver = flash2pPT(eos, **kwargs)
      self.psatsolver = PsatPT(eos, **psatkwargs)
    elif maxNp > 2:
      raise NotImplementedError(
        'The multiphase saturation pressure solver is not implemented yet.'
      )
    else:
      raise ValueError(
        'The maximum number of phases must be greater than or equal to 2.'
      )
    self.c5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.c5pi]
    pass

  def run(
    self,
    PP: Vector[Double],
    T: float,
    yi: Vector[Double],
    Psat0: float,
    n0: float = 1.,
  ) -> LabResult:
    """Performs the DL experiment for a given range of pressures, fixed
    temperature, composition of the mixture and initial guess of the
    saturation pressure.

    Parameters
    ----------
    PP: Vector[Double], shape (Ns,)
      A 1d-array with pressures corresponding to the stages of an
      experiment, where `Ns` is the number of stages. All pressures
      above the saturation pressure will be ignored.

    T: float
      The temperature held constant during the experiment.

    yi: Vector[Double], shape (Nc,)
      The global mole fractions of a mixture at the beginning of the
      experiment.

    Psat0: float
      An initial guess of the saturation pressure. It may be obtained
      from existing experimental data, if available. The initial guess
      of the saturation pressure would be refined by internal
      procedures.

    n0: float
      The initial amount of the fluid in a cell. Default is `1.0` [mol].

    Returns
    -------
    Differential liberation simulation results as an instance of the
    `LabResult`.
    """
    logger.info('Differential liberation (DL).')
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *yi)
    res = self.psatsolver.run(Psat0, T, yi, True)
    Psat = res.P
    logger.info('Saturation pressure: %.1f Pa', Psat)
    PP_filtered = PP[PP < Psat]
    Pk = np.hstack([Psat, PP_filtered])
    Nk = Pk.shape[0]
    ykji = np.zeros(shape=(Nk, self.maxNp, self.eos.Nc))
    Zkj = np.zeros(shape=(Nk, self.maxNp))
    fkj = np.zeros_like(Zkj)
    nk = np.zeros_like(Pk)
    props = np.empty(shape=(Nk, 11))
    Np = res.Np
    ykji[0, :Np] = res.yji
    Zkj[0, :Np] = res.Zj
    fkj[0, :Np] = res.fj
    nk[0] = n0
    n = n0
    gasexists = 0 in res.sj
    oilexists = 1 in res.sj
    if gasexists and oilexists:
      j = np.argwhere(res.sj == 0).flatten()[0]
      bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
      j = np.argwhere(res.sj == 1).flatten()[0]
      bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
      props[0] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
    elif gasexists:
      j = np.argwhere(res.sj == 0).flatten()[0]
      bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
      props[0] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
    elif oilexists:
      j = np.argwhere(res.sj == 1).flatten()[0]
      bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
      props[0] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
    else:
      props[0] = -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.
    for k, P in enumerate(PP_filtered, 1):
      res = self.solver.run(P, T, yi, n)
      ykji[k] = res.yji
      Zkj[k] = res.Zj
      fkj[k] = res.fj
      gasexists = 0 in res.sj
      oilexists = 1 in res.sj
      if gasexists and oilexists:
        j = np.argwhere(res.sj == 0).flatten()[0]
        bg, mug, deng, cc, c5p, denc = self._gasprops(res.yji[j], res.vj[j])
        n -= res.nj[j]
        if oilexists:
          j = np.argwhere(res.sj == 1).flatten()[0]
          bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
          props[k] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
        else:
          props[k] = bg, mug, deng, cc, c5p, denc, -1., -1., -1., -1., -1.
      elif oilexists:
        j = np.argwhere(res.sj == 1).flatten()[0]
        bo, muo, dendo, gor, densg = self._oilprops(res.yji[j], res.vj[j])
        props[k] = -1., -1., -1., -1., -1., -1., bo, muo, dendo, gor, densg
      else:
        props[k] = -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.
      nk[k] = n
      if np.isclose(n, 0.) or n < 0.:
        logger.warning('There is only gas in the cell.'
                       'The DL experiment can not be completed.')
        break
    logger.info('Total produced amount of gas: %.3f mol.', n0 - n)
    out = LabResult(Pk, nk, fkj, ykji, Zkj, props)
    logger.info('%s', out)
    return out


class swellPT(_labprops):
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

    - `getPT_Z_lnphii_dP(P: float, T: float, yi: Vector[Double])
       -> tuple[float, Vector[Double], Vector[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple of:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector[Double]` of shape `(Nc,)`.

    If solution method for the saturation pressure determination is
    based on Newton's method, then it also must have:

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

    - `getPT_Z_lnphii_dP_dnj(P: float, T: float, yi: Vector[Double],
       n: float) -> tuple[float, Vector[Double], Vector[Double],
       Matrix[Double]]`
      For a given pressure [Pa], temperature [K], and mole composition
      (`Vector[Double]` of shape `(Nc,)`), this method must return a
      tuple of:

      - the compressibility factor of the mixture,
      - logarithms of fugacity coefficients of components as a
        `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
      - partial derivatives of logarithms of fugacity coefficients
        with respect to mole numbers of components as a `Matrix[Double]`
        of shape `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector[Double]`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  sepg: LabSeparator
    Before calculating properties of the gas phase (gas formation
    volume factor, condensate solubility in gas, etc.) for the current
    experiment stage, the gas mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the gas mixture collected from
    the cell during the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  sepo: LabSeparator
    Before calculating properties of the oil phase (oil formation
    volume factor, gas solubility in oil, etc.) for the current
    experiment stage, the oil mixture may undergo a series of separation
    steps, similar to what would be performed in the field. The gas and
    liquid phase compositions from the specified separation steps will
    be used to calculate properties of the oil mixture at the current
    stage of the CVD experiment.

    This parameter should be represented as an initialized instance of
    a class that must have the following methods:

    - `run(yi: Vector[Double], n: float = 1.) -> SeparatorResult`
      For a given composition of a mixture this method could perform
      a series of separation steps and return their outputs as an
      instance of `SeparatorResult`.

    Also, this instance must have attributes:

    - `gasstream: str`
      The name of the gas phase stream from which flash calculation
      results will be taken to calculate the gas phase properties at
      the standard conditions.

    - `gasstage: int`
      The stage number (index) of the gas phase stream from which flash
      calculation results will be taken to calculate the gas phase
      properties at the standard conditions.

    - `oilstream: str`
      The name of the oil phase stream from which flash calculation
      results will be taken to calculate the oil phase properties at
      the standard conditions.

    - `oilstage: int`
      The stage number (index) of the oil phase stream from which flash
      calculation results will be taken to calculate the oil phase
      properties at the standard conditions.

  mwc5: float
    Molar mass used to filter components for calculating pseudo oil-gas
    ratio. Default is `0.07215` [kg/mol].

  **kwargs: dict
    Other arguments for the saturation pressure solver. It can contain
    such arguments as `method`, `tol`, `maxiter` and others, depending
    on the selected solver.
  """
  def __init__(
    self,
    eos: EosPsatPT,
    sepg: LabSeparator,
    sepo: LabSeparator,
    mwc5: float = 0.07215,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.sepg = sepg
    self.sepo = sepo
    self.solver = PsatPT(eos, densort=False, **kwargs)
    self.c5pi = eos.mwi >= mwc5
    self.mwc5pi = eos.mwi[self.c5pi]
    pass

  def run(
    self,
    P0: float,
    T: float,
    zi: Vector[Double],
    yi: Vector[Double],
    fj: Vector[Double],
  ) -> LabResult:
    """Performs the swelling experiment.

    Parameters
    ----------
    P0: float
      Initial guess of the saturation pressure [Pa] of the reservoir
      fluid.

    T: float
      Experiment temperature [K].

    zi: Vector[Double], shape (Nc,)
      Mole fractions of components of the reservoir fluid as a
      `Vector[Double]` of shape `(Nc,)`, where `Nc` is the number of
      components.

    yi: Vector[Double], shape (Nc,)
      Mole fractions of components of the injected fluid as a
      `Vector[Double]` of shape `(Nc,)`.

    fj: Vector[Double], shape (Ns,)
      Mole fractions of the injected fluid that need to be dissolved in
      the reservoir fluid. It should be represented as a
      `Vector[Double]` of shape `(Ns,)`, where `Ns` is the number of
      stages of the experiment.

    Returns
    -------
    Swelling simulation results as an instance of the `LabResult`.
    """
    logger.info('Swelling test.')
    logger.info('T = %.2f K, zi =' + self.eos.Nc * '%7.4f', T, *zi)
    xi = zi.copy()
    Ns = fj.shape[0]
    Ps = np.zeros(shape=(Ns,))
    ysji = np.zeros(shape=(Ns, 2, self.eos.Nc))
    Zsj = np.zeros(shape=(Ns, 2))
    fsj = np.zeros_like(Zsj)
    ns = np.zeros_like(Ps)
    props = np.empty(shape=(Ns, 11))
    for i, F in enumerate(fj):
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
        fsj[i, 0] = 1.
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[0], vj[0])
        bo, muo, dendo, gor, densg = self._oilprops(yji[1], vj[1])
      else:
        fsj[i, 1] = 1.
        bg, mug, deng, cc, c5p, denc = self._gasprops(yji[1], vj[1])
        bo, muo, dendo, gor, densg = self._oilprops(yji[0], vj[0])
      props[i] = bg, mug, deng, cc, c5p, denc, bo, muo, dendo, gor, densg
    out = LabResult(Ps, ns, fsj, ysji, Zsj, props)
    logger.info('%s', out)
    return out
