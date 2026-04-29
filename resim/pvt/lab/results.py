from typing import (
  ClassVar,
  Self,
)

from dataclasses import (
  dataclass,
  field,
)

from numpy import (
  array as np_array,
)

from resim.pvt.datatypes import (
  arr2str,
  Float,
  Integer,
  Vector,
  State,
)

from resim.pvt.utils import (
  PhaseFinder,
  findex,
  denmin,
  denmax,
)

from resim.pvt.eos import (
  Eos,
)

from resim.pvt.sep import (
  Separator,
  stsep,
)


@dataclass(eq=False, slots=True)
class BoState(object):
  """A dataclass for black oil parameters that characterize a state of
  a mixture.

  Attributes
  ----------
  Np: int
    The number of phases.

  dj: Vector[Float], shape (Np,)
    Mass densities of `Np` phases [kg/rm³].

  bj: Vector[Float], shape (Np,)
    Formation volume factors of `Np` phases [rm³/sm³].

  rj: Vector[Float], shape (Np,)
    Solution ratios of `Np` phases [sm³/sm³]. For gas-like phases,
    this parameter characterizes condensate solubility in gas. For
    liquid-like phases, this parameter characterizes gas solubility
    in liquid.

  muj: Vector[Float], shape (Np,)
    Viscosities of `Np` phases [cP].

  sj: Vector[Float], shape (Np,)
    Volume fractions of `Np` phases.

  pidj: Vector[Integer], shape (Np,)
    Phase designation indices of `Np` phases (`0` = vapour,
    `1` = liquid, etc.).

  tmpl: str
    A template for the table representation of this dataclass. This
    attribute is valued internally after class initialization.

  Class variables
  ---------------
  gassep: ClassVar[Separator]
    A callable object that can be used to perform a separation procedure
    for the gas phase. Default is `stsep`.

  liqsep: ClassVar[Separator]
    A callable object that can be used to perform a separation procedure
    for the liquid phase. Default is `stsep`.

  gasfinder: ClassVar[PhaseFinder]
    A callable object that finds the gas phase in a resulting state
    obtained from a separation procedure. Default is `findex`.

  condfinder: ClassVar[PhaseFinder]
    A callable object that finds the condensate phase in a resulting
    state obtained from the gas separation procedure. Default is
    `denmin`.

  liqfinder: ClassVar[PhaseFinder]
    A callable object that finds other liquid phases (oil or water) in
    a resulting state obtained from the liquid separation procedure.
    Default is `denmax`.
  """
  Np: int
  P: float
  T: float
  V: float
  n: float
  sj: Vector[Float]
  dj: Vector[Float]
  pidj: Vector[Integer]
  bj: Vector[Float]
  rj: Vector[Float]
  muj: Vector[Float]
  tmpl: str = field(init=False, default='')
  gassep: ClassVar[Separator] = stsep
  liqsep: ClassVar[Separator] = stsep
  gasfinder: ClassVar[PhaseFinder] = findex
  condfinder: ClassVar[PhaseFinder] = denmin
  liqfinder: ClassVar[PhaseFinder] = denmax

  def __post_init__(self):
    Np = self.Np
    self.tmpl = (
      '{0:7.3f}{1:8.2f}{2:9.1e}{3:9.4f}'
      + ''.join(
        [
          (
            f'{{{i}:5d}}{{{i+Np}:9.4f}}{{{i+2*Np}:12.5g}}{{{i+3*Np}:13.5f}}'
            f'{{{i+4*Np}:13.3e}}{{{i+5*Np}:9.4f}}'
          )
          for i in range(4, 4 + Np)
        ]
      )
    )
    pass

  def __str__(self, prefix: str = '', shift: str = '') -> str:
    """Generate the text representation of the dataclass.

    Parameters
    ----------
    prefix: str
      The prefix for the text representation. Default is `''`.

    shift: str
      The shift for each row of the text representation.
      Default is `''`.

    Returns
    -------
    The text representation of the dataclass.
    """
    return prefix + (
      f'{shift}Pressure: {self.P / 1e6:.3f} [MPa].\n'
      f'{shift}Temperature: {self.T - 273.15:.2f} [°C].\n'
      f'{shift}Volume: {self.V:.3e} [m³].\n'
      f'{shift}Mole number: {self.n:.4f} [mol].\n'
      f'{shift}Phase IDs: {self.pidj}.\n'
      f'{shift}Phase volume fractions: {arr2str(self.sj, 4)}.\n'
      f'{shift}Phase densities: {arr2str(self.dj, 3)} [kg/m³].\n'
      f'{shift}Phase FVFs: {arr2str(self.bj, 5)} [rm³/sm³].\n'
      f'{shift}Phase solution ratios: {arr2str(self.rj, 3)} [sm³/sm³].\n'
      f'{shift}Phase viscosities: {arr2str(self.muj, 3)} [cP].'
    )

  def __repr__(
    self,
    phases: int | None = None,
    head_prefix: str = '',
    row_prefix: str = '',
  ) -> str:
    """Generate the table representation of the dataclass.

    Parameters
    ----------
    phases: int | None
      The number of phases for which a table header must be generated.
      If it is `0`, the table header is omitted. If it is `None`, the
      number of phases for the header is equal to the `Np` attribute of
      the dataclass. Default is `None`.

    head_prefix: str
      The prefix for the table header. Default is `''`.

    row_prefix: str
      The prefix for the row of the table. Default is `''`.

    Returns
    -------
    The table representation of the dataclass.
    """
    if phases is None:
      head = (
        head_prefix
        + 'P [MPa]  T [°C]   V [m³]  n [mol]'
        + self.Np * (
          '  PID  s [fr.]  ρ [kg/rm³]  b [rm³/sm³]  r [sm³/sm³]   μ [cP]'
        )
        + '\n'
      )
    elif phases == 0:
      head = ''
    else:
      head = (
        head_prefix
        + 'P [MPa]  T [°C]   V [m³]  n [mol]'
        + phases * (
          '  PID  s [fr.]  ρ [kg/rm³]  b [rm³/sm³]  r [sm³/sm³]   μ [cP]'
        )
        + '\n'
      )
    return head + row_prefix + self.tmpl.format(
      self.P / 1e6, self.T - 273.15, self.V, self.n,
      *self.pidj, *self.sj, *self.dj, *self.bj, *self.rj, *self.muj,
    )

  def __format__(self, fmt: str) -> str:
    """Generate a representation of the dataclass for a given format.

    Parameters
    ----------
    fmt: str
      A given format. Available options are listed in the following
      table.

      +---------------+------------------------------------------------+
      | Format        | Description                                    |
      +===============+================================================+
      | `'s'` or `''` | The text representation of the dataclass.      |
      +---------------+------------------------------------------------+
      | `'r'`         | The table representation of the dataclass.     |
      +---------------+------------------------------------------------+

    Returns
    -------
    A representation of the dataclass.

    Raises
    -----
    ValueError
      This exceptioon is raised if an unknown format is passed.
    """
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')

  @classmethod
  def from_state(cls, s: State, eos: Eos, *args) -> Self:
    """A class method used to create a new instance of the dataclass
    from a given `State`.

    Parameters
    ----------
    s: State
      A given instance of the `State` dataclass.

    eos: Eos
      An initialized instance of an equation of state that can be
      used to perform separation procedures to calculate black-oil
      properties of phases.

    *args
      Other arguments that will be ignored.

    Returns
    -------
    An instance of the dataclass.
    """
    muj = []
    bj = []
    rj = []
    for (pid, yi, vrc) in zip(s.pidj, s.yji, s.vj):
      if pid == 0:
        sepres = cls.gassep(eos, yi)
        try:
          liqstate = sepres.get_liqstate()
          j = cls.condfinder(liqstate, 1)
          Vlsc = liqstate.Vj[j]
        except (KeyError, IndexError):
          Vlsc = 0.
        try:
          gasstate = sepres.get_gasstate()
          j = cls.gasfinder(gasstate, 0)
          Vgsc = gasstate.Vj[j]
          bg = vrc / Vgsc
          Cl = Vlsc / Vgsc
        except (KeyError, IndexError):
          bg = -1.
          Cl = -1.
        muj.append(-1.)
        bj.append(bg)
        rj.append(Cl)
      else:
        sepres = cls.liqsep(eos, yi)
        try:
          gasstate = sepres.get_gasstate()
          j = cls.gasfinder(gasstate, 0)
          Vgsc = gasstate.Vj[j]
        except (KeyError, IndexError):
          Vgsc = 0.
        try:
          liqstate = sepres.get_liqstate()
          j = cls.liqfinder(liqstate, 1)
          Vlsc = liqstate.Vj[j]
          bl = vrc / Vlsc
          Cg = Vgsc / Vlsc
        except (KeyError, IndexError):
          bl = -1.
          Cg = -1.
        muj.append(-1.)
        bj.append(bl)
        rj.append(Cg)
    return cls(
      s.Np, s.P, s.T, s.V, s.n, s.sj, s.dj, s.pidj,
      np_array(bj), np_array(rj), np_array(muj),
    )
