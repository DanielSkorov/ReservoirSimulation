import logging

import numpy as np

from custom_types import (
  ScalarType,
  VectorType,
  EOSPTType,
)

from constants import (
  R,
)

from flash import (
  flash2pPT,
)

from boundary import (
  PsatPT,
)


logger = logging.getLogger('lab')


class LabResult(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      s = (f"Pressure steps:\n{self.Ps} Pa\n"
           f"Phase mole fractions:\n{self.Fsj.T}\n"
           f"Phase compressibility factors:\n{self.Zsj.T}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


def cvdPT(
  T: ScalarType,
  yi: VectorType,
  eos: EOSPTType,
  PP: VectorType,
  Psat0: ScalarType = np.float64(20e6),
  n: ScalarType = 1.,
  flashkwargs: dict = {},
  psatkwargs: dict = {},
) -> LabResult:
  logger.debug(
    'Constant volume depletion for:\n\tT = %s K\n\tyi = %s', T, yi,
  )
  flash = flash2pPT(eos, **flashkwargs)
  zi = yi.copy()
  Psatres = PsatPT(eos, **psatkwargs).run(T, zi, Psat0)
  Psat = Psatres.P
  logger.debug(
    'Saturation pressure calculation:\n\tPsat = %s Pa\n\tsuccess = %s',
    Psat, Psatres.success,
  )
  Zj = Psatres.Zj
  V0 = n * Zj[0] * R * T / Psat
  PP_filtered = PP[PP < Psat]
  PP_out = np.hstack([Psat, PP_filtered])
  Npoints = PP_out.shape[0]
  yji_out = np.empty(shape=(Npoints, 2, eos.Nc))
  yji_out[0] = Psatres.yji
  Zj_out = np.empty(shape=(Npoints, 2))
  Zj_out[0] = Zj
  Fj_out = np.empty(shape=(Npoints, 2))
  Fj_out[0] = np.array([1., 0.])
  n_out = np.empty_like(PP_out)
  n_out[0] = n
  for i, P in enumerate(PP_filtered):
    flashres = flash.run(P, T, zi)
    yji = flashres.yji
    Zj = flashres.Zj
    Fj = flashres.Fj
    yji_out[i+1] = yji
    Zj_out[i+1] = Zj
    Fj_out[i+1] = Fj
    n_out[i+1] = n
    logger.debug(
      'For pressure %s Pa, composition %s and mole number %s mol:\n'
      '\tZj = %s\n\tFj = %s\n\tflash was successful: %s',
      P, zi, n, Zj, Fj, flashres.success,
    )
    nj = Fj * n
    V = R * T * nj.dot(Zj) / P
    rhog = P / (Zj[0] * R * T)
    dV = V - V0
    dng = dV * rhog
    dngi = yji[0] * dng
    ngi = yji[0] * nj[0]
    nginew = ngi - dngi
    ngnew = nj[0] - dng
    n = nj[1] + ngnew
    zi = (nginew + nj[1] * yji[1]) / n
  return LabResult(Ps=PP_out, ysji=yji_out, Zsj=Zj_out, Fsj=Fj_out, ns=n_out,
                   success=True)


def ccePT(
  T: ScalarType,
  yi: VectorType,
  eos: EOSPTType,
  PP: VectorType,
  n: ScalarType = 1.,
  flashkwargs: dict = {},
) -> LabResult:
  logger.debug(
    'Constant composition expansion for:\n\tT = %s K\n\tyi = %s', T, yi,
  )
  Npoints = PP.shape[0]
  yji_out = np.empty(shape=(Npoints, 2, eos.Nc))
  Zj_out = np.empty(shape=(Npoints, 2))
  Fj_out = np.empty(shape=(Npoints, 2))
  n_out = np.empty_like(PP)
  flash = flash2pPT(eos, **flashkwargs)
  for i, P in enumerate(PP):
    flashres = flash.run(P, T, yi)
    yji = flashres.yji
    Zj = flashres.Zj
    Fj = flashres.Fj
    yji_out[i] = yji
    Zj_out[i] = Zj
    Fj_out[i] = Fj
    n_out[i] = n
    logger.debug(
      'For pressure step %s Pa:\n'
      '\tZj = %s\n\tFj = %s\n\tflash was successful: %s',
      P, Zj, Fj, flashres.success,
    )
  return LabResult(Ps=PP, ysji=yji_out, Zsj=Zj_out, Fsj=Fj_out, ns=n_out,
                   success=True)

