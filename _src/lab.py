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
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      s = (f"Pressure steps:\n{self.Ps} Pa\n"
           f"Phase mole fractions:\n{self.Fsj.T}\n"
           f"Phase compressibility factors:\n{self.Zsj.T}\n")
    return s


def cvdPT(
  PP: Vector,
  T: Scalar,
  yi: Vector,
  eos: PsatEosPT,
  Psat0: Scalar,
  n0: Scalar = 1.,
  flashkwargs: dict = {},
  psatkwargs: dict = {},
) -> LabResult:
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
