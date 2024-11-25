import numpy as np


def linsolver2d(A, b):
  D = 1. / (A[0,0] * A[1,1] - A[0,1] * A[1,0])
  return np.array([
    D * (A[1,1] * b[0] - A[0,1] * b[1]),
    D * (A[0,0] * b[1] - A[1,0] * b[0]),
  ])


def brentopt(func, x0, minx, maxx, Niter=50, tol=1e-8):
  golden = 0.3819660
  # x = w = v = maxx
  # x = w = v = minx
  x = w = v = x0
  fw = fv = fx = func(x)
  delta2 = delta = 0
  i = 0
  while i < Niter:
    mid = .5 * (minx + maxx)
    fract1 = tol * np.abs(x) + .25 * tol
    fract2 = 2. * fract1
    if np.abs(x - mid) <= (fract2 - .5 * (maxx - minx)):
      break
    if np.abs(delta2) > fract1:
      r = (x - w) * (fx - fv)
      q = (x - v) * (fx - fw)
      p = (x - v) * q - (x - w) * r
      q = 2. * (q - r)
      if q > 0.:
        p = -p
      q = np.abs(q)
      td = delta2
      delta2 = delta
      if ((np.abs(p) >= np.abs(.5 * q * td))
          or (p <= q * (minx - x))
          or (p >= q * (maxx - x))):
        if x >= mid:
          delta2 = minx - x
        else:
          delta2 = maxx - x
        delta = golden * delta2
      else:
        delta = p / q
        u = x + delta
        if ((u - minx) < fract2) or ((maxx - u) < fract2):
          if mid - x < 0.:
            # delta = -np.abs(fract1)
            delta = -fract1
          else:
            # delta = np.abs(fract1)
            delta = fract1
    else:
      if x >= mid:
        delta2 = minx - x
      else:
        delta2 = maxx - x
      delta = golden * delta2
    if np.abs(delta) >= fract1:
      u = x + delta
    else:
      if delta > 0.:
        # u = x + np.abs(fract1)
        u = x + fract1
      else:
        # u = x - np.abs(fract1)
        u = x - fract1
    fu = func(u)
    if fu <= fx:
      if u >= x:
        minx = x
      else:
        maxx = x
      v = w;
      w = x
      x = u
      fv = fw
      fw = fx
      fx = fu
    else:
      if u < x:
        minx = u
      else:
        maxx = u
      if (fu <= fw) or (w == x):
        v = w
        w = u
        fv = fw
        fw = fu
      elif (fu <= fv) or (v == x) or (v == w):
        v = u
        fv = fu
    i += 1
  return x


