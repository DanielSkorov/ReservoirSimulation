import numpy as np

from custom_types import (
  Vector,
  Matrix,
  LinAlgError,
)

from futils import (
  linalg as fla,
)

def dgem(A: Matrix, b: Vector) -> Vector:
  assert A.dtype == b.dtype == np.float64
  fA = np.asfortranarray(A)
  fb = np.asfortranarray(b)
  x, singular = fla.dgem(fA, fb)
  if singular:
    raise LinAlgError('Singular matrix.')
  return x


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


def mineig_newton(Q, x0, lmbd0=0., tol=1e-5, Niter=10):
  lmbd = lmbd0
  x = x0
  u = np.hstack([x, lmbd])
  I = np.identity(x.shape[0])
  M = Q - lmbd * I
  J = np.block([[M, -x[:,None]], [2. * x, 0.]])
  b = np.hstack([M.dot(x), x.dot(x) - 1.])
  # print(f'Iteration #0:\n\t{x = }\n\t{lmbd = }\n\t{b = }')
  k = 1
  while (np.linalg.norm(b) > tol) & (k < Niter):
    du = np.linalg.solve(J, b)
    u -= du
    x = u[:-1]
    lmbd = u[-1]
    M = Q - lmbd * I
    J = np.block([[M, -x[:,None]], [2. * x, 0.]])
    b = np.hstack([M.dot(x), x.dot(x) - 1.])
    # print(f'Iteration #{k}:\n\t{x = }\n\t{lmbd = }\n\t{b = }')
    k += 1
  return k, x, lmbd


def mineig_inviter(Q, b0=None, s=0., tol=1e-5, Niter=10):
  k = 0
  if b0 is None:
    bk = np.ones(shape=(Q.shape[0],), dtype=Q.dtype)
  else:
    bk = b0
  if s != 0.:
    # M = Q - s * np.identity(n)
    M = np.linalg.inv(Q - np.diagflat(np.full_like(bk, s)))
  else:
    # M = Q
    M = np.linalg.inv(Q)
  # bkp1 = np.linalg.solve(M, bk)
  bkp1 = M.dot(bk)
  bkp1 /= np.linalg.norm(bkp1)
  k += 1
  while (np.linalg.norm(bkp1 - bk) > tol) & (k < Niter):
    bk = bkp1
    # print(f'Iteration #{k}: {bk = }')
    # bkp1 = np.linalg.solve(M, bk)
    bkp1 = M.dot(bk)
    bkp1 /= np.linalg.norm(bkp1)
    k += 1
  lmbd = Q[0].dot(bkp1) / bkp1[0]
  return k, bkp1, lmbd


def mineig_rayquot(Q, x0=None, lmbd0=None, tol=1e-5, Niter=10):
  if x0 is None:
    # xk = np.random.uniform(size=Q.shape[0], dtype=Q.dtype)
    # xk /= np.linalg.norm(xk)
    xk = np.ones(shape=(Q.shape[0],), dtype=Q.dtype)
  else:
    xk = x0
  if lmbd0 is None:
    lmbdk = xk.dot(Q).dot(xk)
  else:
    lmbdk = lmbd0
  # print(f'Iteration #0:\n\t{xk = }\n\t{lmbdk = }')
  M = Q - np.diagflat(np.full_like(xk, lmbdk))
  xkp1 = np.linalg.solve(M, xk)
  xkp1 /= np.linalg.norm(xkp1)
  lmbdkp1 = xkp1.dot(Q).dot(xkp1)
  k = 1
  while (np.abs((lmbdkp1 - lmbdk) / lmbdk) > tol) & (k < Niter):
    xk = xkp1
    lmbdk = lmbdkp1
    # print(f'Iteration #{k}:\n\t{xk = }\n\t{lmbdk = }')
    M = Q - np.diagflat(np.full_like(xk, lmbdk))
    xkp1 = np.linalg.solve(M, xk)
    xkp1 /= np.linalg.norm(xkp1)
    lmbdkp1 = xkp1.dot(Q).dot(xkp1)
    k += 1
  return k, xkp1, lmbdkp1

