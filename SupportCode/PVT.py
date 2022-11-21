import numpy as np


class core():
    R = 8.314472

    @staticmethod
    def repeat(arr, axis, times=None):
        if times is None:
            times = arr.shape[0]
        arr0 = np.empty(shape=np.insert(arr.shape, axis, times))
        arr0[:] = arr.reshape(np.insert(arr.shape, axis, 1))
        return arr0

    @staticmethod
    def calc_cardano(k1, k2, k3):
        q = (k1**2 - 3 * k2) / 9
        r = (2 * k1**3 - 9 * k1 * k2 + 27 * k3) / 54
        if r**2 > q**3:
            s = -np.sign(r) * (np.abs(r) + (r**2 - q**3)**0.5)**(1/3)
            t = q / s if s != 0 else 0
            return [(s + t) - k1 / 3]
        else:
            theta = np.arccos(r / q**(3/2))
            return [-2 * q**0.5 * np.cos(theta / 3) - k1 / 3,
                    -2 * q**0.5 * np.cos((theta + 2 * np.pi) / 3) - k1 / 3,
                    -2 * q**0.5 * np.cos((theta - 2 * np.pi) / 3) - k1 / 3]

    @staticmethod
    def calc_cardano_arr(k1_arr, k2_arr, k3_arr):
        sol = []
        for i, (k1, k2, k3) in enumerate(zip(k1_arr, k2_arr, k3_arr)):
            q = (k1**2 - 3 * k2) / 9
            r = (2 * k1**3 - 9 * k1 * k2 + 27 * k3) / 54
            if r**2 > q**3:
                s = -np.sign(r) * (np.abs(r) + (r**2 - q**3)**0.5)**(1/3)
                t = q / s if s != 0 else 0
                sol.append([(s + t) - k1 / 3])
            else:
                theta = np.arccos(r / q**(3/2))
                sol.append([-2 * q**0.5 * np.cos(theta / 3) - k1 / 3,
                            -2 * q**0.5 * np.cos((theta + 2 * np.pi) / 3) - k1 / 3,
                            -2 * q**0.5 * np.cos((theta - 2 * np.pi) / 3) - k1 / 3])
        return sol

    @staticmethod
    def empty_object():
        class Object(object): pass
        return Object

    @staticmethod
    def replace(arr, index, value):
        copy = arr.copy()
        np.put(copy, index, value)
        return copy

    @staticmethod
    def replace_duplicates(string):
        elements = {i: string.count(i) for i in string}
        return [e + str(i) if elements[e] > 1 else e for e in elements for i in range(elements[e])]

    pass


# Атрибут comp_type класса eos_srk_pr определяет тип компонента для выбора корреляции коэффициента попарного взаимодействия между компонентами:
# 0 - углеводородный компонент, для расчета коэффициентов попарного взаимодействия которого используется корреляция GCM, если молярные объемы в критической точке (vc) компонентов не заданы (None), иначе - корреляция Oellrich
# 1 - углеводородный компонент, для расчета коэффициента попарного взаимодействия с компонентом "вода" которого применяется корреляция Сорейде-Уитсона (только в водной фазе)
# 2 - вода, как компонент
# 3 - диоксид углерода (корреляция Сорейде-Уитсона, только для водной фазы)
# 4 - сероводород (корреляция Сорейде-Уитсона, только для водной фазы)
# 5 - азот (корреляция Сорейде-Уитсона, только для водной фазы)


class eos_srk_pr(core):
    def __init__(self, Pc, Tc, w, comp_type, c, Akl=None, Bkl=None, alpha_matrix=None, vc=None, bip_exp=1.2, dij=None, n=1):
        self.Nc = len(Pc)
        self.comp_type = comp_type
        self.c = c
        if c == 0:
            self.delta1 = 0
            self.delta2 = 1
            self.kappa = 0.48508 + 1.55171 * w - 0.15613 * w**2
        elif c == 1:
            self.delta1 = 1 - 2**0.5
            self.delta2 = 1 + 2**0.5
            self.kappa = np.where(w <= 0.491, 0.37464 + 1.54226 * w - 0.26992 * w**2, 0.379642 + 1.48503 * w - 0.164423 * w**2 + 0.016666 * w**3)
        self.Pc = Pc
        self.Tc = Tc
        self.n = n
        self.w = w
        self.ai = 0.45724 * (self.R * Tc)**2 / Pc
        self.bi = 0.07780 * self.R * Tc / Pc
        self.sw = False
        if dij is None:
            if vc is None:
                self.Akl = Akl
                self.Bkl = Bkl
                self.Ng = len(Akl)
                self.DS_pow = np.divide(Bkl, Akl, out=np.ones_like(Akl), where=Akl!=0) - 1
                self.alpha_matrix = alpha_matrix
                self.gcm = True
            else:
                self.gcm = False
                self.dij = self.calc_bips_oellrich(vc, bip_exp)
        else:
            self.gcm = False
            self.dij = dij
        pass

    def calc_alphai(self, T):
        return self.ai * (1 + self.kappa * (1 - (T / self.Tc)**0.5))**2

    def calc_bips_gcm(self, T, alphai):
        DS = np.sum((self.alpha_matrix.reshape(self.Nc, 1, self.Ng, 1) - self.alpha_matrix.reshape(1, self.Nc, self.Ng, 1)) * (self.alpha_matrix.reshape(self.Nc, 1, 1, self.Ng) - self.alpha_matrix.reshape(1, self.Nc, 1, self.Ng)) * \
                    self.Akl.reshape(1, 1, self.Ng, self.Ng) * (298.15 / T) ** (np.divide(self.Bkl, self.Akl, out=np.ones_like(self.Akl), where=self.Akl!=0) - 1).reshape(1, 1, self.Ng, self.Ng), axis=(2, 3))
        deltai = alphai**0.5 / self.bi
        return (DS / (-2) - (deltai.reshape(1, self.Nc) - deltai.reshape(self.Nc, 1))**2) / (2 * np.outer(deltai, deltai))

    def calc_bips_oellrich(self, vc, bip_exp):
        vc13 = vc**(1/3)
        dij = 1 - (2 * np.sqrt(np.outer(vc13, vc13)) / (vc13.reshape(1, self.Nc) + vc13.reshape(self.Nc, 1)))**bip_exp
        for i in range(0, self.Nc):
            for j in range(i, self.Nc):
                if i != j:
                    if self.comp_type[i] == 0 and self.comp_type[j] == 5 or self.comp_type[j] == 0 and self.comp_type[i] == 5:
                        if self.comp_type[i] == 0:
                            hc = i
                        else:
                            hc = j
                        if self.w[hc] <= 0.098:
                            dij[i][j] = dij[j][i] = 0.035
                        elif self.w[hc] <= 0.251:
                            dij[i][j] = dij[j][i] = 0.095
                        elif self.w[hc] <= 0.444:
                            dij[i][j] = dij[j][i] = 0.1
                        else:
                            dij[i][j] = dij[j][i] = 0.12
                    elif self.comp_type[i] == 0 and self.comp_type[j] == 3 or self.comp_type[j] == 0 and self.comp_type[i] == 3:
                        dij[i][j] = dij[j][i] = 0.125
                    elif self.comp_type[i] == 0 and self.comp_type[j] == 4 or self.comp_type[j] == 0 and self.comp_type[i] == 4:
                        if self.comp_type[i] == 0:
                            hc = i
                        else:
                            hc = j
                        if self.w[hc] <= 0.152:
                            dij[i][j] = dij[j][i] = 0.07
                        elif self.w[hc] <= 0.275:
                            dij[i][j] = dij[j][i] = 0.06
                        elif self.w[hc] <= 0.391:
                            dij[i][j] = dij[j][i] = 0.04
                        else:
                            dij[i][j] = dij[j][i] = 0.0
                    elif self.comp_type[i] == 0 and self.comp_type[j] == 2 or self.comp_type[j] == 0 and self.comp_type[i] == 2:
                        dij[i][j] = dij[j][i] = 0.48
                    elif self.comp_type[i] == 3 and self.comp_type[j] == 5 or self.comp_type[j] == 3 and self.comp_type[i] == 5:
                        dij[i][j] = dij[j][i] = -0.02
                    elif self.comp_type[i] == 4 and self.comp_type[j] == 5 or self.comp_type[j] == 4 and self.comp_type[i] == 5:
                        dij[i][j] = dij[j][i] = 0.176
                    elif self.comp_type[i] == 2 and self.comp_type[j] == 5 or self.comp_type[j] == 2 and self.comp_type[i] == 5:
                        dij[i][j] = dij[j][i] = 0.275
                    elif self.comp_type[i] == 3 and self.comp_type[j] == 4 or self.comp_type[j] == 3 and self.comp_type[i] == 4:
                        dij[i][j] = dij[j][i] = 0.096
                    elif self.comp_type[i] == 3 and self.comp_type[j] == 2 or self.comp_type[j] == 3 and self.comp_type[i] == 2:
                        dij[i][j] = dij[j][i] = 0.2
                    elif self.comp_type[i] == 4 and self.comp_type[j] == 2 or self.comp_type[j] == 4 and self.comp_type[i] == 2:
                        dij[i][j] = dij[j][i] = 0.12
        return dij

    def calc_Z(self, A, B, Np, Zl, Zv):
        Zs = self.calc_cardano_arr(self.c * B - 1, A - (self.c + 1) * B - (2 * self.c + 1) * B**2, -A * B + self.c * (B**2 + B**3))
        if Zl:
            return np.min(Zs, axis=1)
        if Zv:
            return np.max(Zs, axis=1)
        Z = np.zeros_like(A)
        for k in range(Np):
            Z[k] = Zs[k][0]
            if len(Zs[k]) > 1:
                for i in range(1, 3):
                    if np.log((Zs[k][i] - B[k]) / (Z[k] - B[k])) + (Z[k] - Zs[k][i]) + A[k] * np.log(((Z[k] + B[k] * self.delta1) * (Zs[k][i] + B[k] * self.delta2)) / \
                                                                                                     ((Z[k] + B[k] * self.delta2) * (Zs[k][i] + B[k] * self.delta1))) / (B[k] * (self.delta2 - self.delta1)) > 0:
                        Z[k] = Zs[k][i]
        return Z

    def eos_run(self, y, P, T, phases, Zl=False, Zv=False):
        res = self.empty_object()
        res.Np = len(phases)
        res.y = y
        res.P = P
        res.T = T
        res.phases = phases
        res.alphai = self.calc_alphai(T)
        if self.gcm:
            res.dij = self.calc_bips_gcm(T, res.alphai)
        else:
            res.dij = self.dij
        res.alphaij = self.repeat(np.outer(res.alphai, res.alphai)**0.5, axis=0, times=res.Np) * (1 - res.dij)
        res.alpham = np.sum(np.einsum('yi,yj->yij', y, y) * res.alphaij, axis=(1, 2)).reshape(res.Np, 1)
        res.bm = np.sum(y * self.bi, axis=1).reshape(res.Np, 1)
        res.A = res.alpham * P / (self.R**2 * T**2)
        res.B = res.bm * P / (self.R * T)
        res.Z = self.calc_Z(res.A, res.B, res.Np, Zl, Zv)
        res.v = res.Z * self.R * T / P
        res.gZ = np.log(res.Z - res.B)
        res.gphi = (2 * np.sum(res.alphaij * self.repeat(y, axis=1, times=self.Nc), axis=2) - res.alpham * self.repeat(self.bi, axis=0, times=res.Np) / res.bm) / (self.R * T * res.bm)
        res.fZ = np.log((res.Z + res.B * self.delta1) / (res.Z + res.B * self.delta2))
        res.lnphi = -res.gZ + self.repeat(self.bi, axis=0, times=res.Np) * (res.Z - 1) / res.bm + res.gphi * res.fZ / (self.delta2 - self.delta1)
        res.lnf = res.lnphi + np.log(y * P)
        res.f = np.exp(res.lnf)
        return res
    pass


class eos_sw(eos_srk_pr):
    def __init__(self, Pc, Tc, w, comp_type, **kwargs):
        super(eos_sw, self).__init__(Pc, Tc, w, comp_type, c=1, **kwargs)
        self.sw = True
        pass

    def calc_alphai(self, T, cw):
        Tcw = self.Tc[np.where(self.comp_type == 2)][0]
        alphai = self.ai * (1 + self.kappa * (1 - (T / self.Tc)**0.5))**2
        for i in range(self.Nc):
            if self.comp_type[i] == 2:
                alphai[i] = self.ai[i] * (1 + 0.4530 * (1 - T * (1 - 0.0103 * cw**1.1) / Tcw) + 0.0034 * ((T / Tcw)**(-3) - 1))**2
        return alphai

    def calc_bips(self, phases, T, cw, res):
        dij = np.empty(shape=(res.Np, self.Nc, self.Nc))
        if self.gcm:
            dij[:] = self.calc_bips_gcm(T, res.alphai)
        else:
            dij[:] = self.dij
        for k, phase in enumerate(phases):
            if phase == 'w':
                for i in range(self.Nc):
                    for j in range(i, self.Nc):
                        if i != j:
                            if (self.comp_type[i] == 2 or self.comp_type[j] == 2) and (self.comp_type[i] != 0 and self.comp_type[j] != 0):
                                dij[k][i][j] = self.calc_bip_sw(i, j, T, cw, res)
                                dij[k][j][i] = dij[k][i][j]
        return dij

    def calc_bip_sw(self, i, j, T, cw, res):
        if self.comp_type[i] == 2:
            water = i
            non_water = j
        else:
            water = j
            non_water = i
        if self.comp_type[non_water] == 1:
            A0 = 1.1120 - 1.7369 * self.w[non_water]**(-0.1)
            A1 = 1.001 + 0.8360 * self.w[non_water]
            A2 = -0.15742 - 1.0988 * self.w[non_water]
            a0 = 0.017407
            a1 = 0.033516
            a2 = 0.011478
            return A0 * (1 + a0 * cw) + A1 * T * (1 + a1 * cw) / self.Tc[non_water] + A2 * (1 + a2 * cw) * (T / self.Tc[non_water])**2
        elif self.comp_type[non_water] == 3:
            return -0.31092 * (1 + 0.15587 * cw**0.7505) + 0.23580 * (1 + 0.17837 * cw**0.979) * T / self.Tc[non_water] - 21.2566 * np.exp(-6.7222 * T / self.Tc[non_water] - cw)
        elif self.comp_type[non_water] == 4:
            return -0.20441 + 0.23426 * T / self.Tc[non_water]
        elif self.comp_type[non_water] == 5:
            return -1.70235 * (1 + 0.025587 * cw**0.75) + 0.44338 * (1 + 0.08126 * cw**0.75) * T / self.Tc[non_water]

    def eos_run(self, y, P, T, phases, cw, Zl=False, Zv=False):
        res = self.empty_object()
        res.Np = len(phases)
        res.y = y
        res.P = P
        res.T = T
        res.cw = cw
        res.phases = phases
        res.alphai = self.calc_alphai(T, cw)
        res.dij = self.calc_bips(phases, T, cw, res)
        res.alphaij = self.repeat(np.outer(res.alphai, res.alphai)**0.5, axis=0, times=res.Np) * (1 - res.dij)
        res.alpham = np.sum(np.einsum('yi,yj->yij', y, y) * res.alphaij, axis=(1, 2)).reshape(res.Np, 1)
        res.bm = np.sum(y * self.bi, axis=1).reshape(res.Np, 1)
        res.A = res.alpham * P / (self.R**2 * T**2)
        res.B = res.bm * P / (self.R * T)
        res.Z = self.calc_Z(res.A, res.B, res.Np, Zl, Zv)
        res.v = res.Z * self.R * T / P
        res.gZ = np.log(res.Z - res.B)
        res.gphi = (2 * np.sum(res.alphaij * self.repeat(y, axis=1, times=self.Nc), axis=2) - res.alpham * self.repeat(self.bi, axis=0, times=res.Np) / res.bm) / (self.R * T * res.bm)
        res.fZ = np.log((res.Z + res.B * self.delta1) / (res.Z + res.B * self.delta2))
        res.lnphi = -res.gZ + self.repeat(self.bi, axis=0, times=res.Np) * (res.Z - 1) / res.bm + res.gphi * res.fZ / (self.delta2 - self.delta1)
        res.lnf = res.lnphi + np.log(y * P)
        res.f = np.exp(res.lnf)
        return res
    pass


class derivatives_eos_2param(core):
    def __init__(self, mr, eos, n, der_T=False, der_nk=False, der_P=False, der_nl=False, der_ni=False, second=False):
        Nc, Np, y, P, T = mr.Nc, eos.Np, eos.y, eos.P, eos.T
        ddm_dA = np.append(self.repeat(np.array([0.0, 1.0]), axis=0, times=Np), -eos.B, axis=1)
        ddm_dB = np.append(np.append(self.repeat(np.array([mr.c]), axis=0, times=Np), -(mr.c + 1) - 2 * (2 * mr.c + 1) * eos.B, axis=1), -eos.A + mr.c * (2 * eos.B + 3 * eos.B**2), axis=1)
        d2dm_dA2 = np.zeros_like(ddm_dA)
        d2dm_dB2 = np.append(self.repeat(np.array([0.0, -2 * (2 * mr.c + 1)]), axis=0, times=Np), 2 * mr.c * (1 + 3 * eos.B), axis=1)
        d2dm_dAdB = self.repeat(np.array([0.0, 0.0, -1.0]), axis=0, times=Np)
        m = np.arange(2, -1, -1)
        dq_dZ = 3 * eos.Z**2 + 2 * (mr.c * eos.B - 1) * eos.Z + (eos.A - (mr.c + 1) * eos.B - (2 * mr.c + 1) * eos.B**2)
        d2q_dZ2 = 6 * eos.Z + 2 * (mr.c * eos.B - 1)
        dgZ_dZ = 1 / (eos.Z - eos.B)
        dgZ_dB = -1 / (eos.Z - eos.B)
        d2gZ_dZ2 = -dgZ_dZ**2
        d2gZ_dB2 = d2gZ_dZ2
        d2gZ_dZdB = -d2gZ_dB2
        dfZ_dZ = 1 / (eos.Z + eos.B * mr.delta1) - 1 / (eos.Z + eos.B * mr.delta2)
        dfZ_dB = mr.delta1 / (eos.Z + eos.B * mr.delta1) - mr.delta2 / (eos.Z + eos.B * mr.delta2)
        d2fZ_dZ2 = -1 / (eos.Z + eos.B * mr.delta1)**2 + 1 / (eos.Z + eos.B * mr.delta2)**2
        d2fZ_dZdB = -mr.delta1 / (eos.Z + eos.B * mr.delta1)**2 + mr.delta2 / (eos.Z + eos.B * mr.delta2)**2
        d2fZ_dB2 = -(mr.delta1 / (eos.Z + eos.B * mr.delta1))**2 + (mr.delta2 / (eos.Z + eos.B * mr.delta2))**2
        if der_T:
            dalphai_dT = -np.sqrt(mr.ai * eos.alphai / (T * mr.Tc)) * mr.kappa
            if mr.sw:
                dalphai_dT = np.where(mr.comp_type == 2, 2 * np.sqrt(mr.ai * eos.alphai) * (-0.4530 * (1 - 0.0103 * eos.cw**1.1) / mr.Tc - 0.0102 * mr.Tc**3 / T**4), dalphai_dT)
            if mr.gcm:
                dDS_dT = self.dDS_dT_calc(mr, T)
                ddij_dT = self.ddij_dT_calc(mr, eos, dDS_dT, dalphai_dT)
            else:
                ddij_dT = 0
            dalphaij_dT = (1 - eos.dij) * (np.outer(eos.alphai, dalphai_dT) + np.outer(dalphai_dT, eos.alphai)) / (2 * np.sqrt(np.outer(eos.alphai, eos.alphai))) - \
                          np.sqrt(np.outer(eos.alphai, eos.alphai)) * ddij_dT
            dalpham_dT = np.sum(np.einsum('yi,yj->yij', y, y) * dalphaij_dT, axis=(1, 2)).reshape(Np, 1)
            dA_dT = P * dalpham_dT / (self.R**2 * T**2) - 2 * eos.A / T
            dB_dT = -eos.bm * P / (self.R * T**2)
            ddm_dT = ddm_dA * dA_dT + ddm_dB * dB_dT
            dq_dT = np.sum(ddm_dT * eos.Z**m, axis=1)
            dZ_dT = - dq_dT.reshape((Np, 1)) * dq_dZ**(-1)
            dgZ_dT = dgZ_dZ * dZ_dT + dgZ_dB * dB_dT
            dgphi_dT = ((2 * np.sum(self.repeat(y, axis=2, times=Nc) * dalphaij_dT, axis=1) - self.repeat(mr.bi, axis=0, times=Np) * dalpham_dT / eos.bm) / (self.R * eos.bm) - eos.gphi) / T
            dfZ_dT = dfZ_dZ * dZ_dT + dfZ_dB * dB_dT
            dlnphi_dT = -dgZ_dT + self.repeat(mr.bi, axis=0, times=Np) * dZ_dT / eos.bm + (dgphi_dT * eos.fZ + dfZ_dT * eos.gphi) / (mr.delta2 - mr.delta1)
            dlnf_dT = dlnphi_dT
        # if der_nk:
        if der_nk or der_ni:
            dalpham_dnk = 2 * (np.sum(eos.alphaij * self.repeat(y, axis=1, times=Nc), axis=2) - eos.alpham) / n
            dbm_dnk = (self.repeat(mr.bi, 0, Np) - eos.bm) / n
            dA_dnk = P * dalpham_dnk / (self.R**2 * T**2)
            dB_dnk = P * dbm_dnk / (self.R * T)
            ddm_dnk = self.repeat(ddm_dA, axis=1, times=Nc) * self.repeat(dA_dnk, axis=2, times=3) + self.repeat(ddm_dB, axis=1, times=Nc) * self.repeat(dB_dnk, axis=2, times=3)
            dq_dnk = np.sum(ddm_dnk * self.repeat(eos.Z**m, axis=1, times=Nc), axis=2)
            dZ_dnk = - dq_dnk * dq_dZ**(-1)
            dgZ_dnk = dgZ_dZ * dZ_dnk + dgZ_dB * dB_dnk
            dfZ_dnk = dfZ_dZ * dZ_dnk + dfZ_dB * dB_dnk
            dgphi_dnk = (2 * (eos.alphaij - self.repeat(np.sum(self.repeat(y, 2, Nc) * eos.alphaij, axis=1), 1, Nc)) / self.repeat(n, 2, 1) - \
                         self.R * T * self.repeat(eos.gphi, 1, Nc) * self.repeat(dbm_dnk, 2, Nc) - self.repeat(self.repeat(mr.bi, 0, Np), 1, Nc) * \
                         self.repeat(dalpham_dnk - dbm_dnk * eos.alpham / eos.bm, 2, Nc) / self.repeat(eos.bm, 2, 1)) / self.repeat(self.R * T * eos.bm, 2, 1)
            dlnphi_dnk = self.repeat(-dgZ_dnk, 2, Nc) + self.repeat(self.repeat(mr.bi, 0, Np), 1, Nc) * self.repeat(dZ_dnk - (eos.Z - 1) * dbm_dnk / eos.bm, 2, Nc) / self.repeat(eos.bm, 2, 1) + \
                         (self.repeat(eos.fZ, 2, 1) * dgphi_dnk + self.repeat(eos.gphi, 1, Nc) * self.repeat(dfZ_dnk, 2, Nc)) / (mr.delta2 - mr.delta1)
            dlnf_dnk = dlnphi_dnk + self.repeat(np.identity(Nc), 0, Np) / self.repeat(n * y, 1, Nc) - 1 / self.repeat(n, 2, 1)
        if der_ni:
            dlnphi_dni = np.diagonal(dlnphi_dnk, axis1=1, axis2=2)
            dlnf_dni = np.diagonal(dlnf_dnk, axis1=1, axis2=2)
        #     dalpham_dni =(np.sum(eos.alphaij * self.repeat(y, 2, Nc), axis=(2, 1)).reshape(Np, 1) + np.sum(np.diagonal(eos.alphaij, axis1=1, axis2=2) * y, axis=1).reshape(Np, 1) - 2 * eos.alpham) / n
        #     dbm_dni = (np.sum(self.repeat(mr.bi, 0, Np), axis=1).reshape(Np, 1) - eos.bm) / n
        #     dA_dni = P * dalpham_dni / (self.R**2 * T**2)
        #     dB_dni = P * dbm_dni / (self.R * T)
        #     ddm_dni = ddm_dA * dA_dni + ddm_dB * dB_dni
        #     dq_dni = np.sum(ddm_dni * eos.Z**m, axis=1).reshape(Np, 1)
        #     dZ_dni = - dq_dni * dq_dZ**(-1)
        #     dgZ_dni = dgZ_dZ * dZ_dni + dgZ_dB * dB_dni
        #     dfZ_dni = dfZ_dZ * dZ_dni + dfZ_dB * dB_dni
        #     dgphi_dni = (2 * (np.diagonal(eos.alphaij, axis1=1, axis2=2) - np.sum(self.repeat(y, 2, Nc) * eos.alphaij, axis=1)) / n - self.R * T * eos.gphi * dbm_dni - \
        #                  self.repeat(mr.bi, 0, Np) * (dalpham_dni - eos.alpham * dbm_dni / eos.bm) / eos.bm) / (self.R * T * eos.bm)
        #     dlnphi_dni = - dgZ_dni + self.repeat(mr.bi, 0, Np) * dZ_dni / eos.bm - (eos.Z - 1) * self.repeat(mr.bi, 0, Np) * dbm_dni / eos.bm**2 + (eos.fZ * dgphi_dni + eos.gphi * dfZ_dni) \
        #                  / (mr.delta2 - mr.delta1)
        #     dlnf_dni = dlnphi_dni + 1 / (y * n) - 1 / n
        if der_P:
            dA_dP = eos.alpham / (self.R**2 * T**2)
            dB_dP = eos.bm / (self.R * T)
            ddm_dP = ddm_dA * dA_dP + ddm_dB * dB_dP
            dq_dP = np.sum(ddm_dP * eos.Z**m, axis=1)
            dZ_dP = - dq_dP.reshape((Np, 1)) * dq_dZ**(-1)
            dgZ_dP = dgZ_dZ * dZ_dP + dgZ_dB * dB_dP
            dfZ_dP = dfZ_dZ * dZ_dP + dfZ_dB * dB_dP
            dlnphi_dP = -dgZ_dP + self.repeat(mr.bi, axis=0, times=Np) * dZ_dP / eos.bm + eos.gphi * dfZ_dP / (mr.delta2 - mr.delta1)
            dlnf_dP = dlnphi_dP + 1 / P
        if der_T and second:
            d2alphai_dT2 = - mr.kappa * np.sqrt(mr.ai * T / (mr.Tc * eos.alphai)) * (dalphai_dT / T - eos.alphai / T**2) / 2
            if mr.sw:
                d2alphai_dT2 = np.where(mr.comp_type == 2, dalphai_dT**2 / (2 * eos.alphai) + 0.0816 * np.sqrt(mr.ai * eos.alphai) * mr.Tc**3 / T**5, d2alphai_dT2)
            if mr.gcm:
                d2DS_dT2 = self.d2DS_dT2_calc(mr, T) - 2 * dDS_dT / T
                d2dij_dT2 = self.d2dij_dT2_calc(mr, eos, ddij_dT, dDS_dT, d2DS_dT2, dalphai_dT, d2alphai_dT2)
            else:
                d2dij_dT2 = 0
            d2alphaij_dT2 = - (self.repeat(eos.alphai, 1) * self.repeat(dalphai_dT, 0) + self.repeat(eos.alphai, 0) * self.repeat(dalphai_dT, 1)) * ddij_dT / np.sqrt(np.outer(eos.alphai, eos.alphai)) + \
                            (1 - eos.dij) * (self.repeat(eos.alphai, 1) * self.repeat(d2alphai_dT2, 0) + self.repeat(eos.alphai, 0) * self.repeat(d2alphai_dT2, 1) + 2 * np.outer(dalphai_dT, dalphai_dT) - \
                                            (self.repeat(eos.alphai, 1) * self.repeat(dalphai_dT, 0) + self.repeat(eos.alphai, 0) * self.repeat(dalphai_dT, 1))**2 / (2 * np.outer(eos.alphai, eos.alphai))) / \
                            (2 * np.sqrt(np.outer(eos.alphai, eos.alphai))) - np.sqrt(np.outer(eos.alphai, eos.alphai)) * d2dij_dT2
            d2alpham_dT2 = np.sum(np.einsum('yi,yj->yij', y, y) * d2alphaij_dT2, axis=(1, 2)).reshape(Np, 1)
            d2A_dT2 = P * (d2alpham_dT2 - dalpham_dT / T) / (self.R**2 * T**2) - 3 * dA_dT / T
            d2B_dT2 = 2 * eos.bm * P / (self.R * T**3)
            d2dm_dT2 = d2dm_dA2 * dA_dT**2 + ddm_dA * d2A_dT2 + d2dm_dB2 * dB_dT**2 + ddm_dB * d2B_dT2 + 2 * d2dm_dAdB * dA_dT * dB_dT
            d2q_dT2 = np.sum(d2dm_dT2 * eos.Z**m, axis=1).reshape((Np, 1))
            d2q_dZdT = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dT[:,:-1] * eos.Z**(m[:-1] - 1), axis=1).reshape((Np, 1))
            d2Z_dT2 = - dq_dZ**(-1) * (d2q_dT2 + 2 * dZ_dT * d2q_dZdT + dZ_dT**2 * d2q_dZ2)
            d2gZ_dT2 = d2gZ_dZ2 * dZ_dT**2 + dgZ_dZ * d2Z_dT2 + d2gZ_dB2 * dB_dT**2 + dgZ_dB * d2B_dT2 + 2 * d2gZ_dZdB * dZ_dT * dB_dT
            d2gphi_dT2 = ((2 * np.sum(self.repeat(y, 2, Nc) * d2alphaij_dT2, 1) - self.repeat(mr.bi, 0, Np) * d2alpham_dT2 / eos.bm) / (self.R * eos.bm) - 2 * dgphi_dT) / T
            d2fZ_dT2 = d2fZ_dZ2 * dZ_dT**2 + dfZ_dZ * d2Z_dT2 + d2fZ_dB2 * dB_dT**2 + dfZ_dB * d2B_dT2 + 2 * d2fZ_dZdB * dZ_dT * dB_dT
            d2lnphi_dT2 = - d2gZ_dT2 + self.repeat(mr.bi, 0, Np) * d2Z_dT2 / eos.bm + (eos.fZ * d2gphi_dT2 + eos.gphi * d2fZ_dT2 + 2 * dgphi_dT * dfZ_dT) / (mr.delta2 - mr.delta1)
            d2lnf_dT2 = d2lnphi_dT2
        if der_P and second:
            d2dm_dP2 = d2dm_dA2 * dA_dP**2 + d2dm_dB2 * dB_dP**2 + 2 * d2dm_dAdB * dA_dP * dB_dP
            d2q_dP2 = np.sum(d2dm_dP2 * eos.Z**m, axis=1)
            d2q_dZdP = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dP[:,:-1] * eos.Z**(m[:-1] - 1), 1)
            d2Z_dP2 = - dq_dZ**(-1) * (d2q_dP2.reshape((Np, 1)) + 2 * dZ_dP * d2q_dZdP.reshape((Np, 1)) + dZ_dP**2 * d2q_dZ2)
            d2gZ_dP2 = d2gZ_dZ2 * dZ_dP**2 + dgZ_dZ * d2Z_dP2 + d2gZ_dB2 * dB_dP**2 + 2 * d2gZ_dZdB * dZ_dP * dB_dP
            d2fZ_dP2 = d2fZ_dZ2 * dZ_dP**2 + dfZ_dZ * d2Z_dP2 + d2fZ_dB2 * dB_dP**2 + 2 * d2fZ_dZdB * dZ_dP * dB_dP
            d2lnphi_dP2 = - d2gZ_dP2 + self.repeat(mr.bi, 0, Np) * d2Z_dP2 / eos.bm + eos.gphi * d2fZ_dP2 / (mr.delta2 - mr.delta1)
            d2lnf_dP2 = d2lnphi_dP2 + P**(-2)
        if der_nk and der_nl:
            d2alpham_dnkdnl = 2 * (eos.alphaij - self.repeat(n, 2, 1) * self.repeat(dalpham_dnk, 1, Nc) - self.repeat(n, 2, 1) * self.repeat(dalpham_dnk, 2, Nc) - self.repeat(eos.alpham, 2, 1)) / \
                              self.repeat(n**2, 2, 1)
            d2bm_dnkdnl = (self.repeat(2 * eos.bm, 2, 1) - self.repeat(self.repeat(mr.bi, 0, Np), 1, Nc) - self.repeat(self.repeat(mr.bi, 0, Np), 2, Nc)) / self.repeat(n**2, 2, 1)
            d2A_dnkdnl = P * d2alpham_dnkdnl / (self.R**2 * T**2)
            d2B_dnkdnl = P * d2bm_dnkdnl / (self.R * T)
            d2dm_dnkdnl = self.repeat(self.repeat(d2dm_dA2, 1, Nc), 2, Nc) * self.repeat(np.einsum('yi,yj->yij', dA_dnk, dA_dnk), 3, 3) + \
                          self.repeat(self.repeat(ddm_dA, 1, Nc), 2, Nc) * self.repeat(d2A_dnkdnl, 3, 3) + \
                          self.repeat(self.repeat(d2dm_dB2, 1, Nc), 2, Nc) * self.repeat(np.einsum('yi,yj->yij', dB_dnk, dB_dnk), 3, 3) + \
                          self.repeat(self.repeat(ddm_dB, 1, Nc), 2, Nc) * self.repeat(d2B_dnkdnl, 3, 3) + \
                          self.repeat(self.repeat(d2dm_dAdB, 1, Nc), 2, Nc) * self.repeat(np.einsum('yi,yj->yij', dA_dnk, dB_dnk) + np.einsum('yi,yj->yij', dB_dnk, dA_dnk), 3, 3)
            d2q_dnkdnl = np.sum(d2dm_dnkdnl * self.repeat(self.repeat(eos.Z**m, 1, Nc), 2, Nc), axis=3)
            d2q_dZdnk = np.sum(self.repeat(self.repeat(m[:-1], 0, Np), 1, Nc) * ddm_dnk[:,:,:-1] * self.repeat(eos.Z**(m[:-1] - 1), 1, Nc), 2)
            d2Z_dnkdnl = self.repeat(-dq_dZ**(-1), 2, 1) * (d2q_dnkdnl + self.repeat(dZ_dnk, 2, Nc) * self.repeat(d2q_dZdnk, 1, Nc) + \
                                                            self.repeat(dZ_dnk, 1, Nc) * self.repeat(d2q_dZdnk, 2, Nc) + np.einsum('yi,yj->yij', dZ_dnk, dZ_dnk) * self.repeat(d2q_dZ2, 2, 1))
            d2gZ_dnkdnl = self.repeat(d2gZ_dZ2, 2, 1) * np.einsum('yi,yj->yij', dZ_dnk, dZ_dnk) + self.repeat(dgZ_dZ, 2, 1) * d2Z_dnkdnl + \
                          self.repeat(d2gZ_dB2, 2, 1) * np.einsum('yi,yj->yij', dB_dnk, dB_dnk) + self.repeat(dgZ_dB, 2, 1) * d2B_dnkdnl + \
                          self.repeat(d2gZ_dZdB, 2, 1) * (np.einsum('yi,yj->yij', dZ_dnk, dB_dnk) + np.einsum('yi,yj->yij', dB_dnk, dZ_dnk))
            d2gphi_dnkdnl = (2 * (2 * self.repeat(self.repeat(np.sum(self.repeat(y, 2, Nc) * eos.alphaij, 1), 1, Nc), 2, Nc) - self.repeat(eos.alphaij, 2, Nc) - self.repeat(eos.alphaij, 1, Nc)) / \
                             self.repeat(self.repeat(n**2, 2, 1), 3, 1) - self.R * T * self.repeat(self.repeat(eos.gphi, 1, Nc), 2, Nc) * self.repeat(d2bm_dnkdnl, 3, Nc) - \
                             self.repeat(self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm, 1, Nc), 2, Nc) * \
                             (self.repeat(d2alpham_dnkdnl, 3, Nc) - self.repeat(self.repeat(eos.alpham / eos.bm, 2, 1), 3, 1) * self.repeat(d2bm_dnkdnl, 3, Nc) - 
                              self.repeat((np.einsum('yi,yj->yij', dbm_dnk, dalpham_dnk) + np.einsum('yi,yj->yij', dalpham_dnk, dbm_dnk) - 
                                           2 * self.repeat(eos.alpham / eos.bm, 2, 1) * np.einsum('yi,yj->yij', dbm_dnk, dbm_dnk)) / self.repeat(eos.bm, 2, 1), 3, Nc)) - \
                             self.R * T * (self.repeat(self.repeat(dbm_dnk, 2, Nc), 3, Nc) * self.repeat(dgphi_dnk, 1, Nc) + \
                                         self.repeat(dgphi_dnk, 2, Nc) * self.repeat(self.repeat(dbm_dnk, 1, Nc), 3, Nc))) / \
                            (self.R * T * self.repeat(self.repeat(eos.bm, 2, 1), 3, 1))
            d2fZ_dnkdnl = self.repeat(d2fZ_dZ2, 2, 1) * np.einsum('yi,yj->yij', dZ_dnk, dZ_dnk) + self.repeat(dfZ_dZ, 2, 1) * d2Z_dnkdnl + \
                          self.repeat(d2fZ_dB2, 2, 1) * np.einsum('yi,yj->yij', dB_dnk, dB_dnk) + self.repeat(dfZ_dB, 2, 1) * d2B_dnkdnl + \
                          self.repeat(d2fZ_dZdB, 2, 1) * (np.einsum('yi,yj->yij', dZ_dnk, dB_dnk) + np.einsum('yi,yj->yij', dB_dnk, dZ_dnk))
            d2lnphi_dnkdnl = -self.repeat(d2gZ_dnkdnl, 3, Nc) + self.repeat(self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm, 1, Nc), 2, Nc) * self.repeat(d2Z_dnkdnl, 3, Nc) - \
                             self.repeat(self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm**2, 1, Nc), 2, Nc) * self.repeat(np.einsum('yi,yj->yij', dZ_dnk, dbm_dnk) + \
                                                                                                                         np.einsum('yi,yj->yij', dbm_dnk, dZ_dnk), 3, Nc) + \
                             self.repeat(self.repeat(self.repeat(mr.bi, 0, Np) * (eos.Z - 1) / eos.bm**2, 1, Nc), 2, Nc) * \
                             self.repeat(2 * np.einsum('yi,yj->yij', dbm_dnk, dbm_dnk) / self.repeat(eos.bm, 2, 1) - d2bm_dnkdnl, 3, Nc) + \
                             (self.repeat(self.repeat(eos.fZ, 2, 1), 3, 1) * d2gphi_dnkdnl + self.repeat(self.repeat(eos.gphi, 1, Nc), 2, Nc) * self.repeat(d2fZ_dnkdnl, 3, Nc) + \
                              self.repeat(dgphi_dnk, 2, Nc) * self.repeat(self.repeat(dfZ_dnk, 1, Nc), 3, Nc) + \
                              self.repeat(dgphi_dnk, 1, Nc) * self.repeat(self.repeat(dfZ_dnk, 2, Nc), 3, Nc)) / (mr.delta2 - mr.delta1)
            d2lnf_dnkdnl = d2lnphi_dnkdnl - self.repeat(np.identity(Nc), 2, Nc) * self.repeat(np.identity(Nc), 1, Nc) / self.repeat(self.repeat(y**2 * n**2, 1, Nc), 2, Nc) + \
                           1 / self.repeat(self.repeat(n**2, 2, 1), 3, 1)
        if der_nk and der_T:
            d2alpham_dnkdT = 2 * (np.sum(dalphaij_dT * self.repeat(y, axis=1, times=Nc), axis=2) - dalpham_dT) / n
            d2A_dnkdT = P * d2alpham_dnkdT / (self.R**2 * T**2) - 2 * dA_dnk / T
            d2B_dnkdT = - dbm_dnk * P / (self.R * T**2)
            d2dm_dnkdT = self.repeat(d2dm_dA2, 1, Nc) * self.repeat(dA_dnk, 2, 3) * self.repeat(dA_dT, 2, 1) + self.repeat(ddm_dA, 1, Nc) * self.repeat(d2A_dnkdT, 2, 3) + \
                         self.repeat(d2dm_dB2, 1, Nc) * self.repeat(dB_dnk, 2, 3) * self.repeat(dB_dT, 2, 1) + self.repeat(ddm_dB, 1, Nc) * self.repeat(d2B_dnkdT, 2, 3) + \
                         self.repeat(d2dm_dAdB, 1, Nc) * self.repeat(dA_dnk * dB_dT + dA_dT * dB_dnk, 2, 3)
            d2q_dnkdT = np.sum(d2dm_dnkdT * self.repeat(eos.Z**m, 1, Nc), axis=2)
            d2q_dZdT = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dT[:,:-1] * eos.Z**(m[:-1] - 1), axis=1).reshape((Np, 1))
            d2q_dZdnk = np.sum(self.repeat(self.repeat(m[:-1], 0, Np), 1, Nc) * ddm_dnk[:,:,:-1] * self.repeat(eos.Z**(m[:-1] - 1), 1, Nc), 2)
            d2Z_dnkdT = -dq_dZ**(-1) * (d2q_dnkdT + dZ_dnk * d2q_dZdT + dZ_dT * d2q_dZdnk + dZ_dnk * dZ_dT * d2q_dZ2)
            d2gZ_dnkdT = d2gZ_dZ2 * dZ_dT * dZ_dnk + dgZ_dZ * d2Z_dnkdT  + d2gZ_dB2 * dB_dT * dB_dnk + dgZ_dB * d2B_dnkdT + d2gZ_dZdB * (dZ_dT * dB_dnk + dB_dT * dZ_dnk)
            d2gphi_dnkdT = (2 * (dalphaij_dT - self.repeat(np.sum(dalphaij_dT * self.repeat(y, 2, Nc), 1), 1, Nc)) / self.repeat(n, 2, 1) - \
                            self.repeat((d2alpham_dnkdT - dalpham_dT * dbm_dnk / eos.bm), 2, Nc) * self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm, 1, Nc)) / self.repeat(self.R * T * eos.bm, 2, 1) - \
                           self.repeat(dbm_dnk, 2, Nc) * self.repeat(dgphi_dT + eos.gphi / T, 1, Nc) / self.repeat(eos.bm, 2, 1) - dgphi_dnk / T
            d2fZ_dnkdT = d2fZ_dZ2 * dZ_dT * dZ_dnk + dfZ_dZ * d2Z_dnkdT  + d2fZ_dB2 * dB_dT * dB_dnk + dfZ_dB * d2B_dnkdT + d2fZ_dZdB * (dZ_dT * dB_dnk + dB_dT * dZ_dnk)
            d2lnphi_dnkdT = self.repeat(-d2gZ_dnkdT, 2, Nc) + self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm, 1, Nc) * self.repeat(d2Z_dnkdT, 2, Nc) - \
                            self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm**2, 1, Nc) * self.repeat(dbm_dnk, 2, Nc) * self.repeat(dZ_dT, 2, 1) + \
                            (d2gphi_dnkdT * self.repeat(eos.fZ, 2, 1) + self.repeat(eos.gphi, 1, Nc) * self.repeat(d2fZ_dnkdT, 2, Nc) + \
                             dgphi_dnk * self.repeat(dfZ_dT, 2, 1) + self.repeat(dgphi_dT, 1, Nc) * self.repeat(dfZ_dnk, 2, Nc)) / (mr.delta2 - mr.delta1)
            d2lnf_dnkdT = d2lnphi_dnkdT
        if der_ni and der_T:
            d2alpham_dnidT = (np.sum(dalphaij_dT * self.repeat(y, 2, Nc), axis=(2, 1)).reshape(Np, 1) + np.sum(np.diagonal(dalphaij_dT, axis1=1, axis2=2) * y, axis=1).reshape(Np, 1) - 2 * dalpham_dT) / n
            d2A_dnidT = P * d2alpham_dnidT / (self.R**2 * T**2) - 2 * dA_dni / T
            d2B_dnidT = - dbm_dni * P / (self.R * T**2)
            d2dm_dnidT = d2dm_dA2 * dA_dni * dA_dT + ddm_dA * d2A_dnidT + d2dm_dB2 * dB_dni * dB_dT + ddm_dB * d2B_dnidT + d2dm_dAdB * (dA_dni * dB_dT + dA_dT * dB_dni)
            d2q_dnidT = np.sum(d2dm_dnidT * eos.Z**m, axis=1).reshape(Np, 1)
            d2q_dZdT = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dT[:,:-1] * eos.Z**(m[:-1] - 1), axis=1).reshape(Np, 1)
            d2q_dZdni = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dni[:,:-1] * eos.Z**(m[:-1] - 1), axis=1).reshape(Np, 1)
            d2Z_dnidT = -dq_dZ**(-1) * (d2q_dnidT + dZ_dni * d2q_dZdT + dZ_dT * d2q_dZdni + dZ_dni * dZ_dT * d2q_dZ2)
            d2gZ_dnidT = d2gZ_dZ2 * dZ_dT * dZ_dni + dgZ_dZ * d2Z_dnidT  + d2gZ_dB2 * dB_dT * dB_dni + dgZ_dB * d2B_dnidT + d2gZ_dZdB * (dZ_dT * dB_dni + dB_dT * dZ_dni)
            d2fZ_dnidT = d2fZ_dZ2 * dZ_dT * dZ_dni + dfZ_dZ * d2Z_dnidT  + d2fZ_dB2 * dB_dT * dB_dni + dfZ_dB * d2B_dnidT + d2fZ_dZdB * (dZ_dT * dB_dni + dB_dT * dZ_dni)
            d2gphi_dnidT = (2 * (np.diagonal(dalphaij_dT, axis1=1, axis2=2) - np.sum(dalphaij_dT * self.repeat(y, 2, Nc), 1)) / n - \
                            (d2alpham_dnidT - dalpham_dT * dbm_dni / eos.bm) * self.repeat(mr.bi, 0, Np) / eos.bm) / (self.R * T * eos.bm) - dbm_dni * (dgphi_dT + eos.gphi / T) / eos.bm - \
                           dgphi_dni / T
            d2lnphi_dnidT = -d2gZ_dnidT + self.repeat(mr.bi, 0, Np) * d2Z_dnidT / eos.bm - self.repeat(mr.bi, 0, Np) * dZ_dT * dbm_dni / eos.bm**2 + \
                            (eos.fZ * d2gphi_dnidT + eos.gphi * d2fZ_dnidT + dgphi_dni * dfZ_dT + dgphi_dT * dfZ_dni) / (mr.delta2 - mr.delta1)
            d2lnf_dnidT = d2lnphi_dnidT
        if der_nk and der_P:
            d2A_dnkdP = dalpham_dnk / (self.R**2 * T**2)
            d2B_dnkdP = dbm_dnk / (self.R * T)
            d2dm_dnkdP = self.repeat(d2dm_dA2, 1, Nc) * self.repeat(dA_dnk, 2, 3) * self.repeat(dA_dP, 2, 1) + self.repeat(ddm_dA, 1, Nc) * self.repeat(d2A_dnkdP, 2, 3) + \
                         self.repeat(d2dm_dB2, 1, Nc) * self.repeat(dB_dnk, 2, 3) * self.repeat(dB_dP, 2, 1) + self.repeat(ddm_dB, 1, Nc) * self.repeat(d2B_dnkdP, 2, 3) + \
                         self.repeat(d2dm_dAdB, 1, Nc) * self.repeat(dA_dnk * dB_dP + dA_dP * dB_dnk, 2, 3)
            d2q_dnkdP = np.sum(d2dm_dnkdP * self.repeat(eos.Z**m, 1, Nc), axis=2)
            d2q_dZdP = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dP[:,:-1] * eos.Z**(m[:-1] - 1), 1)
            d2q_dZdnk = np.sum(self.repeat(self.repeat(m[:-1], 0, Np), 1, Nc) * ddm_dnk[:,:,:-1] * self.repeat(eos.Z**(m[:-1] - 1), 1, Nc), 2)
            d2Z_dnkdP = -dq_dZ**(-1) * (d2q_dnkdP + dZ_dnk * d2q_dZdP.reshape((Np, 1)) + dZ_dP * d2q_dZdnk + dZ_dnk * dZ_dP * d2q_dZ2)
            d2gZ_dnkdP = d2gZ_dZ2 * dZ_dP * dZ_dnk + dgZ_dZ * d2Z_dnkdP  + d2gZ_dB2 * dB_dP * dB_dnk + dgZ_dB * d2B_dnkdP + d2gZ_dZdB * (dZ_dP * dB_dnk + dB_dP * dZ_dnk)
            d2fZ_dnkdP = d2fZ_dZ2 * dZ_dP * dZ_dnk + dfZ_dZ * d2Z_dnkdP  + d2fZ_dB2 * dB_dP * dB_dnk + dfZ_dB * d2B_dnkdP + d2fZ_dZdB * (dZ_dP * dB_dnk + dB_dP * dZ_dnk)
            d2lnphi_dnkdP = self.repeat(-d2gZ_dnkdP, 2, Nc) + self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm, 1, Nc) * self.repeat(d2Z_dnkdP, 2, Nc) - \
                            self.repeat(self.repeat(mr.bi, 0, Np) / eos.bm**2, 1, Nc) * self.repeat(dbm_dnk, 2, Nc) * self.repeat(dZ_dP, 2, 1) + \
                            (self.repeat(eos.gphi, 1, Nc) * self.repeat(d2fZ_dnkdP, 2, Nc) + dgphi_dnk * self.repeat(dfZ_dP, 2, 1)) / (mr.delta2 - mr.delta1)
            d2lnf_dnkdP = d2lnphi_dnkdP
        if der_P and der_T:
            d2A_dPdT = dalpham_dT / (self.R**2 * T**2) - 2 * dA_dP / T
            d2B_dPdT = - eos.bm / (self.R * T**2)
            d2dm_dPdT = d2dm_dA2 * dA_dP * dA_dT + ddm_dA * d2A_dPdT + d2dm_dB2 * dB_dP * dB_dT + ddm_dB * d2B_dPdT + d2dm_dAdB * (dA_dP * dB_dT + dA_dT * dB_dP)
            d2q_dPdT = np.sum(d2dm_dPdT * eos.Z**m, axis=1)
            d2q_dZdT = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dT[:,:-1] * eos.Z**(m[:-1] - 1), 1)
            d2q_dZdP = np.sum(self.repeat(m[:-1], 0, Np) * ddm_dP[:,:-1] * eos.Z**(m[:-1] - 1), 1)
            d2Z_dPdT = -dq_dZ**(-1) * (d2q_dPdT.reshape((Np, 1)) + dZ_dP * d2q_dZdT.reshape((Np, 1)) + dZ_dT * d2q_dZdP.reshape((Np, 1)) + dZ_dP * dZ_dT * d2q_dZ2)
            d2gZ_dPdT = d2gZ_dZ2 * dZ_dP * dZ_dT + dgZ_dZ * d2Z_dPdT  + d2gZ_dB2 * dB_dP * dB_dT + dgZ_dB * d2B_dPdT + d2gZ_dZdB * (dZ_dP * dB_dT + dB_dP * dZ_dT)
            d2fZ_dPdT = d2fZ_dZ2 * dZ_dP * dZ_dT + dfZ_dZ * d2Z_dPdT  + d2fZ_dB2 * dB_dP * dB_dT + dfZ_dB * d2B_dPdT + d2fZ_dZdB * (dZ_dP * dB_dT + dB_dP * dZ_dT)
            d2lnphi_dPdT = -d2gZ_dPdT + self.repeat(mr.bi, 0, Np) * d2Z_dPdT / eos.bm + (eos.gphi * d2fZ_dPdT + dgphi_dT * dfZ_dP) / (mr.delta2 - mr.delta1)
            d2lnf_dPdT = d2lnphi_dPdT
        self.derivatives = locals()
        pass

    def get(self, *args):
        res = self.empty_object()
        for arg in args:
            setattr(res, arg, self.derivatives[arg])
        return res

    @staticmethod
    def dDS_dT_calc(mr, T):
        m = np.zeros(shape=(mr.Nc, mr.Nc))
        for i in range(mr.Nc):
            for j in range(mr.Nc):
                if i != j:
                    if m[i][j] == 0:
                        m[i][j] = 149.075 * np.sum(np.outer(mr.alpha_matrix[i] - mr.alpha_matrix[j], mr.alpha_matrix[i] - mr.alpha_matrix[j]) * (mr.Bkl - mr.Akl) * (298.15 / T) ** (mr.DS_pow - 1)) / \
                                  T**2
                        m[j][i] = m[i][j]
        return m

    @staticmethod
    def d2DS_dT2_calc(mr, T):
        m = np.zeros(shape=(mr.Nc, mr.Nc))
        for i in range(mr.Nc):
            for j in range(mr.Nc):
                if i != j:
                    if m[i][j] == 0:
                        m[i][j] = - 298.15**2 * np.sum(np.outer(mr.alpha_matrix[i] - mr.alpha_matrix[j], mr.alpha_matrix[i] - mr.alpha_matrix[j]) * (mr.Bkl - mr.Akl) * (mr.DS_pow - 1) * \
                                                       (298.15 / T) ** (mr.DS_pow - 2)) / (2 * T**4)
                        m[j][i] = m[i][j]
        return m

    def ddij_dT_calc(self, mr, eos, dDS_dT, dalphai_dT):
        alphai_alphaj = np.outer(eos.alphai, eos.alphai)
        sqrt_alphai = np.sqrt(eos.alphai)
        res = (np.outer(mr.bi, mr.bi) / np.sqrt(alphai_alphaj) * (dDS_dT - (self.repeat(sqrt_alphai / mr.bi, 0) - self.repeat(sqrt_alphai / mr.bi, 1)) * \
                                                                  (self.repeat(dalphai_dT / (mr.bi * sqrt_alphai), 0) - self.repeat(dalphai_dT / (mr.bi * sqrt_alphai), 1))) - 
               eos.dij * (np.outer(eos.alphai, dalphai_dT) + np.outer(dalphai_dT, eos.alphai)) / alphai_alphaj) / 2
        if mr.sw:
            for k, phase in enumerate(eos.phases):
                if phase == 'w':
                    for i in range(mr.Nc):
                        for j in range(mr.Nc):
                            if i != j:
                                if mr.comp_type[i] == 2 or mr.comp_type[j] == 2:
                                    if mr.comp_type[i] == 2:
                                        water = i
                                        non_water = j
                                    else:
                                        water = j
                                        non_water = i
                                    if mr.comp_type[non_water] == 1:
                                        A1 = 1.001 + 0.8360 * mr.w[non_water]
                                        A2 = -0.15742 - 1.0988 * mr.w[non_water]
                                        a1 = 0.033516
                                        a2 = 0.011478
                                        res[k][j][i] = A1 * (1 + a1 * eos.cw) / mr.Tc[non_water] + 2 * A2 * eos.T * (1 + a2 * eos.cw) / mr.Tc[non_water]**2
                                    if mr.comp_type[non_water] == 3:
                                        res[k][j][i] = (0.23580 * (1 + 0.17837 * eos.cw**0.979) + 142.8911 * np.exp(-6.7222 * eos.T / mr.Tc[non_water] - eos.cw)) / mr.Tc[non_water]
                                    if mr.comp_type[non_water] == 4:
                                        res[k][j][i] = 0.23426 / mr.Tc[non_water]
                                    if mr.comp_type[non_water] == 5:
                                         res[k][j][i] = 0.44338 * (1 + 0.08126 * eos.cw**0.75) / mr.Tc[non_water]
        return res

    def d2dij_dT2_calc(self, mr, eos, ddij_dT, dDS_dT, d2DS_dT2, dalphai_dT, d2alphai_dT2):
        sqrt_alphai = np.sqrt(eos.alphai)
        alphai_alphaj = np.outer(eos.alphai, eos.alphai)
        bi_bj = np.outer(mr.bi, mr.bi)
        sqrt_alphai_bi = sqrt_alphai / mr.bi
        da_bi_sqrt = dalphai_dT / (mr.bi * sqrt_alphai)
        d2a_bi_sqrt = (d2alphai_dT2 - dalphai_dT**2 / (2 * eos.alphai)) / (mr.bi * sqrt_alphai)
        a1 = self.repeat(sqrt_alphai_bi, 0) - self.repeat(sqrt_alphai_bi, 1)
        a2 = self.repeat(da_bi_sqrt, 0) - self.repeat(da_bi_sqrt, 1)
        a3 = self.repeat(eos.alphai, 1) * self.repeat(dalphai_dT, 0) + self.repeat(eos.alphai, 0) * self.repeat(dalphai_dT, 1)
        a4 = self.repeat(d2a_bi_sqrt, 0) - self.repeat(d2a_bi_sqrt, 1)
        a5 = self.repeat(eos.alphai, 1) * self.repeat(d2alphai_dT2, 0) + self.repeat(eos.alphai, 0) * self.repeat(d2alphai_dT2, 1) + 2 * np.outer(dalphai_dT, dalphai_dT)
        res = (dDS_dT - a1 * a2) * (bi_bj / alphai_alphaj**(3/2)) * a3 / (-4) + \
              bi_bj * (d2DS_dT2 - a2**2 / 2 - a1 * a4) / (2 * np.sqrt(alphai_alphaj)) - \
              (a3 * ddij_dT + eos.dij * a5) / (2 * alphai_alphaj) + \
              eos.dij * a3**2 / (2 * alphai_alphaj**2)
        if mr.sw:
            for k, phase in enumerate(eos.phases):
                if phase == 'w':
                    for i in range(mr.Nc):
                        for j in range(mr.Nc):
                            if i != j:
                                if mr.comp_type[i] == 2 or mr.comp_type[j] == 2:
                                    if mr.comp_type[i] == 2:
                                        water = i
                                        non_water = j
                                    else:
                                        water = j
                                        non_water = i
                                    if mr.comp_type[non_water] == 1:
                                        A2 = -0.15742 - 1.0988 * mr.w[non_water]
                                        a2 = 0.011478
                                        res[k][j][i] = 2 * A2 * (1 + a2 * eos.cw) / mr.Tc[non_water]**2
                                    if mr.comp_type[non_water] == 3:
                                        res[k][j][i] = - 960.5426 * np.exp(-6.7222 * eos.T / mr.Tc[non_water] - eos.cw) / mr.Tc[non_water]**2
                                    if mr.comp_type[non_water] == 4 or mr.comp_type[non_water] == 5:
                                        res[k][j][i] = 0.0
        return res


class parameters_2param(core):
    def __init__(self, cp_matrix, cp_ig_pows=None):
        self.cp_matrix = cp_matrix
        if cp_ig_pows is None:
            self.cp_ig_pows = self.repeat(np.arange(0, cp_matrix.shape[0], 1), 1, cp_matrix.shape[1])
        else:
            self.cp_ig_pows = cp_ig_pows
        pass

    def heat_capacity_ig(self, eos, nj):
        return np.sum(nj * eos.y * self.repeat(np.sum(self.cp_matrix * eos.T ** self.cp_ig_pows, axis=0), 0, eos.Np), axis=1).reshape(eos.Np, 1)

    def enthalpy_ig(self, eos, nj, Tref):
        h_ig_pows = self.cp_ig_pows + 1
        return np.sum(nj * eos.y * self.repeat(np.sum(self.cp_matrix * (eos.T**h_ig_pows - Tref**h_ig_pows) / h_ig_pows, axis=0), 0, eos.Np), axis=1).reshape(eos.Np, 1)

    def enthalpy(self, mr, eos, nj, Tref):
        return self.enthalpy_ig(eos, nj, Tref) - self.R * eos.T**2 * nj * np.sum(eos.y * derivatives_eos_2param(mr, eos, nj, der_T=True).get('dlnphi_dT').dlnphi_dT, axis=1).reshape(eos.Np, 1)

    pass


class derivatives_parameters_2param(parameters_2param):
    def __init__(self):
        pass

    def dH_dT(self, mr, eos, params, n):
        ders = derivatives_eos_2param(mr, eos, n, der_T=True, second=True).get('dlnphi_dT', 'd2lnphi_dT2')
        cpig = params.heat_capacity_ig(eos, n)
        return cpig - self.R * eos.T**2 * n * np.reshape(2 * np.sum(eos.y * ders.dlnphi_dT, axis=1) / eos.T + np.sum(eos.y * ders.d2lnphi_dT2, axis=1), (eos.Np, 1))

    def dH_dni(self):
        return

    def dH_dnk(self, mr, eos, params, n, Tref):
        ders = derivatives_eos_2param(mr, eos, n, der_T=True, der_nk=True).get('dalpham_dT', 'dalpham_dnk', 'dbm_dnk', 'dZ_dnk', 'dfZ_dnk', 'd2alpham_dnkdT')
        hig = params.enthalpy_ig(eos, n, Tref)
        return hig + self.R * eos.T * (eos.Z - 1 + n * ders.dZ_dnk) + \
               eos.fZ * (eos.bm * (eos.alpham - eos.T * ders.dalpham_dT + n * (ders.dalpham_dnk - eos.T * ders.d2alpham_dnkdT)) - n * (eos.alpham - eos.T * ders.dalpham_dT) * ders.dbm_dnk) / \
               ((mr.delta2 - mr.delta1) * eos.bm**2) + ders.dfZ_dnk * (n * (eos.alpham - eos.T * ders.dalpham_dT)) / ((mr.delta2 - mr.delta1) * eos.bm)
    pass


class flash_isothermal_ssi(core):
    def __init__(self, mr, z, ssi_rr_eps=1e-8, ssi_eq_eps=1e-8, ssi_use_opt=False, ssi_negative_flash=False, ssi_eq_max_iter=300,
                 ssi_eps_r=0.6, ssi_eps_v=1e-2, ssi_eps_l=1e-5, ssi_eps_u=1e-3, ssi_switch=False, full_output=True, ssi_rr_newton=True):
        self.z = z
        self.mr = mr
        self.ssi_rr_eps = ssi_rr_eps
        self.ssi_eq_eps = ssi_eq_eps
        self.ssi_use_opt = ssi_use_opt
        self.ssi_negative_flash = ssi_negative_flash
        self.ssi_eq_max_iter = ssi_eq_max_iter
        self.ssi_eps_r = ssi_eps_r
        self.ssi_eps_v = ssi_eps_v
        self.ssi_eps_u = ssi_eps_u
        self.ssi_eps_l = ssi_eps_l
        self.ssi_switch = ssi_switch
        self.full_output = full_output
        self.ssi_rr_newton = ssi_rr_newton
        pass

    # def calc_kv_init(self, P, T, phases, level=0, kv_go=None, kv_gw=None, y=None):
    def calc_kv_init(self, P, T, phases, level=0, kv_go=None, kv_gw=None):
        # kv = []
        # if y_test is None:
        #     y_test = self.z
        # kv_go = self.mr.Pc * np.exp(5.3727 * (1 + self.mr.w) * (1 - self.mr.Tc / T)) / P
        # kv_gw = np.where(self.mr.comp_type == 2, 0.1, 10**6 * self.mr.Pc * T / (P * self.mr.Tc))
        # kv_go_dct = {0: kv_go,
        #              1: 1 / kv_go,
        #              2: kv_go**(1/3),
        #              3: kv_go**(-1/3),
        #              4: np.array([0.98, 0.01, 0.01])}
        # kv_go_double = lambda i: self.replace(0.1 / (y_test * (y_test.shape[0] - 1)), i, 0.9 / y_test[i])
        # kv_dct = {'go': kv_go_dct[go_type], 'gw': kv_gw}
        # i = -1
        # double = False
        # for phase in phases[:-1]:
        #     pp = phase + phases[-1]
        #     if 'o' in pp:
        #         if double:
        #             kv_dct.update({'go': kv_go_double(i)})
        #             i -= 1
        #         kv_dct.update({'ow': np.where(self.mr.comp_type == 2, 0.01, kv_dct['gw'] / kv_dct['go'])})
        #         double = True
        #     if pp in kv_dct:
        #         kv.append(kv_dct[pp])
        #     else:
        #         kv.append(kv_dct[pp[::-1]]**(-1))
        # return np.array(kv)
        if kv_go is None:
            kv_go = self.mr.Pc * np.exp(5.3727 * (1 + self.mr.w) * (1 - self.mr.Tc / T)) / P
        if kv_gw is None:
            kv_gw = np.where(self.mr.comp_type == 2, 0.1, 10**6 * self.mr.Pc * T / (P * self.mr.Tc))
        kv_go_dct = {0: kv_go,
                     1: kv_go**(-1),
                     2: kv_go**(1/3),
                     3: kv_go**(-1/3),
                     4: 10**np.linspace(2, -2, self.mr.Nc)}
        # if y is not None:
        #     kv_go_dct.update({i+5: np.array([y[j] if j == i else 1e+8 for j in range(self.mr.Nc)]) for i in range(self.mr.Nc)})
        kv_gw_dct = {0: kv_gw,
                     1: kv_gw**(-1),
                     2: kv_gw**(1/3),
                     3: kv_gw**(-1/3),
                     4: 10**np.linspace(3, -3, self.mr.Nc)}
        # if y is not None:
        #     kv_gw_dct.update({i+5: np.array([y[j]**2 if j == i else 1e+5 for j in range(self.mr.Nc)]) for i in range(self.mr.Nc)})
        kv_go = kv_go_dct[level]
        kv_gw = kv_gw_dct[level]
        kv_ow = kv_gw / kv_go
        kv_ow = np.where(self.mr.comp_type == 2, 100.0 if np.all(kv_ow < 1) else 0.01, kv_ow)
        if phases == 'go':
            return np.array([kv_go])
        if phases == 'og':
            return np.array([1 / kv_go])
        if phases == 'gw':
            return np.array([kv_gw])
        if phases == 'wg':
            return np.array([1 / kv_gw])
        if phases == 'ow':
            return np.array([kv_ow])
        if phases == 'wo':
            return np.array([1 / kv_ow])
        if phases == 'ogw':
            return np.array([kv_ow, kv_gw])
        if phases == 'gow':
            return np.array([kv_gw, kv_ow])
        if phases == 'gwo':
            return np.array([kv_go, 1 / kv_ow])
        if phases == 'wgo':
            return np.array([1 / kv_ow, kv_go])
        if phases == 'owg':
            return np.array([1 / kv_go, 1 / kv_gw])
        if phases == 'wog':
            return np.array([1 / kv_gw, 1 / kv_go])

    def calc_rr_gradient(self, x0, kv):
        x = x0
        kv = 1 - kv
        Np, Nc = kv.shape
        two_phase = False if Np > 1 else True
        eq = np.ones_like(x0)
        # i = 0
        while np.all(np.abs(eq) > self.ssi_rr_eps):
            ti = 1 - np.sum(x * kv, axis=0)
            eq = np.sum(kv * self.z / ti, axis=1).reshape(Np, 1)
            # print('eq', eq)
            grad = np.sum(kv.reshape(Np, 1, Nc) * kv.reshape(1, Np, Nc) * self.z / ti**2, axis=2)
            # print('grad', grad)
            if two_phase:
                x = x - eq / grad
            else:
                x = x - np.linalg.inv(grad).dot(eq)
            # i += 1
        # print('iters', i)
        return x

    def calc_rr_newton(self, kv, x0=None):
        Np, Nc = kv.shape
        two_phase = False if Np > 1 else True
        bi = np.minimum(1 - self.z, np.amin(1 - kv * self.z, axis=0)).reshape(Nc, 1)
        # print('bi', bi)
        kv = 1 - kv
        aiT = kv.T
        # print('aiT', aiT)
        if x0 is None:
            x = np.linalg.inv(kv.dot(aiT)).dot(kv).dot(bi)
            # print('x0', x)
            x = np.where(x < 0, np.zeros_like(x), x)
            # print('x0 (fix negative)', x)
            # print('condition', aiT.dot(x) <= bi)
        else:
            x = x0
        grad = np.ones_like(x)
        # i = 0
        while np.all(np.abs(grad) > self.ssi_rr_eps):
            ti = 1 - np.sum(x * kv, axis=0)
            grad = np.sum(kv * self.z / ti, axis=1).reshape(Np, 1)
            if np.all(np.abs(grad) < self.ssi_rr_eps):
                return x
            # print('grad', grad)
            hessian = np.sum(kv.reshape(Np, 1, Nc) * kv.reshape(1, Np, Nc) * self.z / ti**2, axis=2)
            # print('hessian', hessian)
            if two_phase:
                d = - grad / hessian
                # print('d', d)
                den = aiT * d
                # print('den', den)
                expr = (bi - aiT * x) / den
                # print('lambda_max array', expr)
                lambda_max = np.min(expr, where=den>0, initial=np.max(expr))
                # print('lambda_max', lambda_max)
                if lambda_max > 1:
                    lambda_max = 1
                s = 1
                grad_s = 1
                while np.abs(grad_s) > 1e-5:
                    ti_s = 1 - np.sum((x + s * lambda_max * d) * kv, axis=0)
                    # print('ti_s', ti_s)
                    grad_s = lambda_max * np.sum(kv * self.z / ti_s, axis=1).reshape(1, Np) * d
                    # print('grad_s', grad_s)
                    grad2_s = lambda_max**2 * d**2 * np.sum(kv.reshape(Np, 1, Nc) * kv.reshape(1, Np, Nc) * self.z / ti_s**2, axis=2)
                    # print('grad2_s', grad2_s)
                    s = s - (grad_s / grad2_s)[0][0]
                    # print('s', s)
                if s > 1:
                    s = 1
            else:
                d = - np.linalg.inv(hessian).dot(grad)
                den = aiT.dot(d)
                expr = (bi - aiT.dot(x)) / den
                # print('lambda_max denominator', den)
                # print('lambda_max array', expr)
                lambda_max = np.min(expr, where=den>0, initial=np.max(expr))
                # print('lambda_max', lambda_max)
                if lambda_max > 1:
                    lambda_max = 1
                s = 1
                grad_s = 1
                while np.abs(grad_s) > 1e-5:
                    ti_s = 1 - np.sum((x + s * lambda_max * d) * kv, axis=0)
                    # print('ti_s', ti_s)
                    grad_s = lambda_max * np.sum(kv * self.z / ti_s, axis=1).reshape(1, Np).dot(d)
                    # print('grad_s', grad_s)
                    grad2_s = lambda_max**2 * d.T.dot(np.sum(kv.reshape(Np, 1, Nc) * kv.reshape(1, Np, Nc) * self.z / ti_s**2, axis=2)).dot(d)
                    # print('grad2_s', grad2_s)
                    s = s - (grad_s / grad2_s)[0][0]
                    # print('s', s)
            x = x + lambda_max * s * d
            # i += 1
            # print('x', x)
        # print('iters', i)
        return x

    def calc_rr_negative_limits(self, kv):
        return 1 / (1 - np.amax(kv, axis=1).reshape(kv.shape[0], 1)), 1 / (1 - np.amin(kv, axis=1).reshape(kv.shape[0], 1))

    def calc_full_F(self, F):
        return np.append(F, 1 - np.sum(F, 0)).reshape(F.shape[0] + 1, 1)

    def calc_y_kv(self, F, kv):
        den = np.sum(F * (kv - 1), axis=0) + 1
        return np.append(self.z * kv / den, [self.z / den], axis=0)

    def check_failed_mole_fractions(self, y):
        return np.isnan(y).any() or (np.abs(np.sum(y, axis=1).reshape(len(y)) - 1) > self.ssi_eq_eps).any()

    def flash_isothermal_ssi_run(self, P, T, phases, kv0=0, **kwargs):
        res = self.empty_object()
        res.isnan = False
        if len(phases) > 1:
            if isinstance(kv0, int):
                kv = self.calc_kv_init(P, T, phases, kv0)
                Np_1 = len(phases) - 1
            else:
                kv = kv0
                Np_1 = len(kv0)
            lambda_pow = 1
            Fmin = np.zeros(shape=(Np_1, 1))
            Fmax = np.ones(shape=(Np_1, 1))
            residuals = np.ones(shape=(Np_1, self.mr.Nc))
            it = 0
            if not self.ssi_rr_newton:
                Fmin, Fmax = self.calc_rr_negative_limits(kv)
                F = (Fmin + Fmax) / 2
            else:
                F = None
            # print('F0', F)
            # print(kv, np.max(np.abs(residuals)))
            while (np.abs(residuals) > self.ssi_eq_eps).any() and it < self.ssi_eq_max_iter:
                # print('kv', kv)
                if self.ssi_rr_newton:
                    F = self.calc_rr_newton(kv, F)
                else:
                    F = self.calc_rr_gradient(F, kv)
                # print('F', F)
                if self.ssi_negative_flash:
                    Fmin, Fmax = self.calc_rr_negative_limits(kv)
                F = np.where(F < Fmin, Fmin, F)
                F = np.where(F > Fmax, Fmax, F)
                y = self.calc_y_kv(F, kv)
                # print('y', y)
                if self.check_failed_mole_fractions(y):
                    res.isnan = True
                    return res
                eos = self.mr.eos_run(y, P, T, phases, **kwargs)
                residuals_prev = residuals
                residuals = np.log(kv) + eos.lnphi[:-1] - eos.lnphi[-1]
                if self.ssi_use_opt:
                    lambda_pow = - lambda_pow * np.sum(residuals_prev**2) / (np.sum(residuals_prev * residuals) - np.sum(residuals_prev**2))
                    kv = kv * np.exp(residuals)**(-lambda_pow)
                else:
                    kv = kv * np.exp(residuals)
                # print('residuals', np.max(np.abs(residuals)))
                it += 1
                if self.ssi_switch and it > 1:
                    if np.sum(residuals**2) / np.sum(residuals_prev**2) > self.ssi_eps_r and np.all(np.abs(F - F_prev) < self.ssi_eps_v) and \
                    self.ssi_eps_l < np.sum(residuals**2) < self.ssi_eps_u and 0 < F < 1:
                        break
                    F_prev = F
            if not res.isnan:
                res.kv = kv
                res.y = self.calc_y_kv(F, kv)
                res.eos = eos
                res.F = self.calc_full_F(F)
                if self.full_output:
                    res.it = it
                    res.residuals = residuals
                    phases_replaced = self.replace_duplicates(phases)
                    res.comp_mole_frac = dict(zip(phases_replaced, res.y))
                    res.phase_mole_frac = dict(zip(phases_replaced, res.F.ravel()))
        else:
            res.y = np.array([self.z])
            res.kv = np.ones_like(res.y)
            res.eos = self.mr.eos_run(res.y, P, T, phases, **kwargs)
            res.F = np.array([[1.0]])
            if self.full_output:
                res.it = 0
                res.residuals = [0.0]
                res.comp_mole_frac = {phases: res.y}
                res.phase_mole_frac = {phases: res.F.ravel()}
        return res
    pass


class flash_isothermal_gibbs(flash_isothermal_ssi):
    def __init__(self, mr, eos_ders, z, gibbs_eps=1e-8, gibbs_max_iter=10, **kwargs):
        super(flash_isothermal_gibbs, self).__init__(mr, z, **kwargs)
        self.eos_ders = eos_ders
        self.gibbs_eps = gibbs_eps
        self.gibbs_max_iter = gibbs_max_iter
        pass

    def calc_gibbs_equation(self, lnf):
        return lnf[:-1] - lnf[-1]

    def calc_gibbs_jacobian(self, dlnf_dnk):
        Np_1 = dlnf_dnk.shape[0] - 1
        jac = np.empty(shape=(Np_1 * self.mr.Nc, Np_1 * self.mr.Nc))
        for j in range(Np_1):
            for k in range(Np_1):
                if k == j:
                    jac[j*self.mr.Nc:self.mr.Nc*(j+1), k*self.mr.Nc:(k+1)*self.mr.Nc] = dlnf_dnk[j] + dlnf_dnk[-1]
                else:
                    jac[j*self.mr.Nc:self.mr.Nc*(j+1), k*self.mr.Nc:(k+1)*self.mr.Nc] = dlnf_dnk[-1]
        return jac

    def flash_isothermal_gibbs_run(self, P, T, phases, kv0=0, **kwargs):
        res = self.empty_object()
        res.isnan = False
        if len(phases) > 1:
            ssi_res = self.flash_isothermal_ssi_run(P, T, phases, kv0, **kwargs)
            if ssi_res.isnan:
                return ssi_res
            Np_1 = len(phases) - 1
            F = ssi_res.F
            y = ssi_res.y
            nij0 = y[:-1] * F[:-1] * self.mr.n
            eos = self.mr.eos_run(y, P, T, phases, **kwargs)
            residuals = self.calc_gibbs_equation(eos.lnf).reshape(Np_1 * self.mr.Nc, 1)
            nij0 = nij0.reshape((Np_1 * self.mr.Nc, 1))
            nij = nij0
            it = 0
            while (np.abs(residuals) > self.gibbs_eps).any() and it < self.gibbs_max_iter:
                nij_reshape = nij.reshape((Np_1, self.mr.Nc))
                F = np.sum(nij_reshape, axis=1).reshape((Np_1, 1)) / self.mr.n
                y = np.append(nij_reshape / (self.mr.n * F), np.array([self.z - np.sum(nij_reshape, axis=0) / self.mr.n]) / (1 - np.sum(F)), axis=0)
                eos = self.mr.eos_run(y, P, T, phases, **kwargs)
                residuals = self.calc_gibbs_equation(eos.lnf).reshape(Np_1 * self.mr.Nc, 1)
                nj = self.calc_full_F(F) * self.mr.n
                nij = nij - np.linalg.inv(self.calc_gibbs_jacobian(self.eos_ders(self.mr, eos, nj, der_nk=True).get('dlnf_dnk').dlnf_dnk)).dot(residuals)
                it += 1
                if np.isnan(nij).any():
                    res.isnan = True
                    return res
            if not res.isnan:
                nij_reshape = nij.reshape((Np_1, self.mr.Nc))
                res.F = np.sum(nij_reshape, axis=1).reshape((Np_1, 1)) / self.mr.n
                res.y = np.append(nij_reshape / (self.mr.n * res.F), np.array([self.z - np.sum(nij_reshape, axis=0) / self.mr.n]) / (1 - np.sum(res.F)), axis=0)
                res.eos = eos
                if 0 in res.y[-1]:
                    res.y[-1] = np.where(res.y[-1] == 0, 1e-100, res.y[-1])
                res.kv = res.y[:-1] / res.y[-1]
                res.F = self.calc_full_F(res.F)
                if self.full_output:
                    res.it = it
                    res.ssi_it = ssi_res.it
                    res.residuals = residuals
                    phases_replaced = self.replace_duplicates(phases)
                    res.comp_mole_frac = dict(zip(phases_replaced, res.y))
                    res.phase_mole_frac = dict(zip(phases_replaced, res.F.ravel()))
        else:
            res.y = np.array([self.z])
            res.kv = np.ones_like(res.y)
            res.eos = self.mr.eos_run(res.y, P, T, phases, **kwargs)
            res.F = np.array([[1.0]])
            if self.full_output:
                res.it = 0
                res.ssi_it = 0
                res.residuals = [0.0]
                res.comp_mole_frac = {phases: res.y}
                res.phase_mole_frac = {phases: res.F.ravel()}
        return res
    pass


class equilibrium_isothermal(flash_isothermal_gibbs):
    def __init__(self, mr, eos_ders, z, stab_update_kv=False, stab_max_phases=3, stab_onephase_only=False, stab_kv_init_levels=range(0, 5, 1),
                 stab_max_iter=15, stab_eps=1e-8, stab_ssi_max_iter=10, stab_include_water=False, stab_onephase_calc_condition=False, **kwargs):
        super(equilibrium_isothermal, self).__init__(mr, eos_ders, z, **kwargs)
        self.stab_onephase_only = stab_onephase_only
        self.stab_kv_init_levels = stab_kv_init_levels
        self.stab_max_iter = stab_max_iter
        self.stab_ssi_max_iter = stab_ssi_max_iter
        self.stab_eps = stab_eps
        self.stab_update_kv = stab_update_kv
        self.stab_max_phases = stab_max_phases
        self.stab_onephase_calc_condition = stab_onephase_calc_condition
        if stab_include_water or mr.sw:
            self.stab_all_phases = ['g', 'w', 'o']
            self.stab_phases = ['g', 'w']
        else:
            self.stab_all_phases = ['g', 'o']
            self.stab_phases = ['g']
        if not stab_update_kv:
            self.stab_phase_states = self.stab_phases.copy()
            p1 = 'g' if self.stab_phases[-1] == 'w' else 'w'
            for i in range(0, stab_max_phases - 1, 1):
                self.stab_phase_states.append(p1 + i * 'o' + self.stab_phases[-1])
        pass

    @staticmethod
    def calc_tpd(flash, flash0):
        return np.sum(flash.F * flash.y * flash.eos.lnf - flash0.F * flash0.y * flash0.eos.lnf)

    @staticmethod
    def calc_stab_onephase_jacobian(Y, ders, n):
        return np.identity(Y.shape[1]) / Y + ders.dlnphi_dnk * n / np.sum(Y)

    @staticmethod
    def calc_stab_onephase_equation(Y, lnphi, z, lnphi0):
        return np.log(Y) + lnphi - np.log(z) - lnphi0

    def calc_stab_onephase(self, P, T, state, flash0, checking_levels, **kwargs):
        check_phases = self.stab_all_phases.copy()
        for phase in self.stab_phases:
            if phase in state and phase in check_phases:
                check_phases.remove(phase)
        # print(state, 'stab_phases', self.stab_phases,  'check_phases', check_phases)
        if self.stab_update_kv:
            Ycurr = None
            PHcurr = None
            KVcurr = None
        for new_phase in check_phases:
            phases = new_phase + state[-1]
            for level in checking_levels:
                kv = self.calc_kv_init(P, T, phases, level)
                # print(state, phases, level, 'kv0', kv)
                Y = kv * flash0.y[-1]
                # print(state, phases, level, 'Y0', Y)
                y = Y / np.sum(Y)
                # print(state, phases, level, 'y0', y)
                eos = self.mr.eos_run(y, P, T, new_phase, **kwargs)
                residuals = self.calc_stab_onephase_equation(Y, eos.lnphi[0], flash0.y[-1], flash0.eos.lnphi[-1]).reshape((self.mr.Nc, 1))
                i = 0
                while np.any(np.abs(residuals) > self.stab_eps) and i < self.stab_max_iter:
                    if i < self.stab_ssi_max_iter:
                        Y = np.array([np.exp(np.log(flash0.y[-1]) + flash0.eos.lnphi[-1] - eos.lnphi[0])])
                    else:
                        nj = np.array([flash0.F[-1]]) * self.mr.n
                        Y = Y - np.linalg.inv(self.calc_stab_onephase_jacobian(Y, self.eos_ders(self.mr, eos, nj, der_nk=True).get('dlnphi_dnk'), nj)).dot(residuals).reshape((1, self.mr.Nc))
                    y = Y / np.sum(Y)
                    # print('iter', i, 'y', y)
                    eos = self.mr.eos_run(y, P, T, new_phase, **kwargs)
                    residuals = self.calc_stab_onephase_equation(Y, eos.lnphi[0], flash0.y[-1], flash0.eos.lnphi[-1]).reshape((self.mr.Nc, 1))
                    i += 1
                Ysum = np.sum(Y)
                # print(state, phases, level, 'Ysum', Ysum, 'iters', i, 'max_residuals', np.max(np.abs(residuals)))
                if self.stab_onephase_calc_condition:
                    nj = np.array([flash0.F[-1]]) * self.mr.n
                    condition = 1 + np.sum(self.stab_eps / np.diag(self.calc_stab_onephase_jacobian(Y, self.eos_ders(self.mr, eos, nj, der_nk=True).get('dlnphi_dnk'), nj)[0]))
                else:
                    condition = 1 + self.stab_eps
                # print('condition', condition, 'Ysum > condition', Ysum > condition)
                # if i < self.stab_max_iter and Ysum > condition:
                if Ysum > condition:
                    if not self.stab_update_kv:
                        return False
                    else:
                        if Ycurr is None:
                            Ycurr = Ysum
                            PHcurr = phases
                            if new_phase == 'g' or new_phase == 'w' or 'o' in state:
                                KVcurr = Y / flash0.y[-1]
                            elif new_phase == 'o':
                                KVcurr = flash0.y[-1] / Y
                        elif Ycurr < (Ysum - condition + 1):
                            Ycurr = Ysum
                            PHcurr = phases
                            if new_phase == 'g' or new_phase == 'w' or 'o' in state:
                                KVcurr = Y / flash0.y[-1]
                            elif new_phase == 'o':
                                KVcurr = flash0.y[-1] / Y
        if not self.stab_update_kv:
            return True
        elif PHcurr is None:
            return True, None, self.stab_kv_init_levels[0]
        else:
            return False, PHcurr, KVcurr if len(state) == 1 else np.append(flash0.kv, KVcurr, axis=0)

    def calc_stab_multiphase(self, P, T, state, flash0, checking_levels, **kwargs):
        if self.stab_update_kv:
            tpds = []
            flashs = []
        for level in checking_levels:
            flash = self.flash_isothermal_gibbs_run(P, T, state, kv0=level, **kwargs)
            # print('mphase', level, flash.isnan)
            if not flash.isnan:
                tpd = self.calc_tpd(flash, flash0)
                # print('mphase', level, 'tpd', tpd)
                if tpd < -self.gibbs_eps:
                    if not self.stab_update_kv:
                        return False
                    else:
                        tpds.append(tpd)
                        flashs.append(flash)
        if not self.stab_update_kv:
            return True
        elif tpds:
            return False, flashs[np.argmin(tpds)]
        else:
            return True, flash0

    def equilibrium_isothermal_run(self, P, T, **kwargs):
        states_checked = {}
        if not self.stab_update_kv:
            for state in self.stab_phase_states:
                # print('state', state)
                for i, kv0 in enumerate(self.stab_kv_init_levels):
                    # print('kv0', kv0)
                    flash0 = self.flash_isothermal_gibbs_run(P, T, state, kv0, **kwargs)
                    # print('flash0.isnan', flash0.isnan)
                    if not flash0.isnan:
                        break
                else:
                    continue
                checking_levels = np.delete(self.stab_kv_init_levels, i).tolist()
                if self.stab_onephase_only or len(state) == 1:
                    stability = self.calc_stab_onephase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                else:
                    stability = self.calc_stab_multiphase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                if stability:
                    break
        else:
            i = 0
            state = self.stab_phases[i]
            kv0 = self.stab_kv_init_levels[0]
            flash0 = None
            stability = False
            while not stability and len(state) <= self.stab_max_phases:
                flash0 = self.flash_isothermal_gibbs_run(P, T, state, kv0, **kwargs)
                # print(state, 'flash0.isnan', flash0.isnan)
                # print('kv0', kv0)
                # print(*('flash0_res', np.max(np.abs(flash0.residuals)), 'flash0_gibbs_it', flash0.it, 'flash0_ssi_it', flash0.ssi_it) if not flash0.isnan else [flash0.isnan])
                if flash0.isnan:
                    for j, kv0 in enumerate(self.stab_kv_init_levels):
                        flash0 = self.flash_isothermal_gibbs_run(P, T, state, kv0, **kwargs)
                        # print('new flash0:', j, flash0.isnan)
                        if not flash0.isnan:
                            break
                    else:
                        state = state[:-1] + 'o' + state[-1]
                        kv0 = self.stab_kv_init_levels[0]
                        continue
                    checking_levels = np.delete(self.stab_kv_init_levels, j).tolist()
                else:
                    checking_levels = self.stab_kv_init_levels
                if len(state) == 1 or self.stab_onephase_only:
                    stability, splitted_phases, kv0 = self.calc_stab_onephase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                    # print('state', state, 'stability', stability)
                    if not stability:
                        state = state[:-1] + splitted_phases
                else:
                    stability, flash = self.calc_stab_multiphase(P, T, state, flash0, checking_levels, **kwargs)
                    states_checked.update({state: stability})
                    if isinstance(kv0, int):
                        kv0 = self.calc_kv_init(P, T, state, kv0)
                    state = state[:-1] + 'o' + state[-1]
                    kv0 = np.append(kv0, [flash.y[-1] / flash.y[-2]], 0)
                if len(state) > self.stab_max_phases and i < len(self.stab_phases) - 1:
                    i += 1
                    state = self.stab_phases[i]
                    kv0 = self.stab_kv_init_levels[0]
                if stability:
                    break
        flash0.states_checked = states_checked
        return flash0