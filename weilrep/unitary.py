r"""

Theta lifts and Borcherds products on unitary groups U(n, 1)

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2021-2023 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from collections import defaultdict

from copy import copy, deepcopy

from re import sub

from sage.arith.functions import lcm
from sage.functions.other import ceil, factorial, floor, frac, sqrt
from sage.geometry.cone import Cone
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import Matrix
from sage.matrix.special import block_diagonal_matrix
from sage.misc.functional import denominator, isqrt
from sage.misc.misc_c import prod
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modular.modform.vm_basis import delta_qexp
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.big_oh import O
from sage.rings.fraction_field import FractionField
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.flatten import FlatteningMorphism
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.ring import Field

from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian, RescaledHyperbolicPlane
from .morphisms import WeilRepAutomorphism
from .positive_definite import OrthogonalModularFormPositiveDefinite, OrthogonalModularFormsPositiveDefinite, WeilRepModularFormPositiveDefinite
from .weilrep import WeilRep
from .weilrep_misc import relations
from .weilrep_modular_forms_class import smf, smf_eta, smf_eisenstein_series, WeilRepModularFormPrincipalPart, WeilRepModularFormsBasis


class HermitianWeilRep(WeilRep):
    r"""
    The Weil representation associated to an even integral Hermitian lattice (L, H) over an imaginary-quadratic number field 'K'.

    This is the WeilRep associated to the integral quadratic form: Q(x) := Tr_{K / QQ} H(x). Elements of the disriminant group L'/L are represented as vectors over K. It can be constructed by calling 'WeilRep', rather than 'HermitianWeilRep'.

    INPUT:

    - ``S`` -- a
    """
    def __init__(self, S, **kwargs):
        K = S.base_ring()
        d = K.discriminant()
        d0 = d
        if d0 % 4 == 0:
            d0 = d0 // 4
        gen = kwargs.pop('gen', None)
        if gen is None:
            if d % 2:
                gen = (-1 + K.gen()) / 2
            else:
                gen = K.gen()
        try:
            ell, m = gen.parts()
        except AttributeError:
            ell, m = gen, 0
        u = ell * ell + m * m * (-d0)
        ell, m, u = ell + ell, m + m, u + u
        n = S.nrows()
        T = Matrix(QQ, 2 * n)
        #'Omega' saves the action of our generator of O_K on the trace form
        Omega = Matrix(QQ, 2 * n)
        m0 = m * d0
        #setup the trace form as a ZZ-lattice with respect to the generator 'gen'
        for i in range(n):
            for j in range(i + 1):
                try:
                    re, im = S[i, j].parts()
                except AttributeError:
                    re, im = S[i, j], 0
                T[i + i, j + j] = 2*re
                T[i + i + 1, j + j] = ell * re + m0 * im
                T[i + i + 1, j + j + 1] = u * re
                T[i + i, j + j + 1] = ell * re - m0 * im
                T[j + j, i + i] = T[i + i, j + j]
                T[j + j + 1, i + i] = T[i + i, j + j + 1]
                T[j + j, i + i + 1] = T[i + i + 1, j + j]
                T[j + j + 1, i + i + 1] = T[i + i + 1, j + j + 1]
            Omega[i+i, i+i+1] = -u // 2
            Omega[i+i+1, i+i] = 1
            Omega[i+i+1, i+i+1] = ell
        super().__init__(T)
        self.__base_field = K
        self.__class__ = HermitianWeilRep
        self.__complex_gram_matrix = S
        self.__w = gen
        self.__winv = -~gen.conjugate()
        self.__Omega = Omega
        self.lift_qexp_representation = 'unitary'
        self._plus_H = kwargs.pop('plus_H', False)

    def __add__(self, other, **kwargs):
        r"""
        Direct sum of Hermitian lattices.
        """
        if isinstance(other, HermitianWeilRep):
            return HermitianWeilRep(block_diagonal_matrix([self.complex_gram_matrix(), other.complex_gram_matrix()], subdivide=False), gen=self.__w)
        elif isinstance(other, RescaledHyperbolicPlane):
            return HermitianRescaledHyperbolicPlane(other._N()).__add__(self)
        return NotImplemented

    def __call__(self, N):
        r"""
        Rescale by integers.
        """
        try:
            N = Integer(N)
            return HermitianWeilRep(N * self.complex_gram_matrix(), gen=self.__w)
        except TypeError:
            return super().__call__(N)

    def dual(self):
        r"""
        Return the dual Weil representation.
        """
        try:
            return self.__dual
        except AttributeError:
            s = HermitianWeilRep(-self.complex_gram_matrix(), gen=self.__w)
            self.__dual = s
            return s

    def __repr__(self):
        S = self.complex_gram_matrix()
        return 'Weil representation associated to the Hermitian Gram matrix\n%s\nwith coefficients in %s' % (S, S.base_ring())

    def base_field(self):
        return self.__base_field

    def complex_gram_matrix(self):
        r"""
        This returns the Gram matrix over O_K.

        If gram_matrix() is called then we instead return the Gram matrix of the trace form over ZZ.
        """
        return self.__complex_gram_matrix

    def _is_hermitian_weilrep(self):
        return True

    def _lifts_have_fourier_jacobi_expansion(self):
        raise NotImplementedError

    def multiplication_by_i(self):
        r"""
        Return the multiplication-by-i map.

        Suppose K = QQ(i). This returns the map x --> i * x as a WeilRepAutomorphism; it can be applied to vectors in self's discriminant group or (componentwise) to vector-valued modular forms.
        """
        d = self.base_field().discriminant()
        if d != -4:
            raise ValueError('The base field does not contain I.')
        w = self._w()
        a, b = w.parts()

        def f(v):
            n = len(v)
            if n == self.complex_gram_matrix().nrows():
                i = (w - a) / b

                def z(x):
                    re, im = x.parts()
                    return frac(re) + i * frac(im)
                return vector(z(i * x) for x in v)
            L = [0] * n
            for i in range(n // 2):
                u = (v[i + i] + v[i + i + 1] * a) / b
                L[i + i] = frac(-a*u - b * v[i + i + 1])
                L[i + i + 1] = frac(u)
            return vector(L)
        return WeilRepAutomorphism(self, f)

    def multiplication_by_zeta(self):
        r"""
        Return the multiplication-by-zeta map.

        Suppose K = QQ(sqrt(-3)). Let \zeta = e^{\pi i / 3} be the sixth root of unity. This returns the map x --> \zeta * x as a WeilRepAutomorphism; it can be applied to vectors in self's discriminant group or (componentwise) to vector-valued modular forms.
        """
        d = self.base_field().discriminant()
        if d != -3:
            raise ValueError('The base field does not contain e^(pi * I / 3).')
        w = self._w()
        a, b = w.parts()
        c, d, e = b - a, a * a + 3 * b * b, a + b

        def f(v):
            n = len(v)
            if n == self.complex_gram_matrix().nrows():
                zeta = (w + b - a) / (b + b)
                isqrt3 = 2 * zeta - 1

                def z(x):
                    re, im = x.parts()
                    return frac(re) + isqrt3 * frac(im)
                return vector(z(zeta * x) for x in v)
            L = [0] * n
            for i in range(n // 2):
                L[i + i] = frac((c * v[i + i] - d * v[i + i + 1]) / (b + b))
                L[i + i + 1] = frac((v[i + i] + e * v[i + i + 1]) / (b + b))
            return vector(L)
        return WeilRepAutomorphism(self, f)

    def _norm_form(self):
        r"""
        Return the norm N_{K/QQ} as a integral quadratic form.
        """
        w = self._w()
        wc = w.galois_conjugate()
        wtr = Integer(w + wc)
        return QuadraticForm(Matrix(ZZ, [[2, wtr], [wtr, 2 * w * wc]]))

    def hds(self):
        r"""
        Compute representatives of the discriminant group L'/L

        This is similar to self.ds() but the results are represented as vectors over K rather than QQ.
        """
        try:
            return self.__ds
        except AttributeError:
            X = self.ds()
            w = self.__w
            n = len(X[0]) // 2
            g = self.base_field().gens()[0]

            def a(x):
                c, d = x.parts()
                return frac(c) + g * frac(d)
            self.__ds = [vector([a(v[i+i] + w*v[i+i+1]) for i in range(n)]) for v in X]
            return self.__ds

    def is_lorentzian(self):
        #ZZ-lattices of signature (n, 1) do not have complex structures
        return False

    def is_lorentzian_plus_II(self):
        return self._plus_H

    def _hds_to_ds(self):
        hds = self.hds()
        ds = self.ds()
        return {tuple(g): tuple(ds[i]) for i, g in enumerate(hds)}

    def _ds_to_hds(self):
        hds = self.hds()
        ds = self.ds()
        return {tuple(g): tuple(hds[i]) for i, g in enumerate(ds)}

    def _hds_dict(self):
        try:
            return self.__hds_dict
        except AttributeError:
            ds = [tuple(x) for x in self.hds()]
            self.__hds_dict = dict(zip(ds, range(len(ds))))
            return self.__hds_dict

    def _h_norm_dict(self):
        try:
            return self.__h_norm_dict
        except AttributeError:
            n = self.norm_list()
            ds = self.hds()
            self.__h_norm_dict = {tuple(g): n[i] for i, g in enumerate(ds)}
            return self.__h_norm_dict

    def trace_form(self):
        r"""
        Return the WeilRep associated to the underlying ZZ-lattice.
        """
        w = WeilRep(self.gram_matrix())
        w.lift_qexp_representation = 'PD+II'
        return w

    def _units(self):
        r"""
        A list of the units of O_K.

        If d_K < -4 then this is just [1, -1].
        """
        try:
            return self.__units
        except AttributeError:
            K = self.base_field()
            d = K.discriminant()
            if d < -4:
                L = [1, -1]
            elif d == -4:
                w = self._w()
                a, b = w.parts()
                i = (w - a) / b
                L = [1, i, -1, -i]
            else:
                w = self._w()
                a, b = w.parts()
                zeta = (w + b - a) / (b + b)
                L = [1]
                for _ in range(5):
                    L.append(zeta * L[-1])
            self.__units = L
            return L

    def unitary_reflection(self, r, alpha=-1):
        r"""
        Construct the complex reflection `\sigma_{r, \alpha}(v) = v - (1 - \alpha) <r, v> r / <r, r>`.

        INPUT:

        - ``r`` -- a dual lattice vector
        - ``\alpha`` -- a unit in O_K which is not 1.

        OUTPUT: WeilRepAutomorphism
        """
        S = self.complex_gram_matrix()
        r_conj = vector([x.galois_conjugate() for x in r])
        r_norm = r * S * r_conj
        if not r_norm:
            raise ValueError('Not a valid reflection')
        d = self._hds_to_ds()
        d_inv = self._ds_to_hds()
        r0 = ((1 - alpha) / r_norm) * (S * r_conj)
        g = self.base_field().gens()[0]
        w = self._w()
        w_conj = w.galois_conjugate()
        M = Matrix(ZZ, [[2, w + w_conj], [0, (w - w_conj)*g]]).inverse()

        def f(x):
            v = vector(d_inv[tuple(x)])
            v = v - (v*r0) * r
            x = [0]*(2 * len(v))
            for i, v in enumerate(v):
                v_conj = v.galois_conjugate()
                a, b = (v + v_conj), (v - v_conj) * g
                x[i+i], x[i+i+1] = M * vector([a, b])
            return tuple(map(frac, x))
        return WeilRepAutomorphism(self, f)

    def biflection(self, r):
        return self.unitary_reflection(r, alpha=-1)

    def triflection(self, r):
        if not self.base_field().discriminant() == -3:
            raise ValueError('This lattice does not admit triflections.')
        return self.unitary_reflection(r, alpha=self._units()[2])

    def tetraflection(self, r):
        if not self.base_field().discriminant() == -4:
            raise ValueError('This lattice does not admit tetraflections.')
        return self.unitary_reflection(r, alpha=self._units()[1])

    def hexaflection(self, r):
        if not self.base_field().discriminant() == -3:
            raise ValueError('This lattice does not admit hexaflections.')
        return self.unitary_reflection(r, alpha=self._units()[1])

    def _w(self):
        r"""
        The generator of O_K we picked when constructing the WeilRep
        """
        return self.__w

    def _winv(self):
        r"""
        Return 1 / self._w()
        """
        return self.__winv

    def _pos_def_gram_matrix(self):
        S = self.gram_matrix()
        return S


class HermitianWeilRepModularForm(WeilRepModularFormPositiveDefinite):

    def __getattr__(self, x):
        return getattr(self.weilrep(), x)

    def borcherds_lift(self, max_prec=None, _flag=False):
        d = self.denominator()
        umf = UnitaryModularForms(self.weilrep())
        K = umf.field()
        if d > 1:
            h, k = (d * self).borcherds_lift(max_prec=max_prec, _flag=True)
            k = ZZ(k)
            try:
                u = h.trailing_monomial()
            except AttributeError:
                u = h.valuation()
            if (K.discriminant() == -3 and (k / d) % 6 == 0) or (K.discriminant() == -4 and (k / d) % 4 == 2):
                h /= K.gens()[0]
            h = _root(h, d, k, umf)
            return UnitaryModularForm(self.complex_gram_matrix(), h, k / d)
        #CM values of e4 / eta**8 and e6 / eta**12
        d = K.discriminant()
        if d == -3:
            isqrt3, = K.gens()
            cm_val = (0, 24 * isqrt3)
        elif d == -4:
            cm_val = (12, 0)
        else:
            return NotImplemented
        F = super().borcherds_lift()
        k = F.weight()
        scale = F.scale()
        try:
            Fdict = F.true_coefficients()
            hol = True
        except AttributeError:
            Fdict = F.fourier_expansion().coefficients()
            hol = False
        prec0 = 2 * F.precision()
        prec = 6 * prec0 - k
        if max_prec:
            prec = min(prec, max_prec)
        rx, x = LaurentPolynomialRing(K, 'x').objgen()
        rt, t = PowerSeriesRing(rx, 't').objgen()
        r = umf._taylor_exp_ring_extended()
        e4 = eisenstein_series_qexp(4, prec0, normalization='constant')
        e6 = eisenstein_series_qexp(6, prec0, normalization='constant')
        w1 = II(1)
        v = vector([0, 0])
        E4 = OrthogonalModularForm(4, w1, e4(t * ~x) * e4(t * x), 1, v)
        E6 = OrthogonalModularForm(6, w1, e6(t * ~x) * e6(t * x), 1, v)
        z = r.gens()
        rank = len(z)
        e = r(0).add_bigoh(prec)
        w = self.weilrep()
        zeta = w._w()
        zeta_conj = zeta.conjugate()
        rho = -zeta_conj
        eta3 = sum((2 * n + 1) * (-1) ** n * t ** (ZZ(n * (n + 1) / 2)) for n in range(isqrt(2 * prec0) + 2)).add_bigoh(prec0)
        L = [[i] for i in range(prec)]
        j = 1
        while j < rank:
            j += 1
            L = [x + [i] for x in L for i in range(prec) if sum(x) + i < prec]
        pb = {}
        if not hol:
            r1 = PowerSeriesRing(K, r.gens())
            exp_z = [y.exp() for b in r1.gens() for y in [b, rho * b]]
        for i, y in enumerate(L):
            g = rt(0)
            if hol:
                for a, c in Fdict.items():
                    g += c * prod((a[2*i] + rho * a[2*i+1])**y[i - 1] for i in range(1, len(a) // 2)) * x**(scale * a[1]) * t**ZZ(scale * a[0])
            else:
                for m, c in Fdict.items():
                    a, b = m.exponents()[0]
                    if a + b < prec0:
                        rc = c.parent()
                        rc, p = PolynomialRing(K, rc.gens()).objgens()
                        c = rc.fraction_field()(c)
                        for i, u in enumerate(y):
                            for j in range(u):
                                c = p[2*i] * c.derivative(p[2*i]) + rho * p[2*i+1] * c.derivative(p[2*i + 1])
                        j = 5
                        exp_z = [y.exp(j) for b in r1.gens() for y in [b, rho * b]]
                        c_n = c.numerator()
                        c_ns = c_n.subs({a: exp_z[i] for i, a in enumerate(c_n.parent().gens())})
                        c_d = c.denominator()
                        c_ds = c_d.subs({a: exp_z[i] for i, a in enumerate(c_d.parent().gens())})
                        while not c_ds:
                            j += 5
                            exp_z = [y.exp(j) for b in r1.gens() for y in [b, rho * b]]
                            c_ns = c_n.subs({a: exp_z[i] for i, a in enumerate(c_n.parent().gens())})
                            c_ds = c_d.subs({a: exp_z[i] for i, a in enumerate(c_d.parent().gens())})
                            if j > 1000:
                                raise RuntimeError  # ??
                        try:
                            u = c_ds.trailing_monomial()
                            c_ns /= u
                            c_ds /= u
                            c = c_ns.constant_coefficient() / c_ds.constant_coefficient()
                        except AttributeError:
                            u = c_ds.valuation()
                            c = c_ns[u] / c_ds[u]
                        g += c * (x ** (scale * (a - b))) * (t ** (scale * (a + b)))
            pb[tuple(y)] = OrthogonalModularForm(k + sum(y), w1, g.add_bigoh(floor(F.precision())) / prod(factorial(a) for a in y), scale, v)
        if d == -3:
            eta = sum((-1)**n * (t**(n * (3 * n + 1) // 2) - t**((n+1) * (3*n+2) // 2)) for n in range(isqrt(2 * prec0 / 3) + 1)).add_bigoh(prec0)
            eta_p = (eta3 * eta) ** 2
            eta_p = OrthogonalModularForm(4, w1, t**2 * eta_p((t * ~x)**3) * eta_p((t * x)**3), 3, v)
            delta = t * eta3 ** 8
            k0 = 4
        elif d == -4:
            eta6 = eta3 ** 2
            eta_p = OrthogonalModularForm(3, w1, t**2 * eta6((t * ~x)**4) * eta6((t * x)**4), 4, v)
            delta = t * eta6 ** 4
            k0 = 3
        else:
            eta_p = eta3 ** 4
            k0 = 6
            delta = t * eta_p * eta_p
            eta_p = OrthogonalModularForm(6, w1, t**2 * eta_p((t * ~x)**2) * eta_p((t * x)**2), 2, v)
        E4_powers, E6_powers, Eta_powers = [1], [1], [1]
        e43 = e4 ** 3
        Delta = OrthogonalModularForm(12, w1, delta(t * x) * e43(t * ~x) - delta(t * ~x) * e43(t * x), 1, v)
        prec1 = floor(k + prec)
        for j in range(prec1 // min(k0, 4) + 1):
            if j * k0 <= prec1:
                Eta_powers.append(eta_p * Eta_powers[-1])
            if j * 4 <= prec1:
                E4_powers.append(E4 * E4_powers[-1])
            if j * 6 <= prec1:
                E6_powers.append(E6 * E6_powers[-1])
        Z = defaultdict(list)
        I = defaultdict(list)
        for i1, x1 in enumerate(Eta_powers):
            for i2, x2 in enumerate(E4_powers):
                i_sum_0 = k0 * i1 + 4 * i2
                if i_sum_0 < k + prec:
                    x1x2 = x1 * x2
                    for i3, x3 in enumerate(E6_powers):
                        i_sum = i_sum_0 + 6 * i3
                        if i_sum < k + prec:
                            f = x1x2 * x3
                            Z[i_sum].append(f)
                            I[i_sum].append((i1, i2, i3, 0))
                            if i_sum + 12 < k + prec:
                                Z[i_sum + 12].append(f * Delta)
                                I[i_sum + 12].append((i1, i2, i3, 1))
        h = r(0)
        e4, e6, eta_p, delta = umf._e4(), umf._e6(), umf._eta_p(), umf._delta()
        for x in L:
            g = pb[tuple(x)]
            x_monomial = prod(z[i]**a for i, a in enumerate(x))
            g_k = g.weight()
            X = relations([g] + Z[g_k]).basis()
            if len(X) == 1:
                v = X[0]
                s = sum(v[i + 1] * eta_p**a * (cm_val[0] * e4)**b * (cm_val[1] * e6)**c * (e4**3 - cm_val[0]**3 * delta)**d
                        for i, (a, b, c, d) in enumerate(I[g_k]))
                h += s * x_monomial
            else:
                h += O(x_monomial)
                if _flag:
                    return h, F.weight()
                return UnitaryModularForm(self.complex_gram_matrix(), h, F.weight(), umf=umf)
        h = h.add_bigoh(prec)
        if _flag:
            return h, F.weight()
        return UnitaryModularForm(self.complex_gram_matrix(), h, F.weight(), umf=umf)

    def theta_lift(self):
        cusp_form = self.is_cusp_form()
        rank = self.complex_gram_matrix().nrows()
        umf = UnitaryModularForms(self.weilrep())
        k = self.weight() + rank
        prec = max(self.precision(), 2)
        K = umf.field()
        d = K.discriminant()
        ## values of E4(\tau) / \eta(\tau)**8 and E6(\tau) / \eta(\tau)**12 at a CM point \tau of discriminant 'd'
        if d == -3:
            isqrt3, = K.gens()
            cm_val = [0, 24*isqrt3]
        elif d == -4:
            cm_val = [12, 0]
        elif d == -7:
            isqrt7, = K.gens()
            x = PolynomialRing(K, 'x').gen()
            L, isqrt3 = K.extension(x * x + 3, 'isqrt3').objgen()
            cm_val = [ZZ(15)/2 * (1 - isqrt3), -27 * isqrt7]
        elif d == -8:
            isqrt2, = K.gens()
            x = PolynomialRing(K, 'x').gen()
            L, sqrt2 = K.extension(x * x - 2, 'sqrt2').objgen()
            cm_val = [20, 56 * sqrt2]
        elif d == -11:
            isqrt11, = K.gens()
            x = PolynomialRing(K, 'x').gen()
            L, isqrt3 = K.extension(x * x + 3, 'isqrt3').objgen()
            cm_val = [16 * (1 + isqrt3), 56 * isqrt11]
        else:
            raise NotImplementedError('This is only supported for lattices over Q(sqrt(-d)), d=1,2,3,7,11.') from None
        rx, x = LaurentPolynomialRing(QQ, 'x').objgen()
        rt, t = PowerSeriesRing(rx, 't').objgen()
        r = umf._taylor_exp_ring()
        E4 = eisenstein_series_qexp(4, prec, normalization='constant')
        E6 = eisenstein_series_qexp(6, prec, normalization='constant')
        Delta = delta_qexp(prec)
        w1 = II(1)
        v = vector([0, 0])
        E4 = OrthogonalModularForm(4, w1, E4(t * ~x) * E4(t * x), 1, v)
        E6 = OrthogonalModularForm(6, w1, E6(t * ~x) * E6(t * x), 1, v)
        Delta = OrthogonalModularForm(12, w1, Delta(t * ~x) * Delta(t * x), 1, v)
        w = self.weilrep()
        z = r.gens()
        zeta = w._w()
        zeta_conj = zeta.conjugate()
        rho = -zeta_conj
        r0 = PolynomialRing(r, ['x%s' % i for i in range(2 * rank)])
        prec1 = 6 * isqrt(prec)

        def a(u):
            S = w.gram_matrix()
            v = S * vector(u)
            y = [v[i+i+1] + rho * v[i+i] for i in range(len(v) // 2)]
            return prod(z[i]*y for i, y in enumerate(y)).exp(prec=prec1)
        theta = w.dual().theta_series(prec, P=r0(1), funct=a, symm=False)
        f = self & theta
        # reorganize f...
        rq1, q = PowerSeriesRing(K, 'q').objgen()
        rz1 = PowerSeriesRing(rq1, r.gens())
        f = sum(q**b * rz1(h) for b, h in f.dict().items())

        def b(h, wt):
            cf = cusp_form or (wt > k)
            if cf:
                X = w1.cusp_forms_basis(wt, prec)
            else:
                X = w1.modular_forms_basis(wt, prec)
            if X:
                return (X * [K(x) for x in h.padded_list()[cf: len(X)+cf]]).theta_lift()
            else:
                return 0
        if len(z) > 1:
            df = {a: b(h, sum(a) + k) for a, h in f.dict().items()}
        else:
            df = {(a,): b(h, a + k) for a, h in f.dict().items()}
        k_max = k + prec1
        Z = defaultdict(list)
        I = defaultdict(list)
        e4_powers, e6_powers, d_powers = [1], [1], [1]
        k12 = ZZ(k_max - 12) // 12 + 3 - cusp_form
        if d == -3:
            ## Eisenstein lattice
            e6, eta8 = r.base_ring().gens()
            delta = eta8 ** 3
            E4_cubed = E4 ** 3
            k6 = ZZ(k_max - 12) // 6 + 3 - cusp_form
            e4_powers, e6_powers, d_powers = [1], [1], [1]
            for j in range(1, k6 + 1):
                e6_powers.append(e6_powers[-1] * E6)
                if j <= k12:
                    e4_powers.append(e4_powers[-1] * E4_cubed)
                    d_powers.append(d_powers[-1] * Delta)
            L = []
            mult = []
            exponents = []
            for i_1 in range(k12):
                for i_2 in range(k6):
                    for i_3 in range(k12):
                        i_sum = 6 * i_2 + 12 * (i_1 + i_3)
                        if k <= i_sum < k_max:
                            Z[i_sum].append(e4_powers[i_1] * e6_powers[i_2] * d_powers[i_3])
                            I[i_sum].append((i_1, i_2, i_3))
            h = r(0)
            maxprec = Infinity
            for a, g in df.items():
                if g:
                    k1 = k + sum(a)
                    if k1 < maxprec:
                        X = relations([g] + Z[k1]).basis()
                        if len(X) > 1:
                            h = h.add_bigoh(k1 - k)
                            maxprec = k1
                        v = X[0]
                        p = sum(v[i+1] * (24*isqrt3 * e6)**i2 * delta**i3 for i, (i1, i2, i3) in enumerate(I[k1]) if i1 == 0)
                        h += p * prod(z[i] ** a for i, a in enumerate(a))
            h = h.add_bigoh(prec1)
            return UnitaryModularForm(self.complex_gram_matrix(), h, k, umf=umf)
        elif d == -4:
            ## Gaussian lattice
            e4, eta6 = r.base_ring().gens()
            delta = eta6 ** 4
            E6_squared = E6 * E6
            k4 = ZZ(k_max - 12) // 4 + 2 - cusp_form
            for j in range(1, k4 + 1):
                e4_powers.append(e4_powers[-1] * E4)
                if j <= k12:
                    e6_powers.append(e6_powers[-1] * E6_squared)
                    d_powers.append(d_powers[-1] * Delta)
            L = []
            mult = []
            exponents = []
            for i_1 in range(k4):
                for i_2 in range(k12):
                    for i_3 in range(k12):
                        i_sum = 4 * i_1 + 12 * (i_2 + i_3)
                        if k <= i_sum < k_max:
                            Z[i_sum].append(e4_powers[i_1] * e6_powers[i_2] * d_powers[i_3])
                            I[i_sum].append((i_1, i_2, i_3))
            h = r(0)
            maxprec = Infinity
            for a, g in df.items():
                k1 = k + sum(a)
                if k1 < maxprec:
                    X = relations([g] + Z[k1]).basis()
                    if len(X) > 1:
                        h = h.add_bigoh(k1)
                        maxprec = k1
                    v = X[0]
                    p = sum(v[i+1] * (12 * e4)**i1 * delta**i3 for i, (i1, i2, i3) in enumerate(I[k1]) if i2 == 0)
                    h += p * prod(z[i] ** a for i, a in enumerate(a))
            h = h.add_bigoh(prec1)
            return UnitaryModularForm(self.complex_gram_matrix(), h, k, umf=umf)
        else:
            e4, e6 = r.base_ring().gens()
            delta = (e4 ** 3 - e6 ** 2) / 1728
            k4 = ZZ(k_max - 12) // 4 + 3 - cusp_form
            k6 = ZZ(k_max - 12) // 6 + 3 - cusp_form
            k12 = ZZ(k_max - 12) // 6 + 3 - cusp_form
            for j in range(1, k4 + 1):
                e4_powers.append(e4_powers[-1] * E4)
                if j <= k6:
                    e6_powers.append(e6_powers[-1] * E6)
                    if j <= k12:
                        d_powers.append(d_powers[-1] * Delta)
            L = []
            mult = []
            exponents = []
            for i_1 in range(k4):
                for i_2 in range(k6):
                    for i_3 in range(k12):
                        i_sum = 4 * i_1 + 6 * i_2 + 12 * i_3
                        if k <= i_sum < k_max:
                            Z[i_sum].append(e4_powers[i_1] * e6_powers[i_2] * d_powers[i_3])
                            I[i_sum].append((i_1, i_2, i_3))
            h = r(0)
            maxprec = Infinity
            for a, g in df.items():
                k1 = k + sum(a)
                if k1 < maxprec:
                    try:
                        X = relations([g] + Z[k1]).basis()
                        if len(X) > 1:
                            h = h.add_bigoh(k1)
                            maxprec = k1
                        v = X[0]
                        p = sum(v[i+1] * (cm_val[0] * e4)**i1 * (cm_val[1] * e6)**i2 * delta**i3 for i, (i1, i2, i3) in enumerate(I[k1]))
                        h += p * prod(z[i] ** a for i, a in enumerate(a))
                    except AttributeError: #g=0
                        pass
            h = h.add_bigoh(prec1)
            return UnitaryModularForm(self.complex_gram_matrix(), h, k, umf=umf)


class UnitaryModularForms(OrthogonalModularFormsPositiveDefinite):
    r"""
    This class represents modular forms for the unitary group U(n, 1) associated to our lattice.

    If 'L' is positive definite then we instead consider the unitary group of L + II(1), i.e. L plus a unimodular plane.
    """

    def __init__(self, *args, **kwargs):
        kwargs['unitary'] = 1
        super().__init__(*args, **kwargs)
        self.__class__ = UnitaryModularForms
        try:
            _ = self.weilrep()._lorentz_gram_matrix
            self._plusH = True
        except AttributeError:
            self._plusH = False

    def __repr__(self):
        S = self.weilrep().complex_gram_matrix()
        s = ''
        if not self._plusH:
            s = ' + H'
        return 'Unitary modular forms associated to the gram matrix\n%s%s\nwith coefficients in %s' % (S, s, S.base_ring())

    def _discriminant(self):
        try:
            return self.__discriminant
        except AttributeError:
            d = self.field().discriminant()
            self.__discriminant = d
            return d

    def field(self):
        return self.complex_gram_matrix().base_ring()

    def nvars(self):
        n = self.complex_gram_matrix().nrows()
        s = self._plusH
        return Integer(n + n + 2 - 4*s)

    def complex_gram_matrix(self):
        return self.weilrep().complex_gram_matrix()

    def _taylor_exp_coeff_ring(self):
        try:
            return self.__taylor_exp_coeff_ring
        except AttributeError:
            K = self.field()
            dK = K.discriminant()
            if dK == -7 or dK == -11:
                x = PolynomialRing(K, 'x').gen()
                K = K.extension(x * x + 3, 'isqrt3')
            elif dK == -8:
                x = PolynomialRing(K, 'x').gen()
                K = K.extension(x * x - 2, 'sqrt2')
            if dK == -3:
                R = PolynomialRing(K, ['e6', 'eta8'])
            elif dK == -4:
                R = PolynomialRing(K, ['e4', 'eta6'])
            else:
                R = PolynomialRing(K, ['e4', 'e6'])
            self.__taylor_exp_coeff_ring = R
            return R

    def _taylor_exp_ring(self):
        try:
            return self.__taylor_exp_ring
        except AttributeError:
            rank = self.complex_gram_matrix().nrows()
            r = self._taylor_exp_coeff_ring()
            if rank == 1:
                rz = PowerSeriesRing(r, 'z')
            else:
                rz = PowerSeriesRing(r, ['z%s' % (n + 1) for n in range(rank)])
            self.__taylor_exp_ring = rz
            return rz

    def _taylor_exp_ring_extended(self):
        try:
            return self.__taylor_exp_ring_extended
        except AttributeError:
            r = self._taylor_exp_coeff_ring()
            dK = self._discriminant()
            if dK == -3:
                e6, eta8 = r.gens()
                r1, e4 = PolynomialRing(r, 'e4').objgen()
                I = r1.ideal(e4**3 - e6**2 - (12 * eta8) ** 3)
                r1 = r1.quotient_ring(I, names='e4')
                self._extra_var = r1(e4)
            elif dK == -4:
                e4, eta6 = r.gens()
                r1, e6 = PolynomialRing(r, 'e6').objgen()
                I = r1.ideal(e6**2 - e4**3 - 1728 * (eta6 ** 4))
                r1 = r1.quotient_ring(I, names='e6')
                self._extra_var = r1(e6)
            else:
                e4, e6 = r.gens()
                r1, eta12 = PolynomialRing(r, 'eta12').objgen()
                I = r1.ideal(eta12**2 - (e4**3 - e6**2) / 1728)
                r1 = r1.quotient_ring(I, names='eta12')
                self._extra_var = r1(eta12)
            rank = self.complex_gram_matrix().nrows()
            if rank == 1:
                rz = PowerSeriesRing(r1, 'z')
            else:
                rz = PowerSeriesRing(r1, ['z%s' % (n + 1) for n in range(rank)])
            self.__taylor_exp_ring_extended = rz
            return rz

    #### e4, e6

    def _e4(self):
        d = self._discriminant()
        r = self._taylor_exp_ring_extended()
        if d == -3:
            return r.base_ring().gens()[0]
        else:
            return r.base_ring().base_ring().gens()[0]

    def _e6(self):
        d = self._discriminant()
        r = self._taylor_exp_ring_extended()
        if d == -4:
            return r.base_ring().gens()[0]
        else:
            return r.base_ring().base_ring().gens()[bool(d + 3)]

    def _eta_p(self):
        d = self._discriminant()
        r = self._taylor_exp_ring_extended()
        if d < -4:
            return r.base_ring().gens()[0]
        else:
            return r.base_ring().base_ring().gens()[1]

    def _delta(self):
        d = self._discriminant()
        eta_p = self._eta_p()
        if d == -3:
            return eta_p ** 3
        elif d == -4:
            return eta_p ** 4
        return eta_p ** 2

    #### Eisenstein series

    def eisenstein_series(self, k, prec):
        w = self.weilrep()
        d = self._discriminant()
        if k == 2 or k % 2 or (d == -3 and k % 6) or (d == -4 and k % 4):
            raise ValueError('Invalid weight %s in Eisenstein series for U(n, 1)' % k)
        prec0 = ceil(prec / 6) ** 2
        rank = self.complex_gram_matrix().nrows()
        f = w.eisenstein_series(k - rank, prec0)
        c = eisenstein_series_qexp(k, 1)[0]
        return f.theta_lift().reduce_precision(prec) / c

    #### theta lifts

    def lifts_basis(self, k, prec, cusp_forms=True):
        w = self.weilrep()
        d = self._discriminant()
        if (d == -3 and k % 3) or (d == -4 and k % 2):
            raise ValueError('Invalid weight %s in theta lift to U(n, 1)' % k)
        prec0 = ceil(prec / 6) ** 2
        rank = self.complex_gram_matrix().nrows()
        if cusp_forms:
            X = w.cusp_forms_basis(k - rank, prec0)
        else:
            X = w.modular_forms_basis(k - rank, prec0)
        return [x.theta_lift().reduce_precision(prec) for x in X]

    #### products

    def _borcherds_product_polyhedron(self, pole_order, prec, verbose=False):
        r"""
        Construct a polyhedron representing a cone of Heegner divisors. For internal use in the methods borcherds_input_basis() and borcherds_input_Qbasis().
        INPUT:
        - ``pole_order`` -- pole order
        - ``prec`` -- precision
        OUTPUT: a tuple consisting of an integral matrix M, a Polyhedron p, and a WeilRepModularFormsBasis X
        """
        K = self.weilrep().base_field()
        O_K = K.maximal_order()
        S = self.gram_matrix()
        wt = self.input_wt()
        w = self.weilrep()
        rds = w.rds()
        norm_dict = w.norm_dict()
        X = w.nearly_holomorphic_modular_forms_basis(wt, pole_order, prec, verbose=verbose)
        N = len([g for g in rds if not norm_dict[tuple(g)]])
        v_list = w.coefficient_vector_exponents(0, 1, starting_from=-pole_order, include_vectors=True)
        exp_list = [v[1] for v in v_list]
        d = w._ds_to_hds()
        v_list = [vector(d[tuple(v[0])]) for v in v_list]
        d = K.discriminant()
        positive = []
        zero = vector([0] * (len(exp_list) + 1))
        M = Matrix([x.coefficient_vector(starting_from=-pole_order, ending_with=0)[:-N] for x in X])
        vs = M.transpose().kernel().basis()
        prec = floor(min(exp_list) / max(filter(bool, exp_list)))
        norm_list = w._norm_form().short_vector_list_up_to_length(prec + 1)
        units = w._units()
        _w = w._w()
        norm_list = [[a + b * _w for a, b in x] for x in norm_list]
        excluded_list = set()
        if d >= -4:
            if d == -4:
                f = w.multiplication_by_i()
                f2 = None
            else:
                f = w.multiplication_by_zeta()
                f2 = f * f
        ys = []
        mult = len(units) // 2
        for i, n in enumerate(exp_list):
            if i not in excluded_list:
                ieq = copy(zero)
                ieq[i+1] = 1
                v1 = v_list[i]
                if d >= -4:
                    v2 = f(v1)
                    j = next(j for j, x in enumerate(v_list) if exp_list[i] == exp_list[j] and (all(t in O_K for t in x - v2) or all(t in O_K for t in x + v2)))
                    ieq[j+1] += 1
                    if i != j:
                        excluded_list.add(j)
                        y = copy(zero)
                        y[i+1] = 1
                        y[j+1] = -1
                        ys.append(y)
                    if f2 is not None:
                        v2 = f2(v1)
                        j = next(j for j, x in enumerate(v_list) if exp_list[i] == exp_list[j] and (all(t in O_K for t in x - v2) or all(t in O_K for t in x + v2)))
                        ieq[j + 1] += 1
                        if i != j:
                            excluded_list.add(j)
                            y = copy(zero)
                            y[i + 1] = 1
                            y[j + 1] = -1
                            ys.append(y)
                for j, m in enumerate(exp_list[:i]):
                    if j not in excluded_list:
                        N = m / n
                        if N in ZZ and N > 1:
                            v2 = v_list[j]
                            ieq[j + 1] = mult * any(all(t in O_K for t in x * v1 + u * v2) for x in norm_list[N] for u in units)
                positive.append(ieq)# * denominator(ieq)
        p = Polyhedron(ieqs=positive, eqns=[vector([0] + list(v)) for v in vs] + ys)
        return M, p, X

    def borcherds_input_basis(self, pole_order, prec, verbose=False):
        r"""
        Compute a basis of input functions into the Borcherds lift with pole order in infinity up to pole_order.

        This method computes a list of Borcherds lift inputs F_0, ..., F_d which is a Q-basis in the following sense: it is minimal with the property that every modular form with pole order at most pole_order whose Borcherds lift is holomorphic can be expressed in the form (k_0 F_0 + ... + k_d F_d) where k_i are nonnegative rational numbers.

        INPUT:
        - ``pole_order`` -- positive number (does not need to be an integer)
        - ``prec`` -- precision of the output

        OUTPUT: WeilRepModularFormsBasis
        """
        S = self.gram_matrix()
        w = self.weilrep()
        K = w.base_field()
        d = K.discriminant()
        wt = self.input_wt()
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose=verbose)
        try:
            b = Matrix(Cone(p).Hilbert_basis())
            if verbose:
                print('I will now try to find Borcherds product inputs.')
            try:
                u = M.solve_left(b)
                Y = [v * X for v in u.rows()]
                Y = WeilRepModularFormsBasis(wt, Y, w)
            except ValueError:
                Y = WeilRepModularFormsBasis(wt, [], w)
        except IndexError:
            Y = WeilRepModularFormsBasis(wt, [], w)
        if wt >= 0:
            X = deepcopy(w.basis1_vanishing_to_order(wt, max(0, -pole_order), prec))
            if X:
                X.extend(Y)
                Y = X
        X = Y._WeilRepModularFormsBasis__basis
        if d >= -4:
            if d == -4:
                f = w.multiplication_by_i()
                n = 2
            else:
                f = w.multiplication_by_zeta()
                f2 = f * f
                n = 3
            for i, x in enumerate(X):
                try:
                    j = X.index(f(x))
                    if j == i:
                        X[i] = x / n
                    else:
                        del X[j]
                except ValueError:
                    pass
                if n == 3:
                    try:
                        j2 = X.index(f2(x))
                        del X[j2]
                    except ValueError:
                        pass
        X.sort(key=lambda x: x.fourier_expansion()[0][2][0])
        return WeilRepModularFormsBasis(wt, X, w)

    def borcherds_input_Qbasis(self, pole_order, prec, verbose=False):
        r"""
        Compute a Q-basis of input functions into the Borcherds lift with pole order in infinity up to pole_order.

        This method computes a list of Borcherds lift inputs F_0, ..., F_d which is a Q-basis in the following sense: it is minimal with the property that every modular form with pole order at most pole_order whose Borcherds lift is holomorphic can be expressed in the form (k_0 F_0 + ... + k_d F_d) where k_i are nonnegative rational numbers.

        INPUT:
        - ``pole_order`` -- positive number (does not need to be an integer)
        - ``prec`` -- precision of the output

        OUTPUT: WeilRepModularFormsBasis
        """
        S = self.gram_matrix()
        w = self.weilrep()
        K = w.base_field()
        d = K.discriminant()
        wt = self.input_wt()
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose=verbose)
        try:
            b = Matrix(Cone(p).rays())
            if verbose:
                print('I will now try to find Borcherds product inputs.')
            try:
                u = M.solve_left(b)
                Y = [v * X for v in u.rows()]
                Y = WeilRepModularFormsBasis(wt, Y, w)
            except ValueError:
                Y = WeilRepModularFormsBasis(wt, [], w)
        except IndexError:
            Y = WeilRepModularFormsBasis(wt, [], w)
        if wt >= 0:
            X = deepcopy(w.basis_vanishing_to_order(wt, max(0, -pole_order), prec))
            if X:
                X.extend(Y)
                Y = X
        X = Y._WeilRepModularFormsBasis__basis
        if d >= -4:
            if d == -4:
                f = w.multiplication_by_i()
                n = 2
            else:
                f = w.multiplication_by_zeta()
                f2 = f * f
                n = 3
            for i, x in enumerate(X):
                try:
                    j = X.index(f(x))
                    if j == i:
                        X[i] = x / n
                    else:
                        del X[j]
                except ValueError:
                    pass
                if n == 3:
                    try:
                        j2 = X.index(f2(x))
                        del X[j2]
                    except ValueError:
                        pass
        X.sort(key=lambda x: x.fourier_expansion()[0][2][0])
        return WeilRepModularFormsBasis(wt, X, w)


class HermitianRescaledHyperbolicPlane(HermitianWeilRep):
    r"""
    Rescaled hyperbolic planes over O_K.

    This should be called with II(n) where n \in O_K.
    """
    def __init__(self, N, K=None, gen=None):
        if K is None:
            K = N.parent()
            K = FractionField(K)
        if K:
            g = K.gen()
            try:
                gn = g.norm()
                Kd = K.discriminant()
            except AttributeError:
                gn = g * g
                Kd = 1
            if gn != abs(Kd):
                g *= 2
            a = N / g
            try:
                S = Matrix(K, [[0, a], [a.galois_conjugate(), 0]])
            except AttributeError:
                S = Matrix(K, [[0, a], [a, 0]])
            super().__init__(S, gen=gen, plus_H=True)
            self.__class__ = HermitianRescaledHyperbolicPlane
        self.__N = N

    def __add__(self, other):
        S = other.complex_gram_matrix()
        N = self.__N
        K = S.base_ring()
        g = K.gen()
        if g.norm() != abs(K.discriminant()):
            g *= 2
        a = N / g
        n = S.nrows()
        A = Matrix(K, n + 2)
        for i in range(n):
            for j in range(n):
                A[i + 1, j + 1] = S[i, j]
        A[0, -1] = a
        A[-1, 0] = a.galois_conjugate()
        plus_H = other.is_positive_definite()
        w = HermitianWeilRep(A, gen=other._w(), plus_H=plus_H)
        w._lorentz_gram_matrix = lambda: Matrix(QQ, n + n + 2) #leave empty
        return w
    __radd__ = __add__


class UnitaryModularForm:
    r"""
    This class represents modular forms on U(n, 1).
    """

    def __init__(self, complex_gram_matrix, taylor_series, weight, umf=None):
        self.__taylor = taylor_series
        self.__complex_gram_matrix = complex_gram_matrix
        self.__rank = complex_gram_matrix.rank()
        self.__weight = weight
        if umf is None:
            self.__umf = UnitaryModularForms(complex_gram_matrix)
        else:
            self.__umf = umf

    def __repr__(self):
        return str(self.taylor_series())

    def taylor_series(self):
        return self.__taylor
    power_series = taylor_series

    def complex_gram_matrix(self):
        return self.__complex_gram_matrix

    def umf(self):
        return self.__umf

    def precision(self):
        try:
            return self.__precision
        except AttributeError:
            prec = self.__taylor.prec()
            self.__precision = prec
            return prec
    prec = precision

    def weight(self):
        return self.__weight

    def _character(self):
        try:
            return self.__character
        except AttributeError:
            f = self.taylor_series()
            try:
                p = f.dict()[f.trailing_monomial().exponents()[0]]
            except AttributeError:
                p = f[f.valuation()]
            j = 0
            r = p.base_ring()
            if isinstance(r, Field):
                self.__character = 0
                self._mod_taylor = f
                return self.__character
            j = 0
            found_char = False
            umf = self.umf()
            _ = umf._taylor_exp_ring_extended()
            x = umf._extra_var
            while p not in r:
                p = p * x
                j += 1
                if j > 3:
                    raise RuntimeError
            self.__character = j
            self._mod_taylor = f.map_coefficients(lambda y: r(y * (x**j)))
            return self.__character

    ## coefficients

    def __getitem__(self, n):
        return self.taylor_series().__getitem__(n)

    def coefficient_vector(self):
        pass

    ## modify

    def reduce_precision(self, new_prec):
        r"""
        Reduce the precision of self's Taylor expansion.
        """
        f = self.taylor_series()
        f = f.add_bigoh(min(f.prec(), new_prec))
        return UnitaryModularForm(self.complex_gram_matrix(), f, self.weight(), umf=self.umf())

    ## arithmetic

    def __add__(self, other):
        if not other:
            return self
        if self.weight() != other.weight():
            raise ValueError('Incompatible weights')
        elif self.complex_gram_matrix() != other.complex_gram_matrix():
            raise ValueError('Incompatible Gram matrices')
        return UnitaryModularForm(self.complex_gram_matrix(), self.taylor_series() + other.taylor_series(), self.weight())
    __radd__ = __add__

    def __div__(self, other):
        f = self.taylor_series()
        S = self.complex_gram_matrix()
        if isinstance(other, UnitaryModularForm):
            if S != other.complex_gram_matrix():
                raise ValueError('Incompatible modular forms')
            n = S.nrows()
            g = other.taylor_series()
            if n > 1:
                u = g.trailing_monomial()
                p = g.coefficients()[u]
                return UnitaryModularForm(S, (f / (p*u)) / (g / (p*u)), self.weight() - other.weight())
            u = g.valuation()
            try:
                p = g[u]
            except IndexError:
                raise ValueError('Insufficient precision') from None
            return UnitaryModularForm(S, (f/p) / (g/p), self.weight() - other.weight())
        return UnitaryModularForm(S, f / other, self.weight())
    __truediv__ = __div__

    def __mul__(self, other):
        try:
            if self.complex_gram_matrix() != other.complex_gram_matrix():
                raise ValueError('Incompatible Gram matrices')
            return UnitaryModularForm(self.complex_gram_matrix(), self.taylor_series() * other.taylor_series(), self.weight() + other.weight())
        except AttributeError:
            return UnitaryModularForm(self.complex_gram_matrix(), self.taylor_series() * other, self.weight())
    __rmul__ = __mul__

    def __neg__(self):
        return UnitaryModularForm(self.complex_gram_matrix(), -self.taylor_series(), self.weight())

    def __pow__(self, n):
        if n in ZZ:
            f = self.taylor_series()
            if n == 0:
                r = f.parent()
                return UnitaryModularForm(self.complex_gram_matrix(), r(1).add_bigoh(f.prec()), 0, umf=self.umf())
            elif n == 1:
                return self
            elif n > 1:
                n_half = ZZ(n) // 2
                return self.__pow__(n_half) * self.__pow__(n - n_half)
        elif n in QQ:
            a, b = n.as_integer_ratio()
            f = self.taylor_series()
            h = _root(f, b, self.weight(), self.umf())
            return UnitaryModularForm(self.complex_gram_matrix(), h, self.weight() / ZZ(b), umf=self.umf()) ** a
        return NotImplemented

    def __sub__(self, other):
        if self.weight() != other.weight():
            raise ValueError('Incompatible weights')
        elif self.complex_gram_matrix() != other.complex_gram_matrix():
            raise ValueError('Incompatible Gram matrices')
        return UnitaryModularForm(self.complex_gram_matrix(), self.taylor_series() - other.taylor_series(), self.weight())

    ## other

    def hessian(self):
        r"""
        Compute the modular Hessian determinant of this modular form.

        Given a modular form "f" of weight k for a subgroup of U(n, 1), its modular Hessian Hf is a modular form
        for the same group (with a character) of weight (n+1)*(k+2).
        """
        umf = self.umf()
        k = self.weight()
        e4, e6, eta_p = umf._e4(), umf._e6(), umf._eta_p()
        r0 = e4.parent()
        d = umf._discriminant()
        if d == -3:
            e4 = e4.lift()
            r = e4.parent()
        elif d == -4:
            e6 = e6.lift()
            r = e6.parent()
        else:
            eta_p = eta_p.lift()
            r = eta_p.parent()
        rz = umf._taylor_exp_ring_extended()
        z = rz.gens()
        r1, z = PowerSeriesRing(r, z).objgen()
        serre_deriv = -ZZ(1)/3 * r(e6) * r.derivation(e4) - ZZ(1)/2 * r(e4) * r(e4) * r.derivation(e6)
        h = r1(self.taylor_series())
        try:
            h = h.map_coefficients(lambda x: x.lift())
        except (AttributeError, TypeError):
            pass
        h_tau = h.map_coefficients(serre_deriv)
        N = self.complex_gram_matrix().rank() + 1
        if N > 2:
            h_grad = [h.derivative(a) for a in z]
            h_double_tau = h_tau.map_coefficients(serre_deriv) - (ZZ(1)/144) * r(e4) * (k * h + sum(a * h_grad[a] for a in z))
        else:
            h_grad = [h.derivative()]
            h_double_tau = h_tau.map_coefficients(serre_deriv) - (ZZ(1)/144) * r(e4) * (k * h + z * h_grad[0])
        h_grad_tau = [f.map_coefficients(serre_deriv) for f in h_grad]
        if N > 2:
            h_hess = [[f.derivative(a) for a in z[i:]] for i, f in enumerate(h_grad)]
        else:
            h_hess = [[h_grad[0].derivative()]]
        M = Matrix(rz, N + 1)
        M[0, 0] = k * (k+1) * rz(h)
        M[0, 1] = (k+1) * rz(h_tau)
        M[1, 0] = M[0, 1]
        M[1, 1] = rz(h_double_tau)
        for i in range(2, N+1):
            M[0, i] = (k+1) * rz(h_grad[i - 2])
            M[i, 0] = M[0, i]
            M[1, i] = rz(h_grad_tau[i - 2])
            M[i, 1] = M[1, i]
            for j in range(i, N+1):
                M[i, j] = rz(h_hess[i-2][j-i])
                M[j, i] = M[i, j]
        d = M.determinant()
        return UnitaryModularForm(self.complex_gram_matrix(), d, (k+2)*(N+1), umf=umf)


def _root(h, n, k, umf):
    r = h.parent()
    r1 = r.base_ring()
    z = r.gens()
    d = h.dict()
    if r.ngens() == 1:
        d = {(a,): b for a, b in d.items()}
    L = sorted(d, key=sum)
    a = L[0]
    try:
        b_pow = d[a].lift()
    except (AttributeError, TypeError):
        b_pow = d[a]
    a = vector(a)
    if len(L) > 1 and sum(L[1]) == sum(a):
        raise RuntimeError
    prefactor = prod(z[i]**ZZ(y / n) for i, y in enumerate(a))
    try:
        b = b_pow.nth_root(ZZ(n))
    except ValueError:
        discr = umf._discriminant()
        b = _hard_root(b_pow, discr, n, k + h.valuation())
    try:
        h1 = sum(r1(x.lift() / b_pow) * prod(z[i]**y for i, y in enumerate(vector(c) - a)) for c, x in d.items()).add_bigoh(h.prec() - sum(a))
    except (AttributeError, TypeError):
        try:
            h1 = sum(r1(x / b_pow) * prod(z[i]**y for i, y in enumerate(vector(c) - a)) for c, x in d.items()).add_bigoh(h.prec() - sum(a))
        except TypeError:
            raise ValueError('Not an nth power') from None
    return prefactor * r1(b) * h1**(1/n)


def _hard_root(p, discr, n, k):
    r"""
    Compute the nth root more carefully.
    """
    prec = ZZ(k) // 12 + 1
    E4 = smf_eisenstein_series(4, prec)
    E6 = smf_eisenstein_series(6, prec)
    Eta = smf_eta(prec)
    r = p.parent()
    phi = FlatteningMorphism(r)
    p = phi(p)
    r1 = p.parent()
    if discr == -3:
        Eta8 = Eta ** 8
        try:
            f = p(E6, Eta8, E4)
            e6, eta_N, e4 = r1.gens()
        except TypeError:
            f = p(E6, Eta8)
            e6, eta_N = r1.gens()
        N = 8
    elif discr == -4:
        Eta6 = Eta**6
        try:
            f = p(E4, Eta6, E6)
            e4, eta_N, e6 = r1.gens()
        except TypeError:
            f = p(E4, Eta6)
            e4, eta_N = r1.gens()
        N = 6
    else:
        try:
            f = p(E4, E6, Eta**12)
            e4, e6, eta_N = r1.gens()
        except TypeError:
            f = p(E4, E6)
            e4, e6 = r1.gens()
        N = 12
    assert f.weight() == k
    val = f.valuation(exact=True)
    h = f / Eta**ZZ(24 * val)
    k1 = ZZ(h.weight())
    k1n = ZZ(k1 / n)
    h_root = smf(k1n, h.qexp()**(ZZ(1)/n))
    X = []
    Y = []
    for a in range(k1n // 6 + 1):
        b = (k1n - 6 * a)
        if b % 4 == 0:
            X.append(E4**ZZ(b / 4) * E6**a)
            Y.append(e4**ZZ(b/4) * e6**a)
    V = relations([h_root] + X).basis()
    if len(V) > 1:
        raise RuntimeError
    elif not V:
        raise ValueError('Not an nth power')
    v = V[0]
    return phi.section()(eta_N**ZZ(24 * val / (n*N)) * sum(v[i + 1] * y for i, y in enumerate(Y)))


def unitary_jacobian(X):
    r"""
    Compute the Jacobian of unitary modular forms.

    This computes the Jacobian of a family of (n+1) modular forms on U(n,1).
    If the forms have weights k0, k1, ..., kn then their Jacobian has weight k0+k1+...+kn+(n+1).

    NOTE: should be called using the method "jacobian()" from lifts.py
    """
    Xref = X[0]
    S = Xref.complex_gram_matrix()
    n = S.nrows()
    if any(x.complex_gram_matrix() != S for x in X[1:]):
        raise ValueError('Incompatible modular forms')
    umf = Xref.umf()
    e4, e6, eta_p = umf._e4(), umf._e6(), umf._eta_p()
    r0 = e4.parent()
    d = umf._discriminant()
    if d == -3:
        e4 = e4.lift()
        r = e4.parent()
    elif d == -4:
        e6 = e6.lift()
        r = e6.parent()
    else:
        eta_p = eta_p.lift()
        r = eta_p.parent()
    rz = umf._taylor_exp_ring_extended()
    z = rz.gens()
    r1, z = PowerSeriesRing(r, z).objgen()
    serre_deriv = -ZZ(1)/3 * r(e6) * r.derivation(e4) - ZZ(1)/2 * r(e4) * r(e4) * r.derivation(e6)
    F = [r1(x.taylor_series()) for x in X]
    K = [x.weight() for x in X]
    Ftau = [f.map_coefficients(serre_deriv) for f in F]
    if n > 1:
        Fz = [[f.derivative(a) for f in F] for a in z]
    else:
        Fz = [[f.derivative() for f in F]]
    M = Matrix([[k*F[i] for i, k in enumerate(K)], Ftau] + Fz)
    h = M.determinant()
    return UnitaryModularForm(S, h, sum(K) + n + 2, umf=umf)


def _umf_relations(X):
    prec = min(x.precision() for x in X)
    X = [x.reduce_precision(prec) for x in X]
    characters = [x._character() for x in X]
    weights = [x.weight() for x in X]
    if len(set(characters)) != 1:
        raise ValueError('Incompatible characters')
    if len(set(weights)) != 1:
        raise ValueError('Incompatible weights')
    power_series = [x._mod_taylor for x in X]
    power_series_dicts = [x.dict() for x in power_series]
    keys = set().union(*power_series_dicts)
    key_keys = {}
    for key in keys:
        h = []
        for y in power_series_dicts:
            try:
                h.append(y[key].dict())
            except KeyError:
                pass
        key_keys[key] = set().union(*h)
    M = []
    for d in power_series_dicts:
        v = []
        for key in keys:
            try:
                d_key = d[key]
                for key in key_keys[key]:
                    try:
                        v.append(d_key[key])
                    except KeyError:
                        v.append(0)
            except KeyError:
                v.extend(0 for _ in key_keys[key])
        M.append(v)
    return Matrix(M).kernel()
