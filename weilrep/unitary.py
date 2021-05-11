r"""

Theta lifts and Borcherds products on unitary groups U(n, 1)

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2021 Brandon Williams
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
from sage.functions.other import floor, frac, sqrt
from sage.geometry.cone import Cone
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import Matrix
from sage.matrix.special import block_diagonal_matrix
from sage.misc.functional import denominator
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ

from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .lorentz import OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian, RescaledHyperbolicPlane
from .morphisms import WeilRepAutomorphism
from .positive_definite import OrthogonalModularFormPositiveDefinite, OrthogonalModularFormsPositiveDefinite
from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularFormPrincipalPart, WeilRepModularFormsBasis

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
            gen = (d + K.gen() * (2 - (d % 4))) / 2
            ell = d
            m = d / (2 - (d % 4))
            u = d * (d - 1) // 2
        else:
            ell, m = gen.parts()
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
            return HermitianWeilRep(block_diagonal_matrix([self.complex_gram_matrix(), other.complex_gram_matrix()], subdivide = False), gen = self.__w)
        elif isinstance(other, RescaledHyperbolicPlane):
            return HermitianRescaledHyperbolicPlane(other._N()).__add__(self)
        return NotImplemented

    def __call__(self, N):
        r"""
        Rescale by integers.
        """
        try:
            N = Integer(N)
            return HermitianWeilRep(N * self.complex_gram_matrix(), gen = self.__w)
        except TypeError:
            return super().__call__(N)

    def dual(self):
        r"""
        Return the dual Weil representation.
        """
        try:
            return self.__dual
        except AttributeError:
            s = HermitianWeilRep(-self.complex_gram_matrix(), gen = self.__w)
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
        return {tuple(g):tuple(ds[i]) for i, g in enumerate(hds)}

    def _ds_to_hds(self):
        hds = self.hds()
        ds = self.ds()
        return {tuple(g):tuple(hds[i]) for i, g in enumerate(ds)}

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
            self.__h_norm_dict = {tuple(g):n[i] for i, g in enumerate(ds)}
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
        return 'Unitary modular forms associated to the gram matrix\n%s%s\nwith coefficients in %s'%(S, s, S.base_ring())

    def nvars(self):
        n = self.complex_gram_matrix().nrows()
        s = self._plusH
        return Integer(n + n + 2 - 4*s)

    def complex_gram_matrix(self):
        return self.weilrep().complex_gram_matrix()

    def _borcherds_product_polyhedron(self, pole_order, prec, verbose = False):
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
        X = w.nearly_holomorphic_modular_forms_basis(wt, pole_order, prec, verbose = verbose)
        N = len([g for g in rds if not norm_dict[tuple(g)]])
        v_list = w.coefficient_vector_exponents(0, 1, starting_from = -pole_order, include_vectors = True)
        exp_list = [v[1] for v in v_list]
        d = w._ds_to_hds()
        v_list = [vector(d[tuple(v[0])]) for v in v_list]
        d = K.discriminant()
        positive = []
        zero = vector([0] * (len(exp_list) + 1))
        M = Matrix([x.coefficient_vector(starting_from = -pole_order, ending_with = 0)[:-N] for x in X])
        vs = M.transpose().kernel().basis()
        prec = floor(min(exp_list) / max(filter(bool, exp_list)))
        norm_list = w._norm_form().short_vector_list_up_to_length(prec + 1, up_to_sign_flag = True)
        units = w._units()
        _w = w._w()
        norm_list = [[a + b * _w for a, b in x] for x in norm_list]
        excluded_list = set([])
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
                            #ieq[j + 1] = mult * any(all(t in O_K for t in x * v1 + u * v2) for x in norm_list[N] for u in units)
                            ieq[j + 1] = mult * len([any(all(t in O_K for t in x * v1 + u * v2) for u in units) for x in norm_list[N]])
                            #if not N.is_square():
                            #    ieq[j+1] *= 2
                positive.append(ieq)
        p = Polyhedron(ieqs = positive, eqns = [vector([0] + list(v)) for v in vs] + ys)
        return M, p, X

    def borcherds_input_basis(self, pole_order, prec, verbose = False):
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
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose = verbose)
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
        X.sort(key = lambda x: x.fourier_expansion()[0][2][0])
        return WeilRepModularFormsBasis(wt, X, w)

    def borcherds_input_Qbasis(self, pole_order, prec, verbose = False):
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
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose = verbose)
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
        X.sort(key = lambda x: x.fourier_expansion()[0][2][0])
        return WeilRepModularFormsBasis(wt, X, w)

class HermitianRescaledHyperbolicPlane(HermitianWeilRep):
    r"""
    Rescaled hyperbolic planes over O_K.

    This should be called with II(n) where n \in O_K.
    """
    def __init__(self, N, K = None, gen = None):
        if K:
            a = N / K.gen()
            S = Matrix(K, [[0, a], [a.galois_conjugate(), 0]])
            super().__init__(S, gen = gen, plus_H = True)
            self.__class__ = UnitaryRescaledHyperbolicPlane
        self.__N = N

    def __add__(self, other):
        S = other.complex_gram_matrix()
        N = self.__N
        K = S.base_ring()
        a = N / K.gen()
        n = S.nrows()
        A = Matrix(K, n + 2)
        for i in range(n):
            for j in range(n):
                A[i + 1, j+ 1] = S[i, j]
        A[0, -1] = a
        A[-1, 0] = a.galois_conjugate()
        plus_H = other.is_positive_definite()
        w = HermitianWeilRep(A, gen = other._w(), plus_H = plus_H)
        w._lorentz_gram_matrix = lambda: Matrix(QQ, n + n + 2) #leave empty
        return w
    __radd__ = __add__

class UnitaryModularForm(OrthogonalModularFormPositiveDefinite):
    r"""
    WARNING: this is unfinished! Use at your own risk.

    Modular forms on unitary groups U(n, 1).

    We try to represent modular forms which are obtained from the orthogonal group O(2n, 2) by restriction as a Fourier series:

    F(\tau, z_1,..., z_{n - 1}) = \sum c(m_0, ..., m_{n-1}, m_n) q^(m_0) \zeta_1^(m_1) ... \zeta_{n-1}^(m_{n-1}) s^(m_n)
    The variable 's' should actually be set to a CM point on the upper half-plane.

    See Theorem 8.1 of [Hofmann] for Fourier expansions of this kind that represent Borcherds products.

    General modular forms on U(n, 1) do not seem to have an expansion of this type in a meaningful sense.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__ = UnitaryModularForm
        self.__base_ring = self.weilrep().complex_gram_matrix().base_ring()
        self.__omega = self.weilrep()._w()

    def __repr__(self):
        try:
            return self.__string
        except AttributeError:
            def a(v, x):
                if x:
                    if x != 1:
                        if x in ZZ:
                            return v + '^%s'%x
                        return v + '^(%s)'%x
                    return v
                return ''
            def f(x):
                x, y = x
                b = ''
                if y:
                    if y != 1:
                        b = str(y) + '*'
                    qp, sp, rp = x[0], x[-1], x[1:-1]
                    if len(rp) > 1:
                        u = '*'.join(filter(bool, [a('q', qp), '*'.join([a('r_%d'%i, z) for i, z in enumerate(rp)]), a('s', sp)]))
                    elif rp:
                        u = '*'.join(filter(bool, [a('q', qp), a('r', rp[0]), a('s', sp)]))
                    else:
                        u = '*'.join(filter(bool, [a('q', qp), a('s', sp)]))
                    if u:
                        return b + u
                    return str(y)
                return ''
            d = self.coefficients()
            s = ' + '.join(filter(bool, map(f, d.items())))
            t = a('O(q, s)', self.precision())
            if s:
                t = s + ' + ' + t
            t = t.replace('+ -', '- ')
            self.__string = t
            return t

    def coefficients(self):
        def f():
            return 0
        d = defaultdict(f, {})
        scale = self.scale()
        f = self.fourier_expansion()
        omega = self.weilrep()._w()
        a, b = 2, omega + omega
        N = (self.nvars() - 2) // 2
        S_inv = self.weilrep().gram_matrix().inverse()
        for (qp, sp), p in f.dict().items():
            qp, sp = Integer(qp), Integer(sp)
            for v, x in p.dict().items():
                v = vector(v) * S_inv
                v = vector([a*v[i + i] + b*v[i + i + 1] for i in range(N)])
                w = tuple([qp / scale] + list(v / scale) + [sp / scale])
                d[w] += x
        return d

    def complex_gram_matrix(self):
        return self.weilrep().complex_gram_matrix()