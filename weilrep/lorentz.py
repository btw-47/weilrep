r"""

Additive and multiplicative theta lifts for lattices split
by a rational hyperbolic plane

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2020-2025 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import cypari2

import math

from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularForm

from re import sub

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli, GCD, is_prime, is_square, kronecker_symbol, prime_divisors, XGCD
from sage.arith.srange import srange
from sage.combinat.combinat import bernoulli_polynomial
from sage.functions.gamma import gamma
from sage.functions.log import exp, log
from sage.functions.other import ceil, factorial, floor, frac
from sage.functions.transcendental import zeta
from sage.matrix.constructor import matrix
from sage.matrix.special import block_diagonal_matrix, identity_matrix
from sage.misc.functional import denominator, isqrt
from sage.misc.misc_c import prod
from sage.modular.arithgroup.congroup_gamma0 import Gamma0_constructor
from sage.modular.modform.constructor import ModularForms
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.big_oh import O
from sage.rings.fraction_field import FractionField
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring import polygen
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi

from .eisenstein_series import quadratic_L_function__correct, twoadic_classify, twoadic_jordan_blocks, twonf_with_change_vars
from .lifts import OrthogonalModularForm, OrthogonalModularForms

sage_one_half = Integer(1) / 2
sage_three_half = Integer(3) / 2

pari = cypari2.Pari()
PariError = cypari2.PariError


class OrthogonalModularFormsLorentzian(OrthogonalModularForms):
    r"""
    This class represents orthogonal modular forms for a Lorentzian lattice.

    (If the lattice is L and H is the hyperbolic plane then these are automorphic forms
    on the symmetric domain for L+H.)

    Can be called simply with OrthogonalModularForms(w), where
    - ``w`` -- a WeilRep instance, or a Gram matrix

    WARNING: the Gram matrix must have a strictly negative upper-left-most entry!
    We always choose (1, 0, ...) as a basis of the negative cone of our lattice!
    """

    def input_wt(self):
        return 1 - self.nvars() / 2

    def nvars(self):
        return Integer(self.weilrep()._lorentz_gram_matrix().nrows())

    def volume(self):
        r"""
        Compute the Hirzebruch--Mumford volume of the orthogonal group.
        """
        rank = self.nvars() + 2
        w = self.weilrep()
        if w.is_lorentzian():
            q = (w.dual() + II(1)).quadratic_form()
            g = 1
        else:
            q = w.dual().quadratic_form()
            a, k = w.dual().genus()._proper_spinor_kernel()
            g = ZZ(len(a)) / ZZ(len(k))
        d = w.discriminant()
        if rank % 2:
            u = prod(zeta(2*i) for i in srange(1, (rank + 1) // 2))
        else:
            if rank % 4:
                d = -d
            #u = (~quadratic_L_function__correct(rank // 2, d) * prod(zeta(2*i) for i in srange(1, rank // 2 + 1)))
            u = (quadratic_L_function__correct(rank // 2, d) * prod(zeta(2*i) for i in srange(1, rank // 2)))
        alpha = 1

        def _P(p, m):
            p = ZZ(p)
            return prod((1 - p**(-2*i)) for i in srange(1, m + 1))

        zero = QuadraticForm(matrix([]))
        for p in prime_divisors(2 * d):
            if p != 2:
                x = q.jordan_blocks_by_scale_and_unimodular(p)
            else:
                x = twoadic_jordan_blocks(twonf_with_change_vars(q.matrix())[0])
            x0 = [y[0] for y in x]
            y = []
            for j in range(x[-1][0] + 1):
                try:
                    i = x0.index(j)
                    y.append(x[i])
                except ValueError:
                    y.append((j, zero))
            x = y
            dim = []
            e = []
            P = []
            Q = []
            parity = []
            for j, ell in x:
                ell_dim = ell.dim()
                dim.append(ell_dim)
                if p == 2:
                    h = twoadic_classify(ell)
                    half_rank = h[1] + h[2]
                    parity.append(bool(len(h[0])))
                    P.append(_P(2, half_rank))
                    if len(h[0]) < 2 or (len(h[0]) == 2 and h[0][0] % 4 != h[0][1] % 4):
                        if (j == 0 or j > 0 and not parity[j - 1]) and (j == len(x)-1 or j < len(x)-1 and not twoadic_classify(x[j + 1][1])[0]):
                            chi = (-1) ** h[1]
                            e.append(((1 + chi * ZZ(2)**(-half_rank)) / 2)**(-1))
                        else:
                            e.append(2)
                    else:
                        e.append(2)
                    if j:
                        if parity[j - 1]:
                            Q.append(dim[-1] + parity[-1])
                else:
                    if ell_dim % 2 or not ell_dim:
                        e.append(1)
                    else:
                        chi = kronecker_symbol(ell.det() * (-1)**(ell_dim // 2), p)
                        e.append(1 / (1 + chi * p**(-ell_dim // 2)))
                    P.append(_P(p, ell_dim // 2))
            w = sum(j * n * ((n + 1) / 2 + sum(dim[k] for k in range(j+1, len(dim)))) for j, n in enumerate(dim))
            if p == 2 and parity[-1]:
                Q.append(ell_dim)
            if p == 2:
                N = rank - 1 - sum(Q)
            else:
                N = len([x for x in dim if x]) - 1
            alpha *= (prod(P) * prod(e) * ZZ(2)**N * p**(w) / _P(p, rank // 2))
            if not rank % 2:
                alpha *= (1 - p**(-rank)) / (1 - kronecker_symbol(d, p) * p**(-rank//2))
        return 2 * u * abs(d)**((rank + 1) / 2) * prod(pi**(-k/2) * gamma(k/2) for k in srange(1, rank+1)) / (g * alpha)

    def dimension_main_term(self):
        r"""
        Compute the asymptotic value of dim M_k(O^+(L)) as k becomes large.
        """
        k = polygen(QQ, 'k')
        vol = self.volume()
        n = self.nvars()
        return 2 * (2 - n % 2) * vol * k**n / factorial(n)


class OrthogonalModularFormLorentzian(OrthogonalModularForm):
    r"""
    This class represents modular forms on the type IV domain attached to a lattice of the form L + II_{1, 1}(N), where L is Lorentzian.
    """

    def __repr__(self):
        r"""
        Various string representations, depending on the parameter 'qexp_representation'
        """
        try:
            return self.__string
        except AttributeError:
            d = self.scale()
            try:
                h = self.fourier_expansion()
            except TypeError as t:
                if self.qexp_representation() == 'shimura':
                    h = self.true_fourier_expansion()
                    return str(h).replace('t', 'q')
                raise t

            def m(obj):
                m1, m2 = obj.span()
                obj_s = obj.string[m1:m2]
                x = obj_s[0]
                if x == '^':
                    u = Integer(obj_s[1:]) / d
                    if u.is_integer():
                        if u == 1:
                            return ''
                        return '^%d' % u
                    return '^(%s)' % u
                return (x, obj_s)[x == '_'] + '^(%s)' % (Integer(1)/d)
            _m = lambda s: sub(r'\^-?\d+|(?<!O\(|\, )(\_\d+|q|s|(?<!e)t|x)(?!\^)', m, s)
            if self.qexp_representation() == 'shimura':
                s = sub(r'(?<!e)t', 'q', str(h))  # two passes?
                if d == 1:
                    self.__string = s
                    return s
                else:
                    self.__string = _m(s)
                    return self.__string
            elif self.has_fourier_jacobi_representation():
                S = self.gram_matrix()
                f = self._q_s_expansion()
                v = self._q_s_valuation()
                if v:
                    qd = f.dict()

                    def c(x, a, b):
                        if not (a or b):
                            return str(x)
                        t = ''
                        u = ''
                        if x == -1:
                            t = '-'
                        elif x in QQ:
                            if x != 1:
                                t = str(x)+'*'
                        else:
                            t = '(%s)*' % x
                        if a:
                            u = 'q^%s' % a
                            if b:
                                u += '*s^%s' % b
                        elif b:
                            u = 's^%s' % b
                        u = u.replace('(', '').replace(')', '')
                        return t+u
                    s = ' + '.join(c(x, a+v, b+v) for (a, b), x in qd.items()).replace('+ -', '- ')
                else:
                    s = str(f)
                if not self._base_ring_is_laurent_polynomial_ring():  # represent 'r'-terms as Laurent polynomials if possible
                    n = self.nvars() - 2
                    r = LaurentPolynomialRing(QQ, [f'r_{i}' for i in range(n)])

                    def _a(obj):
                        obj_s = obj.string[slice(*obj.span())]
                        j = 1
                        if obj_s[:2] == '((':
                            obj_s = obj_s[1:]
                            j = 2
                        i = obj_s.index(')/')
                        return '('*j + str(r(obj_s[:(i+1)]) / r(obj_s[i+2:])) + ')'*j
                    s = sub(r'\([^()]*?\)\/((\((r_\d*(\^\d*)?\*?)+\))|(r_\d*(\^\d*)?\*?)+)', _a, s)
                if v:
                    s = s.replace('q^0*', '').replace('s^0', '').replace('* ', ' ')
                    s = s.replace('q^1', 'q').replace('s^1', 's')
                    s = s + ' + O(q, s)^%s' % (d * (self.precision() + v))
                self.__string = s
                d = self.__q_s_scale
                qs = len(f.parent().gens()) - 1
                if not qs:
                    self.__string = sub(r'O\(q.*\)', lambda x: 'O(q, s)%s' % x.string[slice(*x.span())][3:-1], self.__string)
                self.__string = self.__string.replace('((', '(')
                self.__string = self.__string.replace('))', ')')
                if d != 1:
                    self.__string = _m(self.__string)
                if self.nvars() == 3:
                    self.__string = self.__string.replace('r_0', 'r')
                return self.__string
            else:
                if d == 1:
                    self.__string = str(h)
                else:
                    self.__string = _m(str(h))
                return self.__string

    def fourier_expansion(self):
        r"""
        Return self's Fourier expansion, as a power series in 'q','s' if available, otherwise as a power series in 't','x'.
        """
        try:
            return self._q_s_expansion()
        except NotImplementedError:
            return self.true_fourier_expansion()

    def fourier_jacobi(self):
        r"""
        Return self's Fourier--Jacobi coefficients as a list.
        """
        from .jacobi_lvl import JacobiFormWithLevel
        if not self.has_fourier_jacobi_representation():
            raise ValueError("This is not a Fourier--Jacobi series.")
        try:
            return self.__fourier_jacobi
        except AttributeError:
            pass
        S = self.gram_matrix()
        nrows = S.nrows() - 2
        w = self.weilrep()
        try:
            N2 = w._N2()
            nrows -= 2
        except AttributeError:
            N2 = 1
        if nrows:
            rb = LaurentPolynomialRing(QQ, [f'w_{i}' for i in range(nrows)])
            if not self._base_ring_is_laurent_polynomial_ring():
                rb = FractionField(rb)
        else:
            rb = QQ
        z = rb.gens()[0]
        r, q = PowerSeriesRing(rb, 'q').objgen()
        k = self.weight()
        N = w._N()
        scale = self.scale()
        N = lcm([N, N2, scale])
        d = N2 / lcm(N2, scale)
        S = w._pos_def_gram_matrix()
        if scale != 1:
            prec = self.precision()
            floor_prec = floor(prec)
            if prec in ZZ:
                floor_prec -= 1
            L = [O(q ** (floor_prec - n)) for n in range(floor_prec)]
            coeffs = self.qs_coefficients()
            for x, y in coeffs.items():
                a = x[0]
                c = x[2]
                b = x[1:-1]
                wscale = 1
                if any(u not in ZZ for u in b):
                    b = [e + e for e in b]
                    wscale = 2
                if nrows > 1:
                    u = rb.monomial(*b)
                elif nrows:
                    u = z**(b[0])
                else:
                    u = 1
                L[floor(c)] += (q ** floor(a)) * u * y
            qshift = frac(c)
            self.__fourier_jacobi = [JacobiFormWithLevel(k, N, n * S, j, w_scale=scale, q_scale=d) for n, j in enumerate(L)]
            return self.__fourier_jacobi
        f = self.fourier_expansion()
        r_old = f.parent()
        s = r_old.gens()[1]
        prec = self.precision()
        f = f.polynomial()
        self.__fourier_jacobi = [JacobiFormWithLevel(k, N, n * S, r({x[0]: rb(y) for x, y in f.coefficient({s: n}).dict().items()}).add_bigoh(prec - n), w_scale=scale, q_scale=d) for n in range(prec)]
        return self.__fourier_jacobi

    def has_fourier_jacobi_representation(self):
        s = self.qexp_representation()
        if s == 'PD+II':
            return True
        if s is None:
            return False
        if len(s) == 3:
            if s[0] == 'hermite' or s[0] == 'siegel':
                return True
        elif s == 'shimura':
            return True
        return False

    def nvars(self):
        r"""
        Return the number of variables in self's Fourier development.
        """
        try:
            return self.__nvars
        except AttributeError:
            w = self.weilrep()
            S = w.gram_matrix()
            N = S.nrows()
            if w.is_lorentzian():
                return Integer(N)
            else:
                return Integer(N - 2)

    def pullback(self, *v, new_prec=None):
        r"""
        Compute the restriction of self to a sublattice.
        """
        from .positive_definite import WeilRepPositiveDefinitePlusII
        try:
            P = matrix(ZZ, v)
        except TypeError:
            v = v[0]
            P = matrix(ZZ, v)
        prec = self.precision()
        scale = self.scale()
        S = self.gram_matrix()
        w = self.weilrep()
        if isinstance(w, WeilRepPositiveDefinitePlusII):
            A = identity_matrix(S.nrows())
            A[0, -1] = -1
            S = A * S * A.transpose()
            P = P * A.inverse()
            w = WeilRep(S)
        w_new = WeilRep(P * S * P.transpose())
        d = self.true_coefficients()
        A = w.change_of_basis_matrix()
        A_inv = A.inverse()
        B = w_new.change_of_basis_matrix()
        C = w.orthogonalized_gram_matrix()
        c = -C[0, 0]
        S1 = C[1:, 1:]
        H = B * P * A_inv
        a = H[0, 0]
        if a < 0:
            H = -H
            a = -a
        if new_prec is None:
            v = vector(H[0, 1:])
            try:
                new_prec = floor(prec * (a - (v*v) * math.sqrt(c / (v * S1 * v))))
            except ZeroDivisionError:
                new_prec = floor(prec * a)
        nrows = w_new.gram_matrix().nrows()
        N = w._N()
        if N <= 2:
            K = QQ
            if N == 1:
                zeta = Integer(1)
            else:
                zeta = -Integer(1)
        else:
            K = CyclotomicField(N, 'mu%d' % N)
            zeta, = K.gens()
        if nrows > 1:
            if nrows > 2:
                rb = LaurentPolynomialRing(K, [f'r_{i}' for i in range(nrows - 2)])
            else:
                rb = K
            rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        else:
            rb_x = K
            x = 1
        r, t = PowerSeriesRing(rb_x, 't').objgen()
        f = r(0).add_bigoh(new_prec)
        for _v, n in d.items():
            v = H * vector(_v) * scale
            a = v[0]
            if a < new_prec:
                m = 1
                if nrows > 1:
                    b = -v[1]
                    if nrows > 2:
                        c = -vector(v[2:])
                        try:
                            m = rb.monomial(*c)
                        except AttributeError:  # univariate Laurent polynomial ring does not have "monomial"?
                            m = rb.gens()[0] ** c[0]
                    m *= (x ** b)
                m *= (t ** a)
                f += m * n
        return OrthogonalModularForm(self.weight(), w_new, f, scale, vector([0]*nrows))

    def pullback_perp(self, *v, **kwargs):
        r"""
        Compute the pullback of self to the orthogonal complement of a dual lattice vector (or set of dual lattice vectors) 'v'.

        NOTE: 'v' must have positive norm! (or if 'v' is a list of vectors, then it must span a positive-definite subspace with respect to the underlying quadratic form)
        """
        S = self.gram_matrix()
        try:
            z = matrix(QQ, v)
        except TypeError:
            v = v[0]
            z = matrix(QQ, v)
        z *= S
        k = z.transpose().integer_kernel()
        y = matrix(k.basis())
        Q = QuadraticForm(matrix(QQ, y * S * y.transpose()))
        b = True
        N = 0
        while b:
            N -= 1
            try:
                v = Q.solve(N)
                b = any(x not in ZZ for x in v)
            except ArithmeticError:
                pass
            if N < -100000:  # ?!
                raise RuntimeError from None
        x = matrix(ZZ, matrix(ZZ, [v]).transpose().echelon_form(transformation=True)[1].inverse()).transpose() * y
        if 'print_basis' in kwargs.keys():
            s = kwargs.pop('print_basis')
            if s:
                print('pullback to basis:', x.rows())
        return self.pullback(x.rows(), **kwargs)

    def _q_s_expansion(self):
        r"""
        Return our Fourier expansion as a q-s expansion if possible.
        """
        if not self.has_fourier_jacobi_representation():
            raise NotImplementedError
        d = self.scale()
        h = self.true_fourier_expansion()
        try:
            return self.__q_s_exp
        except AttributeError:
            qsval = ZZ(h.valuation()) / 2
            hval = min(0, qsval)
            self.__qs_valuation = 0
            hprec = h.prec()
            if not h:
                q, s = PowerSeriesRing(self.base_ring(), ('q', 's')).gens()
                self.__q_s_exp = O(q**hprec)
                self.__q_s_scale = 1
                self.__q_s_prec = hprec
                return self.__q_s_exp
            elif self.qexp_representation() == 'shimura':
                q, = PowerSeriesRing(QQ, 'q').gens()
                self.__q_s_exp = h(q)
                self.__q_s_scale = 1
                self.__q_s_prec = h.prec()
                return self.__q_s_exp
            v = 0
            if isinstance(h.parent(), LaurentSeriesRing):
                v = ZZ(max(h.valuation(), 0))
                h = h.valuation_zero_part()
            else:
                qsval = 0
            m = ZZ(max(max(x.degree(), -x.valuation()) - i for i, x in enumerate(h.list())))
            if m > 0:
                h = h.shift(m)
                qsval -= m / 2
            else:
                m = 0
                qsval = 0
            self.__qs_valuation = hval
            try:
                q, s = PowerSeriesRing(self.base_ring(), ('q', 's')).gens()
                self.__q_s_exp = sum((q ** (ZZ(i + v - n) / 2)) * (s ** (ZZ(i + v + n) / 2)) * p[n] for i, p in enumerate(h.list()) for n in p.exponents()).O(h.prec() + v)
            except ValueError:
                mapdict = {u: u*u for u in self.base_ring().base_ring().gens()}
                hprec += hprec
                d += d
                self.__q_s_exp = sum((q ** ((i + v - n))) * (s ** (i + v + n)) * p.coefficients()[j].subs(mapdict) for i, p in enumerate(h.list()) for j, n in enumerate(p.exponents())).O(hprec)
            self.__q_s_scale = d
            self.__q_s_prec = hprec
            return self.__q_s_exp

    def _q_s_valuation(self):
        try:
            return self.__qs_valuation
        except AttributeError:
            _ = self._q_s_expansion()
            return self.__qs_valuation

    # ## other

    def phi(self):
        r"""
        Apply the Siegel Phi operator.

        WARNING: this is only defined for lattices of the form L + II(m) + II(n), where L is positive-definite

        The Siegel Phi operator sets s->0 in self's Fourier--Jacobi expansion. The result is an OrthogonalModularForm on a lattice of signature (2, 1).

        EXAMPLES::

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[2, 1], [1, 2]])) + II(2)
            sage: X = m.borcherds_input_Qbasis(1, 10)
            sage: X[1].borcherds_lift().phi()
            q + 8*q^2 + 28*q^3 + 64*q^4 + 126*q^5 + 224*q^6 + O(q^7)

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[6]])) + II(3)
            sage: m.eisenstein_series(4, 10).phi()
            1 + 240*q^3 + 2160*q^6 + 6720*q^9 + O(q^10)
        """
        if self.nvars() == 1:
            return self
        elif self.has_fourier_jacobi_representation():
            N = self.gram_matrix()[0, 0]
            f = self.true_fourier_expansion()
            if self._base_ring_is_laurent_polynomial_ring():
                R = PowerSeriesRing(QQ, 't')
            else:
                R = PowerSeriesRing(self.base_ring(), 't')
            prec = f.prec()
            f = R([f[j][-j] for j in range(prec)]).O(prec)
            S = matrix([[N]])
            return OrthogonalModularFormLorentzian(self.weight() / 2, WeilRepLorentzian(S), f, scale=self.scale(), weylvec=vector([self.weyl_vector()[0]]), qexp_representation='shimura')
        return NotImplemented

    def witt(self):
        r"""
        Apply the Witt operator.

        WARNING: this is only defined for lattices of the form L + II(m) + II(n), where L is positive-definite

        The Witt operator sets all r_i to 1 in self's Fourier--Jacobi expansion. The result is an OrthogonalModularForm on a lattice of signature (2, 2).

        EXAMPLES::

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[2, 1], [1, 2]])) + II(2)
            sage: X = m.borcherds_input_Qbasis(1/2, 15)
            sage: X[1].borcherds_lift().witt()
            q + 8*q^2 - 16*q*s + 28*q^3 - 128*q^2*s + 112*q*s^2 + 64*q^4 - 448*q^3*s + 896*q^2*s^2 - 448*q*s^3 + 126*q^5 - 1024*q^4*s + 3136*q^3*s^2 - 3584*q^2*s^3 + 1136*q*s^4 + 224*q^6 - 2016*q^5*s + 7168*q^4*s^2 - 12544*q^3*s^3 + 9088*q^2*s^4 - 2016*q*s^5 + 344*q^7 - 3584*q^6*s + 14112*q^5*s^2 - 28672*q^4*s^3 + 31808*q^3*s^4 - 16128*q^2*s^5 + 3136*q*s^6 + O(q, s)^8

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[2, 1], [1, 2]])) + II(2)
            sage: m.eisenstein_series(4, 5).witt()
            1 + 240*q^2 + 3840*q*s + 240*s^2 + 30720*q^2*s + 30720*q*s^2 + 2160*q^4 + 107520*q^3*s + 303360*q^2*s^2 + 107520*q*s^3 + 2160*s^4 + O(q, s)^5
        """
        if self.has_fourier_jacobi_representation():
            if self.nvars() == 2:
                return self
            f = self.true_fourier_expansion()
            R = f.base_ring()
            while R != R.base_ring():
                R = R.base_ring()
            a = lambda x: sum(x for _, x in x.dict().items())
            Rx, x = LaurentPolynomialRing(R, 'x').objgen()
            Rt = PowerSeriesRing(Rx, 't')

            def b(z):
                u, v = z.polynomial_construction()
                return Rx(u.map_coefficients(a)) * x**v
            f = Rt(f.map_coefficients(b))
            N = self.gram_matrix()[0, -1]
            S = matrix([[-(N + N), N], [N, 0]])
            return OrthogonalModularFormLorentzian(self.weight(), WeilRepLorentzian(S), f, scale=self.scale(), weylvec=vector([self.weyl_vector()[0], self.weyl_vector()[-1]]), qexp_representation='PD+II')
        return NotImplemented


class WeilRepLorentzian(WeilRep):
    r"""
    WeilRep for Lorentzian lattices. The Gram matrix should have signature (n, 1) for some n (possibly 0). To compute lifts either the top-left entry must be strictly negative or the WeilRep must have been constructed as w + II(N) where w is positive-definite.
    """
    def __init__(self, S, lift_qexp_representation=None):
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self.lift_qexp_representation = lift_qexp_representation

    def __add__(self, other, _flag=None):
        r"""
        Tensor product of Weil representations.

        If 'other' is a rescaled hyperbolic plane then we rearrange it so that 'other' goes in the first and last coordinates.
        """
        from .weilrep import WeilRep
        ell = self.is_lorentzian()
        if not _flag and isinstance(other, RescaledHyperbolicPlane):
            S = self.gram_matrix()
            n = S.nrows()
            N = other._N()
            S_new = matrix(ZZ, n + 2)
            for i in range(n):
                for j in range(n):
                    S_new[i + 1, j + 1] = S[i, j]
            S_new[0, -1], S_new[-1, 0] = N, N
            if self._is_positive_definite_plus_II():
                from .positive_definite import WeilRepPositiveDefinitePlus2II
                return WeilRepPositiveDefinitePlus2II(S_new, self._pos_def_gram_matrix(), self._N(), N, lift_qexp_representation=self.lift_qexp_representation)
            return WeilRepLorentzianPlusII(S_new, S, N, lift_qexp_representation=self.lift_qexp_representation)
        elif isinstance(other, WeilRep):
            return WeilRep(block_diagonal_matrix([self.gram_matrix(), other.gram_matrix()], subdivide=False))
        return NotImplemented

    def __radd__(self, other, **kwargs):
        return other.__add__(self, **kwargs)

    def change_of_basis_matrix(self):
        r"""
        Computes a change-of-basis matrix that splits our lattice (over QQ) in the form <-b> + L_0 where L_0 is positive-definite and b > 0.

        This matrix is generally not invertible over ZZ.

        OUTPUT: a matrix a \in ZZ for which a * S * a.transpose() consists of a positive-definite upper-left block and a negative-definite bottom-right entry, where S is self's Gram matrix.
        """
        try:
            return self.__change_of_basis_matrix
        except AttributeError:
            S = self._lorentz_gram_matrix()
            b = S.rows()[0]
            b_perp = matrix(b).transpose().kernel().basis()
            e0 = vector([0] * S.nrows())
            e0[0] = 1
            self.__change_of_basis_matrix = matrix([e0] + b_perp)
            return self.__change_of_basis_matrix

    def is_lorentzian(self):
        return True

    def is_lorentzian_plus_II(self):
        return False

    def is_positive_definite(self):
        return False

    def _lifts_have_fourier_jacobi_expansion(self):
        s = self.lift_qexp_representation
        if s == 'PD+II':
            return True
        elif s is None:
            return False
        elif len(s) == 3:
            if s[0] == 'hermite':
                return True
        return False

    def orthogonalized_gram_matrix(self):
        r"""
        Compute a Gram matrix of a finite-index sublattice that splits orthogonally into a positive-definite lattice and a negative-definite line.

        This returns a * S * a.transpose(), where ``S`` is self's Gram matrix and ``a`` is the result of self.change_of_basis_matrix()
        """
        try:
            return self.__orthogonalized_gram_matrix
        except AttributeError:
            S = self._lorentz_gram_matrix()
            a = self.change_of_basis_matrix()
            self.__orthogonalized_gram_matrix = a * S * a.transpose()
            return self.__orthogonalized_gram_matrix

    def _N(self):
        return Integer(1)

    def _lorentz_gram_matrix(self):
        return self.gram_matrix()


class RescaledHyperbolicPlane(WeilRepLorentzian):
    r"""
    Represents the rescaled hyperbolic plane (i.e. Z^2, with quadratic form (x, y) -> Nxy for some N).

    The main use of this is to provide certain orthogonal modular forms with Fourier--Jacobi expansions. You can add a RescaledHyperbolicPlane to a positive-definite WeilRep to produce input forms whose lifts are given as Fourier--Jacobi series instead of the default 't', 'x', 'r_0,...' For example:

    w = WeilRep(matrix([[2]]))
    w = w + II(3)
    """

    def __init__(self, N, **kwargs):
        if 'K' in kwargs.keys() or N not in ZZ:
            from .unitary import HermitianRescaledHyperbolicPlane
            self.__class__ = HermitianRescaledHyperbolicPlane
            HermitianRescaledHyperbolicPlane.__init__(self, N, **kwargs)
            return None
        self.__N = N
        S = matrix([[0, N], [N, 0]])
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self._WeilRep__vals = {}
        self._WeilRep__valsm = {}
        self.lift_qexp_representation = 'PD+II'

    def _N(self):
        return self.__N

    def _is_positive_definite_plus_II(self):
        return True

    def _pos_def_gram_matrix(self):
        return matrix([])


def II(N, **kwargs):  # short constructor for rescaled hyperbolic planes
    return RescaledHyperbolicPlane(N, **kwargs)


class WeilRepModularFormLorentzian(WeilRepModularForm):
    r"""

    This is a class for modular forms attached to Lorentzian lattices or Lorentzian lattices + II(N) for some N. It provides the additive theta lift and the Borcherds lift.

    """

    def __init__(self, k, f, w):
        self._WeilRepModularForm__weight = k
        self._WeilRepModularForm__fourier_expansions = f
        self._WeilRepModularForm__weilrep = w
        self._WeilRepModularForm__gram_matrix = w.gram_matrix()

    def _weight_one_theta_lift_constant_term(self):
        r"""
        Compute the constant term in the additive theta lift to weight 1.

        This should not be called directly.

        The additive theta lift does not map cusp forms to cusp forms when the target has weight 1. (for subgroups of SL_2 this means weight 2). We try to compute the missing constant term here.

        NOTE: theta lifts of weight 1 only exist for lattices of signature (2, n) with n <= 4.

        OUTPUT: a rational number
        """
        if not self:
            return 0
        w = self.weilrep()
        nrows = w.gram_matrix().nrows()
        extra_plane = w.is_lorentzian_plus_II()
        N = w._N()
        if w.lift_qexp_representation == 'PD+II':
            a = identity_matrix(nrows - 2*extra_plane)
            a[-1, 0] = -1
            if extra_plane:
                w = WeilRepLorentzian(a.transpose() * w.gram_matrix()[1:-1, 1:-1] * a) + II(N)
            else:
                w = WeilRepLorentzian(a.transpose() * w.gram_matrix() * a)
            if extra_plane:
                z = matrix([[1]])
                a = block_diagonal_matrix([z, a, z])
            x = self.conjugate(a, w=w)
        else:
            x = self
        if nrows > 1:
            if extra_plane:
                a = matrix(nrows)
                a[0, 0], a[-1, 1] = 1, 1
                for i in range(nrows - 2):
                    a[i + 1, (i + 2) % nrows] = 1
                x = x.conjugate(a)
                nrows -= 2
            while nrows != 1:
                x = x.theta_contraction()
                nrows -= 1
        S = x.gram_matrix()
        s = S[-1, -1]
        if extra_plane:
            A = matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            x = x.conjugate(A, w=WeilRep(matrix([[s]])) + II(N))
        x = x.theta_lift(constant_term_weight_one=False)
        f = x.fourier_expansion()
        m = ModularForms(Gamma0_constructor(-s*N // 2), 2, prec=x.precision()).echelon_basis()
        f -= sum(z.qexp() * f[z.qexp().valuation()] for z in m[1:])
        i = f.exponents()[0]
        return f[i] / m[0].qexp()[i]

    def theta_lift(self, prec=None, constant_term_weight_one=True):
        r"""
        Compute the (additive) theta lift.

        This computes the additive theta lift (e.g. Shimura lift; Doi--Naganuma lift; etc) of the given vector-valued modular form.

        INPUT:
        - ``prec`` -- max precision (default None). (This is limited by the precision of the input. If prec is None then we compute as much as possible.)
        - ``constant_term_weight_one`` -- boolean (default True) for internal use. If False then we don't bother correcting the constant term when the result has weight 1.

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: WeilRep(matrix([[-2, 0, 0], [0, 2, 1], [0, 1, 2]])).cusp_forms_basis(11/2, 5)[0].theta_lift()
            t + ((-6*r_0^-1 - 6)*x^-1 + (-6*r_0^-1 + 12 - 6*r_0) + (-6 - 6*r_0)*x)*t^2 + ((15*r_0^-2 + 24*r_0^-1 + 15)*x^-2 + (24*r_0^-2 - 24*r_0^-1 - 24 + 24*r_0)*x^-1 + (15*r_0^-2 - 24*r_0^-1 + 162 - 24*r_0 + 15*r_0^2) + (24*r_0^-1 - 24 - 24*r_0 + 24*r_0^2)*x + (15 + 24*r_0 + 15*r_0^2)*x^2)*t^3 + O(t^4)

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[2]]))
            sage: (w + II(1) + II(4)).modular_forms_basis(1/2, 15)[0].theta_lift()
            -1/4 - q - s - q^2 + (-r^-2 - 2 - r^2)*q*s - s^2 + (-2*r^-2 - 2*r^2)*q^2*s + (-2*r^-2 - 2*r^2)*q*s^2 - q^4 + (-r^-4 - 2 - r^4)*q^2*s^2 - s^4 + (-2)*q^5 + (-r^-4 - 2 - r^4)*q^4*s + (-r^-4 - 2 - r^4)*q*s^4 + (-2)*s^5 + (-2*r^-4 - 2*r^-2 - 2*r^2 - 2*r^4)*q^5*s + (-2*r^-4 - 2*r^4)*q^4*s^2 + (-2*r^-4 - 2*r^4)*q^2*s^4 + (-2*r^-4 - 2*r^-2 - 2*r^2 - 2*r^4)*q*s^5 + (-2*r^-6 - 2*r^-2 - 2*r^2 - 2*r^6)*q^5*s^2 + (-2*r^-6 - 2*r^-2 - 2*r^2 - 2*r^6)*q^2*s^5 + O(q, s)^8
        """
        w = self.weilrep()
        extra_plane = w.is_lorentzian_plus_II()
        s = w.lift_qexp_representation
        if w._lifts_have_fourier_jacobi_expansion():
            S = w.gram_matrix()
            A = identity_matrix(S.nrows())
            if extra_plane:
                A[-2, 1] = -1
                N = w._N()
                S = A.transpose() * S * A
                w = WeilRepLorentzianPlusII(S, S[1:-1, 1:-1], N)
            else:
                A[-1, 0] = -1
                w = WeilRepLorentzian(A.transpose() * S * A)
            w.lift_qexp_representation = s
            X = self.conjugate(A, w=w)
        else:
            X = self
        prec0 = self.precision()
        val = self.valuation()
        if val < 0:
            raise ValueError('Nonholomorphic input function in theta lift.')
        if prec is None:
            prec = isqrt(4 * prec0)
        else:
            prec = min(prec, isqrt(4 * prec0))
        wt = self.weight()
        coeffs = X.coefficients()
        S = w._lorentz_gram_matrix()
        s_0 = w.orthogonalized_gram_matrix()
        if self.is_symmetric() == 1:
            eps = 1
        else:
            eps = -1
        nrows = Integer(s_0.nrows())
        k = wt + nrows/2 - 1
        C = 0
        if k == 1 and constant_term_weight_one:
            if val < 0:
                return NotImplemented
            try:
                C = X._weight_one_theta_lift_constant_term()
            except IndexError:
                print('Warning: I could not find the correct constant term! Please use a higher precision.')  # WARNING: we will keep computing!! even though the output is almost certainly wrong!
        elif k <= 0:
            return NotImplemented
        N = w._N()
        if N <= 2:
            K = QQ
            if N == 1:
                zeta = 1
            else:
                zeta = -1
        else:
            K = CyclotomicField(N, 'mu%d' % N)
            zeta, = K.gens()
        if nrows > 1:
            if nrows > 2:
                rb = LaurentPolynomialRing(K, [f'r_{i}' for i in range(nrows - 2)])
            else:
                rb = K
            rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        else:
            rb_x = K
            x = 1
        extra_plane = False
        t, = PowerSeriesRing(rb_x, 't').gens()
        a = w.change_of_basis_matrix()
        a_tr = a.transpose()
        scale = Integer(a.determinant())
        b_norm = s_0[0, 0]
        s_0 = s_0[1:, 1:]
        s_0inv = s_0.inverse()
        new_prec = ceil(prec * prec * scale * scale / (-4 * b_norm) + prec)
        if nrows >= 3:
            v_matrix = _, _, vs_matrix = pari(s_0inv).qfminim(new_prec, flag=2)
            vs_list = vs_matrix.sage().columns()
            rb0 = rb.gens()[0]
        elif nrows == 2:
            vs_list = [vector([n]) for n in range(1, isqrt(4 * new_prec * s_0[0, 0]))]
        else:
            vs_list = []
        lift = O(t ** prec)
        negative = lambda v: next(s for s in v if s) < 0
        if not w.is_lorentzian():
            extra_plane = True
            if k % 2 == 0:
                y, = PolynomialRing(QQ, 'y').gens()
                bp = bernoulli_polynomial(y, k)
                for i in srange(N):
                    zeta_i = zeta ** i
                    try:
                        c = coeffs[tuple([i / N] + [0]*(nrows + 2))]
                        if c:
                            lift -= c * sum([bp(j / N) * (zeta_i ** j) for j in srange(1, N + 1)])
                    except KeyError:
                        pass
                lift *= (N ** (k - 1)) / (k + k)
        elif k % 2 == 0:
            try:
                c = coeffs[tuple([0]*(nrows + 1))]
                if c:
                    lift -= c * bernoulli(k) / (k + k)
            except KeyError:
                pass
        for v in vs_list:
            sv = s_0inv * v
            j = 1
            while j < prec:
                prec_j = prec//j + 1
                v_big = vector([-j / b_norm] + list(sv))
                z = a_tr * v_big
                v_big_2 = vector([-j / b_norm] + list(-sv))
                sz = S * z
                z_2 = a_tr * v_big_2
                sz_2 = S * z_2
                m = x ** v[0]
                if nrows >= 3:
                    if nrows >= 4:
                        m *= rb.monomial(*v[1:])
                    else:
                        m *= rb0 ** v[1]
                if negative(sz):
                    norm_z = z * sz / 2
                    if extra_plane:
                        z = vector([0] + list(z) + [0])
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        lift += c * sum([n ** (k - 1) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                    except KeyError:
                        if -norm_z >= prec0:
                            prec = j
                            break
                        pass
                    if extra_plane:
                        for i in srange(1, N):
                            z[0] = i / N
                            zeta_i = zeta ** i
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                lift += c * sum([n ** (k - 1) * (zeta_i ** n) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                            except KeyError:
                                pass
                if negative(sz_2):
                    z = z_2
                    sz = sz_2
                    norm_z = z * sz / 2
                    m = ~m
                    if extra_plane:
                        z = vector([0] + list(z) + [0])
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        lift += c * sum([n ** (k - 1) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                    except KeyError:
                        if -norm_z >= prec0:
                            prec = j
                            break
                        pass
                    if extra_plane:
                        for i in srange(1, N):
                            z[0] = i / N
                            zeta_i = zeta ** i
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                lift += c * sum([n ** (k - 1) * (zeta_i ** n) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                            except KeyError:
                                pass
                j += 1
        v_big = vector([QQ(0)] * nrows)
        for j in srange(1, prec):
            v_big[0] = -j / b_norm
            z = a_tr * v_big
            sz = S * z
            norm_z = z * sz / 2
            if extra_plane:
                z = vector([0] + list(z) + [0])
                for i in srange(N):
                    z[0] = i / N
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        lift += c * sum([n ** (k - 1) * (zeta ** (i * n)) * t ** (n * j) for n in srange(1, prec//j + 1)])
                    except KeyError:
                        pass
            else:
                try:
                    c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                    lift += c * sum([n ** (k - 1) * t ** (n * j) for n in srange(1, prec//j + 1)])
                except KeyError:
                    if -norm_z >= self.precision():
                        prec = j
                        break
                    pass
        if eps == -1 and extra_plane and N >= 3:
            lift /= sum(zeta**i - zeta**(-i) for i in range(1, (N + 1)//2))
        return OrthogonalModularForm(k, w, lift + C + O(t ** prec), scale=1, weylvec=vector([0]*nrows), qexp_representation=w.lift_qexp_representation)

    def weyl_vector(self):
        r"""
        Compute the Weyl vector for the Borcherds lift.

        For lattices of the form L + II(m) + II(n) where w is positive-definite we use Borcherds Theorem 10.4. Otherwise we use a recursive method (which may have bugs and should not be trusted!!)

        OUTPUT: a vector
        """
        w = self.weilrep()
        if w._lifts_have_fourier_jacobi_expansion():
            S = self.gram_matrix()
            nrows = S.nrows()
            M = identity_matrix(nrows)
            M[-1, 0] = 1
            X = self.conjugate(M)
            v = X.__weyl_vector_II()
            return vector([v[0] + v[-1], v[-1] - v[0]] + list(v[1:-1]))
        S = w._lorentz_gram_matrix()
        nrows = Integer(S.nrows())
        if not self:
            return vector([0] * nrows)
        orth = any(S[0, i] for i in range(1, nrows))
        if orth:
            a = w.change_of_basis_matrix()
            X = self.conjugate(a.transpose())
            return X.__weyl_vector()
        return self.__weyl_vector()

    def __weyl_vector(self):
        r"""

        Auxiliary function for the Weyl vector. This should not be called directly.

        WARNING: this has bugs! it should not be trusted!

        """
        w = self.weilrep()
        S = w._lorentz_gram_matrix()
        nrows = Integer(S.nrows())
        val = self.valuation(exact=True)
        # suppose v is a primitive vector. extend_vector computes a matrix M in GL_n(Z) whose left column is v. If v has a denominator then we divide it off first.
        extend_vector = lambda v: matrix(ZZ, matrix([v]).transpose().echelon_form(transformation=True)[1].inverse())
        if nrows > 1:
            N = nrows - 1
            v = vector(list(self.theta_contraction().__weyl_vector()) + [QQ(0)])
            u = vector([0] + [1] * N)
            norm = u * S * u / 2
            u = -S[0, 0] // 2
            prime = GCD(norm, u) == 1 and not (norm.is_square() and u.is_square())
            z = vector([1 + isqrt(4 * norm * (1 / (4 * u) - val))] + [-1] * N)
            norm = Integer(z * S * z // 2)
            while (norm >= 0 or is_square(-norm)) or (prime and not is_prime(-norm)):
                z[0] += 1
                norm = Integer(z * S * z // 2)
            u = extend_vector(z)
            X = self.conjugate(u)
            while N > 0:
                X = X.theta_contraction()
                N -= 1
            h = X.__weyl_vector()[0]
            v[-1] = (h - v[:-1] * z[:-1]) / z[-1]
            return v
        else:
            N = -Integer(S[0, 0] / 2)
            q, = PowerSeriesRing(QQ, 'q').gens()
            if (N == 1 or N.is_prime()):
                f = self.fourier_expansion()
                if val < 0:
                    L = []
                    ds = []
                    for i, x in enumerate(f):
                        if x[2].valuation() <= 0:
                            ds.append(x[0])
                            L.append(i)
                    indices = [None] * len(ds)
                    e = w.dual().eisenstein_series(sage_three_half, max(1, 1 - floor(val)), allow_small_weight=True, components=(ds, indices)).fourier_expansion()
                    s = sum([(f[j][2] * e[j][2] * q ** (floor(f[j][1] + e[j][1])))[0] for j in L])
                else:
                    s = f[0][2][0]
                return vector([s * (1 + N) / 24])
            i = 0
            L = []
            mult = max(1, ceil(Integer(1)/4 - val))
            j = 1
            s = isqrt(N)
            s = s * s == N
            sqr = False
            if s:
                if N % 2:
                    if -val < 10:
                        X = WeilRepModularForm(-sage_one_half, matrix([[4]]), [((0,), 0, 4 + 16*q + 56*q**2 + 160*q**3 + 400*q**4 + 928*q**5 + 2016*q**6 + 4160*q**7 + 8248*q**8 + 15792*q**9 + O(q**10)), ((QQ(1)/4,), QQ(-1)/8, 1 - q + q**2 - 2*q**3 + 3*q**4 - 4*q**5 + 5*q**6 - 7*q**7 + 10*q**8 - 13*q**9 + 16*q**10 + O(q**11)), ((QQ(1)/2,), QQ(-1)/2, -8*q - 32*q**2 - 96*q**3 - 256*q**4 - 616*q**5 - 1376*q**6 - 2912*q**7 - 5888*q**8 - 11456*q**9 - 21600*q**10 + O(q**11)), ((QQ(3)/4,), QQ(-1)/8, 1 - q + q**2 - 2*q**3 + 3*q**4 - 4*q**5 + 5*q**6 - 7*q**7 + 10*q**8 - 13*q**9 + 16*q**10 + O(q**11))])
                    else:
                        X = WeilRep(matrix([[4]])).nearly_holomorphic_modular_forms_basis(-sage_one_half, 1, ceil(-val))[0]
                    p_0 = 2
                elif N % 3:
                    if -val < 10:
                        X = WeilRepModularForm(-sage_one_half, matrix([[6]]), [((0,), 0, 2 + 4*q + 12*q**2 + 24*q**3 + 52*q**4 + 96*q**5 + 180*q**6 + 312*q**7 + 540*q**8 + 892*q**9 + O(q**10)), ((QQ(1)/6,), QQ(-1)/12, 1 + 2*q + 5*q**2 + 12*q**3 + 24*q**4 + 46*q**5 + 85*q**6 + 150*q**7 + 257*q**8 + 430*q**9 + 701*q**10 + O(q**11)), ((QQ(1)/3,), QQ(-1)/3, -2*q - 4*q**2 - 10*q**3 - 20*q**4 - 40*q**5 - 72*q**6 - 132*q**7 - 224*q**8 - 380*q**9 - 620*q**10 + O(q**11)), ((QQ(1)/2,), QQ(-3)/4, -2*q - 6*q**2 - 14*q**3 - 30*q**4 - 60*q**5 - 114*q**6 - 206*q**7 - 360*q**8 - 612*q**9 - 1014*q**10 + O(q**11)), ((QQ(2)/3,), QQ(-1)/3, -2*q - 4*q**2 - 10*q**3 - 20*q**4 - 40*q**5 - 72*q**6 - 132*q**7 - 224*q**8 - 380*q**9 - 620*q**10 + O(q**11)), ((QQ(5)/6,), QQ(-1)/12, 1 + 2*q + 5*q**2 + 12*q**3 + 24*q**4 + 46*q**5 + 85*q**6 + 150*q**7 + 257*q**8 + 430*q**9 + 701*q**10 + O(q**11))])
                    else:
                        X = WeilRep(matrix([[6]])).nearly_holomorphic_modular_forms_basis(-sage_one_half, 1, ceil(-val))[0]
                    p_0 = 3
                else:
                    sqr = True
            if sqr or not s:
                if -val < 10:
                    X = WeilRepModularForm(-sage_one_half, matrix([[2]]), [((0,), 0, 10 + 108*q + 808*q**2 + 4016*q**3 + 16524*q**4 + 58640*q**5 + 188304*q**6 + 556416*q**7 + 1541096*q**8 + 4038780*q**9 + O(q**10)), ((QQ(1)/2,), QQ(-1)/4, 1 - 64*q - 513*q**2 - 2752*q**3 - 11775*q**4 - 43200*q**5 - 141826*q**6 - 427264*q**7 - 1201149*q**8 - 3189120*q**9 - 8067588*q**10 + O(q**11))])
                else:
                    X = WeilRep(matrix([[2]])).nearly_holomorphic_modular_forms_basis(-sage_one_half, 1, ceil(-val))[0]
                p_0 = 1
            X = self * X
            m = X.valuation(exact=True)
            Xcoeff = X.principal_part_coefficients()
            e = X.weilrep().dual().eisenstein_series(2, ceil(-m), allow_small_weight=True).coefficients()
            scale = 1 + isqrt(p_0 / N - m) + sum(e[tuple(list(g[:-1]) + [-g[-1]])] * n / 24 for g, n in Xcoeff.items() if n < 0 and tuple(list(g[:-1]) + [-g[-1]]) in e.keys())  # dubious
            v = []
            h = [None] * 2
            bound = 2
            j = ceil(scale)
            while i < bound:  # we can raise this bound for some redundant checks that the weyl vector is correct
                jN = j * j * N
                for f in range(1, 1 + ceil(j / scale)):
                    norm = f * f * p_0 - jN
                    if norm < 0 and norm.is_squarefree():
                        if sqr or is_prime(-norm):
                            y = vector([j, -f])
                            v.append(y)
                            L.append(extend_vector(y))
                            i += 1
                    if i == bound:
                        break
                j += 1
            for j, a in enumerate(L):
                x = X.conjugate(a).theta_contraction()
                h[j] = QQ(x.__weyl_vector()[0])
            return vector([(matrix(v).solve_right(vector(h)) * Integer(p_0) / 12)[0]])

    def __weyl_vector_II(self):
        r"""

        Auxiliary function for the Weyl vector. This should not be called directly.

        """
        S = self.gram_matrix()
        nrows = S.nrows()
        N = S[0, -1]
        XK = self.reduce_lattice(z=vector([1] + [0]*(nrows - 1)), z_prime=vector([0]*(nrows - 1) + [Integer(1) / N]))
        val = self.valuation()
        K = S[1:-1, 1:-1]
        K_inv = K.inverse()
        if XK:
            Theta_K = WeilRep(-K).theta_series(1 - val)
        else:
            Theta_K = WeilRep(-K).zero(K.nrows()/2, 1-val)
        Xcoeff = self.principal_part_coefficients()
        rho = vector([0] * (nrows - 2))
        rho_z = 0
        negative = lambda v: next(s for s in reversed(v) if s) < 0
        try:
            _, _, vs_matrix = pari(K_inv).qfminim(1 - val, flag=2)
            vs_list = vs_matrix.sage().columns()
        except PariError:
            vs_list = [vector([n]) for n in range(1, isqrt(2 * K[0, 0] * (-val)) + 1)]
        for v in vs_list:
            y = list(map(frac, K_inv * v))
            if negative(v):
                v *= -1
            v_norm = -v * K_inv * v / 2
            try:
                rho += Xcoeff[tuple([0] + y + [0, v_norm])] * v
            except KeyError:
                pass
            for i in srange(N):
                j = i / N
                try:
                    c = Xcoeff[tuple([j] + y + [0, v_norm])]
                    rho_z += c * (j * (j - 1) + Integer(1) / 6) / 2
                except KeyError:
                    pass
        for i in srange(N):
            j = i / N
            try:
                c = Xcoeff[tuple([j] + [0]*nrows)]
                rho_z += c * (j * (j - 1) + Integer(1) / 6) / 4
            except KeyError:
                pass
        e2 = eisenstein_series_qexp(2, 1 - val)
        rho_z_prime = -((XK & Theta_K) * e2)[0]
        return vector([rho_z_prime] + list(rho/2) + [N * rho_z])

    def weyl_vector_III(self, z):
        val = self.valuation()
        w = self.weilrep()
        s = w.gram_matrix()
        nrows = s.nrows()
        sz = s * z

        def xgcd_v(x):  # xgcd for more than two arguments
            if len(x) > 1:
                g, a = xgcd_v(x[:-1])
                if g == 1:
                    return g, vector(list(a) + [0])
                new_g, s, t = XGCD(g, x[-1])
                return new_g, vector(list(a * s) + [t])
            return x[0], vector([1])
        _, zeta = xgcd_v(sz)
        _, sz_prime = xgcd_v(z)
        z_prime, szeta, n = s.inverse() * sz_prime, s * zeta, sz * zeta
        zeta_norm, z_prime_norm = zeta * szeta, z_prime * sz_prime
        z_mod = z_prime - z_prime_norm * z
        zeta_K = zeta - n * z_mod - (szeta * z_prime) * z
        k = matrix(matrix([sz, sz_prime]).transpose().integer_kernel().basis())
        try:
            k_k = k * s * k.transpose()
            k_k_inv = k_k.inverse()
        except TypeError:
            k_k = matrix([])
            k_k_inv = k_k
        w_k = WeilRep(k_k)
        theta = w_k.dual().theta_series(1 - val)
        Xcoeff = self.principal_part_coefficients()
        try:
            w_k.lift_qexp_representation = w.lift_qexp_representation
        except AttributeError:
            pass
        ds_k_dict, ds_k, ds = w_k.ds_dict(), w_k.ds(), w.ds()
        Y = [None] * len(ds_k)
        X = self.fourier_expansion()
        r, q = X[0][2].parent().objgen()
        prec = self.precision()
        r_dict = {}
        for i, g in enumerate(ds):
            gsz = Integer(g * sz)
            if not gsz % n:
                g_k = g - gsz * z_mod - (g * sz_prime) * z
                pg = g_k - gsz * zeta_K / n
                try:
                    pg = vector(map(frac, k.solve_left(pg)))
                except ValueError:
                    pg = vector([])
                j = ds_k_dict[tuple(pg)]
                if Y[j] is None:
                    Y[j] = [pg, -frac(pg * k_k * pg / 2), X[i][2]]
                else:
                    Y[j][2] += X[i][2]
                g = tuple(g)
                try:
                    r_dict[g].append(i)
                except KeyError:
                    r_dict[g] = [i]
        for j, g in enumerate(ds_k):
            if Y[j] is None:
                o = -frac(g * k_k * g / 2)
                Y[j] = g, o, r.O(prec - floor(o))
        x = WeilRepModularForm(self.weight(), k_k, Y, w_k)
        e2 = eisenstein_series_qexp(2, 1 - val)
        rho_z_prime = -((x & theta) * e2)[0]
        negative = lambda v: next(s for s in reversed(v) if s) < 0
        try:
            _, _, vs_matrix = pari(k_k_inv).qfminim(1 - val, flag=2)
            vs_list = vs_matrix.sage().columns()
        except PariError:
            vs_list = [vector([n]) for n in range(1, isqrt(2 * k_k[0, 0] * (-val)) + 1)]
        rho_z = -rho_z_prime * z_prime_norm
        rho = vector([0] * (nrows - 2))
        for v in vs_list:
            y = list(map(frac, k_k_inv * v))
            if negative(v):
                v *= -1
            v_norm = -v * k_k_inv * v / 2
            L = r_dict[tuple(y)]
            for i in L:
                g = ds[i]
                j = frac(g * sz_prime)
                try:
                    c = Xcoeff[tuple(list(g) + [v_norm])]
                    rho_z += c * (j * (j - 1) + Integer(1) / 6) / 2
                    if not j:
                        rho += c * v
                except KeyError:
                    pass
        try:
            c = Xcoeff[tuple([0] * (nrows + 1))]
            rho_z += c / 24
        except KeyError:
            pass
        v = rho_z * z + rho_z_prime * z_prime
        print('z, rho_z:', z, rho_z)
        print('z_prime, rho_z_prime:', z_prime, rho_z_prime)
        try:
            return v - k * k_k_inv * rho / 2
        except TypeError:
            return v

    def weyl_vector_IV(self):
        S = self.gram_matrix()
        nrows = S.nrows()
        N = -(S[0, 0] // 2)
        A1 = matrix([[2]])
        # D4 = CartanMatrix(['D', 4])
        v = QuadraticForm(QQ, A1).solve(N)
        # v = QuadraticForm(QQ, D4).solve(N)
        # big_S = block_diagonal_matrix([S, D4])
        big_S = block_diagonal_matrix([S, A1])
        v1 = vector([1] + [0] * (nrows - 1) + list(v))
        v2 = vector([-1] + [0] * (nrows - 1) + list(v))
        # b = matrix(ZZ, matrix(ZZ, [v1, v2]).transpose().echelon_form(transformation = True)[1].inverse())
        b = matrix(ZZ, [big_S * v1, big_S * v2]).transpose().integer_kernel().basis_matrix()
        b = matrix(ZZ, [v1] + b.rows() + [v2]).transpose()
        # psi = OrthogonalModularForms(D4).borcherds_input_Qbasis(Integer(1)/2, self.precision())[0]
        psi = OrthogonalModularForms(A1).borcherds_input_Qbasis(Integer(1)/2, self.precision())[0]
        X = (self * psi).conjugate(b)
        S_conj = X.gram_matrix()
        N, S_conj = S_conj[0, -1], S_conj[1:-1, 1:-1]
        w = WeilRep(S_conj) + II(N)
        S_conj = w.gram_matrix()
        X = WeilRepModularForm(X.weight(), X.gram_matrix(), X.fourier_expansion(), weilrep=w)
        v = X.__weyl_vector_II()
        print('v:', v)
        return b, S_conj.inverse() * v
        # return b.inverse()*vector([v[0] + v[-1], v[-1] - v[0]] + list(v[1:-1]))
        # z = QuadraticForm(QQ, S).solve(0)
        # a = extend_vector([z])
        # a1 = a[:,1:
        # S1 = a1.transpose() * S * a1
        # z_prime = a1 * QuadraticForm(QQ, S1).solve(0)
        # N = z * S * z_prime
        # b = extend_vector([z, z_prime])
        return b, big_S

    def borcherds_lift(self, prec=None, omit_weyl_vector=False, weyl_vector=None, verbose=False):
        r"""
        Compute the Borcherds lift.

        INPUT:
        - ``prec`` -- precision (optional). The precision of the output is limited by the precision of the input. However if ``prec`` is given then the output precision will not exceed ``prec``.

        OUTPUT: OrthogonalModularForm of weight equal to (1/2) of self's constant term.

        EXAMPLES::

            sage: from weilrep import *
            sage: WeilRep(matrix([[-10]])).theta_series(50).borcherds_lift()
            q^(1/4) - q^(5/4) - q^(9/4) + q^(25/4) + 2*q^(29/4) - 2*q^(41/4) + q^(45/4) - q^(49/4) + O(q^(61/4))

            sage: from weilrep import *
            sage: x = polygen(QQ, 'x')
            sage: K.<sqrt5> = NumberField(x * x - 5)
            sage: HMF(K).borcherds_input_Qbasis(1, 5)[0].borcherds_lift()
            -q1^(-1/10*sqrt5 + 1/2)*q2^(1/10*sqrt5 + 1/2) + q1^(1/10*sqrt5 + 1/2)*q2^(-1/10*sqrt5 + 1/2) + q1^(-2/5*sqrt5 + 1)*q2^(2/5*sqrt5 + 1) + 10*q1^(-1/5*sqrt5 + 1)*q2^(1/5*sqrt5 + 1) - 10*q1^(1/5*sqrt5 + 1)*q2^(-1/5*sqrt5 + 1) - q1^(2/5*sqrt5 + 1)*q2^(-2/5*sqrt5 + 1) - 120*q1^(-3/10*sqrt5 + 3/2)*q2^(3/10*sqrt5 + 3/2) + 108*q1^(-1/10*sqrt5 + 3/2)*q2^(1/10*sqrt5 + 3/2) - 108*q1^(1/10*sqrt5 + 3/2)*q2^(-1/10*sqrt5 + 3/2) + 120*q1^(3/10*sqrt5 + 3/2)*q2^(-3/10*sqrt5 + 3/2) - 10*q1^(-4/5*sqrt5 + 2)*q2^(4/5*sqrt5 + 2) + 108*q1^(-3/5*sqrt5 + 2)*q2^(3/5*sqrt5 + 2) + 156*q1^(-2/5*sqrt5 + 2)*q2^(2/5*sqrt5 + 2) + 140*q1^(-1/5*sqrt5 + 2)*q2^(1/5*sqrt5 + 2) - 140*q1^(1/5*sqrt5 + 2)*q2^(-1/5*sqrt5 + 2) - 156*q1^(2/5*sqrt5 + 2)*q2^(-2/5*sqrt5 + 2) - 108*q1^(3/5*sqrt5 + 2)*q2^(-3/5*sqrt5 + 2) + 10*q1^(4/5*sqrt5 + 2)*q2^(-4/5*sqrt5 + 2) - q1^(-11/10*sqrt5 + 5/2)*q2^(11/10*sqrt5 + 5/2) - 108*q1^(-9/10*sqrt5 + 5/2)*q2^(9/10*sqrt5 + 5/2) + 140*q1^(-7/10*sqrt5 + 5/2)*q2^(7/10*sqrt5 + 5/2) - 625*q1^(-1/2*sqrt5 + 5/2)*q2^(1/2*sqrt5 + 5/2) - 810*q1^(-3/10*sqrt5 + 5/2)*q2^(3/10*sqrt5 + 5/2) + 728*q1^(-1/10*sqrt5 + 5/2)*q2^(1/10*sqrt5 + 5/2) - 728*q1^(1/10*sqrt5 + 5/2)*q2^(-1/10*sqrt5 + 5/2) + 810*q1^(3/10*sqrt5 + 5/2)*q2^(-3/10*sqrt5 + 5/2) + 625*q1^(1/2*sqrt5 + 5/2)*q2^(-1/2*sqrt5 + 5/2) - 140*q1^(7/10*sqrt5 + 5/2)*q2^(-7/10*sqrt5 + 5/2) + 108*q1^(9/10*sqrt5 + 5/2)*q2^(-9/10*sqrt5 + 5/2) + q1^(11/10*sqrt5 + 5/2)*q2^(-11/10*sqrt5 + 5/2) + O(q1, q2)^6

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(II(1))
            sage: X = m.borcherds_input_Qbasis(1, 15)
            sage: X[1].borcherds_lift()
            q^-1 + (-s^-1 - 196884*s - 21493760*s^2 - 864299970*s^3 - 20245856256*s^4) + 196884*q + 21493760*q^2 + 864299970*q^3 + 20245856256*q^4 + O(q, s)^5

        """
        w = self.weilrep()
        extra_plane = w.is_lorentzian_plus_II()
        s = w.lift_qexp_representation
        if w._lifts_have_fourier_jacobi_expansion():
            S = w.gram_matrix()
            A = identity_matrix(S.nrows())
            if extra_plane:
                A[-2, 1] = -1
                N = w._N()
                S = A.transpose() * S * A
                w = WeilRepLorentzianPlusII(S, S[1:-1, 1:-1], N)
            else:
                A[-1, 0] = -1
                w = WeilRepLorentzian(A.transpose() * S * A)
            w.lift_qexp_representation = s
            X = self.conjugate(A, w=w)
        else:
            X = self
        prec0 = self.precision()
        val = self.valuation()
        prec0val = prec0 - val
        if prec is None:
            prec = isqrt(4 * (prec0+val))
        else:
            prec = min(prec, isqrt(4 * (prec0+val)))
        prec += 1
        wt = self.weight()
        coeffs = dict(X.coefficients())
        S = w._lorentz_gram_matrix()
        s_0 = w.orthogonalized_gram_matrix()
        nrows = Integer(S.nrows())
        N = w._N()
        if N <= 2:
            K = QQ
            if N == 1:
                zeta = Integer(1)
            else:
                zeta = -Integer(1)
        else:
            K = CyclotomicField(N, 'mu%d' % N)
            zeta, = K.gens()
        try:
            k = Integer(coeffs[tuple([0] * (nrows + 1))]) / 2
        except KeyError:
            try:
                k = Integer(coeffs[tuple([0] * (nrows + 3))]) / 2
            except KeyError:
                k = Integer(0)
        if nrows > 1:
            if nrows > 2:
                rb = LaurentPolynomialRing(K, [f'r_{i}' for i in range(nrows - 2)])
            else:
                rb = K
            rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        else:
            rb_x = K
            x = 1
        r, t = PowerSeriesRing(rb_x, 't').objgen()
        if omit_weyl_vector:
            weyl_vector = vector([0] * nrows)
        elif weyl_vector is not None:
            pass
        elif extra_plane:
            weyl_vector = X.reduce_lattice(z=vector([1] + [0] * (nrows + 1))).weyl_vector()
        else:
            weyl_vector = X.weyl_vector()
        a = w.change_of_basis_matrix()
        a_tr = a.transpose()
        d = denominator(weyl_vector)
        weyl_vector *= d
        scale = Integer(a.determinant())
        b_norm = s_0[0, 0]
        s_0 = s_0[1:, 1:]
        s_0inv = s_0.inverse()
        new_prec = ceil(prec * (prec * scale * scale / (-4 * b_norm) + 2))
        if nrows >= 3:
            v_matrix = _, _, vs_matrix = pari(s_0inv).qfminim(new_prec, flag=2)
            vs_list = vs_matrix.sage().columns()
            rb0 = rb.gens()[0]
        elif nrows == 2:
            vs_list = [vector([n]) for n in range(1, isqrt(4 * new_prec * s_0[0, 0]))]
        else:
            vs_list = []
        h = O(t ** prec)
        log_f = h
        const_f = rb_x(1)
        val = self.valuation(exact=True)
        excluded_vectors = set()
        rpoly, tpoly = PolynomialRing(K, 'tpoly').objgen()
        rpoly_ff = FractionField(rpoly)
        negative = lambda v: v[0] < 0 or next(s for s in reversed(v[1:]) if s) < 0
        if nrows >= 2:
            weyl_diff = weyl_vector[0] - weyl_vector[1]
        else:
            weyl_diff = 0
        row0 = S.rows()[0]
        for v in vs_list:
            sv = s_0inv * v
            vnorm = v * sv / 2
            j_bound = 1
            if j_bound < prec:
                v *= d
                m = x**v[0]
                if nrows >= 3:
                    if nrows >= 4:
                        m *= rb.monomial(*v[1:])
                    else:
                        m *= rb0**v[1]
                for j in srange(j_bound, prec):
                    v_big = vector([-j / b_norm] + list(sv))
                    z = a_tr * v_big
                    sz = S * z
                    norm_z = j * j / (b_norm + b_norm) + vnorm
                    if extra_plane:
                        z = vector([0] + list(z) + [0])
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        log_f += c * log(1 - t**j * m + h)
                        if verbose:
                            print('Multiplying by factor: (%s)^%s' % (1 - t**j * m, c))
                    except KeyError:
                        if -norm_z >= prec0:
                            prec = j
                            h = O(t ** j)
                            break
                        pass
                    if extra_plane:
                        for i in srange(1, N):
                            z[0] = i / N
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                log_f += c * log(1 - (zeta**i) * (t ** j) * m + h)
                                if verbose:
                                    print('Multiplying by factor: (%s)^%s' % (1 - zeta**i * t**j * m, c))
                            except KeyError:
                                pass
                    v_big = vector([-j / b_norm] + list(-sv))
                    z = a_tr * v_big
                    if extra_plane:
                        z = vector([0] + list(z) + [0])
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        log_f += c * log(1 - t**j * ~m + h)
                        if verbose:
                            print('Multiplying by factor: (%s)^%s' % (1 - t**j * ~m, c))
                    except KeyError:
                        if -norm_z >= prec0:
                            prec = j
                            h = O(t ** j)
                            break
                        pass
                    if extra_plane:
                        for i in srange(1, N):
                            z[0] = i / N
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                log_f += c * log(1 - (zeta ** i) * (t ** j) * ~m + h)
                                if verbose:
                                    print('Multiplying by factor: (%s)^%s' % (1 - zeta**i * t**j * ~m, c))
                            except KeyError:
                                pass
            if nrows > 1:
                p = rpoly(1)
                if (not extra_plane) and tuple(v) not in excluded_vectors and tuple(-v) not in excluded_vectors:
                    v_big = vector([0] + list(sv))
                    z = a_tr * v_big
                    sz = S * z
                    if not negative(sz):
                        v *= -1
                        m = ~m
                        sv *= -1
                        z *= -1
                        sz *= -1
                    norm_z = z * sz / 2
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        if c > 0:
                            const_f *= (1 - m + h) ** c
                        else:
                            p *= (1 - tpoly) ** c
                            for j in range(2, isqrt(-val / norm_z) + 1):
                                try:
                                    c_new = coeffs[tuple([frac(j * y) for y in z] + [-j * j * norm_z])]
                                    p *= (1 - tpoly**j) ** c_new
                                    excluded_vectors.add(tuple(j * v))
                                except KeyError:
                                    pass
                    except KeyError:
                        pass
                elif extra_plane:
                    v_big = vector([0] + list(sv))
                    z = a_tr * v_big
                    sz = S * z
                    if not negative(sz):
                        v *= -1
                        m = ~m
                        sv *= -1
                        z *= -1
                        sz *= -1
                    norm_z = z * sz / 2
                    z = vector([0] + list(z) + [0])
                    for i in srange(N):
                        z[0] = i / N
                        mu = zeta ** i
                        vtuple_i = tuple([i] + list(v))
                        if vtuple_i not in excluded_vectors:
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                if c > 0:
                                    const_f *= (1 - mu * m + h) ** c
                                else:
                                    p *= (1 - mu * tpoly) ** c
                                    for j in range(2, isqrt(-val / norm_z) + 1):
                                        v_new = tuple([j*i % N] + list(j * v))
                                        if v_new not in excluded_vectors:
                                            try:
                                                c_new = coeffs[tuple([frac(j * y) for y in z] + [-j * j * norm_z])]
                                                p *= (1 - mu**j * (tpoly**j)) ** c_new
                                                excluded_vectors.add(v_new)
                                            except KeyError:
                                                pass
                            except KeyError:
                                pass
                try:
                    const_f *= rb_x(rpoly(p).subs({tpoly: m}))
                    if verbose:
                        print('Multiplying by: %s' % rb_x(rpoly(p).subs({tpoly: m})))
                except TypeError:
                    const_f *= rpoly_ff(p).subs({tpoly: m})
        v = a.rows()[0]
        norm_v = v * S * v / 2
        for j in srange(1, prec):
            jb = -j/b_norm
            norm_z = -norm_v * jb * jb
            if extra_plane:
                for i in srange(N):
                    try:
                        c = coeffs[tuple([i/N] + list(map(frac, jb * v)) + [0, norm_z])]
                        log_f += c * log(1 - zeta**i * t**j + h)
                        if verbose:
                            print('Multiplying by factor: (%s)^%s' % (1 - zeta**i * t**j, c))
                    except KeyError:
                        if not i and norm_z >= prec0:
                            prec = j
                            h = O(t ** j)
                            log_f += h
                            break
                        pass
            else:
                try:
                    c = coeffs[tuple([frac(jb * y) for y in v] + [norm_z])]
                    log_f += c * log(1 - t**j + h)
                    if verbose:
                        print('Multiplying by factor: (%s)^%s' % (1 - t**j, c))
                except KeyError:
                    if norm_z >= prec0:
                        prec = j
                        h = O(t ** j)
                    pass
        weyl_monomial = 1
        if nrows >= 2:
            weyl_monomial = x ** (weyl_vector[1])
            if nrows >= 3:
                if nrows >= 4:
                    weyl_monomial *= rb.monomial(*weyl_vector[2:])
                else:
                    weyl_monomial *= rb0**weyl_vector[-1]
        if extra_plane and N > 1:
            C = Integer(1)
            for i in srange(1, N // 2):
                try:
                    c = Integer(coeffs[tuple([i / N] + [0] * (nrows + 2))])
                    C *= (1 - zeta**i)**c
                except KeyError:
                    pass
            try:
                c = Integer(coeffs[tuple([Integer(1) / 2] + [0] * (nrows + 2))])
                C *= Integer(2)**Integer(c / 2)
            except (KeyError, TypeError):
                pass
            const_f *= C
        f = exp(log_f) * const_f
        return OrthogonalModularForm(k, w, f.V(d) * weyl_monomial * (t ** weyl_vector[0]), scale=d, weylvec=weyl_vector, qexp_representation=w.lift_qexp_representation, ppcoeffs=self.principal_part_coefficients())


class WeilRepLorentzianPlusII(WeilRepLorentzian):

    def __init__(self, S, lorentz_S, N, lift_qexp_representation=None):
        # S should be a Lorentzian lattice in which the bottom-right entry is negative!!
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self.lift_qexp_representation = lift_qexp_representation
        self.__lorentz_gram_matrix = lorentz_S
        self.__N = N

    def _N(self):
        return self.__N

    def _lorentz_gram_matrix(self):
        return self.__lorentz_gram_matrix

    def is_lorentzian(self):
        return False

    def is_lorentzian_plus_II(self):
        return True


def _theta_lifts(X, prec=None, constant_term_weight_one=True):
    Xref = X[0]
    w = Xref.weilrep()
    extra_plane = w.is_lorentzian_plus_II()
    s = w.lift_qexp_representation
    if w._lifts_have_fourier_jacobi_expansion():
        from .weilrep_modular_forms_class import WeilRepModularFormsBasis
        S = w.gram_matrix()
        A = identity_matrix(S.nrows())
        if extra_plane:
            A[-2, 1] = -1
            N = w._N()
            S = A.transpose() * S * A
            w = WeilRepLorentzianPlusII(S, S[1:-1, 1:-1], N)
        else:
            A[-1, 0] = -1
            w = WeilRepLorentzian(A.transpose() * S * A)
        w.lift_qexp_representation = s
        X = WeilRepModularFormsBasis(X.weight(), [x.conjugate(A, w=w) for x in X], w)
    prec0 = X.precision()
    val = X.valuation()
    if val < 0:
        raise ValueError('Nonholomorphic input function in theta lift.')
    if prec is None:
        prec = isqrt(4 * prec0)
    else:
        prec = min(prec, isqrt(4 * prec0))
    wt = X.weight()
    coeffs = X.coefficients()
    S, s_0 = w._lorentz_gram_matrix(), w.orthogonalized_gram_matrix()
    if self.is_symmetric() == 1:
        eps = 1
    else:
        eps = -1
    nrows = Integer(s_0.nrows())
    k = wt + nrows/2 - 1
    C = 0
    if k == 1 and constant_term_weight_one:
        if val < 0:
            return NotImplemented
        try:
            C = [x._weight_one_theta_lift_constant_term() for x in X]
        except IndexError:
            print('Warning: I could not find the correct constant term! Please use a higher precision.')  # WARNING: we will keep computing!! even though the output is almost certainly wrong!
    elif k <= 0:
        return NotImplemented
    N = w._N()
    if N <= 2:
        K = QQ
        if N == 1:
            zeta = 1
        else:
            zeta = -1
    else:
        K = CyclotomicField(N, 'mu%d' % N)
        zeta, = K.gens()
    if nrows > 1:
        if nrows > 2:
            rb = LaurentPolynomialRing(K, [f'r_{i}' for i in range(nrows - 2)])
        else:
            rb = K
        rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
    else:
        rb_x = K
        x = 1
    extra_plane = False
    r, t = PowerSeriesRing(rb_x, 't').objgen()
    a = w.change_of_basis_matrix()
    a_tr = a.transpose()
    scale = Integer(a.determinant())
    b_norm = s_0[0, 0]
    s_0 = s_0[1:, 1:]
    s_0inv = s_0.inverse()
    new_prec = ceil(prec * prec * scale * scale / (-4 * b_norm) + prec)
    if nrows >= 3:
        v_matrix = _, _, vs_matrix = pari(s_0inv).qfminim(new_prec, flag=2)
        vs_list = vs_matrix.sage().columns()
        rb0 = rb.gens()[0]
    elif nrows == 2:
        vs_list = [vector([n]) for n in range(1, isqrt(4 * new_prec * s_0[0, 0]))]
    else:
        vs_list = []
    lift = [r.O(prec) for _ in X]
    negative = lambda v: next(s for s in v if s) < 0
    if not w.is_lorentzian():
        extra_plane = True
        if not k % 2:
            y, = PolynomialRing(QQ, 'y').gens()
            bp = bernoulli_polynomial(y, k)
            for i in srange(N):
                zeta_i = zeta ** i
                try:
                    c = coeffs[tuple([i / N] + [0]*(nrows + 2))]
                    if c:
                        lift -= c * sum([bp(j / N) * (zeta_i ** j) for j in srange(1, N + 1)])
                except KeyError:
                    pass
                lift *= (N ** (k - 1)) / (k + k)
        elif k % 2 == 0:
            try:
                c = coeffs[tuple([0]*(nrows + 1))]
                if c:
                    lift -= c * bernoulli(k) / (k + k)
            except KeyError:
                pass
        for v in vs_list:
            pass

    def theta_lift(self, prec=None, constant_term_weight_one=True):
        r"""
        Compute the (additive) theta lift.

        This computes the additive theta lift (e.g. Shimura lift; Doi--Naganuma lift; etc) of the given vector-valued modular form.

        INPUT:
        - ``prec`` -- max precision (default None). (This is limited by the precision of the input. If prec is None then we compute as much as possible.)
        - ``constant_term_weight_one`` -- boolean (default True) for internal use. If False then we don't bother correcting the constant term when the result has weight 1.

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: WeilRep(matrix([[-2, 0, 0], [0, 2, 1], [0, 1, 2]])).cusp_forms_basis(11/2, 5)[0].theta_lift()
            t + ((-6*r_0^-1 - 6)*x^-1 + (-6*r_0^-1 + 12 - 6*r_0) + (-6 - 6*r_0)*x)*t^2 + ((15*r_0^-2 + 24*r_0^-1 + 15)*x^-2 + (24*r_0^-2 - 24*r_0^-1 - 24 + 24*r_0)*x^-1 + (15*r_0^-2 - 24*r_0^-1 + 162 - 24*r_0 + 15*r_0^2) + (24*r_0^-1 - 24 - 24*r_0 + 24*r_0^2)*x + (15 + 24*r_0 + 15*r_0^2)*x^2)*t^3 + O(t^4)

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[2]]))
            sage: (w + II(1) + II(4)).modular_forms_basis(1/2, 15)[0].theta_lift()
            -1/4 - q - s - q^2 + (-r^-2 - 2 - r^2)*q*s - s^2 + (-2*r^-2 - 2*r^2)*q^2*s + (-2*r^-2 - 2*r^2)*q*s^2 - q^4 + (-r^-4 - 2 - r^4)*q^2*s^2 - s^4 + (-2)*q^5 + (-r^-4 - 2 - r^4)*q^4*s + (-r^-4 - 2 - r^4)*q*s^4 + (-2)*s^5 + (-2*r^-4 - 2*r^-2 - 2*r^2 - 2*r^4)*q^5*s + (-2*r^-4 - 2*r^4)*q^4*s^2 + (-2*r^-4 - 2*r^4)*q^2*s^4 + (-2*r^-4 - 2*r^-2 - 2*r^2 - 2*r^4)*q*s^5 + (-2*r^-6 - 2*r^-2 - 2*r^2 - 2*r^6)*q^5*s^2 + (-2*r^-6 - 2*r^-2 - 2*r^2 - 2*r^6)*q^2*s^5 + O(q, s)^8
        """
        w = self.weilrep()
        extra_plane = w.is_lorentzian_plus_II()
        s = w.lift_qexp_representation
        if w._lifts_have_fourier_jacobi_expansion():
            S = w.gram_matrix()
            A = identity_matrix(S.nrows())
            if extra_plane:
                A[-2, 1] = -1
                N = w._N()
                S = A.transpose() * S * A
                w = WeilRepLorentzianPlusII(S, S[1:-1, 1:-1], N)
            else:
                A[-1, 0] = -1
                w = WeilRepLorentzian(A.transpose() * S * A)
            w.lift_qexp_representation = s
            X = self.conjugate(A, w=w)
        else:
            X = self
        prec0 = self.precision()
        val = self.valuation()
        if val < 0:
            raise ValueError('Nonholomorphic input function in theta lift.')
        if prec is None:
            prec = isqrt(4 * prec0)
        else:
            prec = min(prec, isqrt(4 * prec0))
        wt = self.weight()
        coeffs = X.coefficients()
        S = w._lorentz_gram_matrix()
        s_0 = w.orthogonalized_gram_matrix()
        if self.is_symmetric() == 1:
            eps = 1
        else:
            eps = -1
        nrows = Integer(s_0.nrows())
        k = wt + nrows/2 - 1
        C = 0
        if k == 1 and constant_term_weight_one:
            if val < 0:
                return NotImplemented
            try:
                C = self._weight_one_theta_lift_constant_term()
            except IndexError:
                print('Warning: I could not find the correct constant term! Please use a higher precision.')  # WARNING: we will keep computing!! even though the output is almost certainly wrong!
        elif k <= 0:
            return NotImplemented
        N = w._N()
        if N <= 2:
            K = QQ
            if N == 1:
                zeta = 1
            else:
                zeta = -1
        else:
            K = CyclotomicField(N, 'mu%d' % N)
            zeta, = K.gens()
        if nrows > 1:
            if nrows > 2:
                rb = LaurentPolynomialRing(K, [f'r_{i}' for i in range(nrows - 2)])
            else:
                rb = K
            rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        else:
            rb_x = K
            x = 1
        extra_plane = False
        t, = PowerSeriesRing(rb_x, 't').gens()
        a = w.change_of_basis_matrix()
        a_tr = a.transpose()
        scale = Integer(a.determinant())
        b_norm = s_0[0, 0]
        s_0 = s_0[1:, 1:]
        s_0inv = s_0.inverse()
        new_prec = ceil(prec * prec * scale * scale / (-4 * b_norm) + prec)
        if nrows >= 3:
            v_matrix = _, _, vs_matrix = pari(s_0inv).qfminim(new_prec, flag=2)
            vs_list = vs_matrix.sage().columns()
            rb0 = rb.gens()[0]
        elif nrows == 2:
            vs_list = [vector([n]) for n in range(1, isqrt(4 * new_prec * s_0[0, 0]))]
        else:
            vs_list = []
        lift = O(t ** prec)
        negative = lambda v: next(s for s in v if s) < 0
        if not w.is_lorentzian():
            extra_plane = True
            if k % 2 == 0:
                y, = PolynomialRing(QQ, 'y').gens()
                bp = bernoulli_polynomial(y, k)
                for i in srange(N):
                    zeta_i = zeta ** i
                    try:
                        c = coeffs[tuple([i / N] + [0]*(nrows + 2))]
                        if c:
                            lift -= c * sum([bp(j / N) * (zeta_i ** j) for j in srange(1, N + 1)])
                    except KeyError:
                        pass
                lift *= (N ** (k - 1)) / (k + k)
        elif k % 2 == 0:
            try:
                c = coeffs[tuple([0]*(nrows + 1))]
                if c:
                    lift -= c * bernoulli(k) / (k + k)
            except KeyError:
                pass
        for v in vs_list:
            sv = s_0inv * v
            j = 1
            while j < prec:
                prec_j = prec//j + 1
                v_big = vector([-j / b_norm] + list(sv))
                z = a_tr * v_big
                v_big_2 = vector([-j / b_norm] + list(-sv))
                sz = S * z
                z_2 = a_tr * v_big_2
                sz_2 = S * z_2
                m = x ** v[0]
                if nrows >= 3:
                    if nrows >= 4:
                        m *= rb.monomial(*v[1:])
                    else:
                        m *= rb0 ** v[1]
                if negative(sz):
                    norm_z = z * sz / 2
                    if extra_plane:
                        z = vector([0] + list(z) + [0])
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        lift += c * sum([n ** (k - 1) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                    except KeyError:
                        if -norm_z >= prec0:
                            prec = j
                            break
                        pass
                    if extra_plane:
                        for i in srange(1, N):
                            z[0] = i / N
                            zeta_i = zeta ** i
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                lift += c * sum([n ** (k - 1) * (zeta_i ** n) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                            except KeyError:
                                pass
                if negative(sz_2):
                    z = z_2
                    sz = sz_2
                    norm_z = z * sz / 2
                    m = ~m
                    if extra_plane:
                        z = vector([0] + list(z) + [0])
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        lift += c * sum([n ** (k - 1) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                    except KeyError:
                        if -norm_z >= prec0:
                            prec = j
                            break
                        pass
                    if extra_plane:
                        for i in srange(1, N):
                            z[0] = i / N
                            zeta_i = zeta ** i
                            try:
                                c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                                lift += c * sum([n ** (k - 1) * (zeta_i ** n) * (m ** n) * t ** (n * j) for n in srange(1, prec_j)])
                            except KeyError:
                                pass
                j += 1
        v_big = vector([QQ(0)] * nrows)
        for j in srange(1, prec):
            v_big[0] = -j / b_norm
            z = a_tr * v_big
            sz = S * z
            norm_z = z * sz / 2
            if extra_plane:
                z = vector([0] + list(z) + [0])
                for i in srange(N):
                    z[0] = i / N
                    try:
                        c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                        lift += c * sum([n ** (k - 1) * (zeta ** (i * n)) * t ** (n * j) for n in srange(1, prec//j + 1)])
                    except KeyError:
                        pass
            else:
                try:
                    c = coeffs[tuple([frac(y) for y in z] + [-norm_z])]
                    lift += c * sum([n ** (k - 1) * t ** (n * j) for n in srange(1, prec//j + 1)])
                except KeyError:
                    if -norm_z >= self.precision():
                        prec = j
                        break
                    pass
        if eps == -1 and extra_plane and N >= 3:
            lift /= sum(zeta**i - zeta**(-i) for i in range(1, (N + 1)//2))
        return OrthogonalModularForm(k, w, lift + C + O(t ** prec), scale=1, weylvec=vector([0]*nrows), qexp_representation=w.lift_qexp_representation)


def _lorentz_laplacian(f):
    r"""
    Apply the Laplace operator.

    WARNING: the Laplace operator does not act on modular forms!
    It is only used to define Rankin--Cohen brackets.
    """
    from weilrep.lifts import OrthogonalModularForm
    w = f.weilrep()
    S = w.orthogonalized_gram_matrix()
    d = f.true_coefficients()
    h = f.true_fourier_expansion()
    rt, t = h.parent().objgen()
    rx, x = h.base_ring().objgen()
    if x != 1:
        r = rx.base_ring()
        rgens = r.gens()
    s = rt(0)
    S_inv = S.inverse()
    scale = f.scale()
    two_scale_sqr = 2 * scale * scale
    for v, c in d.items():
        v = vector(scale * v)
        v0 = v[0]
        monom = t**v0
        if len(v) > 1:
            v1 = v[1]
            monom *= x**v1
            if len(v) > 2:
                monom *= prod(rgens[i]**v for i, v in enumerate(u))
        s += c * (v*S_inv*v/two_scale_sqr) * monom
    return OrthogonalModularForm(f.weight() + 2, w, s.add_bigoh(scale * f.precision()), scale, f.weyl_vector(), qexp_representation=f.qexp_representation())
