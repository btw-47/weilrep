r"""

Hermitian modular forms of degree two

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2020-2024 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import math

from sage.arith.misc import gcd, next_prime, xgcd
from sage.arith.srange import srange
from sage.functions.other import ceil, floor
from sage.matrix.constructor import matrix
from sage.misc.functional import denominator, isqrt
from sage.modular.arithgroup.congroup_gamma0 import is_Gamma0
from sage.modules.free_module_element import vector
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import NumberField
from sage.rings.polynomial.polydict import ETuple
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian
from .positive_definite import OrthogonalModularFormPositiveDefinite, OrthogonalModularFormsPositiveDefinite
from .weilrep import WeilRep
from .weilrep_misc import relations

sage_one_half = Integer(1) / Integer(2)


class HermitianModularForms(OrthogonalModularFormsPositiveDefinite):

    def __init__(self, K, level=1):
        d = K.discriminant()
        self.__base_field = K
        self.__discriminant = d
        self.__level = level
        if d % 2:
            S = matrix([[2, 1], [1, Integer((1 - d) // 2)]])
        else:
            S = matrix([[2, 0], [0, Integer(-d // 2)]])
        w = WeilRep(S)
        if level != 1:
            w += II(level)
            self.__class__ = HermitianModularFormsWithLevel
            self._OrthogonalModularFormsLorentzian__gram_matrix = w.gram_matrix()
        self._OrthogonalModularForms__gram_matrix = S
        self._OrthogonalModularForms__weilrep = w
        w.lift_qexp_representation = 'hermite', K, level

    def __repr__(self):
        return 'Hermitian modular forms of degree two over %s'%str(self.__base_field)

    def base_field(self):
        return self.__base_field

    def level(self):
        return self.__level

    def hecke_operator(self, p):
        return HermitianHeckeOperator(self, p)

    def eigenforms(self, X, _p = 2, _name = '', _final_recursion = True, _K_list = []):
        r"""
        Decompose a space X into common eigenforms of the Hecke operators.

        This will raise a ValueError if X is not invariant under the Hecke operators.
        """
        _p = Integer(_p)
        while self.__discriminant % _p == 0:
            _p = next_prime(_p)
        K0 = self.base_field()
        L = K0.factor(_p)
        if len(L) > 1:
            (pi, _), _ = L
            if pi.is_principal():
                _p, = pi.gens_reduced()
        T = self.hecke_operator(_p)
        M = T.matrix(X)
        chi = M.characteristic_polynomial()
        F = chi.factor()
        L = []
        i = 0
        K_list = []
        chi_list = []
        for x, n in F:
            if x.degree() > 1:
                name = 'a_%s%s'%(_name, i)
                K = NumberField(x, name)
                i += 1
            else:
                K = QQ
            M_K = matrix(K, M)
            V = x(M_K).transpose().kernel().basis_matrix()
            V_rows = V.rows()
            if n == 1:
                if len(V_rows) > 1:
                    P = matrix(K, [V.solve_left(M_K * v) for v in V_rows])
                    for p in P.eigenvectors_left(extend = False):
                        c = p[0].charpoly()
                        if c not in chi_list:
                            L.append(vector(p[1][0]) * V)
                            K_list.append(K)
                            chi_list.append(c)
                else:
                    L.append(V_rows[0])
                    K_list.append(K)
            else:
                _name = _name + '%s_'%i
                K_list_2, eigenvectors = self.eigenforms(V_rows, _p = next_prime(_p), _name = _name, _final_recursion = False, _K_list = K_list)
                K_list.extend(K_list_2)
                L.extend(eigenvectors)
        L = [sum(X[i] * y for i, y in enumerate(x)) for x in L]
        eigenforms = []
        for i, x in enumerate(L):
            a, y = next(y for y in enumerate(x.true_fourier_expansion().list()) if y[1])
            a = Integer(a)
            yd = y.dict()
            b = min([abs(b) for b in yd.keys() if yd[b]])
            b = Integer(b)
            y = yd[b]
            c, n = next(iter(y.dict().items()))
            x /= n
            x.__class__ = HermitianEigenform
            x._HermitianEigenform__field = K_list[i]
            eigenforms.append(x)
        return eigenforms


class HermitianModularForm(OrthogonalModularFormPositiveDefinite):

    r"""
    Hermitian modular forms.

    INPUT: call ``HermitianModularForms(K, level)`` where
    - ``K`` -- the base field (an imaginary quadratic number field)
    - ``level`` -- level (default 1)
    """

    def __init__(self, base_field, level = 1):
        self.__base_field = base_field

    def __repr__(self):
        r"""
        Represent self's Fourier expansion as a power series c(a,b,c) q^a r1^v r2^(v') s^c, where v runs through elements in the dual of the ring of integers.
        """
        K = self.__base_field
        h = self.fourier_expansion()
        S = self.gram_matrix()
        S_inv = S.inverse()
        hprec = h.prec()
        d = self.scale()
        if h:
            D = K.discriminant()
            sqrtD = K(D).sqrt()
            if D % 4:
                omega = (1 + sqrtD) / 2
                omega_c = (1 - sqrtD) / 2
            else:
                omega = sqrtD / 2
                omega_c = -omega
            s = ''
            sign = False
            hdict = h.dict()
            for i in h.exponents():
                p = hdict[i]
                a = Integer(i[0])/d
                c = Integer(i[1])/d
                if a:
                    if a != 1:
                        if a.is_integer():
                            q_power = 'q^%s'%a
                        else:
                            q_power = 'q^(%s)'%a
                    else:
                        q_power = 'q'
                else:
                    q_power = ''
                if c:
                    if c != 1:
                        if c.is_integer():
                            s_power = 's^%s'%c
                        else:
                            s_power = 's^(%s)'%c
                    else:
                        s_power = 's'
                else:
                    s_power = ''
                z = (a or c)
                for (b1, b2), C in p.dict().items():
                    if C:
                        v = vector([b1, b2]) * S_inv
                        r1exp = v[0] + v[1] * omega
                        r2exp = v[0] + v[1] * omega_c
                        coef = True
                        if sign:
                            if C > 0 and C!= 1:
                                s += ' + ' + str(C)
                            elif C + 1 and C!= 1:
                                s += ' - ' + str(-C)
                            elif C + 1:
                                s += ' + '
                                coef = False
                            elif C - 1:
                                s += ' - '
                                coef = False
                        else:
                            if abs(C) != 1 or not z:
                                s += str(C)
                            else:
                                coef = False
                                if C == -1:
                                    s += '-'
                            sign = True
                        if r1exp:
                            if coef:
                                  s += '*'
                            if r1exp != r2exp or not r1exp.is_integer():
                                  s += q_power+'*r1^(%s)*r2^(%s)*'%(r1exp, r2exp)+s_power
                            elif r1exp != 1:
                                  s += q_power+'*r1^%s*r2^%s*'%(r1exp, r2exp)+s_power
                            else:
                                  s += q_power+'*r1*r2*'+s_power
                        else:
                            if coef and z:
                                s += '*'
                            if a:
                                s += q_power
                                if c:
                                    s += '*'
                            s += s_power
            if hprec % d:
                self.__string = s + ' + O(q, s)^(%s)'%(hprec/d)
            else:
                self.__string = s + ' + O(q, s)^%s'%(hprec/d)
        else:
            if hprec % d:
                self.__string = 'O(q, s)^(%s)'%(hprec/d)
            else:
                self.__string = 'O(q, s)^%s'%(hprec/d)
        return self.__string

    ## basic attributes

    def base_field(self):
        return self.__base_field

    def discriminant(self):
        return self.base_field().discriminant()

    def hmf(self):
        return HermitianModularForms(self.__base_field)

    def nvars(self):
        return 4

    ## get coefficients

    def __getitem__(self, a):
        r"""
        Extract Fourier coefficients.

        The exponent should be hermitian and (in the case of trivial character) have integral diagonal and off-diagonal entries contained in the dual lattice O_K'.

        EXAMPLES::

            sage: from weilrep import *
            sage: K.<i> = NumberField(x*x + 1)
            sage: h = HermitianModularForms(K)
            sage: f = h.eisenstein_series(4, 5)
            sage: A = matrix([[1, 1/2 + i/2], [1/2 - i/2, 1]])
            sage: f[A]
            2880
        """
        try:
            a = a.list()
        except AttributeError:
            pass
        a, b, _, d = a
        S = self.gram_matrix()
        try:
            b, c = b.parts()
            if S[0, 1]:
                b, c = b + b, b + c * (2 * S[1, 1] - 1)
            else:
                b, c = b + b, c * S[1, 1]
        except AttributeError:
            if S[0, 1]:
                b, c = b + b, b
            else:
                b, c = b + b, 0
        f = self.true_fourier_expansion()
        s = self.scale()
        try:
            return f[Integer((a + d) * s)][Integer((a - d) * s)].dict()[ETuple([Integer(b * s), Integer(c * s)])]
        except KeyError:
            return 0

    def restrict_to_siegel(self):
        r"""
        Restrict the Hermitian modular form to the Siegel upper half-space. The result is a Siegel modular form.
        """
        f = self.pullback(vector([1, 0]))
        return OrthogonalModularForm(self.weight(), f.weilrep(), f.true_fourier_expansion(), f.scale(), f.weyl_vector(), qexp_representation = 'siegel')

    ## hecke operators

    def hecke_operator(self, p):
        K = self.base_field()
        prec = self.precision()
        if p not in ZZ:
            coefficient = lambda A: hecke_coeff_split(self, p, A)
            bd = p.norm()
        else:
            p = ZZ(p)
            bd = p
            if p.is_prime():
                coefficient = lambda A: hecke_coeff(self, p, A)
            else:
                coefficient = lambda A: hecke_coeff_inert(self, isqrt(p), A)
        bd = floor(prec / bd)
        f = self.true_fourier_expansion()
        r, t = f.parent().objgen()
        x, = r.base_ring().gens()
        r0, r1 = x.base_ring().gens()
        S = self.gram_matrix()
        D = K.discriminant()
        sqrtD = K(D).sqrt()
        if D % 4:
            def convert_exponent(b, c):
                return b / 2 + sqrtD * (b - 2*c) / (4 * S[1, 1] - 2), b / 2 - sqrtD * (b - 2*c) / (4 * S[1, 1] - 2)
        else:
            sqrtD /= 2
            def convert_exponent(b, c):
                return b / 2 + sqrtD * c / S[1, 1], b / 2 - sqrtD * c / S[1, 1]
        f_img = r(0).add_bigoh(bd)
        for a in srange(bd):
            h = f[a]
            for d, g in h.dict().items():
                d = ZZ(d)
                for (b, c), N in g.dict().items():
                    b = ZZ(b)
                    c = ZZ(c)
                    u, v = convert_exponent(b, c)
                    A = matrix(K, [[(a + d) / 2, u], [v, (a - d) / 2]])
                    try:
                        C = coefficient(A)
                        f_img += C * t**a * x**d * r0**b * r1**c
                    except IndexError:
                        bd = a
                        f_img = f_img.add_bigoh(a)
                        return OrthogonalModularForm(self.weight(), self.weilrep(), f_img, self.scale(), self.weyl_vector(), qexp_representation = self.qexp_representation())
        return OrthogonalModularForm(self.weight(), self.weilrep(), f_img, self.scale(), self.weyl_vector(), qexp_representation = self.qexp_representation())

class HermitianEigenform(HermitianModularForm):

    def hecke_field(self):
        r"""
        This is the number field generated by self's Hecke eigenvalues.
        """
        return self.__field

    def eigenvalue(self, p):
        r"""
        Compute self's eigenvalue under the Hecke operator.
        """
        K = self.base_field()
        f = self.true_fourier_expansion()
        S = self.gram_matrix()
        S_inv = S.inverse()
        t = self.hmf().hecke_operator(p)
        a, x = next(x for x in enumerate(f.list()) if x[1])
        a = Integer(a)
        xd = x.dict()
        b = min([abs(b) for b in xd.keys() if xd[b]])
        b = Integer(b)
        x = xd[b]
        c, n = next(iter(x.dict().items()))
        D = K.discriminant()
        sqrtD = K(D).sqrt()
        if D % 4:
            omega = (1 + sqrtD) / 2
            omega_c = (1 - sqrtD) / 2
        else:
            omega = sqrtD / 2
            omega_c = -omega
        v = vector(c) * S_inv
        r1exp = v[0] + v[1] * omega
        r2exp = v[0] + v[1] * omega_c
        n2 = t._get_coefficient(self, matrix([[Integer((a + b)/2), r1exp], [r2exp, Integer((a - b)/2)]]))
        return n2 / n

    def euler_factor(self, p):
        r"""
        Compute the Euler factor at the prime p in self's spinor zeta function.
        """
        K = self.base_field()
        L = K.ideal(p).factor()
        X, = PolynomialRing(self.hecke_field(), 'X').gens()
        k = self.weight()
        if len(L) == 2: #split case
            (pi_1, _), (pi_2, _) = L
            try:
                pi_1, = pi_1.gens_reduced()
                pi_2, = pi_2.gens_reduced()
                e1 = p**(2 - k) * self.eigenvalue(pi_1)
                e2 = p**(2 - k) * self.eigenvalue(pi_2)
                e = p**(4 - 2*k) * self.eigenvalue(p)
                f = 1 - e*X + (p**(1 - k) * e1 * e2 - p**(4 - (k+k))) * X**2 - (p**(3 - 2*k) * (e1*e1 + e2*e2) - 2 * e * p**(4 - (k+k))) * X**3 + p**(4 - (k+k)) * (p**(1 - k) * e1 * e2 - p**(4 - (k+k))) * X**4 - p**(8 - 4*k) * e * X**5 + p**(12 - 6*k) * X**6
                return f(p**(2 * k - 4) * X)
            except ValueError:
                raise ValueError('%s splits into non-principal ideals in %s'%(p, K)) from None
        else:
            _, n = L[0]
            if n > 1:
                raise NotImplementedError
        e1 = p**(4 - 2*k) * self.eigenvalue(p)
        e2 = p**(4 - 3*k) * self.eigenvalue(p * p)
        f = (1 - p**(4 - 2*k) * X**2) * (1 - e1 * X + (p * e2 + p**(1 - (k + k)) * (p**3 + p**2 - p + 1)) * X**2 - p**(4 - (k + k)) * e1 * X**3 + p**(8 - 4*k) * X**4)
        return f(p**(2 * k - 4) * X)

class HermitianHeckeOperator:
    r"""
    Hecke operators on Hermitian modular forms.
    (Only for non-ramified primes!)
    """
    def __init__(self, h, p):
        self.__hmf = h
        self.__index = p

    def __repr__(self):
        return 'Hecke operator of index %s acting on Hermitian modular forms over %s'%(self.index(), self.hmf().base_field())

    def hmf(self):
        return self.__hmf

    def index(self):
        return self.__index

    def __call__(self, f):
        return f.hecke_operator(self.__index)

    def _get_coefficient(self, f, A):
        p = self.index()
        if p not in ZZ:
            return hecke_coeff_split(f, p, A)
        elif p.is_prime():
            return hecke_coeff(f, p, A)
        return hecke_coeff_inert(f, isqrt(p), A)

    def matrix(self, X):
        r"""
        Compute the matrix representing this Hecke operator on a basis X.
        """
        L = []
        R = []
        rank = 0
        target_rank = len(X)
        Xref = X[0]
        F = [x.true_fourier_expansion() for x in X]
        prec = Xref.precision()
        f = Xref.true_fourier_expansion()
        K = self.hmf().base_field()
        S = self.hmf().gram_matrix()
        S_inv = S.inverse()
        D = K.discriminant()
        sqrtD = K(D).sqrt()
        if D % 4:
            omega = (1 + sqrtD) / 2
            omega_c = (1 - sqrtD) / 2
            def convert_exponent(b, c):
                return b / 2 + sqrtD * (b - 2*c) / (4 * S[1, 1] - 2), b / 2 - sqrtD * (b - 2*c) / (4 * S[1, 1] - 2)
        else:
            sqrtD /= 2
            omega = sqrtD
            omega_c = -omega
            def convert_exponent(b, c):
                return b / 2 + sqrtD * c / S[1, 1], b / 2 - sqrtD * c / S[1, 1]
        for a in srange(prec):
            for b in srange(-a - 1, a + 1):
                if a % 2 == b % 2:
                    g = f[a][b]
                    for u, v in g.dict().keys():
                        u = ZZ(u)
                        v = ZZ(v)
                        v = vector([u, v]) * S_inv
                        r1exp = v[0] + v[1] * omega
                        r2exp = v[0] + v[1] * omega_c
                        A = matrix(K, [[(a + b) / 2, r1exp], [r2exp, (a - b) / 2]])
                        try:
                            L1 = [x for x in L]
                            L1.append([self._get_coefficient(f, A) for f in X])
                            M = matrix(L1)
                            rank1 = M.rank()
                            if rank1 > rank:
                                L = L1
                                rank = rank1
                                R.append([f[A] for f in X])
                            if rank == target_rank:
                                return matrix(R).solve_right(matrix(L))
                        except (IndexError, ValueError):
                            pass
        raise ValueError('Insufficient precision') from None


## formulas for Hecke operators on U(2, 2)

def hecke_coeff(f, p, A):
    r"""
    Compute the coefficient of A in the image of f unter the Hecke operator T_p,
    i.e. the double coset of diag(1, 1, p, p).
    """
    k = f.weight()
    s = f[p * A]
    a = ZZ(A[0, 0])
    b = A[0, 1]
    c = ZZ(A[1, 1])
    K = f.base_field()
    d = K.discriminant()
    isqrtd = K(d).sqrt()
    O = K.maximal_order()
    if a % p == 0 and c % p == 0 and b * isqrtd / p in O:
        s = s + p**(k + k - 4) * f[A / p]
    r1 = 1
    if d % 4:
        r2 = (-1 + isqrtd)/2
    else:
        r2 = isqrtd
    if c % p == 0:
        a1, b1, c1 = (p * a, b, c / p)
        s = s + p**(k - 3) * f[matrix([[a1, b1], [b1.conjugate(), c1]])]
    bc = b.conjugate()
    fl = floor(p/2)
    if p == 2:
        max_range = 1
    else:
        max_range = fl + 1
    for x in range(-fl, max_range):
        for y in range(-fl, max_range):
            d = x*r1 + y*r2
            dc = d.conjugate()
            a1 = ZZ(a + d*b + dc*bc + d*dc*c)/p
            b1 = b + dc*c
            c1 = p*c
            if a1 in ZZ:
                a1, b1, c1 = (a1, b1, c1)
                A1 = matrix([[a1, b1], [b1.conjugate(), c1]])
                s = s + (p**(k - 3)) * f[A1]
            if d and ZZ(d*dc) % p == 0:
                a2 = p*a
                b2 = b+d*a
                c2 = ZZ(c + dc*b + d*bc + d*dc*a)/p
                if c2 in ZZ:
                    a2, b2, c2 = (a2, b2, c2)
                    A2 = matrix([[a2, b2], [b2.conjugate(), c2]])
                    s = s + (p**(k - 3)) * f[A2]
                if ZZ((1+d)*(1+dc)) % p == 0:
                    a3 = ZZ(a + b + bc + c)
                    b3 = b + c + a3 * d
                    a3 = p * a3
                    c3 = ZZ(c * (1 + d) * (1 + dc) + b * dc + bc * d + (a + b + bc)*d*dc) / p
                    if c3 in ZZ:
                        a3, b3, c3 = (a3, b3, c3)
                        A3 = matrix([[a3, b3], [b3.conjugate(), c3]])
                        s = s + (p**(k - 3)) * f[A3]
    return s

def hecke_coeff_inert(f, p, A):
    r"""
    Compute the coefficient of A in the image of f unter the Hecke operator T_{p^2},
    i.e. the double coset of diag(1, p, p^2, p).

    Here p should be a prime that is *inert* in the quadratic field K.
    """
    k = f.weight()
    s = 1 - p*p
    a = ZZ(A[0, 0])
    b = A[0, 1]
    c = ZZ(A[1, 1])
    K = f.base_field()
    d = K.discriminant()
    O = K.maximal_order()
    isqrtd = K(d).sqrt()
    if a % p:
        if c % p:
            s += -2
        else:
            s += (p - 2)
    elif c % p:
        s += (p - 2)
    else:
        s += (p + p - 2)
    r1 = 1
    if d % 4:
        r2 = (-1 + isqrtd)/2
    else:
        r2 = isqrtd
    fl = floor(p / 2)
    if p == 2:
        max_range = 1
    else:
        max_range = fl + 1
    h = f[matrix([[a, p*b], [p*b.conjugate(), p*p*c]])]
    if a % (p*p) == 0 and b * isqrtd / p in O:
        h += p**(2*k - 4) * f[matrix([[a / (p*p), b / p], [b.conjugate() / p, c]])]
    for x in range(-fl, max_range):
        for y in range(-fl, max_range):
            d = x * r1 + y * r2
            if d and ZZ(a + d*b + (d*b).conjugate() + d * d.conjugate() * c) % p == 0:
                s += p
            a1 = a
            b1 = (b - d.conjugate() * a) / p
            c1 = (c - d*b - (d*b).conjugate() + a*(d*d.conjugate())) / (p*p)
            if c1 in ZZ and b1 * isqrtd in O:
                M1 = matrix([[a1, b1], [b1.conjugate(), c1]])
                h = h + (p**(2*k - 4)) * f[M1]
            a2 = p*p*a
            b2 = p*(b - a * d.conjugate())
            c2 = c - b*d - (b*d).conjugate() + a*d*d.conjugate()
            M2 = matrix([[a2, b2], [b2.conjugate(), c2]])
            h = h + f[M2]
    return h + p**(k - 4) * s * f[A]

def hecke_coeff_split(f, pi, A):
    r"""
    Compute the coefficient of A in the image of f unter the Hecke operator T_{pi},
    i.e. the double coset of diag(1, pi, p, pi).

    Here p = (pi) * (pi conjugate)
    should be a prime that *splits* into *principal* ideals in the quadratic field K.
    """
    p = abs(pi.norm())
    K = pi.parent()
    k = f.weight()
    a = ZZ(A[0, 0])
    b = K(A[0, 1])
    bc = b.conjugate()
    c = ZZ(A[1, 1])
    d = K.discriminant()
    O = K.maximal_order()
    isqrtd = K(d).sqrt()
    r1 = 1
    if d % 4:
        r2 = (-1 + isqrtd)/2
    else:
        r2 = isqrtd
    fl = floor(p / 2)
    if p == 2:
        max_range = 1
    else:
        max_range = fl + 1
    a1, b1, c1 = (a, b * pi.conjugate(), p * c)
    s = f[matrix([[a1, b1], [b1.conjugate(), c1]])]
    b1 = b / pi
    c1 = c / p
    if b1 * isqrtd in O and c1 in ZZ:
        s = s + (p**(k - 2)) * f[matrix([[a, b1], [b1.conjugate(), c1]])]
    for d in range(-fl, max_range):
        d = -d
        s = s + f[matrix([[p * a, pi * (b - a * d)], [pi.conjugate() * (b.conjugate() - a * d), c - (d*b).trace() + a * d*d]])]
        a1 = (a + d * (b + bc) + c * d * d)/p
        b1 = (b + c*d) / pi.conjugate()
        c1 = c
        if b1 * isqrtd in O and a1 in ZZ:
            s = s + (p**(k - 2)) * f[matrix([[a1, b1], [b1.conjugate(), c1]])]
    return s