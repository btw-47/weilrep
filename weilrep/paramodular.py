r"""

Paramodular forms of degree two

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



class ParamodularForms(OrthogonalModularFormsPositiveDefinite):

    r"""
    Paramodular forms of level N.

    INPUT: call ``ParamodularForms(N)`` where
    - ``N`` -- the level
    """

    def __init__(self, N):
        S = matrix([[Integer(N + N)]])
        w = WeilRep(S)
        w.lift_qexp_representation = 'siegel'
        OrthogonalModularFormsPositiveDefinite.__init__(self, w)
        self.__class__ = ParamodularForms

    def __repr__(self):
        return 'Paramodular forms of level %d'%self.level()

    def level(self):
        return self.gram_matrix()[0, 0] / 2

    def hecke_operator(self, p, d=1):
        return ParamodularHeckeOperator(self, p, d)

    def eigenforms(self, X, _p = 2, _name = '', _final_recursion = True, _K_list = []):
        r"""
        Decompose a space X into common eigenforms of the Hecke operators.

        This will raise a ValueError if X is not invariant under the Hecke operators.
        """
        while self.level() % _p == 0:
            _p = next_prime(_p)
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
                K = M.base_ring()
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
            x.__class__ = ParamodularEigenform
            x._ParamodularEigenform__field = K_list[i]
            eigenforms.append(x)
        return eigenforms

class ParamodularForm(OrthogonalModularFormPositiveDefinite):

    def level(self):
        return Integer(self.gram_matrix()[0, 0] / 2)

    def __getitem__(self, a):
        r"""
        Extract Fourier coefficients.

        The exponent should be symmetric half-integral (in the case of trivial character); i.e. integral diagonal and half-integral off-diagonal entries.

        EXAMPLES::

            sage: from weilrep import *
            sage: f = ParamodularForms(1).eisenstein_series(4, 5)
            sage: A = matrix([[2, 1/2], [1/2, 1]])
            sage: f[A]
            138240
        """
        def paramodular_reduce(N, a, b, c):
            r"""
            reduction of binary quadratic forms under Gamma0(N)
            """
            if a <= 0:
                return (a, b, c)
            if a > c:
                return paramodular_reduce(N, c, b, a)   ## for some reason, calling paramodular_reduce(N, c, -b, a) here is incorrect
            if abs(b) > N * a:
                k = floor(1/2 - b / (2 * N * a))
                return paramodular_reduce(N, a, b + 2*N*k*a, c + k*b + N*k*k*a)
            return (a, b, c)
        try:
            a = a.list()
        except AttributeError:
            pass
        a, b, _, d = a
        a1, b1, d1 = a, b, d
        N = self.level()
        s = self.scale()
        b = Integer(2 * b * s)
        a = Integer(a * s / N)
        d *= s
        a, b, d = paramodular_reduce(N, a, b, d)
        a, d = a + d, a - d
        f = self.true_fourier_expansion()
        try:
            return f[a][d][b]
        except KeyError:
            return 0

    def hecke_operator(self, p, d=1):
        r"""
        Compute the image under the Hecke operator T_{p, d}.

        If d = 1, then T_{p, 1} is the Hecke operator associated to the double coset of diag(1, 1, p, p).
        If d = 2, then T_{p, 2} is the Hecke operator associated to the double coset of diag(1, p, p^2, p).

        'd' can theoretically be any natural number, but using d >= 3 would be silly.
        """
        p = Integer(p)
        if not p.is_prime():
            raise ValueError('Hecke operators are only implemented for prime index')
        N = self.level()
        if N % p == 0:
            raise ValueError('Hecke operators are not implemented for primes dividing the level')
        k = self.weight()
        f = self.true_fourier_expansion()
        prec = Integer(self.precision())
        r = f.parent()
        r0 = r.base_ring()
        r00 = r0.base_ring()
        bound = ceil(prec / (p))
        h = r(0)
        t, = r.gens()
        x, = r0.gens()
        _r, = r00.gens()
        p_pow_1 = p**(k - 2)
        p_pow_2 = p * p_pow_1 * p_pow_1
        if d == 1:
            lower_bd = ceil(-p / 2)
            upper_bd = ceil(p / 2)
            U = [matrix([[p, 0], [0, 1]])] + [matrix([[1, 0], [N * j, p]]) for j in srange(lower_bd, upper_bd)]
            def hecke_operator_coefficient(a, b, c):
                nonlocal bound
                M = matrix([[N * (a - b) / 2, c / 2], [c / 2, (a + b) / 2]])
                n = self.__getitem__(p * M)
                try:
                    n += p_pow_2 * self.__getitem__(M / p)#f[a / p][b / p][c / p]
                except TypeError:
                    pass
                for u in U:
                    M_u = u.transpose() * M * u / p
                    a1, b1 = M_u.rows()[0]
                    _, c1 = M_u.rows()[1]
                    if a1 in ZZ and b1 + b1 in ZZ and c1 in ZZ:
                        try:
                            n += p_pow_1 * self.__getitem__(M_u)
                        except IndexError:
                            bound = a
                            return 0
                return n
        else:
            U = []
            for i in range(d + 1):
                U_i = []
                lower_bd = ceil(-p**i / 2)
                upper_bd = ceil(p**i / 2)
                for j in range(lower_bd, upper_bd):
                    U_i.append(matrix([[1, 0], [N * j, 1]]))
                if i > 0:
                    p_i = p**(i - 1)
                    for j in range(p_i):
                        j1 = j
                        while gcd(j1, N) > 1:
                            j1 = j1 - p_i
                        _, a, b = xgcd(p * j1, N)
                        U_i.append(matrix([[p * j1, b], [N, a]]))
                U.append(U_i)
            def hecke_operator_coefficient(a, b, c):
                nonlocal bound
                M = matrix([[N * (a - b) / 2, c / 2], [c / 2, (a + b) / 2]])
                n = 0
                for beta in range(d + 1):
                    for gamma in range(d + 1 - beta):
                        p_factor = p_pow_1 ** beta * p_pow_2 ** gamma
                        alpha = d - (beta + gamma)
                        for u in U[beta]:
                            M_u = u.transpose() * M * u
                            a1, b1 = M_u.rows()[0]
                            _, c1 = M_u.rows()[1]
                            if a1.valuation(p) >= (beta + gamma) and (b1 + b1).valuation(p) >= gamma and c1.valuation(p) >= gamma:
                                a2 = Integer(a1 * p**(alpha - beta - gamma))
                                b2 = b1 * p**(alpha - gamma)
                                c2 = c1 * p**(alpha + beta - gamma)
                                try:
                                    n += p_factor * self.__getitem__(matrix([[a2, b2], [b2, c2]]))
                                except IndexError:
                                    bound = a
                                    return 0
                return n
        a = -Integer(1)
        while a < bound:
            a += 1
            for b in srange(-a - 1, a + 1):
                if a % 2 == b % 2:
                    rbound = isqrt(N * (a * a - b * b)) + 1
                    for c in srange(-rbound, rbound):
                        try:
                            h += hecke_operator_coefficient(a, b, c) * t**a * x**b * _r**c
                        except IndexError:
                            bound = a
                            break
        h = h.add_bigoh(bound)
        return OrthogonalModularForm(k, self.weilrep(), h, 1, vector([0] * 3), qexp_representation = 'siegel')

    def level(self):
        return self.gram_matrix()[0, 0] / 2

    def level_raising_operator_1(self, m):
        r"""
        Apply the level-raising operator
        \theta : M_k(K(N)) --> M_k(K(m * N))

        REFERENCE: Roberts, Schmidt - On modular forms for the paramodular group
        """
        fj = self.fourier_jacobi()
        return m * ParamodularForms(m * self.level()).modular_form_from_fourier_jacobi_expansion([x.hecke_V(m) for x in fj])

    def level_raising_operator_2(self, m):
        r"""
        Apply the level-raising operator
        \theta' : M_k(K(N)) --> M_k(K(p * N))

        REFERENCE: Roberts, Schmidt - On modular forms for the paramodular group
        """
        fj = self.fourier_jacobi()
        a = len(fj)

    def make_eigenform(self):
        self.__class__ = ParamodularEigenform
        self._ParamodularEigenform__field = self.true_fourier_expansion().base_ring().base_ring().base_ring()


class ParamodularEigenform(ParamodularForm):

    def hecke_field(self):
        r"""
        Field of definition.
        This is the field generated by self's Hecke eigenvalues.
        """
        return self.__field

    def eigenvalue(self, p, d=1):
        f = self.true_fourier_expansion()
        a, x = next(x for x in enumerate(f.list()) if x[1])
        a = Integer(a)
        xd = x.dict()
        b = min([abs(b) for b in xd.keys() if xd[b]])
        b = Integer(b)
        x = xd[b]
        c, n = next(iter(x.dict().items()))
        c = Integer(c)
        N = self.level()
        A = matrix([[N * (a - b) / 2, c/2], [c/2, (a + b) / 2]])
        h = ParamodularForms(self.level()).hecke_operator(p, d=d)._get_coefficient(self, A)
        return h / n

    def spinor_euler_factor(self, p):
        r"""
        Compute the Euler factor at the prime 'p' in self's spinor zeta function.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = ParamodularForms(1)
            sage: e4, e6, e12 = [m.eisenstein_series(k, 10) for k in [4, 6, 12]]
            sage: X = m.eigenforms([e4^3, e6^2, e12])
            sage: X[1].spinor_euler_factor(2).factor()
            (4398046511104) * (X - 1/1024) * (X - 1/2048) * (X^2 + 9/65536*X + 1/2097152)
        """
        K = self.hecke_field()
        r = PolynomialRing(K, 'X')
        eigenvalue_1 = self.eigenvalue(p)
        eigenvalue_2 = self.eigenvalue(p, d=2)
        k = self.weight()
        return r([1, -eigenvalue_1, eigenvalue_1**2 - eigenvalue_2 - p**(2 * k - 4), -eigenvalue_1 * p**(2 * k - 3), p**(4 * k - 6)])

    def standard_euler_factor(self, p):
        r"""
        Compute the Euler factor at the prime 'p' in self's standard zeta function.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = ParamodularForms(1)
            sage: e4, e6, e12 = [m.eisenstein_series(k, 10) for k in [4, 6, 12]]
            sage: X = m.eigenforms([e4^3, e6^2, e12])
            sage: X[1].standard_euler_factor(2).factor()
            (-1) * (X - 1) * (X^2 + 9/64*X + 1/2) * (X^2 + 9/32*X + 2)
        """
        f = self.spinor_euler_factor(p)
        L = f.splitting_field('a')
        fL = f.change_ring(L)
        R = fL.roots()
        X, = fL.parent().gens()
        A = []
        for r, n in R:
            A.extend([r]*n)
        R = A
        for i in range(4):
            for j in range(i+1, 4):
                a, = [(m, n) for m in range(4) for n in range(4) if m not in [i, j] and n not in [i, j] and m < n]
                m, n = a
                if R[i] * R[j] == R[m] * R[n]:
                    g = (1 - X * R[m] / R[i]) * (1 - X * R[i] / R[m]) * (1 - X * R[m] / R[j]) * (1 - X * R[j] / R[m])
                    return (1 - f.parent().gens()[0]) * f.parent()(g)
        raise RuntimeError


class ParamodularHeckeOperator:
    r"""
    Hecke operators on paramodular forms.

    NOTE: this might only work when the paramodular level is a prime
    """
    def __init__(self, m, p, d):
        self.__index = p
        self.__degree = d
        self.__omf = m
        self.__level = m.level()

    def __repr__(self):
        return "Hecke operator of index %s acting on paramodular forms of level %s"%(self.__index, self.__level)

    def degree(self):
        return self.__degree

    def index(self):
        return self.__index

    def level(self):
        return self.__level

    def __call__(self, f):
        return f.hecke_operator(self.__index, self.__degree)

    def _get_coefficient(self, f, M):
        N = self.level()
        d = self.__degree
        p = self.__index
        k = f.weight()
        p_pow_1 = p**(k - 2)
        p_pow_2 = p * p_pow_1 * p_pow_1
        if d == 1:
            lower_bd = ceil(-p / 2)
            upper_bd = ceil(p / 2)
            U = [matrix([[p, 0], [0, 1]])] + [matrix([[1, 0], [N * j, p]]) for j in srange(lower_bd, upper_bd)]
            try:
                n = f.__getitem__(p * M)
            except IndexError:
                raise ValueError('Unknown coefficient') from None
            try:
                n += p_pow_2 * f.__getitem__(M / p)
            except TypeError:
                pass
            for u in U:
                M_u = u.transpose() * M * u / p
                a1, b1 = M_u.rows()[0]
                _, c1 = M_u.rows()[1]
                if a1 in ZZ and c1 in ZZ and (b1 + b1) in ZZ:
                    try:
                        n += p_pow_1 * f.__getitem__(M_u)
                    except IndexError:
                        raise ValueError('Unknown coefficient') from None
            return n
        else:
            U = []
            for i in range(d + 1):
                U_i = []
                lower_bd = ceil(-p**i / 2)
                upper_bd = ceil(p**i / 2)
                for j in range(lower_bd, upper_bd):
                    U_i.append(matrix([[1, 0], [N * j, 1]]))
                if i > 0:
                    p_i = p**(i - 1)
                    for j in range(p_i):
                        j1 = j
                        while gcd(j1, N) > 1:
                            j1 = j1 - p_i
                        _, u, v = xgcd(p * j1, N)
                        U_i.append(matrix([[p * j1, v], [N, u]]))
                U.append(U_i)
            n = 0
            for beta in range(d + 1):
                for gamma in range(d + 1 - beta):
                    p_factor = p_pow_1 ** beta * p_pow_2 ** gamma
                    alpha = d - (beta + gamma)
                    for u in U[beta]:
                        M_u = u.transpose() * M * u
                        a1, b1 = M_u.rows()[0]
                        _, c1 = M_u.rows()[1]
                        if a1.valuation(p) >= (beta + gamma) and (b1 + b1).valuation(p) >= gamma and c1.valuation(p) >= gamma:
                            a2 = Integer(a1 * p**(alpha - beta - gamma))
                            b2 = b1 * p**(alpha - gamma)
                            c2 = c1 * p**(alpha + beta - gamma)
                            try:
                                n += p_factor * f.__getitem__(matrix([[a2, b2], [b2, c2]]))
                            except IndexError:
                                raise ValueError('Unknown coefficient') from None
            return n

    def matrix(self, X):
        p = self.__index
        d = self.__degree
        L = []
        R = []
        rank = 0
        target_rank = len(X)
        Xref = X[0]
        F = [x.true_fourier_expansion() for x in X]
        prec = Xref.precision()
        N = self.level()
        for a in srange(prec):
            for b in srange(-a - 1, a + 1):
                if a % 2 == b % 2:
                    rbound = isqrt(N * (a * a - b * b)) + 1
                    for c in srange(-rbound, rbound):
                        A = matrix([[N * (a - b) / 2, c / 2], [c/2, (a + b)/2]])
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