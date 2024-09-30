r"""

Lifts for Hilbert modular forms over real-quadratic fields

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

from sage.all import next_prime
from sage.arith.misc import bernoulli
from sage.arith.srange import srange
from sage.functions.other import ceil, floor, sqrt
from sage.matrix.constructor import matrix
from sage.misc.functional import isqrt
from sage.modules.free_module_element import vector
from sage.rings.big_oh import O
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.number_field.number_field import NumberField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing_generic
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ


from .lifts import OrthogonalModularForm
from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian
from .weilrep import WeilRep
# from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis


class HMFCharacter:
    r"""
    This class represents characters of Hilbert modular forms. These are not meant to be constructed directly.
    """
    def __init__(self, K, sl2_val, t_val, omega=None):
        self.base_field = K
        self.sl2_val = sl2_val % 24
        self.t_val = t_val % 24
        if omega:
            self.omega = omega
        else:
            d = K.discriminant()
            sqrtd = K(d).sqrt()
            if d % 4:
                self.omega = (1 + sqrtd) / 2
            else:
                self.omega = sqrtd / 2

    def __repr__(self):
        if self.sl2_val or self.t_val:
            prefix = ('th', 'st', 'nd', 'rd', 'th')[min(self.sl2_val, 4)]
            if self.t_val % 12:
                t_val_str = 'e^(2*pi*i*(%s))' % (self.t_val/24)
            elif self.t_val % 24:
                t_val_str = '-1'
            else:
                t_val_str = '1'
            if self.sl2_val % 24:
                s = ('Character', 'Multiplier system')[self.sl2_val % 2]
                return '%s acting on SL_2(Z) as the %d%s power of the eta multiplier and by %s on translation by %s' % (s, self.sl2_val, prefix, t_val_str, self.omega)
            return 'Character acting trivially on SL_2(Z) and by %s on translation by %s' % (t_val_str, self.omega)
        return 'Trivial character'

    def __mul__(self, other):
        if self.base_field != other.base_field:
            raise ValueError('Incompatible base fields')
        return HMFCharacter(self.base_field, self.sl2_val + other.sl2_val, self.t_val + other.t_val)

    def __pow__(self, n):
        return HMFCharacter(self.base_field, n * self.sl2_val, n * self.t_val)


def hmf_inputs(K, level=1):
    r"""
    Constructs a WeilRep instance whose lifts are Hilbert modular forms.

    INPUT:
    - ``K`` -- a real-quadratic number field

    OUTPUT: WeilRep
    """
    from .weilrep import WeilRep
    if not (K.is_totally_real() and K.degree() == 2):
        raise ValueError('HMF only accepts real-quadratic number fields.')
    a, b = K.discriminant().quo_rem(2)
    S = matrix([[-2, b], [b, a]])
    w = WeilRep(S)
    if level > 1:
        w += II(level)
    w.lift_qexp_representation = 'hilbert', K, level
    return w


class HilbertModularForms(OrthogonalModularFormsLorentzian):
    r"""
    This class represents spaces of Hilbert modular forms for the full modular group in real-quadratic number fields (as a special case of modular forms for orthogonal groups of signature 2, 2).

    INPUT:
    Construct a HilbertModularForms instance with either
    HilbertModularForms(K)
    or
    HMF(K)
    where:

    - ``K`` -- a real-quadratic number field

    Optional parameter:

    - ``level`` -- positive integer (default 1); this represents Hilbert modular forms for the subgroup \Gamma_1(N)
    """
    def __init__(self, K, level=1):
        self.__base_field = K
        w = hmf_inputs(K, level=level)
        self._OrthogonalModularForms__weilrep = w
        self._OrthogonalModularForms__gram_matrix = w.gram_matrix()
        d = K.discriminant()
        sqrtd = K(d).sqrt()
        if d % 4:
            self.__omega = (1 + sqrtd) / 2
        else:
            self.__omega = sqrtd / 2
        self.__sqrtd = sqrtd

    def __repr__(self):
        return 'Hilbert modular forms over %s' % str(self.__base_field)

    def base_field(self):
        r"""
        Return self's base field.
        """
        return self.__base_field

    def doi_naganuma_lift(self, x):
        r"""
        Compute the Doi--Naganuma lift of 'x'. Here 'x' may be either a vector-valued modular form (for the appropriate representation) or a scalar modular form (of the correct level and character and satisfying the ``plus``-condition).

        INPUT:
        - ``x`` -- a WeilRepModularForm or a ModularForm

        OUTPUT: HilbertModularForm

        EXAMPLES::

            sage: from weilrep import *
            sage: x = polygen(QQ, 'x')
            sage: K.<sqrt5> = NumberField(x * x - 5)
            sage: chi = DirichletGroup(5)[2]
            sage: HMF(K).doi_naganuma_lift(CuspForms(chi, 6, prec = 20).basis()[0])
            1/2*q1^(-1/10*sqrt5 + 1/2)*q2^(1/10*sqrt5 + 1/2) + 1/2*q1^(1/10*sqrt5 + 1/2)*q2^(-1/10*sqrt5 + 1/2) + 1/2*q1^(-2/5*sqrt5 + 1)*q2^(2/5*sqrt5 + 1) + 10*q1^(-1/5*sqrt5 + 1)*q2^(1/5*sqrt5 + 1) - 45*q1*q2 + 10*q1^(1/5*sqrt5 + 1)*q2^(-1/5*sqrt5 + 1) + 1/2*q1^(2/5*sqrt5 + 1)*q2^(-2/5*sqrt5 + 1) - 45*q1^(-1/2*sqrt5 + 3/2)*q2^(1/2*sqrt5 + 3/2) + 45*q1^(-3/10*sqrt5 + 3/2)*q2^(3/10*sqrt5 + 3/2) + 126*q1^(-1/10*sqrt5 + 3/2)*q2^(1/10*sqrt5 + 3/2) + 126*q1^(1/10*sqrt5 + 3/2)*q2^(-1/10*sqrt5 + 3/2) + 45*q1^(3/10*sqrt5 + 3/2)*q2^(-3/10*sqrt5 + 3/2) - 45*q1^(1/2*sqrt5 + 3/2)*q2^(-1/2*sqrt5 + 3/2) + O(q1, q2)^4
        """
        try:
            return x.theta_lift()
        except AttributeError:
            w = self.weilrep()
            return w.bb_lift(x).theta_lift()

    def eisenstein_series(self, k, prec):
        r"""
        Compute the Hilbert Eisenstein series E_k(tau1, tau2).

        This is a simple algorithm based on the theta lift. We do not use a closed formula for Eisenstein series coefficients.

        INPUT:
        - ``k`` -- the weight (an even integer >= 2)
        - ``prec`` -- the precision of the output

        OUTPUT: HilbertModularForm

        EXAMPLES::

            sage: from weilrep import *
            sage: x = polygen(QQ, 'x')
            sage: K.<sqrt5> = NumberField(x^2 - 5)
            sage: HMF(K).eisenstein_series(2, 6)
            1 + 120*q1^(-1/10*sqrt5 + 1/2)*q2^(1/10*sqrt5 + 1/2) + 120*q1^(1/10*sqrt5 + 1/2)*q2^(-1/10*sqrt5 + 1/2) + 120*q1^(-2/5*sqrt5 + 1)*q2^(2/5*sqrt5 + 1) + 600*q1^(-1/5*sqrt5 + 1)*q2^(1/5*sqrt5 + 1) + 720*q1*q2 + 600*q1^(1/5*sqrt5 + 1)*q2^(-1/5*sqrt5 + 1) + 120*q1^(2/5*sqrt5 + 1)*q2^(-2/5*sqrt5 + 1) + 720*q1^(-1/2*sqrt5 + 3/2)*q2^(1/2*sqrt5 + 3/2) + 1200*q1^(-3/10*sqrt5 + 3/2)*q2^(3/10*sqrt5 + 3/2) + 1440*q1^(-1/10*sqrt5 + 3/2)*q2^(1/10*sqrt5 + 3/2) + 1440*q1^(1/10*sqrt5 + 3/2)*q2^(-1/10*sqrt5 + 3/2) + 1200*q1^(3/10*sqrt5 + 3/2)*q2^(-3/10*sqrt5 + 3/2) + 720*q1^(1/2*sqrt5 + 3/2)*q2^(-1/2*sqrt5 + 3/2) + 600*q1^(-4/5*sqrt5 + 2)*q2^(4/5*sqrt5 + 2) + 1440*q1^(-3/5*sqrt5 + 2)*q2^(3/5*sqrt5 + 2) + 2520*q1^(-2/5*sqrt5 + 2)*q2^(2/5*sqrt5 + 2) + 2400*q1^(-1/5*sqrt5 + 2)*q2^(1/5*sqrt5 + 2) + 3600*q1^2*q2^2 + 2400*q1^(1/5*sqrt5 + 2)*q2^(-1/5*sqrt5 + 2) + 2520*q1^(2/5*sqrt5 + 2)*q2^(-2/5*sqrt5 + 2) + 1440*q1^(3/5*sqrt5 + 2)*q2^(-3/5*sqrt5 + 2) + 600*q1^(4/5*sqrt5 + 2)*q2^(-4/5*sqrt5 + 2) + 120*q1^(-11/10*sqrt5 + 5/2)*q2^(11/10*sqrt5 + 5/2) + 1440*q1^(-9/10*sqrt5 + 5/2)*q2^(9/10*sqrt5 + 5/2) + 2400*q1^(-7/10*sqrt5 + 5/2)*q2^(7/10*sqrt5 + 5/2) + 3720*q1^(-1/2*sqrt5 + 5/2)*q2^(1/2*sqrt5 + 5/2) + 3600*q1^(-3/10*sqrt5 + 5/2)*q2^(3/10*sqrt5 + 5/2) + 3840*q1^(-1/10*sqrt5 + 5/2)*q2^(1/10*sqrt5 + 5/2) + 3840*q1^(1/10*sqrt5 + 5/2)*q2^(-1/10*sqrt5 + 5/2) + 3600*q1^(3/10*sqrt5 + 5/2)*q2^(-3/10*sqrt5 + 5/2) + 3720*q1^(1/2*sqrt5 + 5/2)*q2^(-1/2*sqrt5 + 5/2) + 2400*q1^(7/10*sqrt5 + 5/2)*q2^(-7/10*sqrt5 + 5/2) + 1440*q1^(9/10*sqrt5 + 5/2)*q2^(-9/10*sqrt5 + 5/2) + 120*q1^(11/10*sqrt5 + 5/2)*q2^(-11/10*sqrt5 + 5/2) + O(q1, q2)^6
        """
        w = self.weilrep()
        try:
            return (-((k + k) / bernoulli(k)) * w.eisenstein_series(k, ceil(prec * prec / 4) + 1)).theta_lift(prec)
        except (TypeError, ValueError, ZeroDivisionError):
            raise ValueError('Invalid weight')

    def omega(self):
        r"""
        Return the generator (1 + sqrt(d_K)) / 2 or sqrt(d_K)/2 of the ring of integers of the underlying number field.
        """
        return self.__omega

    def _sqrtd(self):
        return self.__sqrtd

    def fundamental_unit(self):
        r"""
        Return the fundamental unit in self's base field.
        """
        try:
            return self.__unit
        except AttributeError:
            K = self.base_field()
            self.__unit = K(K.unit_group().gens()[1])
            return self.__unit

    def hecke_operator(self, p):
        return HilbertHeckeOperator(self, p)

    def eigenforms(self, X, _p=2, _name='', _final_recursion=True, _K_list=[]):
        r"""
        Decompose a space X into common eigenforms of the Hecke operators.

        This will raise a ValueError if X is not invariant under the Hecke operators.
        """
        _p = Integer(_p)
        K0 = self.base_field()
        D = K0.discriminant()
        while D % _p == 0:
            _p = next_prime(_p)
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
                name = 'a_%s%s' % (_name, i)
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
                    for p in P.eigenvectors_left(extend=False):
                        c = p[0].charpoly()
                        if c not in chi_list:
                            L.append(vector(p[1][0]) * V)
                            K_list.append(K)
                            chi_list.append(c)
                else:
                    L.append(V_rows[0])
                    K_list.append(K)
            else:
                _name = _name + '%s_' % i
                K_list_2, eigenvectors = self.eigenforms(V_rows, _p=next_prime(_p), _name=_name, _final_recursion=False, _K_list=K_list)
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
            x /= yd[b]
            x.__class__ = HilbertEigenform
            x._HilbertEigenform__field = K_list[i]
            eigenforms.append(x)
        return eigenforms


HMF = HilbertModularForms


class HilbertModularForm(OrthogonalModularFormLorentzian):
    r"""
    This class represents Hilbert modular forms for the full modular group in real-quadratic number fields.
    """
    def __init__(self, base_field, level):
        self.__base_field = base_field
        self.__level = level

    def __repr__(self):
        r"""
        Represent self's Fourier expansion as a power series c(v) q1^v q2^(v'), where v runs through totally-positive elements in the dual of the ring of integers.
        """
        K = self.__base_field
        h = self.true_fourier_expansion()
        hprec = h.prec()
        d = self.scale()
        if h:
            D = K.discriminant()
            sqrtD = K(D).sqrt()
            if not D % 4:
                sqrtD /= 2
            s = ''
            sign = False
            _b = False
            if not isinstance(h.base_ring(), LaurentPolynomialRing_generic):
                X = LaurentSeriesRing(self.base_ring(), 'x')
                _b = True
            for i, p in enumerate(h.list()):
                i = ZZ(i)
                if _b:
                    p = X(p)
                for n in p.exponents():
                    c = p[n]
                    if c:
                        q2exp = (i - n/sqrtD)/(d + d)
                        q1exp = i / d - q2exp
                        coef = True
                        if sign:
                            if c > 0 and c != 1:
                                s += ' + ' + str(c)
                            elif c + 1 and c != 1:
                                s += ' - ' + str(-c)
                            elif c + 1:
                                s += ' + '
                                coef = False
                            elif c - 1:
                                s += ' - '
                                coef = False
                        else:
                            if abs(c) != 1 or not q1exp:
                                s += str(c)
                            else:
                                coef = False
                                if c == -1:
                                    s += '-'
                            sign = True
                        if q1exp:
                            if coef:
                                s += '*'
                            if q1exp != q2exp or q1exp not in ZZ:
                                s += 'q1^(%s)*q2^(%s)' % (q1exp, q2exp)
                            elif q1exp != 1:
                                s += 'q1^%s*q2^%s' % (q1exp, q2exp)
                            else:
                                s += 'q1*q2'
                        sign = True
            if hprec % d:
                self.__string = s + ' + O(q1, q2)^(%s)' % (hprec/d)
            else:
                self.__string = s + ' + O(q1, q2)^%s' % (hprec/d)
        else:
            if hprec % d:
                self.__string = 'O(q1, q2)^(%s)' % (hprec/d)
            else:
                self.__string = 'O(q1, q2)^%s' % (hprec/d)
        return self.__string

    def base_field(self):
        r"""
        Return self's base field.
        """
        return self.__base_field

    def character(self):
        r"""
        Compute self's character.
        """
        scale = self.scale()
        d = self.base_field().discriminant()
        X = self.fourier_expansion()
        val = X.valuation()
        r = X[val].valuation()
        if d % 4:
            return HMFCharacter(self.base_field(), (24 * val)/scale, (12 * (val + r))/scale)
        return HMFCharacter(self.base_field(), (24 * val)/scale, (24 * r)/scale)

    def nvars(self):
        return 2

    def hmf(self):
        try:
            return self.__hmf
        except AttributeError:
            self.__hmf = HilbertModularForms(self.base_field())
            return self.__hmf

    # get Fourier coefficients

    def coefficients(self, prec=+Infinity):
        r"""
        Return self's Fourier coefficients as a dictionary.
        """
        d = self.scale()
        d_prec = d * prec
        K = self.base_field()
        D = K.discriminant()
        sqrtD = K(D).sqrt()
        if not D % 4:
            sqrtD /= 2
        h = self.fourier_expansion()
        return {(i + n/sqrtD)/(d + d): p[n]
                for i, p in enumerate(h.list()) if i < d_prec for n in p.exponents()}

    def __getitem__(self, a):
        r"""
        Extract the Fourier coefficient indexed by ``a``.

        INPUT:
        - ``a`` -- a number field element

        OUTPUT: a rational number

        EXAMPLES::

            sage: from weilrep import *
            sage: x = polygen(QQ, 'x')
            sage: K.<sqrt2> = NumberField(x^2 - 2)
            sage: e2 = HMF(K).eisenstein_series(2, 5)
            sage: e2[2 - sqrt2]
            1488
        """
        scale = self.scale()
        K = self.base_field()
        D = K.discriminant()
        try:
            tt = K(a).trace()
            i = ZZ(scale * tt)
            if D % 4:
                sqrtD = K(D).sqrt()
            else:
                sqrtD = K(D).sqrt() / 2
            n = ZZ(scale * (2 * a - tt) * sqrtD)
            return self.fourier_expansion()[i][n]
        # except TypeError:
        #    return 0
        except IndexError:
            u = self.hmf().fundamental_unit()
            au = a * u
            k = self.weight()
            if au.trace() < K(a).trace():
                return (-1)**k * self.__getitem__(au)
            au = a / u
            if au.trace() < K(a).trace():
                return (-1)**k * self.__getitem__(au)
            raise IndexError('coefficient not known') from None

    # other methods

    def hecke_operator(self, p):
        r"""
        Apply the Hecke operator T_p.

        p should be a prime element of K.
        """

    def hz_pullback(self, mu):
        r"""
        Compute the pullbacks to Hirzebruch--Zagier curves.

        This computes the pullback f(\tau * \mu, \tau * \mu') of f to the embedded half-plane H * (\mu, \mu') where \mu' is the conjugate of \mu. The result is a modular form of level equal to the norm of \mu.

        INPUT:
        - ``mu`` -- a totally-positive integer in the base-field K.

        OUTPUT: an OrthogonalModularForm for a signature (2, 1) lattice

        EXAMPLES::

            sage: from weilrep import *
            sage: x = polygen(QQ, 'x')
            sage: K.<sqrt13> = NumberField(x * x - 13)
            sage: HMF(K).eisenstein_series(2, 15).hz_pullback(4 - sqrt13)
            1 + 24*q + O(q^2)
        """
        K = self.base_field()
        mu = K(mu)
        nn = mu.norm()
        tt = mu.trace()
        if tt <= 0 or nn <= 0:
            raise ValueError('You called hz_pullback with a number that is not totally-positive!')
        d = K.discriminant()
        a = isqrt((tt * tt - 4 * nn) / d)
        h = self.fourier_expansion()
        t, = PowerSeriesRing(QQ, 't').gens()
        d = K.discriminant()
        prec = floor(h.prec() / (tt/2 + 2 * a / sqrt(d)))
        if d % 4:
            f = sum([p[n] * t ** ((i*tt + n * a)/2) for i, p in enumerate(h.list()) for n in p.exponents()]) + O(t**prec)
        else:
            f = sum([p[n] * t ** ((i*tt + 2 * n * a)/2) for i, p in enumerate(h.list()) for n in p.exponents()]) + O(t**prec)
        return OrthogonalModularFormLorentzian(self.weight(), WeilRep(matrix([[-2 * nn]])), f, scale=self.scale(), weylvec=vector([0]), qexp_representation='shimura')


class HilbertEigenform(HilbertModularForm):

    def hecke_field(self):
        r"""
        This is the number field generated by self's Hecke eigenvalues.
        """
        return self.__field

    def eigenvalue(self, p):
        T = self.hmf().hecke_operator(p)
        g = self.fourier_expansion()
        d = self.scale()
        sqrtD = self.hmf()._sqrtd()
        for i, u in enumerate(g.list()):
            if u:
                for n, y in u.dict().items():
                    N = (i + n/sqrtD)/(d + d)
                    try:
                        return T._get_coefficient(self, N) / y
                    except IndexError:
                        pass
        raise ValueError('Insufficient precision') from None

    def euler_factor(self, p):
        h = self.hmf()
        K = h.base_field()
        X, = PolynomialRing(self.hecke_field(), 'X').gens()
        L = K.ideal(p).factor()
        k = self.weight()
        eps = h.fundamental_unit()
        if len(L) == 2:
            (pi_1, _), (pi_2, _) = L
            try:
                pi_1, = pi_1.gens_reduced()
                pi_2, = pi_2.gens_reduced()
                if pi_1.norm() < 0:
                    pi_1 *= eps
                if pi_1.trace() < 0:
                    pi_1 = -pi_1
                if pi_2.norm() < 0:
                    pi_2 *= eps
                if pi_2.trace() < 0:
                    pi_2 = -pi_2
                e1 = self.eigenvalue(pi_1)
                e2 = self.eigenvalue(pi_2)
                return (1 - e1 * X + p**(k - 1) * X * X) * (1 - e2 * X + p**(k - 1) * X * X)
            except ValueError:
                raise ValueError('%s splits into non-principal ideals in %s' % (p, K))
        e = self.eigenvalue(p)
        return 1 - e * X * X + p**(2 * k - 2) * X**4

    def asai_euler_factor(self, p):
        h = self.hmf()
        K = self.base_field()
        X, = PolynomialRing(self.hecke_field(), 'X').gens()
        L = K.ideal(p).factor()
        k = self.weight()
        eps = h.fundamental_unit()
        if len(L) == 2:
            (pi_1, _), (pi_2, _) = L
            try:
                pi_1, = pi_1.gens_reduced()
                pi_2, = pi_2.gens_reduced()
                if pi_1.norm() < 0:
                    pi_1 *= eps
                if pi_1.trace() < 0:
                    pi_1 = -pi_1
                if pi_2.norm() < 0:
                    pi_2 *= eps
                if pi_2.trace() < 0:
                    pi_2 = -pi_2
                e1 = self.eigenvalue(pi_1)
                e2 = self.eigenvalue(pi_2)
                return 1 - e1 * e2 * X + p**(k - 1) * (e1**2 + e2**2 - 2 * p**(k - 1)) * X * X - p**(k + k - 2) * e1 * e2 * X**3 + p**(4 * k - 4) * X**4
            except ValueError:
                raise ValueError('%s splits into non-principal ideals in %s' % (p, K)) from None
        e = self.eigenvalue(p)
        return (1 - e * X + p**(2 * k - 2) * X**2) * (1 - (p**(k - 1) * X)**2)


class HilbertHeckeOperator:
    def __init__(self, hmf, p):
        self.__hmf = hmf
        self.__index = p

    def __repr__(self):
        return 'Hecke operator of index %s acting on Hilbert modular forms over %s' % (self.__index, self.__hmf.base_field())

    def hmf(self):
        return self.__hmf

    def index(self):
        return self.__index

    def _get_coefficient(self, f, N):
        r"""
        Get the coefficient of q^N = exp(2*pi*I * (N * tau1 + N' * tau2)) in the image of the Hilbert modular form f under self.
        """
        p = self.__index
        h = self.hmf()
        O = h.base_field().maximal_order()
        if p in ZZ:
            norm = p**2
        else:
            norm = p.norm()
        k = f.weight()
        if N * h._sqrtd() not in O:
            return 0
        s = f.__getitem__(p * N)
        if N * h._sqrtd() / p in O:
            s += f.__getitem__(N / p) * norm**(k - 1)
        return s

    def __call__(self, f):
        K = self.hmf().base_field()
        O = K.maximal_order()
        p = K(self.index())
        norm = p.norm()
        k = f.weight()
        d = f.scale()
        g = f.fourier_expansion()
        t, = g.parent().gens()
        x, = g.base_ring().gens()
        sqrtD = self.hmf()._sqrtd()
        sqrtd_n = math.sqrt(K.discriminant())
        h = g.parent()(0)
        for i, u in enumerate(g.list()):
            i = ZZ(i)
            for n in range(ceil(-i * sqrtd_n), floor(i * sqrtd_n) + 1):
                if n % 2 == i % 2:
                    N = (i + n/sqrtD)/(d + d)
                    try:
                        c = f.__getitem__(p * N)
                        if N * sqrtD / p in O:
                            c += f.__getitem__(N / p) * norm**(k - 1)
                        h += c * t**i * x**n
                    except IndexError:
                        return OrthogonalModularForm(k, f.weilrep(),
                                                     h.add_bigoh(i - 1),
                                                     d, f.weyl_vector(),
                                                     qexp_representation=f.qexp_representation())

    def matrix(self, X):
        L = []
        R = []
        rank = 0
        target_rank = len(X)
        Xref = X[0]
        d = Xref.scale()
        prec = Xref.precision()
        K = self.hmf().base_field()
        D = K.discriminant()
        sqrtD = self.hmf()._sqrtd()
        sqrtd_n = math.sqrt(D)
        for i in srange(prec):
            for n in range(ceil(-i * sqrtd_n), floor(i * sqrtd_n) + 1):
                if n % 2 == i % 2:
                    N = (i + n/sqrtD)/(d + d)
                    try:
                        L1 = list(L)
                        L1.append([self._get_coefficient(f, N) for f in X])
                        M = matrix(L1)
                        rank1 = M.rank()
                        if rank1 > rank:
                            L = L1
                            rank = rank1
                            R.append([f[N] for f in X])
                        if rank == target_rank:
                            return matrix(R).solve_right(matrix(L))
                    except NotImplementedError:  # (IndexError, ValueError):
                        pass
        raise ValueError('Insufficient precision') from None
