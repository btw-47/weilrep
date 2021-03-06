r"""

Lifts for Hilbert modular forms over real-quadratic fields

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2020 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli
from sage.functions.other import ceil, floor, frac, sqrt
from sage.matrix.constructor import matrix
from sage.misc.functional import denominator, isqrt
from sage.modules.free_module_element import vector
from sage.rings.big_oh import O
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ



from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian
from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis

class HMFCharacter:
    r"""
    This class represents characters of Hilbert modular forms. These are not meant to be constructed directly.
    """
    def __init__(self, K, sl2_val, t_val, omega = None):
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
                t_val_str = 'e^(2*pi*i*(%s))'%(self.t_val/24)
            elif self.t_val % 24:
                t_val_str = '-1'
            else:
                t_val_str = '1'
            if self.sl2_val % 24:
                s = ('Character', 'Multiplier system')[self.sl2_val%2]
                return '%s acting on SL_2(Z) as the %d%s power of the eta multiplier and by %s on translation by %s'%(s, self.sl2_val, prefix, t_val_str, self.omega)
            return 'Character acting trivially on SL_2(Z) and by %s on translation by %s'%(t_val_str, self.omega)
        return 'Trivial character'

    def __mul__(self,other):
        if self.base_field != other.base_field:
            raise ValueError('Incompatible base fields')
        return HMFCharacter(self.base_field, self.sl2_val + other.sl2_val, self.t_val + other.t_val)

    def __pow__(self,n):
        return HMFCharacter(self.base_field, n * self.sl2_val, n * self.t_val)

def hmf_inputs(K, level = 1):
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
    def __init__(self, K, level = 1):
        self.__base_field = K
        w = hmf_inputs(K, level = level)
        self._OrthogonalModularForms__weilrep = w
        self._OrthogonalModularForms__gram_matrix = w.gram_matrix()
        d = K.discriminant()
        sqrtd = K(d).sqrt()
        if d % 4:
            self.__omega = (1 + sqrtd) / 2
        else:
            self.__omega = sqrtd / 2

    def __repr__(self):
        return 'Hilbert modular forms over %s'%str(self.__base_field)

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
            sage: x = var('x')
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
            sage: x = var('x')
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
            for i, p in enumerate(h.list()):
                i = ZZ(i)
                for n in p.exponents():
                    c = p[n]
                    if c:
                        q2exp = (i - n/sqrtD)/(d + d)
                        q1exp = i / d - q2exp
                        coef = True
                        if sign:
                            if c > 0 and c!= 1:
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
                                  s += 'q1^(%s)*q2^(%s)'%(q1exp, q2exp)
                            elif q1exp != 1:
                                  s += 'q1^%s*q2^%s'%(q1exp, q2exp)
                            else:
                                  s += 'q1*q2'
                        sign = True
            if hprec % d:
                self.__string = s + ' + O(q1, q2)^(%s)'%(hprec/d)
            else:
                self.__string = s + ' + O(q1, q2)^%s'%(hprec/d)
        else:
            if hprec % d:
                self.__string = 'O(q1, q2)^(%s)'%(hprec/d)
            else:
                self.__string = 'O(q1, q2)^%s'%(hprec/d)
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

    #get Fourier coefficients

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
        X = {}
        h = self.fourier_expansion()
        return {(i + n/sqrtD)/(d + d):p[n] for i, p in enumerate(h.list()) if i < d_prec for n in p.exponents()}

    def __getitem__(self, a):
        r"""
        Extract the Fourier coefficient indexed by ``a``.

        INPUT:
        - ``a`` -- a number field element

        OUTPUT: a rational number

        EXAMPLES::

            sage: from weilrep import *
            sage: x = var('x')
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
        except TypeError:
            return 0

    #other methods

    def hz_pullback(self, mu):
        r"""
        Compute the pullbacks to Hirzebruch--Zagier curves.

        This computes the pullback f(\tau * \mu, \tau * \mu') of f to the embedded half-plane H * (\mu, \mu') where \mu' is the conjugate of \mu. The result is a modular form of level equal to the norm of \mu.

        INPUT:
        - ``mu`` -- a totally-positive integer in the base-field K.

        OUTPUT: an OrthogonalModularForm for a signature (2, 1) lattice

        WARNING: the output's weyl vector is not implemented

        EXAMPLES::

            sage: from weilrep import *
            sage: x = var('x')
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
        return OrthogonalModularFormLorentzian(self.weight(), WeilRep(matrix([[-2 * nn]])), f, scale = self.scale(), weylvec = vector([0]), qexp_representation = 'shimura')