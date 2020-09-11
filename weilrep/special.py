r"""

Degree two Paramodular forms and Hermitian modular forms

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



from sage.misc.functional import denominator
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polydict import ETuple

from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian
from .positive_definite import OrthogonalModularFormPositiveDefinite, OrthogonalModularFormsPositiveDefinite
from .weilrep import WeilRep


class ParamodularForms(OrthogonalModularFormsPositiveDefinite):

    r"""
    Paramodular forms of level N.

    INPUT: call ``ParamodularForms(N)`` where
    - ``N`` -- the level
    """

    def __init__(self, N):
        S = matrix([[N + N]])
        w = WeilRep(S)
        w.lift_qexp_representation = 'siegel'
        OrthogonalModularFormsPositiveDefinite.__init__(self, w)
        self.__class__ = ParamodularForms

    def __repr__(self):
        return 'Paramodular forms of level %d'%self.level()

    def level(self):
        return self.gram_matrix()[0, 0] / 2

class ParamodularForm(OrthogonalModularFormPositiveDefinite):

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
        try:
            a = a.list()
        except AttributeError:
            pass
        a, b, _, d = a
        a, b, d = a + d, b + b, a - d
        f = self.true_fourier_expansion()
        s = self.scale()
        N = self.level()
        try:
            return f[ZZ(a * s)][ZZ(d * s) / N][ZZ(b * s)]
        except KeyError:
            return 0

    def level(self):
        return self.gram_matrix()[0, 0] / 2


class HermitianModularForms(OrthogonalModularFormsPositiveDefinite):

    def __init__(self, K, level=1):
        d = K.discriminant()
        self.__base_field = K
        self.__discriminant = d
        if d % 2:
            S = matrix([[2, 1], [1, ZZ((1 - d) / 2)]])
        else:
            S = matrix([[2, 0], [0, ZZ(-d / 2)]])
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

class HermitianModularFormsWithLevel(OrthogonalModularFormsLorentzian):

    def __repr__(self):
        return 'Hermitian modular forms of degree two over %s for the congruence subgroup Gamma1(%d)'%str(self.__base_field, self.__level)

class HermitianModularForm(OrthogonalModularForm):

    r"""
    Hermitian modular forms.

    INPUT: call ``HermitianModularForms(K, level)`` where
    - ``K`` -- the base field (an imaginary quadratic number field)
    - ``level`` -- level (default 1)
    """

    def __init__(self, base_field, level = 1):
        self.__base_field = base_field
        self.__level = level
        if level == 1:
            self.__class__ = HermitianModularFormPositiveDefinite
        else:
            self.__class__ = HermitianModularFormWithLevel

    def __repr__(self):
        r"""
        Represent self's Fourier expansion as a power series c(a,b,c) q^a r1^v r2^(v') s^c, where v runs through elements in the dual of the ring of integers.
        """
        K = self.base_field()
        h = self.fourier_expansion()
        S = self.gram_matrix()
        if self.level() != 1:
            S = S[1:-1,1:-1]
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
                a = ZZ(i[0])/d
                c = ZZ(i[1])/d
                if a:
                    if a != 1:
                        if a in ZZ:
                            q_power = 'q^%s'%a
                        else:
                            q_power = 'q^(%s)'%a
                    else:
                        q_power = 'q'
                else:
                    q_power = ''
                if c:
                    if c != 1:
                        if c in ZZ:
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
                            if r1exp != r2exp or r1exp not in ZZ:
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

    def level(self):
        return self.__level

    ## get coefficients

    def __getitem__(self, a):
        r"""
        Extract Fourier coefficients.

        The exponent should be hermitian (in the case of trivial character) with integral diagonal and with off-diagonal entries in the dual lattice O_K'.

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
            return f[ZZ((a + d) * s)][ZZ((a - d) * s)].dict()[ETuple([ZZ(b * s), ZZ(c * s)])]
        except KeyError:
            return 0

class HermitianModularFormPositiveDefinite(HermitianModularForm, OrthogonalModularFormPositiveDefinite):
    pass

class HermitianModularFormWithLevel(HermitianModularForm, OrthogonalModularFormLorentzian):
    pass