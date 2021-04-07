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



from sage.matrix.constructor import matrix
from sage.misc.functional import denominator
from sage.modular.arithgroup.congroup_gamma0 import is_Gamma0
from sage.modules.free_module_element import vector
from sage.rings.integer import Integer
from sage.rings.polynomial.polydict import ETuple

from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian
from .positive_definite import OrthogonalModularFormPositiveDefinite, OrthogonalModularFormsPositiveDefinite
from .weilrep import WeilRep

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
            return f[Integer(a * s)][Integer(d * s) / N][Integer(b * s)]
        except KeyError:
            return 0

    def level(self):
        return self.gram_matrix()[0, 0] / 2


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

class HermitianModularFormsWithLevel(OrthogonalModularFormsLorentzian, HermitianModularForms):

    def __repr__(self):
        return 'Hermitian modular forms of degree two over %s for the congruence subgroup Gamma1(%d)'%(str(self.base_field()), self.level())

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
        K = self.__base_field
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

    def level(self):
        return self.__level

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


class HermitianModularFormPositiveDefinite(HermitianModularForm, OrthogonalModularFormPositiveDefinite):
    pass

class HermitianModularFormWithLevel(HermitianModularForm, OrthogonalModularFormLorentzian):
    pass

class KohnenPlusSpace:
    r"""
    The Kohnen Plus space.
    (Unfinished!!)
    """

    def __init__(self, level):
        if is_Gamma0(level):
            N = Integer(level.level())
        else:
            N = level
        if N % 4:
            raise NotImplementedError
        self.__N = N
        self.__weilrep1 = WeilRep(matrix([[ N // 2]]))
        self.__weilrep2 = self.__weilrep1.dual()

    def __repr__(self):
        return 'Kohnen plus space of level Gamma0(%d)'%self.__N

    def _plus_form(self, X):
        X = X.fourier_expansion()
        N = self.__N
        return sum(x[2].V(N).shift(Integer(N * x[1])) for x in X)

    def _weilrep(self, k):
        k0 = Integer(k - sage_one_half)
        if k0 % 2:
            return self.__weilrep1
        return self.__weilrep2

    ## Constructions of modular forms ##

    def basis(self, k, prec, *args, **kwargs): #fix?
        cusp_forms = kwargs.pop('cusp_forms', 0)
        N = self.__N
        precn = prec // self.__N + 1
        w = self._weilrep(k)
        if cusp_forms:
            X = w.cusp_forms_basis(k, precn, *args, **kwargs)
        else:
            X = w.basis(k, precn, *args, **kwargs)
        X = [KohnenPlusSpaceForm(k, N, self._plus_form(x).add_bigoh(prec), x) for x in X]
        return X
    modular_forms_basis = basis

    def cusp_forms_basis(self, k, prec, *args, **kwargs):
        kwargs['cusp_forms'] = 1
        return self.basis(k, prec, *args, **kwargs)

    def eisenstein_series(self, k, prec, *args, **kwargs):
        precn = prec // self.__N + 1
        return self._plus_form(self._weilrep(k).eisenstein_series(k, precn, *args, **kwargs)).add_bigoh(prec)

    def theta_series(self, prec, *args, **kwargs):
        precn = prec // self.__N + 1
        return self._plus_form(self.__weilrep2.theta_series(precn, *args, **kwargs)).add_bigoh(prec)



class HalfIntegralWeightModularForm(object):

    def __init__(self, k, N, f):
        self.__weight = k
        self.__level = N
        self.__qexp = f

    def __repr__(self):
        return self.qexp().__repr__()

    def level(self):
        return self.__level

    def qexp(self):
        return self.__qexp

    def weight(self):
        return self.__weight

class KohnenPlusSpaceForm(HalfIntegralWeightModularForm):

    def __init__(self, k, N, f, X):
        super().__init__(k, N, f)
        self.__vvmf = X

    def vvmf(self):
        return self.__vvmf

    def involution(self):
        return HalfIntegralWeightModularForm(self.weight(), self.level(), self.vvmf()[0][2])

cohen_eisenstein_series = lambda *x: KohnenPlusSpace(4).eisenstein_series(*x)