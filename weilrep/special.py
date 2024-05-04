r"""

Unfinished things.

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
from sage.functions.other import ceil
from sage.matrix.constructor import matrix
from sage.misc.functional import denominator, isqrt
from sage.modular.arithgroup.congroup_gamma0 import is_Gamma0
from sage.modules.free_module_element import vector
from sage.rings.integer import Integer
from sage.rings.polynomial.polydict import ETuple
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .lorentz import II, OrthogonalModularFormLorentzian, OrthogonalModularFormsLorentzian
from .positive_definite import OrthogonalModularFormPositiveDefinite, OrthogonalModularFormsPositiveDefinite
from .weilrep import WeilRep
from .weilrep_misc import relations

sage_one_half = Integer(1) / Integer(2)








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