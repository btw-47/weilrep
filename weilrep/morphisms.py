r"""

Sage code for automorphisms of Weil representations

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
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from .weilrep_modular_forms_class import WeilRepModularForm

class WeilRepMorphism:
    r"""
    This class represents morphisms of discriminant forms.

    INPUT:
    Construct a WeilRepAutomorphism using WeilRepAutomorphism(w1, w2, f), where
    - `` w1 ``, `` w2 `` -- a WeilRep instance
    - `` f `` -- a function that inputs a vector in w1.ds() and outputs another vector in w2.ds()

    f is meant to be an morphism, i.e. f(x+y) = f(x) + f(y) and  Q2(f(x)) = Q1(x) for all x, y if Q1 is the quadratic form of w1 and Q2 of w2.
    """

    def __init__(self, w1, w2, f):
        self.__input_weilrep = w1
        self.__output_weilrep = w2
        self.__f = f
        dsdict = w2.ds_dict()
        ds = w1.ds()
        n1 = w1.norm_list()
        n2 = w2.norm_list()
        Z = [None] * len(ds)
        Z_inv = [None] * len(n2)
        for i, g in enumerate(ds):
            j = dsdict[tuple(f(g))]
            if n1[i] != n2[j] or Z[i] is not None:
                raise ValueError('This is not a morphism.')
            Z[i] = j
            if Z_inv and Z_inv[j] is None:
                Z_inv[j] = i
            else:
                Z_inv = None
        self.__indices = Z
        self.__inverse = Z_inv
        if w1 == w2:
            self.__class__ = WeilRepAutomorphism

    def __repr__(self):
        w = self.input_weilrep()
        f = self.__f
        return 'Morphism from %s to %s\nmapping %s'%(w, self.output_weilrep(), ', '.join(['%s->%s'%(x, f(x)) for x in w.ds()]))

    def f(self):
        return self.__f

    def input_weilrep(self):
        return self.__input_weilrep

    def output_weilrep(self):
        return self.__output_weilrep

    def inverse(self):
        return ~self

    def __call__(self, X):
        try:
            if X.weilrep() != self.input_weilrep():
                raise ValueError('Invalid Weil representation')
            Xf = X.fourier_expansion()
            Z = self.__indices
            return WeilRepModularForm(X.weight(), X.gram_matrix(), [(Xf[i][0], Xf[i][1], Xf[z][2]) for i,z in enumerate(Z)], weilrep = X.weilrep())
        except AttributeError:
            return self.f()(X)

    def __eq__(self, other):
        w = self.input_weilrep()
        if w == other.input_weilrep():
            f1 = self.f()
            f2 = other.f()
            return all(f1(x) == f2(x) for x in w.ds())
        return False

    def __invert__(self):
        Z = self.__inverse
        if not Z:
            raise ValueError('This morphism is not invertible')
        w1, w2 = self.input_weilrep(), self.output_weilrep()
        d1, d2 = w1.ds(), w2.ds_dict()
        return WeilRepMorphism(w2, w1, lambda x: d1[Z[d2[tuple(x)]]])

    def __mul__(self, other):
        r"""
        Composition of morphisms.
        """
        w1 = other.input_weilrep()
        w2 = self.output_weilrep()
        return WeilRepMorphism(w1, w2, lambda x: self.f()(other.f()(x)))

    def __neg__(self):
        r"""
        Negative of self.
        """
        f = self.f()
        return WeilRepMorphism(self.input_weilrep(), self.output_weilrep(), lambda x: vector(map(frac, -f(x))))

    def matrix(self):
        f = self.f()
        r = QQ/ZZ
        x = self.input_weilrep().ds()
        return matrix(r, x).solve_right(matrix(r, map(f, x)))


class WeilRepAutomorphism(WeilRepMorphism):
    r"""
    This class represents automorphisms of discriminant forms.

    INPUT:
    Construct a WeilRepAutomorphism using WeilRepAutomorphism(w, f), where
    - `` w `` -- a WeilRep instance
    - `` f `` -- a function that inputs a vector in w.ds() and outputs another vector in w.ds()

    f is supposed to be an automorphism, i.e. a bijection and Q(f(x)) = Q(x) for all x if Q is the quadratic form of w. We check this at initialization.
    """

    def __init__(self, weilrep, f):
        super().__init__(weilrep, weilrep, f)

    def weilrep(self):
        return self.input_weilrep()

    def __repr__(self):
        ds = self.weilrep().ds()
        f = self.f()
        return 'Automorphism of %s\nmapping %s'%(self.weilrep(), ', '.join(['%s->%s'%(x, f(x)) for x in ds]))

    def __pow__(self, n):
        if not n:
            return self.weilrep()._identity_morphism()
        elif n < 0:
            return ~self.__pow__(-n)
        elif n == 1:
            return self
        nhalf = n // 2
        return self.__pow__(nhalf).__mul__(self.__pow__(n - nhalf))