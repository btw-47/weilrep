r"""

Heegner divisors on O(n, 2)

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2025 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from collections import defaultdict

from sage.arith.misc import moebius
from sage.functions.other import ceil, frac
from sage.misc.functional import isqrt
from sage.modules.free_module_element import vector
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ

from .lifts import OrthogonalModularForms


def heegner_divisor_iterator(w, max_discr = 1):
    r"""
    Iterate through Heegner divisors of discriminant up to a given bound (default 1).

    INPUT:
    - ``w`` -- a WeilRep
    - ``max_discr`` -- the bound for the discriminant

    OUTPUT: an iterator over tuples of the form (x_1,...,x_r, N), corresponding to the Heegner divisor H( (x_1,...,x_r) + ZZ^N, N).
    """
    w_dual = w.dual()
    rds = w_dual.sorted_rds()
    n = w_dual.norm_dict()
    m = 1
    while m <= ceil(max_discr):
        for g in rds:
            N = m + n[g]
            if N <= max_discr:
                yield tuple(list(g) + [N])
        m += 1


def primitive_heegner_divisor(w, g, _flag=0):
    r"""
    Decompose a Heegner divisor (g) attached to the weilrep (w) into irreducible divisors.

    INPUT:
    - ``w`` -- a WeilRep
    - ``g`` -- a tuple
    """
    N = g[-1]
    h = heegner_divisor_iterator(w, max_discr = N / 4)
    x = [(1, g)]
    for y in h:
        m = isqrt(N // y[-1])
        if _flag == 1:
            moebius_m = 1
        else:
            moebius_m = moebius(m)
        if moebius_m:
            if y[-1] * (m * m) == N:
                if all(m * y[i] - v in ZZ for i, v in enumerate(g[:-1])):
                    x += [(moebius_m, y)]
                if all(m * y[i] + v in ZZ for i, v in enumerate(g[:-1])) and 2 % vector(g[:-1]).denominator():
                    x += [(moebius_m, y)]
    return x


class HeegnerDivisor(object):
    r"""
    This class represents Heegner divisors.
    """
    def __init__(self, w, d, k, m=None):
        self.__dict = defaultdict(int, d)
        self.__w = w
        self.__k = k
        if m:
            self.__m = m
        else:
            self.__m = OrthogonalModularForms(w)

    def __repr__(self):
        def a(n):
            if n == 1:
                return ''
            return str(n) + '*'

        def b(x):
            v = vector(x[:-1])
            c = 2 % v.denominator()
            x = str(x)
            j = x.rindex(',')
            s = x[:j] + ';' + x[j+1:]
            if c:
                c = '+/-'
            else:
                c = ''
            return 'H(%s(%s); %s)' % (c, x[1:j], x[j + 1:-1])
        return ' + '.join('%s%s' % (a(y), b(x)) for x, y in self.__dict.items() if y)

    def dict(self):
        return self.__dict

    def __getitem__(self, x):
        return self.__dict[x]

    def weight(self):
        return self.__k

    def weilrep(self):
        return self.__w

    def restriction(self, v):
        r"""
        Compute the restriction (self)|_{v^{\perp}} to a hyperplane v^{\perp}.
        """
        from .lorentz import II
        from .weilrep_modular_forms_class import WeilRepModularForm
        v = vector(v)
        w = self.weilrep()
        N = w.gram_matrix().nrows()
        c = 0
        if len(v) == N + 2:
            w = w + II(1)
            c = 1
        elif len(v) == N + 4:
            w = w + II(1) + II(1)
            c = 2
        elif len(v) != N:
            raise ValueError('%s is not a lattice vector' % v)
        S = w.gram_matrix()
        norm = v * S * v / 2
        if norm < 0:
            raise ValueError('The vector %s has negative norm %s' % (v, norm))
        d = w.ds_dict()
        k = 1 - self.__m.nvars() / 2
        X = w.zero(k).fourier_expansion()
        q = X[0][2].parent().gen()
        for x, y in self.dict().items():
            x1 = tuple([0]*c + list(map(frac, x[:-1])) + [0] * c)
            x2 = tuple([0]*c + [frac(-a) for a in x[:-1]] + [0]*c)
            i1, i2 = d[x1], d[x2]
            a = y * q ** ceil(-x[-1])
            X[i1][2] += a
            if i2 != i1:
                X[i2][2] += a
        f = WeilRepModularForm(k, S, X, weilrep=w)
        f = f.pullback_perp(v)
        w = f.weilrep()
        indices = w.rds(indices = True)
        dsdict = w.ds_dict()
        d = f.principal_part_coefficients()
        d = {tuple(list(x[:-1]) + [-x[-1]]): y for x, y in d.items() if indices[dsdict[tuple(x[:-1])]] is None and x[-1]}
        return HeegnerDivisor(w, d, self.weight() - Integer(1) / 2)

    def P(self):
        d = defaultdict(int, {})
        for x, y in self.__dict.items():
            x = primitive_heegner_divisor(self.__w, x, _flag = 1)
            for x in x:
                d[x[1]] += y
        return PrimitiveHeegnerDivisor(self.__w, d, self.__k)

    def __add__(self, other):
        if isinstance(other, PrimitiveHeegnerDivisor):
            return self.__add__(other.H())
        dict1 = self.dict()
        dict2 = other.dict()
        return HeegnerDivisor(self.__w, {x: dict1[x] + dict2[x] for x in set(dict1.keys()).union(set(dict2.keys()))}, self.__k)
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, PrimitiveHeegnerDivisor):
            return self.__sub__(other.H())
        dict1 = self.dict()
        dict2 = other.dict()
        return HeegnerDivisor(self.__w, {x: dict1[x] - dict2[x] for x in set(dict1.keys()).union(set(dict2.keys()))}, self.__k)

    def __mul__(self, n):
        d = self.dict()
        return HeegnerDivisor(self.__w, {x: n * y for x, y in d.items()}, self.__k)
    __rmul__ = __mul__

    def __neg__(self):
        return self.__mul__(-1)

    def functional(self, X = None):
        if X is None:
            w = self.__w
            k = self.__k
            w_dual = w.dual()
            bd = max(x[-1] for x in self.dict().keys())
            X = [w_dual.eisenstein_series(k, bd + 1)] + w_dual.cusp_forms_basis(k, bd + 1)
        return sum(c * vector(coefficient_functional(X, [(1, g)])) for g, c in self.dict().items())

    def degree(self):
        r"""
        Compute the arithmetic degree of self.
        """
        bound = ceil(max(c[1] for c in self.__dict)) + 1
        vol = self.__m.volume()
        e = self.__w.dual().eisenstein_series(self.__k, bound).coefficients()
        return -sum(e[x] * y for x, y in self.__dict.items()) * vol

    def _intersection(self, other):
        d_self = self.dict()
        d_other = other.dict()
        m = self.__m
        return sum(y1 * y2 * intersection_number(m, vector(x1[:-1]), x1[-1], vector(x2[:-1]), x2[-1]) for x1, y1 in d_self.items() for x2, y2 in d_other.items())


class PrimitiveHeegnerDivisor(object):
    def __init__(self, w, d, k):
        self.__dict = defaultdict(int, d)
        self.__w = w
        self.__k = k

    def __repr__(self):
        def a(n):
            if n == 1:
                return ''
            return str(n) + '*'

        def b(x):
            v = vector(x[:-1])
            c = 2 % v.denominator()
            x = str(x)
            j = x.rindex(',')
            s = x[:j] + ';' + x[j+1:]
            if c:
                c = '+/-'
            else:
                c = ''
            return 'P(%s(%s); %s)' % (c, x[1:j], x[j + 1:-1])
        return ' + '.join('%s%s' % (a(y), b(x)) for x, y in self.__dict.items() if y)

    def dict(self):
        return self.__dict

    def __getitem__(self, x):
        return self.__dict[x]

    def H(self):
        d = defaultdict(int, {})
        for x, y in self.__dict.items():
            x = primitive_heegner_divisor(self.__w, x)
            for x in x:
                d[x[1]] += x[0] * y
        return HeegnerDivisor(self.__w, d, self.__k)

    def restriction(self, v):
        return self.H().restriction(v).P()

    def __add__(self, other):
        if isinstance(other, HeegnerDivisor):
            return self.__add__(other.P())
        dict1 = self.dict()
        dict2 = other.dict()
        return PrimitiveHeegnerDivisor(self.__w, {x: dict1[x] + dict2[x] for x in set(dict1.keys()).union(set(dict2.keys()))}, self.__k)
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, HeegnerDivisor):
            return self.__add__(other.P())
        dict1 = self.dict()
        dict2 = other.dict()
        return PrimitiveHeegnerDivisor(self.__w, {x: dict1[x] + dict2[x] for x in set(dict1.keys()).union(set(dict2.keys()))}, self.__k)

    def __mul__(self, n):
        d = self.dict()
        return PrimitiveHeegnerDivisor(self.__w,
                                       {x: n * y for x, y in d.items()},
                                       self.__k)

    __rmul__ = __mul__

    def __neg__(self):
        return self.__mul__(-1)

    def functional(self, X = None):
        return self.H().functional(X = X)

    def degree(self):
        return self.H().degree()
