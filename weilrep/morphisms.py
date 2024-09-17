r"""

Sage code for automorphisms of Weil representations

"""

# ****************************************************************************
#       Copyright (C) 2020-2023 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from inspect import isfunction
from itertools import chain

from sage.functions.other import frac
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from .weilrep_modular_forms_class import WeilRepModularForm

class WeilRepMorphism:
    r"""
    This class represents morphisms of discriminant forms.

    INPUT:

    Construct a WeilRepMorphism using WeilRepMorphism(w1, w2, f), where

    - ``w1``, ``w2`` -- WeilRep instances

    - ``f`` -- a function that inputs a vector in w1.ds() and outputs another vector in w2.ds()

    f is meant to be an morphism, i.e. f(x+y) = f(x) + f(y) and  Q2(f(x)) = Q1(x) for all x, y if Q1 is the quadratic form of w1 and Q2 of w2.
    """

    def __init__(self, w1, w2, f, check = True):
        if not isfunction(f):
            A = f
            f = lambda x: vector(map(frac, A * x))
        self.__input_weilrep = w1
        self.__output_weilrep = w2
        self.__f = f
        if check:
            dsdict = w2.ds_dict()
            dsgens = w1.ds_gens()
            s1 = w1.gram_matrix()
            s2 = w2.gram_matrix()
            gen_imgs = [None] * len(dsgens)
            for i, g in enumerate(dsgens):
                h = f(g)
                if frac(g * s1 * g / 2) != frac(h * s2 * h / 2):
                    raise ValueError('This is not a morphism.')
                for j, g2 in enumerate(dsgens[: i]):
                    h2 = f(g2)
                    if frac(g * s1 * g2) != frac(h * s2 * h2):
                        raise ValueError('This is not a morphism.')
                gen_imgs[i] = h
            self.__gen_imgs = gen_imgs
        if w1 == w2:
            self.__class__ = WeilRepAutomorphism

    def __repr__(self, gens = None):
        w = self.input_weilrep()
        f = self.__f
        if gens is None:
            gens = w.ds_gens()
        return 'Morphism from %s to %s\nmapping %s'%(w, self.output_weilrep(), ', '.join(['%s->%s'%(x, f(x)) for x in gens]))

    def f(self):
        r"""
        The underlying function.
        """
        return self.__f

    def _compute_indices(self):
        w1 = self.input_weilrep()
        w2 = self.output_weilrep()
        f = self.__f
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

    def input_weilrep(self):
        return self.__input_weilrep

    def output_weilrep(self):
        return self.__output_weilrep

    def inverse(self):
        r"""
        See __invert__
        """
        return ~self

    def __call__(self, X):
        r"""
        Apply the morphism to X, where X is
        -- a vector (specifically, an element of the input weilrep's discriminant group), or
        -- a WeilRepModularForm
        """
        try:
            if X.weilrep() != self.input_weilrep():
                raise ValueError('Invalid Weil representation')
            Xf = X.fourier_expansion()
            try:
                Z = self.__indices
            except AttributeError:
                self._compute_indices()
                Z = self.__indices
            f = self.f()
            w = self.output_weilrep()
            return WeilRepModularForm(X.weight(), w.gram_matrix(), [(f(Xf[i][0]), Xf[i][1], Xf[z][2]) for i,z in enumerate(Z)], weilrep = w)
        except AttributeError:
            return self.f()(X)

    def __eq__(self, other):
        w = self.input_weilrep()
        if w == other.input_weilrep():
            f1 = self.f()
            f2 = other.f()
            return all(f1(x) == f2(x) for x in w.ds())
        return False

    def __hash__(self):
        return hash(self.f())

    def __invert__(self):
        Z = self.__inverse
        if not Z:
            raise ValueError('This morphism is not invertible')
        w1, w2 = self.input_weilrep(), self.output_weilrep()
        d1, d2 = w1.ds(), w2.ds_dict()
        return WeilRepMorphism(w2, w1, lambda x: d1[Z[d2[tuple(x)]]])

    def matrix(self):
        raise NotImplementedError
        #try:
        #    return self.__matrix
        #except AttributeError:
        #    raise NotImplementedError('This morphism was not constructed as a matrix') from None
        #L = []
        #w = self.input_weilrep()
        #ds = w.ds()
        #i = 0
        #S = w.gram_matrix()
        #n = S.nrows()
        #A = matrix(L)
        #r = A.rank()
        #while r < n:
        #    v = ds[i]
        #    B = matrix(L + [v])
        #    if B.rank() > r:
        #        A = B
        #        r = A.rank()
        #        L = L + [v]
        #    i += 1
        #return (A.inverse() * matrix([self(a) for a in A])).transpose()

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

    def __repr__(self, gens = None):
        w = self.weilrep()
        f = self.f()
        if gens is None:
            gens = self.weilrep().ds_gens()
        try:
            d = w._unitary_ds_to_ds()
            d2 = w._ds_to_unitary_ds()
            uds = self.weilrep().unitary_ds()
            return 'Automorphism of %s\nmapping %s'%(self.weilrep(), ', '.join(['%s->%s'%(x, d2[tuple(f(d[tuple(x)]))]) for x in uds]))
        except AttributeError:
            return 'Automorphism of %s\nmapping %s'%(self.weilrep(), ', '.join(['%s->%s'%(x, f(x)) for x in gens]))

    def __pow__(self, n):
        r"""
        Iterated composition.
        """
        if not n:
            return self.weilrep()._identity_morphism()
        elif n < 0:
            return ~self.__pow__(-n)
        elif n == 1:
            return self
        nhalf = n // 2
        return self.__pow__(nhalf).__mul__(self.__pow__(n - nhalf))

class WeilRepAutomorphismGroup:

    def __init__(self, weilrep, G, group, name = None):
        self.__weilrep = weilrep
        self.__G = G
        self.__group = group
        self.__name = name
        if name:
            setG = set(G)
            self.__indices = [G.index(x) for x in setG]
            self.__G = [G[i] for i in self.__indices]

    def __iter__(self):
        for x in self.__G:
            yield x

    def __getitem__(self, x):
        return self.__G[x]

    def __len__(self):
        return len(self.__G)

    def __repr__(self):
        if self.__name:
            return '%s of %s'%(self.__name, self.weilrep())
        return 'Automorphism group of %s'%self.weilrep()

    def weilrep(self):
        return self.__weilrep

    def display(self, gens = None):
        h = '\n' + '-'*80 + '\n'
        print(h.join(x.__repr__(gens = gens) for x in self.__G))

    def gens(self):
        r"""
        Compute a list of automorphisms that generate this group.
        """
        gens = []
        G = self.group()
        LG = list(G)
        return [self.__G[LG.index(x)] for x in G.gens()]

    def group(self):
        return self.__group

    def characters(self):
        r"""
        Return the 'group of characters'. These are actually lists chi = [chi(g1),...,chi(gn)] where self.__G = [g1,...,gn]. Automorphism groups of discriminant forms are always generated by reflections so chi takes only values +/- 1
        """
        G = self.group()
        X = G.character_table()
        X = [x for x in X.rows() if x[0] == 1]
        Y = G.conjugacy_classes()
        Z = [list(chain.from_iterable([[ZZ(x)]*len(Y[i]) for i, x in enumerate(x)])) for x in X]
        Z = [[1]*len(G)] + [z for z in Z if any(x != 1 for x in z)]
        if self.__name:
            return [[z[i] for i in self.__indices] for z in Z]
        return Z

    def index(self, x):
        return self.__G.index(x)

    def orbits(self):
        r"""
        Compute the orbits of self's action on the discriminant group.

        OUTPUT: a list of lists

        EXAMPLES::

            sage: from weilrep import *
            sage: II(3).automorphism_group().orbits()
            [[(0, 0)], [(0, 2/3), (1/3, 0), (0, 1/3), (2/3, 0)], [(1/3, 1/3), (2/3, 2/3)], [(2/3, 1/3), (1/3, 2/3)]]
        """
        X = []
        L = []
        ds = self.weilrep().ds()
        r = []
        for i, g in enumerate(ds):
            tuple_g = tuple(g)
            try:
                _ = next(x for x in L if tuple_g in x)
            except StopIteration:
                y = set(tuple(h(g)) for h in self)
                L.append(y)
                X.append(list(y))
                r.append(i)
        self._orbit_rep_indices = r
        return X

    def orbit_representatives(self):
        r"""
        Compute a system of representatives of self's orbits.

        OUTPUT: a list

        EXAMPLES::

            sage: from weilrep import *
            sage: II(4).automorphism_group().orbit_representatives()
            [(0, 0), (1/4, 0), (1/2, 0), (1/4, 1/4), (1/2, 1/4), (3/4, 1/4), (1/2, 1/2)]
        """
        ds = self.weilrep().ds()
        return [ds[i] for i in self._orbit_representatives_indices()]

    def _orbit_representatives_indices(self):
        r"""
        Compute a list [i_1, i_2, i_3, ...] where ds[i_k] represent the orbits of self on the discriminant group.

        OUTPUT: a list

        EXAMPLES::

            sage: from weilrep import *
            sage: II(4).automorphism_group()._orbit_representatives_indices()
            [0, 1, 2, 5, 6, 7, 10]
        """
        try:
            return self._orbit_rep_indices
        except AttributeError:
            _ = self.orbits()
            return self._orbit_rep_indices
