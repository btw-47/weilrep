r"""

Algebraic modular forms on O(n)

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

from collections import Counter, defaultdict
from itertools import combinations, product

import cypari2
pari = cypari2.Pari()

import random

from sage.arith.misc import kronecker_symbol, next_prime
from sage.arith.srange import srange
from sage.functions.generalized import sgn
from sage.functions.other import binomial, ceil, floor
from sage.graphs.graph import Graph
from sage.matrix.constructor import matrix
from sage.matrix.special import identity_matrix
from sage.misc.cachefunc import cached_function
from sage.misc.prandom import randrange
from sage.modular.arithgroup.congroup_gamma0 import Gamma0_constructor as Gamma0
from sage.modules.free_module import span
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.finite_rings.finite_field_constructor import FiniteField as GF
from sage.rings.finite_rings.integer_mod import mod, square_root_mod_prime
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import NumberField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR

from .eisenstein_series import local_normal_form_with_change_vars
from .weilrep import WeilRep
from .weilrep_misc import relations




class AlgebraicModularForms(object):
    r"""
    This class represents spaces of algebraic modular forms on the compact groups O(n).

    INPUT: an OrthogonalModularForms instance is constructed by calling ``OrthogonalModularForms(S)``, where:
    - ``S`` -- a Gram matrix for a positive-definite lattice; or
    - ``S`` -- a quadratic form; or
    - ``S`` -- the WeilRep instance WeilRep(S)
    """

    def __init__(self, w, **kwargs):
        try:
            S = w.gram_matrix()
        except AttributeError:
            w = WeilRep(w)
            S = w.gram_matrix()
        if not w.is_positive_definite():
            raise ValueError('This lattice is not positive-definite.')
        self.__weilrep = w
        self.__gram_matrix = S
        self.__qf = w.quadratic_form()
        #dictionaries to store computations
        self.__bases = {}
        self.__hs_spin = {}
        self.__molien_spin = {}
        self.__neighbor_matrices = {}

    def __repr__(self):
        return 'Algebraic modular forms associated to the Gram matrix\n%s'%str(self.__gram_matrix)

    ## Attributes

    def automorphism_group(self):
        r"""
        Return the automorphism group of self.

        The result is a GAP matrix group. To use its elements (e.g. to multiply with them) you usually need to make it into a Sage matrix.
        (for example by mapping g to matrix(n, n, g))
        """
        try:
            return self.__aut
        except AttributeError:
            q = self.quadratic_form()
            g = q.automorphism_group()
            self.__aut = g
            return g

    def _base_ring(self):
        S = self.__gram_matrix
        return PolynomialRing(QQ, ['x_%s'%i for i in range(S.nrows())])

    def classes(self):
        r"""
        Return a list of AlgebraicModularForms instances representing the classes in self's genus.

        We compute them using p-neighbors which are then saved for later use.
        (This is slow if one only wants to know the classes because we compute all the p-neighbors.)

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(matrix([[4, 1, 1], [1, 6, 1], [1, 1, 4]]))
            sage: len(m.classes())
            2
        """
        try:
            return self.__cg
        except AttributeError:
            p = 2
            while self.level() % p == 0:
                p = next_prime(p)
            _ = self.neighbor_matrices(p, 1)
            return self.__cg

    def _automorphism_group_conj_classes(self):
        try:
            return self.__cc
        except AttributeError:
            g = self.automorphism_group()
            x = g.conjugacy_classes()
            self.__cc = x
            return x

    def genus(self):
        return self.weilrep().genus()

    def gram_matrix(self):
        return self.__gram_matrix

    def is_split(self, p):
        r"""
        Determine whether O(L) is split over \QQ_p.
        In other words whether L_p is hyperbolic.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['A', 6]))
            sage: m.is_split(2)
            True
        """
        S = self.gram_matrix()
        n = S.nrows()
        if n % 2:
            return False
        _, j = hyperbolic_splitting(S, p)
        return j + j == n

    def level(self):
        return self.weilrep().level()

    def _local_isometries(self):
        r"""
        Compute \QQ-isometries between the class representatives of self's genus.

        If the class representatives (specifically those computed by .classes()) are Gram matrices labelled S_1,...,S_n,
        then this computes isometries A_1,...,A_n over QQ, with denominators coprime to the discriminant, such that A_i.transpose() * S_i * A_i is self's Gram matrix.
        These are used to 'canonically' identify the classes of self's genus as lattices in self x QQ so that the spinor norm is well-defined.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]])
            sage: I = m._local_isometries()
            sage: X = m.classes()
            sage: all(I[i].transpose() * X[i].gram_matrix() * I[i] == m.gram_matrix() for i in range(len(X)))
            True
        """
        try:
            return self.__equiv
        except AttributeError:
            _ = self.classes()
            return self.__equiv

    def mass(self):
        r"""
        Compute the mass of self's genus.
        """
        return self.quadratic_form().conway_mass()

    def neighbor_graph(self, p, k = 1):
        r"""
        Compute the p^k-neighbor graph.

        INPUT:
        - ``p`` -- a prime not dividing the discriminant
        - ``k`` -- an integer

        OUTPUT: the multigraph whose vertices are the equivalence classes of the genus, where there is an edge between two vertices for each realization as p^k-neighbors.
        """
        classes = self.classes()
        _ = self.neighbor_matrices(p, k)
        d = {i: [] for i in range(len(classes))}
        for i, x in enumerate(classes):
            for j, y in enumerate(x.neighbor_matrices(p, k)):
                d[i].extend([j] * len(y))
        G = Graph(d)
        return G

    def neighbor_matrices(self, p, k, classes = None, num_automorphisms = None, equivalences = None):
        r"""
        Compute change-of-basis matrices between p^k-neighbors.

        NOTE: if representatives of the genus are not already known, then we compute them here. The representatives obtained this way are random.
        """
        try:
            return self.__neighbor_matrices[(p, k)]
        except KeyError:
            try:
                classes = self.__cg
                num_automorphisms = [x.number_of_automorphisms() for x in classes]
                equivalences = self._local_isometries()
            except AttributeError:
                if classes is None:
                    classes = [self]
                    num_automorphisms = [self.number_of_automorphisms()]
                    equivalences = [identity_matrix(self.gram_matrix().nrows())]
                pass
            def isometry(x, z, i):
                r"""
                Find a pair j, m such that m is an isometry between x and classes[j], or add a new class
                """
                nonlocal classes, equivalences, num_automorphisms
                q = QuadraticForm(x)
                for j, y in enumerate(classes):
                    q2 = y.quadratic_form()
                    a = q2.is_globally_equivalent_to(q, return_matrix = True)
                    if a:
                        return j, a
                classes.append(AlgebraicModularForms(x))
                num_automorphisms.append(Integer(q.number_of_automorphisms()))
                equivalences.append(z * equivalences[i])
                return len(classes) - 1, equivalences[0]
            U = []
            for i, x in enumerate(classes):
                u = [[] for _ in classes]
                S = x.gram_matrix()
                G = x.automorphism_group()
                L = pk_neighbors(S, G, p, k)
                for y, a, n in L:
                    j, b = isometry(y, a, i)
                    c = b*a
                    for h in n:
                        h = h.transpose().inverse()
                        try:
                            u[j].append(c*h)
                        except IndexError:
                            u.append([c*h])
                U.append(u)
            if sum(~x for x in num_automorphisms) != self.mass(): #lets try another prime
                _ = self.neighbor_matrices(next_prime(p), 1, classes = classes, num_automorphisms = num_automorphisms, equivalences = equivalences)
                old_classes = classes
                classes = self.__cg
                for u in U:
                    if len(u) != len(classes):
                        u.extend([0] * (len(classes) - len(u)))
                for i, x in enumerate(classes):
                    if i > len(old_classes):
                        u = [[] for _ in classes]
                        S = x.gram_matrix()
                        G = x.automorphism_group()
                        L = pk_neighbors(S, G, p, k)
                        for y, a, n in L:
                            j, b = isometry(y)
                            c = b * a
                            for h in n:
                                h = h.transpose()
                                try:
                                    u[j].append(c*h)
                                except IndexError:
                                    u.append([c*h])
            for i, x in enumerate(classes):
                x._AlgebraicModularForms__neighbor_matrices[(p, k)] = U[i]
            self.__cg = classes
            self.__equiv = equivalences
            return self.__neighbor_matrices[(p, k)]

    def number_of_automorphisms(self):
        return Integer(self.__qf.number_of_automorphisms())

    def quadratic_form(self):
        return self.__qf

    def _spin_numbers(self):
        r"""
        Let D be self's discriminant. For a divisor d | D we compute the numbers s_d defined as follows:

        s_d = -1 if there exists a self-isometry whose spinor norm is nontrivial mod d
        s_d = +1 otherwise.

        s_1 is always +1.

        NOTE: I do not know what the proper name for these numbers are.
        """
        try:
            return self.__spin
        except AttributeError:
            l = self.level()
            S = self.gram_matrix()
            n = S.nrows()
            spin = {(d, 1):1 for d in l.divisors() if d.is_squarefree()}
            spin.update({(d, -1):1 for d in l.divisors() if d.is_squarefree()})
            g = self.automorphism_group()
            for x in g.gens():
                x = matrix(n, n, x)
                N = spinor_norm(S, x)
                xdet = x.determinant()
                for d in l.divisors():
                    if d.is_squarefree():
                        chi = (-1)**sum(N.valuation(p) for p in d.prime_divisors())
                        if chi != 1 or xdet != 1:
                            spin[(d, xdet)] = -1
            self.__spin = spin
            return spin

    def weilrep(self):
        return self.__weilrep

    def __eq__(self, other):
        if not isinstance(other, AlgebraicModularForms):
            return False
        return self.gram_matrix() == other.gram_matrix()

    ## modular forms

    def basis(self, k, spin = 1, det = 1):
        r"""
        Compute a basis of algebraic modular forms of weight Har_k.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['D', 4]))
            sage: m.basis(6)
            [Algebraic modular form mapping
            [ 2 -1  0  0]
            [-1  2 -1 -1]
            [ 0 -1  2  0]
            [ 0 -1  0  2]
            to
            -x_0^6 + 3*x_0^5*x_1 - 5*x_0^3*x_1^3 + 3*x_0*x_1^5 - x_1^6 - 5*x_0^4*x_1*x_2 + 10*x_0^3*x_1^2*x_2 + 5*x_0^2*x_1^3*x_2 - 10*x_0*x_1^4*x_2 + 3*x_1^5*x_2 + 5*x_0^4*x_2^2 - 10*x_0^3*x_1*x_2^2 + 5*x_0*x_1^3*x_2^2 - 10*x_0^2*x_1*x_2^3 + 10*x_0*x_1^2*x_2^3 - 5*x_1^3*x_2^3 + 5*x_0^2*x_2^4 - 5*x_0*x_1*x_2^4 + 3*x_1*x_2^5 - x_2^6 - 5*x_0^4*x_1*x_3 + 10*x_0^3*x_1^2*x_3 + 5*x_0^2*x_1^3*x_3 - 10*x_0*x_1^4*x_3 + 3*x_1^5*x_3 - 30*x_0^2*x_1^2*x_2*x_3 + 30*x_0*x_1^3*x_2*x_3 - 10*x_1^4*x_2*x_3 + 30*x_0^2*x_1*x_2^2*x_3 - 30*x_0*x_1^2*x_2^2*x_3 + 5*x_1^3*x_2^2*x_3 + 10*x_1^2*x_2^3*x_3 - 5*x_1*x_2^4*x_3 + 5*x_0^4*x_3^2 - 10*x_0^3*x_1*x_3^2 + 5*x_0*x_1^3*x_3^2 + 30*x_0^2*x_1*x_2*x_3^2 - 30*x_0*x_1^2*x_2*x_3^2 + 5*x_1^3*x_2*x_3^2 - 30*x_0^2*x_2^2*x_3^2 + 30*x_0*x_1*x_2^2*x_3^2 - 10*x_1*x_2^3*x_3^2 + 5*x_2^4*x_3^2 - 10*x_0^2*x_1*x_3^3 + 10*x_0*x_1^2*x_3^3 - 5*x_1^3*x_3^3 + 10*x_1^2*x_2*x_3^3 - 10*x_1*x_2^2*x_3^3 + 5*x_0^2*x_3^4 - 5*x_0*x_1*x_3^4 - 5*x_1*x_2*x_3^4 + 5*x_2^2*x_3^4 + 3*x_1*x_3^5 - x_3^6]
        """
        try:
            return self.__bases[(k, spin, det)]
        except KeyError:
            pass
        dim = self.dimension(k, separate_classes = True, spin = spin, det = det)
        R = self._base_ring()
        L = []
        for i, x in enumerate(self.classes()):
            if k:
                y = invariant_weight_k_polynomials_with_dim_bound(x.gram_matrix(), x.automorphism_group(), k, dim[i], spin = spin, det = det)
                y = [b / b.content() for b in y]
            elif x._spin_numbers()[(spin, det)]:
                y = [R(1)]
            else:
                y = []
            L.append(y)
        a = []
        zero = R(0)
        n = len(L)
        for i, y in enumerate(L):
            b = [AlgebraicModularForm(self, [zero]*i + [x] + [zero]*(n - 1 - i), k, spin = spin, det = det) for x in y]
            a.extend(b)
        self.__bases[(k, spin)] = a
        return a

    def eisenstein_series(self):
        r"""
        Compute the Eisenstein series.
        (This is the modular form of weight 0 that maps each class to 1.)
        """
        x = self.classes()
        R = self._base_ring()
        return AlgebraicModularForm(self, [R(1) for _ in x], 0)

    def hilbert_series(self, spin = 1, det=1):
        r"""
        Compute the Hilbert series
        \sum (dim M(Har_k)) * t^k
        as a rational function of t.

        INPUT:
        - ``spin`` (optional, default = 1). This is a squarefree divisor of the discriminant. If spin = d is nontrivial, then we twist by the spinor character spin_d
        (if d = p is prime, then this is just the spinor norm over QQ_p).

        - ``det`` (optional, default = 1). If det = -1, then we twist by the determinant character det of O(L).

        Twisting by a character \chi means that we compute the Hilbert series
        \sum (dim M(Har_k x \chi)) * t^k

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['D', 4]))
            sage: m.hilbert_series()
            -1/(t^26 - t^20 - t^18 - t^14 + t^12 + t^8 + t^6 - 1)

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['D', 4]))
            sage: m.hilbert_series(spin = 2)
            -t^12/(t^26 - t^20 - t^18 - t^14 + t^12 + t^8 + t^6 - 1)

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['D', 4]))
            sage: m.hilbert_series(det = -1)
            -t^24/(t^26 - t^20 - t^18 - t^14 + t^12 + t^8 + t^6 - 1)
        """
        try:
            if spin == 1 and det == 1:
                return self.__hs
            else:
                return self.__hs_spin[(spin, det)]
        except (AttributeError, KeyError):
            r, t = PolynomialRing(QQ, 't').objgen()
            x = self.classes()
            s = 0
            molien_series = []
            n = self.gram_matrix().nrows()
            for m in x:
                f = 0
                S = m.gram_matrix()
                if spin == 1:
                    if det == 1:
                        character = lambda y: 1
                    else:
                        character = lambda y: y.determinant()
                else:
                    if det == 1:
                        def character(y):
                            N = spinor_norm(S, y)
                            return (-1)**sum(N.valuation(p) for p in spin.prime_factors())
                    else:
                        def character(y):
                            N = spinor_norm(S, y)
                            return y.determinant() * (-1)**sum(N.valuation(p) for p in spin.prime_factors())
                N = m.number_of_automorphisms()
                cc = m._automorphism_group_conj_classes()
                for u in cc:
                    y = matrix(n, n, u.representative())
                    f += character(y) * u.cardinality() * (1 - t * t) / (N * r(t**n * y.charpoly()(~t)))
                molien_series.append(f)
                s += f
            if spin == 1 and det == 1:
                self.__hs = s
                self.__molien = molien_series
            else:
                self.__hs_spin[(spin, det)] = s
                self.__molien_spin[(spin, det)] = molien_series
            return s

    def dimension(self, k, spin = 1, det = 1, separate_classes = False):
        r"""
        Compute the dimension of the space of algebraic modular forms of weight Har_k.

        INPUT:
        - ``k`` -- the weight (a natural number or 0)
        - ``spin`` -- the spinor character (see .hilbert_series())
        - ``det`` -- the determinant character (see .hilbert_series())
        - ``separate_classes`` -- Boolean (default False). If True then we ONLY compute the space of harmonic polynomials that transform under self's orthogonal group O(L) with the specified character.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['D', 5]))
            sage: m.dimension(40)
            28

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['A', 5]))
            sage: m.dimension(5, spin=3)
            1
        """
        if self.level() % spin:
            raise ValueError('The spin character must correspond to a divisor of the level %s'%self.level())
        r = PowerSeriesRing(QQ, 't', (k + 1))
        p = self.hilbert_series(spin = spin, det = det)
        if separate_classes:
            if spin == 1 and det == 1:
                p = self.__molien
            else:
                p = self.__molien_spin[(spin, det)]
            f = [r(p.numerator()) / r(p.denominator()) for p in p]
            return [f[k] for f in f]
        f = r(p.numerator()) / r(p.denominator())
        return f[k]

    def theta_kernel(self, k, spin = 1, det = 1):
        r"""
        Compute a basis of the kernel of the theta map.

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(matrix([[2, 1, 0, 0, 0], [1, 2, 1, 0, 0], [0, 1, 2, 1, 0], [0, 0, 1, 2, 0], [0, 0, 0, 0, 2]]))
            sage: len(m.theta_kernel(8))
            1
        """
        X = self.basis(k, spin = spin, det = det)
        V = polynomial_relations([x.theta().polynomial() for x in X]).basis_matrix()
        return [sum(v[i] * x for i, x in enumerate(X)) for v in V.rows()]


    ## Hecke etc

    def eigenforms(self, X, spin = 1, det = 1, _p = 2, _name = '', _final_recursion = True, _K_list = []):
        r"""
        Compute eigenforms.

        If k is a natural number or 0, then
        self.eigenforms(k)
        produces Galois representatives of the eigenforms of weight k.

        If X is a list of AlgebraicModularForm instances, then
        self.eigenforms(X)
        produces Galois representatives of the eigenforms contained in span(X).
        (If span(X) is not invariant under the Hecke operators then this should produce some sort of error!!)

        EXAMPLES::

            sage: from weilrep import *
            sage: m = AlgebraicModularForms(CartanMatrix(['A', 4]))
            sage: m.eigenforms(8)
            [Algebraic modular form mapping
            [ 2 -1  0  0]
            [-1  2 -1  0]
            [ 0 -1  2 -1]
            [ 0  0 -1  2]
            to
            (-71/25536*a_0 + 85/336)*x_0^8 + (71/6384*a_0 - 85/84)*x_0^7*x_1 + (-65/17024*a_0 + 275/224)*x_0^6*x_1^2 + (-1403/51072*a_0 - 95/672)*x_0^5*x_1^3 + (367/8512*a_0 - 45/112)*x_0^4*x_1^4 + (-1403/51072*a_0 - 95/672)*x_0^3*x_1^5 + (-65/17024*a_0 + 275/224)*x_0^2*x_1^6 + (71/6384*a_0 - 85/84)*x_0*x_1^7 + (-71/25536*a_0 + 85/336)*x_1^8 + (-2573/76608*a_0 + 4855/1008)*x_0^6*x_1*x_2 + (2573/25536*a_0 - 4855/336)*x_0^5*x_1^2*x_2 + (-20351/153216*a_0 + 43885/2016)*x_0^4*x_1^3*x_2 + (197/2016*a_0 - 9805/504)*x_0^3*x_1^4*x_2 + (233/51072*a_0 + 5045/672)*x_0^2*x_1^5*x_2 + (-1403/38304*a_0 - 95/504)*x_0*x_1^6*x_2 + (71/6384*a_0 - 85/84)*x_1^7*x_2 + (2573/76608*a_0 - 4855/1008)*x_0^6*x_2^2 + (-2573/25536*a_0 + 4855/336)*x_0^5*x_1*x_2^2 + (1165/51072*a_0 + 25225/672)*x_0^4*x_1^2*x_2^2 + (4685/38304*a_0 - 49975/504)*x_0^3*x_1^3*x_2^2 + (-301/3648*a_0 + 2135/48)*x_0^2*x_1^4*x_2^2 + (233/51072*a_0 + 5045/672)*x_0*x_1^5*x_2^2 + (-65/17024*a_0 + 275/224)*x_1^6*x_2^2 + (301/1368*a_0 - 2135/18)*x_0^4*x_1*x_2^3 + (-301/684*a_0 + 2135/9)*x_0^3*x_1^2*x_2^3 + (4685/38304*a_0 - 49975/504)*x_0^2*x_1^3*x_2^3 + (197/2016*a_0 - 9805/504)*x_0*x_1^4*x_2^3 + (-1403/51072*a_0 - 95/672)*x_1^5*x_2^3 + (-301/2736*a_0 + 2135/36)*x_0^4*x_2^4 + (301/1368*a_0 - 2135/18)*x_0^3*x_1*x_2^4 + (1165/51072*a_0 + 25225/672)*x_0^2*x_1^2*x_2^4 + (-20351/153216*a_0 + 43885/2016)*x_0*x_1^3*x_2^4 + (367/8512*a_0 - 45/112)*x_1^4*x_2^4 + (-2573/25536*a_0 + 4855/336)*x_0^2*x_1*x_2^5 + (2573/25536*a_0 - 4855/336)*x_0*x_1^2*x_2^5 + (-1403/51072*a_0 - 95/672)*x_1^3*x_2^5 + (2573/76608*a_0 - 4855/1008)*x_0^2*x_2^6 + (-2573/76608*a_0 + 4855/1008)*x_0*x_1*x_2^6 + (-65/17024*a_0 + 275/224)*x_1^2*x_2^6 + (71/6384*a_0 - 85/84)*x_1*x_2^7 + (-71/25536*a_0 + 85/336)*x_2^8 + (-2573/76608*a_0 + 4855/1008)*x_0^6*x_2*x_3 + (2573/25536*a_0 - 4855/336)*x_0^5*x_1*x_2*x_3 + (-18517/51072*a_0 + 265295/672)*x_0^4*x_1^2*x_2*x_3 + (3049/5472*a_0 - 55115/72)*x_0^3*x_1^3*x_2*x_3 + (-18517/51072*a_0 + 265295/672)*x_0^2*x_1^4*x_2*x_3 + (2573/25536*a_0 - 4855/336)*x_0*x_1^5*x_2*x_3 + (-2573/76608*a_0 + 4855/1008)*x_1^6*x_2*x_3 + (31/3192*a_0 - 10685/42)*x_0^4*x_1*x_2^2*x_3 + (-31/1596*a_0 + 10685/21)*x_0^3*x_1^2*x_2^2*x_3 + (-1289/6384*a_0 + 17515/84)*x_0^2*x_1^3*x_2^2*x_3 + (193/912*a_0 - 5555/12)*x_0*x_1^4*x_2^2*x_3 + (2573/25536*a_0 - 4855/336)*x_1^5*x_2^2*x_3 + (301/1368*a_0 - 2135/18)*x_0^4*x_2^3*x_3 + (-301/684*a_0 + 2135/9)*x_0^3*x_1*x_2^3*x_3 + (5627/6384*a_0 - 90145/84)*x_0^2*x_1^2*x_2^3*x_3 + (-12667/19152*a_0 + 240545/252)*x_0*x_1^3*x_2^3*x_3 + (-20351/153216*a_0 + 43885/2016)*x_1^4*x_2^3*x_3 + (-193/912*a_0 + 5555/12)*x_0^2*x_1*x_2^4*x_3 + (193/912*a_0 - 5555/12)*x_0*x_1^2*x_2^4*x_3 + (197/2016*a_0 - 9805/504)*x_1^3*x_2^4*x_3 + (-2573/25536*a_0 + 4855/336)*x_0^2*x_2^5*x_3 + (2573/25536*a_0 - 4855/336)*x_0*x_1*x_2^5*x_3 + (233/51072*a_0 + 5045/672)*x_1^2*x_2^5*x_3 + (-1403/38304*a_0 - 95/504)*x_1*x_2^6*x_3 + (71/6384*a_0 - 85/84)*x_2^7*x_3 + (2573/76608*a_0 - 4855/1008)*x_0^6*x_3^2 + (-2573/25536*a_0 + 4855/336)*x_0^5*x_1*x_3^2 + (18517/51072*a_0 - 265295/672)*x_0^4*x_1^2*x_3^2 + (-3049/5472*a_0 + 55115/72)*x_0^3*x_1^3*x_3^2 + (18517/51072*a_0 - 265295/672)*x_0^2*x_1^4*x_3^2 + (-2573/25536*a_0 + 4855/336)*x_0*x_1^5*x_3^2 + (2573/76608*a_0 - 4855/1008)*x_1^6*x_3^2 + (-31/3192*a_0 + 10685/42)*x_0^4*x_1*x_2*x_3^2 + (31/1596*a_0 - 10685/21)*x_0^3*x_1^2*x_2*x_3^2 + (1289/6384*a_0 - 17515/84)*x_0^2*x_1^3*x_2*x_3^2 + (-193/912*a_0 + 5555/12)*x_0*x_1^4*x_2*x_3^2 + (-2573/25536*a_0 + 4855/336)*x_1^5*x_2*x_3^2 + (-301/912*a_0 + 2135/12)*x_0^4*x_2^2*x_3^2 + (301/456*a_0 - 2135/6)*x_0^3*x_1*x_2^2*x_3^2 + (-1289/1064*a_0 + 17515/14)*x_0^2*x_1^2*x_2^2*x_3^2 + (5627/6384*a_0 - 90145/84)*x_0*x_1^3*x_2^2*x_3^2 + (1165/51072*a_0 + 25225/672)*x_1^4*x_2^2*x_3^2 + (1289/6384*a_0 - 17515/84)*x_0^2*x_1*x_2^3*x_3^2 + (-1289/6384*a_0 + 17515/84)*x_0*x_1^2*x_2^3*x_3^2 + (4685/38304*a_0 - 49975/504)*x_1^3*x_2^3*x_3^2 + (18517/51072*a_0 - 265295/672)*x_0^2*x_2^4*x_3^2 + (-18517/51072*a_0 + 265295/672)*x_0*x_1*x_2^4*x_3^2 + (-301/3648*a_0 + 2135/48)*x_1^2*x_2^4*x_3^2 + (233/51072*a_0 + 5045/672)*x_1*x_2^5*x_3^2 + (-65/17024*a_0 + 275/224)*x_2^6*x_3^2 + (301/1368*a_0 - 2135/18)*x_0^4*x_2*x_3^3 + (-301/684*a_0 + 2135/9)*x_0^3*x_1*x_2*x_3^3 + (301/456*a_0 - 2135/6)*x_0^2*x_1^2*x_2*x_3^3 + (-301/684*a_0 + 2135/9)*x_0*x_1^3*x_2*x_3^3 + (301/1368*a_0 - 2135/18)*x_1^4*x_2*x_3^3 + (31/1596*a_0 - 10685/21)*x_0^2*x_1*x_2^2*x_3^3 + (-31/1596*a_0 + 10685/21)*x_0*x_1^2*x_2^2*x_3^3 + (-301/684*a_0 + 2135/9)*x_1^3*x_2^2*x_3^3 + (-3049/5472*a_0 + 55115/72)*x_0^2*x_2^3*x_3^3 + (3049/5472*a_0 - 55115/72)*x_0*x_1*x_2^3*x_3^3 + (4685/38304*a_0 - 49975/504)*x_1^2*x_2^3*x_3^3 + (197/2016*a_0 - 9805/504)*x_1*x_2^4*x_3^3 + (-1403/51072*a_0 - 95/672)*x_2^5*x_3^3 + (-301/2736*a_0 + 2135/36)*x_0^4*x_3^4 + (301/1368*a_0 - 2135/18)*x_0^3*x_1*x_3^4 + (-301/912*a_0 + 2135/12)*x_0^2*x_1^2*x_3^4 + (301/1368*a_0 - 2135/18)*x_0*x_1^3*x_3^4 + (-301/2736*a_0 + 2135/36)*x_1^4*x_3^4 + (-31/3192*a_0 + 10685/42)*x_0^2*x_1*x_2*x_3^4 + (31/3192*a_0 - 10685/42)*x_0*x_1^2*x_2*x_3^4 + (301/1368*a_0 - 2135/18)*x_1^3*x_2*x_3^4 + (18517/51072*a_0 - 265295/672)*x_0^2*x_2^2*x_3^4 + (-18517/51072*a_0 + 265295/672)*x_0*x_1*x_2^2*x_3^4 + (1165/51072*a_0 + 25225/672)*x_1^2*x_2^2*x_3^4 + (-20351/153216*a_0 + 43885/2016)*x_1*x_2^3*x_3^4 + (367/8512*a_0 - 45/112)*x_2^4*x_3^4 + (-2573/25536*a_0 + 4855/336)*x_0^2*x_2*x_3^5 + (2573/25536*a_0 - 4855/336)*x_0*x_1*x_2*x_3^5 + (-2573/25536*a_0 + 4855/336)*x_1^2*x_2*x_3^5 + (2573/25536*a_0 - 4855/336)*x_1*x_2^2*x_3^5 + (-1403/51072*a_0 - 95/672)*x_2^3*x_3^5 + (2573/76608*a_0 - 4855/1008)*x_0^2*x_3^6 + (-2573/76608*a_0 + 4855/1008)*x_0*x_1*x_3^6 + (2573/76608*a_0 - 4855/1008)*x_1^2*x_3^6 + (-2573/76608*a_0 + 4855/1008)*x_1*x_2*x_3^6 + (-65/17024*a_0 + 275/224)*x_2^2*x_3^6 + (71/6384*a_0 - 85/84)*x_2*x_3^7 + (-71/25536*a_0 + 85/336)*x_3^8]
        """
        if isinstance(X, Integer):
            X = self.basis(X, spin = spin, det = det)
            return self.eigenforms(X, spin = spin, det = det)
        if not X:
            return []
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
            else: ## this will get ugly if multiplicity-one fails for some reason. For maximal lattices it is probably OK.
                _name = _name + '%s_'%i
                K_list_2, eigenvectors = self.eigenforms([sum(v[i]*X[i] for i in range(len(v))) for v in V_rows], _p = next_prime(_p), _name = _name, _final_recursion = False, _K_list = K_list)
                K_list.extend(K_list_2)
                L.extend(eigenvectors)
        L = [sum(X[i] * y for i, y in enumerate(x)) for x in L]
        eigenforms = []
        for i, x in enumerate(L):
            x.__class__ = AlgebraicModularFormEigenform
            x._AlgebraicModularFormEigenform__field = K_list[i]
            eigenforms.append(x)
        return eigenforms

    def hecke_operator(self, p, d = 1, safe = True):
        r"""
        Construct the Hecke operator T_{p, d}.
        This is the Hecke operator defined using p^d-neighbors.

        INPUT:
        - ``p`` -- a prime
        - ``d`` -- a natural number (default d=1)

        OUTPUT: AlgebraicModularFormHeckeOperator
        """
        if self.level() % p or safe == False:
            return AlgebraicModularFormHeckeOperator(self, p, d = d)
        raise NotImplementedError('Hecke operators are only implemented for primes not dividing the discriminant.')







class AlgebraicModularForm(object):
    r"""
    Class to represent an algebraic modular form.

    Should not be called directly.
    Initialize with the following data
    - ``amf`` -- AlgebraicModularForms instance (generally the one that created this form)
    - ``f`` -- a list of polynomials, one for each class in the genus of amf
    - ``weight`` -- self's weight (an integer k, representing Har_k)
    - ``spin``, ``det`` -- character twist
    """

    def __init__(self, amf, f, weight, spin = 1, det = 1):
        self.__amf = amf
        self.__f = f
        self.__weight = weight
        self.__spin = spin
        self.__det = det

    def __repr__(self):
        r"""
        When printed. This can be long, so don't do it unless you need to.
        """
        try:
            return self.__str
        except AttributeError:
            f = self.__f
            s = 'Algebraic modular form mapping\n' + '\n'.join(['%s\nto\n%s,\n'%(str(x.gram_matrix()), str(f[i])) for i, x in enumerate(self.__amf.classes())])
            s = s[:-2]
            self.__str = s
            return s

    def amf(self):
        r"""
        Return self's AlgebraicModularForms instance.
        """
        return self.__amf

    def base_field(self):
        r"""
        Return the coefficient field. If self is a Hecke eigenform then this is the field generated by the eigenvalues.
        """
        return self._f()[0].base_ring()

    def _f(self):
        return self.__f

    def gram_matrix(self):
        return self.__amf.gram_matrix()

    def det(self):
        return self.__det

    def spin(self):
        return self.__spin

    def weight(self):
        return self.__weight

    def __call__(self, x):
        r"""
        Return the polynomial corresponding to x.

        x should be a quadratic form in the correct genus.
        """
        amf = self.__amf
        classes = amf.classes()
        f = self.__f
        try:
            x = x.gram_matrix()
        except AttributeError:
            try:
                x = x.matrix()
            except AttributeError:
                pass
        classes = [y.quadratic_form() for y in classes]
        g = [y.matrix() for y in classes]
        try:
            i = g.index(x)
            return f[i]
        except ValueError:
            q = QuadraticForm(ZZ, x)
            try:
                i = next(i for i, y in enumerate(classes) if y.is_globally_equivalent_to(q))
                return f[i]
            except StopIteration:
                raise ValueError('%s is not in the correct genus!'%str(x)) from None

    ## Arithmetic operations

    def __add__(self, other):
        if not other:
            return self
        if self.amf() != other.amf() or self.weight() != other.weight() or self.spin() != other.spin() or self.det() != other.det():
            raise TypeError
        f1 = self._f()
        f2 = other._f()
        return AlgebraicModularForm(self.amf(), [f1[i] + f2[i] for i in range(len(f1))], self.weight(), spin=self.spin(), det=self.det())
    __radd__ = __add__

    def __sub__(self, other):
        if self.amf() != other.amf() or self.weight() != other.weight() or self.spin() != other.spin() or self.det() != other.det():
            raise TypeError
        f1 = self._f()
        f2 = other._f()
        return AlgebraicModularForm(self.amf(), [f1[i] - f2[i] for i in range(len(f1))], self.weight(), spin=self.spin(), det=self.det())

    def __mul__(self, other):
        r"""
        Multiplication of algebraic modular forms is defined by means of the harmonic projection.
        This makes algebraic modular forms a commutative graded ring.
        """
        if isinstance(other, AlgebraicModularForm):
            amf = self.amf()
            if amf != other.amf():
                raise TypeError
            f1 = self._f()
            f2 = other._f()
            classes = amf.classes()
            spin = (self.spin() * other.spin()).squarefree_part()
            return AlgebraicModularForm(self.amf(), [harmonic_projection(f1[i] * f2[i], classes[i].gram_matrix()) for i in range(len(f1))], self.weight() + other.weight(), spin=spin)
        return AlgebraicModularForm(self.amf(), [other * x for x in self._f()], self.weight(), spin=self.spin(), det=self.det())
    __rmul__ = __mul__

    def __div__(self, N):
        return AlgebraicModularForm(self.amf(), [x / N for x in self._f()], self.weight(), spin=self.spin(), det=self.det())
    __truediv__ = __div__


    ## Theta map

    def theta(self, bound = None):
        r"""
        Apply the Theta map to self.

        The Fourier coefficients of the theta functions are computed up to bound.
        If bound is not given then we use the Sturm bound.
        """
        amf = self.__amf
        classes = amf.classes()
        f = self._f()
        q, = PowerSeriesRing(self.base_field(), 'q').gens()
        lvl = amf.level()
        k = self.weight()
        n = amf.gram_matrix().nrows() + k + k
        if bound is None:
            bound = floor(Gamma0(lvl).index() * n / 24) + 1
        if k > 0:
            def theta_series(x, p):
                if not p:
                    return 0
                x = x.gram_matrix()
                _, _, v = pari(x).qfminim(bound + bound + 1)
                v = v.sage().columns()
                return sum(q ** (Integer(v * x * v / 2)) * p(*v) for v in v)
            return sum(theta_series(classes[i], p) / classes[i].number_of_automorphisms() for i, p in enumerate(f)).add_bigoh(bound)
        return sum(f[i][[0]*n] * x.quadratic_form().theta_series(bound) / x.number_of_automorphisms() for i, x in enumerate(classes))



class AlgebraicModularFormHeckeOperator(object):
    r"""
    This class represents Hecke operators that act on algebraic modular forms for a definite orthogonal group O(n) = O(S).

    If p is a prime and d \in \NN, then the Hecke operator T_{p, d} is defined using the "p^d-neighbor" relation.
    """
    def __init__(self, amf, p, d = 1):
        self.__index = p
        self.__degree = d
        self.__amf = amf

    def __repr__(self):
        return 'Hecke operator of index %s and degree %s acting on %s'%(self.__index, self.__degree, self.__amf)

    def amf(self):
        return self.__amf

    def degree(self):
        return self.__degree

    def index(self):
        return self.__index

    def __call__(self, f):
        r"""
        Apply the Hecke operator (self) to the AlgebraicModularForm f.

        INPUT:
        - "f" -- AlgebraicModularForm

        OUTPUT: AlgebraicModularForm
        """
        amf = self.__amf
        k = f.weight()
        p = self.__index
        d = self.__degree
        S = amf.gram_matrix()
        R = amf._base_ring()
        rgens = vector(R.gens())
        spin = f.spin()
        det = f.det()
        classes = amf.classes()
        def spinor_character(A):
            return (-1)**len([p for p in spinor_norm(S, A).prime_factors() if spin % p == 0])
        if det == 1:
            if spin == 1:
                character = lambda A: 1
            else:
                character = spinor_character
        else:
            if spin == 1:
                character = lambda A: A.determinant()
            else:
                character = lambda A: A.determinant() * spinor_character(A)
        h = [R(0)] * len(classes)
        iso = amf._local_isometries()
        for i, x in enumerate(classes):
            nm = x.neighbor_matrices(p, d)
            b1 = iso[i]
            for j, b_list in enumerate(nm):
                if b_list:
                    b0 = iso[j]
                    for b in b_list:
                        c = character(b0.inverse() * b * b1)
                        if k:
                            h[i] += c * f._f()[j](*(b * rgens))
                        else:
                            h[i] += c * f._f()[j]
        #normalization factor p^k?
        p_k = p ** k
        return AlgebraicModularForm(amf, [p_k * x for x in h], k, spin=spin, det=det)

    def _evaluate_at_point(self, f, v):
        r"""
        Apply the Hecke operator to the algebraic modular form f, then evaluate at a point v \in \QQ^n.

        This is a lot faster than __call__ because we do not have to manipulate polynomials.
        We use it to compute the matrix representation of Hecke operators with respect to a basis.
        """
        amf = self.__amf
        k = f.weight()
        p = self.__index
        d = self.__degree
        spin = f.spin()
        classes = amf.classes()
        det = f.det()
        def spinor_character(A):
            return (-1)**len([p for p in spinor_norm(S, A).prime_factors() if spin % p == 0])
        if det == 1:
            if spin == 1:
                character = lambda A: 1
            else:
                character = spinor_character
        else:
            if spin == 1:
                character = lambda A: A.determinant()
            else:
                character = lambda A: A.determinant() * spinor_character(A)
        h = [0] * len(classes)
        nm0 = amf.neighbor_matrices(p, d)
        S = amf.gram_matrix()
        iso = amf._local_isometries()
        for i, x in enumerate(classes):
            nm = x.neighbor_matrices(p, d)
            b1 = iso[i]
            for j, b_list in enumerate(nm):
                if b_list:
                    b0 = iso[j]
                    for b in b_list:
                        c = character(b0.inverse() * b * b1)
                        if k:
                            h[i] += c * f._f()[j](*(b * v))
                        else:
                            h[i] += c * f._f()[j]
        #normalization factor p^k?
        p_k = p ** k
        return [p_k * x for x in h]


    def matrix(self, X):
        r"""
        Compute the representation matrix of the Hecke operator with respect to a basis.

        We do this by evaluating the basis and its image at random points in \QQ^n and comparing the results.
        """
        x = []
        w = []
        n = self.__amf.gram_matrix().nrows()
        m = 0
        len_X = len(X)
        if not len_X:
            return matrix([[]])
        N = len(X[0]._f())
        denominator = len_X * n
        k = X[0].weight()
        if k:
            while m < len(X):
                v = vector(RR(random.random()).nearby_rational(max_denominator = denominator) for _ in range(n))
                y = x
                for i in range(N):
                    y = y + [[f._f()[i](*v) for f in X]]
                m1 = matrix(y).rank()
                if m1 > m:
                    x = y
                    w.append(v)
                    m = m1
            m = [[y for v in w for u in self._evaluate_at_point(f, v) for y in u] for f in X]
            return matrix(x).solve_right(matrix(m).transpose())
        else:
            x = [[f._f()[i].base_ring()(f._f()[i]) for f in X] for i in range(N)]
            Y = [self.__call__(f) for f in X]
            m = [[y._f()[i].base_ring()(y._f()[i]) for y in Y] for i in range(N)]
        return matrix(x).solve_left(matrix(m))


class AlgebraicModularFormEigenform(AlgebraicModularForm):

    def eigenvalue(self, p, d=1):
        K = self.base_field()
        if not self.amf().level() % p:
            raise ValueError('The prime %s divides the lattice level %s.'%(p, self.amf().level()))
        T = self.amf().hecke_operator(p, d=d)
        n = self.amf().gram_matrix().nrows()
        denominator = 10 * n
        while 1:
            v = vector(RR(random.random()).nearby_rational(max_denominator = denominator) for _ in range(n))
            f = T._evaluate_at_point(self, v)
            try:
                x = next(x for x in enumerate(f) if x[1])
                g = self._f()[x[0]]
                return K(x[1] / g(*v))
            except StopIteration:
                pass

    def euler_factor(self, p):
        r"""
        Compute the Euler factor at a "good" prime p in self's L-function.

        NOTE: this only works if the rank of the lattice is <= 8.
        We use Murphy's formulas for the explicit Satake transform.
        """
        rank = self.gram_matrix().nrows()
        k = self.weight()
        p_k = p**k
        X, = PolynomialRing(self.field(), 'X').gens()
        if rank == 3:
            e1 = self.eigenvalue(p)
            return 1 - e1*X + p**(2 * k + 1) * X * X
        if rank == 4:
            e1 = self.eigenvalue(p) / p_k
            if self.amf().is_split(p):
                e2 = self.eigenvalue(p, d=2) / p_k
                g = 1 - e1 * X + p * (e2 + 2) * X**2 - p * p * e1 * X**3 + p**4 * X**4
                return g(p_k * X)
            g = (1 - p**2 * X**2) * (1 - e1 * X + p**2 * X * X)
            return g(p_k * X)
        if rank == 5:
            e1 = self.eigenvalue(p) / p_k
            e2 = self.eigenvalue(p, d = 2) / p_k
            g = 1 - e1*X + (e2 + 1 + p * p) * p * X * X - e1 * p**3 * X**3 + p**6 * X**4
            return g(p_k * X)
        if rank == 6:
            e1 = self.eigenvalue(p) / p_k
            e2 = self.eigenvalue(p, d=2) / p_k
            if self.amf().is_split(p):
                e3 = self.eigenvalue(p, d=3) / p_k
                g = 1 - e1 * X + (e2 + 1 + p + p*p) * p * X * X - (e1 + e1 + e3) * (p * X)**3 + (e2 + 1 + p + p*p) * p**5 * X**4 - e1 * p**8 * X**5 + p**12 * X**6
                return g(p_k * X)
            g = (1 - p**4 * X * X) * (1 - e1 * X + (e2 + p**3 + p**2 - p + 1) * p * X * X - p**4 * e1 * X**3 + p**8 * X**4)
            return g(p_k * X)
        if rank == 7:
            e1 = self.eigenvalue(p) / p_k
            e2 = self.eigenvalue(p, d=2) / p_k
            e3 = self.eigenvalue(p, d=3) / p_k
            g = 1 - e1 * X + p * (e2 + p**4 + p * p + 1) * X**2 - p**3 * ((1 + p * p ) * e1 + e3) * X**3 + p**6 * (e2 + p**4 + p * p + 1) * X**4 - p**10 * e1 * X**5 + p**15 * X**6
            return g(p_k * X)
        if rank == 8:
            e1 = self.eigenvalue(p) / p_k
            e2 = self.eigenvalue(p, d=2) / p_k
            e3 = self.eigenvalue(p, d=3) / p_k
            if self.amf().is_split(p):
                e4 = self.eigenvalue(p, d=4) / p_k
                g = 1 - e1 * X + p * (e2 + p**4 + 2 * p**2 + 1) * X**2 - p**3 * (e3 + e1 * (p**2 + p + 1)) * X**3 + p**6 * (e4 + 2 * (e2 + p**4 + p**2 + 1)) * X**4 - p**9 * (e3 + e1 * (p**2 + p + 1)) * X**5 + p**13 * (e2 + p**4 + 2 * p**2 + 1) * X**6 - p**18 * e1 * X**7 + p**24 * X**8
                return g(p_k * X)
            g = (1 - p**6 * X**2) * (1 - e1 * X + p * (e2 + p**5 + p**4 + 1) * X**2 - p**3 * (e3 + e1 * (p**3 + p**2 - p + 1)) * X**3 + p**7 * (e2 + p**5 + p**4 + 1) * X**4 - p**12 * e1 * X**5 + p**18 * X**6)
            return g(p_k * X)
        return NotImplemented

    def field(self):
        r"""
        Field of definition.
        This is the field generated by self's Hecke eigenvalues.
        """
        return self.__field



## Implement the algorithms of Chapter 5 of Jeffery Hein's thesis

def isotropic_vector(S, V, p):
    r"""
    Compute an isotropic vector in an isotropic quadratic form over Z/pZ.

    INPUT:
    - ``S`` -- a Gram matrix for the quadratic form, defined over ZZ. (I.e. a symmetric integral matrix with even diagonal.)
    - ``V`` -- a set of vectors
    - ``p`` -- a prime

    OUTPUT: a vector v in span(V) that is isotropic, i.e. v*S*v/2 = 0 mod p.

    WARNING: if S is anisotropic then this algorithm will probably never terminate!
    Don't use this code on anisotropic lattices.

    Note: the vector v is random.
    """
    V = V[:3]
    W = matrix(ZZ, V)
    S1 = W * S * W.transpose()
    K = matrix(GF(p), S1).kernel().basis_matrix()
    if p > 2 and K.nrows():
        x = vector(ZZ, K.rows()[0])
        v = x[0] * V[0] + x[1] * V[1] + x[2] * V[2]
        return v
    S00 = S1[0, 0] // 2
    S11 = S1[1, 1] // 2
    if S00 % p == 0:
        return V[0]
    if S11 % p == 0:
        return V[1]
    n = len(V)
    S01 = S1[0, 1]
    if p == 2 and S01 % 2 == 0:
        return V[0] + V[1]
    if n == 2:
        if p == 2:
            raise RuntimeError
        d = S01 * S01 - 4 * S00 * S11
        d = square_root_mod_prime(mod(d, p))
        return V[1] - Integer((d + S01) // (S00 + S00)) * V[0]
    if p == 2:
        e = S1[0, 2] % 2
        g = (e * (S1[1, 2] + 1) + S1[2, 2] // 2) % 2
        return g * V[0] + e * V[1] + V[2]
    D, P = local_normal_form_with_change_vars(S1, p)
    P = P % p
    V0, V1, V2 = matrix(ZZ, P * W).rows()
    if D[0, 0] % p == 0:
        return V0
    if D[1, 1] % p == 0:
        return V1
    if D[2, 2] % p == 0:
        return V2
    while 1:
        a = randrange(1, p)
        b = randrange(p)
        u = -((a * a * D[0, 0] + b * b * D[1, 1]) / D[2, 2]) % p
        if kronecker_symbol(u, p) == 1:
            return vector(ZZ, a * V0 + b * V1 + square_root_mod_prime(mod(u, p)) * V2)
        a = randrange(p)
        b = randrange(1, p)
        u = -((a * a * D[0, 0] + b * b * D[1, 1]) / D[2, 2]) % p
        if kronecker_symbol(u, p) == 1:
            return vector(ZZ, a * V0 + b * V1 + square_root_mod_prime(mod(u, p)) * V2)

def hyperbolic_complement(S, V, p, X = []):
    r"""
    Compute a hyperbolic complement to an isotropic subspace.
    """
    n = len(V)
    if n:
        W = matrix(V).transpose()
        S1 = S * W
        s = S1.columns()[0]
        if X:
            U = matrix(ZZ, matrix(GF(p), matrix(X) * S).transpose().kernel().basis_matrix())
            x = next(x for x in U if x * s % p)
            s = x / (x * s)
        else:
            j, x = next(x for x in enumerate(s) if x[1] % p)
            s = vector(ZZ, [0]*(j) + [Integer(x).inverse_mod(p)] + [0]*(S.nrows() - j - 1))# - V[0] * ( S[j, j] / (2 * x * x) ) % p
        X = X + [s, V[0]]
        if n > 1:
            Z = hyperbolic_complement(S, V[1:], p, X = X)
            s = s % p
            s = s - sum(s * S1.columns()[i] * Z[i - 1] % p for i in range(1, n)) - sum(s * S * Z[i - 1] * V[i] % p for i in range(1, n))
            s = s - V[0] * (s * S * s) / 2
            return [s % p] + Z
        s = s - V[0] * (s * S * s) / 2
        return [s % p]
    return []


def hyperbolic_splitting(S, p, V = None, X = [], j = 0):
    r"""
    Compute a maximal hyperbolic splitting.

    OUTPUT: a list H such that matrix(H) * S * matrix(H).transpose() splits a maximal number of hyperbolic planes over Z/p, together with a number i (the number of hyperbolic planes)

    WARNING: the output is random
    """
    if V is None:
        V = identity_matrix(S.nrows()).rows()
    n = len(V)
    if n:
        if n == 1:
            return [V[0]], 0
        elif n == 2:
            S0 = V[0] * S
            S00 = S0 * V[0] // 2
            S01 = S0 * V[1]
            S11 = V[1] * S * V[1] // 2
            if p == 2:
                if S00 % 2 and S11 % 2:
                    return V[:2], 0
            elif p > 2:
                d = S01 * S01 - 4 * S00 * S11
                if kronecker_symbol(d, p) != 1:
                    return V[:2], 0
        W = matrix(V)
        v1 = isotropic_vector(S, V[:3], p)
        v2, = hyperbolic_complement(S, [v1], p, X = X)
        v2 = (v2 / (v1 * S * v2)) % p
        Sv1 = (W * S) * v1
        Sv2 = (W * S) * v2
        U = matrix(ZZ, matrix(GF(p), [Sv1, Sv2]).transpose().kernel().basis_matrix())
        X = X + [v1, v2]
        H, j = hyperbolic_splitting(S, p, V = (U * W).rows(), X = X, j = j)
        return [v1, v2] + H, j + 1
    return [], 0

def isotropic_subspaces(S, p, k):
    r"""
    Compute all k-dimensional isotropic subspaces in a quadratic form S over Z/pZ.
    """
    H, j = hyperbolic_splitting(S, p)
    H = matrix(ZZ, H)
    S_H = H * S * H.transpose()
    H_t = H.transpose()
    if not j:
        return []
    two_j = j + j
    two_k = k + k
    n = S.nrows() - two_j
    if n:
        Sn = S_H[(-n):, (-n):]
        d = {i:[] for i in range(p)}
        if n == 1:
            Sn = Sn[0, 0] / 2
            for a in range(p):
                d[Sn * a * a % p].append([a])
        elif n == 2:
            Sn00, Sn01 = Sn.rows()[0]
            Sn11 = Sn[1, 1] / 2
            Sn00 /= 2
            for a in range(p):
                for b in range(p):
                    N = a * (a * Sn00 + b * Sn01) + b * b * Sn11
                    d[N % p].append([a, b])
    else:
        d = {i:[] for i in range(p)}
        d[0] = [[]]
    @cached_function
    def subspaces_with_pivot(P):
        n = len(P)
        r = list(range(p))
        P0 = P[0]
        if n == 1:
            v = [0] * (P0 + 2 - (P0 % 2))
            v[P0] = 1
            X = [r for _ in range(2 * floor(P0 / 2) + 2, two_j)]
            isotropic_lines = []
            L = list(product(*X))
            N = Integer(len(X))
            if P0 % 2:
                for x in L:
                    u = sum(-x[i + i] * x[i + i + 1] % p for i in range(N / 2)) % p
                    h = v + list(x)
                    isotropic_lines.extend([h + y for y in d[u]])
            else:
                for a in range(p):
                    v[P0 + 1] = a
                    for x in L:
                        u = (-a + sum(-x[i + i] * x[i + i + 1] % p for i in range(N / 2))) % p
                        h = v + list(x)
                        isotropic_lines.extend([h + y for y in d[u]])
            return [[vector(ZZ, x)] for x in isotropic_lines]
        Q = subspaces_with_pivot(P[:1])
        R = subspaces_with_pivot(P[1:])
        return [x+y for x in Q for y in R if not any(x[0][i] for i in P[1:]) and not any(x[0] * S_H * z % p for z in y)]
    subspaces = []
    for pivot in combinations(range(two_j), k):
        X = subspaces_with_pivot(pivot)
        X = [[H_t * x % p for x in x] for x in X]
        subspaces += X
    subspaces_with_pivot.clear_cache()
    return subspaces

def lift_p_to_psqr(S, p, X, Z):
    r"""
    Lift a hyperbolic pair (X, Z) mod p to a hyperbolic pair (X, Z) mod p^2
    """
    X2 = []
    Z2 = []
    SX = [S * x for x in X]
    SZ = [S * z for z in Z]
    for i, x in enumerate(X):
        z = Z[i]
        y1 = x - (x * SX[i]) * Z[i] / 2
        y2 = z - (z * SZ[i]) * X[i] / 2
        for j in range(i):
            y1 -= (X[j] * SX[i]) * Z[j]
            y2 -= (Z[j] * SZ[i]) * X[j]
        X2.append(y1)
        Z2.append(y2)
    return X2, Z2
    Z3 = [x - p * sum( ZZ(((x * S * X2[j]) - (i == j))/p) % p * Z2[i] for i in range(len(X2))) for j, x in enumerate(Z2)]
    return X2, Z3


def antisymmetric_matrices_mod(n, p):
    r"""
    Iterate through antisymmetric matrices of size (n x n) modulo p
    """
    r = list(range(p))
    X = product(*[r for _ in range(ZZ(n * (n - 1) / 2))])
    for x in X:
        y = matrix(ZZ, n, n)
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                y[i, j] = x[s]
                y[j, i] = -x[s]
                s += 1
        yield y

def orbits(X, G, p):
    r"""
    Split X into G-orbits
    """
    K = GF(p)
    n = len(X[0][0])
    G_reduce = [matrix(K, n, n, g) for g in G]
    Y = []
    Z = []
    for x in X:
        x = matrix(K, x).rref()
        if x not in Z:
            H = []
            N = []
            for i, g in enumerate(G_reduce):
                y = x * g
                y = y.rref()
                if y not in Z:
                    Z.append(y)
                    y = y.lift()
                    H.append((y, G[i]))
            Y.append(H)
    return Y

def pk_neighbors_from_X(S, X, p):
    r"""
    Compute (p^k)-neighbors of Q given a k-dimensional isotropic subspace X
    """
    n = S.nrows()
    Z = hyperbolic_complement(S, X, p)
    X, Z = lift_p_to_psqr(S, p, X, Z)
    X = matrix(ZZ, X)
    Z = matrix(ZZ, Z)
    M = antisymmetric_matrices_mod(X.nrows(), p)
    L = []
    K = GF(p)
    psqr_I = p * p * identity_matrix(n)
    U = matrix(K, X * S).transpose().kernel().basis_matrix().lift()
    V = (p * U).stack(psqr_I)
    for A in M:
        B = (X + p * A * Z).stack(V)
        B = B.hermite_form()[:n, :] / p
        Bt = B.transpose()
        S1 = (B*S*Bt).change_ring(ZZ)
        L.append([S1, Bt.inverse()])
    return L

def pk_neighbors(S, G, p, k):
    r"""
    Compute (p^k)-neighbors with change-of-basis matrices.
    """
    X = isotropic_subspaces(S, p, k)
    K = GF(p)
    n = S.nrows()
    G_tr = [matrix(n, n, g).transpose() for g in G]
    Y = orbits(X, G_tr, p)
    Z = []
    for y in Y:
        x, g = y[0]
        P = pk_neighbors_from_X(S, x.rows(), p)
        Z.extend([(x, b, [y[1] for y in y]) for x, b in P])
    return Z



## miscellaneous functions


def harmonic_projection(p, S):
    r"""
    Compute the harmonic projection h of a homogeneous polynomial p with respect to a quadratic form Q.
    This is the unique harmonic (with respect to Q) polynomial with the property

    p(X) = h(X) + Q(X) * f(X)

    with a polynomial f of degree deg(p)-2.

    INPUT:
    -- "p" - a homogeneous polynomial in dim(S) variables
    -- "S" - a Gram matrix for the quadratic form Q

    OUTPUT: harmonic homogeneous polynomial
    """
    R = p.parent()
    n = S.nrows()
    v = vector(R.gens())
    u = v * S * v / 2
    u_powers = [1]
    D = [R.derivation(x) for x in R.gens()]
    S_inv = S.inverse()
    def laplacian(f):
        v = S_inv * vector([d(f) for d in D])
        return 2 * sum(d(v[i]) for i, d in enumerate(D))
    laplacians = [p]
    d = ZZ(p.degree())
    k = d // 2
    for i in range(k):
        u_powers.append(u_powers[-1] * u)
        laplacians.append(laplacian(laplacians[-1]))
    def double_factorial(n):
        if n == 0 or n == 1:
            return 1
        return n * double_factorial(n - 2)
    f = 0
    for j in srange(k + 1):
        c = (-1)**j * double_factorial(n + 2 * (d - j - 2)) / (double_factorial(2 * j) * double_factorial(n + 2 * d - 4))
        f += c * u_powers[j] * laplacians[j]
    return f


def harmonic_invariant_polynomial_generator(S, G, n, d, spin = 1, det = 1):
    r = PolynomialRing(QQ, ['x_%s'%i for i in range(n)])
    def lists_of_fixed_sum(nvars, N):
        if nvars == 1:
            return [[N]]
        return [[j] + x for j in range(N + 1) for x in lists_of_fixed_sum(nvars - 1, N - j)]
    L = lists_of_fixed_sum(n, d)
    L_dict = {tuple(x):i for i, x in enumerate(L)}
    M = [r.monomial(*x) for x in L]
    rgens = vector(r.gens())
    if spin == 1:
        if det == 1:
            character = lambda y:1
        else:
            character = lambda y: y.determinant()
    else:
        if det == 1:
            character = lambda y: (-1)**len([p for p in spinor_norm(S, y).prime_factors() if spin % p == 0])
        else:
            character = lambda y: y.determinant() * (-1)**len([p for p in spinor_norm(S, y).prime_factors() if spin % p == 0])
    def permutation_matrix_minus_identity(g):
        g = matrix(n, n, g)
        z = g * rgens
        x = []
        chi = character(g)
        for i, m in enumerate(M):
            y = [0] * len(L)
            f = m(*z)
            c = f.coefficients()
            d = f.exponents()
            for j, a in enumerate(d):
                y[L_dict[tuple(a)]] = c[j]
            y[i] -= chi
            x.append(y)
        return matrix(x).transpose()
    G_gens = G.gens()
    u = permutation_matrix_minus_identity(G_gens[0])
    for g in G_gens[1:]:
        u = u.stack(permutation_matrix_minus_identity(g))
    Z = matrix(u).transpose().kernel().basis_matrix()
    for z in Z.rows():
        yield harmonic_projection(r({tuple(x):z[i] for i, x in enumerate(L)}), S)



def invariant_harmonic_projection(f, G, S, spin = 1, det = 1):
    n = S.nrows()
    v = vector(f.parent().gens())
    if spin == 1:
        if det == 1:
            h = sum(f(*(matrix(n, n, y) * v)) for y in G)
        else:
            h = sum(matrix(n, n, y).det() * f(*(matrix(n, n, y) * v)) for y in G)
    else:
        if det == 1:
            character = lambda y: (-1)**len([p for p in spinor_norm(S, matrix(n, n, y)).prime_factors() if spin % p == 0])
        else:
            character = lambda y: matrix(n, n, y).det() * (-1)**len([p for p in spinor_norm(S, matrix(n, n, y)).prime_factors() if spin % p == 0])
        h = sum(f(*(matrix(n, n, y) * v)) * character(y) for y in G)
    return harmonic_projection(h, S)

def monomial_iterator(r, d):
    r"""
    Iterate through monomials in the multivariate polynomial ring r that are homogeneous of degree d.
    """
    def list_with_sum(n, d):
        if n == 1:
            yield [d]
        else:
            for j in range(d):
                for x in list_with_sum(n - 1, d - j):
                    yield [j] + x
            yield [d] + [0]*(n-1)
    n = len(r.gens())
    for x in list_with_sum(n, d):
        yield r.monomial(*x)



def invariant_weight_k_polynomials_with_dim_bound(S, G, k, bound, spin = 1, det = 1):
    current_rank = 0
    current_list = []
    R = PolynomialRing(QQ, ['x_%s'%i for i in range(S.nrows())])
    N = S.nrows()
    dim = binomial(N + k, N)
    if len(G) > 10 * dim: #cutoff where linear algebra might be less awful than the Reynolds projector
        for f in harmonic_invariant_polynomial_generator(S, G, S.nrows(), k, spin = spin, det = det):
            new_list = current_list + [f]
            V = polynomial_relations(new_list)
            if not V.dimension():
                current_list = new_list
            if len(current_list) == bound:
                return current_list
    else:
        for x in monomial_iterator(R, k):
            f = R(invariant_harmonic_projection(x, G, S, spin = spin, det = det))
            new_list = current_list + [f]
            V = polynomial_relations(new_list)
            if not V.dimension():
                current_list = new_list
            if len(current_list) == bound:
                return current_list
    return []



def polynomial_linear_combination(f, X):
    r"""
    Express the polynomial "f" as a linear combination of the list of polynomials "X".
    """
    dicts = [defaultdict(int, x.dict()) for x in ([f]+X)]
    keys = set().union(*dicts)
    M = matrix(QQ, [[d[i] for i in keys] for d in dicts])
    v = M.kernel()
    try:
        v = v.basis_matrix().rows()[0]
        return vector((-v/v[0])[1:])
    except IndexError:
        raise ValueError('No linear combination') from None

def polynomial_relations(X):
    dicts = [defaultdict(int, x.dict()) for x in X]
    keys = set().union(*dicts)
    M = matrix([[d[i] for i in keys] for d in dicts])
    return M.kernel()

def spinor_norm(S, A):
    r"""
    Compute spinor norms.

    INPUT:
    -- "S" - a Gram matrix for the quadratic form
    -- "A" - an element of the orthogonal group O(S)

    OUTPUT: the spinor norm of A (an element of Q^x / (Q^x)^2).
    """
    n = A.nrows()
    I = identity_matrix(n)
    s = Integer(1)
    i = 0
    bound = n + 2
    while 1:
        try:
            i += 1
            v = next(v for v in (A - I).columns() if v)
            n = v * S * v / 2
            v = matrix(v)
            s *= n
            R = I - v.transpose() * v * S / n
            A *= R
            if i > bound:
                raise RuntimeError
        except StopIteration:
            return s.squarefree_part()

def _amf_relations(X):
    r"""
    Linear relations among a set of modular forms.
    """
    if not X:
        return matrix([])
    N = len(X)
    if N == 1:
        try:
            Y = X[0]
            Xref = Y[0]
            X = Y
        except (IndexError, TypeError):
            Xref = X[0]
    else:
        Xref = X[0]
    Y = [x._f() for x in X]
    k = Xref.weight()
    if k > 0:
        V = polynomial_relations([y[0] for y in Y])
        if not V.dimension():
            return V
        for i in range(1, len(Y[0])):
            V = V.intersection(polynomial_relations([y[i] for y in Y]))
            if not V.dimension():
                return V
    else:
        n = Xref.gram_matrix().nrows()
        V = matrix([[y[[0]*n] for y in x] for x in Y]).kernel()
    return V