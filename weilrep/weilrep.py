r"""

Sage code for spaces of vector-valued modular forms

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

import cmath
import math

import cypari2
pari = cypari2.Pari()
PariError = cypari2.PariError

from copy import copy
from itertools import product

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli, divisors, euler_phi, factor, fundamental_discriminant, GCD, is_prime, is_square, kronecker_symbol, moebius, prime_divisors, valuation, XGCD
from sage.arith.srange import srange
from sage.combinat.root_system.cartan_matrix import CartanMatrix
from sage.combinat.root_system.weyl_group import WeylGroup
from sage.functions.bessel import bessel_I, bessel_J
from sage.functions.generalized import sgn
from sage.functions.log import log
from sage.functions.other import ceil, floor, frac, sqrt
from sage.functions.transcendental import zeta
from sage.functions.trig import cos
from sage.matrix.constructor import matrix
from sage.matrix.matrix_space import MatrixSpace
from sage.matrix.special import block_matrix, diagonal_matrix, identity_matrix
from sage.misc.functional import denominator, round, isqrt
from sage.misc.misc_c import prod
from sage.modular.dirichlet import DirichletGroup, kronecker_character
from sage.modular.modform.constructor import CuspForms, ModularForms
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modular.modform.element import is_ModularFormElement
from sage.modular.modform.j_invariant import j_invariant_qexp
from sage.modular.modform.vm_basis import delta_qexp
from sage.modules.free_module import span
from sage.modules.free_module_element import vector
from sage.modules.free_quadratic_module import FreeQuadraticModule
from sage.modules.torsion_quadratic_module import TorsionQuadraticForm
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.all import CC
from sage.rings.big_oh import O
from sage.rings.fast_arith import prime_range
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.monomials import monomials
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RealField, RR

from .eisenstein_series import *
from .mock import WeilRepMockModularForm, WeilRepQuasiModularForm
from .morphisms import WeilRepAutomorphism, WeilRepAutomorphismGroup
from .weilrep_modular_forms_class import smf,  WeilRepModularForm, WeilRepModularFormsBasis

sage_one_half = Integer(1) / Integer(2)
sage_three_half = Integer(3) / Integer(2)
sage_five_half = Integer(5) / Integer(2)
sage_seven_half = Integer(7) / Integer(2)
sage_nine_half = Integer(9) / Integer(2)


class WeilRep(object):
    r"""
    The WeilRep class represents the module of vector-valued modular forms whi ch transform with the dual Weil representation.

    INPUT:

    A WeilRep instance is constructed by calling WeilRep(S), where
    - ``S`` -- a symmetric integral matrix with even diagonal and nonzero determinant (this is not checked), OR
    - ``S`` -- a nondegenerate quadratic form

    OUTPUT: WeilRep
    """

    def __init__(self, S):
        if not S:
            S = matrix([])
        if isinstance(S, CartanMatrix):
            self._cartan = S
        else:
            self._cartan = None
        if isinstance(S.parent(), MatrixSpace):
            S = matrix(ZZ, S)
            self.__gram_matrix = S
            self.__quadratic_form = QuadraticForm(S)
        elif isinstance(S, QuadraticForm):
            self.__quadratic_form = S
            self.__gram_matrix = S.matrix()
        else:
            raise TypeError('Invalid input')
        self.__eisenstein = {}
        self.__cusp_forms_basis = {}
        self.__modular_forms_basis = {}
        self.__vals = {}
        self.__valsm = {}
        if self.is_positive_definite():
            from .positive_definite import WeilRepPositiveDefinite
            self.__class__ = WeilRepPositiveDefinite
        if self.is_lorentzian():
            if S.nrows() > 1:
                self.lift_qexp_representation = None
            else:
                self.lift_qexp_representation = 'shimura'
            from .lorentz import WeilRepLorentzian
            self.__class__ = WeilRepLorentzian

    def __repr__(self):
        #when printed:
        return 'Weil representation associated to the Gram matrix\n%s' % (self.gram_matrix())

    ## rescale, add weilrep's

    def dual(self):
        r"""
        Compute the dual representation.

        This is simply the Weil representation obtained by multiplying the underlying quadratic form by (-1).

        NOTE: w.dual() is the same as w(-1) except that it is cached.

        OUTPUT: a WeilRep instance

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).dual()
            Weil representation associated to the Gram matrix
            [-2 -1]
            [-1 -2]

        """
        try:
            return self.__dual
        except AttributeError:
            w = WeilRep(-self.gram_matrix())
            w._cartan = self._cartan
            self.__dual = w
            return w

    def __call__(self, N):
        r"""
        If ``N`` is an integer, then: Rescale the underlying lattice.
        If ``N`` is a matrix in SL_2(ZZ)
        """
        try:
            N = Integer(N)
            w = WeilRep(N * self.gram_matrix())
            w._cartan = self._cartan
            return w
        except TypeError:
            a, b, c, d = tuple(ZZ(x) for x in N.list())
            if not a*d == 1 + b*c:
                raise ValueError('Matrix is not in SL_2(ZZ)') from None
            if self.signature() % 2:
                return self._evaluate_metaplectic(a, b, c, d)[0]
            return self._evaluate(a, b, c, d)

    def __add__(self, other):
        if isinstance(other, WeilRep):
            return WeilRep(block_diagonal_matrix([self.gram_matrix(), other.gram_matrix()], subdivide = False))
        return NotImplemented

    __radd__ = __add__

    def __hash__(self):
        return hash(tuple(self.gram_matrix().coefficients()))

    ## basic attributes

    def gram_matrix(self):
        return self.__gram_matrix

    def quadratic_form(self):
        return self.__quadratic_form

    def signature(self):
        r"""
        Return the signature of the underlying quadratic form.
        """
        try:
            return self.__signature
        except AttributeError:
            self.__true_signature = self.quadratic_form().signature()
            self.__is_positive_definite = self.__true_signature == self.__gram_matrix.nrows()
            self.__signature = self.__true_signature % 8
            return self.__signature

    def discriminant(self):
        r"""
        Return the discriminant (without sign) of the underlying quadratic form.
        """
        try:
            return self.__discriminant
        except AttributeError:
            try:
                self.__discriminant = Integer(len(self.__ds))
                return self.__discriminant
            except AttributeError:
                self.__discriminant = abs(self.gram_matrix().determinant())
                return self.__discriminant

    def discriminant_form(self):
        r"""
        Return the discriminant form as a Torsion Quadratic Module.
        """
        try:
            return self.__discriminant_form
        except AttributeError:
            self.__discriminant_form = TorsionQuadraticForm(self.gram_matrix().inverse())
            return self.__discriminant_form

    def __eq__(self, other):
        r"""
        Test whether self and other are equal.

        We do not check whether 'other' is actually a WeilRep. (Hopefully this doesn't matter!)
        """
        return self.gram_matrix() == other.gram_matrix()

    def __ne__(self, other):
        return not(self.__eq__(other))

    def is_lorentzian(self):
        r"""
        Test whether self has signature (n, 1) for some n.

        (Note n=0 is allowed.)
        """
        try:
            return self.__is_lorentzian
        except AttributeError:
            signature = self.signature()
            self.__is_lorentzian = self.__true_signature + 2 == self.__gram_matrix.nrows()
            return self.__is_lorentzian

    def is_lorentzian_plus_II(self):
        r"""
        Test whether self is of the form L + II(n) for some Lorentzian WeilRep L.

        Always returns False. When we construct L+II(n) this method is overwritten to sometimes be True.
        """
        return False

    def is_positive_definite(self):
        r"""
        Test whether self has signature (n, 0) for some n.
        """
        try:
            return self.__is_positive_definite
        except AttributeError:
            self.__is_positive_definite = self.__quadratic_form.is_positive_definite()
            return self.__is_positive_definite

    def is_symmetric_weight(self, weight):
        r"""
        Computes whether the given weight is symmetric.

        INPUT:
        - ``weight`` -- a half-integer

        OUTPUT: 1, if all modular forms of this weight are symmetric. 0, if all modular forms of this weight are antisymmetric. None, otherwise.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,4]])).is_symmetric_weight(4)
            0

        """
        i = (Integer(weight + weight) + self.signature()) % 4
        if i == 0:
            return 1
        elif i == 2:
            return 0
        return None

    def _evaluate(self, a, b, c, d): #bad
        A = (a, b, c, d)
        try:
            return self.__vals[A]
        except KeyError:
            exp = cmath.exp
            if A == (0, -1, 1, 0):
                G = self.gram_matrix()
                ds = self.ds()
                M = matrix(CC, len(ds))
                two_pi_i = complex(0.0, 2 * math.pi)
                for i, x in enumerate(ds):
                    x *= G
                    for j, y in enumerate(ds[:(i + 1)]):
                        z = exp(two_pi_i * (x*y).n())
                        M[i, j] = z
                        M[j, i] = z
                X = (exp(self.signature() * two_pi_i / 8) / math.sqrt(len(ds))) * M
                self.__vals[A] = X
                return X
            elif A == (1, 1, 0, 1):
                two_pi_i = complex(0.0, 2 * math.pi)
                X = diagonal_matrix([exp(two_pi_i * x.n()) for x in self.norm_list()])
                self.__vals[A] = X
                return X
            elif A == (1, 0, 0, 1):
                X = identity_matrix(self.discriminant()), lambda x:1
                self.__vals[A] = X
                return X
            elif A == (-1, 0, 0, -1):
                dsdict = self.ds_dict()
                ds = self.ds()
                Z = [dsdict[tuple(map(frac, -x))] for x in ds]
                M = matrix(CC, len(ds))
                f = complex(0.0, 1.0) ** self.signature()
                for i, j in enumerate(Z):
                    M[i, j] = f
                self.__vals[A] = M
                return M
            elif A == (0, 1, -1, 0):
                X1 = self._evaluate(0, -1, 1, 0)
                X2 = self._evaluate(-1, 0, 0, -1)
                X = X1 * X2
                self.__vals[A] = X
                return X
            elif c:
                if abs(a) > abs(c):
                    q, r = Integer(a).quo_rem(c)
                    X1 = self._evaluate(1, 1, 0, 1)
                    X2 = self._evaluate(r, b - q * d, c, d)
                    X = (X1**q)*X2
                    self.__vals[A] = X
                    return X
                else:
                    X1 = self._evaluate(0, -1, 1, 0)
                    X2 = self._evaluate(c, d, -a, -b)
                    X = X1*X2
                    X = X
                    self.__vals[A] = X
                    return X
            elif (a - 1):
                X1 = self._evaluate(-1, 0, 0, -1)
                X2 = self._evaluate(-a, -b, -c, -d)
                X = X1*X2
                self.__vals[A] = X
                return X
            else:
                X1 = self._evaluate(1, 1, 0, 1)
                X = X1**b
                self.__vals[A] = X
                return X

    def _evaluate_metaplectic(self, a, b, c, d): #worse
        A = (a, b, c, d)
        try:
            return self.__valsm[A]
        except KeyError:
            exp = cmath.exp
            I = complex(0.0, 1.0)
            if A == (0, -1, 1, 0):
                G = self.gram_matrix()
                ds = self.ds()
                M = matrix(CC, len(ds))
                two_pi_i = complex(0.0, 2 * math.pi)
                for i, x in enumerate(ds):
                    x *= G
                    for j, y in enumerate(ds[:(i + 1)]):
                        z = exp(two_pi_i * (x*y).n())
                        M[i, j] = z
                        M[j, i] = z
                X = (exp(self.signature() * two_pi_i / 8) / math.sqrt(len(ds))) * M, cmath.sqrt
                self.__valsm[A] = X
                return X
            elif A == (1, 1, 0, 1):
                two_pi_i = complex(0.0, 2 * math.pi)
                X = diagonal_matrix([exp(two_pi_i * x.n()) for x in self.norm_list()]), lambda x: 1.0
                self.__valsm[A] = X
                return X
            elif A == (1, 0, 0, 1):
                X = identity_matrix(self.discriminant()), lambda x: 1.0
                self.__valsm[A] = X
                return X
            elif A == (-1, 0, 0, -1):
                dsdict = self.ds_dict()
                ds = self.ds()
                Z = [dsdict[tuple(map(frac, -x))] for x in ds]
                M = matrix(CC, len(ds))
                f = I ** self.signature()
                for i, j in enumerate(Z):
                    M[i, j] = f
                X = M, lambda x: I
                self.__valsm[A] = X
                return X
            elif A == (0, 1, -1, 0):
                X1, _ = self._evaluate_metaplectic(0, -1, 1, 0)
                X2, _ = self._evaluate_metaplectic(-1, 0, 0, -1)
                X = X1 * X2, cmath.sqrt
                self.__valsm[A] = X
                return X
            elif c:
                if abs(a) > abs(c):
                    q, r = Integer(a).quo_rem(c)
                    X1, _ = self._evaluate_metaplectic(1, 1, 0, 1)
                    X2, f2 = self._evaluate_metaplectic(r, b - q * d, c, d)
                    X = (X1**q)*X2, f2
                    self.__valsm[A] = X
                    return X
                else:
                    X1, f1 = self._evaluate_metaplectic(0, -1, 1, 0)
                    X2, f2 = self._evaluate_metaplectic(c, d, -a, -b)
                    f = lambda x: f1((c * x + d) / (-a * x - b)) * f2(x)
                    X = X1*X2
                    if f(I).real < 0:
                        f = lambda x: -f1((c * x + d) / (-a * x - b)) * f2(x)
                        if self.signature() % 2:
                            X = -X
                    X = X, f
                    self.__valsm[A] = X
                    return X
            elif (a - 1):
                X1, _ = self._evaluate_metaplectic(-1, 0, 0, -1)
                X2, f2 = self._evaluate_metaplectic(-a, -b, -c, -d)
                X = X1*X2
                if f2(I).imag > 0:
                    f = lambda x: -I * f2(x)
                    if self.signature() % 2:
                        X = -X
                else:
                    f = lambda x: I * f2(x)
                X = X, f
                self.__valsm[A] = X
                return X
            else:
                X1, f = self._evaluate_metaplectic(1, 1, 0, 1)
                X = X1**b, f
                self.__valsm[A] = X
                return X

    ## methods for the discriminant form

    def coefficient_vector_exponents(self, prec, symm, starting_from = 0, include_vectors = False):
        r"""
        Interpret the ``coefficient vectors`` of modular forms for this Weil representation.

        INPUT:
        - ``prec`` -- the precision
        - ``symm`` -- 0 if we work with anti-symmetric modular forms and 1 otherwise
        - ``starting_from`` -- valuation of the modular form (default 0)
        - ``include_vectors`` -- a boolean (default False)

        OUTPUT: if include_vectors = False then this outputs a list of exponents [n_0, n_1, ...] such that if a modular form of this representation has ``coefficient vector`` (c_0, c_1, ...) (see weilrep_modular_forms_class.sage) then its Fourier expansion takes the form c_0 q^(n_0) e_(g_0) + c_1 q^(n_1) e_(g_1) + ... If include_vectors = True then we output the list of lists [[g_0,n_0], [g_1,n_1]...] instead.

        NOTE: we only take one representative from each pair {g, -g}. If symm = 0 then we also exclude g for which g = -g.

        NOTE: if include_vectors = True then the vectors are given as tuples!

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).coefficient_vector_exponents(3,1)
            [0, 2/3, 1, 5/3, 2, 8/3]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).coefficient_vector_exponents(3,0)
            [2/3, 5/3, 8/3]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,4]])).coefficient_vector_exponents(3,1,include_vectors = True)
            [[(0, 0), 0], [(5/7, 4/7), 3/7], [(4/7, 6/7), 5/7], [(1/7, 5/7), 6/7], [(0, 0), 1], [(5/7, 4/7), 10/7], [(4/7, 6/7), 12/7], [(1/7, 5/7), 13/7], [(0, 0), 2], [(5/7, 4/7), 17/7],   [(4/7, 6/7), 19/7], [(1/7, 5/7), 20/7]]

        """
        if starting_from == 0:
            try:
                old_prec = self.__coefficient_vector_prec
                if old_prec == prec:
                    return [self.__coefficient_vector_exponents[symm], self.__coefficient_vector_exponents_including_vectors[symm]][include_vectors]
            except AttributeError:
                pass
            finally:
                pass
        G = self.sorted_rds()
        n_dict = self.norm_dict()
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        prec = ceil(prec)
        for n in range(floor(starting_from),prec+1):
            for g in G:
                true_n = n_dict[g] + n
                if starting_from <= true_n < prec:
                    X1.append(true_n)
                    Y1.append([g,true_n])
                    if 2 % vector(g).denominator():
                        X2.append(true_n)
                        Y2.append([g,true_n])
        if starting_from == 0:
            self.__coefficient_vector_prec = prec
            self.__coefficient_vector_exponents = [X2, X1]
            self.__coefficient_vector_exponents_including_vectors = [Y2,Y1]
        return [[X2,X1][symm], [Y2,Y1][symm]][include_vectors]

    def ds(self):
        r"""
        Compute representatives of the discriminant group of the underlying quadratic form.

        OUTPUT: a list of vectors which represent the cosets of S^(-1)*Z^N modulo Z^N, where S is the Gram matrix.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).ds()
            [(0, 0), (2/3, 2/3), (1/3, 1/3)]
            """
        try:
            return self.__ds
        except AttributeError:
            if not self.gram_matrix():
                self.__ds = [vector([])]
            else:
                D, _, V = self.gram_matrix().smith_form()
                self._smith_form = D, V
                L = [vector(range(n)) / n for n in D.diagonal()]
                L = (matrix(product(*L)) * V.transpose()).rows()
                self.__ds = [vector(map(frac, x)) for x in L]
            return self.__ds

    def ds_dict(self):
        r"""
        Compute the discriminant group of the underlying quadratic form as a dictionary.

        OUTPUT: a dictionary whose keys are the elements of the discriminant group (as tuples) and whose values are their index in self.ds()

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,4]])).ds_dict() == {(2/7, 3/7): 4, (5/7, 4/7): 3, (0, 0): 0, (1/7, 5/7): 2, (3/7, 1/7): 6, (4/7, 6/7): 1, (6/7, 2/7): 5}
            True

        """
        try:
            return self.__ds_dict
        except AttributeError:
            try:
                _ds = [tuple(g) for g in self.__ds]
                self.__ds_dict = dict(zip(_ds, range(len(_ds))))
                return self.__ds_dict
            except AttributeError:
                D, U, V = self.gram_matrix().smith_form()
                L = [vector(range(D[k, k])) / D[k, k] for k in range(D.nrows())]
                ds = [None] * prod(D.diagonal())
                ds_dict = {}
                for i, r in enumerate(product( *L )):
                    frac_r = list(map(frac, V * vector(r)))
                    ds[i] = vector(frac_r)
                    ds_dict[tuple(frac_r)] = i
                self.__ds = ds
                self.__ds_dict = ds_dict
                return self.__ds_dict

    def embiggen(self, b, m):
        r"""
        Construct a WeilRep for a bigger matrix.

        Given a vector b \in L' and a rational number m \in ZZ - Q(b), this computes the WeilRep attached to the integral quadratic form

        \tilde Q(x, \lambda) = Q(x + \lambda b) + m \lambda^2

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[-12]]))
            sage: w.embiggen(vector([1/12]), 1/24)
            Weil representation associated to the Gram matrix
            [-12  -1]
            [ -1   0]
        """
        S = self.gram_matrix()
        tilde_b = b*S
        shift_m = m + b*tilde_b/2
        tilde_b = matrix(tilde_b)
        return WeilRep(block_matrix(ZZ,[[S, tilde_b.transpose()], [tilde_b, 2*shift_m]], subdivide = False))

    def level(self):
        r"""
        Return self's level.
        """
        try:
            return self.__level
        except:
            self.__level = self.quadratic_form().level()
            return self.__level

    def norm_dict(self):
        r"""
        Compute the values of the quadratic form Q on the discriminant group as a dictionary.

        OUTPUT: a dictionary whose keys are the elements of the discriminant group (as tuples) and whose values are the *minus* norms -Q(x) in QQ/ZZ (represented by a rational number: -1 < -Q(x) <= 0)

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).norm_dict() == {(2/3, 2/3): -1/3, (0, 0): 0, (1/3, 1/3): -1/3}
            True

        """
        try:
            return self.__norm_dict
        except:
            _ds = self.ds()
            S = self.gram_matrix()
            self.__norm_dict = {tuple(g):-frac(g*S*g/2) for g in _ds}
            return self.__norm_dict

    def norm_list(self):
        r"""
        Compute the values of the quadratic form Q on the discriminant group as a list.

        OUTPUT: a list whose values are the *minus* norms -Q(x) in QQ/ZZ (represented by a rational number: -1 < -Q(x) <= 0) where x runs through the result of self.ds()

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).norm_list()
            [0, -1/3, -1/3]

        """
        try:
            return self.__norm_list
        except:
            _ds = self.ds()
            S = self.gram_matrix()
            x = [-frac(g*S*g/2) for g in _ds]
            self.__norm_list = x
            self.__norm_dict = dict(zip(map(tuple, _ds), x)) #create a norm_dict while we're here
            return x

    def rds(self, indices = False):
        r"""
        Reduce the representatives of the discriminant group modulo equivalence g ~ -g

        OUTPUT:
        - If indices = False then output a sublist of discriminant_group(S) containing exactly one element from each pair {g,-g}. 
        - If indices = True then output a list X of indices defined as follows. Let ds = self.ds(). Then: X[j] = i if j > i and ds[j] = -ds[i] mod Z^N, and X[j] = None if no such i exists.

        NOTE: as we run through the discriminant group we also store the denominators of the vectors as a list

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).rds()
            [(0, 0), (2/3, 2/3)]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).rds(indices = True)
            [None, None, 1]

        """
        try:
            return [self.__rds, self.__rds_indices][indices]
        except AttributeError:
            if not self.gram_matrix():
                self.__rds = [vector([])]
                self.__rds_indices = [None]
                self.__ds_denominators_list = [1]
                self.__rds_denominators_list = [1]
                self.__order_two_in_ds_list = [1]
                self.__order_two_in_rds_list = [1]
            else:
                L = []
                set_L = set(L)
                Gdict = self.ds_dict()
                G = self.ds()
                len_G = len(G)
                X = [None]*len_G
                X2 = [None]*len_G
                Y = [0]*len_G
                Z = []
                order_two_ds = [0]*len_G
                order_two_rds = []
                for i, g in enumerate(G):
                    u = vector(map(frac, -g))
                    dg = denominator(g)
                    Y[i] = dg
                    order_two_ds[i] = not(2 % dg)
                    tu = tuple(u)
                    if tu in set_L:
                        X[i] = Gdict[tu]
                    else:
                        tg = tuple(g)
                        L.append(g)
                        set_L.add(tg)
                        Z.append(dg)
                        order_two_rds.append(order_two_ds[i])
                self.__rds = L
                self.__rds_indices = X
                self.__ds_denominators_list = Y
                self.__rds_denominators_list = Z
                self.__order_two_in_ds_list = order_two_ds
                self.__order_two_in_rds_list = order_two_rds
            return [self.__rds, self.__rds_indices][indices]

    def sorted_rds(self):
        r"""
        Computes a copy of the reduced discriminant group self.rds(), sorted by the norm dictionary norm_dict().

        OUTPUT: a list of *tuples*

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,4]])).sorted_rds()
            [(5/7, 4/7), (4/7, 6/7), (1/7, 5/7), (0, 0)]
        """
        S = self.gram_matrix()
        try:
            return self.__sorted_rds
        except AttributeError:
            n_dict = self.norm_dict()
            G = list(map(tuple, self.rds()))
            G.sort(key = lambda g: n_dict[g])
            self.__sorted_rds = G
            return G


    ## constructors of modular forms for this representation. See also weilrep_modular_forms_class.sage

    def bb_lift(self, mf):
        r"""
        Construct vector-valued modular forms of prime level via the Bruinier--Bundschuh lift.

        NOTE: this works *only* when self has odd prime discriminant. (In particular the lattice rank must be even.)

        INPUT:
        - ``mf`` -- a modular form of level equal to self's discriminant, with the quadratic character and lying in the correct plus/minus space

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[2,1],[1,2]]))
            sage: mf = ModularForms(Gamma1(3), 3, prec = 20).basis()[0]
            sage: w.bb_lift(mf)
            [(0, 0), 1 + 72*q + 270*q^2 + 720*q^3 + 936*q^4 + 2160*q^5 + O(q^6)]
            [(2/3, 2/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + O(q^(20/3))]
            [(1/3, 1/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + O(q^(20/3))]

        """
        if not is_ModularFormElement(mf):
            raise TypeError('The Bruinier-Bundschuh lift takes modular forms as input')
        p = mf.level()
        if not p.is_prime() and p != 2 and self.discriminant() == p:
            raise TypeError('The Bruinier-Bundschuh lift takes modular forms of odd prime level as input')
        if not mf.character() == DirichletGroup(p)[(p - 1) // 2]:
            raise TypeError('Invalid character')
        mfq = mf.qexp()
        R, q = mfq.parent().objgen()
        mf_coeffs = mfq.padded_list()
        prec = len(mf_coeffs) // p
        ds = self.ds()
        norm_list = self.norm_list()
        Y = [None]*len(ds)
        for i, g in enumerate(ds):
            offset = norm_list[i]
            if not g:
                f = R(mf_coeffs[::p]) + O(q ** prec)
                Y[i] = g, offset, f
                mfq -= f.V(p)
            else:
                u = q * R(mf_coeffs[p + Integer(p*offset) :: p])/2 + O(q**(prec+1))
                Y[i] = g, offset, u
                mfq -= u.V(p) * q ** Integer(p * offset)
        if mfq:
            raise ValueError('This modular form does not lie in the correct plus/minus subspace')
        return WeilRepModularForm(mf.weight(), self.gram_matrix(), Y, weilrep = self)

    def eisenstein_series(self, k, prec, allow_small_weight = False, components = None):
        r"""
        Constuct Eisenstein series attached to the vector e_0.

        This constructs the Eisenstein series E_(k,0) of weight k and constant term e_0 with Fourier expansion up to precision `prec`.

        INPUT:
        - ``k`` -- a weight (half-integer, and such that 2k + signature = 0 mod 4). also ``k`` can be a list of weights (then we produce a list of Eisenstein series).
        - ``prec`` -- precision
        - ``allow_small_weight`` -- a boolean (default False). If True then we compute the Eisenstein series in weights less than or equal to 2 (where it may not be a true modular form)
        - ``components`` -- optional parameter (default None). A sublist L of self's discriminant group and a list of indices (e.g. [None]*len(L) ) can be passed here as a tuple.

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).eisenstein_series(3,20)
            [(0, 0), 1 + 72*q + 270*q^2 + 720*q^3 + 936*q^4 + 2160*q^5 + 2214*q^6 + 3600*q^7 + 4590*q^8 + 6552*q^9 + 5184*q^10 + 10800*q^11 + 9360*q^12 + 12240*q^13 + 13500*q^14 + 17712*q^15 + 14760*q^16 + 25920*q^17 + 19710*q^18 + 26064*q^19 + O(q^20)]
            [(2/3, 2/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + 2808*q^(20/3) + 4752*q^(23/3) + 4590*q^(26/3) + 7560*q^(29/3) + 7371*q^(32/3) + 10800*q^(35/3) + 9774*q^(38/3) + 15120*q^(41/3) + 14040*q^(44/3) + 19872*q^(47/3) + 16227*q^(50/3) + 25272*q^(53/3) + 22950*q^(56/3) + 31320*q^(59/3) + O(q^(62/3))]
            [(1/3, 1/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + 2808*q^(20/3) + 4752*q^(23/3) + 4590*q^(26/3) + 7560*q^(29/3) + 7371*q^(32/3) + 10800*q^(35/3) + 9774*q^(38/3) + 15120*q^(41/3) + 14040*q^(44/3) + 19872*q^(47/3) + 16227*q^(50/3) + 25272*q^(53/3) + 22950*q^(56/3) + 31320*q^(59/3) + O(q^(62/3))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,0],[0,6]])).eisenstein_series(5,5)
            [(0, 0), 1 - 1280/11*q - 20910/11*q^2 - 104960/11*q^3 - 329040/11*q^4 + O(q^5)]
            [(0, 1/6), -915/11*q^(11/12) - 1590*q^(23/12) - 93678/11*q^(35/12) - 304980/11*q^(47/12) - 757335/11*q^(59/12) + O(q^(71/12))]
            [(0, 1/3), -255/11*q^(2/3) - 9984/11*q^(5/3) - 65775/11*q^(8/3) - 234240/11*q^(11/3) - 612510/11*q^(14/3) + O(q^(17/3))]
            [(0, 1/2), -5/11*q^(1/4) - 3198/11*q^(5/4) - 33215/11*q^(9/4) - 142810/11*q^(13/4) - 428040/11*q^(17/4) + O(q^(21/4))]
            [(0, 2/3), -255/11*q^(2/3) - 9984/11*q^(5/3) - 65775/11*q^(8/3) - 234240/11*q^(11/3) - 612510/11*q^(14/3) + O(q^(17/3))]
            [(0, 5/6), -915/11*q^(11/12) - 1590*q^(23/12) - 93678/11*q^(35/12) - 304980/11*q^(47/12) - 757335/11*q^(59/12) + O(q^(71/12))]
            [(1/2, 0), -410/11*q^(3/4) - 12010/11*q^(7/4) - 75030/11*q^(11/4) - 255918/11*q^(15/4) - 651610/11*q^(19/4) + O(q^(23/4))]
            [(1/2, 1/6), -240/11*q^(2/3) - 10608/11*q^(5/3) - 61440/11*q^(8/3) - 248880/11*q^(11/3) - 576480/11*q^(14/3) + O(q^(17/3))]
            [(1/2, 1/3), -39/11*q^(5/12) - 5220/11*q^(17/12) - 44205/11*q^(29/12) - 176610/11*q^(41/12) - 493155/11*q^(53/12) + O(q^(65/12))]
            [(1/2, 1/2), -1360/11*q - 19680/11*q^2 - 111520/11*q^3 - 307200/11*q^4 + O(q^5)]
            [(1/2, 2/3), -39/11*q^(5/12) - 5220/11*q^(17/12) - 44205/11*q^(29/12) - 176610/11*q^(41/12) - 493155/11*q^(53/12) + O(q^(65/12))]
            [(1/2, 5/6), -240/11*q^(2/3) - 10608/11*q^(5/3) - 61440/11*q^(8/3) - 248880/11*q^(11/3) - 576480/11*q^(14/3) + O(q^(17/3))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[0,0,2],[0,-4,0],[2,0,0]])).eisenstein_series(5/2,5)
            [(0, 0, 0), 1 - 8*q - 102*q^2 - 48*q^3 - 184*q^4 + O(q^5)]
            [(0, 3/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^(41/8))]
            [(0, 1/2, 0), -14*q^(1/2) - 16*q^(3/2) - 80*q^(5/2) - 64*q^(7/2) - 350*q^(9/2) + O(q^(11/2))]
            [(0, 1/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^(41/8))]
            [(0, 0, 1/2), -16*q - 64*q^2 - 96*q^3 - 128*q^4 + O(q^5)]
            [(0, 3/4, 1/2), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^(41/8))]
            [(0, 1/2, 1/2), -8*q^(1/2) - 32*q^(3/2) - 64*q^(5/2) - 128*q^(7/2) - 200*q^(9/2) + O(q^(11/2))]
            [(0, 1/4, 1/2), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^(41/8))]
            [(1/2, 0, 0), -16*q - 64*q^2 - 96*q^3 - 128*q^4 + O(q^5)]
            [(1/2, 3/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^(41/8))]
            [(1/2, 1/2, 0), -8*q^(1/2) - 32*q^(3/2) - 64*q^(5/2) - 128*q^(7/2) - 200*q^(9/2) + O(q^(11/2))]
            [(1/2, 1/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^(41/8))]
            [(1/2, 0, 1/2), -8*q^(1/2) - 32*q^(3/2) - 64*q^(5/2) - 128*q^(7/2) - 200*q^(9/2) + O(q^(11/2))]
            [(1/2, 3/4, 1/2), -8*q^(5/8) - 40*q^(13/8) - 80*q^(21/8) - 120*q^(29/8) - 200*q^(37/8) + O(q^(45/8))]
            [(1/2, 1/2, 1/2), -16*q - 64*q^2 - 96*q^3 - 128*q^4 + O(q^5)]
            [(1/2, 1/4, 1/2), -8*q^(5/8) - 40*q^(13/8) - 80*q^(21/8) - 120*q^(29/8) - 200*q^(37/8) + O(q^(45/8))]

            sage: WeilRep(matrix([])).eisenstein_series(2, 10, allow_small_weight = True)
            [(), 1 - 24*q - 72*q^2 - 96*q^3 - 168*q^4 - 144*q^5 - 288*q^6 - 192*q^7 - 360*q^8 - 312*q^9 + O(q^10)]

        """
        #check input
        prec = ceil(prec)
        k_is_list = type(k) is list
        if not k_is_list:
            if prec <= 0:
                raise ValueError('Precision must be at least 0')
            if not self.is_symmetric_weight(k):
                raise ValueError('Invalid weight in Eisenstein series')
            try:#did we do this already?
                try:
                    old_prec, e = self.__eisenstein[k]
                except ValueError:
                    old_prec = -1
                if old_prec >= prec:
                    return e.reduce_precision(prec, in_place = False)
                raise RuntimeError
            except (KeyError, RuntimeError):
                pass
            if 1 < k <= 2 and not allow_small_weight: #weight 3/2, 2
                if k == 2:
                    s = self.eisenstein_series_shadow(prec)
                    if s:
                        self.__eisenstein[k] = WeilRepQuasiModularForm(k, self.gram_matrix(), [-12 * s / sqrt(self.discriminant()), self.eisenstein_series(k, prec, allow_small_weight = True)], weilrep = self)
                    else:
                        self.__eisenstein[k] = self.eisenstein_series(k, prec, allow_small_weight = True)
                else:
                    s = self.eisenstein_series_shadow_wt_three_half(prec)
                    if s:
                        self.__eisenstein[k] = WeilRepMockModularForm(k, self.gram_matrix(), self.eisenstein_series(k, prec, allow_small_weight = True).fourier_expansion(), s, weilrep = self)
                    else:
                        self.__eisenstein[k] = self.eisenstein_series(k, prec, allow_small_weight = True)
                return self.__eisenstein[k]
        elif not k:
            return []
        #shortcut if unimodular
        dets = self.discriminant()
        S = self.gram_matrix()
        if dets == 1:
            X = WeilRepModularForm(k, self.gram_matrix(), [(vector([0] * S.nrows()), 0, eisenstein_series_qexp(k, prec, normalization = 'constant'))], weilrep=self)
            self.__eisenstein[k] = X
            return X
        #setup
        try:
            k = Integer(k)
        except TypeError:
            pass
        R, q = PowerSeriesRing(QQ, 'q').objgen()
        if components:
            _ds, _indices = components
            _norm_list = [-frac(g*S*g/2) for g in _ds]
        else:
            _ds = self.ds()
            _indices = self.rds(indices = True)
            _norm_list = self.norm_list()
        eps = (-1) ** (self.signature() in range(3, 7))
        S_rows_gcds = list(map(GCD, S.rows()))
        S_rows_sums = sum(S)
        level = self.level()
        L_half_s = (matrix(_ds) * S).rows() #fix?
        _dim = S.nrows()
        precomputed_lists = {}
        dets_primes = dets.prime_divisors()
        X = [None] * len(_ds)
        if k_is_list:
            len_k = len(k)
            X = [copy(X) for _ in range(len_k)]
        #guess which Lvalues we have to look at. (this is always enough but sometimes its too many)
        def eisenstein_series_create_lists(g):
            d_gamma = denominator(g)
            d_gamma_squared = d_gamma * d_gamma
            old_modulus = 2 * d_gamma_squared * dets
            mod_value = old_modulus * _norm
            gcd_mm = GCD(old_modulus, mod_value)
            modulus = Integer(old_modulus / gcd_mm)
            mod_value = Integer(mod_value / gcd_mm)
            m = mod_value + modulus * modified_prec
            prime_list_1 = prime_range(2, m)
            prime_list_1.extend([p for p in dets_primes if p >= m])
            little_n_list = [mod_value / modulus + j for j in range(1, modified_prec)]
            prime_list = []
            n_lists = []
            removable_primes = []
            for p in prime_list_1:
                if level % p == 0:
                    if p != 2 and any(L_half[i] % p and not S_rows_gcds[i] % p for i in range(_dim)):# and (p != 2 or any(L_half[i]%2 and not S[i,i]%4 for i in range(_dim))):#???
                        removable_primes.append(p)
                    else:
                        prime_list.append(p)
                        n_lists.append([little_n_list,list(range(modified_prec - 1)),[],[]])
                else:
                    p_sqr = p * p
                    index_list_p = []
                    index_list_p_ii = []
                    n_list_p = []
                    n_list_p_ii = []
                    mod_value_mod_p = (modulus.inverse_mod(p_sqr) * mod_value) % p_sqr
                    i0 = p_sqr - mod_value_mod_p
                    i = p - (mod_value_mod_p % p)
                    while 1:#find exponents divisible by p
                        try:
                            N = little_n_list[i-1]
                            if (i-i0) % p_sqr:
                                n_list_p_ii.append(N)
                                index_list_p_ii.append(i-1)
                            else:
                                n_list_p.append(N)
                                index_list_p.append(i-1)
                            i += p
                        except:
                            break
                    if n_list_p or n_list_p_ii:
                        prime_list.append(p)
                        n_lists.append([n_list_p, index_list_p, n_list_p_ii, index_list_p_ii])
            return [little_n_list, n_lists, old_modulus, prime_list, removable_primes]
        #odd rank:
        if _dim % 2:
            if k_is_list:
                k_shift_list = [Integer(j - sage_one_half) for j in k]
                two_k_shift_list = [j + j for j in k_shift_list]
                front_multiplier_list = [eps / zeta(1 - j) for j in two_k_shift_list]
                k_shift = k_shift_list[0]
            else:
                k_shift = Integer(k - sage_one_half)
                two_k_shift = k_shift + k_shift
                front_multiplier = eps / zeta(1 - two_k_shift)
            for i_g, g in enumerate(_ds):
                _norm = _norm_list[i_g]
                if _indices[i_g] is None: #have we computed the negative component yet?
                    L_half = L_half_s[i_g]
                    L = L_half + L_half
                    modified_prec = prec - floor(_norm)
                    little_n_list, n_lists, old_modulus, prime_list, removable_primes = eisenstein_series_create_lists(g)
                    gSg = g * S * g
                    if k_is_list:
                        t, = PolynomialRing(QQ, 't').gens()
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, 0, t=t) for i_p, p in enumerate(prime_list)]
                    else:
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, k) for i_p, p in enumerate(prime_list)]
                    try:#some things depend only on the exponents not the component-vector "g"
                        D_list, main_term_list = precomputed_lists[_norm]
                    except:
                        if k_is_list:
                            main_term_list = [[0] for _ in range(len_k)]
                            D_list = [0]
                            for i_n, n in enumerate(little_n_list):
                                D = ((-1) ** k_shift) * old_modulus
                                for p, e in factor(n):
                                    if p==2 or (dets % p == 0):
                                        D *= (p ** e)
                                    else:
                                        D *= (p ** (e % 2))
                                little_D = abs(fundamental_discriminant(D))
                                sqrt_factor = sqrt(2 * n  * dets / little_D)
                                correct_L_function_list = [quadratic_L_function__corrector(k_shift, D) * quadratic_L_function__cached(1 - k_shift, D) for k_shift in k_shift_list]
                                D_list.append(D)
                                for j, k_shift in enumerate(k_shift_list):
                                    main_term_list[j].append(correct_L_function_list[j] * ((4 * n / little_D) ** k_shift) / sqrt_factor)
                        else:
                            main_term_list = [0]
                            D_list = [0]
                            for i_n, n in enumerate(little_n_list):
                                D = ((-1) ** k_shift) * old_modulus
                                for p, e in factor(n):
                                    if p==2 or (dets % p == 0):
                                        D *= (p ** e)
                                    else:
                                        D *= (p ** (e % 2))
                                little_D = abs(fundamental_discriminant(D))
                                sqrt_factor = sqrt(2 * n  * dets / little_D)
                                correct_L_function = quadratic_L_function__corrector(k_shift, D) * quadratic_L_function__cached(1 - k_shift, D)
                                D_list.append(D)
                                main_term_list.append(correct_L_function * ((4 * n / little_D) ** k_shift) / sqrt_factor)
                    local_factor_list = [1] * (modified_prec)
                    if k_is_list:
                        local_factor_list = [copy(local_factor_list) for _ in range(len_k)]
                        for i, p in enumerate(prime_list):
                            p = prime_list[i]
                            p_e_power = p ** ((1 + _dim) // 2)
                            for index_k, k_shift in enumerate(k_shift_list):
                                p_k_shift = p ** (-k_shift)
                                p_k_shift_squared = p_k_shift * p_k_shift
                                p1_mult = 1 / (1 + p_k_shift)
                                p2_mult = (1 + p_k_shift) / (1 - p_k_shift_squared)
                                p_pow = p_k_shift * p_e_power
                                p_tuple = (p2_mult, ~(1 - p_k_shift_squared), p1_mult)
                                for j in range(len(n_lists[i][0])):
                                    index_n = n_lists[i][1][j] + 1
                                    D = D_list[index_n]
                                    local_factor_list[index_k][index_n] *= Lvalue_list[i][j](p_pow) * p_tuple[kronecker_symbol(D, p) + 1]
                    else:
                        for i, p in enumerate(prime_list):
                            p = prime_list[i]
                            p_k_shift = p ** (-k_shift)
                            p_k_shift_squared = p_k_shift * p_k_shift
                            p1_mult = 1 / (1 + p_k_shift)
                            p2_mult = (1 + p_k_shift) / (1 - p_k_shift_squared)
                            p_tuple = (p2_mult, ~(1 - p_k_shift_squared), p1_mult)
                            for j in range(len(n_lists[i][0])):
                                index_n = n_lists[i][1][j] + 1
                                D = D_list[index_n]
                                local_factor_list[index_n] *= Lvalue_list[i][j] * p_tuple[kronecker_symbol(D, p) + 1]
                    E = (old_modulus == dets + dets) + O(q ** modified_prec)
                    if k_is_list:
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, E + front_multiplier_list[i] * (prod(1 / (1 - p ** (-two_k_shift_list[i])) for p in removable_primes)) * R([local_factor_list[i][j] * main_term_list[i][j] for j in range(modified_prec)])
                    else:
                        if (k == 1) and gSg == 0:
                            E = E - weight_one_zero_val(g,S)
                        X[i_g] = g, _norm, E + front_multiplier * (prod(1 / (1 - p ** (-two_k_shift)) for p in removable_primes)) * R([local_factor_list[j] * main_term_list[j] for j in range(modified_prec)])
                    precomputed_lists[_norm] = D_list, main_term_list
                else:
                    if k_is_list:
                        ind_g = _indices[i_g]
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, X[i][ind_g][2]
                    else:
                        X[i_g] = g, _norm, X[_indices[i_g]][2]
            if k_is_list:
                return [WeilRepModularForm(k[i], S, X[i], weilrep = self) for i in range(len_k)]
            else:
                e = WeilRepModularForm(k, S, X, weilrep = self)
                self.__eisenstein[k] = prec, e
                return e
        #even rank
        else:
            if k_is_list:
                D = ((-1) ** k[0]) * dets
                littleD = fundamental_discriminant(D)
                sqrt_factor = QQ(2 / isqrt(abs(littleD * dets)))
                multiplier_list = [eps * (littleD ** k_i) * sqrt_factor / (quadratic_L_function__corrector(k_i, D) * quadratic_L_function__cached(1 - k_i, littleD)) for i, k_i in enumerate(k)]
            else:
                D = ((-1) ** k) * dets
                littleD = fundamental_discriminant(D)
                corrector = 1 / quadratic_L_function__corrector(k, D)
                sqrt_factor = QQ(2/isqrt(abs(littleD * dets)))
                multiplier = QQ(eps * corrector * (littleD ** k) * sqrt_factor / quadratic_L_function__cached(1 - k, littleD))
            for i_g, g in enumerate(_ds):
                _norm = _norm_list[i_g]
                modified_prec = prec - floor(_norm)
                if _indices[i_g] is None: #have we computed the negative component yet?
                    L_half = L_half_s[i_g]
                    L = L_half + L_half
                    _norm = _norm_list[i_g]
                    little_n_list, n_lists, old_modulus, prime_list, removable_primes = eisenstein_series_create_lists(g)
                    gSg = g * S * g
                    if k_is_list:
                        t, = PolynomialRing(QQ, 't').gens()
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, 0, t=t) for i_p, p in enumerate(prime_list)]
                    else:
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, k) for i_p, p in enumerate(prime_list)]
                    try:
                        main_term_list = precomputed_lists[_norm]#some things depend only on the exponents not the component-vector "g"
                    except:
                        if k_is_list:
                            main_term_list = [[0] for _ in range(len_k)]
                            for i, k_i in enumerate(k):
                                main_term_list[i].extend([n ** (k_i - 1) for n in little_n_list])
                        else:
                            main_term_list = [0]
                            main_term_list.extend([n ** (k-1) for n in little_n_list])
                    local_factor_list = [1] * (modified_prec)
                    if k_is_list:
                        local_factor_list = [copy(local_factor_list) for _ in range(len_k)]
                        for i, p in enumerate(prime_list):
                            kron = kronecker_symbol(D, p)
                            p_pow_e = p ** (1 + _dim//2)
                            for index_k, k_i in enumerate(k):
                                p_k = p ** (-k_i)
                                kron_p = kron * p_k
                                p_pow = p_pow_e * p_k
                                p_factor = 1 + (p * kron_p)
                                quot = 1 / (1 - kron_p)
                                for j in range(len(n_lists[i][0])):
                                    index_n = n_lists[i][1][j] + 1
                                    local_factor_list[index_k][index_n] *= (Lvalue_list[i][j](p_pow) * quot)
                                for j in range(len(n_lists[i][2])):
                                    index_n = n_lists[i][3][j] + 1
                                    local_factor_list[index_k][index_n] *= p_factor
                    else:
                        local_factor_list = [1] * (modified_prec)
                        for i, p in enumerate(prime_list):
                            kron_p = kronecker_symbol(D, p) * (p ** (-k))
                            p_factor = 1 + (p * kron_p)
                            quot = 1 / (1 - kron_p)
                            for j in range(len(n_lists[i][0])):
                                index_n = n_lists[i][1][j] + 1
                                local_factor_list[index_n] *= (Lvalue_list[i][j] * quot)
                            for j in range(len(n_lists[i][2])):#p is bad but not too bad at N=local_factor_list[index_n]
                                index_n = n_lists[i][3][j] + 1
                                local_factor_list[index_n] *= p_factor
                    E = (old_modulus == dets + dets) + O(q ** modified_prec)
                    if k_is_list:
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, E + multiplier_list[i] * R([local_factor_list[i][j] * main_term_list[i][j] for j in range(modified_prec)])
                    else:
                        if (k == 1) and _norm == 0:
                            E = E - weight_one_zero_val(g,S)
                        X[i_g] = g, _norm, E + multiplier * R([local_factor_list[j] * main_term_list[j] for j in range(modified_prec)])
                    precomputed_lists[_norm] = main_term_list
                else:
                    if k_is_list:
                        index = _indices[i_g]
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, X[i][index][2]
                    else:
                        X[i_g] = g, _norm, X[_indices[i_g]][2]
            if k_is_list:
                return [WeilRepModularForm(k[i], S, X[i], weilrep = self) for i in range(len_k)]
            else:
                e = WeilRepModularForm(k, S, X, weilrep = self)
                self.__eisenstein[k] = prec, e
                return e

    def eisenstein_newform(self, k, b, prec, allow_small_weight = False, print_exact = False):
        ## WARNING!! This is not fully tested and very likely has bugs!! it is also slow! ##

        r"""
        Compute Eisenstein newforms.

        This computes the sum:
        \sum_{\chi} E_{k, \beta, \chi}
        over all primitive Dirichlet characters \chi modulo the denominator of \beta.

        ALGORITHM: We use the formula of Theorem 1.4 of [Sch] for the Eisenstein series with character E_{k, \beta, \chi}. Their Fourier coefficients are computed numerically. We recover the (rational) power series using known bounds on the denominators of the Fourier coefficients.

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- a vector in self's discriminant group
        - ``prec`` -- the precision
        - ``allow_small_weight`` -- boolean (default False); if True then we do not check whether the weight is > 2
        - ``print_exact`` -- boolean (default False); if True then print some debugging information

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[18]])).eisenstein_newform(5/2, vector([1/3]), 5)
            [(0), O(q^5)]
            [(1/18), 72*q^(35/36) + 120*q^(71/36) + 396*q^(107/36) + 384*q^(143/36) + 804*q^(179/36) + O(q^(215/36))]
            [(1/9), -54*q^(8/9) - 156*q^(17/9) - 276*q^(26/9) - 504*q^(35/9) - 660*q^(44/9) + O(q^(53/9))]
            [(1/6), -4*q^(3/4) + 24*q^(7/4) - 36*q^(11/4) - 48*q^(15/4) + 168*q^(19/4) + O(q^(23/4))]
            [(2/9), 24*q^(5/9) + 108*q^(14/9) + 264*q^(23/9) + 438*q^(32/9) + 540*q^(41/9) + O(q^(50/9))]
            [(5/18), -12*q^(11/36) - 72*q^(47/36) - 276*q^(83/36) - 264*q^(119/36) - 672*q^(155/36) + O(q^(191/36))]
            [(1/3), 2 - 12*q + 18*q^2 + 28*q^3 - 108*q^4 + O(q^5)]
            [(7/18), 24*q^(23/36) + 156*q^(59/36) + 192*q^(95/36) + 516*q^(131/36) + 480*q^(167/36) + O(q^(203/36))]
            [(4/9), -6*q^(2/9) - 84*q^(11/9) - 216*q^(20/9) - 312*q^(29/9) - 528*q^(38/9) + O(q^(47/9))]
            [(1/2), O(q^(23/4))]
            [(5/9), 6*q^(2/9) + 84*q^(11/9) + 216*q^(20/9) + 312*q^(29/9) + 528*q^(38/9) + O(q^(47/9))]
            [(11/18), -24*q^(23/36) - 156*q^(59/36) - 192*q^(95/36) - 516*q^(131/36) - 480*q^(167/36) + O(q^(203/36))]
            [(2/3), -2 + 12*q - 18*q^2 - 28*q^3 + 108*q^4 + O(q^5)]
            [(13/18), 12*q^(11/36) + 72*q^(47/36) + 276*q^(83/36) + 264*q^(119/36) + 672*q^(155/36) + O(q^(191/36))]
            [(7/9), -24*q^(5/9) - 108*q^(14/9) - 264*q^(23/9) - 438*q^(32/9) - 540*q^(41/9) + O(q^(50/9))]
            [(5/6), 4*q^(3/4) - 24*q^(7/4) + 36*q^(11/4) + 48*q^(15/4) - 168*q^(19/4) + O(q^(23/4))]
            [(8/9), 54*q^(8/9) + 156*q^(17/9) + 276*q^(26/9) + 504*q^(35/9) + 660*q^(44/9) + O(q^(53/9))]
            [(17/18), -72*q^(35/36) - 120*q^(71/36) - 396*q^(107/36) - 384*q^(143/36) - 804*q^(179/36) + O(q^(215/36))]
        """

        def l_value(k, psi0):
            psi = psi0.primitive_character()
            f_psi = psi.conductor()
            l_value_bar = CC(-psi.bar().bernoulli(k) / k)
            delta = psi.is_odd()
            L_val = l_value_bar * (2 * RRpi / f_psi)**k * CC(psi.gauss_sum_numerical(prec = lazy_bound)) / (2 * CC(1j)**delta * RR(cos(math.pi * (k - delta) / 2)) * RR(k).gamma())
            for p in prime_divisors(psi0.modulus() / f_psi):
                L_val = L_val * (1 - psi(p) / (p **k))
            return L_val
        def lazy_l_value(g, n, S, p, k, u = None):
            L = 2 * g * S
            c = g * S * g + 2*n
            return L_values(2 * g * S, [c], S, p, k, t = u)[0]
        #check input
        prec = ceil(prec)
        if prec <= 0:
            raise ValueError('Precision must be at least 0')
        if k <= 2 and not allow_small_weight and (k < 2 or self.discriminant().is_squarefree()):
            raise ValueError('Weight must be at least 5/2')
        symm = self.is_symmetric_weight(k)
        if symm is None:
            raise ValueError('Invalid weight in Eisenstein series')
        #setup
        S = self.gram_matrix()
        e = S.nrows()
        if e % 2:
            denom = lcm([2*k-1, self.discriminant(), max(bernoulli(Integer(2*k - 1)).numerator(), 1)])
            for p in self.discriminant().prime_divisors():
                denom = lcm(denom, p ** (Integer(2*k - 1)) - 1)
        else:
            denom = lcm([k, self.discriminant(), max(bernoulli(k).numerator(), 1)])
            for p in self.discriminant().prime_divisors():
                denom = lcm(denom, p ** k - 1)
        if print_exact:
            print('denominator: ', denom)
        lazy_bound = max(4 * ceil(log(prec + 2) * (k - 1) + log(denom) + log(60) + log(k+1)), 53)
        RR = RealField(lazy_bound)
        CC = RR.complex_field()
        N_b = denominator(b)
        N_b_factor = factor(N_b)
        chi_list = [chi for chi in DirichletGroup(N_b) if chi.is_primitive() and chi.is_even() == symm]
        if not chi_list:
            raise ValueError('Not a newform')
        chi_decompositions = [chi.decomposition() for chi in chi_list]
        chi_gauss_sums = [chi.gauss_sum_numerical(prec = lazy_bound) for chi in chi_list]
        ds = self.ds()
        indices = self.rds(indices = True)
        discr = self.discriminant()
        sqrt_discr = sqrt(RR(discr))
        RRpi = RR.pi()
        RRgamma_k = RR(k).gamma()
        first_factor = (RR(2) ** (k+1)) * RRpi**k * CC(1j) ** (k + self.signature() / 2) / (sqrt_discr * RRgamma_k)
        norm_list = self.norm_list()
        RPoly, t = PolynomialRing(QQ, 't').objgen()
        X = []
        R, q = PowerSeriesRing(QQ, 'q').objgen()
        for i, g in enumerate(ds):
            if indices[i] is None:
                offset = norm_list[i]
                prec_g = prec - floor(offset)
                coeff_list = []
                for n_ceil in range(1, prec_g):
                    n = n_ceil + offset
                    N_bSg = Integer(N_b * (b * S * g))
                    gcd_b_gb = GCD(N_b, N_bSg)
                    N_g = prod([p ** d for (p, d) in factor(N_b) if gcd_b_gb % p == 0])
                    N_g_prime = N_b // N_g
                    D_g = DirichletGroup(N_g)
                    D_g_prime = DirichletGroup(N_g_prime)
                    chi_g_list = [prod([D_g(psi) for psi in chi_decompositions[i] if N_g % psi.modulus() == 0]) for i in range(len(chi_list))]
                    L_s = [[D_g_prime(psi) for psi in chi_decompositions[i] if N_g % psi.modulus()] for i in range(len(chi_list))]
                    chi_g_prime_list = [prod(L) if L else lambda x:1 for L in L_s]
                    front_factor = first_factor * RR(n) ** (k-1)
                    eps_factors = [chi_gauss_sums[i] * chi_g_prime(N_bSg)**(-1) / N_g for i, chi_g_prime in enumerate(chi_g_prime_list)]
                    D = Integer(2 * N_g * N_g * n * S.determinant())
                    bad_primes = (D).prime_divisors()
                    if e % 2 == 1:
                        D0 = fundamental_discriminant(D * (-1)**((e+1)/2))
                    else:
                        D0 = fundamental_discriminant((-1)**(e/2) * S.determinant())
                    main_terms = [RR(1)]*len(chi_list)
                    chi0 = kronecker_character(D0)
                    for p in bad_primes:
                        main_term_L_val = lazy_l_value(g, n, S, p, k, u = t)
                        p_power = p ** Integer(1 + e/2 - k)
                        Euler_factors = [CC(main_term_L_val(chi(p)*p_power)) if chi(p) else 1 for chi in chi_list]
                        chi0_p = CC(chi0(p))
                        if e % 2:
                            p_pow_2 = RR(p ** Integer(1 / 2 - k))
                            for i in range(len(chi_list)):
                                chi_p = CC(chi_list[i](p))
                                main_terms[i] *= ( (1 - chi_p * chi0_p * p_pow_2) * Euler_factors[i] / (1 - (chi_p * p_pow_2) ** 2))#.n()
                        else:
                            for i in range(len(chi_list)):
                                main_terms[i] *= Euler_factors[i] / (1 - chi_list[i](p) * chi0_p * (p ** (-k)))
                    if e % 2:
                        for i, chi in enumerate(chi_list):
                            G = DirichletGroup(lcm(chi.modulus(), chi0.modulus()))
                            main_terms[i] *= CC(l_value(Integer(k - sage_one_half), G(chi)*G(chi0)))
                            main_terms[i] /= CC(l_value(Integer(2*k - 1), chi * chi))
                    else:
                        for i, chi in enumerate(chi_list):
                            G = DirichletGroup(lcm(chi.modulus(), chi0.modulus()))
                            main_terms[i] /= CC(l_value(Integer(k), G(chi)*G(chi0)))
                    finite_parts = [1 for _ in chi_list]
                    for p in prime_divisors(gcd_b_gb):
                        p_sum = 0
                        p_power_N_b = p ** (N_b.valuation(p))
                        vp_g = gcd_b_gb.valuation(p)
                        w_p = 1 + 2 * (2*N_b * N_g * n).valuation(p)
                        N_g_over_p = N_g / (p ** N_g.valuation(p))
                        Dp_prime = DirichletGroup(N_g_over_p)
                        Dp = DirichletGroup(p ** N_g.valuation(p))
                        L_s = [[Dp_prime(psi) for psi in chi_decomposition if not N_g_over_p % psi.modulus()] for chi_decomposition in chi_decompositions]
                        chi_p_prime_list = [prod(L) if L else lambda x:1 for L in L_s]
                        chi_p_list = [prod([Dp(psi) for psi in chi_decomposition if not psi.modulus() % p]) for chi_decomposition in chi_decompositions]
                        s = vector([0] * len(chi_list))
                        for alpha in range(w_p + 1):
                            p_alpha = p ** alpha
                            p_pow_list = [CC(chi_p_prime(p_alpha)) * (p_alpha ** (1 - e/2 - k)) for chi_p_prime in chi_p_prime_list]
                            p_e_alpha = (p_alpha ** e) / p
                            s_alpha = vector([0]*len(chi_list))
                            for v in range(p_power_N_b):
                                new_g = g - v * (N_b // p_power_N_b) * b
                                for u in range(p_power_N_b):
                                    if u % p:
                                        LvalueSeries = RPoly(lazy_l_value(new_g, n + p_alpha * v * u / p_power_N_b , S, p, k, u = t))
                                        s_alpha += CC(p_e_alpha * (p * LvalueSeries[alpha] - LvalueSeries[alpha - 1])) * vector(CC(chi_p.bar()(u)) for chi_p in chi_p_list)
                            s = s + vector(p_pow_list[i] * s_alpha[i] for i in range(len(p_pow_list)))
                        for i in range(len(finite_parts)):
                            finite_parts[i] *= s[i]
                    coeff = (front_factor * sum(eps_factors[i] * main_terms[i] * finite_parts[i] for i in range(len(chi_list))))
                    coeff_numerator = RR(coeff.real()) * denom
                    if print_exact:
                        print(g, n, 'exact: ', coeff_numerator, front_factor.n(), eps_factors[0].n(), main_terms[0].n(), finite_parts[0].n())
                    coeff = round(coeff_numerator) / denom
                    coeff_list.append(coeff)
                f = q * R(coeff_list) + O(q ** prec_g)
                const_term = Integer(0)
                for n in range(N_b):
                    if denominator(g - n * b) == 1:
                        const_term = const_term + 2 * sum(chi(n).n() for chi in chi_list)
                const_term = round(const_term.real() * denom) / denom
                f = f + const_term
                X.append([g, offset, f])
            else:
                eps = 1 if symm else -1
                X.append([g, norm_list[indices[i]], eps * X[indices[i]][2]])
        return WeilRepModularForm(k, S, X, weilrep = self)

    def eisenstein_oldform(self, k, b, prec, allow_small_weight = False):
        r"""
        Compute certain Eisenstein oldforms.

        This computes the sum over E_{k, \beta} where \beta runs through all multiples of ``b`` in self's discriminant group.

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- an element of self's discriminant group with integral norm
        - ``prec`` -- precision

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[8]])).eisenstein_oldform(7/2, vector([1/2]), 5)
            [(0), 1 + 126*q + 756*q^2 + 2072*q^3 + 4158*q^4 + O(q^5)]
            [(1/8), O(q^(95/16))]
            [(1/4), 56*q^(3/4) + 576*q^(7/4) + 1512*q^(11/4) + 4032*q^(15/4) + 5544*q^(19/4) + O(q^(23/4))]
            [(3/8), O(q^(87/16))]
            [(1/2), 1 + 126*q + 756*q^2 + 2072*q^3 + 4158*q^4 + O(q^5)]
            [(5/8), O(q^(87/16))]
            [(3/4), 56*q^(3/4) + 576*q^(7/4) + 1512*q^(11/4) + 4032*q^(15/4) + 5544*q^(19/4) + O(q^(23/4))]
            [(7/8), O(q^(95/16))]
        """
        eps = 1 if self.is_symmetric_weight(k) else -1
        E = self.eisenstein_series(k, prec, allow_small_weight = allow_small_weight)
        d_b = denominator(b)
        if d_b == 1:
            return E
        q, = PowerSeriesRing(QQ, 'q').gens()
        S = self.gram_matrix()
        X = E.components()
        ds = self.ds()
        indices = self.rds(indices = True)
        norm_list = self.norm_list()
        Y = [None] * len(ds)
        for i, g in enumerate(ds):
            if indices[i] is None:
                g_b = frac(g * S * b)
                if g_b:
                    Y[i] = g, norm_list[i], O(q ** (prec - floor(norm_list[i])))
                else:
                    f = sum(X[tuple(map(frac, g + j * b))] for j in range(d_b))
                    Y[i] = g, norm_list[i], f
            else:
                Y[i] = g, norm_list[i], eps*Y[indices[i]][2]
        return WeilRepModularForm(k, S, Y, weilrep = self)

    def eisenstein_series_shadow(self, prec):
        r"""
        Compute the shadow of the weight two Eisenstein series (if it exists).

        INPUT:
        - ``prec`` -- precision

        OUTPUT: a WeilRepModularForm of weight 0

        NOTE: the precision is irrelevant because modular forms of weight 0 are constant.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,0],[0,-2]])).eisenstein_series_shadow(5)
            [(0, 0), 1 + O(q^5)]
            [(1/2, 0), O(q^(23/4))]
            [(0, 1/2), O(q^(21/4))]
            [(1/2, 1/2), 1 + O(q^5)]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[8,0],[0,-2]])).eisenstein_series_shadow(5)
            [(0, 0), 1 + O(q^5)]
            [(1/8, 0), O(q^(95/16))]
            [(1/4, 0), O(q^(23/4))]
            [(3/8, 0), O(q^(87/16))]
            [(1/2, 0), 1 + O(q^5)]
            [(5/8, 0), O(q^(87/16))]
            [(3/4, 0), O(q^(23/4))]
            [(7/8, 0), O(q^(95/16))]
            [(0, 1/2), O(q^(21/4))]
            [(1/8, 1/2), O(q^(83/16))]
            [(1/4, 1/2), 1 + O(q^5)]
            [(3/8, 1/2), O(q^(91/16))]
            [(1/2, 1/2), O(q^(21/4))]
            [(5/8, 1/2), O(q^(91/16))]
            [(3/4, 1/2), 1 + O(q^5)]
            [(7/8, 1/2), O(q^(83/16))]

        """
        prec = ceil(prec)
        if not self.is_symmetric_weight(0):
            raise NotImplementedError
        _ds = self.ds()
        _indices = self.rds(indices = True)
        _dets = self.discriminant()
        n_list = self.norm_list()
        q,= PowerSeriesRing(QQ, 'q').gens()
        o_q_prec, o_q_prec_plus_one = O(q**prec), O(q**(prec + 1))
        try:
            A_sqrt = Integer(sqrt(_dets))
        except:
            return self.zero(prec)
        S = self.gram_matrix()
        _nrows = S.nrows()
        bad_primes = (2*_dets).prime_divisors()
        X = [None] * len(_ds)
        t, = PolynomialRing(QQ, 't').gens()
        for i, g in enumerate(_ds):
            L =  g * S
            c = L * g
            L = L + L
            if _indices[i]:
                x = X[_indices[i]]
                X[i] = g, x[1], x[2]
            else:
                offset = n_list[i]
                if offset:
                    X[i] = g, offset, o_q_prec_plus_one
                else:
                    X[i] = g, 0, o_q_prec + prod(L_values(L, [-c], -S, p, 2)[0] / (1 + 1/p) for p in bad_primes)
        return WeilRepModularForm(0, S, X, weilrep = self)

    def eisenstein_series_shadow_wt_three_half(self, prec):
        r"""
        Compute the shadow of the weight 3/2 Eisenstein series (if it exists).

        INPUT:
        - ``prec`` -- precision

        OUTPUT: a WeilRepModularForm of weight 1/2, which is sqrt(|det S| / 2) * the shadow of the weight 3/2 Eisenstein series for the weilrep 'self'. (This factor makes the result have rational Fourier coefficients.) This transforms with the dual representation (its WeilRep is self.dual())
        """
        prec = ceil(prec)
        s = self.signature()
        if not self.is_symmetric_weight(sage_three_half):
            raise NotImplementedError
        dets = self.discriminant()
        wdual = self(-1)
        ds = wdual.ds()
        indices = wdual.rds(indices = True)
        n_list = wdual.norm_list()
        q, = PowerSeriesRing(QQ, 'q').gens()
        o_q_prec, o_q_prec_plus_one = O(q**prec), O(q**(prec + 1))
        S = wdual.gram_matrix()
        X = [None] * len(ds)
        for i, g in enumerate(ds):
            if indices[i]:
                x = X[indices[i]]
                X[i] = g, x[1], x[2]
            else:
                dg = denominator(g)
                dg *= dg
                L =  g * S
                c, L = L * g, L + L
                offset = n_list[i]
                if offset:
                    f = o_q_prec_plus_one
                else:
                    f = o_q_prec + prod(L_values(L, [c], S, p, sage_three_half)[0] / (1 + ~p) for p in prime_divisors(dets))
                for n in range(1, prec - floor(offset)):
                    _n = n + offset
                    if is_square(2 * _n * dets):
                        f += (q ** n) * 2 * prod(L_values(L, [_n + _n + c], S, p, sage_three_half)[0] / (1 + ~p) for p in prime_divisors(dets * _n * dg))
                X[i] = g, offset, 24 * f
        return WeilRepModularForm(sage_one_half, S, X, weilrep = wdual)

    def poincare_series(self, k, b, m, prec, nterms = 10, _flag = None):
        r"""
        Compute Poincare series.

        This computes a numerical approximation to the Poincare series of weight k and index (b, m) up to precision 'prec'.

        INPUT:

        - ``k`` -- the weight
        - ``b`` -- a vector; typically an element of self.ds()
        - ``m`` -- the index: a rational number such that q^m e_b is an exponent in self's Fourier expansion
        - ``prec`` -- the precision
        - ``nterms`` -- integer (default 10). We compute the coefficient formula (a series for c=1 to infinity over Bessel functions and Kloosterman sums) for c=1 to nterms.
        """
        s = self.is_symmetric_weight(k)
        eps = -1
        if s is None or k < 2 or (k == 2 and not m):
            raise ValueError('Invalid weight.')
        elif s:
            eps = 1
        dsdict = self.ds_dict()
        ds = self.ds()
        rds = self.rds(indices = True)
        j = dsdict[tuple(map(frac, b))]
        j1 = dsdict[tuple(map(frac, -b))]
        nl = self.norm_list()
        if _flag or nl[j] + m not in ZZ:
            raise ValueError('Invalid index.')
        two_pi_i, four_pi = complex(0.0, 2 * math.pi), 4 * math.pi
        e = cmath.exp
        if m > 0:
            J = lambda x: bessel_J(k - 1, x)
        elif m < 0:
            J = lambda x: bessel_I(k - 1, x)
        else:
            gamma_k = math.gamma(k)
        r, q = PowerSeriesRing(RR, 'q').objgen()
        X = [vector(RR, [0]*(prec - floor(u))) for u in nl]
        s1 = e(-two_pi_i * k / 4)
        exponent = k - 1
        #if _flag == 'mock':
        #    exponent *= -1
        for c in range(1, nterms):
            two_pi_i_c = two_pi_i / c
            four_pi_c = four_pi / c
            Y = [vector(RR, [0]*len(x)) for x in X]
            for d in range(c):
                g, a, b = XGCD(c, d)
                if g == 1:
                    M = self._evaluate_metaplectic(b, -a, c, d)[0]
                    zeta1, zeta2 = e(two_pi_i_c * m * b) , e(two_pi_i_c * d)
                    for i, g in enumerate(ds):
                        if rds[i] is None:
                            u = nl[i] + 1
                            zeta = s1 * zeta1 * e(two_pi_i_c * u * d)
                            for n in range(1, len(Y[i])):
                                Y[i][n] += (M[j, i].conjugate() * zeta).real()
                                zeta *= zeta2
            for i, y in enumerate(Y):
                if rds[i] is None:
                    u = nl[i]
                    for n in range(1, len(Y[i])):
                        if m:
                            X[i][n] += 2 * math.pi * math.sqrt((n + u) / abs(m))**exponent * Y[i][n] * J(four_pi_c * math.sqrt(abs(m) * (n + u))) / c
                        else:
                            X[i][n] += (four_pi_c * (n + u) / 2.0)**k  * Y[i][n] / (gamma_k * (n + u))
        for i, x in enumerate(X):
            if rds[i]:
                X[i] = [eps*x for x in X[rds[i]]]
        X = [[ds[i], nl[i],  r(x).add_bigoh(ceil(prec - nl[i]))] for i, x in enumerate(X)]
        X[j][2] += 0.5 * q**ceil(m)
        X[j1][2] += 0.5 * q**ceil(m)
        return WeilRepModularForm(k, self.gram_matrix(), X, weilrep = self)


    def pss(self, weight, b, m, prec, weilrep = None, fix = True):
        r"""
        Compute Poincare square series.

        These are obtained by theta-contraction of Eisenstein series attached to other lattices.

        INPUT:
        - ``weight`` -- the weight (a half-integer) which is at least 5/2
        - ``b`` -- a vector for which b*S is integral (where S is our Gram matrix)
        - ``m`` -- a rational number for which m + b*S*b/2 is a positive integer
        - ``prec`` -- precision (natural number)
        - ``weilrep`` -- WeilRep (optional) the result of self.embiggen(b, m) if known
        - ``fix`` -- boolean (default True) if false then do not fix the result in weight 2 or 5/2

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,0],[0,-2]])).pss(6,vector([1/2,1/2]),1,5)
            [(0, 0), 1 - 8008/31*q - 8184*q^2 - 1935520/31*q^3 - 262392*q^4 + O(q^5)]
            [(1/2, 0), -1918/31*q^(3/4) - 130460/31*q^(7/4) - 1246938/31*q^(11/4) - 5912724/31*q^(15/4) - 19187894/31*q^(19/4) + O(q^(23/4))]
            [(0, 1/2), -11/62*q^(1/4) - 24105/31*q^(5/4) - 919487/62*q^(9/4) - 2878469/31*q^(13/4) - 11002563/31*q^(17/4) + O(q^(21/4))]
            [(1/2, 1/2), -7616/31*q - 8448*q^2 - 1876736/31*q^3 - 270336*q^4 + O(q^5)]

        """
        if weight < 2:
            raise NotImplementedError
        S = self.gram_matrix()
        if not weilrep:
            if S:
                tilde_b = b*S
                shift_m = m + b*tilde_b/2
                tilde_b = matrix(tilde_b)
                S_new = block_matrix(ZZ,[[S,tilde_b.transpose()],[tilde_b,2*shift_m]])
            else:
                S_new = matrix(ZZ, [[2*m]])
            w = WeilRep(S_new)
        else:
            w = weilrep
        new_k = weight - sage_one_half
        _components = [self.ds(), self.rds(indices = True)]
        X = w.eisenstein_series(new_k, prec, allow_small_weight = True).theta_contraction(components = _components)
        if weight > sage_five_half or not fix:
            return X
        elif weight == sage_five_half:#result might be wrong so lets fix it
            dets = w.discriminant()
            try:
                epsilon = QQ(Integer(24) * (-1)**((1 + self.signature())/4) / sqrt(abs(dets))) #maybe we will need this number
            except TypeError:
                return X #result was ok
            q, = PowerSeriesRing(QQ, 'q').gens()
            theta = w.eisenstein_series_shadow(prec+1).theta_contraction(components = _components).fourier_expansion()
            Y = X.fourier_expansion()
            Z = [None] * len(theta)
            for i in range(len(theta)):
                offset = theta[i][1]
                theta_f = list(theta[i][2])
                Z[i] = Y[i][0], Y[i][1], Y[i][2] - epsilon * sum((n + offset) * theta_f[n] * (q ** n) for n in range(1, len(theta_f)) if theta_f[n])
            return WeilRepModularForm(weight, S, Z, weilrep = self)
        elif weight == 2:
            from .weilrep_misc import weight_two_pss_fix
            return X + weight_two_pss_fix(self, b, m, prec, w)


    def pssd(self, weight, b, m, prec, weilrep = None, fix = True):
        r"""
        Compute antisymmetric modular forms.

        These are obtained by theta-contraction of Eisenstein series attached to other lattices.

        INPUT:
        - ``weight`` -- the weight (a half-integer) which is at least 7/2
        - ``b`` -- a vector for which b*S is integral (where S is our Gram matrix)
        - ``m`` -- a rational number for which m + b*S*b/2 is a positive integer
        - ``prec`` -- precision (natural number)
        - ``weilrep`` -- WeilRep (default None) should be the result of self.embiggen(b, m)
        - ``fix`` -- boolean (default True) if false then we do not fix the result in weights 3 or 7/2

        NOTE: if b has order 2 in our discriminant group then this is zero!

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[0,0,3],[0,-2,0],[3,0,0]])).pssd(7/2, vector([0,1/2,1/3]), 1/4, 5)
            [(0, 0, 0), O(q^5)]
            [(0, 1/2, 1/3), 1/2*q^(1/4) - 3*q^(5/4) + 9/2*q^(9/4) + 6*q^(13/4) - 21*q^(17/4) + O(q^(21/4))]
            [(0, 0, 2/3), q - 6*q^2 + 9*q^3 + 10*q^4 + O(q^5)]
            [(0, 1/2, 0), O(q^(21/4))]
            [(0, 0, 1/3), -q + 6*q^2 - 9*q^3 - 10*q^4 + O(q^5)]
            [(0, 1/2, 2/3), -1/2*q^(1/4) + 3*q^(5/4) - 9/2*q^(9/4) - 6*q^(13/4) + 21*q^(17/4) + O(q^(21/4))]
            [(1/3, 0, 0), q - 6*q^2 + 9*q^3 + 10*q^4 + O(q^5)]
            [(1/3, 1/2, 1/3), O(q^(71/12))]
            [(1/3, 0, 2/3), q^(1/3) - 6*q^(4/3) + 10*q^(7/3) + 4*q^(10/3) - 20*q^(13/3) + O(q^(16/3))]
            [(1/3, 1/2, 0), -1/2*q^(1/4) + 3*q^(5/4) - 9/2*q^(9/4) - 6*q^(13/4) + 21*q^(17/4) + O(q^(21/4))]
            [(1/3, 0, 1/3), O(q^(17/3))]
            [(1/3, 1/2, 2/3), -q^(7/12) + 5*q^(19/12) - 3*q^(31/12) - 19*q^(43/12) + 20*q^(55/12) + O(q^(67/12))]
            [(2/3, 0, 0), -q + 6*q^2 - 9*q^3 - 10*q^4 + O(q^5)]
            [(2/3, 1/2, 1/3), q^(7/12) - 5*q^(19/12) + 3*q^(31/12) + 19*q^(43/12) - 20*q^(55/12) + O(q^(67/12))]
            [(2/3, 0, 2/3), O(q^(17/3))]
            [(2/3, 1/2, 0), 1/2*q^(1/4) - 3*q^(5/4) + 9/2*q^(9/4) + 6*q^(13/4) - 21*q^(17/4) + O(q^(21/4))]
            [(2/3, 0, 1/3), -q^(1/3) + 6*q^(4/3) - 10*q^(7/3) - 4*q^(10/3) + 20*q^(13/3) + O(q^(16/3))]
            [(2/3, 1/2, 2/3), O(q^(71/12))]

        """
        if weight < 3:
            raise NotImplementedError
        if not weilrep:
            S = self.gram_matrix()
            if S:
                tilde_b = b*S
                shift_m = m + b*tilde_b/2
                tilde_b = matrix(tilde_b)
                S_new = block_matrix(ZZ,[[S,tilde_b.transpose()],[tilde_b,2*shift_m]])
            else:
                S_new = matrix(ZZ, [[2*m]])
            w = WeilRep(S_new)
        else:
            w = weilrep
        new_k = weight - sage_three_half
        X = w.eisenstein_series(new_k, prec, allow_small_weight = True).theta_contraction(odd = True, weilrep = self)
        if weight > sage_seven_half:
            return X
        elif weight == sage_seven_half:
            try:
                epsilon = QQ(Integer(8) * (-1)**((1 + self.signature())/4) / sqrt(w.discriminant()))
            except TypeError:
                return X
            q, = PowerSeriesRing(QQ, 'q').gens()
            theta = w.eisenstein_series_shadow(prec+1).theta_contraction(odd = True).fourier_expansion()
            Y = X.fourier_expansion()
            Z = [None] * len(theta)
            for i in range(len(theta)):
                offset = theta[i][1]
                theta_f = list(theta[i][2])
                Z[i] = Y[i][0], Y[i][1], Y[i][2] - epsilon * sum((n + offset) * theta_f[n] * (q ** n) for n in range(1, len(theta_f)) if theta_f[n])
            return WeilRepModularForm(weight, self.gram_matrix(), Z, weilrep = self)
        elif weight == 3:
            from .weilrep_misc import weight_three_pssd_fix
            return X + weight_three_pssd_fix(self, b, m, prec, w)

    def pss_double(self, weight, b, m, prec):#to be used when weight is even and >= 3??
        r"""
        Compute the double theta-contraction of an Eisenstein series attached to a lattice of self's rank + 2

        (Usually this is slower.)

        INPUT:
        - ``weight`` -- the weight (a half-integer) which is at least 3
        - ``b`` -- a vector for which b*S is integral (where S is our Gram matrix) OR a matrix of size (2 x S.nrows())
        - ``m`` -- a rational number for which m + b*S*b/2 is a positive integer OR a symmetric rational matrix of size (2x2) for which 2*m + b*S*b.transpose() is a positive-definite integral matrix with even diagonal
        - ``prec`` -- precision (natural number)

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[4]])).pss_double(7/2, vector([1/2]), 1/2, 5)
            [(0), 1 + 84*q + 574*q^2 + 1288*q^3 + 3444*q^4 + O(q^5)]
            [(1/4), 64*q^(7/8) + 448*q^(15/8) + 1344*q^(23/8) + 2688*q^(31/8) + 4928*q^(39/8) + O(q^(47/8))]
            [(1/2), 14*q^(1/2) + 280*q^(3/2) + 840*q^(5/2) + 2368*q^(7/2) + 3542*q^(9/2) + O(q^(11/2))]
            [(3/4), 64*q^(7/8) + 448*q^(15/8) + 1344*q^(23/8) + 2688*q^(31/8) + 4928*q^(39/8) + O(q^(47/8))]
        """
        if weight < 3:
            raise NotImplementedError
        S = self.gram_matrix()
        try:
            _is_vector = b.is_vector() #???
            tilde_b = b*S
            shift_m = m + b*tilde_b/2
            tilde_b = matrix(tilde_b)
            zero_v = matrix([0] * S.nrows())
            S_new = block_matrix(ZZ,[[S, tilde_b.transpose(), zero_v.transpose()],[tilde_b, 2*shift_m, 0],[zero_v, 0, 2]])
        except AttributeError:
            tilde_b = b*S
            two_shift_m = 2*m + tilde_b * b.transpose()
            S_new = block_matrix(ZZ, [[S, tilde_b.transpose()], [tilde_b, two_shift_m]])
        new_k = weight - 1
        w = WeilRep(S_new)
        _components = [self.ds(), self.rds(indices = True)]
        X = w.eisenstein_series(weight - 1, prec, allow_small_weight = True).theta_contraction().theta_contraction(components = _components)
        if weight > 3:
            return X
        else:
            raise NotImplementedError #to be fixed
            q, = PowerSeriesRing(QQ, 'q').gens()
            theta = w.eisenstein_series_shadow(prec+1).theta_contraction().theta_contraction(components = _components).fourier_expansion()
            Y = X.fourier_expansion()
            Z = [None] * len(theta)
            for i in range(len(theta)):
                offset = theta[i][1]
                theta_f = list(theta[i][2])
                Z[i] = Y[i][0], Y[i][1], Y[i][2] - epsilon * sum((n + offset) * theta_f[n] * (q ** n) for n in range(1, len(theta_f)) if theta_f[n])
        return WeilRepModularForm(weight, S, Z, weilrep = self)

    def recover_modular_form_from_coefficient_vector(self, k, coefficient_vector, prec, starting_from = 0):
        r"""
        Recover a WeilRepModularForm for this representation from its coefficient vector.

        INPUT:
        - ``k`` -- the weight of the modular form
        - ``coefficient_vector`` -- a vector of coefficients
        - ``prec`` -- precision
        - ``starting_from`` -- the exponent at which the vector of coefficients begins (default 0)

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[2,0],[0,-2]]))
            sage: e4 = w.eisenstein_series(4, 5)
            sage: v = e4.coefficient_vector()
            sage: e4 == w.recover_modular_form_from_coefficient_vector(4, v, 5)
            True

        """
        q, = PowerSeriesRing(QQ, 'q').gens()
        symm = self.is_symmetric_weight(k)
        Y = self.coefficient_vector_exponents(prec, symm = symm, starting_from = starting_from, include_vectors = True)
        eps = 2 * symm - 1
        _ds = self.ds()
        _ds_dict = self.ds_dict()
        _indices = self.rds(indices = True)
        _norm_list = self.norm_list()
        X = [None] * len(_ds)
        for i, c in enumerate(coefficient_vector):
            g, n = Y[i]
            j = _ds_dict[tuple(g)]
            if X[j]:
                X[j][2] += c * q**(ceil(n))
            else:
                X[j] = [vector(g), _norm_list[j], O(q**(prec - floor(_norm_list[j]))) + c * q**(ceil(n))]
            minus_g = tuple(frac(-x) for x in g)
            if minus_g != g:
                j2 = _ds_dict[minus_g]
                if X[j2]:
                    X[j2][2] += eps * c * q**(ceil(n))
                else:
                    X[j2] = [vector(minus_g), _norm_list[j], O(q**(prec - floor(_norm_list[j]))) + eps * c * q**(ceil(n))]
        for i, g in enumerate(_ds):
            if X[i] is None:
                X[i] = g, _norm_list[i], O(q**(prec - floor(_norm_list[i])))
            else:
                X[i] = tuple(X[i])
        return WeilRepModularForm(k, self.gram_matrix(), X, weilrep = self)

    def theta_series(self, prec, P = None, _list = False):
        r"""
        Construct vector-valued theta series.

        This computes the theta series \sum_x P(x) q^(-Q(x)) e_x, where Q is a negative-definite quadratic form, P is a harmonic homogeneous polynomial and x runs through the *dual* lattice of Q.

        NOTE: We take *negative-definite* quadratic forms because the Weil representation here is the *dual* of the theta representation. This is somewhat unfortunate, but the convention here is more natural for working with Jacobi forms.

        ALGORITHM: try to apply PARI qfminim on the inverse of the Gram matrix. if this fails then we rescale the inverse of the Gram matrix by the level of Q to obtain something integral. this seems to be necessary to get the nontrivial components of the theta series. (The e_0 component is simply the result of (-Q).theta_series())

        INPUT:
        - ``prec`` -- the precision
        - ``P`` -- a polynomial which is homogeneous and is harmonic with respect to the underlying quadratic form
        - ``test_P`` -- a boolean (default True). If False then we do not test whether P is homogeneous and harmonic. (If P is not harmonic then the theta series is only a quasi-modular form!)

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(-matrix([[2,1],[1,4]])).theta_series(5)
            [(0, 0), 1 + 2*q + 4*q^2 + 6*q^4 + O(q^5)]
            [(3/7, 1/7), 2*q^(2/7) + q^(9/7) + 5*q^(16/7) + 2*q^(23/7) + O(q^(37/7))]
            [(6/7, 2/7), q^(1/7) + 4*q^(8/7) + 4*q^(22/7) + 2*q^(29/7) + O(q^(36/7))]
            [(2/7, 3/7), 3*q^(4/7) + 2*q^(11/7) + 2*q^(18/7) + q^(25/7) + 6*q^(32/7) + O(q^(39/7))]
            [(5/7, 4/7), 3*q^(4/7) + 2*q^(11/7) + 2*q^(18/7) + q^(25/7) + 6*q^(32/7) + O(q^(39/7))]
            [(1/7, 5/7), q^(1/7) + 4*q^(8/7) + 4*q^(22/7) + 2*q^(29/7) + O(q^(36/7))]
            [(4/7, 6/7), 2*q^(2/7) + q^(9/7) + 5*q^(16/7) + 2*q^(23/7) + O(q^(37/7))]

            sage: from weilrep import WeilRep
            sage: R.<x,y> = PolynomialRing(QQ)
            sage: P = x^2 - 2*y^2
            sage: WeilRep(matrix([[-2,-1],[-1,-4]])).theta_series(10, P = P)
            [(0, 0), 2*q - 6*q^2 + 10*q^4 - 14*q^7 - 6*q^8 + 18*q^9 + O(q^10)]
            [(3/7, 1/7), 3/7*q^(2/7) - 9/7*q^(9/7) + 11/7*q^(16/7) - 18/7*q^(23/7) + 38/7*q^(37/7) + 30/7*q^(44/7) - 162/7*q^(58/7) + O(q^(72/7))]
            [(6/7, 2/7), -1/7*q^(1/7) + 3/7*q^(8/7) - 18/7*q^(22/7) + 54/7*q^(29/7) - 45/7*q^(36/7) - 58/7*q^(43/7) + 75/7*q^(50/7) + 13*q^(64/7) + O(q^(71/7))]
            [(2/7, 3/7), -5/7*q^(4/7) + 6/7*q^(11/7) + 27/7*q^(18/7) - 25/7*q^(25/7) - 45/7*q^(32/7) + 54/7*q^(46/7) + 6/7*q^(53/7) + 118/7*q^(67/7) + O(q^(74/7))]
            [(5/7, 4/7), -5/7*q^(4/7) + 6/7*q^(11/7) + 27/7*q^(18/7) - 25/7*q^(25/7) - 45/7*q^(32/7) + 54/7*q^(46/7) + 6/7*q^(53/7) + 118/7*q^(67/7) + O(q^(74/7))]
            [(1/7, 5/7), -1/7*q^(1/7) + 3/7*q^(8/7) - 18/7*q^(22/7) + 54/7*q^(29/7) - 45/7*q^(36/7) - 58/7*q^(43/7) + 75/7*q^(50/7) + 13*q^(64/7) + O(q^(71/7))]
            [(4/7, 6/7), 3/7*q^(2/7) - 9/7*q^(9/7) + 11/7*q^(16/7) - 18/7*q^(23/7) + 38/7*q^(37/7) + 30/7*q^(44/7) - 162/7*q^(58/7) + O(q^(72/7))]

        """
        Q = self.__quadratic_form
        if not Q.is_negative_definite():
            if self.is_positive_definite():
                raise ValueError('Theta series define modular forms for the *dual* Weil representation only for negative-definite lattices. Please replace your WeilRep w by w.dual() and try again.')
            raise ValueError('Not a negative-definite lattice.')
        q, = PowerSeriesRing(QQ, 'q').gens()
        _ds = self.ds()
        _ds_dict = self.ds_dict()
        n_dict = self.norm_dict()
        if P == 0:
            if _list:
                return [self.zero(prec = prec)]
            return self.zero(prec = prec)
        S_inv = -self.gram_matrix().inverse()
        deg_P = 0
        if P:#test whether P is OK
            variables = P.parent().gens()
            deg_P = P.degree()
            if len(variables) != Q.dim():
                raise ValueError('The number of variables in P does not equal the lattice rank')
            if not P.is_homogeneous():
                raise ValueError('Not a homogeneous polynomial')
            u = vector(P.gradient())*S_inv
            P1 = sum(x.derivative(variables[i]) for i, x in enumerate(u))
            if P1:
                t = self.theta_series(prec, P = -P1/2, _list = True)
            else:
                t = []
                #raise ValueError('Not a harmonic polynomial')
        else:
            P = lambda x: 1
            t = []
        try:
            _, _, vs_matrix = pari(S_inv).qfminim(prec + prec + 1, flag=2)
            vs_list = vs_matrix.sage().columns()
            X = [[g, n_dict[tuple(g)], O(q ** (prec - floor(n_dict[tuple(g)])))] for g in _ds]
            for v in vs_list:
                g = S_inv * v
                P_val = P(list(g))
                v_norm_with_offset = ceil(v*g/2)
                list_g = map(frac, g)
                g = tuple(list_g)
                j1 = _ds_dict[g]
                X[j1][2] += P_val * q ** (v_norm_with_offset)
                if v:
                    minus_g = tuple(frac(-x) for x in g)
                    j2 = _ds_dict[minus_g]
                    X[j2][2] += (-1)**deg_P * P_val * q ** (v_norm_with_offset)
            X[0][2] += P([0]*S_inv.nrows())
        except PariError: #when we are not allowed to use pari's qfminim with flag=2 for some reason. Then multiply S inverse to get something integral. The code below is a little slower
            level = Q.level()
            Q_adj = QuadraticForm(level * S_inv)
            vs_list = Q_adj.short_vector_list_up_to_length(level*prec)
            X = [[g, n_dict[tuple(g)], O(q ** (prec - floor(n_dict[tuple(g)])))] for g in _ds]
            for i in range(len(vs_list)):
                v_norm_offset = ceil(i/level)
                vs = vs_list[i]
                for v in vs:
                    S_inv_v = -S_inv*v
                    v_frac = tuple(frac(x) for x in S_inv_v)
                    j = _ds_dict[v_frac]
                    X[j][2] += P(list(S_inv_v)) * q ** v_norm_offset
        X = WeilRepModularForm(Q.dim()/2 + deg_P, self.gram_matrix(), X, weilrep = self)
        if t:
            if _list:
                t.append(X)
                return t
            else:
                r = len(t)
                f = [1]*(r + 1)
                k = r - 1
                for j in range(1, r + 1):
                    f[k] = j * f[k + 1]
                    k -= 1
                return WeilRepQuasiModularForm(Q.dim()/2 + deg_P, self.gram_matrix(), [y / f[i] for i, y in enumerate(t)] + [X], weilrep = self)
        else:
            if _list:
                return [X]
            return X

    def zero(self, weight = 0, prec = 20):
        r"""
        Construct a WeilRepModularForm of weight 'weight' and precision 'prec' which is identically zero.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).zero(3, 5)
            [(0, 0), O(q^5)]
            [(2/3, 2/3), O(q^(17/3))]
            [(1/3, 1/3), O(q^(17/3))]

        """

        n_list = self.norm_list()
        _ds = self.ds()
        q, = PowerSeriesRing(QQ, 'q').gens()
        o_q = O(q ** prec)
        o_q_2 = O(q ** (prec + 1))
        X = [[g, n_list[i], [o_q_2,o_q][n_list[i] == 0]] for i, g in enumerate(_ds)]
        return WeilRepModularForm(weight, self.gram_matrix(), X, weilrep = self)

    ## dimensions of spaces of modular forms associated to this representation

    def rank(self, symm):
        r"""
        Compute the rank of self's modular forms as a module over the ring M_*(SL_2(Z)) of scalar-valued modular forms.

        INPUT:
        - ``symm`` -- boolean: if True then we consider the module of symmetric forms, if False then antisymmetric forms

        OUTPUT: a natural number

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[4,0],[0,4]]))
            sage: w.rank(0), w.rank(1)
            (6, 10)

        """

        try:
            return [self.__antisymm_rank, self.__symm_rank][symm]
        except AttributeError:
            rds_list = self.rds()
            self.__symm_rank = len(self.rds())
            self.__antisymm_rank = self.__symm_rank - sum(self.__order_two_in_rds_list)
            return [self.__antisymm_rank, self.__symm_rank][symm]

    def cusp_forms_dimension(self, weight, eta_twist = 0, force_Riemann_Roch = False, do_not_round = False):
        r"""
        Compute the dimension of spaces of cusp forms.

        This computes the dimension of the space of cusp forms using Riemann-Roch. The formula is valid in weights > 2. In weights 1/2,1,3/2,2 we compute a --basis-- of the space and take its length. (This is slow!)

        INPUT:
        - ``weight`` -- the weight; a half-integer
        - ``eta_twist`` -- an integer (default 0). This computes instead the dimension of cusp forms after twisting the Weil representation by this power of the eta multiplier
        - ``force_Riemann_Roch`` -- a boolean (default False). If True then we produce the output of the Riemann-Roch formula, regardless of whether this represents the dimension
        - ``do_not_round`` -- a boolean (default False). If True then do not convert the output to an integer. (This is probably only useful for debugging.)

        OUTPUT: the dimension of the space of cusp forms of the given weight and eta twist as an integer (unless specified otherwise)

        NOTE: the first time we compute any dimension we have to compute some Gauss sums. After this it should be faster.

        NOTE: we actually compute all of the dimensions dim(rho * chi^N) where chi is the eta character and where N in Z/24Z. This is not much extra work but the eta-twisted dimensions are never used (yet...)!

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).cusp_forms_dimension(11)
            1

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[4]])).cusp_forms_dimension(21/2)
            1
        """
        eta_twist %= 24
        d = self.discriminant()
        symm = self.is_symmetric_weight(weight - eta_twist/2)
        if weight <= 0 or symm is None:
            return 0
        elif weight >= sage_three_half or force_Riemann_Roch:
            eps = 1 if symm else -1
            modforms_rank = self.rank(symm)
            pi_i = complex(0.0, math.pi)
            sig = self.signature()
            try:
                gauss_sum_1 = self.__gauss_sum_1
                gauss_sum_2 = self.__gauss_sum_2
                gauss_sum_3 = self.__gauss_sum_3
                alpha_T = self.__alpha_T[eta_twist]
                if eps == -1:
                    alpha_T -= self.__alpha_T_order_two[eta_twist]
                sqrt_A = math.sqrt(self.discriminant())
                count_isotropic_vectors = self.__count_isotropic_vectors
                count_isotropic_vectors_of_order_two = self.__count_isotropic_vectors_of_order_two
            except AttributeError:
                S = self.gram_matrix()
                rds_grp = self.rds()
                order_two_indices = self.__order_two_in_rds_list
                sqrt_A = math.sqrt(self.discriminant())
                count_isotropic_vectors = [0]*24
                count_isotropic_vectors_of_order_two = [0]*24
                self.__alpha_T = vector([0]*24)
                self.__alpha_T_order_two = vector([0]*24)
                gauss_sum_1 = cmath.exp(pi_i * sig / 4) * sqrt_A #milgram formula. unfortunately we are going to iterate over the discriminant group anyway
                self.__gauss_sum_1 = gauss_sum_1
                gauss_sum_2 = 0.0
                gauss_sum_3 = 0.0
                for i, g in enumerate(rds_grp):
                    gsg = (g * S * g)/2
                    multiplier = 2 - order_two_indices[i]
                    v = vector([frac(-gsg + QQ(N)/24) for N in range(24)]) #fix this?
                    for j in range(len(v)):
                        if v[j] == 0:
                            count_isotropic_vectors[j] += 1
                            if order_two_indices[i]:
                                count_isotropic_vectors_of_order_two[j] += 1
                    self.__alpha_T += v
                    if order_two_indices[i]:
                        self.__alpha_T_order_two += v
                    pi_i_gsg = pi_i * gsg.n()
                    gauss_sum_2 += multiplier * cmath.exp(4 * pi_i_gsg)
                    gauss_sum_3 += multiplier * cmath.exp(-6 * pi_i_gsg)
                self.__gauss_sum_2 = gauss_sum_2
                self.__gauss_sum_3 = gauss_sum_3
                self.__count_isotropic_vectors = count_isotropic_vectors
                self.__count_isotropic_vectors_of_order_two = count_isotropic_vectors_of_order_two
                alpha_T = self.__alpha_T[eta_twist]
                alpha_T_order_two = self.__alpha_T_order_two[eta_twist]
                if not symm:
                    alpha_T = alpha_T - alpha_T_order_two
            g2 = gauss_sum_2.real if symm else gauss_sum_2.imag
            result_dim = modforms_rank * (weight + 5)/12 +  (cmath.exp(pi_i * (2*weight + sig + 1 - eps - eta_twist)/4) * g2).real / (4*sqrt_A) - alpha_T -  (cmath.exp(pi_i * (3*sig - 2*eta_twist + 4 * weight - 10)/12) * (gauss_sum_1 + eps * gauss_sum_3)).real / (3 * math.sqrt(3) * sqrt_A) - count_isotropic_vectors[eta_twist] + (1 - symm) * count_isotropic_vectors_of_order_two[eta_twist]
            if not force_Riemann_Roch:
                if weight == 2:
                    result_dim += self._invariants_dim()
                elif weight == sage_three_half:
                    result_dim += len(self.dual()._weight_one_half_basis(1))
            if do_not_round:
                return result_dim
            else:
                return Integer(round(result_dim))
        else:#not good
            if eta_twist:
                raise ValueError('Not yet implemented')
            return len(self.cusp_forms_basis(weight))

    def modular_forms_dimension(self, weight, eta_twist = 0, force_Riemann_Roch = False, do_not_round = False):
        r"""
        Compute the dimension of spaces of modular forms.

        This computes the dimension of the space of modular forms using Riemann-Roch. The formula is valid in weights >= 2. In weights 0,1/2,1,3/2 we compute a *basis* of the space and then take its length. (This is slow!)

        INPUT:
        - ``weight`` -- the weight; a half-integer
        - ``eta_twist`` -- an integer (default 0). This computes instead the dimension of modular forms after twisting the Weil representation by this power of the eta multiplier
        - ``force_Riemann_Roch`` -- a boolean (default False). If True then we produce the output of the Riemann-Roch formula, regardless of whether this represents the dimension
        - ``do_not_round`` -- a boolean (default False). If True then do not convert the output to an integer. (This is probably only useful for debugging.)

        OUTPUT: the dimension of the space of modular forms of the given weight and eta twist as an integer (unless specified otherwise)

        NOTE: the first time we compute any dimension, a lot of Gauss sums have to be computed. After this it should be fast.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).modular_forms_dimension(11)
            2

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[4]])).modular_forms_dimension(21/2)
            1

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[4]])).modular_forms_dimension(17/2, 2)
            2

        """
        eta_twist %= 24
        k = weight - eta_twist / 2
        if k >= 2 or force_Riemann_Roch:
            symm = self.is_symmetric_weight(k)
            cusp_dim = self.cusp_forms_dimension(weight, eta_twist, force_Riemann_Roch = True, do_not_round = do_not_round)
            return cusp_dim + self.__count_isotropic_vectors[eta_twist] + (symm - 1) * self.__count_isotropic_vectors_of_order_two[eta_twist]
        elif weight < 0:
            return 0
        elif eta_twist:
            raise ValueError('Not yet implemented')
        elif weight == 0:
            return self._invariants_dim()
        elif weight == sage_three_half:
            wdual = self.dual()
            s = wdual.cusp_forms_dimension(sage_one_half, force_Riemann_Roch = True, do_not_round = do_not_round)
            return len(wdual.cusp_forms_basis(sage_one_half, 1)) - s
        else:
            return len(self.modular_forms_basis(weight))

    def quasimodular_forms_dimension(self, weight):
        return sum(self.modular_forms_dimension(weight - 2*k) for k in range(floor(weight / 2) + 1))

    def invariant_cusp_forms_dimension(self, weight, G = None, chi = None, force_Riemann_Roch = False, do_not_round = False):
        return self.invariant_forms_dimension(weight, cusp_forms = True, G = G, chi = chi, force_Riemann_Roch = force_Riemann_Roch, do_not_round = do_not_round)

    def invariant_forms_dimension(self, weight, cusp_forms = False, G = None, chi=None, force_Riemann_Roch = False, do_not_round = False):
        r"""

        INPUT:
        - ``weight`` -- the weight
        - ``cusp_forms`` -- boolean (default False) if True then count only cusp forms
        - ``G`` -- a group of automorphisms of self (typically constructed via self.automorphism_group()) Default None; if G = None then we assume G is the full automorphism group. WARNING: ``G`` must contain the ``canonical involution`` x -> -x
        - ``chi`` -- a character chi : G --> C^* (Default None) This should be a list of values [chi(g1),...,chi(g_n)], where G = [g1,...,g_n]. If chi=None then we assume chi is the trivial character. WARNING: we do not check whether chi is actually a character!
        - ``force_Riemann_Roch`` -- boolean (Default False) If True then use the Riemann--Roch formula even when it does not apply

        EXAMPLES::

            sage: from weilrep import *
            sage: w = II(5)
            sage: w.invariant_forms_dimension(8)
            5
        """
        symm = self.is_symmetric_weight(weight)
        sqrt_d = math.sqrt(self.discriminant())
        if weight <= 0 or symm is None:
            return 0
        if G is None:
            G = self.automorphism_group()
        if chi is None:
            chi = [1]*len(G)
        try:
            i = G.index(self.canonical_involution())
            if (-1)**symm + chi[i]:
                raise ValueError('This character does not satisfy chi(-1) = (-1)^k.')
        except AttributeError:
            raise ValueError('The automorphism group you provided does not contain x --> -x.') from None
        if weight >= sage_five_half or force_Riemann_Roch:
            eps = 1 if symm else -1
            modforms_rank = self.rank(symm)
            two_pi_i = complex(0.0, 2 * math.pi)
            sig = self.signature()
            dsdict = self.ds_dict()
            ds = self.ds()
            if not symm:
                rds = self.rds(indices = True)
            else:
                rds = ds
            d = [[0]*len(ds) for _ in ds]
            e = set([])
            s1 = 0
            s2 = 0
            s3 = []
            n = self.norm_list()
            for j, x in enumerate(ds):
                sx = self.gram_matrix() * x
                qx = n[j]
                if tuple(x) not in e:
                    new = True
                    g = frac(qx)
                    if g < 0:
                        g += 1
                    s3.append(-g)
                else:
                    new = False
                    s3.append(0)
                qx *= two_pi_i
                for i, g in enumerate(G):
                    gx = g(x)
                    u = chi[i]
                    h = two_pi_i * (gx * sx).n()
                    gx = tuple(gx)
                    k = dsdict[gx]
                    s1 += cmath.exp(h) * u
                    s2 += cmath.exp(h + qx) * u
                    if new:
                        d[j][k] += u
                    e.add(gx)
            s3 = sum(x for i, x in enumerate(s3) if any(d[i]))
            if cusp_forms:
                s4 = len([i for i, x in enumerate(d) if not n[i] and any(x)])
            d = sum(1 for _ in filter(any, d))
            zeta1 = cmath.exp(two_pi_i * (2 * weight + sig) / 8)
            zeta2 = cmath.exp(two_pi_i * (4 * weight + 3 * sig - 10)/ 24)
            s = d * (weight + 5) / 12 + zeta1 * s1 / (4 * len(G) * sqrt_d) - (zeta2 * s2 / (len(G) * sqrt_d)).real * (2 / (3 * math.sqrt(3))) + s3
            s = s.real
            if cusp_forms:
                s -= s4
            if do_not_round:
                return s
            return Integer(round(s))
        else:
            if cusp_forms:
                return len(self.invariant_cusp_forms_basis(weight, 1, G = G, chi = chi))
            return len(self.invariant_forms_basis(weight, 1, G = G, chi = chi))


    def hilbert_series(self, polynomial = False):
        r"""
        Compute the Hilbert series \sum_k dim M_{floor k}(rho) t^k.

        INPUT:
        - ``polynomial`` -- boolean (default False). If True then output the Hilbert Polynomial f * (1 - t^4) * (1 - t^6) instead.

        EXAMPLES::

            sage: from weilrep import *
            sage: w = WeilRep(CartanMatrix(['D', 4]))
            sage: w.hilbert_series()
            (2*t^6 + t^4 + t^2)/(t^10 - t^6 - t^4 + 1)
        """
        eps = Integer(self.signature() % 2)
        r, t = PolynomialRing(ZZ, 't').objgen()
        d = []
        p = []
        discr = self.discriminant()
        k = eps / 2
        s = 0
        while s < discr:
            p.append(self.modular_forms_dimension(k))
            d.append(p[-1])
            if len(p) > 4:
                p[-1] -= d[-5]
                if len(p) > 6:
                    p[-1] -= d[-7]
                    if len(p) > 10:
                        p[-1] += d[-11]
            k += 1
            s += p[-1]
        if polynomial:
            return r(p)
        return r(p) / ((1 - t**4) * (1 - t**6))

    def hilbert_polynomial(self):
        return self.hilbert_series(polynomial = True)

    ## bases of spaces associated to this representation

    def _eisenstein_packet(self, k, prec, dim = None, include_E = False):#packet of cusp forms that can be computed using only Eisenstein series
        try:
            k = Integer(k)
        except TypeError:
            pass
        j = floor((k - 1) / 2)
        dim = floor(min(dim, j) if dim else j)
        k_list = [k - (j + j) for j in srange(dim)]
        E = self.eisenstein_series(k_list, ceil(prec))
        if include_E:
            X = [E[0]]
        else:
            X = []
        def repeated_serre_deriv(x, N):
            if N <= 0:
                return x
            return repeated_serre_deriv(x.serre_derivative(normalize_constant_term = True), N - 1)
        if len(k_list) > 1:
            e4 = eisenstein_series_qexp(4, prec, normalization='constant')
            if len(k_list) > 2:
                e6 = eisenstein_series_qexp(6, prec, normalization='constant')
        for c in srange(dim // 3 + 1):
            if c:
                s6 = smf(6 * c, e6 ** c)
            three_c = c + c + c
            for b in srange((dim - 3*c) // 2 + 1):
                if b:
                    s4 = smf(4*b, e4 ** b)
                i = three_c + b + b
                for a in range(dim - i):
                    if (i or a) and len(X) < (4/3) * dim: #random multiple...
                        u = E[a + i]
                        if b:
                            u = u * s4
                        if c:
                            u = u * s6
                        u = repeated_serre_deriv(u, a)
                        if include_E:
                            u *= u.denominator()
                            X.append(u)
                        else:
                            u = E[0] - u
                            u *= u.denominator()
                            X.append(u)
        b = WeilRepModularFormsBasis(k, X, self)
        if include_E:
            return b
        return E[0], b

    def cusp_forms_basis(self, k, prec=None, verbose = False, E = None, dim = None, save_pivots = False, echelonize = True, symmetry_data = None):#basis of cusp forms
        r"""
        Compute a basis of the space of cusp forms.

        ALGORITHM: If k is a symmetric weight, k >= 5/2, then we compute a basis from linear combinations of self's eisenstein_series() and pss(). If k is an antisymmetric weight, k >= 7/2, then we compute a basis from self's pssd(). Otherwise, we compute S_k as the intersection
        S_k(\rho^*) = E_4^(-1) * S_{k+4}(\rho^*) intersect E_6^(-1) * S_{k+6}(\rho^*). (This is slow!!)
        The basis is converted to echelon form (i.e. a ``Victor Miller basis``)

        INPUT:
        - ``k`` -- the weight (half-integer)
        - ``prec`` -- precision (default None). If precision is not given then we use the Sturm bound.
        - ``verbose`` -- boolean (default False). If True then add comments throughout the computation.
        - ``E`` -- WeilRepModularForm (default None). If this is given then the computation assumes that E is the Eisenstein series of weight k.
        - ``dim`` -- (default None) If given then we stop computing after having found 'dim' vectors. (this is automatically minimized to the true dimension)
        - ``save_pivots`` -- boolean (default False) If True then we also output the pivots of each element of the basis' coefficient-vectors
        - ``echelonize`` -- boolean (default True) If False then we skip some echelonization steps.

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[4,0],[0,4]])).cusp_forms_basis(6, 5)
            [(0, 0), O(q^5)]
            [(1/4, 0), O(q^(47/8))]
            [(1/2, 0), O(q^(11/2))]
            [(3/4, 0), O(q^(47/8))]
            [(0, 1/4), 6*q^(7/8) - 46*q^(15/8) + 114*q^(23/8) - 72*q^(31/8) + 42*q^(39/8) + O(q^(47/8))]
            [(1/4, 1/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^(23/4))]
            [(1/2, 1/4), q^(3/8) + 3*q^(11/8) - 75*q^(19/8) + 282*q^(27/8) - 276*q^(35/8) + O(q^(43/8))]
            [(3/4, 1/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^(23/4))]
            [(0, 1/2), O(q^(11/2))]
            [(1/4, 1/2), O(q^(43/8))]
            [(1/2, 1/2), O(q^5)]
            [(3/4, 1/2), O(q^(43/8))]
            [(0, 3/4), -6*q^(7/8) + 46*q^(15/8) - 114*q^(23/8) + 72*q^(31/8) - 42*q^(39/8) + O(q^(47/8))]
            [(1/4, 3/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^(23/4))]
            [(1/2, 3/4), -q^(3/8) - 3*q^(11/8) + 75*q^(19/8) - 282*q^(27/8) + 276*q^(35/8) + O(q^(43/8))]
            [(3/4, 3/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^(23/4))]
            ------------------------------------------------------------
            [(0, 0), O(q^5)]
            [(1/4, 0), 6*q^(7/8) - 46*q^(15/8) + 114*q^(23/8) - 72*q^(31/8) + 42*q^(39/8) + O(q^(47/8))]
            [(1/2, 0), O(q^(11/2))]
            [(3/4, 0), -6*q^(7/8) + 46*q^(15/8) - 114*q^(23/8) + 72*q^(31/8) - 42*q^(39/8) + O(q^(47/8))]
            [(0, 1/4), O(q^(47/8))]
            [(1/4, 1/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^(23/4))]
            [(1/2, 1/4), O(q^(43/8))]
            [(3/4, 1/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^(23/4))]
            [(0, 1/2), O(q^(11/2))]
            [(1/4, 1/2), q^(3/8) + 3*q^(11/8) - 75*q^(19/8) + 282*q^(27/8) - 276*q^(35/8) + O(q^(43/8))]
            [(1/2, 1/2), O(q^5)]
            [(3/4, 1/2), -q^(3/8) - 3*q^(11/8) + 75*q^(19/8) - 282*q^(27/8) + 276*q^(35/8) + O(q^(43/8))]
            [(0, 3/4), O(q^(47/8))]
            [(1/4, 3/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^(23/4))]
            [(1/2, 3/4), O(q^(43/8))]
            [(3/4, 3/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^(23/4))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[-12]])).cusp_forms_basis(1/2, 20)
            [(0), O(q^20)]
            [(11/12), q^(1/24) - q^(25/24) - q^(49/24) + q^(121/24) + q^(169/24) - q^(289/24) - q^(361/24) + O(q^(481/24))]
            [(5/6), O(q^(121/6))]
            [(3/4), O(q^(163/8))]
            [(2/3), O(q^(62/3))]
            [(7/12), -q^(1/24) + q^(25/24) + q^(49/24) - q^(121/24) - q^(169/24) + q^(289/24) + q^(361/24) + O(q^(481/24))]
            [(1/2), O(q^(41/2))]
            [(5/12), -q^(1/24) + q^(25/24) + q^(49/24) - q^(121/24) - q^(169/24) + q^(289/24) + q^(361/24) + O(q^(481/24))]
            [(1/3), O(q^(62/3))]
            [(1/4), O(q^(163/8))]
            [(1/6), O(q^(121/6))]
            [(1/12), q^(1/24) - q^(25/24) - q^(49/24) + q^(121/24) + q^(169/24) - q^(289/24) - q^(361/24) + O(q^(481/24))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2, 0], [0, 2]])).cusp_forms_basis(9, 10)
            [(0, 0), -20*q + 144*q^2 - 448*q^3 + 2240*q^4 - 12200*q^5 + 29440*q^6 - 6272*q^7 - 81664*q^8 + 68460*q^9 + O(q^10)]
            [(1/2, 0), 128*q^(7/4) - 1280*q^(11/4) + 4480*q^(15/4) - 3840*q^(19/4) - 14464*q^(23/4) + 40704*q^(27/4) - 35840*q^(31/4) - 2560*q^(35/4) + 88960*q^(39/4) + O(q^(43/4))]
            [(0, 1/2), 8*q^(3/4) - 16*q^(7/4) - 200*q^(11/4) + 400*q^(15/4) + 2280*q^(19/4) - 4528*q^(23/4) - 15600*q^(27/4) + 30400*q^(31/4) + 70880*q^(35/4) - 132720*q^(39/4) + O(q^(43/4))]
            [(1/2, 1/2), q^(1/2) - 80*q^(3/2) + 610*q^(5/2) - 1120*q^(7/2) - 3423*q^(9/2) + 14800*q^(11/2) - 5470*q^(13/2) - 48800*q^(15/2) + 73090*q^(17/2) + 15600*q^(19/2) + O(q^(21/2))]
            ------------------------------------------------------------
            [(0, 0), O(q^10)]
            [(1/2, 0), q^(3/4) - 18*q^(7/4) + 135*q^(11/4) - 510*q^(15/4) + 765*q^(19/4) + 1242*q^(23/4) - 7038*q^(27/4) + 8280*q^(31/4) + 9180*q^(35/4) - 27710*q^(39/4) + O(q^(43/4))]
            [(0, 1/2), -q^(3/4) + 18*q^(7/4) - 135*q^(11/4) + 510*q^(15/4) - 765*q^(19/4) - 1242*q^(23/4) + 7038*q^(27/4) - 8280*q^(31/4) - 9180*q^(35/4) + 27710*q^(39/4) + O(q^(43/4))]
            [(1/2, 1/2), O(q^(21/2))]
         """
        if symmetry_data is not None:
            return self.invariant_cusp_forms_basis(k, prec, G = symmetry_data[0], chi = symmetry_data[1], verbose = verbose)
        try:
            k = Integer(k)
        except TypeError:
            k = QQ(k)
        if k <= 0 or (dim is not None and dim <= 0):
            return WeilRepModularFormsBasis(k, [], self)
        if not save_pivots:
            try:
                old_prec, X = self.__cusp_forms_basis[k]
                if old_prec >= prec or not X:
                    if old_prec == prec or not X:
                        return X
                    X = WeilRepModularFormsBasis(k, [x.reduce_precision(prec, in_place = False) for x in X], self)
                    return X
            except KeyError:
                pass
        S = self.gram_matrix()
        if verbose:
            print('I am now looking for cusp forms for the Weil representation for the Gram matrix\n%s'%S)
        _norm_dict = self.norm_dict()
        symm = self.is_symmetric_weight(k)
        if symm is None:
            return WeilRepModularFormsBasis(k, [], self) #should this raise an error instead?
        def return_pivots():
            if verbose:
                print('Done!')
            if save_pivots:
                return X, pivots
            return X
        X = WeilRepModularFormsBasis(k, [], self)
        if k > 2:
            true_dim = self.cusp_forms_dimension(k)
            if dim is None:
                dim = true_dim
            else:
                dim = Integer(min(dim, true_dim))
            if not dim:
                self.__cusp_forms_basis[k] = prec, X
                pivots = []
                return return_pivots()
            elif verbose:
                print('I need to find %d cusp forms of weight %s.' %(dim, k))
        sturm_bound = k / 12
        if not prec:
            prec = ceil(sturm_bound)
        else:
            prec = ceil(max(prec, sturm_bound))
        if k == sage_one_half:
            X = self._weight_one_half_basis(prec)
            return WeilRepModularFormsBasis(sage_one_half, [x for x in X if x.valuation(exact = True)], self)
        X = WeilRepModularFormsBasis(k, [], self)
        rank = 0
        if k >= sage_seven_half or (k >= sage_five_half and symm):
            deltasmf = [smf(12, delta_qexp(prec))]
            try:
                oldprec, Y = self.__cusp_forms_basis[k - 2]
                if oldprec >= prec:
                    X = WeilRepModularFormsBasis(k, [y.reduce_precision(prec, in_place = False).serre_derivative() for y in Y], self)
                    rank = X.rank()
                    if rank >= dim:
                        if verbose:
                            print('I found %d cusp forms using the cached cusp forms basis of weight %s.'%(len(X), k-2))
                        pivots = X.echelonize(save_pivots = save_pivots)
                        self.__cusp_forms_basis[k] = prec, X
                        return return_pivots()
            except KeyError:
                if k - 6 >= sage_seven_half or (k - 6 >= sage_five_half and symm):
                    e4 = smf(4, eisenstein_series_qexp(4, prec, normalization = 'constant'))
                    e6 = smf(6, eisenstein_series_qexp(6, prec, normalization = 'constant'))
                    X6 = self.cusp_forms_basis(k - 6, prec, verbose = verbose, echelonize=False)
                    X4 = [x.serre_derivative() for x in X6]
                    if verbose:
                        print('-'*80)
                    X = WeilRepModularFormsBasis(k, [x*e4 for x in X4], self) + WeilRepModularFormsBasis(k, [x*e6 for x in X6], self) + WeilRepModularFormsBasis(k, [x.serre_derivative().serre_derivative() for x in X4], self)
                    X.remove_nonpivots()
                    rank = len(X)
                    if rank == dim:
                        X.echelonize()
                        return X
                else:
                    rank = 0
            if symm and k >= sage_nine_half:
                E, Y = self._eisenstein_packet(k, prec, dim = dim+1)
                if verbose and Y:
                    print('I computed a packet of %d cusp forms using Eisenstein series. (They may be linearly dependent.)'%len(Y))
                    X.extend(Y)
            elif symm and not E:
                E = self.eisenstein_series(k, prec)
                if verbose:
                    print('I computed the Eisenstein series of weight %s up to precision %s.' %(k, prec))
            if X:
                rank = X.rank()
                if rank >= dim:
                    pivots = X.echelonize(save_pivots = save_pivots)
                    self.__cusp_forms_basis[k] = prec, X
                    return return_pivots()
                pass
            m0 = 1
            G = self.sorted_rds()
            indices = self.rds(indices = True)
            ds = self.ds()
            norm_list = self.norm_list()
            b_list = [i for i in range(len(ds)) if not (indices[i] or norm_list[i])]
            def t_packet_1(X, k, m, b, max_dim, prec, verbose = False): #symmetric
                w_new = self.embiggen(b, m)
                if max_dim > 3 and k in ZZ:
                    z = w_new.cusp_forms_basis(k - sage_one_half, prec, echelonize = False, verbose = verbose, dim = max_dim).theta(weilrep = self)
                    if z and verbose:
                        print('-'*80)
                        print('I am returning to the Gram matrix\n%s'%self.gram_matrix())
                        print('I computed a packet of %d cusp forms using the index %s.'%(len(z), (b, m)))
                    X.extend(z)
                if k > sage_nine_half:
                    _, x = w_new._eisenstein_packet(k - sage_one_half, prec, dim = dim_rank)
                    X.extend(x.theta(weilrep = self))
                    if x and verbose:
                        print('I computed a packet of %d cusp forms using the index %s.'%(len(x), (b, m)))
                X.append(E - self.pss(k, b, m, prec, weilrep = w_new))
                if verbose:
                    print('I computed a Poincare square series of index %s and weight %s.'%([b, m], k))
                if m <= 1:
                    j = 1
                    k_j = k - 12
                    while k_j > sage_five_half:
                        try:
                            delta_j = deltasmf[j - 1]
                        except IndexError:
                            deltasmf.append(deltasmf[-1] * deltasmf[0])
                            delta_j = deltasmf[-1]
                            X.append(delta_j * (self.eisenstein_series(k_j, prec-j) - self.pss(k_j, b, m, prec - j, weilrep = w_new)))
                            if verbose:
                                 print('I computed a Poincare square series of index %s and weight %s.'%([b, m], k_j))
                        j += 1
                        k_j -= 12
                rank = len(X)
                if rank >= dim:
                    X.remove_nonpivots()
                    rank = len(X)
                del w_new
                return X, rank
            def t_packet_2(X, k, m, b, max_dim, prec, verbose = False): #anti-symmetric
                w_new = self.embiggen(b, m)
                if max_dim > 3:
                    z = w_new.cusp_forms_basis(k - sage_three_half, prec, echelonize = False, verbose = verbose, dim = max_dim).theta(weilrep = self, odd = True)
                    if z:
                        X.extend(z)
                        if verbose:
                            print('I computed a packet of %d cusp forms using the index %s.'%(len(z), (b, m)))
                X.append(self.pssd(k, b, m, prec, weilrep = w_new))
                if verbose:
                    print('I computed a Poincare square series of index %s and weight %s.'%([b, m], k))
                if m <= 1:
                    j = 1
                    k_j = k - 12
                    while k_j > sage_seven_half:
                        try:
                            delta_j = deltasmf[j - 1]
                        except IndexError:
                            deltasmf.append(deltasmf[-1] * deltasmf[0])
                            delta_j = deltasmf[-1]
                            X.append(delta_j * self.pssd(k_j, b, m, prec - j, weilrep = w_new))
                            if verbose:
                                 print('I computed a Poincare square series of index %s and weight %s.'%([b, m], k_j))
                        j += 1
                        k_j -= 12
                rank = len(X)
                if rank >= dim:
                    X.remove_nonpivots()
                    rank = len(X)
                del w_new
                return X, rank
            while rank < dim:
                for b_tuple in G:
                    b = vector(b_tuple)
                    if symm or b.denominator() > 2:
                        m = m0 + _norm_dict[b_tuple]
                        dim_rank = dim - rank
                        if dim_rank <= 0:
                            X.remove_nonpivots()
                            rank = len(X)
                            dim_rank = dim - rank
                        if symm:
                            if dim_rank > 2:
                                X, rank = t_packet_1(X, k, m, b, dim_rank, prec, verbose = verbose)
                            elif dim_rank:
                                X.append(E - self.pss(k, b, m, prec))
                                if verbose:
                                    print('I computed a Poincare square series of index %s.'%([b, m]))
                                rank += 1
                        else:
                            if dim_rank > 2:
                                X, rank = t_packet_2(X, k, m, b, dim_rank, prec, verbose = verbose)
                            elif dim_rank:
                                X.append(self.pssd(k, b, m, prec))
                                if verbose:
                                    print('I computed a Poincare square series of index %s.'%([b, m]))
                                rank += 1
                        if rank >= dim:
                            X.remove_nonpivots()
                            rank = len(X)
                            if verbose:
                                print('I have found %d out of %d cusp forms.'%(rank, dim))
                        if rank >= dim:
                            if echelonize:
                                if verbose:
                                    print('I am computing an echelon form.')
                                pivots = X.echelonize(save_pivots = save_pivots)
                                self.__cusp_forms_basis[k] = prec, X
                            return return_pivots()
                m0 += 1
                if m0 > prec and rank < dim:#this will probably never happen but lets be safe
                    raise RuntimeError('Something went horribly wrong!')
            if echelonize:
                if verbose:
                    print('I am computing an echelon form.')
                pivots = X.echelonize(save_pivots = save_pivots)
                self.__cusp_forms_basis[k] = prec, X
            return return_pivots()
        else:
            p = self.discriminant()
            if symm and p.is_prime() and p != 2:
                if verbose:
                    print('The discriminant is prime so I can construct cusp forms via the Bruinier--Bundschuh lift.')
                chi = DirichletGroup(p)[(p-1)//2]
                cusp_forms = CuspForms(chi, k, prec = ceil(p*prec)).echelon_basis()
                mod_sturm_bound = ceil(p * k / 12)
                sig = self.signature()
                eps = sig == 0 or sig == 6
                eps = 1 - 2 * eps
                m = matrix([[y for i, y in enumerate(x.coefficients(mod_sturm_bound)) if kronecker(i + 1, p) == eps] for x in cusp_forms])
                v_basis = m.kernel().basis()
                L = [sum([mf * v[i] for i, mf in enumerate(cusp_forms)]) for v in v_basis]
                L = [2*self.bb_lift(x) if x.valuation() % p else self.bb_lift(x) for x in L]
                X = WeilRepModularFormsBasis(k, L, self)
                if echelonize:
                    pivots = X.echelonize(save_pivots = save_pivots)
                    self.__cusp_forms_basis[k] = prec, X
                return return_pivots()
            if verbose:
                print('I am going to compute the spaces of cusp forms of weights %s and %s.' %(k+4, k+6))
            prec = max([2, prec, ceil(sturm_bound + sage_one_half)])
            e4 = smf(-4, ~eisenstein_series_qexp(4, prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6, prec))
            X1 = self.cusp_forms_basis(k + 4, prec, verbose = verbose)
            X2 = self.cusp_forms_basis(k + 6, prec, verbose = verbose)
            if verbose:
                print('I will intersect the spaces E_4^(-1) * S_%s and E_6^(-1) * S_%s.' %(k +4, k +6))
            try:
                V1 = span((x * e4).coefficient_vector() for x in X1)
                V2 = span((x * e6).coefficient_vector() for x in X2)
                X = WeilRepModularFormsBasis(k, [self.recover_modular_form_from_coefficient_vector(k, v, prec) for v in V1.intersection(V2).basis()], self)
                if echelonize:
                    pivots = X.echelonize(save_pivots = save_pivots)
                    self.__cusp_forms_basis[k] = prec, X
                return return_pivots()
            except AttributeError: #we SHOULD only get ``AttributeError: 'Objects_with_category' object has no attribute 'base_ring'`` when X1 or X2 is empty...
                X = WeilRepModularFormsBasis(k, [], self)
                self.__cusp_forms_basis[k] = prec, X
                return return_pivots()

    def modular_forms_basis(self, weight, prec = 0, eisenstein = False, verbose = False, symmetry_data = None):
        r"""
        Compute a basis of the space of modular forms.

        ALGORITHM: If k is a symmetric weight, k >= 5/2, then we compute a basis from linear combinations of self's eisenstein_series() and pss(). If k is an antisymmetric weight, k >= 7/2, then we compute a basis from self's pssd(). Otherwise, we compute M_k as the intersection
        M_k(\rho^*) = E_4^(-1) * M_{k+4}(\rho^*) intersect E_6^(-1) * M_{k+6}(\rho^*). (This is slow!!)
        Note: Eisenstein series at nonzero cusps are not implemented yet so when the Eisenstein space has dim > 1 we instead compute the image in S_{k+12} of Delta * M_k. (This is even slower!!)
        The basis is always converted to echelon form (in particular the Fourier coefficients are integers).

        INPUT:
        - ``k`` -- the weight (half-integer)
        - ``prec`` -- precision (default None). If precision is not given then we use the Sturm bound.
        - ``eisenstein`` -- boolean (default False). If True and weight >= 5/2 then the first element in the output is always the Eisenstein series (i.e. we do not pass to echelon form).
        - ``verbose`` -- boolean (default False). If true then we add comments throughout the computation.

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,0],[0,4]])).modular_forms_basis(5, 10)
            [(0, 0), 1 - 150*q - 2270*q^2 - 11820*q^3 - 36750*q^4 - 89888*q^5 - 188380*q^6 - 344640*q^7 - 589230*q^8 - 954210*q^9 + O(q^10)]
            [(0, 1/4), -80*q^(7/8) - 1808*q^(15/8) - 9840*q^(23/8) - 32320*q^(31/8) - 82160*q^(39/8) - 171360*q^(47/8) - 320528*q^(55/8) - 559600*q^(63/8) - 891600*q^(71/8) - 1365920*q^(79/8) + O(q^(87/8))]
            [(0, 1/2), -10*q^(1/2) - 740*q^(3/2) - 5568*q^(5/2) - 21760*q^(7/2) - 59390*q^(9/2) - 130980*q^(11/2) - 257600*q^(13/2) - 461056*q^(15/2) - 747540*q^(17/2) - 1166180*q^(19/2) + O(q^(21/2))]
            [(0, 3/4), -80*q^(7/8) - 1808*q^(15/8) - 9840*q^(23/8) - 32320*q^(31/8) - 82160*q^(39/8) - 171360*q^(47/8) - 320528*q^(55/8) - 559600*q^(63/8) - 891600*q^(71/8) - 1365920*q^(79/8) + O(q^(87/8))]
            [(1/2, 0), -40*q^(3/4) - 1440*q^(7/4) - 7720*q^(11/4) - 30496*q^(15/4) - 68520*q^(19/4) - 166880*q^(23/4) - 283600*q^(27/4) - 551040*q^(31/4) - 787200*q^(35/4) - 1396960*q^(39/4) + O(q^(43/4))]
            [(1/2, 1/4), -24*q^(5/8) - 1000*q^(13/8) - 6880*q^(21/8) - 24840*q^(29/8) - 65880*q^(37/8) - 145352*q^(45/8) - 276600*q^(53/8) - 485960*q^(61/8) - 805280*q^(69/8) - 1233120*q^(77/8) + O(q^(85/8))]
            [(1/2, 1/2), -368*q^(5/4) - 3520*q^(9/4) - 17040*q^(13/4) - 43840*q^(17/4) - 117440*q^(21/4) - 205440*q^(25/4) - 421840*q^(29/4) - 632000*q^(33/4) - 1117680*q^(37/4) + O(q^(41/4))]
            [(1/2, 3/4), -24*q^(5/8) - 1000*q^(13/8) - 6880*q^(21/8) - 24840*q^(29/8) - 65880*q^(37/8) - 145352*q^(45/8) - 276600*q^(53/8) - 485960*q^(61/8) - 805280*q^(69/8) - 1233120*q^(77/8) + O(q^(85/8))]
            ------------------------------------------------------------
            [(0, 0), -12*q + 56*q^2 - 72*q^3 + 80*q^4 - 352*q^5 + 336*q^6 + 704*q^7 - 1056*q^8 + 540*q^9 + O(q^10)]
            [(0, 1/4), 8*q^(7/8) - 24*q^(15/8) - 40*q^(23/8) + 160*q^(31/8) + 24*q^(39/8) - 272*q^(47/8) + 104*q^(55/8) - 360*q^(63/8) + 72*q^(71/8) + 1424*q^(79/8) + O(q^(87/8))]
            [(0, 1/2), -2*q^(1/2) - 12*q^(3/2) + 112*q^(5/2) - 224*q^(7/2) + 90*q^(9/2) + 52*q^(11/2) - 112*q^(13/2) + 672*q^(15/2) - 452*q^(17/2) - 268*q^(19/2) + O(q^(21/2))]
            [(0, 3/4), 8*q^(7/8) - 24*q^(15/8) - 40*q^(23/8) + 160*q^(31/8) + 24*q^(39/8) - 272*q^(47/8) + 104*q^(55/8) - 360*q^(63/8) + 72*q^(71/8) + 1424*q^(79/8) + O(q^(87/8))]
            [(1/2, 0), 6*q^(3/4) - 16*q^(7/4) - 26*q^(11/4) + 48*q^(15/4) + 134*q^(19/4) + 80*q^(23/4) - 756*q^(27/4) - 320*q^(31/4) + 1920*q^(35/4) - 48*q^(39/4) + O(q^(43/4))]
            [(1/2, 1/4), -4*q^(5/8) + 4*q^(13/8) + 48*q^(21/8) - 44*q^(29/8) - 228*q^(37/8) + 180*q^(45/8) + 492*q^(53/8) - 268*q^(61/8) - 240*q^(69/8) - 208*q^(77/8) + O(q^(85/8))]
            [(1/2, 1/2), q^(1/4) + 8*q^(5/4) - 45*q^(9/4) - 8*q^(13/4) + 226*q^(17/4) - 96*q^(21/4) - 335*q^(25/4) + 88*q^(29/4) - 156*q^(33/4) + 456*q^(37/4) + O(q^(41/4))]
            [(1/2, 3/4), -4*q^(5/8) + 4*q^(13/8) + 48*q^(21/8) - 44*q^(29/8) - 228*q^(37/8) + 180*q^(45/8) + 492*q^(53/8) - 268*q^(61/8) - 240*q^(69/8) - 208*q^(77/8) + O(q^(85/8))]
        """
        if symmetry_data is not None:
            return self.invariant_forms_basis(weight, prec = prec, G = symmetry_data[0], chi = symmetry_data[1], verbose = verbose)
        prec = ceil(prec)
        if not eisenstein:
            try:
                old_prec, X = copy(self.__modular_forms_basis[weight])
                if old_prec >= prec or not X:
                    if old_prec == prec or not X:
                        return X
                    X = WeilRepModularFormsBasis(weight, [x.reduce_precision(prec, in_place = False) for x in X], self)
                    return X
            except KeyError:
                pass
        symm = self.is_symmetric_weight(weight)
        if symm is None:
            return []
        _ds = self.ds()
        _indices = self.rds(indices = True)
        _norm_list = self.norm_list()
        try:
            weight = Integer(weight)
        except TypeError:
            weight = QQ(weight)
        sturm_bound = weight / 12
        prec = max(prec, sturm_bound)
        if weight == 0:
            return self._invariants(prec)
        elif weight == sage_one_half:
            return self._weight_one_half_basis(prec)
        b_list = [i for i in range(len(_ds)) if not (_indices[i] or _norm_list[i]) and (self.__ds_denominators_list[i] < 5 or self.__ds_denominators_list[i] == 6)]
        if weight > 3 or (symm and weight > 2):
            dim1 = self.modular_forms_dimension(weight)
            dim2 = self.cusp_forms_dimension(weight)
            if verbose:
                print('I need to find %d modular forms of weight %s to precision %d.' %(dim1, weight, prec))
            if (symm and dim1 <= dim2 + len(b_list)):
                if verbose:
                    print('I found %d Eisenstein series.' %len(b_list))
                    if dim2 > 0:
                        print('I am now going to look for %d cusp forms of weight %s.' %(dim2, weight))
                L = WeilRepModularFormsBasis(weight, [self.eisenstein_oldform(weight, _ds[i], prec) for i in b_list], self)
                if eisenstein:
                    L.extend(self.cusp_forms_basis(weight, prec, verbose = verbose, E = L[0]))
                    return L
                else:
                    X = WeilRepModularFormsBasis(weight, [x for x in self.cusp_forms_basis(weight, prec, verbose = verbose, E = L[0])], self)
                    X.extend(L)
                    X.echelonize()
                    self.__modular_forms_basis[weight] = prec, X
                    return X
            elif dim1 == dim2:
                X = self.cusp_forms_basis(weight, prec, verbose = verbose)
                self.__modular_forms_basis[weight] = prec, X
                return X
            else:
                pass
        p = self.discriminant()
        if symm and p.is_prime() and p != 2:
            if weight == 0:
                return []
            if verbose:
                print('The discriminant is prime so I can construct modular forms via the Bruinier--Bundschuh lift.')
            chi = DirichletGroup(p)[(p-1)//2]
            mod_forms = ModularForms(chi, weight, prec = ceil(p*prec)).echelon_basis()
            mod_sturm_bound = p * ceil(weight / 12)
            sig = self.signature()
            if (sig == 0 or sig == 6):
                eps = -1
            else:
                eps = 1
            m = matrix([[y for i, y in enumerate(x.coefficients(mod_sturm_bound)) if kronecker_symbol(i + 1, p) == eps] for x in mod_forms])
            v_basis = m.kernel().basis()
            L = [sum([mf * v[i] for i, mf in enumerate(mod_forms)]) for v in v_basis]
            L = [2*self.bb_lift(x) if x.valuation() % p else self.bb_lift(x) for x in L]
            X = WeilRepModularFormsBasis(weight, L, self)
            self.__modular_forms_basis[weight] = prec, X
            return X
        dim1 = self.modular_forms_dimension(weight+4)
        dim2 = self.cusp_forms_dimension(weight+4)
        if symm and (dim1 <= dim2 + len(b_list)):
            if verbose:
                print('I am going to compute the spaces of modular forms of weights %s and %s.' %(weight+4, weight+6))
            prec = max([2, prec, ceil(sturm_bound + sage_one_half)])
            e4 = smf(-4, ~eisenstein_series_qexp(4, prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6, prec))
            X1 = self.modular_forms_basis(weight+4, prec, verbose = verbose)
            X2 = self.modular_forms_basis(weight+6, prec, verbose = verbose)
            if verbose:
                print('I am now going to compute M_%s by intersecting the spaces E_4^(-1) * M_%s and E_6^(-1) * M_%s.' %(weight, weight +4, weight +6))
            try:
                V1 = span([(x * e4).coefficient_vector() for x in X1])
                V2 = span([(x * e6).coefficient_vector() for x in X2])
                V = (V1.intersection(V2)).basis()
                X = WeilRepModularFormsBasis(weight, [self.recover_modular_form_from_coefficient_vector(weight, v, prec) for v in V], self)
                X.echelonize()
                self.__modular_forms_basis[weight] = prec, X
                return X
            except AttributeError:
                X = WeilRepModularFormsBasis(weight, [], self)
                self.__modular_forms_basis[weight] = prec, X
                return X
        else:
            if verbose:
                print('I do not know how to find enough Eisenstein series. I am going to compute the image of M_%s under multiplication by Delta.' %weight)
            return self.nearly_holomorphic_modular_forms_basis(weight, 0, prec, inclusive = True, reverse = False, force_N_positive = True, verbose = verbose)

    basis = modular_forms_basis

    def basis_vanishing_to_order(self, k, N=0, prec=0, inclusive = False,  inclusive_except_zero_component = False, keep_N = False, symmetry_data = None, verbose = False):
        r"""
        Compute bases of modular forms that vanish to a specified order at infinity.

        ALGORITHM: We first try to reduce to lower weight by dividing by a power of the modular Delta function. Then compute the full cusp space and pick out the forms with given vanishing order. (This is easy because cusp_forms_basis computes an echelon form.)

        INPUT:
        - ``k`` -- the weight
        - ``N`` -- the minimum order of vanishing (default 0)
        - ``prec`` -- the precision (default 0); will be raised to at least the Sturm bound
        - ``inclusive`` -- boolean (default False); if True then we also exclude modular forms whose order of vanishing is *exactly* N
        - ``inclusive_except_zero_component`` -- boolean (default False); if True then we exclude modular forms any of whose components (except the zero component) has order of vanishing exactly N
        - ``keep_N`` -- boolean (default False); if True then we skip the first step of trying to reduce to lower weight (for internal use)
        - ``symmetry_data`` -- default None. If not None then it should be a list [G, chi] where G is a group of automorphisms and chi : G --> C^* is a character. (See also invariant_forms_dimension().)
        - ``verbose`` -- boolean (default False); if True then we add commentary.

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[-6]])).basis_vanishing_to_order(21/2, 3/4, 10)
            [(0), -2*q + 24*q^2 - 56*q^3 - 588*q^4 + 4740*q^5 - 13680*q^6 + 8464*q^7 + 51768*q^8 - 137754*q^9 + O(q^10)]
            [(5/6), 3*q^(13/12) - 45*q^(25/12) + 255*q^(37/12) - 519*q^(49/12) - 879*q^(61/12) + 5916*q^(73/12) - 5610*q^(85/12) - 18123*q^(97/12) + 34017*q^(109/12) + O(q^(121/12))]
            [(2/3), -6*q^(4/3) + 102*q^(7/3) - 720*q^(10/3) + 2568*q^(13/3) - 3876*q^(16/3) - 3246*q^(19/3) + 20904*q^(22/3) - 22440*q^(25/3) - 4692*q^(28/3) + O(q^(31/3))]
            [(1/2), q^(3/4) - 10*q^(7/4) + 3*q^(11/4) + 360*q^(15/4) - 1783*q^(19/4) + 1716*q^(23/4) + 11286*q^(27/4) - 35466*q^(31/4) + 1080*q^(35/4) + 148662*q^(39/4) + O(q^(43/4))]
            [(1/3), -6*q^(4/3) + 102*q^(7/3) - 720*q^(10/3) + 2568*q^(13/3) - 3876*q^(16/3) - 3246*q^(19/3) + 20904*q^(22/3) - 22440*q^(25/3) - 4692*q^(28/3) + O(q^(31/3))]
            [(1/6), 3*q^(13/12) - 45*q^(25/12) + 255*q^(37/12) - 519*q^(49/12) - 879*q^(61/12) + 5916*q^(73/12) - 5610*q^(85/12) - 18123*q^(97/12) + 34017*q^(109/12) + O(q^(121/12))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[0,2],[2,0]])).basis_vanishing_to_order(8, 1/2, 10)
            [(0, 0), 8*q - 64*q^2 + 96*q^3 + 512*q^4 - 1680*q^5 - 768*q^6 + 8128*q^7 - 4096*q^8 - 16344*q^9 + O(q^10)]
            [(0, 1/2), -8*q + 64*q^2 - 96*q^3 - 512*q^4 + 1680*q^5 + 768*q^6 - 8128*q^7 + 4096*q^8 + 16344*q^9 + O(q^10)]
            [(1/2, 0), -8*q + 64*q^2 - 96*q^3 - 512*q^4 + 1680*q^5 + 768*q^6 - 8128*q^7 + 4096*q^8 + 16344*q^9 + O(q^10)]
            [(1/2, 1/2), q^(1/2) + 12*q^(3/2) - 210*q^(5/2) + 1016*q^(7/2) - 2043*q^(9/2) + 1092*q^(11/2) + 1382*q^(13/2) - 2520*q^(15/2) + 14706*q^(17/2) - 39940*q^(19/2) + O(q^(21/2))]
        """
        if verbose:
            print('I am now looking for modular forms of weight %s which vanish to order %s at infinity.' %(k, N))
        if inclusive and inclusive_except_zero_component:
            raise ValueError('At most one of "inclusive" and "inclusive_except_zero_component" may be true')
        try:
            k = Integer(k)
        except TypeError:
            k = QQ(k)
        symm = self.is_symmetric_weight(k)
        if symm is None:
            return WeilRepModularFormsBasis(k, [], self)
        sturm_bound = k/12
        prec = max(prec,sturm_bound)
        if N > sturm_bound:
            return []
        elif N == 0:
            if inclusive:
                if verbose:
                    print('The vanishing condition is trivial so I am looking for all cusp forms.')
                return self.cusp_forms_basis(k, prec, verbose = verbose, symmetry_data = symmetry_data)
            else:
                if verbose:
                    print('The vanishing condition is trivial so I am looking for all modular forms.')
                return self.modular_forms_basis(k, prec, eisenstein = False, verbose = verbose, symmetry_data = symmetry_data)
        elif N >= 1 and not keep_N:
            frac_N = frac(N)
            floor_N = floor(N)
            computed_weight = k - 12*floor_N
            if computed_weight <= 2:
                frac_N += 1
                floor_N -= 1
            if frac_N < N:
                smf_delta_N = smf(12*floor_N, delta_qexp(prec)^floor_N)
                if verbose:
                    print('I am going to find a basis of modular forms of weight %s which vanish to order %s at infinity and multiply them by Delta^%d.' %(computed_weight, frac_N, floor_N))
                X = self.basis_vanishing_to_order(computed_weight, frac_N, prec, inclusive, verbose = verbose, symmetry_data = symmetry_data)
                return WeilRepModularFormsBasis(k, [x * smf_delta_N for x in X], self)
        U = self.cusp_forms_basis(k, prec, verbose = verbose, save_pivots = True)
        try:
            cusp_forms, pivots = U
            ell = len(cusp_forms)
        except (TypeError, ValueError):
            if inclusive:
                return WeilRepModularFormsBasis(k, [x for x in U if x.valuation(exact = True) > N], self)
            return WeilRepModularFormsBasis(k, [x for x in U if x.valuation(exact = True) >= N], self)
        Y = self.coefficient_vector_exponents(prec, symm, include_vectors = inclusive_except_zero_component)
        try:
            if inclusive:
                j = next(i for i in range(ell) if Y[pivots[i]] > N)
            elif inclusive_except_zero_component:
                j = next(i for i in range(ell) if (Y[0][pivots[i]] >= N) and (Y[0][pivots[i]] > N or not Y[1][pivots[i]]))
            else:
                j = next(i for i in range(ell) if Y[pivots[i]] >= N)
        except StopIteration:
            return []
        Z = WeilRepModularFormsBasis(k, cusp_forms[j:], self)
        if type(Z) is list:
            Z = WeilRepModularFormsBasis(k, Z, self)
        Z.echelonize()
        return Z

    def nearly_holomorphic_modular_forms_basis(self, k, pole_order, prec = 0, inclusive = True, reverse = True, force_N_positive = False, symmetry_data = None, verbose = False):
        r"""
        Computes a basis of nearly holomorphic modular forms.

        A nearly-holomorphic modular form is a function f : H -> C[L' / L] that is holomorphic on H and meromorphic at cusps and satisfies the usual transformations. In other words they are allowed to have a finite principal part.

        ALGORITHM: compute spaces of modular forms of higher weight and divide by Delta.

        INPUT:
        - ``k`` -- the weight
        - ``pole_order`` -- the worst pole the modular forms are allowed to have
        - ``prec`` -- precision (default 0); will be raised at least to a Sturm bound
        - ``inclusive`` -- boolean (default True); if True then we allow forms with pole order *exactly* pole_order
        - ``reverse`` -- boolean (default True); if True then output forms in reverse echelon order
        - ``symmetry_data`` -- default None. If not None then it should be a list [G, chi] where G is a group of automorphisms and chi : G --> C^* is a character. (See also invariant_forms_dimension().)
        - ``force_N_positive`` -- boolean (default False); if True then we always divide by Delta at least once in the computation. (for internal use)
        - ``verbose`` -- boolean (default False); if True then we add commentary.

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[0,2],[2,0]])).nearly_holomorphic_modular_forms_basis(0, 1, 10)
            [(0, 0), O(q^10)]
            [(0, 1/2), 1 + O(q^10)]
            [(1/2, 0), -1 + O(q^10)]
            [(1/2, 1/2), O(q^(21/2))]
            ------------------------------------------------------------
            [(0, 0), 1 + O(q^10)]
            [(0, 1/2), O(q^10)]
            [(1/2, 0), 1 + O(q^10)]
            [(1/2, 1/2), O(q^(21/2))]
            ------------------------------------------------------------
            [(0, 0), 2048*q + 49152*q^2 + 614400*q^3 + 5373952*q^4 + 37122048*q^5 + 216072192*q^6 + 1102430208*q^7 + 5061476352*q^8 + 21301241856*q^9 + O(q^10)]
            [(0, 1/2), -2048*q - 49152*q^2 - 614400*q^3 - 5373952*q^4 - 37122048*q^5 - 216072192*q^6 - 1102430208*q^7 - 5061476352*q^8 - 21301241856*q^9 + O(q^10)]
            [(1/2, 0), -24 - 2048*q - 49152*q^2 - 614400*q^3 - 5373952*q^4 - 37122048*q^5 - 216072192*q^6 - 1102430208*q^7 - 5061476352*q^8 - 21301241856*q^9 + O(q^10)]
            [(1/2, 1/2), q^(-1/2) + 276*q^(1/2) + 11202*q^(3/2) + 184024*q^(5/2) + 1881471*q^(7/2) + 14478180*q^(9/2) + 91231550*q^(11/2) + 495248952*q^(13/2) + 2390434947*q^(15/2) + 10487167336*q^(17/2) + 42481784514*q^(19/2) + O(q^(21/2))]
            ------------------------------------------------------------
            [(0, 0), 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(0, 1/2), -98304*q - 10747904*q^2 - 432144384*q^3 - 10122952704*q^4 - 166601228288*q^5 - 2126011957248*q^6 - 22328496095232*q^7 - 200745446014976*q^8 - 1588220107653120*q^9 + O(q^10)]
            [(1/2, 0), q^-1 - 24 + 98580*q + 10745856*q^2 + 432155586*q^3 + 10122903552*q^4 + 166601412312*q^5 + 2126011342848*q^6 + 22328497976703*q^7 + 200745440641024*q^8 + 1588220122131300*q^9 + O(q^10)]
            [(1/2, 1/2), -4096*q^(1/2) - 1228800*q^(3/2) - 74244096*q^(5/2) - 2204860416*q^(7/2) - 42602483712*q^(9/2) - 611708977152*q^(11/2) - 7039930359808*q^(13/2) - 68131864608768*q^(15/2) - 572940027371520*q^(17/2) - 4286110556078080*q^(19/2) + O(q^(21/2))]
            ------------------------------------------------------------
            [(0, 0), 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(0, 1/2), q^-1 + 98580*q + 10745856*q^2 + 432155586*q^3 + 10122903552*q^4 + 166601412312*q^5 + 2126011342848*q^6 + 22328497976703*q^7 + 200745440641024*q^8 + 1588220122131300*q^9 + O(q^10)]
            [(1/2, 0), -24 - 98304*q - 10747904*q^2 - 432144384*q^3 - 10122952704*q^4 - 166601228288*q^5 - 2126011957248*q^6 - 22328496095232*q^7 - 200745446014976*q^8 - 1588220107653120*q^9 + O(q^10)]
            [(1/2, 1/2), -4096*q^(1/2) - 1228800*q^(3/2) - 74244096*q^(5/2) - 2204860416*q^(7/2) - 42602483712*q^(9/2) - 611708977152*q^(11/2) - 7039930359808*q^(13/2) - 68131864608768*q^(15/2) - 572940027371520*q^(17/2) - 4286110556078080*q^(19/2) + O(q^(21/2))]
            ------------------------------------------------------------
            [(0, 0), q^-1 + 98580*q + 10745856*q^2 + 432155586*q^3 + 10122903552*q^4 + 166601412312*q^5 + 2126011342848*q^6 + 22328497976703*q^7 + 200745440641024*q^8 + 1588220122131300*q^9 + O(q^10)]
            [(0, 1/2), 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(1/2, 0), 24 + 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(1/2, 1/2), 4096*q^(1/2) + 1228800*q^(3/2) + 74244096*q^(5/2) + 2204860416*q^(7/2) + 42602483712*q^(9/2) + 611708977152*q^(11/2) + 7039930359808*q^(13/2) + 68131864608768*q^(15/2) + 572940027371520*q^(17/2) + 4286110556078080*q^(19/2) + O(q^(21/2))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[2,1],[1,2]])).nearly_holomorphic_modular_forms_basis(0, 1, 10)
            [(0, 0), O(q^10)]
            [(2/3, 2/3), q^(-1/3) + 248*q^(2/3) + 4124*q^(5/3) + 34752*q^(8/3) + 213126*q^(11/3) + 1057504*q^(14/3) + 4530744*q^(17/3) + 17333248*q^(20/3) + 60655377*q^(23/3) + 197230000*q^(26/3) + 603096260*q^(29/3) + O(q^(32/3))]
            [(1/3, 1/3), -q^(-1/3) - 248*q^(2/3) - 4124*q^(5/3) - 34752*q^(8/3) - 213126*q^(11/3) - 1057504*q^(14/3) - 4530744*q^(17/3) - 17333248*q^(20/3) - 60655377*q^(23/3) - 197230000*q^(26/3) - 603096260*q^(29/3) + O(q^(32/3))]

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[4]])).nearly_holomorphic_modular_forms_basis(5/2, 1, 7)
            [(0), O(q^7)]
            [(1/4), q^(-1/8) + 243*q^(7/8) + 2889*q^(15/8) + 15382*q^(23/8) + 62451*q^(31/8) + 203148*q^(39/8) + 593021*q^(47/8) + 1551069*q^(55/8) + O(q^(63/8))]
            [(1/2), O(q^(15/2))]
            [(3/4), -q^(-1/8) - 243*q^(7/8) - 2889*q^(15/8) - 15382*q^(23/8) - 62451*q^(31/8) - 203148*q^(39/8) - 593021*q^(47/8) - 1551069*q^(55/8) + O(q^(63/8))]

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[-2]]))
            sage: w.nearly_holomorphic_modular_forms_basis(1/2, 3, 10)
            [(0), 1 + 2*q + 2*q^4 + 2*q^9 + O(q^11)]
            [(1/2), 2*q^(1/4) + 2*q^(9/4) + 2*q^(25/4) + O(q^(45/4))]
            ------------------------------------------------------------
            [(0), 26752*q + 1707264*q^2 + 44330496*q^3 + 708938752*q^4 + 8277534720*q^5 + 77092288000*q^6 + 604139268096*q^7 + 4125992712192*q^8 + 25168873498752*q^9 + 139625296473600*q^10 + O(q^11)]
            [(1/2), q^(-3/4) - 248*q^(1/4) - 85995*q^(5/4) - 4096248*q^(9/4) - 91951146*q^(13/4) - 1343913984*q^(17/4) - 14733025125*q^(21/4) - 130880766200*q^(25/4) - 988226335125*q^(29/4) - 6548115718144*q^(33/4) - 38948971203675*q^(37/4) - 211482206208000*q^(41/4) + O(q^(45/4))]
            ------------------------------------------------------------
            [(0), q^-1 + 143376*q + 18473000*q^2 + 818626500*q^3 + 20556578688*q^4 + 357139677440*q^5 + 4764286992816*q^6 + 51954490735875*q^7 + 482593381088000*q^8 + 3929750661380112*q^9 + 28649527223209200*q^10 + O(q^11)]
            [(1/2), 492*q^(1/4) + 565760*q^(5/4) + 51180012*q^(9/4) + 1912896000*q^(13/4) + 43222528000*q^(17/4) + 697599931392*q^(21/4) + 8806299845100*q^(25/4) + 91956846489088*q^(29/4) + 824582094336000*q^(33/4) + 6520094118720000*q^(37/4) + 46360773296627712*q^(41/4) + O(q^(45/4))]
            ------------------------------------------------------------
            [(0), 8288256*q + 5734772736*q^2 + 922836934656*q^3 + 68729335136256*q^4 + 3111743030394880*q^5 + 98680901501952000*q^6 + 2386107127409246208*q^7 + 46509370523454046208*q^8 + 759717116804833376256*q^9 + 10699245000803629670400*q^10 + 132733789550418603008000*q^11 + O(q^12)]
            [(1/2), q^(-7/4) - 4119*q^(1/4) - 52756480*q^(5/4) - 22505066244*q^(9/4) - 2873089916928*q^(13/4) - 185508750165739*q^(17/4) - 7604567359488000*q^(21/4) - 223888934996798550*q^(25/4) - 5107069401161728000*q^(29/4) - 94944169224036717354*q^(33/4) - 1490934365617426071552*q^(37/4) - 20305295688695624366375*q^(41/4) - 244721335271284263813120*q^(45/4) + O(q^(49/4))]
            ------------------------------------------------------------
            [(0), q^-2 + 26124256*q + 29071392966*q^2 + 6737719296672*q^3 + 682490104576256*q^4 + 40516549954629120*q^5 + 1641525762880154250*q^6 + 49720608503585987968*q^7 + 1195216754150697417216*q^8 + 23772960843078093836256*q^9 + 403344553606117980671100*q^10 + 5973695362216702643292000*q^11 + O(q^12)]
            [(1/2), 7256*q^(1/4) + 190356480*q^(5/4) + 125891591256*q^(9/4) + 22750533217280*q^(13/4) + 1976975628705792*q^(17/4) + 105494320850688000*q^(21/4) + 3946621101455219800*q^(25/4) + 112295534019596928000*q^(29/4) + 2565871033971612229632*q^(33/4) + 48923498171391496059904*q^(37/4) + 800782892347774482432000*q^(41/4) + 11497804660218756445701120*q^(45/4) + O(q^(49/4))]
            ------------------------------------------------------------
            [(0), 561346944*q + 2225561184000*q^2 + 1367414747712000*q^3 + 315092279732380672*q^4 + 38588394037390571520*q^5 + 3008750483804157633024*q^6 + 166390610901572307712000*q^7 + 7004778717830225359104000*q^8 + 235828007010736097976072576*q^9 + 6582181403795814354977753600*q^10 + O(q^11)]
            [(1/2), q^(-11/4) - 33512*q^(1/4) - 5874905295*q^(5/4) - 12538686997224*q^(9/4) - 5735833218391375*q^(13/4) - 1102406000357376000*q^(17/4) - 119018814818782615506*q^(21/4) - 8446021610391873684200*q^(25/4) - 433852271199314133818652*q^(29/4) - 17202986580356952256512000*q^(33/4) - 551006240536440788537208750*q^(37/4) + O(q^(41/4))]
            ------------------------------------------------------------
            [(0), q^-3 + 1417904008*q + 8251987131648*q^2 + 6806527806672384*q^3 + 2010450259344635008*q^4 + 306416810529400510485*q^5 + 29115811870352951744000*q^6 + 1931312964722362632972288*q^7 + 96301285200298420431291648*q^8 + 3800832540589599304293040008*q^9 + O(q^10)]
            [(1/2), 53008*q^(1/4) + 16555069440*q^(5/4) + 50337738805008*q^(9/4) + 30484812119150592*q^(13/4) + 7446765882399621120*q^(17/4) + 994703198461400064000*q^(21/4) + 85648872591829475184400*q^(25/4) + 5259105072740190518784000*q^(29/4) + 246314320014783801021136896*q^(33/4) + 9227686751471465432298344448*q^(37/4) + 286377961374919807536439296000*q^(41/4) + O(q^(45/4))]
        """
        if verbose:
            print('I am now looking for modular forms of weight %s which are holomorphic on H and have a pole of order at most %s in infinity.' %(k, pole_order))
        try:
            k = Integer(k)
        except TypeError:
            k = QQ(k)
        sturm_bound = k/12
        prec = max(prec, sturm_bound)
        dual_sturm_bound = Integer(1)/Integer(6) - sturm_bound
        symm = self.is_symmetric_weight(k)
        if symm is None:
            return []
        if pole_order >= dual_sturm_bound + 2:
            if verbose:
                print('The pole order is large so I will compute modular forms with a smaller pole order and multiply them by the j-invariant.')
            j_order = floor(pole_order - dual_sturm_bound - 1)
            new_pole_order = pole_order - j_order
            X = self.nearly_holomorphic_modular_forms_basis(k, new_pole_order, prec = prec + j_order + 1, inclusive = inclusive, reverse = reverse, force_N_positive = force_N_positive, symmetry_data = symmetry_data, verbose = verbose)
            j = j_invariant_qexp(prec + j_order + 1) - 744
            j = [smf(0, j ** n) for n in range(1, j_order + 1)]
            Y = copy(X)
            Y.extend(WeilRepModularFormsBasis(k, [x * j[n] for n in range(j_order) for x in X], self))
            for y in Y:
                y.reduce_precision(prec)
            Y.echelonize(starting_from = -pole_order)
            if reverse:
                Y.reverse()
            return Y
        ceil_pole_order = ceil(pole_order)
        computed_weight = k + 12 * ceil_pole_order
        N = ceil_pole_order
        while computed_weight < sage_seven_half or (symm and computed_weight < sage_five_half):
            computed_weight += 12
            N += 1
        if force_N_positive and N <= pole_order:
            N += 1
            computed_weight += 12
        if verbose:
            print('I will compute modular forms of weight %s which vanish in infinity to order %s and divide them by Delta^%d.' %(computed_weight, N - pole_order, N))
        X = self.basis_vanishing_to_order(computed_weight, N - pole_order, prec + N + 1, not inclusive, keep_N = True, symmetry_data = symmetry_data, verbose = verbose)
        delta_power = smf(-12 * N, ~(delta_qexp(max(ceil(prec) + N + 1, 1)) ** N))
        Y = WeilRepModularFormsBasis(k, [(x * delta_power) for x in X], self)
        if verbose:
            print('I am computing an echelon form.')
        Y.echelonize(starting_from = -N, ending_with = sturm_bound)
        if reverse:
            Y.reverse()
        return Y.reduce_precision(prec)

    weakly_holomorphic_modular_forms_basis = nearly_holomorphic_modular_forms_basis

    def invariant_cusp_forms_basis(self, k, prec = 0, G = None, chi = None, verbose = False):
        r"""
        Compute a basis of cusp forms with specified symmetries.

        Let G be a WeilRepAutomorphismGroup (typically the output of self.automorphism_group()) and chi a 'character' -- a list of +/-1 of length equal to len(G) such that chi[i] * chi[j] = chi[k] whenever G[i] * G[j] = G[k]. This method computes a basis of the space of cusp forms f of weight 'k' for which G[i](f) = chi[i] * f for i = 1,...,|G|.

        NOTE: the character 'chi' must satisfy chi(-1) = (-1)^\kappa, where \kappa = k + signature / 2

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- precision
        - ``G`` -- a WeilRepAutomorphismGroup. Default 'None'; if 'None' then we use the full automorphism group.
        - ``chi`` -- a character, i.e. a list of +/-1 for which chi[i] * chi[j] = chi[k] whenever G[i] * G[j] = G[k]. Typically constructed using G.characters() Default: 'None'; if 'None' then we use the trivial character.
        - ``verbose`` -- boolean (default False) if True then add comments.

        EXAMPLES::

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[4, 0], [0, 4]]))
            sage: w.invariant_cusp_forms_basis(7, 5)
            [(0, 0), 32*q - 128*q^2 - 256*q^3 + 1536*q^4 + O(q^5)]
            [(1/4, 0), -10*q^(7/8) + 10*q^(15/8) + 338*q^(23/8) - 1352*q^(31/8) + 1466*q^(39/8) + O(q^(47/8))]
            [(1/2, 0), O(q^(11/2))]
            [(3/4, 0), -10*q^(7/8) + 10*q^(15/8) + 338*q^(23/8) - 1352*q^(31/8) + 1466*q^(39/8) + O(q^(47/8))]
            [(0, 1/4), -10*q^(7/8) + 10*q^(15/8) + 338*q^(23/8) - 1352*q^(31/8) + 1466*q^(39/8) + O(q^(47/8))]
            [(1/4, 1/4), O(q^(23/4))]
            [(1/2, 1/4), q^(3/8) + 31*q^(11/8) - 243*q^(19/8) + 498*q^(27/8) - 100*q^(35/8) + O(q^(43/8))]
            [(3/4, 1/4), O(q^(23/4))]
            [(0, 1/2), O(q^(11/2))]
            [(1/4, 1/2), q^(3/8) + 31*q^(11/8) - 243*q^(19/8) + 498*q^(27/8) - 100*q^(35/8) + O(q^(43/8))]
            [(1/2, 1/2), -32*q + 128*q^2 + 256*q^3 - 1536*q^4 + O(q^5)]
            [(3/4, 1/2), q^(3/8) + 31*q^(11/8) - 243*q^(19/8) + 498*q^(27/8) - 100*q^(35/8) + O(q^(43/8))]
            [(0, 3/4), -10*q^(7/8) + 10*q^(15/8) + 338*q^(23/8) - 1352*q^(31/8) + 1466*q^(39/8) + O(q^(47/8))]
            [(1/4, 3/4), O(q^(23/4))]
            [(1/2, 3/4), q^(3/8) + 31*q^(11/8) - 243*q^(19/8) + 498*q^(27/8) - 100*q^(35/8) + O(q^(43/8))]
            [(3/4, 3/4), O(q^(23/4))]
            ------------------------------------------------------------
            [(0, 0), 4*q - 48*q^2 + 224*q^3 - 448*q^4 + O(q^5)]
            [(1/4, 0), O(q^(47/8))]
            [(1/2, 0), q^(1/2) - 8*q^(3/2) + 10*q^(5/2) + 80*q^(7/2) - 231*q^(9/2) + O(q^(11/2))]
            [(3/4, 0), O(q^(47/8))]
            [(0, 1/4), O(q^(47/8))]
            [(1/4, 1/4), -2*q^(3/4) + 20*q^(7/4) - 62*q^(11/4) - 20*q^(15/4) + 486*q^(19/4) + O(q^(23/4))]
            [(1/2, 1/4), O(q^(43/8))]
            [(3/4, 1/4), -2*q^(3/4) + 20*q^(7/4) - 62*q^(11/4) - 20*q^(15/4) + 486*q^(19/4) + O(q^(23/4))]
            [(0, 1/2), q^(1/2) - 8*q^(3/2) + 10*q^(5/2) + 80*q^(7/2) - 231*q^(9/2) + O(q^(11/2))]
            [(1/4, 1/2), O(q^(43/8))]
            [(1/2, 1/2), 4*q - 48*q^2 + 224*q^3 - 448*q^4 + O(q^5)]
            [(3/4, 1/2), O(q^(43/8))]
            [(0, 3/4), O(q^(47/8))]
            [(1/4, 3/4), -2*q^(3/4) + 20*q^(7/4) - 62*q^(11/4) - 20*q^(15/4) + 486*q^(19/4) + O(q^(23/4))]
            [(1/2, 3/4), O(q^(43/8))]
            [(3/4, 3/4), -2*q^(3/4) + 20*q^(7/4) - 62*q^(11/4) - 20*q^(15/4) + 486*q^(19/4) + O(q^(23/4))]

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[-12]]))
            sage: w.invariant_cusp_forms_basis(1/2, 5, chi = w.automorphism_group().characters()[1])
            [(0), O(q^5)]
            [(11/12), q^(1/24) - q^(25/24) - q^(49/24) + O(q^(121/24))]
            [(5/6), O(q^(31/6))]
            [(3/4), O(q^(43/8))]
            [(2/3), O(q^(17/3))]
            [(7/12), -q^(1/24) + q^(25/24) + q^(49/24) + O(q^(121/24))]
            [(1/2), O(q^(11/2))]
            [(5/12), -q^(1/24) + q^(25/24) + q^(49/24) + O(q^(121/24))]
            [(1/3), O(q^(17/3))]
            [(1/4), O(q^(43/8))]
            [(1/6), O(q^(31/6))]
            [(1/12), q^(1/24) - q^(25/24) - q^(49/24) + O(q^(121/24))]
        """
        try:
            k = Integer(k)
        except TypeError:
            k = QQ(k)
        symm = self.is_symmetric_weight(k)
        sturm_bound = k / 12
        if not prec:
            prec = ceil(sturm_bound)
        else:
            prec = ceil(max(prec, sturm_bound))
        if k <= 0:
            return WeilRepModularFormsBasis(k, [], self)
        elif k >= sage_seven_half or (symm and k >= sage_five_half):
            if verbose:
                print('I am computing the dimension...')
            dim = self.invariant_forms_dimension(k, cusp_forms = True, G = G, chi = chi)
            if G is None:
                G = self.automorphism_group()
            if chi is None:
                chi = [1] * len(G)
            if verbose:
                print('I need to find %d cusp forms.'%dim)
            X = WeilRepModularFormsBasis(k, [], self)
            if not dim:
                return X
            dsdict = self.ds_dict()
            rds = self.sorted_rds()
            indices = self.rds(indices = True)
            orbits = []
            e = set([])
            n = self.norm_dict()
            n0 = []
            for i, b in enumerate(rds):
                if b not in e:
                    vb = vector(b)
                    x = [g(vb) for g in G]
                    d = {}
                    for i, s in enumerate(chi):
                        y = x[i]
                        if indices[dsdict[tuple(y)]] is None:
                            eps = 1 + bool(2 % denominator(y))
                            try:
                                d[tuple(y)] += s * eps
                            except KeyError:
                                d[tuple(y)] = s * eps
                    for x in x:
                        e.add(tuple(x))
                    orbits.append(d)
                    n0.append(n[b])
            r = 0
            m = 1
            if symm:
                e = self.eisenstein_series(k, prec)
            while r < dim:
                for i, x in enumerate(orbits):
                    a = n0[i]
                    f = self.zero(k, prec)
                    for y, s in x.items():
                        if s:
                            if symm:
                                f += s * (self.pss(k, vector(y), m+a, prec) - e)
                            else:
                                f += s * self.pssd(k, vector(y), m+a, prec)
                    if verbose:
                        print('I computed several Poincare square series of index %s.'%(m+a))
                    if f:
                        X.append(f)
                    if len(X) >= dim:
                        r = X.rank()
                        if r == dim:
                            break
                m += 1
            if verbose:
                print('I am computing an echelon form.')
            X.echelonize()
            return X
        else:
            if verbose:
                print('I am going to compute the spaces of invariant cusp forms of weights %s and %s.' %(k+4, k+6))
            prec = max([2, prec, ceil(sturm_bound + sage_one_half)])
            e4 = smf(-4, ~eisenstein_series_qexp(4, prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6, prec))
            X1 = self.invariant_cusp_forms_basis(k + 4, prec, G=G, chi=chi, verbose = verbose)
            X2 = self.invariant_cusp_forms_basis(k + 6, prec, G=G, chi=chi, verbose = verbose)
            if verbose:
                print('I will intersect the spaces E_4^(-1) * S_%s and E_6^(-1) * S_%s.' %(k +4, k +6))
            try:
                V1 = span((x * e4).coefficient_vector() for x in X1)
                V2 = span((x * e6).coefficient_vector() for x in X2)
                X = WeilRepModularFormsBasis(k, [self.recover_modular_form_from_coefficient_vector(k, v, prec) for v in V1.intersection(V2).basis()], self)
                return X
            except AttributeError:
                return WeilRepModularFormsBasis(k, [], self)

    def invariant_forms_basis(self, k, prec = 0, G = None, chi = None, verbose = False):
        r"""
        Compute a basis of cusp forms with specified symmetries.

        Let G be a WeilRepAutomorphismGroup (typically the output of self.automorphism_group()) and chi a 'character' -- a list of +/-1 of length equal to len(G) such that chi[i] * chi[j] = chi[k] whenever G[i] * G[j] = G[k]. This method computes a basis of the space of cusp forms f of weight 'k' for which G[i](f) = chi[i] * f for i = 1,...,|G|.

        NOTE: the character 'chi' must satisfy chi(-1) = (-1)^\kappa, where \kappa = k + signature / 2

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- precision
        - ``G`` -- a WeilRepAutomorphismGroup. Default 'None'; if 'None' then we use the full automorphism group.
        - ``chi`` -- a character, i.e. a list of +/-1 for which chi[i] * chi[j] = chi[k] whenever G[i] * G[j] = G[k]. Typically constructed using G.characters() Default: 'None'; if 'None' then we use the trivial character.
        - ``verbose`` -- boolean (default False) if True then add comments.

        EXAMPLES::

            sage: from weilrep import *
            sage: w = WeilRep(CartanMatrix(['D', 4]))
            sage: w.invariant_forms_basis(6, 5)
            [(0, 0, 0, 0), 1 + 264*q + 7944*q^2 + 64416*q^3 + 253704*q^4 + O(q^5)]
            [(0, 0, 1/2, 1/2), 8*q^(1/2) + 1952*q^(3/2) + 25008*q^(5/2) + 134464*q^(7/2) + 474344*q^(9/2) + O(q^(11/2))]
            [(1/2, 0, 0, 1/2), 8*q^(1/2) + 1952*q^(3/2) + 25008*q^(5/2) + 134464*q^(7/2) + 474344*q^(9/2) + O(q^(11/2))]
            [(1/2, 0, 1/2, 0), 8*q^(1/2) + 1952*q^(3/2) + 25008*q^(5/2) + 134464*q^(7/2) + 474344*q^(9/2) + O(q^(11/2))]
        """
        try:
            k = Integer(k)
        except TypeError:
            k = QQ(k)
        prec = ceil(max(prec, k / 12))
        symm = self.is_symmetric_weight(k)
        if symm is None:
            return []
        elif k < 0:
            return WeilRepModularFormsBasis(k, [], self)
        ds = self.ds()
        dsdict = self.ds_dict()
        indices = self.rds(indices = True)
        n = self.norm_list()
        b = [b for i, b in enumerate(ds) if not n[i] and indices[i] is None]
        if G is None:
            G = self.automorphism_group()
        if chi is None:
            chi = [1] * len(G)
        if any(denominator(x) not in [1, 2, 3, 4, 6] for x in b):
            if verbose:
                print('I do not know how to find enough Eisenstein series. I am going to compute the image of M_%s under multiplication by Delta.')
            X = self.nearly_holomorphic_modular_forms_basis(k, 0, prec, inclusive = True, reverse = False, force_N_positive = True, symmetry_data = [G, chi], verbose = verbose)
        elif k >= sage_seven_half or (symm and k >= sage_five_half):
            X = self.invariant_cusp_forms_basis(k, prec = prec, G = G, chi = chi, verbose = verbose)
            e = set([])
            orbits = []
            for i, b in enumerate(b):
                tb = tuple(b)
                if tb not in e:
                    x = [g(b) for g in G]
                    d = {}
                    for i, s in enumerate(chi):
                        y = x[i]
                        if indices[dsdict[tuple(y)]] is None:
                            eps = 1 + bool(2 % denominator(y))
                            try:
                                d[tuple(y)] += s * eps
                            except KeyError:
                                d[tuple(y)] = s * eps
                    for x in x:
                        e.add(tuple(x))
                    orbits.append(d)
            for x in orbits:
                f = self.zero(k, prec)
                for y, s in x.items():
                    if s:
                        f += s * self.eisenstein_oldform(k, vector(y), prec)
                X.append(f)
            X.echelonize()
            return X
        else:
            prec = max(prec, 2)
            e4 = smf(-4, ~eisenstein_series_qexp(4, prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6, prec))
            X1 = self.invariant_forms_basis(k+4, prec, G = G, chi = chi, verbose = verbose)
            X2 = self.invariant_forms_basis(k+6, prec, G = G, chi = chi, verbose = verbose)
            if verbose:
                print('I am now going to compute M_%s by intersecting the spaces E_4^(-1) * M_%s and E_6^(-1) * M_%s.' %(k, k +4, k +6))
            try:
                V1 = span([(x * e4).coefficient_vector() for x in X1])
                V2 = span([(x * e6).coefficient_vector() for x in X2])
                V = (V1.intersection(V2)).basis()
                X = WeilRepModularFormsBasis(k, [self.recover_modular_form_from_coefficient_vector(k, v, prec) for v in V], self)
                X.echelonize()
                return X
            except AttributeError:
                X = WeilRepModularFormsBasis(k, [], self)
                return X

    def borcherds_obstructions(self, weight, prec, reverse = True, verbose = False):
        r"""
        Compute a basis of the Borcherds obstruction space.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: WeilRep(matrix([[-8]])).borcherds_obstructions(5/2, 5)
            [(0), 1 - 24*q - 72*q^2 - 96*q^3 - 358*q^4 + O(q^5)]
            [(7/8), -1/2*q^(1/16) - 24*q^(17/16) - 72*q^(33/16) - 337/2*q^(49/16) - 192*q^(65/16) + O(q^(81/16))]
            [(3/4), -5*q^(1/4) - 24*q^(5/4) - 125*q^(9/4) - 120*q^(13/4) - 240*q^(17/4) + O(q^(21/4))]
            [(5/8), -25/2*q^(9/16) - 121/2*q^(25/16) - 96*q^(41/16) - 168*q^(57/16) - 264*q^(73/16) + O(q^(89/16))]
            [(1/2), -46*q - 48*q^2 - 144*q^3 - 192*q^4 + O(q^5)]
            [(3/8), -25/2*q^(9/16) - 121/2*q^(25/16) - 96*q^(41/16) - 168*q^(57/16) - 264*q^(73/16) + O(q^(89/16))]
            [(1/4), -5*q^(1/4) - 24*q^(5/4) - 125*q^(9/4) - 120*q^(13/4) - 240*q^(17/4) + O(q^(21/4))]
            [(1/8), -1/2*q^(1/16) - 24*q^(17/16) - 72*q^(33/16) - 337/2*q^(49/16) - 192*q^(65/16) + O(q^(81/16))]
        """
        prec = ceil(prec)
        d = self.discriminant()
        if weight > 2 or (weight == 2 and ((d % 4 and d.is_squarefree()) or (d//4).is_squarefree())):
            if verbose:
                print('I am looking for obstructions to Borcherds products of weight %s.' %weight)
            E = self.eisenstein_series(weight, prec)
            if verbose:
                print('I computed the Eisenstein series and will now compute cusp forms.')
            L = [E]
            L.extend(self.cusp_forms_basis(weight, prec, E = E, verbose = verbose))
            return WeilRepModularFormsBasis(weight, L, self)
        elif weight == 0:
            if d == 1:
                return self._invariants(prec)
            return []
        elif weight == sage_one_half:
            X = self._weight_one_half_basis(prec)
            n = self.norm_dict()
            return WeilRepModularFormsBasis(sage_one_half, [x for x in X if not any(h[0] and h[2][0] for h in x.fourier_expansion())], self)
        else:
            if verbose:
                print('I am going to compute the obstruction spaces in weights %s and %s.' %(weight+4, weight+6))
            e4 = smf(-4, ~eisenstein_series_qexp(4,prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6,prec))
            X1 = self.borcherds_obstructions(weight+4, prec, verbose = verbose)
            X2 = self.borcherds_obstructions(weight+6, prec, verbose = verbose)
            try:
                V1 = span([(x * e4).coefficient_vector() for x in X1])
                V2 = span([(x * e6).coefficient_vector() for x in X2])
                V = (V1.intersection(V2)).echelonized_basis()
                Y = [self.recover_modular_form_from_coefficient_vector(weight, v, prec) for v in V]
                if reverse:
                    Y.reverse()
                return WeilRepModularFormsBasis(weight, Y, self)
            except AttributeError:
                return []

    def quasimodular_forms_basis(self, k, prec, verbose = False):
        if k < 2:
            return self.modular_forms_basis(k, prec, verbose=verbose)
        e2 = WeilRep([]).eisenstein_series(2, prec)
        if verbose:
            print('I will compute a basis of quasimodular forms of weight %s.'%k)
        X = self.quasimodular_forms_basis(k - 2, prec, verbose=verbose)
        X = self.modular_forms_basis(k, prec, verbose=verbose) + WeilRepModularFormsBasis(k, [e2 * x for x in X], self, flag = 'quasimodular')
        if verbose:
            print('I am computing an echelon form.')
        X.echelonize()
        return X

    ## automorphisms ##

    def identity_morphism(self):
        r"""
        Construct the identity morphism x --> x.
        """
        return WeilRepAutomorphism(self, lambda x:x)

    def canonical_involution(self):
        r"""
        Construct the map x --> -x
        """
        return WeilRepAutomorphism(self, lambda x: vector(map(frac, -x)))

    def reflection(self, r):
        r"""
        Compute the reflection by a vector r \in L \otimes \QQ as a WeilRepAutomorphism.

        This is the morphism
        s_r : L'/L --> L'/L,  s_r(x) = x - <r, x> (r / Q(r)).
        If r has norm Q(r) = 0 or if s_r does not map L' into L' and L into L, then raise a ValueError.

        INPUT:
        - ``r`` -- a rational vector of length equal to self's lattice rank

        OUTPUT: WeilRepAutomorphism
        """
        S = self.gram_matrix()
        r0 = S * r
        try:
            r0 = 2 * r0 / (r * r0)
        except ZeroDivisionError:
            raise ValueError('Not a valid reflection') from None
        try:
            return WeilRepAutomorphism(self, lambda x: vector(map(frac, x - (r0 * x) * r)))
        except KeyError:
            raise ValueError('Not a valid reflection') from None

    def automorphism_group(self):
        r"""
        Compute all automorphisms of this WeilRep.

        If (A, Q) is the discriminant form then an automorphism is an isomorphism of groups f : A -> A with Q(f(x)) = Q(x) for all x in A.

        OUTPUT: a WeilRepAutomorphismGroup

        EXAMPLES::

            sage: from weilrep import *
            sage: w = II(3)
            sage: list(w.automorphism_group())
            [Automorphism of Weil representation associated to the Gram matrix
            [0 3]
            [3 0]
            mapping (0, 0)->(0, 0), (0, 1/3)->(0, 1/3), (0, 2/3)->(0, 2/3), (1/3, 0)->(1/3, 0), (1/3, 1/3)->(1/3, 1/3), (1/3, 2/3)->(1/3, 2/3), (2/3, 0)->(2/3, 0), (2/3, 1/3)->(2/3, 1/3), (2/3, 2/3)->(2/3, 2/3), Automorphism of Weil representation associated to the Gram matrix
            [0 3]
            [3 0]
            mapping (0, 0)->(0, 0), (0, 1/3)->(1/3, 0), (0, 2/3)->(2/3, 0), (1/3, 0)->(0, 1/3), (1/3, 1/3)->(1/3, 1/3), (1/3, 2/3)->(2/3, 1/3), (2/3, 0)->(0, 2/3), (2/3, 1/3)->(1/3, 2/3), (2/3, 2/3)->(2/3, 2/3), Automorphism of Weil representation associated to the Gram matrix
            [0 3]
            [3 0]
            mapping (0, 0)->(0, 0), (0, 1/3)->(0, 2/3), (0, 2/3)->(0, 1/3), (1/3, 0)->(2/3, 0), (1/3, 1/3)->(2/3, 2/3), (1/3, 2/3)->(2/3, 1/3), (2/3, 0)->(1/3, 0), (2/3, 1/3)->(1/3, 2/3), (2/3, 2/3)->(1/3, 1/3), Automorphism of Weil representation associated to the Gram matrix
            [0 3]
            [3 0]
            mapping (0, 0)->(0, 0), (0, 1/3)->(2/3, 0), (0, 2/3)->(1/3, 0), (1/3, 0)->(0, 2/3), (1/3, 1/3)->(2/3, 2/3), (1/3, 2/3)->(1/3, 2/3), (2/3, 0)->(0, 1/3), (2/3, 1/3)->(2/3, 1/3), (2/3, 2/3)->(1/3, 1/3)]

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[6, 3], [3, 4]]))
            sage: w.automorphism_group()
            Automorphism group of Weil representation associated to the Gram matrix
            [6 3]
            [3 4]
        """
        try:
            return self.__automorphism_group
        except AttributeError:
            pass
        S = self.gram_matrix().inverse()
        S1, d = S._clear_denom()
        _, U, V = S1.smith_form()
        D = U * S * V
        q = FreeQuadraticModule(ZZ, S.nrows(), inner_product_matrix = (d * d) * S)
        denoms = [x.denominator() for x in D.diagonal()]
        r = q.span(diagonal_matrix(ZZ, denoms) * U)
        v = q / r
        Z = matrix([g.lift() for g in v.gens()])
        n = S.nrows()
        i = sum(1 for x in D.diagonal() if x == 1)
        if i == n:
            return [self.identity_morphism()]
        I = identity_matrix(i)
        S = self.gram_matrix()
        if i:
            A = matrix(ZZ, (S * ((Z * S1).transpose().echelon_form(transformation = True)[1].inverse())).columns()[Z.nrows():])
            Z = block_matrix([[A], [Z]])
        Z_inv = Z.inverse()
        def a(g):
            g = Z_inv * block_diagonal_matrix([I, g.matrix()]) * Z
            return lambda x: vector(map(frac, g * x))
        G = self.discriminant_form().orthogonal_group()
        X = G.conjugacy_classes()
        G = WeilRepAutomorphismGroup(self, [WeilRepAutomorphism(self, a(g)) for x in X for g in x], G)
        self.__automorphism_group = G
        return G

    ## low weight ##

    def _s_matrix(self):
        r"""
        Auxiliary function for Weil invariants.

        There is probably no reason to call this directly.

        OUTPUT: a tuple (s, v) where:
        - ``s`` -- an integer matrix
        - ``v`` -- a list of tuples (i, g) where `g` is an isotropic vector and `i` is its index in self.ds()
        """
        try:
            return self.__s_matrix
        except AttributeError:
            pass
        @cached_function
        def local_moebius(n):
            return moebius(n)
        @cached_function
        def local_phi(n):
            return euler_phi(n)
        def tr_sqrt_f_zeta_N(f, N, n): #trace of sqrt(f) * zeta_N
            if f == 1:
                d = N // GCD(N, n)
                return local_moebius(d) * (phi // local_phi(d))
            h, r = N.quo_rem(4 * f)
            if r:
                return 0
            s = 0
            if f % 4 == 1:
                N4 = 0
            else:
                N4 = N // 4
            if f % 2:
                for ell in range(abs(f)):
                    u = GCD(n - N4 + 4 * h * ell* ell, N)
                    d = N//u
                    s += local_moebius(d) * (phi // local_phi(d))
                return s * sgn(f)
            else:
                for ell in range(abs(4 * f)):
                    u1, u2 = GCD(n + h * ell * ell, N), GCD(n - N4 + h * ell * ell, N)
                    d1, d2 = N//u1, N//u2
                    s += (local_moebius(d1) * (phi // local_phi(d1)) + local_moebius(d2) * (phi // local_phi(d2)))
                return s * sgn(f) // 4
        n = self.norm_list()
        s = self.signature()
        eps = 1
        j = 1
        if s == 4 or s == 6:
            j = -1
        S = self.gram_matrix()
        indices = self.rds(indices = True)
        isotropic_vectors = [x for x in enumerate(self.ds()) if not n[x[0]] and indices[x[0]] is None]
        m = len(isotropic_vectors)
        N = denominator(S.inverse())
        if s % 4:
            eps = -1
        D = self.discriminant()
        f = D.squarefree_part()
        Dsqr, f = j * isqrt(D // f), f * eps
        while N % (4 * f):
            N += N
        phi = local_phi(N)
        M = []
        for i, b in isotropic_vectors:
            Sb = N * (S * b)
            L = [[0]*m for _ in range(phi)]
            for j, (jj, g) in enumerate(isotropic_vectors):
                d = 2 % denominator(g)
                gSb = Integer(g * Sb) % N
                for k in srange(phi):
                    kg = k - gSb
                    u = GCD(kg, N)
                    h = N // u
                    z = (phi // local_phi(h))
                    z1 = z * local_moebius(h)
                    L[k][j] += z1
                    if i == jj:
                        L[k][j] -= Dsqr * tr_sqrt_f_zeta_N(f, N, kg)
                    if d:
                        kg = k + gSb
                        h = N // GCD(kg, N)
                        z = (phi // local_phi(h))
                        z1 = z * local_moebius(h)
                        L[k][j] += eps * z1
            M.extend(filter(any, L))
        self.__s_matrix = matrix(M), isotropic_vectors
        return self.__s_matrix

    def _invariants(self, prec = 1):
        r"""
        Compute invariants of the Weil representation.

        This computes a basis of invariants under the Weil representation or equivalently modular forms of weight 0. It is called from self.modular_forms_basis(k, prec) when k = 0.

        NOTE: it does not matter which convention we use here (dual vs nondual) as the invariants of \rho and its dual are the same. (N.B. we always use the 'dual' convention)

        ALGORITHM: it is based on Ehlen--Skoruppa's algorithm [ES]

        INPUT:
        - ``prec`` -- the precision (default 1)

        OUTPUT:
        WeilRepModularFormsBasis

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[2, 0], [0, -2]]))
            sage: w._invariants(5)
            [(0, 0), 1 + O(q^5)]
            [(1/2, 0), O(q^(23/4))]
            [(0, 1/2), O(q^(21/4))]
            [(1/2, 1/2), 1 + O(q^5)]

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[2, 1], [1, 2]]))
            sage: (w + II(3))._invariants(5)
            [(0, 0, 0, 0), O(q^5)]
            [(0, 0, 0, 1/3), 1 + O(q^5)]
            [(0, 0, 0, 2/3), -1 + O(q^5)]
            [(0, 2/3, 2/3, 0), O(q^(17/3))]
            [(0, 2/3, 2/3, 1/3), O(q^(17/3))]
            [(0, 2/3, 2/3, 2/3), O(q^(17/3))]
            [(0, 1/3, 1/3, 0), O(q^(17/3))]
            [(0, 1/3, 1/3, 1/3), O(q^(17/3))]
            [(0, 1/3, 1/3, 2/3), O(q^(17/3))]
            [(1/3, 0, 0, 0), -1 + O(q^5)]
            [(1/3, 0, 0, 1/3), O(q^(17/3))]
            [(1/3, 0, 0, 2/3), O(q^(16/3))]
            [(1/3, 2/3, 2/3, 0), O(q^(17/3))]
            [(1/3, 2/3, 2/3, 1/3), O(q^(16/3))]
            [(1/3, 2/3, 2/3, 2/3), -1 + O(q^5)]
            [(1/3, 1/3, 1/3, 0), O(q^(17/3))]
            [(1/3, 1/3, 1/3, 1/3), O(q^(16/3))]
            [(1/3, 1/3, 1/3, 2/3), -1 + O(q^5)]
            [(2/3, 0, 0, 0), 1 + O(q^5)]
            [(2/3, 0, 0, 1/3), O(q^(16/3))]
            [(2/3, 0, 0, 2/3), O(q^(17/3))]
            [(2/3, 2/3, 2/3, 0), O(q^(17/3))]
            [(2/3, 2/3, 2/3, 1/3), 1 + O(q^5)]
            [(2/3, 2/3, 2/3, 2/3), O(q^(16/3))]
            [(2/3, 1/3, 1/3, 0), O(q^(17/3))]
            [(2/3, 1/3, 1/3, 1/3), 1 + O(q^5)]
            [(2/3, 1/3, 1/3, 2/3), O(q^(16/3))]
        """
        eps = self.is_symmetric_weight(0)
        if eps == 0:
            eps = -1
        elif not(eps):
            return []
        if self.discriminant() == 1: #unimodular
            f = self.zero(0, prec)
            f._WeilRepModularForm__fourier_expansions[0][2] += 1
            return WeilRepModularFormsBasis(0, [f], self)
        s, isotropic_vectors = self._s_matrix()
        k = s.right_kernel().basis_matrix()
        indices = self.rds(indices = True)
        X = []
        for x in k.rows():
            f = self.zero(0, prec)
            for i, g in enumerate(isotropic_vectors):
                f._WeilRepModularForm__fourier_expansions[g[0]][2] += x[i]
            for i, j in filter(all, enumerate(indices)):
                f._WeilRepModularForm__fourier_expansions[i][2] = eps * f._WeilRepModularForm__fourier_expansions[j][2]
            X.append(f)
        return WeilRepModularFormsBasis(0, X, self)

    def _invariants_dim(self):
        r"""
        Compute the dimension of the space of invariants.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[2, 0], [0, -2]]))
            sage: w._invariants_dim()
            1
        """
        if self.discriminant() == 1: #unimodular
            return 1
        S, _ = self._s_matrix()
        return S.ncols() - S.rank()

    def _weight_one_half_basis(self, prec = 1):
        r"""
        Compute weight one half modular forms.

        This computes a basis of the space of weight 1/2 modular forms. It is called from self.modular_forms_basis(k, prec) when k = 1/2.

        ALGORITHM: by the Serre Stark basis theorem the theta-contractions of Weil invariants yield a basis. See Skoruppa's paper [S] for the exact statement.

        EXAMPLES::

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[-12]]))
            sage: w._weight_one_half_basis(10)
            [(0), 1 + 2*q^6 + O(q^10)]
            [(11/12), q^(25/24) + q^(49/24) + O(q^(241/24))]
            [(5/6), q^(1/6) + q^(25/6) + q^(49/6) + O(q^(61/6))]
            [(3/4), q^(3/8) + q^(27/8) + q^(75/8) + O(q^(83/8))]
            [(2/3), q^(2/3) + q^(8/3) + O(q^(32/3))]
            [(7/12), q^(1/24) + q^(121/24) + q^(169/24) + O(q^(241/24))]
            [(1/2), 2*q^(3/2) + O(q^(21/2))]
            [(5/12), q^(1/24) + q^(121/24) + q^(169/24) + O(q^(241/24))]
            [(1/3), q^(2/3) + q^(8/3) + O(q^(32/3))]
            [(1/4), q^(3/8) + q^(27/8) + q^(75/8) + O(q^(83/8))]
            [(1/6), q^(1/6) + q^(25/6) + q^(49/6) + O(q^(61/6))]
            [(1/12), q^(25/24) + q^(49/24) + O(q^(241/24))]
            ------------------------------------------------------------
            [(0), O(q^10)]
            [(11/12), q^(1/24) - q^(25/24) - q^(49/24) + q^(121/24) + q^(169/24) + O(q^(241/24))]
            [(5/6), O(q^(61/6))]
            [(3/4), O(q^(83/8))]
            [(2/3), O(q^(32/3))]
            [(7/12), -q^(1/24) + q^(25/24) + q^(49/24) - q^(121/24) - q^(169/24) + O(q^(241/24))]
            [(1/2), O(q^(21/2))]
            [(5/12), -q^(1/24) + q^(25/24) + q^(49/24) - q^(121/24) - q^(169/24) + O(q^(241/24))]
            [(1/3), O(q^(32/3))]
            [(1/4), O(q^(83/8))]
            [(1/6), O(q^(61/6))]
            [(1/12), q^(1/24) - q^(25/24) - q^(49/24) + q^(121/24) + q^(169/24) + O(q^(241/24))]
        """
        prec = max(1, ceil(prec))
        S = self.gram_matrix()
        n = S.nrows()
        X = WeilRepModularFormsBasis(sage_one_half, [], self)
        if not(n % 2) or not((n + 1) % 8 or len(set([x for x in S.elementary_divisors() if x - 1])) > 1): #skoruppa theorem 10
            return X
        N = self.level() // 4
        b = vector([0] * n)
        for d in divisors(N):
            if (N // d).is_squarefree():
                X.extend(self.embiggen(b, d)._invariants(prec).theta(weilrep = self))
        X.echelonize()
        return X