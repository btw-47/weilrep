r"""

Sage code for Jacobi forms

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2020-2021 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import cypari2
pari = cypari2.Pari()
PariError = cypari2.PariError

from collections import Counter
from itertools import product
from re import sub

from sage.arith.misc import divisors, is_square, XGCD
from sage.calculus.var import var
from sage.combinat.root_system.root_system import RootSystem
from sage.functions.other import binomial, ceil, floor, frac
from sage.matrix.constructor import matrix
from sage.matrix.special import block_diagonal_matrix, block_matrix, identity_matrix
from sage.misc.functional import denominator, isqrt
from sage.misc.latex import latex
from sage.misc.misc_c import prod
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modular.modform.element import is_ModularFormElement
from sage.modules.free_module import span
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.all import CC
from sage.rings.big_oh import O
from sage.rings.fraction_field import FractionField
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ

from .weilrep import WeilRep
from .weilrep_modular_forms_class import EtaCharacterPower, smf, WeilRepModularForm, WeilRepModularFormWithCharacter, WeilRepModularFormsBasis

sage_one_half = Integer(1) / Integer(2)
sage_three_half = Integer(3) / Integer(2)


class JacobiForms:
    r"""
    The JacobiForms class represents the graded module of Jacobi forms for some lattice index.

    INPUT:

    A JacobiForms instance is constructed by calling JacobiForms(m), where:
    - ``m`` -- a positive-definite Gram matrix (symmetric, integral, even diagonal); OR:
    - ``m`` -- a natural number (not 0)

    """
    def __init__(self, index_matrix = None, weilrep=None):
        if index_matrix in QQ:
            self.__index_matrix = matrix([[2 * index_matrix]])
        elif index_matrix:
            if isinstance(index_matrix, WeilRep):
                if not index_matrix.is_positive_definite():
                    raise ValueError('This index is not positive definite.')
                self.__index_matrix = index_matrix.gram_matrix()
            else:
                try:
                    _ = index_matrix.nrows()
                except AttributeError:
                    index_matrix = matrix(index_matrix)
                self.__index_matrix = index_matrix
        else:
            self.__index_matrix = matrix([])
        if weilrep:
            self.__weilrep = weilrep

    def __repr__(self):
        S = self.index_matrix()
        N = S.nrows()
        if N > 1:
            return 'Jacobi forms of index \n%s' % S
        else:
            return 'Jacobi forms of index %s' % self.index()

    def __call__(self, N):
        r"""
        Rescale the index by N.

        INPUT:
        - ``N`` -- a natural number

        OUTPUT: a JacobiForms
        """
        return JacobiForms(self.__index_matrix * N)

    def discriminant(self):
        r"""
        Return the discriminant of self's index.
        """
        try:
            return self.__discriminant
        except AttributeError:
            self.__discriminant = self.index_matrix().determinant()
            return self.__discriminant

    def index(self):
        r"""
        Return self's index (as a scalar if one elliptic variable).

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).index()
            1

            sage: from weilrep import *
            sage: JacobiForms(matrix([[2, 1],[1, 2]])).index()
            [2 1]
            [1 2]
        """
        S = self.index_matrix()
        if S.nrows() == 1:
            try:
                return Integer(S[0, 0] / 2)
            except TypeError:
                return QQ(S[0, 0]) / 2
        else:
            return S

    def index_matrix(self):
        r"""
        Return self's index as a matrix.

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).index_matrix()
            [2]
        """
        return self.__index_matrix

    def is_even(self):
        return not any(x % 2 for x in self.index_matrix().diagonal())

    def nvars(self):
        r"""
        Return the number of elliptic variables.

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).nvars()
            1
        """
        try:
            return self.__nvars
        except:
            self.__nvars = Integer(self.index_matrix().nrows())
            return self.__nvars

    def theta_decomposition(self):
        r"""
        Return the WeilRep associated to this Gram matrix.

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).theta_decomposition()
            Weil representation associated to the Gram matrix
            [2]
        """
        try:
            return self.__weilrep
        except AttributeError:
            from weilrep import WeilRep
            self.__weilrep = WeilRep(self.index_matrix())
            return self.__weilrep

    weilrep = theta_decomposition

    ##construction of Jacobi Forms associated to this index

    def eisenstein_series(self, k, prec, allow_small_weight=False):
        r"""
        Compute the Jacobi Eisenstein series of weight k.

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- the precision of the Fourier expansion

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).eisenstein_series(4, 5)
            1 + (w^-2 + 56*w^-1 + 126 + 56*w + w^2)*q + (126*w^-2 + 576*w^-1 + 756 + 576*w + 126*w^2)*q^2 + (56*w^-3 + 756*w^-2 + 1512*w^-1 + 2072 + 1512*w + 756*w^2 + 56*w^3)*q^3 + (w^-4 + 576*w^-3 + 2072*w^-2 + 4032*w^-1 + 4158 + 4032*w + 2072*w^2 + 576*w^3 + w^4)*q^4 + O(q^5)

        """

        return self.weilrep().eisenstein_series(k - self.nvars() / 2, prec, allow_small_weight=allow_small_weight).jacobi_form()

    def eisenstein_newform(self, k, b, prec, allow_small_weight=False):
        r"""
        Compute certain newform Jacobi Eisenstein series of weight k.

        WARNING: this is experimental and also slow!!

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- a vector, or possibly (if self's index is an integer) an integer
        - ``prec`` -- the precision of the Fourier expansion

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(9).eisenstein_newform(3, 1, 5)
            (-2*w^-6 + 12*w^-5 - 24*w^-4 + 4*w^-3 + 54*w^-2 - 72*w^-1 + 72*w - 54*w^2 - 4*w^3 + 24*w^4 - 12*w^5 + 2*w^6)*q + (6*w^-8 - 24*w^-7 + 12*w^-6 + 72*w^-5 - 108*w^-4 - 24*w^-3 + 156*w^-2 - 120*w^-1 + 120*w - 156*w^2 + 24*w^3 + 108*w^4 - 72*w^5 - 12*w^6 + 24*w^7 - 6*w^8)*q^2 + (-6*w^-10 + 84*w^-8 - 156*w^-7 - 18*w^-6 + 276*w^-5 - 264*w^-4 + 36*w^-3 + 276*w^-2 - 396*w^-1 + 396*w - 276*w^2 - 36*w^3 + 264*w^4 - 276*w^5 + 18*w^6 + 156*w^7 - 84*w^8 + 6*w^10)*q^3 + (2*w^-12 + 24*w^-11 - 84*w^-10 + 216*w^-8 - 192*w^-7 - 28*w^-6 + 264*w^-5 - 438*w^-4 + 48*w^-3 + 504*w^-2 - 384*w^-1 + 384*w - 504*w^2 - 48*w^3 + 438*w^4 - 264*w^5 + 28*w^6 + 192*w^7 - 216*w^8 + 84*w^10 - 24*w^11 - 2*w^12)*q^4 + O(q^5)

        """
        if b in ZZ:
            m = self.index()
            f = m.squarefree_part()
            f0 = isqrt(m / f)
            b = vector([b / f0])
        return self.weilrep().eisenstein_newform(k - self.nvars() / 2, b, prec, allow_small_weight=allow_small_weight).jacobi_form()

    def eisenstein_oldform(self, k, b, prec, allow_small_weight=False):
        r"""
        Compute certain oldform Jacobi Eisenstein series of weight k.

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- a vector, or possibly (if self's index is an integer) an integer
        - ``prec`` -- the precision of the Fourier expansion

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(4).eisenstein_oldform(4, 1, 5)
            1 + (w^-4 + 56*w^-2 + 126 + 56*w^2 + w^4)*q + (126*w^-4 + 576*w^-2 + 756 + 576*w^2 + 126*w^4)*q^2 + (56*w^-6 + 756*w^-4 + 1512*w^-2 + 2072 + 1512*w^2 + 756*w^4 + 56*w^6)*q^3 + (w^-8 + 576*w^-6 + 2072*w^-4 + 4032*w^-2 + 4158 + 4032*w^2 + 2072*w^4 + 576*w^6 + w^8)*q^4 + O(q^5)

        """
        if b in ZZ:
            m = self.index()
            f = m.squarefree_part()
            f0 = isqrt(m / f)
            b = vector([b / f0])
        return self.weilrep().eisenstein_oldform(k - self.nvars() / 2, b, prec, allow_small_weight=allow_small_weight).jacobi_form()

    def poincare_series(self, k, n, r, prec, nterms = 10):
        r"""
        Compute a numerical approximation to the Jacobi Poincare series of index (n, r).

        This is (up to a constant multiple) the Jacobi form that extracts the Fourier coefficient of e^(2*pi*i * (n * \tau + r * z)) with respect to the Petersson inner product.

        INPUT:
        - ``k`` -- the weight
        - ``n`` -- an integer
        - ``r`` -- an integral vector of length equal to self's number of elliptic variables. (If that number is 1 then r may be given as an integer.)
        - ``prec`` -- the precision of the Fourier expansion
        """
        if r in ZZ:
            r = vector([r])
        m = self.index_matrix()
        mr = m.inverse() * r
        return self.weilrep().poincare_series(k - self.nvars() / 2, mr, n - r * mr / 2, prec, nterms = nterms).jacobi_form()

    ## dimensions associated to this index

    def cusp_forms_dimension(self, weight, **kwargs):
        r"""
        Dimension of Jacobi cusp forms.

        This computes the dimension of the space of Jacobi cusp forms of self's index and the given weight.

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).cusp_forms_dimension(10)
            1

            sage: from weilrep import *
            sage: JacobiForms(13).cusp_forms_dimension(3)
            1
        """
        if not self.is_even():
            return len(self.cusp_forms_basis(weight, prec = ceil(weight / 12 - self.nvars() / 24) + 1, **kwargs))
        eta_twist = kwargs.pop('eta_twist', 0)
        if not eta_twist:
            if self.nvars() == 1 and weight == 2:
                dim = self.weilrep().cusp_forms_dimension(sage_three_half, force_Riemann_Roch=True)
                N = self.index()
                sqrtN = isqrt(N)
                return dim + Integer(len(divisors(N)) + N.is_square()) / 2
            if weight <= 1:
                return 0
        return self.weilrep().cusp_forms_dimension(weight - self.nvars() / 2, eta_twist = eta_twist, **kwargs)

    def dimension(self, weight, **kwargs):
        r"""
        Dimension of Jacobi forms.

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(2).dimension(11)
            1

            sage: from weilrep import *
            sage: JacobiForms(4).dimension(8)
            3

            sage: from weilrep import *
            sage: JacobiForms(37).dimension(2)
            1

            sage: from weilrep import *
            sage: JacobiForms(49).dimension(2)
            2
        """
        if not self.is_even():
            return len(self.basis(weight, prec = ceil(weight / 12 - self.nvars() / 24) + 1, **kwargs))
        eta_twist = kwargs.pop('eta_twist', 0)
        if not eta_twist:
            if self.nvars() == 1 and weight == 2:
                dim = self.weilrep().modular_forms_dimension(sage_three_half, force_Riemann_Roch=True)
                N = self.index()
                sqrtN = isqrt(N)
                return dim + len([d for d in divisors(N) if d <= sqrtN and N % (d * d)])
            if weight <= 1:
                return 0
        return self.weilrep().modular_forms_dimension(weight - self.nvars() / 2, eta_twist = eta_twist, **kwargs)

    jacobi_forms_dimension = dimension

    def rank(self):
        r"""
        Compute the rank of the Jacobi forms of this index as a graded module over the ring of scalar modular forms.

        ALGORITHM: it's the determinant of self's index (matrix)
        """
        return self.index_matrix().determinant()

    def weak_forms_dimension(self, k):
        r"""
        Compute the dimension of the space of weak Jacobi forms of weight k.

        INPUT:
        - ``k`` -- weight

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(matrix([[2, 1], [1, 2]])).weak_forms_dimension(-3)
            1
        """
        n = self.nvars()
        if n == 1:
            m = Integer(self.index_matrix()[0, 0])
            k_plus_m = k + m
            if k_plus_m < 0:
                return 0
            r, t = PolynomialRing(ZZ, 't').objgen()
            f = (r([1,0]+[1]*(m - 1)) * (1 + t**4 + t**8) * (1 + t**6))
            return sum(f[k_plus_m - 12*n] * (n + 1) for n in range(1 + k_plus_m // 12))
        if not self.is_even():
            return len(self.weak_forms_basis(k))
        svn = self._short_vector_norms_by_component()
        w = self.weilrep()
        rds = w.rds()
        ds = w.ds()
        k_dual = 2 - k + n/2
        if k_dual <= 0:
            rds = w.rds(indices = True)
            d = sum(ceil(n) for i, n in enumerate(svn) if rds[i] is None )
            return self.dimension(k) + d
        N = max(svn) + 1
        #N = max(N, (k_dual / 12) + 1)
        wdual = w.dual()
        X = wdual.modular_forms_basis(k_dual, N)
        if not X:
            rds = w.rds(indices = True)
            d = sum(ceil(n) for i, n in enumerate(svn) if rds[i] is None and (not(k % 2) or 2 % denominator(ds[i])))
            return d
        v_list = wdual.coefficient_vector_exponents(N, 1 - k % 2, include_vectors = True)
        len_v_list = len(v_list)
        dsdict = wdual.ds_dict()
        I = [i for i, (g, n) in enumerate(v_list) if n <= svn[dsdict[g]]]
        Y1 = [x.coefficient_vector(starting_from = 0) for x in X]
        Y = matrix([[y[i] for i in I] for y in Y1])
        return Y.ncols() - Y.rank()


    def hilbert_series(self, polynomial = False):
        r"""
        Compute the Hilbert series f = \sum_k dim J_k t^k.

        INPUT:

        - ``polynomial`` -- boolean (default False). If True then output the Hilbert Polynomial f * (1 - t^4) * (1 - t^6) instead.

        """
        r, t = PolynomialRing(ZZ, 't').objgen()
        d = []
        p = []
        discr = self.discriminant()
        k, s = 0, 0
        while s < discr:
            p.append(self.dimension(k))
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

    def weak_hilbert_series(self, polynomial = False, verbose = False):
        r"""
        Compute the Hilbert series of weak Jacobi forms f = \sum_k dim J_k^w t^k.

        INPUT:

        - ``polynomial`` -- boolean (default False). If True then output the Hilbert Polynomial f * (1 - t^4) * (1 - t^6) instead.
        - ``verbose`` -- verbosity (default False)

        """
        if self.nvars() == 1:
            p = self.weak_hilbert_polynomial()
            if polynomial:
                return p
            t, = p.parent().gens()
            return p/((1 - t**4) * (1 - t**6))
        r, t = LaurentPolynomialRing(ZZ, 't').objgen()
        d = []
        p = []
        discr = self.discriminant()
        N = self._longest_short_vector_norm()
        k_min = floor(-12 * N)
        _ = [self.weilrep().dual().cusp_forms_basis(k + self.nvars() / 2, N, verbose = verbose) for k in range(k_min)]
        k, s = k_min, 0
        while s < discr:
            p.append(self.weak_forms_dimension(k))
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
            return r(p) * t**k_min
        return r(p) * (t ** k_min) / ((1 - t**4) * (1 - t**6))

    def weak_hilbert_polynomial(self):
        if self.nvars() == 1:
            r, t = LaurentPolynomialRing(ZZ, 't').objgen()
            m = Integer(self.index_matrix()[0, 0])
            return t**(-m) * r([1,0]+[1]*(m - 1))
        return self.weak_hilbert_series(polynomial = True)

    def zero(self, *args, **kwargs):
        return self.weilrep().zero(*args, **kwargs).jacobi_form()

    ## bases of spaces associated to this index

    def cusp_forms_basis(self, weight, prec=1, try_theta_blocks=None, **kwargs):
        r"""
        Compute a basis of Jacobi cusp forms.

        This computes a basis of Jacobi cusp forms of the given weight up to precision "prec". If "prec" is not given then a Sturm bound is used.

        INPUT:
         - ``weight`` -- the weight
         - ``prec`` -- the precision of the Fourier expansions (with respect to the variable 'q')
         - ``try_theta_blocks`` -- if True then in weight 2 or 3 we first try to find theta blocks which span the space.
         - ``verbose`` -- boolean (default False); if True then we add commentary throughout the computation
         - ``eta_twist`` -- integer (default 0); if this is 'k' then compute Jacobi forms that transform with the k^th power of the eta multiplier

        OUTPUT: a list of JacobiForm's

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).cusp_forms_basis(10, 5)
            [(w^-1 - 2 + w)*q + (-2*w^-2 - 16*w^-1 + 36 - 16*w - 2*w^2)*q^2 + (w^-3 + 36*w^-2 + 99*w^-1 - 272 + 99*w + 36*w^2 + w^3)*q^3 + (-16*w^-3 - 272*w^-2 - 240*w^-1 + 1056 - 240*w - 272*w^2 - 16*w^3)*q^4 + O(q^5)]

            sage: from weilrep import *
            sage: JacobiForms(2).cusp_forms_basis(11, 5)
            [(-w^-1 + w)*q + (w^-3 + 21*w^-1 - 21*w - w^3)*q^2 + (-21*w^-3 - 189*w^-1 + 189*w + 21*w^3)*q^3 + (-w^-5 + 189*w^-3 + 910*w^-1 - 910*w - 189*w^3 + w^5)*q^4 + O(q^5)]

        """
        verbose = kwargs.pop('verbose', False)
        eta_twist = kwargs.pop('eta_twist', 0)
        if not self.is_even():
            from .weilrep_misc import relations
            S = self.index_matrix()
            d = self.index_matrix().diagonal()
            odd_indices = [i for i, x in enumerate(d) if x % 2]
            A = matrix(S.nrows())
            U = []
            n = Integer(0)
            for i in odd_indices:
                A[i, i] = 1
                U.append(A.rows()[i])
                n += 1
            J = JacobiForms(S + A)
            X = J.cusp_forms_basis(weight + n/2, prec = prec, eta_twist = eta_twist + 3 * n, verbose = verbose, try_theta_blocks = try_theta_blocks)
            if not X:
                return []
            prec = max(1, X[0].precision())
            V = relations([x.substitute_zero([odd_indices[0]]) for x in X])
            for i in odd_indices[1:]:
                V = V.intersection(relations([x.substitute_zero([i]) for x in X]))
            V_rows = V.basis_matrix().rows()
            if not V_rows:
                return []
            Z = [sum(x * u[i] for i, x in enumerate(X)) for u in V_rows]
            f = jacobi_theta_series(prec = prec)
            f = prod(f.pullback(matrix(u).transpose()) for u in U)
            return [z / f for z in Z]
        if not eta_twist and self.nvars() == 1 and weight <= 3:
            dim = self.cusp_forms_dimension(weight)
            if not dim:
                return []
            jacobi_dim = self.dimension(weight)
            if verbose:
                print('I need to find %d Jacobi cusp form'%dim + ['s.', '.'][dim == 1])
            if try_theta_blocks is None:
                try_theta_blocks = (dim < 3) and (jacobi_dim == dim)  #arbitrary?
            if try_theta_blocks and (jacobi_dim == dim):
                from .weilrep_misc import weight_two_basis_from_theta_blocks, weight_three_basis_from_theta_blocks
                if verbose:
                    print('I will look for theta blocks of weight %d.' %weight)
                if weight == 2:
                    return weight_two_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms=self, verbose=verbose)
                else:
                    return weight_three_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms=self, verbose=verbose)
        wt = weight - self.nvars() / 2
        if verbose:
            print('I will look for the corresponding vector-valued cusp forms of weight %s.'% wt)
            print('-' * 60)
        L = self.weilrep().cusp_forms_basis(wt, prec, verbose=verbose, eta_twist = eta_twist, **kwargs)
        if not L:
            return []
        if verbose:
            print('-' * 60)
            print('I will now convert these cusp forms to Jacobi forms.')
        if len(L) > 1:
            return L.jacobi_forms()
        return [L[0].jacobi_form()]

    def jacobi_forms_basis(self, weight, prec=1, try_theta_blocks=None, **kwargs):
        r"""
        Compute a basis of Jacobi forms.

        This computes a basis of Jacobi forms of the given weight up to precision "prec". If "prec" is not given then a Sturm bound is used.

        NOTE: can also be called simply with "basis()"

        INPUT:
         - ``weight`` -- the weight
         - ``prec`` -- the precision of the Fourier expansions (with respect to the variable 'q')
         - ``try_theta_blocks`` -- if True then in weight 2 or 3 we first try to find theta blocks which span the space.
         - ``verbose`` -- boolean (default False); if True then we add commentary throughout the computation
         - ``eta_twist`` -- integer (default 0); if this is 'k' then compute Jacobi forms that transform with the k^th power of the eta multiplier

        OUTPUT: a list of JacobiForm's

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(1).jacobi_forms_basis(10, 5)
            [1 + (w^-2 - 266 + w^2)*q + (-266*w^-2 - 26752*w^-1 - 81396 - 26752*w - 266*w^2)*q^2 + (-81396*w^-2 - 1225728*w^-1 - 2582328 - 1225728*w - 81396*w^2)*q^3 + (w^-4 - 26752*w^-3 - 2582328*w^-2 - 17211264*w^-1 - 29700762 - 17211264*w - 2582328*w^2 - 26752*w^3 + w^4)*q^4 + O(q^5), (w^-1 - 2 + w)*q + (-2*w^-2 - 16*w^-1 + 36 - 16*w - 2*w^2)*q^2 + (w^-3 + 36*w^-2 + 99*w^-1 - 272 + 99*w + 36*w^2 + w^3)*q^3 + (-16*w^-3 - 272*w^-2 - 240*w^-1 + 1056 - 240*w - 272*w^2 - 16*w^3)*q^4 + O(q^5)]

        """
        verbose = kwargs.pop('verbose', False)
        eta_twist = kwargs.pop('eta_twist', 0)
        if not self.is_even():
            from .weilrep_misc import relations
            S = self.index_matrix()
            d = self.index_matrix().diagonal()
            odd_indices = [i for i, x in enumerate(d) if x % 2]
            A = matrix(S.nrows())
            U = []
            n = Integer(0)
            for i in odd_indices:
                A[i, i] = 1
                U.append(A.rows()[i])
                n += 1
            J = JacobiForms(S + A)
            X = J.basis(weight + n/2, prec = prec, eta_twist = eta_twist + 3 * n, verbose = verbose, try_theta_blocks = try_theta_blocks)
            if not X:
                return []
            prec = max(1, X[0].precision())
            V = relations([x.substitute_zero([odd_indices[0]]) for x in X])
            for i in odd_indices[1:]:
                V = V.intersection(relations([x.substitute_zero([i]) for x in X]))
            V_rows = V.basis_matrix().rows()
            if not V_rows:
                return []
            Z = [sum(x * u[i] for i, x in enumerate(X)) for u in V_rows]
            f = jacobi_theta_series(prec = prec)
            f = prod(f.pullback(matrix(u).transpose()) for u in U)
            return [z / f for z in Z]
        if not eta_twist and self.nvars() == 1 and weight <= 3:
            if weight <= 1:
                return []
            dim = self.dimension(weight)
            if not dim:
                return []
            if verbose:
                print('I need to find %d Jacobi form' % dim +['s.', '.'][dim == 1])
            if try_theta_blocks is None:
                try_theta_blocks = (dim < 3)  #arbitrary?
            if try_theta_blocks:
                from .weilrep_misc import weight_two_basis_from_theta_blocks, weight_three_basis_from_theta_blocks
                if verbose:
                    print('I will look for theta blocks of weight %d.' %
                          weight)
                if weight == 2:
                    return weight_two_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms=self, verbose=verbose)
                else:
                    return weight_three_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms=self, verbose=verbose)
        wt = weight - self.nvars() / 2
        if verbose:
            print('I will look for the corresponding vector-valued modular forms of weight %s.'% wt)
            print('-' * 60)
        L = self.weilrep().modular_forms_basis(wt, prec, verbose=verbose, eta_twist = eta_twist, **kwargs)
        if not L:
            return []
        if verbose:
            print('-' * 60)
            print('I will now convert these modular forms to Jacobi forms.')
        if len(L) > 1:
            return L.jacobi_forms()
        return [L[0].jacobi_form()]

    basis = jacobi_forms_basis

    def weak_forms_basis(self, weight, prec=1, verbose=False, convert_to_Jacobi_forms=True, **kwargs):
        r"""
        Compute a basis of weak Jacobi forms.

        A weak Jacobi form is a holomorphic function f(tau, z) satisfying the transformations of Jacobi forms and with a Fourier expansion of the form
        f(tau, z) = \sum_{n=0}^{\infty} \sum_r c(n,r) q^n \zeta^r
        i.e. with the usual restriction on 'n' but without the restriction on 'r'.

        This computes a basis of weak Jacobi forms of the given weight up to precision "prec". If "prec" is not given then a Sturm bound is used.

        INPUT:
        - ``weight`` -- the weight
        - ``prec`` -- the precision of the Fourier expansion (with respect to the variable 'q')
        - ``verbose`` -- boolean (default False); if True then we add commentary throughout the computation
        - ``convert_to_Jacobi_forms`` -- boolean (default True); if True then we convert the computed vector-valued modular forms to Jacobi forms.

        OUTPUT: a list of JacobiForm's

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(3).weak_forms_basis(0, 5)
            [w^-1 + 2 + w + (-2*w^-3 - 2*w^-2 + 2*w^-1 + 4 + 2*w - 2*w^2 - 2*w^3)*q + (w^-5 - 2*w^-4 - 6*w^-3 - 4*w^-2 + 5*w^-1 + 12 + 5*w - 4*w^2 - 6*w^3 - 2*w^4 + w^5)*q^2 + (2*w^-6 + 2*w^-5 - 4*w^-4 - 14*w^-3 - 10*w^-2 + 12*w^-1 + 24 + 12*w - 10*w^2 - 14*w^3 - 4*w^4 + 2*w^5 + 2*w^6)*q^3 + (w^-7 + 4*w^-6 + 5*w^-5 - 10*w^-4 - 30*w^-3 - 20*w^-2 + 24*w^-1 + 52 + 24*w - 20*w^2 - 30*w^3 - 10*w^4 + 5*w^5 + 4*w^6 + w^7)*q^4 + O(q^5), w^-2 + 14 + w^2 + (w^-4 + 40*w^-3 - 76*w^-2 - 168*w^-1 + 406 - 168*w - 76*w^2 + 40*w^3 + w^4)*q + (-76*w^-4 + 768*w^-3 - 1048*w^-2 - 1792*w^-1 + 4296 - 1792*w - 1048*w^2 + 768*w^3 - 76*w^4)*q^2 + (14*w^-6 - 168*w^-5 - 1048*w^-4 + 7192*w^-3 - 7998*w^-2 - 12656*w^-1 + 29328 - 12656*w - 7998*w^2 + 7192*w^3 - 1048*w^4 - 168*w^5 + 14*w^6)*q^3 + (406*w^-6 - 1792*w^-5 - 7998*w^-4 + 45312*w^-3 - 45558*w^-2 - 68096*w^-1 + 155452 - 68096*w - 45558*w^2 + 45312*w^3 - 7998*w^4 - 1792*w^5 + 406*w^6)*q^4 + O(q^5), w^-3 + 34 + w^3 + (-186*w^-3 + 2430*w^-2 - 8262*w^-1 + 12036 - 8262*w + 2430*w^2 - 186*w^3)*q + (2430*w^-4 - 35307*w^-3 + 175932*w^-2 - 425493*w^-1 + 564876 - 425493*w + 175932*w^2 - 35307*w^3 + 2430*w^4)*q^2 + (34*w^-6 - 8262*w^-5 + 175932*w^-4 - 1281814*w^-3 + 4623318*w^-2 - 9567396*w^-1 + 12116376 - 9567396*w + 4623318*w^2 - 1281814*w^3 + 175932*w^4 - 8262*w^5 + 34*w^6)*q^3 + (12036*w^-6 - 425493*w^-5 + 4623318*w^-4 - 24202674*w^-3 + 72869868*w^-2 - 137425977*w^-1 + 169097844 - 137425977*w + 72869868*w^2 - 24202674*w^3 + 4623318*w^4 - 425493*w^5 + 12036*w^6)*q^4 + O(q^5)]

            sage: from weilrep import *
            sage: JacobiForms(matrix([[2,1],[1,2]])).weak_forms_basis(-3, 5)
            [-w_0*w_1 + w_0 + w_1 - w_1^-1 - w_0^-1 + w_0^-1*w_1^-1 + (w_0^2*w_1^2 - w_0^2 - 8*w_0*w_1 - w_1^2 + 8*w_0 + 8*w_1 - 8*w_1^-1 - 8*w_0^-1 + w_1^-2 + 8*w_0^-1*w_1^-1 + w_0^-2 - w_0^-2*w_1^-2)*q + (-w_0^3*w_1^2 - w_0^2*w_1^3 + w_0^3*w_1 + 8*w_0^2*w_1^2 + w_0*w_1^3 - 8*w_0^2 - 44*w_0*w_1 - 8*w_1^2 + w_0^2*w_1^-1 + 44*w_0 + 44*w_1 + w_0^-1*w_1^2 - w_0*w_1^-2 - 44*w_1^-1 - 44*w_0^-1 - w_0^-2*w_1 + 8*w_1^-2 + 44*w_0^-1*w_1^-1 + 8*w_0^-2 - w_0^-1*w_1^-3 - 8*w_0^-2*w_1^-2 - w_0^-3*w_1^-1 + w_0^-2*w_1^-3 + w_0^-3*w_1^-2)*q^2 + (-8*w_0^3*w_1^2 - 8*w_0^2*w_1^3 + 8*w_0^3*w_1 + 44*w_0^2*w_1^2 + 8*w_0*w_1^3 - 44*w_0^2 - 192*w_0*w_1 - 44*w_1^2 + 8*w_0^2*w_1^-1 + 192*w_0 + 192*w_1 + 8*w_0^-1*w_1^2 - 8*w_0*w_1^-2 - 192*w_1^-1 - 192*w_0^-1 - 8*w_0^-2*w_1 + 44*w_1^-2 + 192*w_0^-1*w_1^-1 + 44*w_0^-2 - 8*w_0^-1*w_1^-3 - 44*w_0^-2*w_1^-2 - 8*w_0^-3*w_1^-1 + 8*w_0^-2*w_1^-3 + 8*w_0^-3*w_1^-2)*q^3 + (w_0^4*w_1^3 + w_0^3*w_1^4 - w_0^4*w_1 - 44*w_0^3*w_1^2 - 44*w_0^2*w_1^3 - w_0*w_1^4 + 44*w_0^3*w_1 + 192*w_0^2*w_1^2 + 44*w_0*w_1^3 - w_0^3*w_1^-1 - 192*w_0^2 - 726*w_0*w_1 - 192*w_1^2 - w_0^-1*w_1^3 + 44*w_0^2*w_1^-1 + 726*w_0 + 726*w_1 + 44*w_0^-1*w_1^2 - 44*w_0*w_1^-2 - 726*w_1^-1 - 726*w_0^-1 - 44*w_0^-2*w_1 + w_0*w_1^-3 + 192*w_1^-2 + 726*w_0^-1*w_1^-1 + 192*w_0^-2 + w_0^-3*w_1 - 44*w_0^-1*w_1^-3 - 192*w_0^-2*w_1^-2 - 44*w_0^-3*w_1^-1 + w_0^-1*w_1^-4 + 44*w_0^-2*w_1^-3 + 44*w_0^-3*w_1^-2 + w_0^-4*w_1^-1 - w_0^-3*w_1^-4 - w_0^-4*w_1^-3)*q^4 + O(q^5)]

        """
        S = self.index_matrix()
        if not self.is_even():
            from .weilrep_misc import relations
            from .weilrep_modular_forms_class import smf_eta
            d = self.index_matrix().diagonal()
            odd_indices = [i for i, x in enumerate(d) if x % 2]
            A = matrix(S.nrows())
            U = []
            n = Integer(0)
            for i in odd_indices:
                A[i, i] = 1
                U.append(A.rows()[i])
                n += 1
            J = JacobiForms(S + A)
            kwargs['reduce_prec'] = False
            X = J.weak_forms_basis(weight - n, prec = prec, verbose = verbose, **kwargs)
            if not X:
                return []
            prec = max(1, X[0].precision())
            V = relations([x.substitute_zero([odd_indices[0]]) for x in X])
            for i in odd_indices[1:]:
                V = V.intersection(relations([x.substitute_zero([i]) for x in X]))
            V_rows = V.basis_matrix().rows()
            if not V_rows:
                return []
            Z = [sum(x * u[i] for i, x in enumerate(X)) for u in V_rows]
            if verbose:
                print(r'I will now divide by the weak Jacobi form \phi_{-1, 0}.')
            f = jacobi_theta_series(prec = prec) / smf_eta(prec = prec)**3
            f = prod(f.pullback(matrix(u).transpose()) for u in U)
            return [z / f for z in Z]
        w = self.weilrep()
        indices = w.rds(indices=True)
        dsdict = w.ds_dict()
        if verbose:
            print('I am looking for weak Jacobi forms of weight %d.' % weight)
        svn = self._short_vector_norms_by_component()
        N = max(svn)
        if verbose:
            print('I will compute nearly-holomorphic modular forms with a pole in infinity of order at most %s.'% N)
            print('-' * 60)
        k = weight - self.nvars() / 2
        L = w.nearly_holomorphic_modular_forms_basis(k, N, prec, verbose=verbose, **kwargs)
        if not L:
            return []
        v_list = w.coefficient_vector_exponents(prec, 1 - (weight % 2), starting_from = -N, include_vectors = True)
        n = len(v_list)
        e = lambda i: vector([0] * i + [1] + [0] * (n - 1 - i))
        Z = [e(j) for j, (v, i) in enumerate(v_list) if i + svn[dsdict[tuple(v)]] >= 0]
        V = span([x.coefficient_vector(starting_from = -N, ending_with = prec)[:n] for x in L])
        V = V.intersection(span(Z)).basis()
        X = WeilRepModularFormsBasis(k, [w.recover_modular_form_from_coefficient_vector(k, v, prec, starting_from = -N) for v in V], w)
        X.reverse()
        if not convert_to_Jacobi_forms:
            return X
        if verbose:
            print('-' * 60)
            print('%d of these nearly-holomorphic forms appear to arise as theta decompositions of Jacobi forms.'% len(X))
            print('I am converting these modular forms to Jacobi forms.')
        jf = X.jacobi_forms()
        for y in jf:
            y._JacobiForm__fourier_expansion = y._JacobiForm__fourier_expansion.power_series()
        return jf

    ## other:

    def _short_vector_norms_by_component(self):
        r"""
        Computes the expression min( Q(x): x in ZZ^N + g) for g in self.ds()

        NOTE: used in weak_forms_basis() to determine which nearly-holomorphic modular forms might product weak Jacobi forms

        NOTE: indices g for which -g appears earlier in self.ds() are given the value (-1)

        TODO: is there a better way to do this?? for now we use PARI qfminim() to find some short vectors

        INPUT:

        - ``parity`` -- (default 1) 1 for even weight, 0 for odd weight

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(matrix([[2,1],[1,2]]))._short_vector_norms_by_component()
            [0, 1/3, -1]

        """
        try:
            return self.__short_vector_norms
        except AttributeError:
            pass
        w = self.weilrep()
        ds_dict = w.ds_dict()
        ds = w.ds()
        rds = w.rds()
        indices = w.rds(indices=True)
        S = w.gram_matrix()
        found_vectors = [g * S * g/2 if (indices[i] is None) else (-indices[i]) for i, g in enumerate(ds)]
        N_triv = max(found_vectors) #a trivial upper bound for longest_short_vector_norm(). next we'll use PARI qfminim() to lower this
        found_vectors[0] = 0
        S_inv = S.inverse()
        try:
            _, _, vs_matrix = pari(S_inv).qfminim(N_triv + 1, flag=2)
            vs_list = vs_matrix.sage().columns()
            for v in vs_list:
                r = S_inv * v
                j = ds_dict[tuple(frac(x) for x in r)]
                if indices[j]:
                    j = indices[j]
                if found_vectors[j] is None:
                    found_vectors[j] = v * r / 2
                else:
                    found_vectors[j] = min(v * r / 2, found_vectors[j])
            self.__short_vector_norms = found_vectors
            return found_vectors
        except PariError:
            lvl = w.level()
            S_adj = S_inv * lvl
            vs = QuadraticForm(S_adj).short_vector_list_up_to_length(lvl * N_triv + 1)
            for n in range(len(vs)):
                r_norm = n / lvl
                for v in vs[n]:
                    r = S_inv * v
                    j = ds_dict[tuple(frac(x) for x in r)]
                if indices[j]:
                    j = indices[j]
                if found_vectors[j] is None:
                    found_vectors[j] = v * r / 2
                if all(x is not None for x in found_vectors):
                    break
        for i, v in enumerate(found_vectors):
            if v < 0:
                found_vectors[i] = found_vectors[-v]
        self.__short_vector_norms = found_vectors
        return self.__short_vector_norms

    def _longest_short_vector_norm(self):
        return max(self._short_vector_norms_by_component())


class JacobiForm:
    r"""
    The JacobiForm class represents Jacobi forms.

    INPUT: (This is not meant to be called directly.) JacobiForm instances are constructed by calling

    JacobiForm(k, S, f)

    where:

    - ``k`` -- the weight (integer)
    - ``S`` -- the index: an integer or a Gram matrix
    - ``f`` -- the Fourier expansion. This is a power series in the variable 'q' over the base ring of Laurent polynomials in the variables 'w_0, ..., w_d' over QQ.

    Optional arguments:
    - ``modform`` -- a WeilRepModularForm which we assume is our theta decomposition (default None)
    - ``weilrep`` -- a WeilRep instance attached to this Jacobi form (default None)
    - ``jacobiforms`` -- a JacobiForms instance attached to this Jacobi form (default None)

    """
    def __init__(self, weight, index_matrix, fourier_expansion, **kwargs):
        try:
            self.__weight = ZZ(weight)
        except TypeError:
            self.__weight = QQ(weight)
        self.__index_matrix = index_matrix
        self.__fourier_expansion = fourier_expansion
        try:
            self.__weilrep = kwargs['weilrep']
        except KeyError:
            pass
        try:
            modform = kwargs['modform']
            if modform is not None:
                self.__theta = modform
        except KeyError:
            pass
        try:
            self.__jacobiforms = kwargs['jacobiforms']
        except KeyError:
            pass
        self.__wscale = kwargs.pop('w_scale', 1)

    def __repr__(self):
        try:
            return self.__string
        except AttributeError:
            s = str(self.fourier_expansion())
            if self.scale() == 2: #divide by scale
                def m(obj):
                    obj_s = obj.string[slice(*obj.span())]
                    x = obj_s[0]
                    if x == '^':
                        u = ZZ(obj_s[1:])/2
                        if u.is_integer():
                            if u == 1:
                                return ''
                            return '^%d'%u
                        return '^(%s)'%u
                    return obj_s + '^(%s)'%sage_one_half
                s = sub(r'((?<!q)\^-?\d+)|w\_\d+(?!\^)', m, s)
            if self.nvars() == 1:
                s = s.replace('w_0', 'w')
            self.__string = s
            return s

    def _latex_(self):
        return latex(self.fourier_expansion())

    ## basic attributes

    def base_ring(self):
        r"""
        Laurent polynomial ring representing self's elliptic variables.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).base_ring()
            Univariate Laurent Polynomial Ring in w_0 over Rational Field
        """
        return self.fourier_expansion().base_ring()

    def character(self):
        return EtaCharacterPower(0)

    def coefficient_vector(self, starting_from=None, ending_with=None, correct=True):
        r"""
        Return self's non-redundant Fourier coefficients c(n, r) as a vector sorted by increasing value of n - r^2 / (4m) (or its appropriate generalization to matrix index)

        This returns the coefficient vector of self's theta decomposition.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).coefficient_vector()
            (1, 56, 126, 576, 756, 1512, 2072, 4032, 4158, 5544)

        """
        if starting_from is None and ending_with is None:
            try:
                return self.__coefficient_vector
            except:
                pass
        v = self.theta_decomposition(correct = correct, same_precision = correct).coefficient_vector(starting_from=starting_from, ending_with=ending_with)
        if correct and starting_from is None and ending_with is None:
            self.__coefficient_vector = v
        return v

    def development_coefficient(self, lattice_basis, v = []):
        r"""
        Compute development coefficients.

        Suppose 'self' is a Jacobi form of weight 'k' and lattice index 'L'. For any sublattice K <= L and any vectors v_1,...,v_N \in L orthogonal to K, the development coefficient D_{v_1,...,v_N} is a Jacobi form of weight k+N and lattice index K. It is a modification of the Taylor coefficient that transforms as a Jacobi form.

        INPUT:
        - ``lattice_basis" -- a list of linearly independent integral vectors (which span the sublattice K). This can be the empty list.
        - ``v" -- the list [v_1,...,v_N]

        NOTE: If 'self' is a Jacobi form of scalar index then we can instead call
        self.development_coefficient(N)
        to yield the N^th development coefficient as originally defined by Eichler--Zagier. (This corresponds to taking lattice_basis = [] and v_1 = ... = v_N = (1).)

        EXAMPLES::

            sage: from weilrep import *
            sage: f = JacobiForms(1).weak_forms_basis(0, 10)[0]
            sage: f.development_coefficient(4)
            4 + 960*q + 8640*q^2 + 26880*q^3 + 70080*q^4 + 120960*q^5 + 241920*q^6 + 330240*q^7 + 561600*q^8 + 726720*q^9 + O(q^10)

            sage: from weilrep import *
            sage: f = JacobiForms(37).cusp_forms_basis(2, 10)[0]
            sage: f.development_coefficient(6)
            O(q^10)

            sage: from weilrep import *
            sage: f = JacobiForms(37).cusp_forms_basis(2, 10)[0]
            sage: f.development_coefficient(10) / 158018273280000
            q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 - 6048*q^6 - 16744*q^7 + 84480*q^8 - 113643*q^9 + O(q^10)

            sage: from weilrep import *
            sage: f = JacobiForms(1).cusp_forms_basis(12, 10)[0]
            sage: f.development_coefficient(0)
            12*q - 288*q^2 + 3024*q^3 - 17664*q^4 + 57960*q^5 - 72576*q^6 - 200928*q^7 + 1013760*q^8 - 1363716*q^9 + O(q^10)

            sage: from weilrep import *
            sage: J = JacobiForms([[2, 1], [1, 4]])
            sage: f = J.cusp_forms_basis(7, 5)[0]
            sage: f.development_coefficient([vector([1, 0])], [vector([1, -2])]*3)
            (360*w^-1 - 720 + 360*w)*q + (-720*w^-2 - 5760*w^-1 + 12960 - 5760*w - 720*w^2)*q^2 + (360*w^-3 + 12960*w^-2 + 35640*w^-1 - 97920 + 35640*w + 12960*w^2 + 360*w^3)*q^3 + (-5760*w^-3 - 97920*w^-2 - 86400*w^-1 + 380160 - 86400*w - 97920*w^2 - 5760*w^3)*q^4 + O(q^5)
        """
        if isinstance(lattice_basis, Integer) and self.nvars() == 1:
            v = [vector([1])]*lattice_basis
            lattice_basis = []
        try:
            f = self.__theta
            return f.development_coefficient(lattice_basis, v=v).jacobi_form()
        except AttributeError:
            pass
        from .weilrep_misc import multilinear_gegenbauer_polynomial
        k = self.weight()
        K = self.base_ring().base_ring()
        S = self.index_matrix()
        S_inv = S.inverse()
        z = matrix(ZZ, lattice_basis)
        z_tr = z.transpose()
        ell = Integer(z.nrows())
        if ell:
            Sz = S * z_tr
            if matrix(v) * Sz:
                raise ValueError('The development coefficient must be evaluated along vectors orthogonal to the sublattice.')
            A = Sz.integer_kernel().basis_matrix()
            Rb = LaurentPolynomialRing(K,list(var('w_%d' % i) for i in range(ell) ))
        else:
            A = identity_matrix(S.nrows())
            Rb = K
            Sz = matrix([])
        j = JacobiForms(z * Sz)
        N = len(v)
        R = PowerSeriesRing(Rb, 'q')
        qshift = self._qshift()
        wscale = self.scale()
        if N:
            P = multilinear_gegenbauer_polynomial(N, k - 1 - j.nvars()/2, v, S)
        else:
            P = lambda *_: 1
        if ell == 1:
            w, = Rb.gens()
            def m(x):
                return w ** x[0]
        elif ell > 1:
            def m(x):
                return Rb.monomial(*x)
        else:
            m = lambda _:1
        def a(n):
            n += qshift
            def b(x):
                if ell:
                    try:
                        return sum( m(vector(x)*z_tr) * P(*S_inv*vector(x) / wscale, n) * y for x, y in x.dict().items() )
                    except AttributeError:
                        return K(x)
                    except TypeError:
                        return sum( m(vector([x])*z_tr) * P(*S_inv*vector([x]) / wscale, n) * y for x, y in x.dict().items() )
                else:
                    try:
                        return sum( P(*S_inv*vector(x) / wscale, n) * y for x, y in x.dict().items() )
                    except AttributeError:
                        return K(x)
                    except TypeError:
                        return sum( P(*S_inv*vector([x]) / wscale, n) * y for x, y in x.dict().items() )
            return b
        f = R( [a(n)(h) for n, h in enumerate(self.q_coefficients())] ).add_bigoh(self.precision())
        if qshift:
            return JacobiFormWithCharacter(k + N, j.index_matrix(), f, jacobiforms = j, character = self.character(), qshift = qshift, w_scale = wscale)
        return JacobiForm(k + N, j.index_matrix(), f, jacobiforms = j, w_scale = wscale)

    def fourier_expansion(self):
        r"""
        Return self's Fourier expansion.

        This is a Power series in the variable 'q' with coefficients in a ring of Laurent polynomials over QQ in variables w_0, ... w_d.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).fourier_expansion()
            1 + (w_0^-2 + 56*w_0^-1 + 126 + 56*w_0 + w_0^2)*q + (126*w_0^-2 + 576*w_0^-1 + 756 + 576*w_0 + 126*w_0^2)*q^2 + (56*w_0^-3 + 756*w_0^-2 + 1512*w_0^-1 + 2072 + 1512*w_0 + 756*w_0^2 + 56*w_0^3)*q^3 + (w_0^-4 + 576*w_0^-3 + 2072*w_0^-2 + 4032*w_0^-1 + 4158 + 4032*w_0 + 2072*w_0^2 + 576*w_0^3 + w_0^4)*q^4 + O(q^5)
        """
        return self.__fourier_expansion

    qexp = fourier_expansion

    def index(self):
        r"""
        Return self's index.

        If the index is a rank one matrix then return self's index as a scalar instead.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).index()
            1
        """
        S = self.index_matrix()
        e = self.nvars()
        if e == 1:
            try:
                return Integer(S[0][0] / 2)
            except TypeError:
                return S[0][0] / 2
        else:
            return S

    def index_matrix(self):
        r"""
        Return self's index as a matrix.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).index_matrix()
            [2]
        """
        return self.__index_matrix

    def is_cusp_form(self):
        r"""
        Return whether self is a Jacobi cusp form.

        WARNING: we only check holomorphy at the cusps! Applying this to Jacobi forms with poles on H x (C \otimes L) itself may still return True
        """
        X = self.theta_decomposition(same_precision = False).fourier_expansion()
        try:
            return all(x[2].valuation() / max(0, x[2].prec()) > 0 for x in X)
        except ZeroDivisionError:
            if self.precision() > 0:
                raise ValueError('I am probably a cusp form. However unless my precision is at least %d it is impossible to say for sure.'%ceil(2 + self.jacobiforms()._longest_short_vector_norm()))
            raise ValueError('My precision must be at least %d to tell whether I am a cusp form.'%ceil(2 + self.jacobiforms()._longest_short_vector_norm()))

    def is_holomorphic(self):
        r"""
        Return whether self is a holomorphic Jacobi form.

        WARNING: we only check holomorphy at the cusps! Applying this to Jacobi forms with poles on H x (C \otimes L) itself may still return True
        """
        X = self.theta_decomposition(same_precision = False).fourier_expansion()
        try:
            return all(x[2].valuation() / max(0, x[2].prec()) >= 0 and (not x[1] or x[2].valuation() / max(0, x[2].prec()) > 0) for x in X)
        except ZeroDivisionError:
            if self.precision() > 0:
                raise ValueError('I am probably a holomorphic Jacobi form. However unless my precision is at least %d it is impossible to say for sure.'%ceil(2 + self.jacobiforms()._longest_short_vector_norm()))
            raise ValueError('My precision must be at least %d to tell whether I am holomorphic.'%ceil(2 + self.jacobiforms()._longest_short_vector_norm()))

    def jacobiforms(self):
        r"""
        Return the JacobiForms of self's index.
        """
        try:
            return self.__jacobiforms
        except AttributeError:
            self.__jacobiforms = JacobiForms(self.index())
            return self.__jacobiforms

    def modform(self, error = True):
        r"""
        Try to return self's theta decomposition. If this is not stored then raises an AttributeError.
        """
        try:
            return self.__theta
        except AttributeError as e:
            if error:
                raise e

    def nvars(self):
        r"""
        Self's number of elliptic variables.
        """
        try:
            return self.__nvars
        except AttributeError:
            self.__nvars = Integer(self.index_matrix().nrows())
            return self.__nvars

    def precision(self):
        r"""
        Return self's precision (with respect to the variable 'q').

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).precision()
            5
        """
        try:
            return self.__precision
        except:
            self.__precision = self.fourier_expansion().prec()
            return self.__precision

    prec = precision

    def q_coefficients(self):
        r"""
        Return self's Fourier coefficients with respect to the variable 'q'.

        OUTPUT: a list of Laurent polynomials

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).q_coefficients()
            [1, w_0^-2 + 56*w_0^-1 + 126 + 56*w_0 + w_0^2, 126*w_0^-2 + 576*w_0^-1 + 756 + 576*w_0 + 126*w_0^2, 56*w_0^-3 + 756*w_0^-2 + 1512*w_0^-1 + 2072 + 1512*w_0 + 756*w_0^2 + 56*w_0^3, w_0^-4 + 576*w_0^-3 + 2072*w_0^-2 + 4032*w_0^-1 + 4158 + 4032*w_0 + 2072*w_0^2 + 576*w_0^3 + w_0^4]
        """
        return list(self.fourier_expansion())

    def _qshift(self):
        return 0

    def reduce_precision(self, new_prec):
        return JacobiForm(self.weight(), self.index_matrix(), self.qexp().add_bigoh(new_prec), modform = self.modform(error = False), weilrep = self.weilrep(), w_scale = self.scale())

    def _rescale(self, a):
        f = self.qexp()
        e = self.nvars()
        rb_w = f.base_ring()
        d = {rb_w('w_%d' % j): rb_w('w_%d' % j)**a for j in range(e)}
        f = f.map_coefficients(lambda x: x.subs(d))
        return JacobiForm(self.weight(), self.index_matrix(), f, modform = self.modform(error = False), weilrep = self.weilrep(), w_scale = self.scale() * a)

    def scale(self):
        return self.__wscale

    def theta_decomposition(self, correct=True, same_precision=True):
        r"""
        Return self's theta decomposition.

        INPUT: optional arguments
        - ``correct`` -- boolean (default True). If False then fill the Fourier expansion with any exponents we find. (this is only useful in the coefficient_vector method)
        - ``same_precision`` -- boolean (default True). If False then we allow the components to have varying precision.

        OUTPUT: a WeilRepModularForm

        WARNING: passing to JacobiForm and back to WeilRepModularForm can incur a severe precision loss! we try to avoid this by caching the original modular form

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).theta_decomposition()
            [(0), 1 + 126*q + 756*q^2 + 2072*q^3 + 4158*q^4 + O(q^5)]
            [(1/2), 56*q^(3/4) + 576*q^(7/4) + 1512*q^(11/4) + 4032*q^(15/4) + 5544*q^(19/4) + O(q^(23/4))]

            sage: from weilrep import *
            sage: (jacobi_eisenstein_series(4, 1, 5) ** 2).theta_decomposition() #we lose precision here! it's unavoidable I guess
            [(0), 1 + 252*q + 23662*q^2 + 324184*q^3 + O(q^4)]
            [(1/4), 112*q^(7/8) + 15376*q^(15/8) + 248112*q^(23/8) + O(q^(31/8))]
            [(1/2), 2*q^(1/2) + 3640*q^(3/2) + 99288*q^(5/2) + O(q^(7/2))]
            [(3/4), 112*q^(7/8) + 15376*q^(15/8) + 248112*q^(23/8) + O(q^(31/8))]
        """
        try:
            return self.__theta
        except AttributeError:
            pass
        w = self.weilrep()
        if w is None:
            return self.hecke_U(2).theta_decomposition()
        if self.scale() == 2: #scale back down to 1. This should be OK now.
            self.__wscale = 1
            f = self.qexp()
            R = self.base_ring()
            if self.nvars() > 1:
                self.__fourier_expansion = f.map_coefficients(lambda x: R({x.escalar_div(2) : y for x, y in x.dict().items()}))
            else:
                self.__fourier_expansion = f.map_coefficients(lambda x: R({x // 2: y for x, y in x.dict().items()}))
        ds_dict = w.ds_dict()
        ds = w.ds()
        j = self.jacobiforms()
        if correct and same_precision:
            try:
                return self.__theta
            except AttributeError:
                pass
            N = j._longest_short_vector_norm()
            norms = [N] * len(ds)
        elif correct:
            norms = j._short_vector_norms_by_component()
        else:
            norms = [0] * len(ds)
        f, S, e, n_list = self.fourier_expansion(), self.index_matrix(), self.nvars(), w.norm_list()
        prec, val, S_inv= f.prec(), f.valuation(), S.inverse()
        lsr, q = LaurentSeriesRing(QQ, 'q').objgen()
        psr, q = PowerSeriesRing(QQ, 'q').objgen()
        if e == 0:
            self.__theta = WeilRepModularForm(self.weight(), matrix([]), [[vector([]), 0, psr(self.qexp())]])
            return self.__theta
        L = [[g, n_list[i], O(q**(prec - ceil(norms[i]) - ceil(n_list[i])))] if prec > (1 + ceil(norms[i]) + ceil(n_list[i])) else [g, n_list[i], O(lsr(q)**(prec - ceil(norms[i]) - ceil(n_list[i])))] for i, g in enumerate(ds)]
        lower_bounds = [None] * len(ds)
        for i in range(val, prec):
            h = f[i]
            h_coeffs = h.coefficients()
            for k, v in enumerate(h.exponents()):
                if e == 1:
                    r = vector(S_inv[0, 0] * v)
                else:
                    r = S_inv * vector(v)
                r_frac = tuple(frac(r[j]) for j in range(e))
                j = ds_dict[r_frac]
                exponent = ceil(i - r * S * r / 2)
                if (lower_bounds[j] is None) or (exponent > lower_bounds[j]):
                    lower_bounds[j] = exponent
                    try:
                        L[j][2] += h_coeffs[k] * q**exponent
                    except ValueError:
                        pass
        if correct:
            self.__theta = WeilRepModularForm(self.weight() - e / 2, S, L, weilrep=self.weilrep())
            return self.__theta
        return WeilRepModularForm(self.weight() - e / 2, S, L, weilrep=self.weilrep())

    def valuation(self):
        r"""
        Return self's valuation (with respect to the variable 'q').

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).valuation()
            0
        """
        try:
            return self.__valuation
        except:
            self.__valuation = self.fourier_expansion().valuation()
            return self.__valuation

    def weight(self):
        r"""
        Return self's weight.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).weight()
            4
        """
        return self.__weight

    def weilrep(self):
        r"""
        Return self's WeilRep instance.

        If self was created without a WeilRep instance then we create one here.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).weilrep()
            Weil representation associated to the Gram matrix
            [2]
        """
        try:
            return self.__weilrep
        except AttributeError:
            from .weilrep import WeilRep
            try:
                self.__weilrep = WeilRep(self.index_matrix())
                return self.__weilrep
            except TypeError:
                return None

    def __bool__(self):
        try:
            return self.modform().__bool__()
        except AttributeError:
            return self.fourier_expansion().__bool__()

    def _frac(self):
        K = self.base_ring().fraction_field()
        f = JacobiForm(self.__weight, self.__index_matrix, self.__fourier_expansion.change_ring(K))
        return f

    ## arithmetic operations

    def __add__(self, other):
        r"""
        Addition of Jacobi forms. Undefined unless both have the same weight and index.
        """
        if not other:
            return self
        if not isinstance(other, JacobiForm):
            raise TypeError('Cannot add these objects')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        if not self.index_matrix() == other.index_matrix():
            raise ValueError('Incompatible indices')
        try:
            modform = self.modform() + other.modform()
        except (AttributeError, TypeError):
            modform = None
        scale = 1
        sf, of = self.qexp(), other.qexp()
        if self.scale() == 2 or other.scale() == 2:
            r = self.base_ring()
            scale = 2
            if self.scale() == 1:
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            if other.scale() == 1:
                of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
        return JacobiForm(self.weight(), self.index_matrix(), sf + of, modform=modform, weilrep=self.weilrep(), jacobiforms = self.jacobiforms(), w_scale = scale)

    __radd__ = __add__

    def __sub__(self, other):
        r"""
        Subtraction of Jacobi forms. Undefined unless both have the same weight and index.
        """
        if not other:
            return self
        if not isinstance(other, JacobiForm):
            raise TypeError('Cannot subtract these objects')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        if not self.index_matrix() == other.index_matrix():
            raise ValueError('Incompatible indices')
        try:
            modform = self.modform() - other.modform()
        except (AttributeError, TypeError):
            modform = None
        scale = 1
        sf, of = self.qexp(), other.qexp()
        if self.scale() == 2 or other.scale() == 2:
            r = self.base_ring()
            scale = 2
            if self.scale() == 1:
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            if other.scale() == 1:
                of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
        return JacobiForm(self.weight(), self.index_matrix(), sf - of, modform=modform, weilrep=self.weilrep(), jacobiforms = self.jacobiforms(), w_scale = scale)

    def __invert__(self):
        f = self.qexp()
        try:
            f_inv = ~f
        except ValueError:
            R = f.parent()
            R_frac = FractionField(R)
            f_inv = ~R_frac(f)
        return JacobiForm(-self.weight(), -self.index_matrix(), f_inv, w_scale = self.scale())

    def __neg__(self):
        r"""
        Return the negative of self.
        """
        try:
            modform = -self.modform()
        except (AttributeError, TypeError):
            modform = None
        return JacobiForm(self.weight(), self.index_matrix(), -self.fourier_expansion(), modform=modform, weilrep=self.weilrep(), jacobiforms = self.jacobiforms(), w_scale = self.scale())

    def __mul__(self, other):
        r"""
        Multiplication of Jacobi forms. Undefined unless both have index of the same rank.
        """
        if isinstance(other, JacobiForm):
            S1 = self.index_matrix()
            S2 = other.index_matrix()
            if not S1.nrows() == S2.nrows():
                raise ValueError('Incompatible indices')
            sf, of = self.qexp(), other.qexp()
            scale = 1
            if self.scale() == 2 or other.scale() == 2:
                r = self.base_ring()
                scale = 2
                if self.scale() == 1:
                    sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
                if other.scale() == 1:
                    of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            f = sf * of
            R = f.base_ring()
            if R is not LaurentPolynomialRing:
                R = LaurentPolynomialRing(QQ, R.gens())
                def a(x):
                    try:
                        u = R(x.numerator()) / R(x.denominator())
                        u = R(u)
                    except AttributeError:
                        u = R(x)
                    return u
                Rq = PowerSeriesRing(R, 'q')
                try:
                    f = Rq({i: a(f[i]) for i in f.exponents()}).add_bigoh(f.prec())
                except TypeError:
                    pass
            return JacobiForm(self.weight() + other.weight(), S1 + S2, f, w_scale = scale)
        elif is_ModularFormElement(other):
            try:
                modform = self.modform() * smf(other.weight(), other.qexp())
            except (AttributeError, TypeError):
                modform = None
            sf, of = self.qexp(), other.qexp()
            f = sf * of.change_ring(sf.base_ring())
            return JacobiForm(self.weight() + other.weight(), self.index_matrix(), f, modform=modform, w_scale = self.scale())
        elif isinstance(other, WeilRepModularForm):
            if other.weilrep().gram_matrix().nrows() == 0:
                try:
                    modform = self.modform() * other
                except (AttributeError, TypeError):
                    modform = None
                sf, of = self.qexp(), other.fourier_expansion()[0][2]
                f = sf * of.change_ring(sf.base_ring())
                qshift = other.fourier_expansion()[0][1]
                if qshift:
                    return JacobiFormWithCharacter(self.weight() + other.weight(), self.index_matrix(), f, modform = modform, w_scale = self.scale(), character = other.character(), qshift = qshift)
                return JacobiForm(self.weight() + other.weight(), self.index_matrix(), f, modform = modform, w_scale = self.scale())
            raise NotImplementedError
        else:
            try:
                modform = self.modform() * other
            except (AttributeError, TypeError):
                modform = None
            return JacobiForm(self.weight(), self.index_matrix(), self.fourier_expansion() * other, modform=modform, weilrep=self.weilrep(), jacobiforms = self.jacobiforms(), w_scale = self.scale())

    __rmul__ = __mul__

    def __truediv__(self, other):
        r"""
        Division of Jacobi forms. Undefined unless both have index of the same rank.
        """
        if isinstance(other, JacobiForm):
            S1 = self.index_matrix()
            S2 = other.index_matrix()
            if not S1.nrows() == S2.nrows():
                raise ValueError('Incompatible indices')
            sf, of = self.fourier_expansion(), other.fourier_expansion()
            scale = 1
            r = self.base_ring()
            if self.scale() == 2 or other.scale() == 2:
                scale = 2
                if self.scale() == 1:
                    sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
                if other.scale() == 1:
                    of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            try:
                f = sf / of
            except ValueError:
                K = r.fraction_field()
                sf, of = sf.change_ring(K), of.change_ring(K)
                z = next(z for z in of if z)
                of /= z
                sf /= z
                f = sf / of
            R = f.base_ring()
            if R is not LaurentPolynomialRing:
                try:
                    R = LaurentPolynomialRing(QQ, R.gens())
                except ValueError:
                    R = QQ
                def a(x):
                    try:
                        u = R(x.numerator()) / R(x.denominator())
                        u = R(u)
                    except AttributeError:
                        u = R(x)
                    return u
                Rq = PowerSeriesRing(R, 'q')
                try:
                    f = Rq({i: a(f[i]) for i in f.exponents()}).add_bigoh(f.prec())
                except TypeError:
                    pass
            return JacobiForm(self.weight() - other.weight(), S1 - S2, f, w_scale = scale)
        elif isinstance(other, WeilRepModularForm):
            if other.weilrep().gram_matrix().nrows() == 0:
                try:
                    modform = self.modform() / other
                except (AttributeError, TypeError):
                    modform = None
                sf, of = self.qexp(), other.fourier_expansion()[0][2]
                f = sf / of
                R = f.base_ring()
                if R is not LaurentPolynomialRing:
                    f = f.change_ring(LaurentPolynomialRing(R.base_ring(), R.gens()))
                qshift = other.fourier_expansion()[0][1]
                if qshift:
                    return JacobiFormWithCharacter(self.weight() - other.weight(), self.index_matrix(), f, modform = modform, w_scale = self.scale(), character = ~other.character(), qshift = -qshift)
                return JacobiForm(self.weight() - other.weight(), self.index_matrix(), f, modform = modform, w_scale = self.scale())
        elif is_ModularFormElement(other):
            try:
                modform = self.modform() / smf(-other.weight(), ~other.qexp())
            except (AttributeError, TypeError):
                modform = None
            f = self.fourier_expansion() / other.qexp()
            return JacobiForm(self.weight() - other.weight(), self.index_matrix(), f, modform=modform, weilrep=self.weilrep(), jacobiforms = self.jacobiforms(), w_scale = self.scale())
        elif other in CC:
            try:
                modform = self.modform() / other
            except (AttributeError, TypeError):
                modform = None
            return JacobiForm(self.weight(), self.index_matrix(), self.fourier_expansion() / other, modform=modform, weilrep=self.weilrep(), jacobiforms = self.jacobiforms(), w_scale = self.scale())
        return NotImplemented

    __div__ = __truediv__

    def __rtruediv__(self, other):
        N = self.nvars()
        if other in QQ:
            x = JacobiForm(0, matrix(N), self.qexp().parent()(other))
        elif is_ModularFormElement(other):
            x = JacobiForm(other.weight(), matrix(N), self.qexp().parent()(other.qexp()))
        elif isinstance(other, WeilRepModularForm):
            if other.weilrep().gram_matrix().nrows() == 0:
                x = JacobiForm(self.weight() + other.weight(), self.index_matrix(), of/sf, modform = modform, w_scale = self.scale())
            raise NotImplementedError
        else:
            x = other
        return x.__truediv__(self)

    __rdiv__ = __rtruediv__

    def __pow__(self, other):  ## "tensor product" i.e. multiply together with separate abelian variables!
        r"""
        Outer product of Jacobi forms.

        If f(tau, z_1,...,z_m) and g(tau, z_1,...,z_n) are our Jacobi forms then this produces the Jacobi form

        f ** g(tau, z_1,...,z_(m+n)) = f(tau, z_1,...,z_m) * g(tau, z_(m+1), ..., z_(m+n)).

        EXAMPLES::

            sage: from weilrep import *
            sage: J4 = jacobi_eisenstein_series(4, 2, 5)
            sage: J6 = jacobi_eisenstein_series(6, 1, 5)
            sage: J4 ** J6
            1 + (14*w_0^2 + w_1^2 + 64*w_0 - 88*w_1 - 246 - 88*w_1^-1 + 64*w_0^-1 + w_1^-2 + 14*w_0^-2)*q + (w_0^4 + 14*w_0^2*w_1^2 + 64*w_0^3 - 1232*w_0^2*w_1 + 64*w_0*w_1^2 - 4340*w_0^2 - 5632*w_0*w_1 - 246*w_1^2 - 1232*w_0^2*w_1^-1 - 20672*w_0 - 11616*w_1 + 64*w_0^-1*w_1^2 + 14*w_0^2*w_1^-2 - 5632*w_0*w_1^-1 - 34670 - 5632*w_0^-1*w_1 + 14*w_0^-2*w_1^2 + 64*w_0*w_1^-2 - 11616*w_1^-1 - 20672*w_0^-1 - 1232*w_0^-2*w_1 - 246*w_1^-2 - 5632*w_0^-1*w_1^-1 - 4340*w_0^-2 + 64*w_0^-1*w_1^-2 - 1232*w_0^-2*w_1^-1 + 64*w_0^-3 + 14*w_0^-2*w_1^-2 + w_0^-4)*q^2 + (w_0^4*w_1^2 - 88*w_0^4*w_1 + 64*w_0^3*w_1^2 - 246*w_0^4 - 5632*w_0^3*w_1 - 4340*w_0^2*w_1^2 - 88*w_0^4*w_1^-1 - 20672*w_0^3 - 83776*w_0^2*w_1 - 20672*w_0*w_1^2 - 88*w_1^3 + w_0^4*w_1^-2 - 5632*w_0^3*w_1^-1 - 196896*w_0^2 - 309760*w_0*w_1 - 34670*w_1^2 + 64*w_0^3*w_1^-2 - 83776*w_0^2*w_1^-1 - 628032*w_0 - 435928*w_1 - 20672*w_0^-1*w_1^2 - 4340*w_0^2*w_1^-2 - 309760*w_0*w_1^-1 - 866700 - 309760*w_0^-1*w_1 - 4340*w_0^-2*w_1^2 - 20672*w_0*w_1^-2 - 435928*w_1^-1 - 628032*w_0^-1 - 83776*w_0^-2*w_1 + 64*w_0^-3*w_1^2 - 34670*w_1^-2 - 309760*w_0^-1*w_1^-1 - 196896*w_0^-2 - 5632*w_0^-3*w_1 + w_0^-4*w_1^2 - 88*w_1^-3 - 20672*w_0^-1*w_1^-2 - 83776*w_0^-2*w_1^-1 - 20672*w_0^-3 - 88*w_0^-4*w_1 - 4340*w_0^-2*w_1^-2 - 5632*w_0^-3*w_1^-1 - 246*w_0^-4 + 64*w_0^-3*w_1^-2 - 88*w_0^-4*w_1^-1 + w_0^-4*w_1^-2)*q^3 + (-246*w_0^4*w_1^2 + 64*w_0^5 - 11616*w_0^4*w_1 - 20672*w_0^3*w_1^2 - 1232*w_0^2*w_1^3 - 34670*w_0^4 - 309760*w_0^3*w_1 - 196896*w_0^2*w_1^2 - 5632*w_0*w_1^3 + w_1^4 - 11616*w_0^4*w_1^-1 - 628032*w_0^3 - 1685040*w_0^2*w_1 - 628032*w_0*w_1^2 - 11616*w_1^3 - 246*w_0^4*w_1^-2 - 309760*w_0^3*w_1^-1 - 3033280*w_0^2 - 3969024*w_0*w_1 - 866700*w_1^2 - 5632*w_0^-1*w_1^3 - 20672*w_0^3*w_1^-2 - 1685040*w_0^2*w_1^-1 - 6790912*w_0 - 5239264*w_1 - 628032*w_0^-1*w_1^2 - 1232*w_0^-2*w_1^3 - 196896*w_0^2*w_1^-2 - 3969024*w_0*w_1^-1 - 8820030 - 3969024*w_0^-1*w_1 - 196896*w_0^-2*w_1^2 - 1232*w_0^2*w_1^-3 - 628032*w_0*w_1^-2 - 5239264*w_1^-1 - 6790912*w_0^-1 - 1685040*w_0^-2*w_1 - 20672*w_0^-3*w_1^2 - 5632*w_0*w_1^-3 - 866700*w_1^-2 - 3969024*w_0^-1*w_1^-1 - 3033280*w_0^-2 - 309760*w_0^-3*w_1 - 246*w_0^-4*w_1^2 - 11616*w_1^-3 - 628032*w_0^-1*w_1^-2 - 1685040*w_0^-2*w_1^-1 - 628032*w_0^-3 - 11616*w_0^-4*w_1 + w_1^-4 - 5632*w_0^-1*w_1^-3 - 196896*w_0^-2*w_1^-2 - 309760*w_0^-3*w_1^-1 - 34670*w_0^-4 - 1232*w_0^-2*w_1^-3 - 20672*w_0^-3*w_1^-2 - 11616*w_0^-4*w_1^-1 + 64*w_0^-5 - 246*w_0^-4*w_1^-2)*q^4 + O(q^5)

            sage: from weilrep import *
            sage: f = JacobiForms(3).weak_forms_basis(3, 6)[0]
            sage: f ** f
            w_0*w_1 - w_0*w_1^-1 - w_0^-1*w_1 + w_0^-1*w_1^-1 + (56*w_0^2*w_1 + 56*w_0*w_1^2 + 256*w_0*w_1 - 56*w_0^2*w_1^-1 - 56*w_0^-1*w_1^2 - 256*w_0*w_1^-1 - 256*w_0^-1*w_1 - 56*w_0*w_1^-2 - 56*w_0^-2*w_1 + 256*w_0^-1*w_1^-1 + 56*w_0^-1*w_1^-2 + 56*w_0^-2*w_1^-1)*q + (-w_0^5*w_1 - w_0*w_1^5 - 56*w_0^4*w_1 - 56*w_0*w_1^4 + w_0^5*w_1^-1 + 3136*w_0^2*w_1^2 + w_0^-1*w_1^5 + 56*w_0^4*w_1^-1 + 7856*w_0^2*w_1 + 7856*w_0*w_1^2 + 56*w_0^-1*w_1^4 + 18410*w_0*w_1 - 7856*w_0^2*w_1^-1 - 7856*w_0^-1*w_1^2 - 3136*w_0^2*w_1^-2 - 18410*w_0*w_1^-1 - 18410*w_0^-1*w_1 - 3136*w_0^-2*w_1^2 - 7856*w_0*w_1^-2 - 7856*w_0^-2*w_1 + 18410*w_0^-1*w_1^-1 + 56*w_0*w_1^-4 + 7856*w_0^-1*w_1^-2 + 7856*w_0^-2*w_1^-1 + 56*w_0^-4*w_1 + w_0*w_1^-5 + 3136*w_0^-2*w_1^-2 + w_0^-5*w_1 - 56*w_0^-1*w_1^-4 - 56*w_0^-4*w_1^-1 - w_0^-1*w_1^-5 - w_0^-5*w_1^-1)*q^2 + (-56*w_0^5*w_1^2 - 56*w_0^2*w_1^5 - 256*w_0^5*w_1 - 3136*w_0^4*w_1^2 - 3136*w_0^2*w_1^4 - 256*w_0*w_1^5 - 7856*w_0^4*w_1 - 7856*w_0*w_1^4 + 256*w_0^5*w_1^-1 + 77056*w_0^2*w_1^2 + 256*w_0^-1*w_1^5 + 56*w_0^5*w_1^-2 + 7856*w_0^4*w_1^-1 + 147736*w_0^2*w_1 + 147736*w_0*w_1^2 + 7856*w_0^-1*w_1^4 + 56*w_0^-2*w_1^5 + 3136*w_0^4*w_1^-2 + 267776*w_0*w_1 + 3136*w_0^-2*w_1^4 - 147736*w_0^2*w_1^-1 - 147736*w_0^-1*w_1^2 - 77056*w_0^2*w_1^-2 - 267776*w_0*w_1^-1 - 267776*w_0^-1*w_1 - 77056*w_0^-2*w_1^2 - 147736*w_0*w_1^-2 - 147736*w_0^-2*w_1 + 3136*w_0^2*w_1^-4 + 267776*w_0^-1*w_1^-1 + 3136*w_0^-4*w_1^2 + 56*w_0^2*w_1^-5 + 7856*w_0*w_1^-4 + 147736*w_0^-1*w_1^-2 + 147736*w_0^-2*w_1^-1 + 7856*w_0^-4*w_1 + 56*w_0^-5*w_1^2 + 256*w_0*w_1^-5 + 77056*w_0^-2*w_1^-2 + 256*w_0^-5*w_1 - 7856*w_0^-1*w_1^-4 - 7856*w_0^-4*w_1^-1 - 256*w_0^-1*w_1^-5 - 3136*w_0^-2*w_1^-4 - 3136*w_0^-4*w_1^-2 - 256*w_0^-5*w_1^-1 - 56*w_0^-2*w_1^-5 - 56*w_0^-5*w_1^-2)*q^3 + (w_0^5*w_1^5 + 56*w_0^5*w_1^4 + 56*w_0^4*w_1^5 + w_0^7*w_1 + 3136*w_0^4*w_1^4 + w_0*w_1^7 - 7856*w_0^5*w_1^2 - 7856*w_0^2*w_1^5 - w_0^7*w_1^-1 - 18410*w_0^5*w_1 - 77056*w_0^4*w_1^2 - 77056*w_0^2*w_1^4 - 18410*w_0*w_1^5 - w_0^-1*w_1^7 - 147736*w_0^4*w_1 - 147736*w_0*w_1^4 + 18410*w_0^5*w_1^-1 + 803072*w_0^2*w_1^2 + 18410*w_0^-1*w_1^5 + 7856*w_0^5*w_1^-2 + 147736*w_0^4*w_1^-1 + 1320816*w_0^2*w_1 + 1320816*w_0*w_1^2 + 147736*w_0^-1*w_1^4 + 7856*w_0^-2*w_1^5 + 77056*w_0^4*w_1^-2 + 2134237*w_0*w_1 + 77056*w_0^-2*w_1^4 - 56*w_0^5*w_1^-4 - 1320816*w_0^2*w_1^-1 - 1320816*w_0^-1*w_1^2 - 56*w_0^-4*w_1^5 - w_0^5*w_1^-5 - 3136*w_0^4*w_1^-4 - 803072*w_0^2*w_1^-2 - 2134237*w_0*w_1^-1 - 2134237*w_0^-1*w_1 - 803072*w_0^-2*w_1^2 - 3136*w_0^-4*w_1^4 - w_0^-5*w_1^5 - 56*w_0^4*w_1^-5 - 1320816*w_0*w_1^-2 - 1320816*w_0^-2*w_1 - 56*w_0^-5*w_1^4 + 77056*w_0^2*w_1^-4 + 2134237*w_0^-1*w_1^-1 + 77056*w_0^-4*w_1^2 + 7856*w_0^2*w_1^-5 + 147736*w_0*w_1^-4 + 1320816*w_0^-1*w_1^-2 + 1320816*w_0^-2*w_1^-1 + 147736*w_0^-4*w_1 + 7856*w_0^-5*w_1^2 + 18410*w_0*w_1^-5 + 803072*w_0^-2*w_1^-2 + 18410*w_0^-5*w_1 - 147736*w_0^-1*w_1^-4 - 147736*w_0^-4*w_1^-1 - w_0*w_1^-7 - 18410*w_0^-1*w_1^-5 - 77056*w_0^-2*w_1^-4 - 77056*w_0^-4*w_1^-2 - 18410*w_0^-5*w_1^-1 - w_0^-7*w_1 - 7856*w_0^-2*w_1^-5 - 7856*w_0^-5*w_1^-2 + w_0^-1*w_1^-7 + 3136*w_0^-4*w_1^-4 + w_0^-7*w_1^-1 + 56*w_0^-4*w_1^-5 + 56*w_0^-5*w_1^-4 + w_0^-5*w_1^-5)*q^4 + (256*w_0^5*w_1^5 + 56*w_0^7*w_1^2 + 7856*w_0^5*w_1^4 + 7856*w_0^4*w_1^5 + 56*w_0^2*w_1^7 + 256*w_0^7*w_1 + 77056*w_0^4*w_1^4 + 256*w_0*w_1^7 - 147736*w_0^5*w_1^2 - 147736*w_0^2*w_1^5 - 256*w_0^7*w_1^-1 - 267776*w_0^5*w_1 - 803072*w_0^4*w_1^2 - 803072*w_0^2*w_1^4 - 267776*w_0*w_1^5 - 256*w_0^-1*w_1^7 - 56*w_0^7*w_1^-2 - 1320816*w_0^4*w_1 - 1320816*w_0*w_1^4 - 56*w_0^-2*w_1^7 + 267776*w_0^5*w_1^-1 + 5226496*w_0^2*w_1^2 + 267776*w_0^-1*w_1^5 + 147736*w_0^5*w_1^-2 + 1320816*w_0^4*w_1^-1 + 8008192*w_0^2*w_1 + 8008192*w_0*w_1^2 + 1320816*w_0^-1*w_1^4 + 147736*w_0^-2*w_1^5 + 803072*w_0^4*w_1^-2 + 12051200*w_0*w_1 + 803072*w_0^-2*w_1^4 - 7856*w_0^5*w_1^-4 - 8008192*w_0^2*w_1^-1 - 8008192*w_0^-1*w_1^2 - 7856*w_0^-4*w_1^5 - 256*w_0^5*w_1^-5 - 77056*w_0^4*w_1^-4 - 5226496*w_0^2*w_1^-2 - 12051200*w_0*w_1^-1 - 12051200*w_0^-1*w_1 - 5226496*w_0^-2*w_1^2 - 77056*w_0^-4*w_1^4 - 256*w_0^-5*w_1^5 - 7856*w_0^4*w_1^-5 - 8008192*w_0*w_1^-2 - 8008192*w_0^-2*w_1 - 7856*w_0^-5*w_1^4 + 803072*w_0^2*w_1^-4 + 12051200*w_0^-1*w_1^-1 + 803072*w_0^-4*w_1^2 + 147736*w_0^2*w_1^-5 + 1320816*w_0*w_1^-4 + 8008192*w_0^-1*w_1^-2 + 8008192*w_0^-2*w_1^-1 + 1320816*w_0^-4*w_1 + 147736*w_0^-5*w_1^2 + 267776*w_0*w_1^-5 + 5226496*w_0^-2*w_1^-2 + 267776*w_0^-5*w_1 - 56*w_0^2*w_1^-7 - 1320816*w_0^-1*w_1^-4 - 1320816*w_0^-4*w_1^-1 - 56*w_0^-7*w_1^2 - 256*w_0*w_1^-7 - 267776*w_0^-1*w_1^-5 - 803072*w_0^-2*w_1^-4 - 803072*w_0^-4*w_1^-2 - 267776*w_0^-5*w_1^-1 - 256*w_0^-7*w_1 - 147736*w_0^-2*w_1^-5 - 147736*w_0^-5*w_1^-2 + 256*w_0^-1*w_1^-7 + 77056*w_0^-4*w_1^-4 + 256*w_0^-7*w_1^-1 + 56*w_0^-2*w_1^-7 + 7856*w_0^-4*w_1^-5 + 7856*w_0^-5*w_1^-4 + 56*w_0^-7*w_1^-2 + 256*w_0^-5*w_1^-5)*q^5 + O(q^6)

        """
        if other in ZZ:
            return JacobiForm(self.weight() * other, self.index_matrix() * other, self.fourier_expansion()**other, w_scale = self.scale())
        elif not isinstance(other, JacobiForm):
            raise ValueError('Cannot multiply these objects')
        S1 = self.index_matrix()
        S2 = other.index_matrix()
        bigS = block_diagonal_matrix([S1, S2])
        K = self.base_ring().base_ring()
        rb = LaurentPolynomialRing(K, list(var('w_%d' % i) for i in range(bigS.nrows())))
        r, q = PowerSeriesRing(rb, 'q').objgen()
        g = rb.gens()
        e1 = S1.nrows()
        e2 = S2.nrows()
        sf, f = self.qexp(), other.qexp()
        scale = 1
        if self.scale() == 2 or other.scale() == 2:
            r0 = self.base_ring()
            scale = 2
            if self.scale() == 1:
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r0.gens()}))
            if other.scale() == 1:
                f = f.map_coefficients(lambda x: x.subs({y: y*y for y in r0.gens()}))
        #val = other.valuation()
        jf = [rb(f[i]).subs({g[j]:g[j+e1] for j in range(e2)}) for i in range(f.valuation(), f.prec())]
        return JacobiForm(self.weight() + other.weight(), bigS, r(sf) * r(jf) + O(q**other.precision()), w_scale = scale)

    def __eq__(self, other):
        sf, of = self.qexp(), other.qexp()
        if self.scale() == 2 or other.scale() == 2:
            r = self.base_ring()
            if self.scale() == 1:
                try:
                    sf = sf.power_series()
                except AttributeError:
                    pass
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            if other.scale() == 1:
                try:
                    of = of.power_series()
                except AttributeError:
                    pass
                of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
        return sf == of

    ## other operations

    def borcherds_lift(self):
        r"""
        Compute the Borcherds lift of a Jacobi form.

        This converts to a vector-valued modular form and computes the Borcherds lift as in the 'lifts.py' file. The Jacobi form should have weight zero; it does not need to be holomorphic (or even a weak Jacobi form).
        """
        return self.theta_decomposition().borcherds_lift()

    def gritsenko_lift(self):
        r"""
        Compute the Gritsenko lift.

        We first try to compute the additive theta-lift of the underlying vector-valued modular form. If this is not available then we compute the Gritsenko lift directly from Gritsenko's definition in terms of hecke V-operators.

        EXAMPLES::

            sage: from weilrep import *
            sage: JacobiForms(37).basis(2, 5)[0].gritsenko_lift()
            (r^-12 - 3*r^-11 + 6*r^-9 - r^-8 - 4*r^-7 - 4*r^-6 + 3*r^-5 + 3*r^-4 + 4*r^-2 - 2*r^-1 - 6 - 2*r + 4*r^2 + 3*r^4 + 3*r^5 - 4*r^6 - 4*r^7 - r^8 + 6*r^9 - 3*r^11 + r^12)*q*s + (-r^-17 + 2*r^-16 + r^-15 - 3*r^-14 + r^-13 - 2*r^-12 + r^-11 + 6*r^-8 - 4*r^-7 + 2*r^-6 - r^-5 - 6*r^-4 - r^-3 + r^-2 + 4*r^-1 + 4*r + r^2 - r^3 - 6*r^4 - r^5 + 2*r^6 - 4*r^7 + 6*r^8 + r^11 - 2*r^12 + r^13 - 3*r^14 + r^15 + 2*r^16 - r^17)*q^2*s + (-r^-17 + 2*r^-16 + r^-15 - 3*r^-14 + r^-13 - 2*r^-12 + r^-11 + 6*r^-8 - 4*r^-7 + 2*r^-6 - r^-5 - 6*r^-4 - r^-3 + r^-2 + 4*r^-1 + 4*r + r^2 - r^3 - 6*r^4 - r^5 + 2*r^6 - 4*r^7 + 6*r^8 + r^11 - 2*r^12 + r^13 - 3*r^14 + r^15 + 2*r^16 - r^17)*q*s^2 + (r^-21 - r^-20 - r^-19 - 2*r^-18 + 2*r^-17 + 3*r^-16 + r^-15 - 3*r^-13 + r^-12 - 2*r^-11 - 4*r^-9 + 2*r^-7 + 4*r^-6 + 3*r^-5 + 2*r^-3 - 2*r^-2 - r^-1 - 6 - r - 2*r^2 + 2*r^3 + 3*r^5 + 4*r^6 + 2*r^7 - 4*r^9 - 2*r^11 + r^12 - 3*r^13 + r^15 + 3*r^16 + 2*r^17 - 2*r^18 - r^19 - r^20 + r^21)*q^3*s + (2*r^-23 - 3*r^-22 - 2*r^-21 + 2*r^-20 - r^-19 + 6*r^-18 - r^-17 - 2*r^-15 - 4*r^-14 + 4*r^-13 - 6*r^-12 + r^-11 + 3*r^-10 + 3*r^-9 - r^-7 - 3*r^-5 + 4*r^-4 + r^-3 - 2*r^-2 - r^-1 - r - 2*r^2 + r^3 + 4*r^4 - 3*r^5 - r^7 + 3*r^9 + 3*r^10 + r^11 - 6*r^12 + 4*r^13 - 4*r^14 - 2*r^15 - r^17 + 6*r^18 - r^19 + 2*r^20 - 2*r^21 - 3*r^22 + 2*r^23)*q^2*s^2 + (r^-21 - r^-20 - r^-19 - 2*r^-18 + 2*r^-17 + 3*r^-16 + r^-15 - 3*r^-13 + r^-12 - 2*r^-11 - 4*r^-9 + 2*r^-7 + 4*r^-6 + 3*r^-5 + 2*r^-3 - 2*r^-2 - r^-1 - 6 - r - 2*r^2 + 2*r^3 + 3*r^5 + 4*r^6 + 2*r^7 - 4*r^9 - 2*r^11 + r^12 - 3*r^13 + r^15 + 3*r^16 + 2*r^17 - 2*r^18 - r^19 - r^20 + r^21)*q*s^3 + O(q, s)^5
        """
        try:
            return self.modform().gritsenko_lift()
        except AttributeError:
            from .lifts import OrthogonalModularForms
            return OrthogonalModularForms(self.weilrep()).modular_form_from_fourier_jacobi_expansion([self.hecke_V(N) for N in range(self.precision())])

    def hecke_P(self, N):
        ##fix this!!
        return self.theta_decomposition().hecke_P().jacobi_form()

    def hecke_T(self, N):
        ##fix this!!
        r"""
        Apply the Nth Hecke T-operator.

        NOTE: this applies the T-operator to self's theta-decomposition and computes the Jacobi form from that.

        INPUT:
        - ``N`` -- a natural number

        OUTPUT: Jacobi form of the same weight and index
        """
        return self.theta_decomposition().hecke_T(N).jacobi_form()

    def hecke_U(self, N):
        r"""
        Apply the Nth Hecke U-operator.

        NOTE: same as pullback(self, A) where A = N * identity_matrix

        INPUT:
        - ``N`` -- a natural number

        OUTPUT: JacobiForm of the same weight and of index N^2 * self.index()

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).hecke_U(3)
            1 + (w^-6 + 56*w^-3 + 126 + 56*w^3 + w^6)*q + (126*w^-6 + 576*w^-3 + 756 + 576*w^3 + 126*w^6)*q^2 + (56*w^-9 + 756*w^-6 + 1512*w^-3 + 2072 + 1512*w^3 + 756*w^6 + 56*w^9)*q^3 + (w^-12 + 576*w^-9 + 2072*w^-6 + 4032*w^-3 + 4158 + 4032*w^3 + 2072*w^6 + 576*w^9 + w^12)*q^4 + O(q^5)

        """
        if N == 1:
            return self
        elif N <= 0:
            return NotImplemented
        scale = self.scale()
        if scale == 2 and not N % 2:
            return JacobiForm(self.weight(), 4 * self.index_matrix(), self.fourier_expansion(), w_scale = 1).hecke_U(N // 2)
        S = self.index_matrix()
        e = S.nrows()
        Rb = self.base_ring()
        F = self.q_coefficients()
        R, q = PowerSeriesRing(Rb, 'q').objgen()
        f = []
        val = self.valuation()
        f = [g.subs({Rb('w_%d' % j): Rb('w_%d' % j)**N for j in range(e)}) for g in F]
        return JacobiForm(self.weight(), N * N * S, q**val * R(f) + O(q**self.precision()), w_scale = self.scale())

    def hecke_V(self, N):
        r"""
        Apply the Nth Hecke V-operator.

        NOTE: if self has precision 'prec' then self.hecke_V(N) has precision floor(prec / N)

        INPUT:
        - ``N`` -- a natural number

        OUTPUT: JacobiForm of the same weight and of index N * self.index()

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).hecke_V(3)
            28 + (56*w^-3 + 756*w^-2 + 1512*w^-1 + 2072 + 1512*w + 756*w^2 + 56*w^3)*q + O(q^2)

        """
        if N == 1:
            return self
        elif N == 0:
            c = self.q_coefficients()[0]
            c = QQ(c.constant_coefficient())
            k = self.weight()
            return JacobiForm(k, matrix([]), c * eisenstein_series_qexp(k, self.precision()))
        S = self.index_matrix()
        e = S.nrows()
        k_1 = self.weight() - 1
        Rb = self.base_ring()
        F = self.q_coefficients()
        q, = PowerSeriesRing(Rb, 'q').gens()
        val = min(self.valuation(), 0)
        max_prec = -(self.precision() // -N)
        f = O(q**(max_prec + val))
        for a in divisors(N):
            d = N // ZZ(a)
            sub_a = {Rb('w_%d' % j): Rb('w_%d' % j)**a for j in range(e)}
            f += sum([a**(k_1) * q**(a * (i + val)) * F[d * (i + val) - val].subs(sub_a) for i in range(max(0, ceil(val * (1 / d - 1))), min(-((len(F) + (1 - d) * val) // -d), max_prec))])
        return JacobiForm(k_1 + 1, N * S, f, w_scale = self.scale())

    def pullback(self, A):  #return self evaluated at tau, A*z
        r"""
        Apply a linear map to self's elliptic variables.

        INPUT:
        - ``A`` -- a matrix in which the number of columns equals self's number of variables.

        OUTPUT: the JacobiForm f(tau, A*z) if self is the JacobiForm f(tau, z)

        EXAMPLES::

            sage: from weilrep import *
            sage: j = jacobi_eisenstein_series(4, matrix([[2,1],[1,4]]), 5)
            sage: j.pullback(matrix([[1, 2]]))
            1 + (7*w^-5 + 8*w^-4 + 21*w^-3 + 35*w^-2 + 28*w^-1 + 42 + 28*w + 35*w^2 + 21*w^3 + 8*w^4 + 7*w^5)*q + (w^-9 + 7*w^-8 + 35*w^-7 + 56*w^-6 + 106*w^-5 + 147*w^-4 + 182*w^-3 + 182*w^-2 + 252*w^-1 + 224 + 252*w + 182*w^2 + 182*w^3 + 147*w^4 + 106*w^5 + 56*w^6 + 35*w^7 + 7*w^8 + w^9)*q^2 + (21*w^-10 + 49*w^-9 + 126*w^-8 + 168*w^-7 + 294*w^-6 + 315*w^-5 + 462*w^-4 + 469*w^-3 + 609*w^-2 + 567*w^-1 + 560 + 567*w + 609*w^2 + 469*w^3 + 462*w^4 + 315*w^5 + 294*w^6 + 168*w^7 + 126*w^8 + 49*w^9 + 21*w^10)*q^3 + (w^-13 + 21*w^-12 + 70*w^-11 + 154*w^-10 + 315*w^-9 + 400*w^-8 + 623*w^-7 + 756*w^-6 + 952*w^-5 + 966*w^-4 + 1281*w^-3 + 1162*w^-2 + 1366*w^-1 + 1386 + 1366*w + 1162*w^2 + 1281*w^3 + 966*w^4 + 952*w^5 + 756*w^6 + 623*w^7 + 400*w^8 + 315*w^9 + 154*w^10 + 70*w^11 + 21*w^12 + w^13)*q^4 + O(q^5)

        """
        f = self.fourier_expansion()
        S = self.index_matrix()
        e = S.nrows()
        new_e = A.nrows()
        Rb = LaurentPolynomialRing(QQ, list(var('w_%d' % i) for i in range(e)))
        Rb_new = LaurentPolynomialRing(QQ, list(var('w_%d' % i) for i in range(new_e)))
        R, q = PowerSeriesRing(Rb_new, 'q').objgen()
        val = f.valuation()
        prec = f.prec()
        if new_e > 1:
            sub_R = {Rb('w_%d' % j): Rb_new.monomial(*x) for j, x in enumerate(A.columns())}
        else:
            w, = Rb_new.gens()
            sub_R = {Rb('w_%d' % j): w**A[0, j] for j in range(e)}
        jf_new = [f[i].subs(sub_R) for i in range(val, prec)]
        return JacobiForm(self.weight(), A * S * A.transpose(), q**val * R(jf_new) + O(q**f.prec()), w_scale = self.scale())

    substitute = pullback

    def serre_derivative(self):
        h = self.scale()
        if h == 1:
            return self.theta_decomposition().serre_derivative().jacobi_form()
        else:
            h = self.hecke_U(2).serre_derivative()
            return JacobiForm(self.weight() + 2, self.index_matrix(), h.fourier_expansion(), w_scale = 2)

    def substitute_zero(self, indices=None):
        r"""
        Set some of our elliptic variables equal to zero.

        This evaluates the Jacobi form f(tau, z_0, ..., z_d) at z_i = 0 for some subset of indices i in {0,...,d}.

        NOTE: this is a special case of self.pullback(A) for a particular choice of A

        INPUT:
        - ``indices`` -- a list of integers between 0 and d = self.nvars()

        OUTPUT: JacobiForm

        EXAMPLES::

            sage: from weilrep import *
            sage: j = jacobi_eisenstein_series(4, matrix([[2, 1],[1, 4]]), 5)
            sage: j.substitute_zero([0])
            1 + (14*w^-2 + 64*w^-1 + 84 + 64*w + 14*w^2)*q + (w^-4 + 64*w^-3 + 280*w^-2 + 448*w^-1 + 574 + 448*w + 280*w^2 + 64*w^3 + w^4)*q^2 + (84*w^-4 + 448*w^-3 + 840*w^-2 + 1344*w^-1 + 1288 + 1344*w + 840*w^2 + 448*w^3 + 84*w^4)*q^3 + (64*w^-5 + 574*w^-4 + 1344*w^-3 + 2368*w^-2 + 2688*w^-1 + 3444 + 2688*w + 2368*w^2 + 1344*w^3 + 574*w^4 + 64*w^5)*q^4 + O(q^5)

        """
        if indices is None:
            indices = list(range(self.nvars()))
        f = self.fourier_expansion()
        S = self.index_matrix()
        e = S.nrows()
        Rb = LaurentPolynomialRing(QQ, list(var('w_%d' % i) for i in range(e)))
        try:
            Rb_new = LaurentPolynomialRing(QQ, list(var('w_%d' % i) for i in range(e - len(indices))))
        except ValueError:
            Rb_new = QQ
        R_new, q = PowerSeriesRing(Rb_new, 'q').objgen()
        val = f.valuation()
        prec = f.prec()
        L = [k for k in range(e) if k not in indices]
        d = {Rb('w_%d' % j): 1 for j in indices}
        d.update({Rb('w_%d' % k): Rb_new('w_%d' % i) for i, k in enumerate(L)})
        jf_sub = R_new([f[i].subs(d) for i in range(val, prec)])
        k = self.character()._k()
        try:
            f = self.modform()
            from .weilrep_modular_forms_class import smf_eta
            if k:
                f *= smf_eta(prec = f.precision())**(24 - k)
            V = []
            for i in L:
                v = vector([0] * e)
                v[i] = 1
                V.append(v)
            f = f.pullback(V)
            if k:
                f /= smf_eta(prec = f.precision())**(24 - k)
        except AttributeError:
            f = None
        return JacobiForm(self.weight(), S[L, L], q**val * jf_sub + O(q**prec), modform = f, scale = self.scale())

    def derivative_at_zero(self, index):
        y = self.base_ring().gens()[index]
        f = JacobiForm(self.weight() + 1, self.index_matrix(), self.fourier_expansion().map_coefficients(lambda x: x.derivative(y)), w_scale = self.scale()) / self.scale()
        return f.substitute_zero([index])

#special jacobi forms

def jacobi_eisenstein_series(k, m, prec = 20, allow_small_weight=False):
    r"""
    Compute the Jacobi Eisenstein series.

    See JacobiForms method eisenstein_series()
    """
    return JacobiForms(m).eisenstein_series(k, prec, allow_small_weight=allow_small_weight)

def jacobi_theta_series(prec = 20):
    return theta_block([1], 0, prec)

def theta_block(a, n, prec, jacobiforms = None):  #theta block corresponding to a=[a1,...,ar] and eta^n
    r"""
    Compute theta blocks.

    This takes a list of nonzero integers [a_1,...,a_r] and an integer n as input and produces the theta block

    theta(tau, a_1 z) * ... * theta(tau, a_r z) * eta(tau)^n

    if the result is a Jacobi form without character. (If the theta block has a character then a ValueError is raised instead.) The a_1,...,a_r do not need to be distinct.

    INPUT:
    - ``a`` -- a list [a_1,...,a_r] of nonzero integers (repeats are allowed)
    - ``n`` -- an integer
    - ``prec`` -- precision
    Optional:
    - ``jacobiforms`` -- a JacobiForms instance associated to this theta block (default None)

    OUTPUT: JacobiForm

    EXAMPLES::

        sage: from weilrep import *
        sage: theta_block([1]*8, 0, 5)
        (w^-4 - 8*w^-3 + 28*w^-2 - 56*w^-1 + 70 - 56*w + 28*w^2 - 8*w^3 + w^4)*q + (-8*w^-5 + 56*w^-4 - 168*w^-3 + 288*w^-2 - 336*w^-1 + 336 - 336*w + 288*w^2 - 168*w^3 + 56*w^4 - 8*w^5)*q^2 + (28*w^-6 - 168*w^-5 + 420*w^-4 - 616*w^-3 + 756*w^-2 - 1008*w^-1 + 1176 - 1008*w + 756*w^2 - 616*w^3 + 420*w^4 - 168*w^5 + 28*w^6)*q^3 + (-56*w^-7 + 288*w^-6 - 616*w^-5 + 896*w^-4 - 1400*w^-3 + 2016*w^-2 - 2024*w^-1 + 1792 - 2024*w + 2016*w^2 - 1400*w^3 + 896*w^4 - 616*w^5 + 288*w^6 - 56*w^7)*q^4 + O(q^5)
    """
    if any(not a for a in a):
        raise ValueError('List "a" contains zeros')
    qval = Integer(3 * len(a) + n)
    qval, qshift = qval.quo_rem(24)
    scale = 1
    s = sum(a)
    if s % 2:
        scale, wval = 2, s
    else:
        wval = s // 2
    prec -= qval
    weight = Integer(len(a) + n) / 2
    a0 = Counter(a)
    a0_list = list(a0)
    rb, w_0 = LaurentPolynomialRing(QQ, 'w_0').objgen()
    q, = PowerSeriesRing(rb, 'q', prec).gens()
    thetas = [O(q**prec)] * len(a0_list)  #compute theta functions. we don't generally spend much time on this part so I won't bother caching them
    eta = O(q**prec)
    bound = isqrt(prec + prec + 1 / 4) + 1
    eps = 1 - 2 * (bound % 2)
    for i in range(-bound, bound):
        eps = -eps
        for j, _a in enumerate(a0_list):
            thetas[j] += eps * w_0**(_a * i * scale) * q**binomial(i + 1, 2)
        j = i * (3 * i + 1) / 2
        if j < prec:
            eta += eps * q**ZZ(j)
    S = matrix([[sum([a * a for a in a])]])
    if jacobiforms is None:
        jacobiforms = JacobiForms(S)
    z = prod([theta**a0[a0_list[i]] for i, theta in enumerate(thetas)]) * (eta**n) * q**qval * w_0**wval
    if qshift:
        return JacobiFormWithCharacter(weight, S, z, jacobiforms = jacobiforms, w_scale = scale, qshift = qshift / 24, character = EtaCharacterPower(qshift))
    return JacobiForm(weight, S, z, jacobiforms = jacobiforms, w_scale = scale)

def root_lattice_theta_block(root_system, prec):
    r"""
    The theta block \vartheta_R of Gritsenko--Skoruppa--Zagier associated to a root system R.

    INPUT:
    - ``root_system`` -- a RootSystem. Can also be a list such as ['A', n], ['B', n], etc.
    - ``prec`` -- precision

    OUTPUT: a JacobiForm

    EXAMPLES::
        sage: from weilrep import *
        sage: root_lattice_theta_block(['A', 2], 5)
        (-w_0*w_1 + w_0 + w_1 - w_1^-1 - w_0^-1 + w_0^-1*w_1^-1)*q^(1/3) + (w_0^2*w_1^2 - w_0^2 - w_1^2 + w_1^-2 + w_0^-2 - w_0^-2*w_1^-2)*q^(4/3) + (-w_0^3*w_1^2 - w_0^2*w_1^3 + w_0^3*w_1 + w_0*w_1^3 + w_0^2*w_1^-1 + w_0^-1*w_1^2 - w_0*w_1^-2 - w_0^-2*w_1 - w_0^-1*w_1^-3 - w_0^-3*w_1^-1 + w_0^-2*w_1^-3 + w_0^-3*w_1^-2)*q^(7/3) + (w_0^4*w_1^3 + w_0^3*w_1^4 - w_0^4*w_1 - w_0*w_1^4 - w_0^3*w_1^-1 - w_0^-1*w_1^3 + w_0*w_1^-3 + w_0^-3*w_1 + w_0^-1*w_1^-4 + w_0^-4*w_1^-1 - w_0^-3*w_1^-4 - w_0^-4*w_1^-3)*q^(13/3) + O(q^(16/3))

    """
    from .weilrep_modular_forms_class import smf_eta
    if isinstance(root_system, list):
        root_system = RootSystem(root_system)
    R = root_system.root_lattice()
    P = list(R.positive_roots())
    coords = [matrix([x.dense_coefficient_list()]).transpose() for x in P]
    theta = jacobi_theta_series(prec)
    eta = smf_eta(prec)
    return eta ** (R.rank() - len(P)) * prod(theta.pullback(x) for x in coords)

class JacobiFormWithCharacter(JacobiForm):
    r"""
    Jacobi forms which transform with a power of the eta multiplier under the action of SL_2(ZZ).
    """
    def __init__(self, *args, **kwargs):
        chi = kwargs.pop('character')
        qshift = kwargs.pop('qshift', 0)
        qfloor = floor(qshift)
        if qfloor:
            f = args[2]
            f = f.shift(qfloor)
            qshift -= qfloor
            args = args[0], args[1], f
        super().__init__(*args, **kwargs)
        if chi:
            self.__class__, self.__character, self.__qshift = JacobiFormWithCharacter, chi, qshift
        else:
            self.__class__ = JacobiForm

    def __repr__(self):
        try:
            return self.__string
        except AttributeError:
            x = self.__qshift
            N = min(0, self.valuation())
            e = self.nvars()
            if super().__bool__():
                if e:
                    s = ' + '.join('(%s)*q^(%s)'%(str(w), i+N+x) for i, w in enumerate(super().q_coefficients()) if w) + ' + O(q^(%s))'%(self.precision() + x)
                else:
                    s = ' + '.join('%s*q^(%s)'%(str(w), i+N+x) for i, w in enumerate(super().q_coefficients()) if w) + ' + O(q^(%s))'%(self.precision() + x)
                    s = s.replace('1*', '')
                    s = s.replace('+ -', '- ')
            else:
                s = 'O(q^(%s))'%(self.precision() + x)
            if self.scale() == 2:
                def m(obj):
                    obj_s = obj.string[slice(*obj.span())]
                    x = obj_s[0]
                    if x == '^':
                        u = ZZ(obj_s[1:])/2
                        if u.is_integer():
                            if u == 1:
                                return ''
                            return '^%d'%u
                        return '^(%s)'%u
                    return obj_s + '^(%s)'%sage_one_half
                s = sub(r'((?<!q)\^-?\d+)|w\_\d+(?!\^)', m, s)
            if self.nvars() == 1:
                s = s.replace('w_0', 'w')
            self.__string = s
            return s

    def character(self):
        return self.__character

    def _qshift(self):
        return self.__qshift

    def __add__(self, other):
        if self.character() != other.character():
            return NotImplemented
        f = super().__add__(other)
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    def __sub__(self, other):
        if self.character() != other.character():
            return NotImplemented
        f = super().__sub__(other)
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    def __mul__(self, other):
        f = super().__mul__(other)
        try:
            k = other.character()
            x = self.character() * k
            qshift = self._qshift() + other._qshift()
        except AttributeError:
            x = self.character()
            qshift = self._qshift()
        try:
            qshift += f._qshift()
            x *= f.character()
        except AttributeError:
            pass
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = x, qshift = qshift, w_scale = f.scale())

    __rmul__ = __mul__

    def __div__(self, other):
        f = super().__div__(other)
        try:
            k = other.character()
            x = self.character() / k
            qshift = self._qshift() - other._qshift()
        except AttributeError:
            x = self.character()
            qshift = self._qshift()
        try:
            qshift += f._qshift()
            x *= f.character()
        except AttributeError:
            pass
        if qshift in ZZ:
            f._JacobiForm__fourier_expansion = f.fourier_expansion().shift(qshift)
            f.__qshift = 0
            f.__class__ = JacobiForm
            return f
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = x, qshift = qshift, w_scale = f.scale())

    __truediv__ = __div__

    def __pow__(self, N):
        f = super().__pow__(N)
        qshift = self._qshift() * N
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character()**N, qshift = qshift, w_scale = f.scale())

    def __neg__(self):
        f = super().__neg__()
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    def __rdiv__(self, other):
        return (~self).__mul__(other)

    __rtruediv__ = __rdiv__

    def __invert__(self):
        f = super().__invert__()
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = ~self.character(), qshift = -self._qshift(), w_scale = f.scale())

    def substitute_zero(self, *args):
        f = super().substitute_zero(*args)
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    def pullback(self, *args):
        f = super().pullback(*args)
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    def theta_decomposition(self, **kwargs): #dumb
        try:
            return self._JacobiForm__theta
        except AttributeError:
            chi = self.character()
            k = chi._k()
            from .weilrep_modular_forms_class import smf_eta
            h = smf_eta(self.precision() + 1) ** (24 - k)
            f = (self * h).theta_decomposition(**kwargs) / h
            self._JacobiForm__theta = f
            return f

    def hecke_T(self, N):
        raise NotImplementedError

    def hecke_U(self, N):
        f = super().hecke_U(N)
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    def hecke_V(self, N):
        r"""
        Apply the Nth Hecke V-operator.

        This was defined for Jacobi forms twisted by a power 'D' of the eta character by Cléry and Gritsenko [CG]. The power 'D' must divide 24.
        NOTE: if self has precision 'prec' then self.hecke_V(N) has precision floor(prec / N)

        INPUT:
        - ``N`` -- a natural number that is coprime to (24 / D)

        OUTPUT: JacobiForm of the same weight and of index N * self.index()
        """
        wt = self.weight()
        chi = self.character()
        k = chi._k()
        if 24 % k :
            return NotImplemented
        Q = 24 // k
        if Q > 2:
            K = CyclotomicField(Q)
            zeta = K.gen()
        else:
            K = QQ
            if Q == 2:
                zeta = -1
            else:
                zeta = 1
        g, x, y = XGCD(N, Q)
        Qy = Q * y
        xQxy = x * (1 + Qy)
        if g != 1:
            return NotImplemented
        S = self.index_matrix()
        e = S.nrows()
        f = self.fourier_expansion()
        F = f.padded_list()
        qshift = self._qshift()
        new_shift = x * qshift
        R, q = f.parent().objgen()
        Rb = R.base_ring()
        s = self.scale()
        wt_1 = wt - 1
        prec = self.precision()
        max_prec = -(prec // -N) - 1
        val = min(self.valuation(), 0)
        h = O(q ** (max_prec + val))
        for a in divisors(N):
            d = N // ZZ(a)
            m = matrix([[d * xQxy, -Qy], [Qy, a]])
            multiplier = zeta ** (chi(m) // k)
            sub_a = {Rb('w_%d' % j): Rb('w_%d' % j)**a for j in range(e)}
            s = 0
            for n in range(max(0, ceil(val * (Q / d - 1))), prec):
                if Q * (n + qshift) % d == 0:
                    u = ZZ(a * (n + qshift) / d - new_shift)
                    s += q ** u * F[n].subs(sub_a)
            if wt in ZZ or is_square(a):
                h += a**wt_1 * multiplier * s
        return JacobiFormWithCharacter(wt, N * S, h, w_scale = self.scale(), character = chi**x, qshift = new_shift)

    def reduce_precision(self, new_prec):
        return JacobiFormWithCharacter(self.weight(), self.index_matrix(), self.qexp().add_bigoh(new_prec), modform = self.modform(error = False), weilrep = self.weilrep(), w_scale = self.scale(), character = self.character(), qshift = self._qshift())

    def _rescale(self, a):
        f = self.qexp()
        e = self.nvars()
        rb_w = f.base_ring()
        d = {rb_w('w_%d' % j): rb_w('w_%d' % j)**a for j in range(e)}
        f = f.map_coefficients(lambda x: x.subs(d))
        return JacobiFormWithCharacter(self.weight(), self.index_matrix(), f, modform = self.modform(error = False), weilrep = self.weilrep(), w_scale = self.scale() * a, character = self.character(), qshift = self._qshift())

    def serre_derivative(self):
        f = super().serre_derivative()
        return JacobiFormWithCharacter(f.weight(), f.index_matrix(), f.fourier_expansion(), modform = f.modform(error=False), weilrep = f.weilrep(), character = self.character(), qshift = self._qshift(), w_scale = f.scale())

    ## lift

    def gritsenko_lift(self, prec=None):
        from .lifts import OrthogonalModularForms
        chi = self.character()
        k = chi._k()
        if 24 % k or not self.is_holomorphic():
            return NotImplemented
        Q = 24 // k
        if prec is None:
            prec = self.precision()
        else:
            prec = min(prec, self.precision())
        qshift = self._qshift()
        if qshift == 0:
            c = self.theta_decomposition().fourier_expansion()[0][2][0]
        else:
            c = 0
        prec_Q = prec // Q
        fj = [None] * (prec_Q + 1)
        wt = self.weight()
        J = JacobiForms([])
        if c:
            fj[0] = c * J.eisenstein_series(wt, prec)
        else:
            fj[0] = J.zero(wt, prec)
        for n in range(prec_Q):
            m = Q * n + 1
            fj[n + 1] = self.hecke_V(m)
        return OrthogonalModularForms(self.index_matrix() / qshift).modular_form_from_fourier_jacobi_expansion(fj)


## Extra functions

def jf_rankin_cohen(N, f1, f2, direct = False):
    r"""
    Compute the Nth Rankin--Cohen bracket of two Jacobi forms.

    This computes the Rankin--Cohen bracket on the level of vector-valued modular forms and converts the result into a Jacobi form. If the optional parameter "direct" is set to True then all multiplications are carried out with separate elliptic variables; the result is a Jacobi form whose index is the direct sum of the indices of f1 and f2 (understood as lattices)

    INPUT:
    - ``N`` -- a natural number (including 0)
    - ``f1`` -- a JacobiForm
    - ``f2`` -- a JacobiForm
    - ``direct`` -- boolean (default False)

    OUTPUT: A JacobiForm. If direct=False and f1 has weight k1 and index m1 and f2 has weight k2 and index m2, then the result has weight k1+k2+2N and index m1+m2
    """
    from .weilrep_modular_forms_class import rankin_cohen
    F = rankin_cohen(N, f1.theta_decomposition(), f2.theta_decomposition())
    if direct:
        return F.jacobi_form()
    nvars1 = f1.nvars()
    nvars2 = f2.nvars()
    if nvars1 != nvars2:
        raise ValueError('The number of variables do not match.')
    I = identity_matrix(ZZ, nvars1)
    zero = matrix(ZZ, nvars1)
    A = block_matrix([[I, zero], [I, I]])
    F = F.conjugate(A)
    for _ in range(nvars1):
        F = F.theta_contraction()
    return F.jacobi_form()

def _jf_relations(X):
    r"""
    Compute all relations among a set of Jacobi forms.
    This should no longer be called directly. Use weilrep_misc.relations instead.
    """
    Xref = X[0]
    if Xref.scale() == 2:
        return _jf_relations([x.hecke_U(2) for x in X])
    if not all(x.weight() == Xref.weight() and x.index() == Xref.index() for x in X[1:]):
        raise ValueError('Incompatible Jacobi forms')
    X = [x.theta_decomposition() for x in X]
    val = min(x.valuation() for x in X)
    prec = min(x.precision() for x in X)
    M = matrix([x.coefficient_vector(starting_from = val, ending_with = prec) for x in X])
    return M.kernel()