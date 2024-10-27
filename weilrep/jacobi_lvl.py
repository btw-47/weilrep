r"""

Sage code for Jacobi forms on congruence subgroups of SL2(ZZ)

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2023-2024 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from re import sub

from sage.arith.functions import lcm
from sage.arith.misc import divisors, gcd, moebius
from sage.arith.srange import srange
from sage.functions.other import ceil, frac, floor
from sage.matrix.constructor import matrix
from sage.matrix.special import block_diagonal_matrix
from sage.misc.functional import isqrt
from sage.modular.arithgroup.congroup_gamma0 import Gamma0_constructor
from sage.modular.arithgroup.congroup_gamma1 import Gamma1_constructor
from sage.modular.modform.element import ModularFormElement
from sage.modules.free_module import span
from sage.modules.free_module_element import vector
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field_base import NumberField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing, LaurentPolynomialRing_generic
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RealField_class

try:
    from sage.rings.complex_mpfr import ComplexField_class
except ModuleNotFoundError:
    from sage.rings.complex_field import ComplexField_class


from .jacobi_forms_class import JacobiForm, JacobiForms, JacobiFormWithCharacter
from .lorentz import II
from .morphisms import WeilRepAutomorphismGroup
from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis


class JacobiFormsGamma0:

    def __init__(self, level, index_matrix, **kwargs):
        self.__level = level
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
        self.__rank = self.__index_matrix.nrows()

    def __repr__(self):
        return "Jacobi forms of level Gamma0(%s) and index %s" % (self.__level, self.index())

    def index(self):
        r"""
        Return self's index (as a scalar if one elliptic variable).
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
        return self.__index_matrix

    def level(self):
        return self.__level

    def rank(self):
        return self.__rank

    # ## dimension ##
    def cusp_forms_dimension(self, k):
        S = self.index_matrix()
        r = S.nrows()
        wt = QQ(k - r / 2)
        if wt <= 2:
            return len(self.cusp_forms_basis(k, 0))

        def a(u, N):
            ui = u.inverse_mod(N)

            def b(x):
                return vector([frac(u * x[0])] + list(x[1:-1]) + [frac(ui * x[-1])])
            return b

        def dim(N):
            if N == 1:
                w = WeilRep(S)
                return w.cusp_forms_dimension(wt)
            w = WeilRep(S) + II(N)
            G = [w.automorphism(a(u, N)) for u in srange(N) if gcd(u, N) == 1]
            phi = w.canonical_involution()
            if phi not in G:
                chi = [1]*len(G) + [(-1)**k] * len(G)
                G = G + [g * phi for g in G]
            else:
                if k % 2:
                    return 0
                chi = [1]*len(G)
            return w.invariant_cusp_forms_dimension(wt, G=G, chi=chi)
        N = self.level()
        return sum(moebius(N / d) * dim(d) for d in divisors(N))

    def jacobi_forms_dimension(self, k):
        S = self.index_matrix()
        r = S.nrows()
        wt = QQ(k - r / 2)
        if wt <= 2:
            return len(self.cusp_forms_basis(k, 0))

        def a(u, N):
            ui = u.inverse_mod(N)

            def b(x):
                return vector([frac(u * x[0])] + list(x[1:-1]) + [frac(ui * x[-1])])
            return b

        def dim(N):
            if N == 1:
                w = WeilRep(S)
                return w.cusp_forms_dimension(wt)
            w = WeilRep(S) + II(N)
            G = [w.automorphism(a(u, N)) for u in srange(N) if gcd(u, N) == 1]
            phi = w.canonical_involution()
            if phi not in G:
                chi = [1]*len(G) + [(-1)**k] * len(G)
                G = G + [g * phi for g in G]
            else:
                if k % 2:
                    return 0
                chi = [1]*len(G)
            return w.invariant_forms_dimension(wt, G=G, chi=chi)
        N = self.level()
        return sum(moebius(N / d) * dim(d) for d in divisors(N))
    dimension = jacobi_forms_dimension

    # ## Jacobi forms ##

    def basis(self, k, prec, **kwargs):
        r"""
        Compute a basis of (holomorphic) Jacobi forms of weight k and level \Gamma_0(N) up to precision prec.

        EXAMPLES::

            sage: from weilrep import *
            sage: J = JacobiForms(Gamma0(3), 3)
            sage: J.basis(5, 10)   # Jacobi forms of weight 5 and index 3 and level \Gamma_0(3) up to precision q^10. This is a 2-dimensional space spanned by phi_{-3, 3}*f_8 and phi_{-1, 3}*f_6, where phi_{k, 3} is the unique weak Jacobi form of index 3 and weight k for the full group SL2(ZZ), and f_k is the unique cusp form of level \Gamma_0(3) and weight k. check that they are linear combinations of the echelon-form basis below.
            [(-w^-2 + w^2)*q + (w^-4 - 2*w^-2 + 12*w^-1 - 12*w + 2*w^2 - w^4)*q^2 + (2*w^-4 - 35*w^-2 + 44*w^-1 - 44*w + 35*w^2 - 2*w^4)*q^3 + (-12*w^-5 + 35*w^-4 - 94*w^-2 + 100*w^-1 - 100*w + 94*w^2 - 35*w^4 + 12*w^5)*q^4 + (-44*w^-5 + 94*w^-4 - 218*w^-2 + 268*w^-1 - 268*w + 218*w^2 - 94*w^4 + 44*w^5)*q^5 + (-w^-8 + 12*w^-7 - 100*w^-5 + 218*w^-4 - 408*w^-2 + 476*w^-1 - 476*w + 408*w^2 - 218*w^4 + 100*w^5 - 12*w^7 + w^8)*q^6 + (-2*w^-8 + 44*w^-7 - 268*w^-5 + 408*w^-4 - 728*w^-2 + 952*w^-1 - 952*w + 728*w^2 - 408*w^4 + 268*w^5 - 44*w^7 + 2*w^8)*q^7 + (-35*w^-8 + 100*w^-7 - 476*w^-5 + 728*w^-4 - 1268*w^-2 + 1248*w^-1 - 1248*w + 1268*w^2 - 728*w^4 + 476*w^5 - 100*w^7 + 35*w^8)*q^8 + (w^-10 - 94*w^-8 + 268*w^-7 - 952*w^-5 + 1268*w^-4 - 1862*w^-2 + 2116*w^-1 - 2116*w + 1862*w^2 - 1268*w^4 + 952*w^5 - 268*w^7 + 94*w^8 - w^10)*q^9 + O(q^10), (-w^-1 + w)*q + (7*w^-2 - 8*w^-1 + 8*w - 7*w^2)*q^2 + (w^-5 - 7*w^-4 + 29*w^-2 - 44*w^-1 + 44*w - 29*w^2 + 7*w^4 - w^5)*q^3 + (8*w^-5 - 29*w^-4 + 92*w^-2 - 112*w^-1 + 112*w - 92*w^2 + 29*w^4 - 8*w^5)*q^4 + (-w^-7 + 44*w^-5 - 92*w^-4 + 212*w^-2 - 275*w^-1 + 275*w - 212*w^2 + 92*w^4 - 44*w^5 + w^7)*q^5 + (-8*w^-7 + 112*w^-5 - 212*w^-4 + 427*w^-2 - 456*w^-1 + 456*w - 427*w^2 + 212*w^4 - 112*w^5 + 8*w^7)*q^6 + (7*w^-8 - 44*w^-7 + 275*w^-5 - 427*w^-4 + 758*w^-2 - 891*w^-1 + 891*w - 758*w^2 + 427*w^4 - 275*w^5 + 44*w^7 - 7*w^8)*q^7 + (29*w^-8 - 112*w^-7 + 456*w^-5 - 758*w^-4 + 1212*w^-2 - 1288*w^-1 + 1288*w - 1212*w^2 + 758*w^4 - 456*w^5 + 112*w^7 - 29*w^8)*q^8 + (92*w^-8 - 275*w^-7 + 891*w^-5 - 1212*w^-4 + 1864*w^-2 - 2227*w^-1 + 2227*w - 1864*w^2 + 1212*w^4 - 891*w^5 + 275*w^7 - 92*w^8)*q^9 + O(q^10)]

            sage: from weilrep import *
            sage: J = JacobiForms(Gamma0(2), matrix([[2, -1], [-1, 2]]))
            sage: J.basis(4, 5)   #Jacobi forms of weight 4 and lattice index A_2 and level \Gamma_0(2) up to precision q^5.
            [1 + (w_0*w_1 + w_0^2*w_1^-1 + w_0^-1*w_1^2 + 18 + w_0*w_1^-2 + w_0^-2*w_1 + w_0^-1*w_1^-1)*q + (18*w_0*w_1 + 18*w_0^2*w_1^-1 + 54*w_0 + 54*w_1 + 18*w_0^-1*w_1^2 + 54*w_0*w_1^-1 + 54*w_0^-1*w_1 + 18*w_0*w_1^-2 + 54*w_1^-1 + 54*w_0^-1 + 18*w_0^-2*w_1 + 18*w_0^-1*w_1^-1)*q^2 + (w_0^3 + w_1^3 + 54*w_0^2 + 54*w_1^2 + 27*w_0 + 27*w_1 + w_0^3*w_1^-3 + 54*w_0^2*w_1^-2 + 27*w_0*w_1^-1 + 180 + 27*w_0^-1*w_1 + 54*w_0^-2*w_1^2 + w_0^-3*w_1^3 + 27*w_1^-1 + 27*w_0^-1 + 54*w_1^-2 + 54*w_0^-2 + w_1^-3 + w_0^-3)*q^3 + (w_0^2*w_1^2 + 18*w_0^3 + 54*w_0^2*w_1 + 54*w_0*w_1^2 + 18*w_1^3 + w_0^4*w_1^-2 + 54*w_0^3*w_1^-1 + 27*w_0^2 + 180*w_0*w_1 + 27*w_1^2 + 54*w_0^-1*w_1^3 + w_0^-2*w_1^4 + 54*w_0^3*w_1^-2 + 180*w_0^2*w_1^-1 + 270*w_0 + 270*w_1 + 180*w_0^-1*w_1^2 + 54*w_0^-2*w_1^3 + 18*w_0^3*w_1^-3 + 27*w_0^2*w_1^-2 + 270*w_0*w_1^-1 + 72 + 270*w_0^-1*w_1 + 27*w_0^-2*w_1^2 + 18*w_0^-3*w_1^3 + 54*w_0^2*w_1^-3 + 180*w_0*w_1^-2 + 270*w_1^-1 + 270*w_0^-1 + 180*w_0^-2*w_1 + 54*w_0^-3*w_1^2 + w_0^2*w_1^-4 + 54*w_0*w_1^-3 + 27*w_1^-2 + 180*w_0^-1*w_1^-1 + 27*w_0^-2 + 54*w_0^-3*w_1 + w_0^-4*w_1^2 + 18*w_1^-3 + 54*w_0^-1*w_1^-2 + 54*w_0^-2*w_1^-1 + 18*w_0^-3 + w_0^-2*w_1^-2)*q^4 + O(q^5), (w_0 + w_1 + w_0*w_1^-1 + 2 + w_0^-1*w_1 + w_1^-1 + w_0^-1)*q + (w_0^2 + 2*w_0*w_1 + w_1^2 + 2*w_0^2*w_1^-1 + 6*w_0 + 6*w_1 + 2*w_0^-1*w_1^2 + w_0^2*w_1^-2 + 6*w_0*w_1^-1 + 10 + 6*w_0^-1*w_1 + w_0^-2*w_1^2 + 2*w_0*w_1^-2 + 6*w_1^-1 + 6*w_0^-1 + 2*w_0^-2*w_1 + w_1^-2 + 2*w_0^-1*w_1^-1 + w_0^-2)*q^2 + (w_0^2*w_1 + w_0*w_1^2 + w_0^3*w_1^-1 + 6*w_0^2 + 10*w_0*w_1 + 6*w_1^2 + w_0^-1*w_1^3 + w_0^3*w_1^-2 + 10*w_0^2*w_1^-1 + 16*w_0 + 16*w_1 + 10*w_0^-1*w_1^2 + w_0^-2*w_1^3 + 6*w_0^2*w_1^-2 + 16*w_0*w_1^-1 + 20 + 16*w_0^-1*w_1 + 6*w_0^-2*w_1^2 + w_0^2*w_1^-3 + 10*w_0*w_1^-2 + 16*w_1^-1 + 16*w_0^-1 + 10*w_0^-2*w_1 + w_0^-3*w_1^2 + w_0*w_1^-3 + 6*w_1^-2 + 10*w_0^-1*w_1^-1 + 6*w_0^-2 + w_0^-3*w_1 + w_0^-1*w_1^-2 + w_0^-2*w_1^-1)*q^3 + (2*w_0^3 + 6*w_0^2*w_1 + 6*w_0*w_1^2 + 2*w_1^3 + 6*w_0^3*w_1^-1 + 16*w_0^2 + 20*w_0*w_1 + 16*w_1^2 + 6*w_0^-1*w_1^3 + 6*w_0^3*w_1^-2 + 20*w_0^2*w_1^-1 + 30*w_0 + 30*w_1 + 20*w_0^-1*w_1^2 + 6*w_0^-2*w_1^3 + 2*w_0^3*w_1^-3 + 16*w_0^2*w_1^-2 + 30*w_0*w_1^-1 + 32 + 30*w_0^-1*w_1 + 16*w_0^-2*w_1^2 + 2*w_0^-3*w_1^3 + 6*w_0^2*w_1^-3 + 20*w_0*w_1^-2 + 30*w_1^-1 + 30*w_0^-1 + 20*w_0^-2*w_1 + 6*w_0^-3*w_1^2 + 6*w_0*w_1^-3 + 16*w_1^-2 + 20*w_0^-1*w_1^-1 + 16*w_0^-2 + 6*w_0^-3*w_1 + 2*w_1^-3 + 6*w_0^-1*w_1^-2 + 6*w_0^-2*w_1^-1 + 2*w_0^-3)*q^4 + O(q^5)]
        """
        N = self.level()
        h = Gamma0_constructor(N).index()
        bound = (Integer(k) / 12) * h
        prec = max(prec, ceil(bound))
        S = self.index_matrix()
        rk = Integer(S.nrows())
        w1 = WeilRep(self.index_matrix())
        w = w1 + II(N)
        try:
            if k - rk / 2 >= 5/2 + (k % 2):
                w._WeilRep__applied_funct = lambda x: _remove_N(x, w1, N)
                w._WeilRep__flag = w1, h
                w.modular_forms_dimension = lambda k: self.jacobi_forms_dimension(k + rk/2)
                w.cusp_forms_dimension = lambda k: self.cusp_forms_dimension(k + rk/2)
                X = w.modular_forms_basis(k - rk / 2, prec, **kwargs)
            else:
                raise NotImplementedError
        except NotImplementedError:
            w = w1 + II(N)

            def a(u, N):
                ui = u.inverse_mod(N)

                def b(x):
                    return vector([frac(u * x[0])] + list(x[1:-1]) + [frac(ui * x[-1])])
                return b

            G = [w.automorphism(a(u, N)) for u in srange(N) if gcd(u, N) == 1]
            phi = w.canonical_involution()
            if phi not in G:
                chi = [1]*len(G) + [(-1)**k] * len(G)
                G = G + [g * phi for g in G]
            else:
                if k % 2:
                    return []
                chi = [1]*len(G)
            G = WeilRepAutomorphismGroup(w, G, None)
            X = w.invariant_forms_basis(k - rk / 2, prec, G=G, chi=chi, **kwargs)
            X = WeilRepModularFormsBasis(X.weight(), [_remove_N(x, w1, N) for x in X], w1)
            X._WeilRepModularFormsBasis__bound = bound
            X.echelonize()
        return [_jacobi_form(x, w1, N) for x in X]

    def cusp_forms_basis(self, k, prec, **kwargs):
        r"""
        Compute a basis of Jacobi cusp forms of weight k and lattice index and level \Gamma_0(N).

        EXAMPLES::

            sage: from weilrep import *
            sage: J = JacobiForms(Gamma0(4), matrix([[2, -1], [-1, 2]]))
            sage: J = J.cusp_forms_basis(6, 5) #cusp forms of weight 6 and level \Gamma_0(4) of A_2-lattice index
        """
        N = self.level()
        h = Gamma0_constructor(N).index()
        bound = (Integer(k) / 12) * h
        prec = max(prec, ceil(bound))
        S = self.index_matrix()
        rk = Integer(S.nrows())
        w1 = WeilRep(S)
        w = w1 + II(N)
        w._WeilRep__applied_funct = lambda x: _remove_N(x, w1, N)
        w._WeilRep__flag = w1, h
        w.cusp_forms_dimension = lambda k: self.cusp_forms_dimension(k + rk/2)
        return [_jacobi_form(x, w1, N) for x in w.cusp_forms_basis(k - rk / 2, prec, **kwargs)]

    def eisenstein_series(self, k, prec):
        S = self.index_matrix()
        w1 = WeilRep(S)
        N = self.level()
        w = w1 + II(N)
        w._WeilRep__applied_funct = lambda x: _remove_N(x, w1, N)
        w._WeilRep__flag = w1, None
        rk = Integer(S.nrows())
        return _jacobi_form(w.eisenstein_series(k - rk/2, prec), w1, N)

    def weak_forms_basis(self, k, prec, **kwargs):
        r"""
        Compute a basis of the space of weak Jacobi forms of weight k and level `\Gamma_0(N)` up to precision prec.

        NOTE: This is inefficient because for any index L, the graded ring of weak Jacobi forms of all weights has a decomposition as a tensor product,
        `J_{*, L}^w( \Gamma_0(N) ) = J_{*, L}^w( SL_2(ZZ) ) \otimes M_*(\Gamma_0(N))`
        and `J_{*, L}^w( SL_2(ZZ) )` can be computed using the faster code from jacobi_forms_class.py

        EXAMPLES::

            sage: from weilrep import *
            sage: J = JacobiForms(Gamma0(2), 1)
            sage: J.weak_forms_basis(0, 5)
            [1 + (w^-2 - 8*w^-1 + 14 - 8*w + w^2)*q + (14*w^-2 - 64*w^-1 + 100 - 64*w + 14*w^2)*q^2 + (-8*w^-3 + 100*w^-2 - 344*w^-1 + 504 - 344*w + 100*w^2 - 8*w^3)*q^3 + (w^-4 - 64*w^-3 + 504*w^-2 - 1472*w^-1 + 2062 - 1472*w + 504*w^2 - 64*w^3 + w^4)*q^4 + O(q^5), (w^-1 + w) + (16*w^-1 - 32 + 16*w)*q + (w^-3 - 32*w^-2 + 127*w^-1 - 192 + 127*w - 32*w^2 + w^3)*q^2 + (16*w^-3 - 192*w^-2 + 688*w^-1 - 1024 + 688*w - 192*w^2 + 16*w^3)*q^3 + (127*w^-3 - 1024*w^-2 + 2945*w^-1 - 4096 + 2945*w - 1024*w^2 + 127*w^3)*q^4 + O(q^5)]
        """
        svn = self._svn()
        s = max(svn)
        for i, x in enumerate(svn):
            if x < 0:
                svn[i] = svn[-x]
        N = self.level()
        h = Gamma0_constructor(N).index()
        bound = (Integer(k) / 12 + 1) * h
        prec = max(prec, ceil(bound))
        S = self.index_matrix()
        rk = Integer(S.nrows())
        w1 = WeilRep(S)
        w = w1 + II(N)
        k1 = k - rk/2
        L = w.nearly_holomorphic_modular_forms_basis(k1, s, prec, **kwargs)
        if not L:
            return []
        v_list = w.coefficient_vector_exponents(prec, 1 - (Integer(k) % 2), starting_from=-s, include_vectors=True)
        n = len(v_list)
        V = span([x.coefficient_vector(starting_from=-s, ending_with=prec)[:n] for x in L])

        def e(i):
            return vector([0] * i + [1] + [0] * (n - 1 - i))

        dsdict = w1.ds_dict()
        Z = [e(j) for j, (v, i) in enumerate(v_list) if i + svn[dsdict[tuple(v[1:-1])]] >= 0]
        V = V.intersection(span(Z)).basis()
        X = WeilRepModularFormsBasis(k1, [w.recover_modular_form_from_coefficient_vector(k1, v, prec, starting_from=-s) for v in V], w)
        X = WeilRepModularFormsBasis(k1, [_remove_N(x, w1, N) for x in X], w1)
        X._WeilRepModularFormsBasis__bound = bound
        X.echelonize(starting_from=-s)
        X.reverse()
        return [_jacobi_form(x, w1, N) for x in X]

    # ## other ##

    def _svn(self):
        r"""
        Short vector norms within cosets of self underlying lattice
        """
        try:
            return self.__svn
        except AttributeError:
            svn = JacobiForms(self.index_matrix())._short_vector_norms_by_component()
            self.__svn = svn
            return svn


class JacobiFormsGamma1:

    def __init__(self, level, index_matrix, **kwargs):
        self.__level = level
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
        self.__rank = self.__index_matrix.nrows()

    def __repr__(self):
        return "Jacobi forms of level Gamma1(%s) and index %s" % (self.__level, self.index())

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
        return self.__index_matrix

    def level(self):
        return self.__level

    def rank(self):
        return self.__rank

    # ## dimension ##
    def cusp_forms_dimension(self, k):
        rk = self.rank()
        k0 = QQ(k - rk/2)
        N = self.level()
        X = [WeilRep(self.index_matrix()) + II(d) for d in divisors(N)]
        return sum(moebius(N / d) * X[i].cusp_forms_dimension(k0) for i, d in enumerate(divisors(N)))

    def dimension(self, k):
        rk = self.rank()
        k0 = QQ(k - rk/2)
        N = self.level()
        X = [WeilRep(self.index_matrix()) + II(d) for d in divisors(N)]
        return sum(moebius(N / d) * X[i].modular_forms_dimension(k0) for i, d in enumerate(divisors(N)))
    jacobi_forms_dimension = dimension

    # ## basis ##

    def basis(self, k, prec):
        N = self.level()
        bound = (Integer(k) / 12) * Gamma1_constructor(N).index()
        prec = max(prec, ceil(bound))
        dim = self.cusp_forms_dimension(k)
        S = self.index_matrix()
        rk = Integer(S.nrows())
        w1 = WeilRep(self.index_matrix())
        w = w1 + II(N)
        A = w.modular_forms_basis(k - rk / 2, prec)
        if k % 2 == 0:
            X = WeilRepModularFormsBasis(A.weight(), [_remove_N(x, w1, N) for x in A], w1)
            X._WeilRepModularFormsBasis__bound = bound
            X.echelonize()
        else:
            X = []
        b = 1
        while len(X) < dim:
            Y = WeilRepModularFormsBasis(A.weight(), [_remove_N(x, w1, N, b=b) for x in A], w1)
            X = X + Y
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X._WeilRepModularFormsBasis__bound = bound
            X.echelonize()
            b += 1
            if b > N:
                raise RuntimeError  # probably won't happen
        return [_jacobi_form(x, w1, N, _flag=1) for x in X if x]

    def cusp_forms_basis(self, k, prec):
        r"""
        Compute a basis of Jacobi cusp forms of weight k and level \Gamma_1(N) up to precision prec.

        EXAMPLES::

            sage: from weilrep import *
            sage: J = JacobiForms(Gamma1(3), matrix([[2, -1], [-1, 2]]))
            sage: J.cusp_forms_basis(6, 5)
            [(w_1 + w_0*w_1^-1 + w_0^-1)*q + (w_0^2 - 7*w_1 - 7*w_0*w_1^-1 + w_0^-2*w_1^2 - 7*w_0^-1 + w_1^-2)*q^2 + (w_0*w_1^2 - 7*w_0^2 + w_0^-1*w_1^3 + w_0^3*w_1^-2 + 14*w_1 + 14*w_0*w_1^-1 - 7*w_0^-2*w_1^2 + w_0^2*w_1^-3 + 14*w_0^-1 - 7*w_1^-2 + w_0^-3*w_1 + w_0^-2*w_1^-1)*q^3 + (-7*w_0*w_1^2 + 14*w_0^2 - 7*w_0^-1*w_1^3 - 7*w_0^3*w_1^-2 + 4*w_1 + 4*w_0*w_1^-1 + 14*w_0^-2*w_1^2 - 7*w_0^2*w_1^-3 + 4*w_0^-1 + 14*w_1^-2 - 7*w_0^-3*w_1 - 7*w_0^-2*w_1^-1)*q^4 + O(q^5), (w_0 + w_0^-1*w_1 + w_1^-1)*q + (w_1^2 - 7*w_0 + w_0^2*w_1^-2 - 7*w_0^-1*w_1 - 7*w_1^-1 + w_0^-2)*q^2 + (w_0^2*w_1 + w_0^3*w_1^-1 - 7*w_1^2 + 14*w_0 + w_0^-2*w_1^3 - 7*w_0^2*w_1^-2 + 14*w_0^-1*w_1 + 14*w_1^-1 + w_0^-3*w_1^2 + w_0*w_1^-3 - 7*w_0^-2 + w_0^-1*w_1^-2)*q^3 + (-7*w_0^2*w_1 - 7*w_0^3*w_1^-1 + 14*w_1^2 + 4*w_0 - 7*w_0^-2*w_1^3 + 14*w_0^2*w_1^-2 + 4*w_0^-1*w_1 + 4*w_1^-1 - 7*w_0^-3*w_1^2 - 7*w_0*w_1^-3 + 14*w_0^-2 - 7*w_0^-1*w_1^-2)*q^4 + O(q^5)]
        """
        N = self.level()
        bound = (Integer(k) / 12) * Gamma1_constructor(N).index()
        prec = max(prec, ceil(bound))
        dim = self.cusp_forms_dimension(k)
        S = self.index_matrix()
        rk = Integer(S.nrows())
        w1 = WeilRep(S)
        w = w1 + II(N)
        A = w.cusp_forms_basis(k - rk / 2, prec)
        if k % 2 == 0:
            X = WeilRepModularFormsBasis(A.weight(), [_remove_N(x, w1, N) for x in A], w1)
            X._WeilRepModularFormsBasis__bound = bound
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X.echelonize()
        else:
            X = []
        b = 1
        while len(X) < dim:
            Y = WeilRepModularFormsBasis(A.weight(), [_remove_N(x, w1, N, b=b) for x in A], w1)
            X = X + Y
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X._WeilRepModularFormsBasis__bound = bound
            X.echelonize()
            b += 1
            if b > N:
                raise RuntimeError  # should not happen
        return [_jacobi_form(x, w1, N, _flag=1) for x in X if x]

    def weak_forms_basis(*args, **kwargs):
        r"""
        TODO (maybe)
        """
        raise NotImplementedError


class JacobiFormsGamma:

    def __init__(self, level, index_matrix, **kwargs):
        self.__level = level
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
        self.__rank = self.__index_matrix.nrows()

    def __repr__(self):
        return "Jacobi forms of level Gamma(%s) and index %s" % (self.__level, self.index())

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
        return self.__index_matrix

    def level(self):
        return self.__level

    def rank(self):
        return self.__rank

    # ## dimension ##
    def cusp_forms_dimension(self, k):
        S = self.index_matrix()
        r = S.nrows()
        wt = QQ(k - r / 2)

        def a(u, N):
            ui = u.inverse_mod(N)

            def b(x):
                return vector([frac(u * x[0])] + list(x[1:-1]) + [frac(ui * x[-1])])
            return b

        def dim(N, h):
            if N == 1:
                w = WeilRep(S)
                return w.cusp_forms_dimension(wt)
            w = WeilRep(S) + II(N)
            G = [w.automorphism(a(u, N)) for u in srange(N) if u % h == 1]
            phi = w.canonical_involution()
            if phi not in G:
                chi = [1]*len(G) + [(-1)**k] * len(G)
                G = G + [g * phi for g in G]
            else:
                if k % 2:
                    return 0
                chi = [1]*len(G)
            G = WeilRepAutomorphismGroup(w, G, None)
            return w.invariant_cusp_forms_dimension(wt, G=G, chi=chi)
        N = self.level()
        Nsqr = N * N
        return sum(moebius(Nsqr / d) * dim(d, N) for d in divisors(Nsqr) if moebius(Nsqr/d))

    def jacobi_forms_dimension(self, k):
        S = self.index_matrix()
        r = S.nrows()
        wt = QQ(k - r / 2)

        def a(u, N):
            ui = u.inverse_mod(N)

            def b(x):
                return vector([frac(u * x[0])] + list(x[1:-1]) + [frac(ui * x[-1])])
            return b

        def dim(N, h):
            if N == 1:
                w = WeilRep(S)
                return w.cusp_forms_dimension(wt)
            w = WeilRep(S) + II(N)
            G = [w.automorphism(a(u, N)) for u in srange(N) if u % h == 1]
            phi = w.canonical_involution()
            if phi not in G:
                chi = [1]*len(G) + [(-1)**k] * len(G)
                G = G + [g * phi for g in G]
            else:
                if k % 2:
                    return 0
                chi = [1]*len(G)
            G = WeilRepAutomorphismGroup(w, G, None)
            return w.invariant_forms_dimension(wt, G=G, chi=chi)
        N = self.level()
        Nsqr = N * N
        return sum(moebius(Nsqr / d) * dim(d, N) for d in divisors(Nsqr) if moebius(Nsqr/d))
    dimension = jacobi_forms_dimension

    # basis

    def basis(self, k, prec):
        N = self.level()
        Nsqr = N*N
        prec = prec * N
        bound = (Integer(k) / 12) * Gamma1_constructor(Nsqr).index()
        prec = max(prec, ceil(bound))
        S = self.index_matrix()
        rk = Integer(S.nrows())
        wt = k - rk / 2
        if wt < 2:
            dim = Infinity
        else:
            dim = self.jacobi_forms_dimension(k)
        w1 = WeilRep(S)
        w = w1 + II(Nsqr)
        A = w.modular_forms_basis(wt, prec)
        if k % 2 == 0:
            X = WeilRepModularFormsBasis(wt, [_remove_N(x, w1, Nsqr) for x in A], w1)
            X._WeilRepModularFormsBasis__bound = bound
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X.echelonize()
        else:
            X = []
        b = N
        while len(X) < dim:
            Y = WeilRepModularFormsBasis(wt, [_remove_N(x, w1, Nsqr, b=b) for x in A], w1)
            X = X + Y
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X._WeilRepModularFormsBasis__bound = bound
            X.echelonize()
            b += N
            if b + b > Nsqr:
                if dim == Infinity:
                    dim = -1
                else:
                    raise RuntimeError  # should not happen
        J = [_jacobi_form(x, w1, Nsqr, _flag=1) for x in X if x]
        for x in J:
            x._JacobiFormWithLevel__level = N
            x._JacobiFormWithLevel__wscale = N
            x._JacobiFormWithLevel__qscale = N
        return J

    def cusp_forms_basis(self, k, prec):
        r"""
        Compute a basis of Jacobi cusp forms of weight k and level \Gamma(N) up to precision prec.

        NOTE: this is slow

        EXAMPLES::

            sage: from weilrep import *
            sage: J = JacobiForms(Gamma1(3), matrix([[2, -1], [-1, 2]]))
            sage: J.cusp_forms_basis(6, 5)
            [(w_1 + w_0*w_1^-1 + w_0^-1)*q + (w_0^2 - 7*w_1 - 7*w_0*w_1^-1 + w_0^-2*w_1^2 - 7*w_0^-1 + w_1^-2)*q^2 + (w_0*w_1^2 - 7*w_0^2 + w_0^-1*w_1^3 + w_0^3*w_1^-2 + 14*w_1 + 14*w_0*w_1^-1 - 7*w_0^-2*w_1^2 + w_0^2*w_1^-3 + 14*w_0^-1 - 7*w_1^-2 + w_0^-3*w_1 + w_0^-2*w_1^-1)*q^3 + (-7*w_0*w_1^2 + 14*w_0^2 - 7*w_0^-1*w_1^3 - 7*w_0^3*w_1^-2 + 4*w_1 + 4*w_0*w_1^-1 + 14*w_0^-2*w_1^2 - 7*w_0^2*w_1^-3 + 4*w_0^-1 + 14*w_1^-2 - 7*w_0^-3*w_1 - 7*w_0^-2*w_1^-1)*q^4 + O(q^5), (w_0 + w_0^-1*w_1 + w_1^-1)*q + (w_1^2 - 7*w_0 + w_0^2*w_1^-2 - 7*w_0^-1*w_1 - 7*w_1^-1 + w_0^-2)*q^2 + (w_0^2*w_1 + w_0^3*w_1^-1 - 7*w_1^2 + 14*w_0 + w_0^-2*w_1^3 - 7*w_0^2*w_1^-2 + 14*w_0^-1*w_1 + 14*w_1^-1 + w_0^-3*w_1^2 + w_0*w_1^-3 - 7*w_0^-2 + w_0^-1*w_1^-2)*q^3 + (-7*w_0^2*w_1 - 7*w_0^3*w_1^-1 + 14*w_1^2 + 4*w_0 - 7*w_0^-2*w_1^3 + 14*w_0^2*w_1^-2 + 4*w_0^-1*w_1 + 4*w_1^-1 - 7*w_0^-3*w_1^2 - 7*w_0*w_1^-3 + 14*w_0^-2 - 7*w_0^-1*w_1^-2)*q^4 + O(q^5)]
        """
        N = self.level()
        Nsqr = N*N
        bound = (Integer(k) / 12) * Gamma1_constructor(Nsqr).index()
        prec = max(prec, ceil(bound))
        S = self.index_matrix()
        rk = Integer(S.nrows())
        wt = k - rk / 2
        if wt + wt < 5:
            dim = Infinity
        else:
            dim = self.cusp_forms_dimension(k)
        w1 = WeilRep(S)
        w = w1 + II(Nsqr)
        A = w.cusp_forms_basis(wt, prec)
        if k % 2 == 0:
            X = WeilRepModularFormsBasis(wt, [_remove_N(x, w1, Nsqr) for x in A], w1)
            X._WeilRepModularFormsBasis__bound = bound
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X.echelonize()
        else:
            X = []
        b = N
        while len(X) < dim:
            Y = WeilRepModularFormsBasis(wt, [_remove_N(x, w1, Nsqr, b=b) for x in A], w1)
            X = X + Y
            for x in X:
                x._use_all_coeffs_in_coeffvector = True
            X._WeilRepModularFormsBasis__bound = bound
            X.echelonize()
            b += N
            if b > Nsqr:
                if dim == Infinity:
                    dim = -1
                else:
                    raise RuntimeError  # should not happen
        J = [_jacobi_form(x, w1, Nsqr, _flag=1) for x in X if x]
        for x in J:
            x._JacobiFormWithLevel__level = N
            x._JacobiFormWithLevel__wscale = N
            x._JacobiFormWithLevel__qscale = N
        return J

    def weak_forms_basis(*args, **kwargs):
        r"""
        TODO (maybe)
        """
        raise NotImplementedError


def _remove_N(f, w1, N, b=0, c=0):
    ds = w1.ds()
    w = f.weilrep()
    dsdict = w.ds_dict()
    nl = w1.norm_list()
    X = f.fourier_expansion()
    Y = [None] * len(ds)
    bN = frac(b / N)
    cN = frac(c / N)
    for i, x in enumerate(ds):
        j = dsdict[tuple([bN]+list(x)+[cN])]
        _, _, h = X[j]
        Y[i] = (x, nl[i], h)
    return WeilRepModularForm(f.weight(), w1.gram_matrix(), Y, weilrep=w1)


def _jacobi_form(f, w1, N, _flag=0, q_scale=1):
    rk = w1.gram_matrix().nrows()
    if _flag:
        j = f.jacobi_form(eps=0, _flag=_flag)
    else:
        eps = (-1) ** Integer(f.weight() + rk/2)
        j = f.jacobi_form(eps=eps)
    return JacobiFormWithLevel(j.weight(), N, j.index_matrix(), j.qexp(), q_scale=q_scale)


class JacobiFormWithLevel:

    def __init__(self, weight, level, index_matrix, qexp, **kwargs):
        try:
            self.__weight = Integer(weight)
        except TypeError:
            self.__weight = QQ(weight)
        self.__index_matrix = index_matrix
        self.__qexp = qexp
        self.__level = level
        self.__wscale = kwargs.pop('w_scale', 1)
        self.__qscale = kwargs.pop('q_scale', 1)

    def __repr__(self):
        try:
            return self.__string
        except AttributeError:
            pass
        s = JacobiForm.__repr__(self)
        q = self.q_scale()
        if q != 1:
            def m(obj):
                obj_s = obj.string[slice(*obj.span())]
                x = obj_s[0]
                if x == '^':
                    u = Integer(obj_s[1:]) / q
                    if u.is_integer():
                        if u == 1:
                            return ''
                        return '^%d' % u
                    return '^(%s)' % u
                return obj_s + '^(1/%s)' % q
            s = sub(r'((?<=q)\^-?\d+)|q+(?!\^)', m, s)
        self.__string = s
        return self.__string

    def base_ring(self):
        r"""
        Laurent polynomial ring representing self's elliptic variables.

        EXAMPLES::

            sage: from weilrep import *
            sage: jacobi_eisenstein_series(4, 1, 5).base_ring()
            Univariate Laurent Polynomial Ring in w_0 over Rational Field
        """
        return self.fourier_expansion().base_ring()

    def _base_ring_is_laurent_polynomial_ring(self):
        r"""
        Is self's base ring actually a Laurent polynomial ring?

        This should return False if it is a FractionField.
        """
        try:
            return self.__brilpr
        except AttributeError:
            r = self.base_ring()
            self.__brilpr = isinstance(r, (LaurentPolynomialRing_generic,
                                           NumberField,
                                           ComplexField_class,
                                           RealField_class))
            # are we missing anything?
            return self.__brilpr

    def __bool__(self):
        return self.fourier_expansion().__bool__()

    def index(self):
        r"""
        Return self's index.

        If the index is a rank one matrix then return self's index as a scalar instead.
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
        """
        return self.__index_matrix

    def jacobiforms(self):
        from .jacobi_forms_class import JacobiForms
        return JacobiForms(self.index_matrix())

    def level(self):
        return self.__level

    def nvars(self):
        return self.index_matrix().nrows()

    def precision(self):
        r"""
        Return self's precision (with respect to the variable 'q').
        """
        try:
            return self.__precision
        except AttributeError:
            self.__precision = self.fourier_expansion().prec()
            return self.__precision
    prec = precision

    def qexp(self):
        return self.__qexp
    fourier_expansion = qexp

    def q_scale(self):
        return self.__qscale
    qscale = q_scale

    def _qshift(self):
        return 0

    def _rescale(self, a):
        f = self.qexp()
        e = self.nvars()
        rb_w = f.base_ring()
        d = {rb_w('w_%d' % j): rb_w('w_%d' % j)**a for j in range(e)}
        f = f.map_coefficients(lambda x: x.subs(d))
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix(), f, q_scale=self.q_scale(), w_scale=self.scale() * a)
    rescale = _rescale

    def _rescale_q(self, a):
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix(), self.qexp().V(a), q_scale = a * self.q_scale(), w_scale = self.scale())

    def scale(self):
        return self.__wscale

    def set_level(self, N):
        r"""
        Change self's level to N. (Not in place.)

        WARNING: we do not test whether self actually defines a Jacobi form of level N. Be careful!
        """
        q_scale = self.q_scale()
        if q_scale % N:
            raise ValueError('This is not a Jacobi form of level %s.' % N)
        if N == 1:
            return JacobiForm(self.weight(), self.index_matrix(), self.qexp(), scale=self.scale())
        return JacobiFormWithLevel(self, self.weight(), N, self.index_matrix(), self.qexp(), w_scale=self.scale(), q_scale=self.q_scale())

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
        except AttributeError:
            self.__valuation = self.fourier_expansion().valuation()
            return self.__valuation

    def weight(self):
        return self.__weight

    def weilrep(self):
        return WeilRep(self.index_matrix())

    # coefficients

    def q_coefficients(self):
        r"""
        Return self's Fourier coefficients with respect to the variable 'q'.

        OUTPUT: a list of Laurent polynomials
        """
        return list(self.fourier_expansion())

    # arithmetic

    def __add__(self, other):
        r"""
        Addition of Jacobi forms. Undefined unless both have the same weight and index.
        """
        if not other:
            return self
        if not isinstance(other, (JacobiForm, JacobiFormWithLevel)):
            raise TypeError('Cannot add these objects')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        if not self.index_matrix() == other.index_matrix():
            raise ValueError('Incompatible indices')
        sf = self.qexp()
        of = other.qexp()
        level = lcm(self.level(), other.level())
        scale = 1
        h = other._qshift()
        if h:
            try:
                d = h.denom()
            except AttributeError:
                d = 1
        else:
            d = 1
        q_scale = lcm([self.q_scale(), other.q_scale(), d])
        q1 = q_scale / self.q_scale()
        q2 = q_scale / other.q_scale()
        if self.scale() == 2 or other.scale() == 2:
            r = self.base_ring()
            q, = self.qexp().parent().gens()
            scale = 2
            if self.scale() == 1:
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            if other.scale() == 1:
                of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
        if q1 != 1:
            sf = sf.V(q1)
        if q2 != 1:
            of = of.V(q2)
        if h:
            of *= q ** Integer(q_scale * d)
        return JacobiFormWithLevel(self.weight(), level, self.index_matrix(), sf + of, w_scale=scale, q_scale=q_scale)

    __radd__ = __add__

    def __sub__(self, other):
        r"""
        Subtraction of Jacobi forms. Undefined unless both have the same weight and index.
        """
        if not other:
            return self
        if not isinstance(other, (JacobiForm, JacobiFormWithLevel)):
            raise TypeError('Cannot subtract these objects')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        if not self.index_matrix() == other.index_matrix():
            raise ValueError('Incompatible indices')
        sf = self.qexp()
        of = other.qexp()
        level = lcm(self.level(), other.level())
        scale = 1
        h = other._qshift()
        if h:
            try:
                d = h.denom()
            except AttributeError:
                d = 1
        else:
            d = 1
        q_scale = lcm([self.q_scale(), other.q_scale(), d])
        q1 = q_scale / self.q_scale()
        q2 = q_scale / other.q_scale()
        if self.scale() == 2 or other.scale() == 2:
            r = self.base_ring()
            q, = self.qexp().parent().gens()
            scale = 2
            if self.scale() == 1:
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            if other.scale() == 1:
                of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
        if q1 != 1:
            sf = sf.V(q1)
        if q2 != 1:
            of = of.V(q2)
        if h:
            of *= q ** Integer(q_scale * d)
        return JacobiFormWithLevel(self.weight(), level, self.index_matrix(), sf - of, w_scale=scale, q_scale=q_scale)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __invert__(self):
        f = self.qexp()
        try:
            f_inv = ~f
        except ValueError:
            R = f.parent()
            R_frac = R.fraction_field()
            f_inv = ~R_frac(f)
        return JacobiFormWithLevel(-self.weight(), self.level(), -self.index_matrix(), f_inv, w_scale=self.scale(), q_scale=self.q_scale())

    def __mul__(self, other):
        if isinstance(other, (JacobiForm, JacobiFormWithLevel)):
            scale = 1
            h = other._qshift()
            try:
                d = h.denom()
            except AttributeError:
                d = 1
            try:
                oq = other.q_scale()
            except AttributeError:
                oq = 1
            q_scale = lcm([self.q_scale(), oq, d])
            q1 = q_scale / self.q_scale()
            q2 = q_scale / oq
            sf = self.qexp()
            of = other.qexp()
            if self.scale() == 2 or other.scale() == 2:
                r = self.base_ring()
                q, = self.qexp().parent().gens()
                scale = 2
                if self.scale() == 1:
                    sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
                if other.scale() == 1:
                    of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r.gens()}))
            if q1 != 1:
                sf = sf.V(q1)
            if q2 != 1:
                of = of.V(q2)
            if h:
                of *= q ** Integer(h * q2)
            S1 = self.index_matrix()
            S2 = other.index_matrix()
            n1 = S1.nrows()
            n2 = S2.nrows()
            level = lcm(self.level(), other.level())
            if not n1:
                return JacobiFormWithLevel(self.weight() + other.weight(), level, other.index_matrix(), sf * of, w_scale=scale, q_scale=q_scale)
            elif not n2:
                return JacobiFormWithLevel(self.weight() + other.weight(), level, self.index_matrix(), sf * of, w_scale=scale, q_scale=q_scale)
            if n1 != n2:
                raise ValueError('Incompatible indices')
            return JacobiFormWithLevel(self.weight() + other.weight(), level, self.index_matrix() + other.index_matrix(), sf * of, w_scale=scale, q_scale=q_scale)
        elif isinstance(other, ModularFormElement):
            sf, of = self.qexp(), other.qexp()
            scale = self.scale()
            q_scale = self.q_scale()
            if q_scale != 1:
                q, = of.parent().gens()
                of = of.V(q_scale)
            f = sf * of.change_ring(sf.base_ring())
            level = lcm(self.level(), other.level())
            return JacobiFormWithLevel(self.weight() + other.weight(), level, self.index_matrix(), f, w_scale=scale, q_scale=q_scale)
        elif isinstance(other, WeilRepModularForm):
            if other.weilrep().gram_matrix().nrows() == 0:
                scale = self.scale()
                sf, of = self.qexp(), other.fourier_expansion()[0][2]
                qshift = other.fourier_expansion()[0][1]
                try:
                    d = qshift.denom()
                except AttributeError:
                    d = 1
                q_scale = lcm(self.q_scale(), d)
                q1 = q_scale / self.q_scale()
                q2 = q_scale / d
                if q_scale != 1:
                    q, = of.parent().gens()
                    of = of.V(q2)
                    f = sf.V(q1) * of.change_ring(sf.base_ring()) * q**Integer(q2 * qshift)
                else:
                    f = sf * of
                return JacobiFormWithLevel(self.weight() + other.weight(), self.level(), self.index_matrix(), f, w_scale=scale, q_scale=q_scale)
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix(), self.fourier_expansion() * other, w_scale=self.scale(), q_scale=self.q_scale())
    __rmul__ = __mul__

    def __neg__(self):
        r"""
        Return the negative of self.
        """
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix(), -self.fourier_expansion(), w_scale=self.scale(), q_scale=self.q_scale())

    def __truediv__(self, other):
        r"""
        Division
        """
        if isinstance(other, WeilRepModularForm):
            if other.weilrep().gram_matrix().nrows() == 0:
                sf, of = self.qexp(), other.fourier_expansion()[0][2]
                scale = self.scale()
                qshift = other.fourier_expansion()[0][1]
                try:
                    d = qshift.denom()
                except AttributeError:
                    d = 1
                q_scale = lcm(self.q_scale(), d)
                q1 = q_scale / self.q_scale()
                q2 = q_scale / d
                if q_scale != 1:
                    q, = of.parent().gens()
                    if qshift > 0:
                        of = of.V(q_scale).change_ring(sf.base_ring()) * q**Integer(q_scale * qshift)
                    else:
                        of = of.V(q_scale).change_ring(sf.base_ring()) / q**Integer(-q_scale * qshift)
                    f = sf.V(q1) / of
                else:
                    f = sf / of.change_ring(sf.base_ring())
                r = f.base_ring()
                if r is not LaurentPolynomialRing:
                    f = f.change_ring(LaurentPolynomialRing(r.base_ring(), r.gens()))
                return JacobiFormWithLevel(self.weight() - other.weight(), self.level(), self.index_matrix(), f, w_scale=scale, q_scale=q_scale)
        elif isinstance(other, ModularFormElement):
            level = lcm(self.level(), other.level())
            of = other.qexp()
            scale = self.scale()
            q_scale = self.q_scale()
            if q_scale != 1:
                q, = of.parent().gens()
                of = of.V(q_scale)
            f = self.fourier_expansion() / of
            return JacobiFormWithLevel(self.weight() - other.weight(), level, self.index_matrix(), f, w_scale=scale, q_scale=q_scale)
        elif isinstance(other, (JacobiForm, JacobiFormWithLevel)):
            try:
                d = other._qshift().denom()
            except AttributeError:
                d = 1
            try:
                oq = other.q_scale()
            except AttributeError:
                oq = 1
            q_scale = lcm([self.q_scale(), oq, d])
            q1 = q_scale / self.q_scale()
            q2 = q_scale / oq
            sf = self.qexp()
            of = other.qexp()
            r = self.base_ring()
            w1 = self.scale()
            w2 = other.scale()
            w_scale = lcm(w1, w2)
            if w1 != w_scale:
                r1 = sf.base_ring()
                sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r1.gens()}))
            if w2 != w_scale:
                r1 = of.base_ring()
                of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r1.gens()}))
            if q1 != 1:
                sf = sf.V(q1)
            if q2 != 1:
                of = of.V(q2)
            h = other._qshift()
            if h:
                q, = of.parent().gens()
                of *= q ** Integer(h * d)
            try:
                f = sf / of
            except ValueError:
                K = r.fraction_field()
                sf, of = sf.change_ring(K), of.change_ring(K)
                z = next(z for z in of if z)
                of /= z
                sf /= z
                f = sf / of
            level = lcm(self.level(), other.level())
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
            S1 = self.index_matrix()
            S2 = other.index_matrix()
            n1 = S1.nrows()
            n2 = S2.nrows()
            if not n1:
                return JacobiFormWithLevel(self.weight() - other.weight(), level, -other.index_matrix(), f, w_scale=w_scale, q_scale=q_scale)
            elif not n2:
                return JacobiFormWithLevel(self.weight() - other.weight(), level, self.index_matrix(), f, w_scale=w_scale, q_scale=q_scale)
            elif n1 != n2:
                raise ValueError('Incompatible indices')
            return JacobiFormWithLevel(self.weight() - other.weight(), level, self.index_matrix() - other.index_matrix(), f, w_scale=w_scale, q_scale=q_scale)
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix(), self.fourier_expansion() / other, w_scale=self.scale(), q_scale=self.q_scale())
    __div__ = __truediv__

    def __pow__(self, other):
        r"""
        Compute the exterior product of Jacobi forms.

        EXAMPLES::

            sage: from weilrep import *
            sage: t = jacobi_theta_series(20, '00')
            sage: print(t*t ** jacobi_thetanull(20, '00').V(2)**2)
            1 + (2*w^-1 + 2*w)*q^(1/2) + (w^-2 + 6 + w^2)*q + (8*w^-1 + 8*w)*q^(3/2) + (6*w^-2 + 12 + 6*w^2)*q^2 + (2*w^-3 + 10*w^-1 + 10*w + 2*w^3)*q^(5/2) + (12*w^-2 + 8 + 12*w^2)*q^3 + (8*w^-3 + 8*w^-1 + 8*w + 8*w^3)*q^(7/2) + (w^-4 + 8*w^-2 + 6 + 8*w^2 + w^4)*q^4 + (10*w^-3 + 16*w^-1 + 16*w + 10*w^3)*q^(9/2) + (6*w^-4 + 6*w^-2 + 24 + 6*w^2 + 6*w^4)*q^5 + (8*w^-3 + 16*w^-1 + 16*w + 8*w^3)*q^(11/2) + (12*w^-4 + 24*w^-2 + 24 + 24*w^2 + 12*w^4)*q^6 + (2*w^-5 + 16*w^-3 + 10*w^-1 + 10*w + 16*w^3 + 2*w^5)*q^(13/2) + (8*w^-4 + 24*w^-2 + 24*w^2 + 8*w^4)*q^7 + (8*w^-5 + 16*w^-3 + 24*w^-1 + 24*w + 16*w^3 + 8*w^5)*q^(15/2) + (6*w^-4 + 12 + 6*w^4)*q^8 + (10*w^-5 + 10*w^-3 + 16*w^-1 + 16*w + 10*w^3 + 10*w^5)*q^(17/2) + (w^-6 + 24*w^-4 + 12*w^-2 + 30 + 12*w^2 + 24*w^4 + w^6)*q^9 + (8*w^-5 + 24*w^-3 + 8*w^-1 + 8*w + 24*w^3 + 8*w^5)*q^(19/2) + (6*w^-6 + 24*w^-4 + 30*w^-2 + 24 + 30*w^2 + 24*w^4 + 6*w^6)*q^10 + (16*w^-5 + 16*w^-3 + 32*w^-1 + 32*w + 16*w^3 + 16*w^5)*q^(21/2) + (12*w^-6 + 24*w^-2 + 24 + 24*w^2 + 12*w^6)*q^11 + (16*w^-5 + 8*w^-3 + 24*w^-1 + 24*w + 8*w^3 + 16*w^5)*q^(23/2) + (8*w^-6 + 12*w^-4 + 24*w^-2 + 8 + 24*w^2 + 12*w^4 + 8*w^6)*q^12 + (2*w^-7 + 10*w^-5 + 32*w^-3 + 18*w^-1 + 18*w + 32*w^3 + 10*w^5 + 2*w^7)*q^(25/2) + (6*w^-6 + 30*w^-4 + 8*w^-2 + 24 + 8*w^2 + 30*w^4 + 6*w^6)*q^13 + (8*w^-7 + 24*w^-5 + 24*w^-3 + 24*w^-1 + 24*w + 24*w^3 + 24*w^5 + 8*w^7)*q^(27/2) + (24*w^-6 + 24*w^-4 + 24*w^-2 + 48 + 24*w^2 + 24*w^4 + 24*w^6)*q^14 + (10*w^-7 + 16*w^-5 + 18*w^-3 + 16*w^-1 + 16*w + 18*w^3 + 16*w^5 + 10*w^7)*q^(29/2) + (24*w^-6 + 24*w^-4 + 48*w^-2 + 48*w^2 + 24*w^4 + 24*w^6)*q^15 + (8*w^-7 + 8*w^-5 + 24*w^-3 + 24*w^-1 + 24*w + 24*w^3 + 8*w^5 + 8*w^7)*q^(31/2) + (w^-8 + 8*w^-4 + 6 + 8*w^4 + w^8)*q^16 + (16*w^-7 + 32*w^-5 + 16*w^-3 + 32*w^-1 + 32*w + 16*w^3 + 32*w^5 + 16*w^7)*q^(33/2) + (6*w^-8 + 12*w^-6 + 24*w^-4 + 6*w^-2 + 48 + 6*w^2 + 24*w^4 + 12*w^6 + 6*w^8)*q^17 + (16*w^-7 + 24*w^-5 + 24*w^-3 + 32*w^-1 + 32*w + 24*w^3 + 24*w^5 + 16*w^7)*q^(35/2) + (12*w^-8 + 30*w^-6 + 48*w^-4 + 48*w^-2 + 36 + 48*w^2 + 48*w^4 + 30*w^6 + 12*w^8)*q^18 + (10*w^-7 + 18*w^-5 + 32*w^-3 + 16*w^-1 + 16*w + 32*w^3 + 18*w^5 + 10*w^7)*q^(37/2) + (8*w^-8 + 24*w^-6 + 36*w^-2 + 24 + 36*w^2 + 24*w^6 + 8*w^8)*q^19 + (24*w^-7 + 24*w^-5 + 32*w^-3 + 32*w^-1 + 32*w + 32*w^3 + 24*w^5 + 24*w^7)*q^(39/2) + O(q^20)
        """
        if other in ZZ:
            return JacobiFormWithLevel(self.weight() * other, self.level(), self.index_matrix() * other, self.fourier_expansion()**other, w_scale=self.scale(), q_scale=self.q_scale())
        elif not isinstance(other, JacobiForm) and not isinstance(other, JacobiFormWithLevel):
            raise ValueError('Cannot multiply these objects')
        elif self.nvars() == 0:
            return other.__pow__(self)
        S1 = self.index_matrix()
        S2 = other.index_matrix()
        bigS = block_diagonal_matrix([S1, S2])
        K = self.base_ring().base_ring()
        rb = LaurentPolynomialRing(K, list('w_%d' % i for i in range(bigS.nrows())))
        r, q = PowerSeriesRing(rb, 'q').objgen()
        g = rb.gens()
        e1 = S1.nrows()
        e2 = S2.nrows()
        sf, of = self.qexp(), other.qexp()
        w1 = self.scale()
        w2 = other.scale()
        w_scale = lcm(w1, w2)
        if w1 != w_scale:
            r1 = sf.base_ring()
            sf = sf.map_coefficients(lambda x: x.subs({y: y*y for y in r1.gens()}))
        if w2 != w_scale:
            r1 = of.base_ring()
            of = of.map_coefficients(lambda x: x.subs({y: y*y for y in r1.gens()}))
        level = lcm(self.level(), other.level())
        q_scale = lcm(self.q_scale(), other.q_scale())
        q1 = q_scale / self.q_scale()
        q2 = q_scale / other.q_scale()
        if q1 != 1:
            sf = sf.V(q1)
        if q2 != 1:
            of = of.V(q2)
        jf = [rb(of[i]).subs({g[j]: g[j+e1] for j in range(e2)}) for i in range(of.valuation(), of.prec())]
        level = lcm(self.level(), other.level())
        return JacobiFormWithLevel(self.weight() + other.weight(), level, bigS, (r(sf) * r(jf)).add_bigoh(other.precision()), w_scale=w_scale, q_scale=q_scale)

    def __eq__(self, other):
        sf, of = self.qexp(), other.qexp()
        return sf == of

    # ## other ##

    def development_coefficient(self, lattice_basis, v=[]):
        r"""
        Compute the development coefficients of a Jacobi form on a congruence subgroup.
        """
        if isinstance(lattice_basis, Integer) and self.nvars() == 1:
            v = [vector([1])]*lattice_basis
            lattice_basis = []
        from .weilrep_misc import multilinear_gegenbauer_polynomial
        k = self.weight()
        K = self.base_ring().base_ring()
        S = self.index_matrix()
        S_inv = S.inverse()
        z = matrix(ZZ, lattice_basis)
        z_tr = z.transpose()
        ell = z.nrows()
        if ell:
            Sz = S * z_tr
            if matrix(v) * Sz:
                raise ValueError('The development coefficient must be evaluated along vectors orthogonal to the sublattice.')
            Rb = LaurentPolynomialRing(K, list('w_%d' % i for i in range(ell)))
        else:
            Rb = K
            Sz = matrix([])
        j = JacobiForms(z * Sz)
        N = len(v)
        R = PowerSeriesRing(Rb, 'q')
        qscale = self.q_scale()
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
            m = lambda _: 1

        def a(n):
            n = Integer(n) / qscale

            def b(x):
                if ell:
                    try:
                        return sum(m(vector(x)*z_tr) * P(*S_inv*vector(x) / wscale, n) * y for x, y in x.dict().items())
                    except AttributeError:
                        return K(x)
                    except TypeError:
                        return sum(m(vector([x])*z_tr) * P(*S_inv*vector([x]) / wscale, n) * y for x, y in x.dict().items())
                else:
                    try:
                        return sum(P(*S_inv*vector(x) / wscale, n) * y for x, y in x.dict().items())
                    except AttributeError:
                        return K(x)
                    except TypeError:
                        return sum(P(*S_inv*vector([x]) / wscale, n) * y for x, y in x.dict().items())
            return b
        f = R([a(n)(h) for n, h in enumerate(self.q_coefficients())]).add_bigoh(self.precision())
        return JacobiFormWithLevel(k + N, self.level(), j.index_matrix(), f, q_scale=qscale, w_scale=wscale)

    def pullback(self, A):
        r"""
        Apply a linear map to self's elliptic variables.
        """
        try:
            new_e = A.nrows()
        except AttributeError:
            A = matrix(A).transpose()
            return self.pullback(A)
        f = self.qexp()
        S = self.index_matrix()
        e = S.nrows()
        Rb = LaurentPolynomialRing(QQ, list('w_%d' % i for i in range(e)))
        Rb_new = LaurentPolynomialRing(QQ, list('w_%d' % i for i in range(new_e)))
        R, q = PowerSeriesRing(Rb_new, 'q').objgen()
        q_scale = self.q_scale()
        val = f.valuation()
        prec = f.prec()
        if new_e > 1:
            sub_R = {Rb('w_%d' % j): Rb_new.monomial(*x) for j, x in enumerate(A.columns())}
        else:
            w, = Rb_new.gens()
            sub_R = {Rb('w_%d' % j): w**A[0, j] for j in range(e)}
        jf_new = [f[i].subs(sub_R) for i in range(val, prec)]
        return JacobiFormWithLevel(self.weight(), self.level(), A * S * A.transpose(), (q**val * R(jf_new)).add_bigoh(f.prec()), w_scale=self.scale(), q_scale=q_scale)
    substitute = pullback

    def reduce_precision(self, new_prec):
        q_scale = self.q_scale()
        f = self.qexp().add_bigoh(ceil(q_scale * new_prec))
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix(), f, w_scale=self.scale(), q_scale=q_scale)

    def _theta_decomposition(self, *args, **kwargs):
        from .jacobi_forms_class import JacobiForm
        return JacobiForm.theta_decomposition(self, *args, **kwargs)

    def U(self, N):
        r"""
        If self is the Jacobi form f(\tau, z), then return the Jacobi form f(\tau,  N * z).
        """
        wscale = self.scale()
        wd = gcd(wscale, N)
        new_wscale = Integer(wscale // wd)
        f = self._rescale(Integer(N // wd)).qexp()
        return JacobiFormWithLevel(self.weight(), self.level(), self.index_matrix() * N * N, f, q_scale=self.q_scale(), w_scale=new_wscale)

    def V(self, N):
        r"""
        If self is the Jacobi form f(\tau, z), then return the Jacobi form f(N * \tau,  N * z).
        """
        qscale = self.q_scale()
        wscale = self.scale()
        qd = gcd(qscale, N)
        wd = gcd(wscale, N)
        new_qscale = Integer(qscale // qd)
        new_wscale = Integer(wscale // wd)
        f = self._rescale(Integer(N // wd)).qexp().V(Integer(N // qd))
        return JacobiFormWithLevel(self.weight(), self.level() * N, self.index_matrix() * N * N, f, q_scale=new_qscale, w_scale=new_wscale)

    # lifts
    # (dubious)

    def borcherds_lift(self, *args, **kwargs):
        return self.set_level(1).borcherds_lift(*args, **kwargs)

    def gritsenko_lift(self, *args, **kwargs):
        return self.set_level(1).gritsenko_lift(*args, **kwargs)


class JacobiFormWithLevelAndCharacter(JacobiFormWithLevel):
    r"""
    Jacobi forms which transform with a power of the eta multiplier under the action of some congruence group \Gamma(N).
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
            self.__class__, self.__character, self.__qshift = JacobiFormWithLevelAndCharacter, chi, qshift
        else:
            self.__class__ = JacobiFormWithLevel

    def __repr__(self):
        try:
            return self.__string
        except AttributeError:
            self.__string = JacobiFormWithCharacter.__repr__(self)
            return self.__string


def jacobi_theta_00(prec):
    r"""
    The Jacobi theta function with characteristic 00.
    """
    rb, w_0 = LaurentPolynomialRing(QQ, 'w_0').objgen()
    q, = PowerSeriesRing(rb, 'q', prec).gens()
    bound = isqrt(2 * prec) + 1
    f = 1 + sum(q**(n*n) * (w_0 ** n + w_0 ** (-n)) for n in range(1, bound))
    return JacobiFormWithLevel(1/2, 2, matrix([[1]]), f.add_bigoh(2 * prec), w_scale=1, q_scale=2)


def jacobi_theta_01(prec):
    r"""
    The Jacobi theta function with characteristic 01.
    """
    rb, w_0 = LaurentPolynomialRing(QQ, 'w_0').objgen()
    q, = PowerSeriesRing(rb, 'q', prec).gens()
    bound = isqrt(2 * prec) + 1
    f = 1 + sum((-q)**(n*n) * (w_0 ** n + w_0 ** (-n)) for n in range(1, bound))
    return JacobiFormWithLevel(1/2, 2, matrix([[1]]), f.add_bigoh(2 * prec), w_scale=1, q_scale=2)


def jacobi_theta_10(prec):
    r"""
    The Jacobi theta function with characteristic 10.
    """
    rb, w_0 = LaurentPolynomialRing(QQ, 'w_0').objgen()
    q, = PowerSeriesRing(rb, 'q', prec).gens()
    bound = isqrt(2 * prec) + 1
    f = sum(q**((2*n - 1)**2) * (w_0 ** (2*n - 1) + w_0 ** (1 - 2*n)) for n in range(1, bound))
    return JacobiFormWithLevel(1/2, 2, matrix([[1]]), f.add_bigoh(8 * prec), w_scale=2, q_scale=8)


def jacobi_thetanull_00(prec):
    r"""
    The Jacobi thetanull with characteristic 00.
    """
    q, = PowerSeriesRing(QQ, 'q', prec).gens()
    bound = isqrt(2 * prec) + 1
    f = 1 + sum(2 * q**(n*n) for n in range(1, bound))
    return JacobiFormWithLevel(1/2, 2, matrix([]), f.add_bigoh(2 * prec), w_scale=1, q_scale=2)


def jacobi_thetanull_01(prec):
    r"""
    The Jacobi thetanull with characteristic 01.
    """
    q, = PowerSeriesRing(QQ, 'q', prec).gens()
    bound = isqrt(2 * prec) + 1
    f = 1 + sum(2 * (-q)**(n*n) for n in range(1, bound))
    return JacobiFormWithLevel(1/2, 2, matrix([]), f.add_bigoh(2 * prec), w_scale=1, q_scale=2)


def jacobi_thetanull_10(prec):
    r"""
    The Jacobi thetanull with characteristic 10.
    """
    q, = PowerSeriesRing(QQ, 'q', prec).gens()
    bound = isqrt(2 * prec) + 1
    f = 2 * sum(q**((2*n - 1)**2) for n in range(1, bound))
    return JacobiFormWithLevel(1/2, 2, matrix([]), f.add_bigoh(8 * prec), w_scale=2, q_scale=8)


def jacobi_thetanull(prec, s):
    if s == '00':
        return jacobi_thetanull_00(prec)
    elif s == '01':
        return jacobi_thetanull_01(prec)
    elif s == '10':
        return jacobi_thetanull_10(prec)
    elif s == '11':
        return 0
    raise ValueError('Undefined theta characteristic: %s' % s)


def _jf_relations_lvl(X):
    Xref = X[0]
    if not all(x.weight() == Xref.weight() and x.index() == Xref.index() for x in X[1:]):
        raise ValueError('Incompatible Jacobi forms')
    Nq = lcm(x.q_scale() for x in X)
    Nw = lcm(x.scale() for x in X)
    X = [x._rescale_q(Nq / x.q_scale()).rescale(Nw / x.scale()) for x in X]
    X = [x._theta_decomposition() for x in X]
    val = min(x.valuation() for x in X)
    prec = min(x.precision() for x in X)
    M = matrix([x.coefficient_vector(starting_from=val, ending_with=prec) for x in X])
    return M.kernel()
