r"""

Basic methods for additive and multiplicative theta lifts

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

from copy import copy

import cypari2
pari = cypari2.Pari()
PariError = cypari2.PariError

import math

from re import sub

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli, divisors, GCD
from sage.arith.srange import srange
from sage.calculus.var import var
from sage.functions.log import exp, log
from sage.functions.other import ceil, floor, frac, sqrt
from sage.geometry.cone import Cone
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import matrix
from sage.matrix.special import block_diagonal_matrix, block_matrix
from sage.misc.functional import denominator, isqrt
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modules.free_module_element import vector
from sage.rings.big_oh import O
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR

from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis
from .jacobi_forms_class import JacobiForm, JacobiForms

class OrthogonalModularForms(object):
    r"""
    This class represents spaces of modular forms on the type IV domain attached to a lattice of the form L + II_{2,2}, where L is positive-definite.

    INPUT: an OrthogonalModularForms instance is constructed by calling ``OrthogonalModularForms(S)``, where:
    - ``S`` -- a Gram matrix for the lattice; or
    - ``S`` -- a quadratic form; or
    - ``S`` -- the WeilRep instance WeilRep(S)
    """

    def __init__(self, w):
        try:
            S = w.gram_matrix()
        except AttributeError:
            w = WeilRep(w)
            S = w.gram_matrix()
        if w.is_lorentzian() or w.is_lorentzian_plus_II():
            from .lorentz import OrthogonalModularFormsLorentzian
            self.__class__ = OrthogonalModularFormsLorentzian
            OrthogonalModularFormsLorentzian.__init__(self, w)
        elif not w.is_positive_definite():
            raise NotImplementedError
        self.__weilrep = w
        self.__gram_matrix = S

    def __repr__(self):
        return 'Orthogonal modular forms associated to the Gram matrix\n%s + 2U'%self.__gram_matrix

    def __add__(self, other):
        from .lorentz import RescaledHyperbolicPlane
        w = self.weilrep()
        if isinstance(other, RescaledHyperbolicPlane):
            return OrthogonalModularForms(w + other)
        raise NotImplementedError
    __radd__ = __add__

    def gram_matrix(self):
        return self.__gram_matrix

    def weilrep(self):
        return self.__weilrep

    ## constructors of modular forms
    def eisenstein_series(self, k, prec):
        r"""
        Compute the orthogonal Eisenstein series.

        This computes the additive lift of the Jacobi Eisenstein series, renormalized to have constant coefficient 1.

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- the precision of the output

        OUTPUT: OrthogonalModularForm

        EXAMPLES::

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[2]]))
            sage: m.eisenstein_series(4, 5)
            1 + 240*q + 240*s + 2160*q^2 + (240*r^-2 + 13440*r^-1 + 30240 + 13440*r + 240*r^2)*q*s + 2160*s^2 + 6720*q^3 + (30240*r^-2 + 138240*r^-1 + 181440 + 138240*r + 30240*r^2)*q^2*s + (30240*r^-2 + 138240*r^-1 + 181440 + 138240*r + 30240*r^2)*q*s^2 + 6720*s^3 + 17520*q^4 + (13440*r^-3 + 181440*r^-2 + 362880*r^-1 + 497280 + 362880*r + 181440*r^2 + 13440*r^3)*q^3*s + (2160*r^-4 + 138240*r^-3 + 604800*r^-2 + 967680*r^-1 + 1239840 + 967680*r + 604800*r^2 + 138240*r^3 + 2160*r^4)*q^2*s^2 + (13440*r^-3 + 181440*r^-2 + 362880*r^-1 + 497280 + 362880*r + 181440*r^2 + 13440*r^3)*q*s^3 + 17520*s^4 + O(q, s)^5
        """
        return orthogonal_eisenstein_series(k, self.gram_matrix(), prec, w = self.weilrep())

    def spezialschar(self, k, prec):
        r"""
        Compute a basis of the Maass Spezialschar of weight ``k`` up to precision ``prec``.

        This computes the theta lifts of a basis of cusp forms.

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- the precision of the output

        OUTPUT: list of OrthogonalModularForm's

        EXAMPLES::

            sage: from weilrep import *
            sage: ParamodularForms(N = 1).spezialschar(10, 5)
            [(r^-1 - 2 + r)*q*s + (-2*r^-2 - 16*r^-1 + 36 - 16*r - 2*r^2)*q^2*s + (-2*r^-2 - 16*r^-1 + 36 - 16*r - 2*r^2)*q*s^2 + (r^-3 + 36*r^-2 + 99*r^-1 - 272 + 99*r + 36*r^2 + r^3)*q^3*s + (-16*r^-3 + 240*r^-2 - 240*r^-1 + 32 - 240*r + 240*r^2 - 16*r^3)*q^2*s^2 + (r^-3 + 36*r^-2 + 99*r^-1 - 272 + 99*r + 36*r^2 + r^3)*q*s^3 + O(q, s)^5]
        """
        S = self.gram_matrix()
        nrows = S.nrows()
        w = self.weilrep()
        X = w.cusp_forms_basis(k - nrows/2, ceil(prec * prec / 4) + 1)
        return [x.theta_lift(prec) for x in X]
    maass_space = spezialschar

    def modular_form_from_fourier_jacobi_expansion(self, k, fj):
        r"""
        Recover an orthogonal modular form from its Fourier--Jacobi coefficients.

        WARNING: we do not check whether the Fourier--Jacobi coefficients actually correspond to a modular form!

        INPUT:
        - ``k`` -- the weight
        - ``fj`` -- a list of Jacobi forms

        OUTPUT: OrthogonalModularForm
        """
        if len(fj) < 2:
            raise ValueError('Too few coefficients')
        rb_w = fj[1].base_ring()
        S = self.gram_matrix()
        nrows = S.nrows()
        rb_r = LaurentPolynomialRing(QQ, list(var('r_%d' % i) for i in range(nrows)))
        rb_x, x = LaurentPolynomialRing(rb_r, 'x').objgen()
        rb_q, q = PowerSeriesRing(rb_r, 'q').objgen()
        t, = PowerSeriesRing(rb_x, 't').gens()
        change_ring = {w_j:rb_r('r_%d'%j) for j, w_j in enumerate(rb_w.gens())}
        f = sum([rb_q(fj[i].fourier_expansion().map_coefficients(lambda x: x.subs(change_ring)))(x*t) * (~x*t)**i for i in range(1, len(fj))]) + O(t ** len(fj))
        h = fj[0].fourier_expansion()
        if h:
            f += rb_q(h)(x * t)
        return OrthogonalModularForm(k, S, f, 1, vector([0] * (nrows + 2)))

    ## methods for borcherds products

    def _borcherds_product_polyhedron(self, pole_order, prec, verbose = False):
        r"""
        Construct a polyhedron representing a cone of Heegner divisors. For internal use in the methods borcherds_input_basis() and borcherds_input_Qbasis().

        INPUT:
        - ``pole_order`` -- pole order
        - ``prec`` -- precision

        OUTPUT: a tuple consisting of an integral matrix M, a Polyhedron p, and a WeilRepModularFormsBasis X

        EXAMPLES::

            sage: from weilrep import *
            sage: m = ParamodularForms(5)
            sage: m._borcherds_product_polyhedron(1/4, 5)[1]
            A 2-dimensional polyhedron in QQ^3 defined as the convex hull of 1 vertex and 2 rays
        """
        S = self.gram_matrix()
        wt = -Integer(S.nrows())/2
        w = self.weilrep()
        rds = w.rds()
        norm_dict = w.norm_dict()
        X = w.nearly_holomorphic_modular_forms_basis(wt, pole_order, prec, verbose = verbose)
        N = len([g for g in rds if not norm_dict[tuple(g)]])
        v_list = w.coefficient_vector_exponents(0, 1, starting_from = -pole_order, include_vectors = True)
        exp_list = [v[1] for v in v_list]
        v_list = [vector(v[0]) for v in v_list]
        positive = [None]*len(exp_list)
        zero = vector([0] * (len(exp_list) + 1))
        M = matrix([x.coefficient_vector(starting_from = -pole_order, ending_with = 0)[:-N] for x in X])
        vs = M.transpose().kernel().basis()
        for i, n in enumerate(exp_list):
            ieq = copy(zero)
            ieq[i+1] = 1
            for j, m in enumerate(exp_list[:i]):
                N = sqrt(m / n)
                if N in ZZ:
                    v1 = v_list[i]
                    v2 = v_list[j]
                    ieq[j + 1] = denominator(v1 * N - v2) == 1 or denominator(v1 * N + v2) == 1
            positive[i] = ieq
        p = Polyhedron(ieqs = positive, eqns = [vector([0] + list(v)) for v in vs])
        return M, p, X

    def borcherds_input_basis(self, pole_order, prec, verbose = False):
        r"""
        Compute a basis of input functions into the Borcherds lift with pole up to pole_order.

        This method computes a list of Borcherds lift inputs F_0, ..., F_d which is a basis in the following sense: it is minimal with the property that every modular form with pole order at most pole_order whose Borcherds lift is holomorphic can be expressed in the form (k_0 F_0 + ... + k_d F_d) where k_i are nonnegative integers.

        WARNING: this can take a long time, and the output list might be longer than you expect!!

        INPUT:
        - ``pole_order`` -- positive number (does not need to be an integer)
        - ``prec`` -- precision of the output

        OUTPUT: WeilRepModularFormsBasis

        EXAMPLES::

            sage: from weilrep import *
            sage: m = ParamodularForms(5)
            sage: m.borcherds_input_basis(1/4, 5)
            [(0), 8 + 98*q + 604*q^2 + 2822*q^3 + 10836*q^4 + O(q^5)]
            [(1/10), q^(-1/20) - 34*q^(19/20) - 253*q^(39/20) - 1381*q^(59/20) - 5879*q^(79/20) - 21615*q^(99/20) + O(q^(119/20))]
            [(1/5), q^(-1/5) + 6*q^(4/5) + 107*q^(9/5) + 698*q^(14/5) + 3436*q^(19/5) + 13644*q^(24/5) + O(q^(29/5))]
            [(3/10), -31*q^(11/20) - 257*q^(31/20) - 1343*q^(51/20) - 5635*q^(71/20) - 20283*q^(91/20) + O(q^(111/20))]
            [(2/5), 9*q^(1/5) + 94*q^(6/5) + 612*q^(11/5) + 2816*q^(16/5) + 10958*q^(21/5) + O(q^(26/5))]
            [(1/2), q^(-1/4) - 2*q^(3/4) - 61*q^(7/4) - 518*q^(11/4) - 2677*q^(15/4) - 11280*q^(19/4) + O(q^(23/4))]
            [(3/5), 9*q^(1/5) + 94*q^(6/5) + 612*q^(11/5) + 2816*q^(16/5) + 10958*q^(21/5) + O(q^(26/5))]
            [(7/10), -31*q^(11/20) - 257*q^(31/20) - 1343*q^(51/20) - 5635*q^(71/20) - 20283*q^(91/20) + O(q^(111/20))]
            [(4/5), q^(-1/5) + 6*q^(4/5) + 107*q^(9/5) + 698*q^(14/5) + 3436*q^(19/5) + 13644*q^(24/5) + O(q^(29/5))]
            [(9/10), q^(-1/20) - 34*q^(19/20) - 253*q^(39/20) - 1381*q^(59/20) - 5879*q^(79/20) - 21615*q^(99/20) + O(q^(119/20))]
            ------------------------------------------------------------
            [(0), 10 + 50*q + 260*q^2 + 1030*q^3 + 3500*q^4 + O(q^5)]
            [(1/10), 6*q^(-1/20) + 16*q^(19/20) + 82*q^(39/20) + 304*q^(59/20) + 1046*q^(79/20) + 3120*q^(99/20) + O(q^(119/20))]
            [(1/5), q^(-1/5) - 24*q^(4/5) - 143*q^(9/5) - 622*q^(14/5) - 2204*q^(19/5) - 6876*q^(24/5) + O(q^(29/5))]
            [(3/10), -16*q^(11/20) - 102*q^(31/20) - 448*q^(51/20) - 1650*q^(71/20) - 5248*q^(91/20) + O(q^(111/20))]
            [(2/5), -q^(1/5) + 14*q^(6/5) + 92*q^(11/5) + 386*q^(16/5) + 1318*q^(21/5) + O(q^(26/5))]
            [(1/2), 20*q^(3/4) + 160*q^(7/4) + 700*q^(11/4) + 2560*q^(15/4) + 8000*q^(19/4) + O(q^(23/4))]
            [(3/5), -q^(1/5) + 14*q^(6/5) + 92*q^(11/5) + 386*q^(16/5) + 1318*q^(21/5) + O(q^(26/5))]
            [(7/10), -16*q^(11/20) - 102*q^(31/20) - 448*q^(51/20) - 1650*q^(71/20) - 5248*q^(91/20) + O(q^(111/20))]
            [(4/5), q^(-1/5) - 24*q^(4/5) - 143*q^(9/5) - 622*q^(14/5) - 2204*q^(19/5) - 6876*q^(24/5) + O(q^(29/5))]
            [(9/10), 6*q^(-1/20) + 16*q^(19/20) + 82*q^(39/20) + 304*q^(59/20) + 1046*q^(79/20) + 3120*q^(99/20) + O(q^(119/20))]
            ------------------------------------------------------------
            [(0), 22 + 342*q + 2156*q^2 + 10258*q^3 + 39844*q^4 + O(q^5)]
            [(1/10), -2*q^(-1/20) - 152*q^(19/20) - 1094*q^(39/20) - 5828*q^(59/20) - 24562*q^(79/20) - 89580*q^(99/20) + O(q^(119/20))]
            [(1/5), 3*q^(-1/5) + 48*q^(4/5) + 571*q^(9/5) + 3414*q^(14/5) + 15948*q^(19/5) + 61452*q^(24/5) + O(q^(29/5))]
            [(3/10), -108*q^(11/20) - 926*q^(31/20) - 4924*q^(51/20) - 20890*q^(71/20) - 75884*q^(91/20) + O(q^(111/20))]
            [(2/5), 37*q^(1/5) + 362*q^(6/5) + 2356*q^(11/5) + 10878*q^(16/5) + 42514*q^(21/5) + O(q^(26/5))]
            [(1/2), 4*q^(-1/4) - 28*q^(3/4) - 404*q^(7/4) - 2772*q^(11/4) - 13268*q^(15/4) - 53120*q^(19/4) + O(q^(23/4))]
            [(3/5), 37*q^(1/5) + 362*q^(6/5) + 2356*q^(11/5) + 10878*q^(16/5) + 42514*q^(21/5) + O(q^(26/5))]
            [(7/10), -108*q^(11/20) - 926*q^(31/20) - 4924*q^(51/20) - 20890*q^(71/20) - 75884*q^(91/20) + O(q^(111/20))]
            [(4/5), 3*q^(-1/5) + 48*q^(4/5) + 571*q^(9/5) + 3414*q^(14/5) + 15948*q^(19/5) + 61452*q^(24/5) + O(q^(29/5))]
            [(9/10), -2*q^(-1/20) - 152*q^(19/20) - 1094*q^(39/20) - 5828*q^(59/20) - 24562*q^(79/20) - 89580*q^(99/20) + O(q^(119/20))]
            ------------------------------------------------------------
            [(0), 36 + 586*q + 3708*q^2 + 17694*q^3 + 68852*q^4 + O(q^5)]
            [(1/10), -5*q^(-1/20) - 270*q^(19/20) - 1935*q^(39/20) - 10275*q^(59/20) - 43245*q^(79/20) - 157545*q^(99/20) + O(q^(119/20))]
            [(1/5), 5*q^(-1/5) + 90*q^(4/5) + 1035*q^(9/5) + 6130*q^(14/5) + 28460*q^(19/5) + 109260*q^(24/5) + O(q^(29/5))]
            [(3/10), -185*q^(11/20) - 1595*q^(31/20) - 8505*q^(51/20) - 36145*q^(71/20) - 131485*q^(91/20) + O(q^(111/20))]
            [(2/5), 65*q^(1/5) + 630*q^(6/5) + 4100*q^(11/5) + 18940*q^(16/5) + 74070*q^(21/5) + O(q^(26/5))]
            [(1/2), 7*q^(-1/4) - 54*q^(3/4) - 747*q^(7/4) - 5026*q^(11/4) - 23859*q^(15/4) - 94960*q^(19/4) + O(q^(23/4))]
            [(3/5), 65*q^(1/5) + 630*q^(6/5) + 4100*q^(11/5) + 18940*q^(16/5) + 74070*q^(21/5) + O(q^(26/5))]
            [(7/10), -185*q^(11/20) - 1595*q^(31/20) - 8505*q^(51/20) - 36145*q^(71/20) - 131485*q^(91/20) + O(q^(111/20))]
            [(4/5), 5*q^(-1/5) + 90*q^(4/5) + 1035*q^(9/5) + 6130*q^(14/5) + 28460*q^(19/5) + 109260*q^(24/5) + O(q^(29/5))]
            [(9/10), -5*q^(-1/20) - 270*q^(19/20) - 1935*q^(39/20) - 10275*q^(59/20) - 43245*q^(79/20) - 157545*q^(99/20) + O(q^(119/20))]
        """
        S = self.gram_matrix()
        w = self.weilrep()
        wt = -S.nrows()/2
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose = verbose)
        if verbose:
            print('I will now try to find a Hilbert basis.')
        b = matrix(Cone(p).Hilbert_basis())
        try:
            u = M.solve_left(b)
        except ValueError:
            return WeilRepModularFormsBasis(wt, [], w)
        Y = [v * X for v in u.rows()]
        Y.sort(key = lambda x: x.fourier_expansion()[0][2][0])
        return WeilRepModularFormsBasis(wt, Y, w)

    def borcherds_input_Qbasis(self, pole_order, prec, verbose = False):
        r"""
        Compute a Q-basis of input functions into the Borcherds lift with pole order in infinity up to pole_order.

        This method computes a list of Borcherds lift inputs F_0, ..., F_d which is a Q-basis in the following sense: it is minimal with the property that every modular form with pole order at most pole_order whose Borcherds lift is holomorphic can be expressed in the form (k_0 F_0 + ... + k_d F_d) where k_i are nonnegative rational numbers.

        INPUT:
        - ``pole_order`` -- positive number (does not need to be an integer)
        - ``prec`` -- precision of the output

        OUTPUT: WeilRepModularFormsBasis

        EXAMPLES::

            sage: from weilrep import *
            sage: m = ParamodularForms(5)
            sage: m.borcherds_input_Qbasis(1/4, 5)
            [(0), 10 + 50*q + 260*q^2 + 1030*q^3 + 3500*q^4 + O(q^5)]
            [(1/10), 6*q^(-1/20) + 16*q^(19/20) + 82*q^(39/20) + 304*q^(59/20) + 1046*q^(79/20) + 3120*q^(99/20) + O(q^(119/20))]
            [(1/5), q^(-1/5) - 24*q^(4/5) - 143*q^(9/5) - 622*q^(14/5) - 2204*q^(19/5) - 6876*q^(24/5) + O(q^(29/5))]
            [(3/10), -16*q^(11/20) - 102*q^(31/20) - 448*q^(51/20) - 1650*q^(71/20) - 5248*q^(91/20) + O(q^(111/20))]
            [(2/5), -q^(1/5) + 14*q^(6/5) + 92*q^(11/5) + 386*q^(16/5) + 1318*q^(21/5) + O(q^(26/5))]
            [(1/2), 20*q^(3/4) + 160*q^(7/4) + 700*q^(11/4) + 2560*q^(15/4) + 8000*q^(19/4) + O(q^(23/4))]
            [(3/5), -q^(1/5) + 14*q^(6/5) + 92*q^(11/5) + 386*q^(16/5) + 1318*q^(21/5) + O(q^(26/5))]
            [(7/10), -16*q^(11/20) - 102*q^(31/20) - 448*q^(51/20) - 1650*q^(71/20) - 5248*q^(91/20) + O(q^(111/20))]
            [(4/5), q^(-1/5) - 24*q^(4/5) - 143*q^(9/5) - 622*q^(14/5) - 2204*q^(19/5) - 6876*q^(24/5) + O(q^(29/5))]
            [(9/10), 6*q^(-1/20) + 16*q^(19/20) + 82*q^(39/20) + 304*q^(59/20) + 1046*q^(79/20) + 3120*q^(99/20) + O(q^(119/20))]
            ------------------------------------------------------------
            [(0), 36 + 586*q + 3708*q^2 + 17694*q^3 + 68852*q^4 + O(q^5)]
            [(1/10), -5*q^(-1/20) - 270*q^(19/20) - 1935*q^(39/20) - 10275*q^(59/20) - 43245*q^(79/20) - 157545*q^(99/20) + O(q^(119/20))]
            [(1/5), 5*q^(-1/5) + 90*q^(4/5) + 1035*q^(9/5) + 6130*q^(14/5) + 28460*q^(19/5) + 109260*q^(24/5) + O(q^(29/5))]
            [(3/10), -185*q^(11/20) - 1595*q^(31/20) - 8505*q^(51/20) - 36145*q^(71/20) - 131485*q^(91/20) + O(q^(111/20))]
            [(2/5), 65*q^(1/5) + 630*q^(6/5) + 4100*q^(11/5) + 18940*q^(16/5) + 74070*q^(21/5) + O(q^(26/5))]
            [(1/2), 7*q^(-1/4) - 54*q^(3/4) - 747*q^(7/4) - 5026*q^(11/4) - 23859*q^(15/4) - 94960*q^(19/4) + O(q^(23/4))]
            [(3/5), 65*q^(1/5) + 630*q^(6/5) + 4100*q^(11/5) + 18940*q^(16/5) + 74070*q^(21/5) + O(q^(26/5))]
            [(7/10), -185*q^(11/20) - 1595*q^(31/20) - 8505*q^(51/20) - 36145*q^(71/20) - 131485*q^(91/20) + O(q^(111/20))]
            [(4/5), 5*q^(-1/5) + 90*q^(4/5) + 1035*q^(9/5) + 6130*q^(14/5) + 28460*q^(19/5) + 109260*q^(24/5) + O(q^(29/5))]
            [(9/10), -5*q^(-1/20) - 270*q^(19/20) - 1935*q^(39/20) - 10275*q^(59/20) - 43245*q^(79/20) - 157545*q^(99/20) + O(q^(119/20))]
        """
        S = self.gram_matrix()
        w = self.weilrep()
        wt = -S.nrows()/2
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose = verbose)
        b = matrix(Cone(p).rays())
        if verbose:
            print('I will now try to find Borcherds product inputs.')
        try:
            u = M.solve_left(b)
        except ValueError:
            return WeilRepModularFormsBasis(wt, [], w)
        Y = [v * X for v in u.rows()]
        Y.sort(key = lambda x: x.fourier_expansion()[0][2][0])
        return WeilRepModularFormsBasis(wt, Y, w)

    def borcherds_input_by_weight(self, k, prec, pole_order = None, verbose = False):
        r"""
        Compute all input functions that yield a holomorphic Borcherds product of the given weight.

        This method computes a list of all input functions that yield holomorphic Borcherds products of weight ``k``.

        INPUT:
        - ``k`` -- the weight of the products
        - ``prec`` -- the precision to which the input functions are given
        - ``pole_order`` -- optional. If given then we look only for input functions with pole order up to ``pole_order``. (Otherwise an appropriate value for pole_order is computed automatically.)

        OUTPUT: WeilRepModularFormsBasis

        EXAMPLE::

            sage: from weilrep import *
            sage: OrthogonalModularForms(CartanMatrix(['A', 2])).borcherds_input_by_weight(45, 5)
            [(0, 0), 90 + 1890*q + 17820*q^2 + 111240*q^3 + 555120*q^4 + O(q^5)]
            [(2/3, 1/3), 5*q^(-1/3) - 410*q^(2/3) - 4460*q^(5/3) - 31170*q^(8/3) - 165660*q^(11/3) - 739540*q^(14/3) + O(q^(17/3))]
            [(1/3, 2/3), 5*q^(-1/3) - 410*q^(2/3) - 4460*q^(5/3) - 31170*q^(8/3) - 165660*q^(11/3) - 739540*q^(14/3) + O(q^(17/3))]
            ------------------------------------------------------------
            [(0, 0), q^-1 + 90 + 100116*q + 8046620*q^2 + 268478010*q^3 + 5499299952*q^4 + O(q^5)]
            [(2/3, 1/3), 16038*q^(2/3) + 2125035*q^(5/3) + 89099838*q^(8/3) + 2095831260*q^(11/3) + 34171889580*q^(14/3) + O(q^(17/3))]
            [(1/3, 2/3), 16038*q^(2/3) + 2125035*q^(5/3) + 89099838*q^(8/3) + 2095831260*q^(11/3) + 34171889580*q^(14/3) + O(q^(17/3))]
        """
        S = self.gram_matrix()
        d = S.determinant()
        e = Integer(S.nrows())
        wt = -e / 2
        l = 2 - wt
        if pole_order is None:
            from .eisenstein_series import quadratic_L_function__correct
            if e % 2:
                eisenstein_bound = (2*math.pi)**l / (5 * (1 - 2**(-3-e)) * math.gamma(l) * RR(l - 0.5).zeta() * math.sqrt(d))
                for p in d.prime_factors():
                    eisenstein_bound *= (1 - p**(-1)) / (1 - p**(-3 - e))
            else:
                D = (-1) ** (e/2) * d
                l = ZZ(l)
                eisenstein_bound = (2*math.pi)**l * (2 - (1 - 2^(1-l))*RR(l - 1.0).zeta()) / (math.sqrt(d) * math.gamma(l) * quadratic_L_function__correct(l, D).n())
                for p in (2 * d).prime_factors():
                    eisenstein_bound *= (1 - p**(-1))
            pole_order = (2 * k / eisenstein_bound) ** (1 / (1 - wt))
        if verbose:
            print('I will compute an obstruction Eisenstein series to precision %d.'%ceil(pole_order))
        w = self.weilrep()
        rds = w.rds()
        norm_dict = w.norm_dict()
        e = w.dual().eisenstein_series(l, ceil(pole_order))
        e_coeffs = e.coefficients()
        N = len([g for g in rds if not norm_dict[tuple(g)]]) - 1
        pole_order = max([g_n[0][-1] for g_n in e_coeffs.items() if g_n[1] + k + k >= 0])
        if verbose:
            print('I need to consider modular forms with a pole order at most %s.'%pole_order)
        X = w.nearly_holomorphic_modular_forms_basis(wt, pole_order, prec, verbose = verbose)
        v_list = w.coefficient_vector_exponents(0, 1, starting_from = -pole_order, include_vectors = True)
        exp_list = [v[1] for v in v_list]
        v_list = [vector(v[0]) for v in v_list]
        positive = []
        zero = vector([0] * (len(exp_list) + 2))
        if N:
            M = matrix([x.coefficient_vector(starting_from = -pole_order, ending_with = 0)[:-N] for x in X])
        else:
            M = matrix([x.coefficient_vector(starting_from = -pole_order, ending_with = 0) for x in X])
        vs = M.transpose().kernel().basis()
        for i, n in enumerate(exp_list):
            ieq = copy(zero)
            ieq[i+1] = 1
            for j, m in enumerate(exp_list[:i]):
                N = sqrt(m / n)
                if N in ZZ:
                    v1 = v_list[i]
                    v2 = v_list[j]
                    ieq[j + 1] = denominator(v1 * N - v2) == 1 or denominator(v1 * N + v2) == 1
            positive.append(ieq)
        r = vector(QQ, [k + k] + [(1 + bool(2 % denominator(g))) * e_coeffs[tuple(list(g) + [-exp_list[i]])] for i, g in enumerate(v_list)] + [0])
        if verbose:
            print('I will now find integral points in a polyhedron.')
        p = Polyhedron(ieqs = positive, eqns = [r] + [vector([0] + list(v)) for v in vs])
        try:
            u = M.solve_left(matrix(p.integral_points()))
            Y = [v * X for v in u.rows()]
        except ValueError:
            Y = []
            pass
        return WeilRepModularFormsBasis(wt, Y, self.weilrep())

    def borcherds_input_by_obstruction(self, k, pole_order = None, verbose = False):
        r"""
        Compute principal parts of Borcherds products using the method of obstructions.

        This command produces *only the principal parts* of nearly-holomorphic modular forms that lift to holomorphic Borcherds products of weight ``k``. It is generally much faster than borcherds_input_by_weight(), but it does not yield any Fourier coefficients of the product and is only suited to proving existence.

        INPUT:
        - ``k`` -- the weight (of the output products)
        - ``pole_order`` -- the max pole order (default None; if None then we use the maximum possible pole order)
        - ``verbose`` -- boolean (default False) if True then we add commentary

        OUTPUT: a list of WeilRepModularForm's

        EXAMPLES::

            sage: from weilrep import *
            sage: OrthogonalModularForms(CartanMatrix(['A', 2])).borcherds_input_by_obstruction(45)
            [(0, 0), 90 + O(q)]
            [(2/3, 1/3), 5*q^(-1/3) + O(q^(2/3))]
            [(1/3, 2/3), 5*q^(-1/3) + O(q^(2/3))]
            ------------------------------------------------------------
            [(0, 0), q^-1 + 90 + O(q)]
            [(2/3, 1/3), O(q^(2/3))]
            [(1/3, 2/3), O(q^(2/3))]
        """
        S = self.gram_matrix()
        d = S.determinant()
        e = Integer(S.nrows())
        wt = -e / 2
        l = 2 - wt
        if pole_order is None:
            from .eisenstein_series import quadratic_L_function__correct
            if e % 2:
                eisenstein_bound = (2*math.pi)**l / (5 * (1 - 2**(-3-e)) * math.gamma(l) * RR(l - 0.5).zeta() * math.sqrt(d))
                for p in d.prime_factors():
                    eisenstein_bound *= (1 - p**(-1)) / (1 - p**(-3 - e))
            else:
                D = (-1) ** (e/2) * d
                l = ZZ(l)
                eisenstein_bound = (2*math.pi)**l * (2 - (1 - 2^(1-l))*RR(l - 1.0).zeta()) / (math.sqrt(d) * math.gamma(l) * quadratic_L_function__correct(l, D).n())
                for p in (2 * d).prime_factors():
                    eisenstein_bound *= (1 - p**(-1))
            pole_order = (2 * k / eisenstein_bound) ** (1 / (1 - wt))
        if verbose:
            print('I will compute the obstruction Eisenstein series to precision %d.'%ceil(pole_order))
        w = self.weilrep()
        w_dual = w.dual()
        e = w_dual.eisenstein_series(l, ceil(pole_order))
        e_coeffs = e.coefficients()
        pole_order = max([g_n[0][-1] for g_n in e_coeffs.items() if g_n[1] + k + k >= 0])
        if verbose:
            print('I need to compute the obstruction space to precision %s.'%pole_order)
        v_list = w_dual.coefficient_vector_exponents(floor(pole_order) + 1, 1, include_vectors = True)
        rds = w_dual.rds()
        norm_dict = w_dual.norm_dict()
        N = len([g for g in rds if not norm_dict[tuple(g)]])
        exp_list = [v[1] for v in v_list if v[1] and v[1] <= pole_order]
        v_list = [vector(v[0]) for v in v_list][N:len(exp_list)+N]
        positive = []
        zero = vector([0] * (len(exp_list) + 1))
        X = w_dual.cusp_forms_basis(l, pole_order, verbose = verbose)
        L = [x.coefficient_vector()[N - 1 : len(v_list) + N] for x in X]
        for i, n in enumerate(exp_list):
            ieq = copy(zero)
            ieq[i + 1] = 1
            for j, m in enumerate(exp_list[i+1:]):
                sqrtm_n = sqrt(m / n)
                if sqrtm_n in ZZ:
                    v1 = v_list[i]
                    v2 = v_list[i + j + 1]
                    ieq[i + j + 2] = bool(denominator(v1 * sqrtm_n - v2) == 1 or denominator(v1 * sqrtm_n + v2) == 1)
            positive.append(ieq)
        r = [k + k]
        for i, g in enumerate(v_list):
            try:
                c = e_coeffs[tuple(list(g) + [exp_list[i]])]
            except KeyError:
                c = 0
            mult = 1 + bool(2 % denominator(g))
            r.append(mult * c)
            if mult != 1:
                for v in L:
                    v[i + 1] += v[i + 1]
        r = vector(r)
        if verbose:
            print('I will now find integral points in a polyhedron.')
        p = Polyhedron(ieqs = positive, eqns = [r] + L)
        q, = PowerSeriesRing(QQ, 'q').gens()
        X = []
        ds = w.ds()
        ds_dict = w.ds_dict()
        norm_list = w.norm_list()
        for v in p.integral_points():
            Y = [None] * len(ds)
            for j, g in enumerate(ds):
                if norm_list[j] or not g:
                    Y[j] = [g, norm_list[j], O(q ** 1)]
                else:
                    Y[j] = [g, 0, O(q ** 0)]
            for i, v_i in enumerate(v):
                if v_i:
                    g = v_list[i]
                    n = ceil(-exp_list[i])
                    j = ds_dict[tuple(g)]
                    j2 = ds_dict[tuple([frac(-x) for x in g])]
                    Y[j][2] += v_i * q ** n
                    if j2 != j:
                        Y[j2][2] += v_i * q ** n
            Y[0][2] += k + k
            X.append(WeilRepModularForm(wt, S, Y, w))
        return WeilRepModularFormsBasis(wt, X, w)

def ParamodularForms(N):
    r"""
    Orthogonal modular forms for the lattice A_1(N) + II_{2,2}.
    """
    return OrthogonalModularForms(matrix([[N + N]]))

class OrthogonalModularForm:
    r"""
    This class represents modular forms on type IV domains attached to lattices of the form L + II_{2,2}, where L is positive-definite.

    INPUT: Orthogonal modular forms are constructed by calling ``OrthogonalModularForm(k, S, f, scale, weylvec)``, where:
    - ``k`` -- the weight
    - ``S`` -- Gram matrix of the underlying positive-definite lattice.
    - ``f`` -- the Fourier expansion. This is a power series in the variables q, s over a base ring of Laurent polynomials in the variables r_0,...,r_d
    - ``scale`` -- a natural number. If scale != 1 then all exponents in the Fourier expansion should be divided by `scale`. (This is done in the __repr__ method)
    - ``weylvec`` -- a ``Weyl vector``. For modular forms that are not constructed as Borcherds products this vector should be zero.
    - ``precision`` -- optional: the total precision of the Fourier expansion
    - ``valuation`` -- optional: the total valuation of the Fourier expansion
    """
    def __init__(self, k, S, f, scale, weylvec):
        self.__weight = k
        self.__gram_matrix = S
        self.__fourier_expansion = f
        self.__scale = scale
        self.__valuation = f.valuation()
        self.__weylvec = weylvec

    def __repr__(self):
        r"""
        Print self's Fourier expansion. If 'scale' > 1 then we divide all exponents by 'scale' in order to print modular forms with characters. If the Gram matrix is 1x1 then we pretend to use the variable 'r' instead of 'r_0'.
        """
        try:
            return self.__string
        except AttributeError:
            s = str(self.fourier_expansion())
            d = self.__scale
            if d == 1:
                self.__string = s
            else: #divide by scale
                def m(obj):
                    obj_s = obj.string[slice(*obj.span())]
                    x = obj_s[0]
                    if x == '^':
                        u = ZZ(obj_s[1:])/d
                        if u.is_integer():
                            if u == 1:
                                return ''
                            return '^%d'%u
                        return '^(%s)'%u
                    return (x, obj_s)[x == '_'] + '^(%s)'%(1/d)
                self.__string = sub(r'\^-?\d+|(?<!O\(|\, )(\_\d+|q|s)(?!\^)', m, s)
            if self.gram_matrix().nrows() == 1:
                self.__string = self.__string.replace('r_0', 'r')
            return self.__string

    ## basic attributes

    def base_ring(self):
        r"""
        Return the underlying Laurent polynomial ring.
        """
        return self.__fourier_expansion.base_ring().base_ring()

    def gram_matrix(self):
        r"""
        Return our Gram matrix.
        """
        return self.__gram_matrix

    def nvars(self):
        return 2 + Integer(self.gram_matrix().nrows())

    def precision(self):
        r"""
        Return our precision.
        """
        return Integer(self.true_fourier_expansion().prec()) / self.scale()

    def rescale(self, d):
        r"""
        Rescale the Fourier expansion of an orthogonal modular form.

        For internal use. This rescales q, s, and all r_i in self's Fourier expansion by the power 'd'. This does not happen in-place.

        NOTE: this method will not appear to have any effect unless you explicitly call .fourier_expansion() or .true_fourier_expansion()
        """
        f = self.true_fourier_expansion()
        rb_x = f.base_ring()
        x, = rb_x.gens()
        rbgens = rb_x.base_ring().gens()
        S = self.__gram_matrix
        rescale_dict = {a : a ** d for a in rbgens}
        return OrthogonalModularForm(self.__weight, S, (f.map_coefficients(lambda y: (x ** (d * y.polynomial_construction()[1])) * rb_x([p.subs(rescale_dict) for p in list(y)]).subs({x : x ** d}))).V(d), scale = self.scale() * d, weylvec = self.weyl_vector())

    def scale(self):
        r"""
        Return self's scale.
        """
        return self.__scale

    def valuation(self):
        r"""
        Return self's valuation.
        """
        return self.__valuation

    def weight(self):
        r"""
        Return self's weight.
        """
        return self.__weight

    def weyl_vector(self):
        r"""
        Return self's Weyl vector.
        """
        return self.__weylvec

    ## methods to extract Fourier coefficients

    def coefficients(self, prec=+Infinity):
        r"""
        Return a dictionary of self's known Fourier coefficients.

        The input into the dictionary should be a tuple of the form (a, b_0, ..., b_d, c). The output will then be the Fourier coefficient of the monomial q^a r_0^(b_0)...r_d^(b_d) s^c.

        EXAMPLES::

            sage: from weilrep import *
            sage: f = ParamodularForms(4).borcherds_input_by_weight(1/2, 10)[0].borcherds_lift()
            sage: f.coefficients()[(1/8, -1/2, 1/8)]
            -1
        """
        L = {}
        d = self.scale()
        nrows = self.gram_matrix().nrows()
        f = self.fourier_expansion()
        coeffs = f.coefficients()
        q, s = f.parent().gens()
        d_prec = d * prec
        for j, x in coeffs.items():
            a, c = j.exponents()[0]
            if a+c < d_prec:
                x_coeffs = x.coefficients()
                if nrows > 1:
                    for i, y in enumerate(x.exponents()):
                        g = tuple([a/d] + list(vector(y)/d) + [c/d])
                        L[g] = x_coeffs[i]
                else:
                    for i, y in enumerate(x.exponents()):
                        g = a/d, y/d, c/d
                        L[g] = x_coeffs[i]
        return L

    def fourier_expansion(self):
        r"""
        Return our Fourier expansion as a power series in 'q' and 's' over a ring of Laurent polynomials in the variables 'r_i'.

        EXAMPLES::

            sage: from weilrep import *
            sage: f = ParamodularForms(1).spezialschar(10, 5)[0]
            sage: f.fourier_expansion()
            (r_0^-1 - 2 + r_0)*q*s + (-2*r_0^-2 - 16*r_0^-1 + 36 - 16*r_0 - 2*r_0^2)*q^2*s + (-2*r_0^-2 - 16*r_0^-1 + 36 - 16*r_0 - 2*r_0^2)*q*s^2 + (r_0^-3 + 36*r_0^-2 + 99*r_0^-1 - 272 + 99*r_0 + 36*r_0^2 + r_0^3)*q^3*s + (-16*r_0^-3 + 240*r_0^-2 - 240*r_0^-1 + 32 - 240*r_0 + 240*r_0^2 - 16*r_0^3)*q^2*s^2 + (r_0^-3 + 36*r_0^-2 + 99*r_0^-1 - 272 + 99*r_0 + 36*r_0^2 + r_0^3)*q*s^3 + O(q, s)^5
        """
        try:
            return self.__qexp
        except AttributeError:
            h = self.__fourier_expansion
            q, s = PowerSeriesRing(self.base_ring(), ('q', 's')).gens()
            self.__qexp = O(q ** h.prec()) + sum([(q ** ((i - n) // 2)) * (s ** ((i + n) // 2)) * p.coefficients()[j] for i, p in enumerate(h.list()) for j, n in enumerate(p.exponents()) ])
            return self.__qexp

    def fourier_jacobi(self):
        r"""
        Computes self's Fourier--Jacobi expansion.

        OUTPUT: a list [phi_0, phi_1, ... phi_N] where N is self's precision. Each phi_i is a JacobiForm of index N * S where S is self's gram matrix. The series \sum_{i=0}^{\infty} \phi_i(\tau, z) s^i recovers the Fourier expansion where q = e^{2pi i \tau} and w_i = e^{2pi i z_i}.

        EXAMPLES::

            sage: from weilrep import *
            sage: f = ParamodularForms(1).spezialschar(10, 5)[0]
            sage: f.fourier_jacobi()
            [O(q^5), (w^-1 - 2 + w)*q + (-2*w^-2 - 16*w^-1 + 36 - 16*w - 2*w^2)*q^2 + (w^-3 + 36*w^-2 + 99*w^-1 - 272 + 99*w + 36*w^2 + w^3)*q^3 + O(q^4), (-2*w^-2 - 16*w^-1 + 36 - 16*w - 2*w^2)*q + (-16*w^-3 + 240*w^-2 - 240*w^-1 + 32 - 240*w + 240*w^2 - 16*w^3)*q^2 + O(q^3), (w^-3 + 36*w^-2 + 99*w^-1 - 272 + 99*w + 36*w^2 + w^3)*q + O(q^2), O(q^1)]

            sage: from weilrep import *
            sage: f = ParamodularForms(4).borcherds_input_by_weight(1/2, 5)[0].borcherds_lift()
            sage: (f ** 8).fourier_jacobi()
            [O(q^6), (w^-4 - 8*w^-3 + 28*w^-2 - 56*w^-1 + 70 - 56*w + 28*w^2 - 8*w^3 + w^4)*q + (-8*w^-5 + 56*w^-4 - 168*w^-3 + 288*w^-2 - 336*w^-1 + 336 - 336*w + 288*w^2 - 168*w^3 + 56*w^4 - 8*w^5)*q^2 + (28*w^-6 - 168*w^-5 + 420*w^-4 - 616*w^-3 + 756*w^-2 - 1008*w^-1 + 1176 - 1008*w + 756*w^2 - 616*w^3 + 420*w^4 - 168*w^5 + 28*w^6)*q^3 + (-56*w^-7 + 288*w^-6 - 616*w^-5 + 896*w^-4 - 1400*w^-3 + 2016*w^-2 - 2024*w^-1 + 1792 - 2024*w + 2016*w^2 - 1400*w^3 + 896*w^4 - 616*w^5 + 288*w^6 - 56*w^7)*q^4 + O(q^5), (-8*w^-5 + 56*w^-4 - 168*w^-3 + 288*w^-2 - 336*w^-1 + 336 - 336*w + 288*w^2 - 168*w^3 + 56*w^4 - 8*w^5)*q + (8*w^-8 - 56*w^-7 + 224*w^-6 - 616*w^-5 + 1120*w^-4 - 1400*w^-3 + 1568*w^-2 - 2024*w^-1 + 2352 - 2024*w + 1568*w^2 - 1400*w^3 + 1120*w^4 - 616*w^5 + 224*w^6 - 56*w^7 + 8*w^8)*q^2 + (-56*w^-9 + 336*w^-8 - 1008*w^-7 + 2016*w^-6 - 2856*w^-5 + 3360*w^-4 - 4536*w^-3 + 6048*w^-2 - 5880*w^-1 + 5152 - 5880*w + 6048*w^2 - 4536*w^3 + 3360*w^4 - 2856*w^5 + 2016*w^6 - 1008*w^7 + 336*w^8 - 56*w^9)*q^3 + O(q^4), (28*w^-6 - 168*w^-5 + 420*w^-4 - 616*w^-3 + 756*w^-2 - 1008*w^-1 + 1176 - 1008*w + 756*w^2 - 616*w^3 + 420*w^4 - 168*w^5 + 28*w^6)*q + (-56*w^-9 + 336*w^-8 - 1008*w^-7 + 2016*w^-6 - 2856*w^-5 + 3360*w^-4 - 4536*w^-3 + 6048*w^-2 - 5880*w^-1 + 5152 - 5880*w + 6048*w^2 - 4536*w^3 + 3360*w^4 - 2856*w^5 + 2016*w^6 - 1008*w^7 + 336*w^8 - 56*w^9)*q^2 + O(q^3), (-56*w^-7 + 288*w^-6 - 616*w^-5 + 896*w^-4 - 1400*w^-3 + 2016*w^-2 - 2024*w^-1 + 1792 - 2024*w + 2016*w^2 - 1400*w^3 + 896*w^4 - 616*w^5 + 288*w^6 - 56*w^7)*q + O(q^2), O(q^1)]

        """
        try:
            return self.__fourier_jacobi
        except AttributeError:
            pass
        S = self.gram_matrix()
        nrows = S.nrows()
        rb = LaurentPolynomialRing(QQ, list(var('w_%d' % i) for i in range(nrows)))
        z = rb.gens()[0]
        r, q = PowerSeriesRing(rb, 'q').objgen()
        k = self.weight()
        if self.scale() != 1:
            v = self.weyl_vector()
            if v[0] in ZZ and v[-1] in ZZ: #ok we'll try this but it's not a good solution
                prec = self.precision()
                L = [O(q ** (prec - n)) for n in range(prec)]
                coeffs = self.coefficients()
                for x, y in coeffs.items():
                    a = x[0]
                    c = x[2]
                    b = x[1:-1]
                    if nrows > 1:
                        u = rb.monomial(*b)
                    else:
                        u = z**b[0]
                    L[c] += (q ** a) * u * y
                self.__fourier_jacobi = [JacobiForm(k, n * S, j) for n, j in enumerate(L)]
                return self.__fourier_jacobi
            raise NotImplementedError('Nontrivial character')
        f = self.fourier_expansion()
        rb_old = f.base_ring()
        r_old = f.parent()
        s = r_old.gens()[1]
        r_new = r_old.remove_var(s)
        change_name = {rb_old('r_%d'%j):rb('w_%d'%j) for j in range(nrows)}
        prec = self.precision()
        f = f.polynomial()
        _change_ring = lambda f: r([x.subs(change_name) for x in f.list()])
        self.__fourier_jacobi = [JacobiForm(k, n * S, _change_ring(r_new(f.coefficient({s : n}))) + O(q ** (prec - n))) for n in range(prec)]
        return self.__fourier_jacobi


    def true_fourier_expansion(self):
        r"""
        Return self's Fourier expansion as it is actually stored (as a univariate power series over a ring of Laurent polynomials).

        EXAMPLE::

            sage: from weilrep import *
            sage: f = ParamodularForms(4).borcherds_input_by_weight(1/2, 10)[0].borcherds_lift()
            sage: f.true_fourier_expansion()
            ((-r_0^-4 + r_0^4))*t^2 + ((r_0^-12 - r_0^12)*x^-8 + (r_0^-12 - r_0^12)*x^8)*t^10 + ((-r_0^-36 + r_0^36))*t^18 + ((-r_0^-20 + r_0^20)*x^-24 + (-r_0^-20 + r_0^20)*x^24)*t^26 + ((r_0^-60 - r_0^60)*x^-16 + (r_0^-60 - r_0^60)*x^16)*t^34 + O(t^50)
        """
        return self.__fourier_expansion


    ## arithmetic operations

    def __add__(self, other):
        r"""
        Add modular forms, rescaling if necessary.
        """
        if not other:
            return self
        if not self.gram_matrix() == other.gram_matrix():
            raise ValueError('Incompatible Gram matrices')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        self_v = self.weyl_vector()
        other_v = other.weyl_vector()
        if self_v or other_v:
            if not denominator(self_v - other_v) == 1:
                raise ValueError('Incompatible characters')
        self_scale = self.scale()
        other_scale = other.scale()
        if not self_scale == other_scale:
            new_scale = lcm(self_scale, other_scale)
            X1 = self.rescale(new_scale // self_scale)
            X2 = other.rescale(new_scale // other_scale)
            return OrthogonalModularForm(self.__weight, self.__gram_matrix, X1.true_fourier_expansion() + X2.true_fourier_expansion(), scale = new_scale, weylvec = self_v)
        return OrthogonalModularForm(self.__weight, self.__gram_matrix, self.true_fourier_expansion() + other.true_fourier_expansion(), scale = self_scale, weylvec = self_v)

    __radd__ = __add__

    def __sub__(self, other):
        r"""
        Subtract modular forms, rescaling if necessary.
        """
        if not other:
            return self
        if not self.gram_matrix() == other.gram_matrix():
            raise ValueError('Incompatible Gram matrices')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        self_v = self.weyl_vector()
        other_v = other.weyl_vector()
        if self_v or other_v:
            if not denominator(self_v - other_v) == 1:
                raise ValueError('Incompatible characters')
        self_scale = self.scale()
        other_scale = other.scale()
        if not self_scale == other_scale:
            new_scale = lcm(self_scale, other_scale)
            X1 = self.rescale(new_scale // self_scale)
            X2 = other.rescale(new_scale // other_scale)
            return OrthogonalModularForm(self.__weight, self.__gram_matrix, X1.true_fourier_expansion() - X2.true_fourier_expansion(), scale = new_scale, weylvec = self_v)
        return OrthogonalModularForm(self.__weight, self.__gram_matrix, self.true_fourier_expansion() - other.true_fourier_expansion(), scale = self_scale, weylvec = self_v)

    def __neg__(self):
        return OrthogonalModularForm(self.__weight, self.__gram_matrix, -self.true_fourier_expansion(), scale = self.scale(), weylvec = self.weyl_vector())

    def __mul__(self, other):
        r"""
        Multiply modular forms, rescaling if necessary.
        """
        if isinstance(other, OrthogonalModularForm):
            if not self.gram_matrix() == other.gram_matrix():
                raise ValueError('Incompatible Gram matrices')
            self_scale = self.scale()
            other_scale = other.scale()
            if self_scale != 1 or other_scale != 1:
                new_scale = lcm(self.scale(), other.scale())
                X1 = self.rescale(new_scale // self_scale)
                X2 = other.rescale(new_scale // other_scale)
                return OrthogonalModularForm(self.__weight + other.weight(), self.__gram_matrix, X1.true_fourier_expansion() * X2.true_fourier_expansion(), scale = new_scale, weylvec = self.weyl_vector() + other.weyl_vector())
            return OrthogonalModularForm(self.weight() + other.weight(), self.__gram_matrix, self.true_fourier_expansion() * other.true_fourier_expansion(), scale = 1, weylvec = self.weyl_vector() + other.weyl_vector())
        elif other in QQ:
            return OrthogonalModularForm(self.weight(), self.__gram_matrix, self.true_fourier_expansion() * other, scale = self.scale(), weylvec = self.weyl_vector())

    __rmul__ = __mul__

    def __div__(self, other):
        r"""
        Divide modular forms, rescaling if necessary.
        """
        if isinstance(other, OrthogonalModularForm):
            if not self.gram_matrix() == other.gram_matrix():
                raise ValueError('Incompatible Gram matrices')
            self_scale = self.scale()
            other_scale = other.scale()
            if self_scale != 1 or other_scale != 1:
                new_scale = lcm(self.scale(), other.scale())
                X1 = self.rescale(new_scale // self_scale)
                X2 = other.rescale(new_scale // other_scale)
                return OrthogonalModularForm(self.__weight - other.weight(), self.__gram_matrix, X1.true_fourier_expansion() / X2.true_fourier_expansion(), scale = new_scale, weylvec = self.weyl_vector() - other.weyl_vector())
            return OrthogonalModularForm(self.weight() - other.weight(), self.__gram_matrix, self.true_fourier_expansion() / other.true_fourier_expansion(), scale = 1, weylvec = self.weyl_vector() - other.weyl_vector())
        elif other in QQ:
            return OrthogonalModularForm(self.weight(), self.__gram_matrix, self.true_fourier_expansion() / other, scale = self.scale(), weylvec = self.weyl_vector())

    __truediv__ = __div__

    def __eq__(self, other):
        self_scale = self.scale()
        other_scale = other.scale()
        if self_scale == other_scale:
            return self.true_fourier_expansion() == other.true_fourier_expansion()
        else:
            new_scale = lcm(self_scale, other_scale)
            X1 = self.rescale(new_scale // self_scale)
            X2 = other.rescale(new_scale // other_scale)
            return X1.true_fourier_expansion() == X2.true_fourier_expansion()

    def __pow__(self, other):
        if not other in ZZ:
            raise ValueError('Not a valid exponent')
        return OrthogonalModularForm(other * self.weight(), self.__gram_matrix, self.true_fourier_expansion() ** other, scale=self.scale(), weylvec = other * self.weyl_vector())

    ## other methods

    def is_lift(self):
        r"""
        Return True if self's Fourier coefficients satisfy the Maass relations, otherwise False

        ALGORITHM: check whether this equals the Gritsenko lift of its first Fourier--Jacobi coefficient

        WARNING: this always outputs True if the precision is too low!

        OUTPUT: True/False

        EXAMPLES::

            sage: from weilrep import *
            sage: ParamodularForms(N = 1).borcherds_input_by_weight(10, 10)[0].borcherds_lift().is_lift()
            True

            sage: from weilrep import *
            sage: ParamodularForms(N = 1).borcherds_input_by_weight(24, 10)[0].borcherds_lift().is_lift()
            False
        """
        try:
            X = self.fourier_jacobi()
            f = X[1]
            return all(f.hecke_V(N) == x for N, x in enumerate(X) if N)
        except NotImplementedError:
            return False

def orthogonal_eisenstein_series(k, S, prec, w = None):
    r"""
    Computes the "orthogonal Eisenstein series" as the theta-lift of the vector-valued Eisenstein series E_{k, 0} (if it exists). We renormalize such that the constant term is 1.

    NOTE: this is called by OrthogonalModularForms(S).eisenstein_series()

    INPUT:
    - ``k`` -- weight
    - ``S`` -- (positive-definite) Gram matrix
    - ``prec`` -- precision
    - ``w`` -- a WeilRep instance (to check if the Eisenstein series has been cached)

    OUTPUT: OrthogonalModularForm

    EXAMPLES::

        sage: from weilrep import *
        sage: orthogonal_eisenstein_series(4, matrix([[4]]), 5)
        1 + 240*q + 240*s + 2160*q^2 + (3360*r^-2 + 15360*r^-1 + 20160 + 15360*r + 3360*r^2)*q*s + 2160*s^2 + 6720*q^3 + (240*r^-4 + 15360*r^-3 + 67200*r^-2 + 107520*r^-1 + 137760 + 107520*r + 67200*r^2 + 15360*r^3 + 240*r^4)*q^2*s + (240*r^-4 + 15360*r^-3 + 67200*r^-2 + 107520*r^-1 + 137760 + 107520*r + 67200*r^2 + 15360*r^3 + 240*r^4)*q*s^2 + 6720*s^3 + 17520*q^4 + (20160*r^-4 + 107520*r^-3 + 201600*r^-2 + 322560*r^-1 + 309120 + 322560*r + 201600*r^2 + 107520*r^3 + 20160*r^4)*q^3*s + (15360*r^-5 + 164640*r^-4 + 322560*r^-3 + 691200*r^-2 + 645120*r^-1 + 987840 + 645120*r + 691200*r^2 + 322560*r^3 + 164640*r^4 + 15360*r^5)*q^2*s^2 + (20160*r^-4 + 107520*r^-3 + 201600*r^-2 + 322560*r^-1 + 309120 + 322560*r + 201600*r^2 + 107520*r^3 + 20160*r^4)*q*s^3 + 17520*s^4 + O(q, s)^5

    """
    if w is None:
        try:
            from .weilrep import WeilRep
            w = WeilRep(S)
        except AttributeError:
            w, S = S, S.gram_matrix()
    nrows = Integer(S.nrows())
    try:
        return (-((k + k) / bernoulli(k)) * w.eisenstein_series(k + Integer(w.is_lorentzian()) - nrows/2, ceil(prec * prec / 4) + 1)).theta_lift(prec)
    except ValueError:
        raise ValueError('Invalid weight')

class WeilRepPositiveDefinite(WeilRep):
    def __init__(self):
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}

    def is_lorentzian(self):
        return False

    def is_lorentzian_plus_II(self):
        return False

    def is_positive_definite(self):
        return True

    def jacobi_forms(self):
        return JacobiForms(self.gram_matrix(), weilrep = self)

    def __add__(self, other):
        from .lorentz import RescaledHyperbolicPlane, WeilRepLorentzian
        if isinstance(other, RescaledHyperbolicPlane):
            S = self.gram_matrix()
            zerom = matrix([[0]])
            zerov = matrix([[0]*S.nrows()])
            zerovt = zerov.transpose()
            N = other._N()
            return WeilRepLorentzian(block_matrix([[zerom, zerov, N], [zerovt, S, zerovt], [N, zerov, -(N + N)]], subdivide = False), lift_qexp_representation = 'PD+II')
        from .weilrep import WeilRep
        if isinstance(other, WeilRep):
            return WeilRep(block_diagonal_matrix([self.gram_matrix(), other.gram_matrix()], subdivide = False))
        return NotImplemented
    __radd__ = __add__

class WeilRepModularFormPositiveDefinite(WeilRepModularForm):
    r"""
    subclass of WeilRepModularForms for positive definite lattices. This adds methods for constructing Jacobi forms and theta lifts.
    """
    def __init__(self, k, f, w):
        self._WeilRepModularForm__weight = k
        self._WeilRepModularForm__fourier_expansions = f
        self._WeilRepModularForm__weilrep = w
        self._WeilRepModularForm__gram_matrix = w.gram_matrix()

    ## special methods

    def jacobi_form(self):
        r"""
        Return the Jacobi form associated to self.

        If the Gram matrix is positive-definite (this is not checked!!) then this returns the Jacobi form whose theta-decomposition is the vector valued modular form that we started with.

        OUTPUT: a JacobiForm

        EXAMPLES::

            sage: from weilrep import *
            sage: WeilRep(matrix([[2,1],[1,2]])).eisenstein_series(3, 3).jacobi_form()
            1 + (w_0^2*w_1 + w_0*w_1^2 + 27*w_0*w_1 + 27*w_0 + 27*w_1 + w_0*w_1^-1 + 72 + w_0^-1*w_1 + 27*w_1^-1 + 27*w_0^-1 + 27*w_0^-1*w_1^-1 + w_0^-1*w_1^-2 + w_0^-2*w_1^-1)*q + (27*w_0^2*w_1^2 + 72*w_0^2*w_1 + 72*w_0*w_1^2 + 27*w_0^2 + 216*w_0*w_1 + 27*w_1^2 + 216*w_0 + 216*w_1 + 72*w_0*w_1^-1 + 270 + 72*w_0^-1*w_1 + 216*w_1^-1 + 216*w_0^-1 + 27*w_1^-2 + 216*w_0^-1*w_1^-1 + 27*w_0^-2 + 72*w_0^-1*w_1^-2 + 72*w_0^-2*w_1^-1 + 27*w_0^-2*w_1^-2)*q^2 + O(q^3)
        """
        X = self.fourier_expansion()
        S = self.gram_matrix()
        prec = self.precision()
        val = self.valuation()
        e = Integer(S.nrows())
        Rb = LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(e) ))
        R, q = PowerSeriesRing(Rb, 'q', prec).objgen()
        if e > 1:
            _ds_dict = self.weilrep().ds_dict()
            jf = [Rb(0)]*(prec-val)
            precval = prec - val
            S_inv = S.inverse()
            try:
                _, _, vs_matrix = pari(S_inv).qfminim(precval + precval + 1, flag = 2)
                vs_list = vs_matrix.sage().columns()
                symm = self.is_symmetric()
                symm = 1 if symm else -1
                for v in vs_list:
                    wv = Rb.monomial(*v)
                    r = S_inv * v
                    r_norm = v*r / 2
                    i_start = ceil(r_norm)
                    j = _ds_dict[tuple(frac(x) for x in r)]
                    f = X[j][2]
                    m = ceil(i_start + val - r_norm)
                    for i in range(i_start, precval):
                        jf[i] += (wv + symm / wv) * f[m]
                        m += 1
                f = X[0][2]#deal with v=0 separately
                for i in range(precval):
                    jf[i] += f[ceil(val) + i]
            except PariError: #oops!
                lvl = Q.level()
                Q = QuadraticForm(S)
                S_adj = lvl*S_inv
                vs = QuadraticForm(S_adj).short_vector_list_up_to_length(lvl*(prec - val))
                for n in range(len(vs)):
                    r_norm = n/lvl
                    i_start = ceil(r_norm)
                    for v in vs[n]:
                        r = S_inv*v
                        rfrac = tuple(frac(r[i]) for i in range(e))
                        wv = Rb.monomial(*v)
                        j = _ds_dict[rfrac]
                        f = X[j][2]
                        m = ceil(i_start + val - r_norm)
                        for i in range(i_start,prec):
                            jf[i] += wv*f[m]
                            m += 1
                pass
        else:
            w, = Rb.gens()
            m = S[0,0] #twice the index
            eps = 2*self.is_symmetric()-1
            jf = [X[0][2][i] + sum(X[r%m][2][ceil(i - r*r / (2*m))]*(w**r + eps * w**QQ(-r)) for r in range(1,isqrt(2*(i-val)*m)+1)) for i in range(val, prec)]
        return JacobiForm(self.weight()+e/2, S, q ** val * R(jf) + O(q**prec), weilrep = self.weilrep(), modform = self)

    def theta_lift(self, prec = None):
        r"""
        Compute the additive theta lift.

        This computes the additive theta lift (Gritsenko lift, Maass lift) of the given vector-valued modular form.

        INPUT:
        - ``prec`` -- precision of the output (default None). The precision is limited by the precision of the input form. If ``prec`` is given then the output precision will not *exceed* ``prec``.

        OUTPUT: an OrthogonalModularForm of weight = input form's weight + (lattice rank) / 2

        EXAMPLES::

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[2, 1], [1, 2]]))
            sage: w.cusp_forms_basis(9, 5)[0].theta_lift()
            (r_0*r_1 + r_0 + r_1 - 6 + r_1^-1 + r_0^-1 + r_0^-1*r_1^-1)*q*s + (r_0^2*r_1^2 - 6*r_0^2*r_1 - 6*r_0*r_1^2 + r_0^2 - 10*r_0*r_1 + r_1^2 - 10*r_0 - 10*r_1 - 6*r_0*r_1^-1 + 90 - 6*r_0^-1*r_1 - 10*r_1^-1 - 10*r_0^-1 + r_1^-2 - 10*r_0^-1*r_1^-1 + r_0^-2 - 6*r_0^-1*r_1^-2 - 6*r_0^-2*r_1^-1 + r_0^-2*r_1^-2)*q^2*s + (r_0^2*r_1^2 - 6*r_0^2*r_1 - 6*r_0*r_1^2 + r_0^2 - 10*r_0*r_1 + r_1^2 - 10*r_0 - 10*r_1 - 6*r_0*r_1^-1 + 90 - 6*r_0^-1*r_1 - 10*r_1^-1 - 10*r_0^-1 + r_1^-2 - 10*r_0^-1*r_1^-1 + r_0^-2 - 6*r_0^-1*r_1^-2 - 6*r_0^-2*r_1^-1 + r_0^-2*r_1^-2)*q*s^2 + (r_0^3*r_1^2 + r_0^2*r_1^3 + r_0^3*r_1 - 10*r_0^2*r_1^2 + r_0*r_1^3 + 90*r_0^2*r_1 + 90*r_0*r_1^2 - 10*r_0^2 + 8*r_0*r_1 - 10*r_1^2 + r_0^2*r_1^-1 + 8*r_0 + 8*r_1 + r_0^-1*r_1^2 + 90*r_0*r_1^-1 - 540 + 90*r_0^-1*r_1 + r_0*r_1^-2 + 8*r_1^-1 + 8*r_0^-1 + r_0^-2*r_1 - 10*r_1^-2 + 8*r_0^-1*r_1^-1 - 10*r_0^-2 + 90*r_0^-1*r_1^-2 + 90*r_0^-2*r_1^-1 + r_0^-1*r_1^-3 - 10*r_0^-2*r_1^-2 + r_0^-3*r_1^-1 + r_0^-2*r_1^-3 + r_0^-3*r_1^-2)*q^3*s + (-6*r_0^3*r_1^3 - 10*r_0^3*r_1^2 - 10*r_0^2*r_1^3 - 10*r_0^3*r_1 + 520*r_0^2*r_1^2 - 10*r_0*r_1^3 - 6*r_0^3 - 540*r_0^2*r_1 - 540*r_0*r_1^2 - 6*r_1^3 + 520*r_0^2 + 310*r_0*r_1 + 520*r_1^2 - 10*r_0^2*r_1^-1 + 310*r_0 + 310*r_1 - 10*r_0^-1*r_1^2 - 540*r_0*r_1^-1 - 1584 - 540*r_0^-1*r_1 - 10*r_0*r_1^-2 + 310*r_1^-1 + 310*r_0^-1 - 10*r_0^-2*r_1 + 520*r_1^-2 + 310*r_0^-1*r_1^-1 + 520*r_0^-2 - 6*r_1^-3 - 540*r_0^-1*r_1^-2 - 540*r_0^-2*r_1^-1 - 6*r_0^-3 - 10*r_0^-1*r_1^-3 + 520*r_0^-2*r_1^-2 - 10*r_0^-3*r_1^-1 - 10*r_0^-2*r_1^-3 - 10*r_0^-3*r_1^-2 - 6*r_0^-3*r_1^-3)*q^2*s^2 + (r_0^3*r_1^2 + r_0^2*r_1^3 + r_0^3*r_1 - 10*r_0^2*r_1^2 + r_0*r_1^3 + 90*r_0^2*r_1 + 90*r_0*r_1^2 - 10*r_0^2 + 8*r_0*r_1 - 10*r_1^2 + r_0^2*r_1^-1 + 8*r_0 + 8*r_1 + r_0^-1*r_1^2 + 90*r_0*r_1^-1 - 540 + 90*r_0^-1*r_1 + r_0*r_1^-2 + 8*r_1^-1 + 8*r_0^-1 + r_0^-2*r_1 - 10*r_1^-2 + 8*r_0^-1*r_1^-1 - 10*r_0^-2 + 90*r_0^-1*r_1^-2 + 90*r_0^-2*r_1^-1 + r_0^-1*r_1^-3 - 10*r_0^-2*r_1^-2 + r_0^-3*r_1^-1 + r_0^-2*r_1^-3 + r_0^-3*r_1^-2)*q*s^3 + O(q, s)^5
        """
        if self.valuation() < 0:
            raise ValueError('The input function must be holomorphic')
        prec0 = self.precision() + 1
        newprec = isqrt(4 * prec0 + 4)
        if prec is None:
            prec = newprec
        else:
            prec = min(prec, newprec)
        S = self.gram_matrix()
        coeffs = self.coefficients()
        S_inv = self.inverse_gram_matrix()
        rb = LaurentPolynomialRing(QQ, list(var('r_%d' % i) for i in range(S.nrows()) ))
        z = rb.gens()[0]
        rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        t, = PowerSeriesRing(rb_x, 't').gens()
        k = self.weight()
        nrows = ZZ(S.nrows())
        wt = k + nrows / 2
        if nrows > 1:
            _, _, vs_matrix = pari(S_inv).qfminim(prec0 + 1, flag = 2)
            vs_list = vs_matrix.sage().columns()
        else:
            vs_list = [vector([n]) for n in srange(1, isqrt(2 * prec0*S[0, 0]) + 1)]
        f = O(t ** prec)
        if wt % 2 == 0:
            try:
                f_0 = coeffs[tuple([0]*(S.nrows() + 1))]
                if f_0:
                    e = eisenstein_series_qexp(wt, prec + 1)
                    f = f + f_0 * (e(t*x) + e(~x * t) - e[0])
            except KeyError:
                pass
        for v in vs_list:
            j = next(j for j, w in enumerate(v) if w)
            if v[j] > 0:
                g = S_inv * v
                v_norm = v * g / 2
                sqrt_v_norm = ceil(sqrt(v_norm))
                if nrows > 1:
                    v_monomial = rb.monomial(*v)
                    v_monomial = v_monomial + (-1)**wt * v_monomial**(-1)
                else:
                    v_monomial = z**v[0] + (-1)**wt * z**(-v[0])
                a = max(1, sqrt_v_norm)
                while a < prec:
                    c = max(1, ceil(v_norm / a))
                    while c < min(a + 1, prec- a):
                        a_plus_c = a + c
                        n = a * c - v_norm
                        if n >= 0:
                            sum_coeff = 0
                            for d in divisors(GCD([a, c] + list(v))):
                                d_wt = d ** (wt - 1)
                                g_n = tuple([frac(y) for y in g / d] + [n / (d * d)])
                                try:
                                    sum_coeff += coeffs[g_n] * d_wt
                                except KeyError:
                                    if n >= prec0:
                                        prec = min(a_plus_c, prec)
                                        f += O(t ** prec)
                                    pass
                            if a != c:
                                f += sum_coeff * t**(a_plus_c) * (x**(a-c) + x**(c-a)) * v_monomial
                            else:
                                f += sum_coeff * t**(a_plus_c) * v_monomial
                        c += 1
                    a += 1
        for a in range(prec):
            for c in range(min(a + 1, prec - a)):
                n = a * c
                a_plus_c = a + c
                if n:
                    sum_coeff = 0
                    for d in divisors(GCD(a,c)):
                        d_wt = d ** (wt - 1)
                        g_n = tuple([0]*nrows + [n / (d * d)])
                        try:
                            sum_coeff = sum_coeff + coeffs[g_n] * d_wt
                        except KeyError:
                            if n >= prec0:
                                prec = min(a_plus_c, prec)
                                f += O(t ** prec)
                            pass
                    if a != c:
                        f += sum_coeff * t**(a_plus_c) * (x**(a - c) + x**(c - a))
                    else:
                        f += sum_coeff * t**(a_plus_c)
        return OrthogonalModularForm(wt, S, f, scale = 1, weylvec = vector([0] * (nrows + 2)))
    additive_lift = theta_lift
    gritsenko_lift = theta_lift
    maass_lift = theta_lift

    def weyl_vector(self):
        r"""
        Compute the Weyl vector in the Borcherds lift.
        """
        q, = PowerSeriesRing(QQ, 'q').gens()
        w = self.weilrep()
        ds_dict = w.ds_dict()
        ds = w.ds()
        coeffs = self.coefficients()
        prec = -self.valuation()
        S = self.gram_matrix()
        S_inv = self.inverse_gram_matrix()
        F = self.fourier_expansion()
        nrows = S.nrows()
        try:
            weight = coeffs[tuple([0] * (nrows + 1))]
        except KeyError:
            weight = 0
        theta = [O(q ** (prec + 1)) for _ in ds]
        theta[0] += 1
        if nrows > 1:
            _, _, weyl_vs_matrix = pari(S_inv).qfminim(prec + 1, flag = 2)
            weyl_vs_list = weyl_vs_matrix.sage().columns()
        else:
            weyl_vs_list = [vector([n]) for n in srange(1, isqrt(2*prec*S[0, 0]) + 1)]
        coeff_sum = 0
        vec_sum = vector([0] * nrows)
        for v in weyl_vs_list:
            g = S_inv * v
            v_norm = g * v / 2
            g_frac = [frac(x) for x in g]
            i = ds_dict[tuple(g_frac)]
            big_tuple = tuple(g_frac + [-v_norm])
            j = next(j for j, w in enumerate(v) if w)
            theta[i] += 2 * q ** floor(v_norm)
            try:
                coeff = coeffs[big_tuple]
                coeff_sum += coeff
                if v[j] > 0:
                    vec_sum += g * coeff
                else:
                    vec_sum -= g * coeff
            except KeyError:
                pass
        e2 = eisenstein_series_qexp(2, prec + 1)
        return vector(QQ, [coeff_sum/12 + weight/24] + list(vec_sum / 2) + [-(e2 * sum([theta[i] * F[i][2] for i in range(len(F))]))[0]])

    def borcherds_lift(self, prec = None):
        r"""
        Compute the Borcherds lift.

        If ``self`` is a nearly-holomorphic modular form of weight -rank(L) / 2 (where L is the underlying positive-definite lattice) then one can associate to it a Borcherds product on the Type IV domain attached to L + II_{2, 2}. The result is a true modular form if all Fourier coefficients in self's principal part are integers and if sufficiently many of them are positive. (For the precise statement see Theorem 13.3 of [B].)

        NOTE: we do not check whether the input actually yields a holomorphic product. We do check that the input is of the correct weight, and you can expect a ValueError if the input has nonintegral coefficients.

        INPUT:
        - ``prec`` -- precision (optional). The precision of the output is limited by the precision of the input. However if ``prec`` is given then the output precision will not exceed ``prec``.

        OUTPUT: OrthogonalModularForm of weight equal to (1/2) of self's constant term.

        EXAMPLES::

            sage: from weilrep import *
            sage: ParamodularForms(5).borcherds_input_basis(1/4, 5)[0].borcherds_lift()
            (r^(-3/2) - r^(-1/2) - r^(1/2) + r^(3/2))*q^(1/2)*s^(1/2) + (-r^(-7/2) - 6*r^(-3/2) + 7*r^(-1/2) + 7*r^(1/2) - 6*r^(3/2) - r^(7/2))*q^(3/2)*s^(1/2) + (-r^(-7/2) - 6*r^(-3/2) + 7*r^(-1/2) + 7*r^(1/2) - 6*r^(3/2) - r^(7/2))*q^(1/2)*s^(3/2) + (r^(-9/2) + 6*r^(-7/2) + 8*r^(-3/2) - 15*r^(-1/2) - 15*r^(1/2) + 8*r^(3/2) + 6*r^(7/2) + r^(9/2))*q^(5/2)*s^(1/2) + (-r^(-13/2) - 7*r^(-11/2) + 42*r^(-9/2) - 15*r^(-7/2) - 60*r^(-3/2) + 41*r^(-1/2) + 41*r^(1/2) - 60*r^(3/2) - 15*r^(7/2) + 42*r^(9/2) - 7*r^(11/2) - r^(13/2))*q^(3/2)*s^(3/2) + (r^(-9/2) + 6*r^(-7/2) + 8*r^(-3/2) - 15*r^(-1/2) - 15*r^(1/2) + 8*r^(3/2) + 6*r^(7/2) + r^(9/2))*q^(1/2)*s^(5/2) + (r^(-11/2) - 7*r^(-9/2) - 8*r^(-7/2) + 15*r^(-3/2) - r^(-1/2) - r^(1/2) + 15*r^(3/2) - 8*r^(7/2) - 7*r^(9/2) + r^(11/2))*q^(7/2)*s^(1/2) + (r^(-17/2) - 15*r^(-13/2) - 41*r^(-11/2) + 36*r^(-9/2) - 31*r^(-7/2) + 72*r^(-3/2) - 22*r^(-1/2) - 22*r^(1/2) + 72*r^(3/2) - 31*r^(7/2) + 36*r^(9/2) - 41*r^(11/2) - 15*r^(13/2) + r^(17/2))*q^(5/2)*s^(3/2) + (r^(-17/2) - 15*r^(-13/2) - 41*r^(-11/2) + 36*r^(-9/2) - 31*r^(-7/2) + 72*r^(-3/2) - 22*r^(-1/2) - 22*r^(1/2) + 72*r^(3/2) - 31*r^(7/2) + 36*r^(9/2) - 41*r^(11/2) - 15*r^(13/2) + r^(17/2))*q^(3/2)*s^(5/2) + (r^(-11/2) - 7*r^(-9/2) - 8*r^(-7/2) + 15*r^(-3/2) - r^(-1/2) - r^(1/2) + 15*r^(3/2) - 8*r^(7/2) - 7*r^(9/2) + r^(11/2))*q^(1/2)*s^(7/2) + O(q, s)^5
        """
        prec0 = self.precision()
        val = self.valuation()
        prec0val = prec0 - val
        if prec is None:
            prec = isqrt(4 * (prec0+val))
        else:
            prec = min(prec, isqrt(4 * (prec0+val)))
        S_inv = self.inverse_gram_matrix()
        S = self.gram_matrix()
        det_S = S.determinant()
        nrows = Integer(S.nrows())
        if not self.weight() == -nrows/2:
            raise ValueError('Incorrect input weight')
        w = self.weyl_vector()
        w = vector([w[0]] + list(S * w[1:-1]) + [w[-1]])
        d = ZZ(denominator(w))
        weyl_v = d * w
        prec *= d
        coeffs = self.coefficients()
        weight = coeffs[tuple([0]*(nrows + 1))] / 2
        rb = LaurentPolynomialRing(QQ, list(var('r_%d' % i) for i in range(nrows)) )
        rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        r, t = PowerSeriesRing(rb_x, 't', prec).objgen()
        rpoly, t0 = PolynomialRing(QQ, 't0').objgen()
        ds_dict = self.weilrep().ds_dict()
        if nrows > 1:
            _, _, vs_matrix = pari(S_inv).qfminim(prec0val + prec0val + 1, flag = 2)
            vs_list = vs_matrix.sage().columns()
        else:
            vs_list = [vector([n]) for n in srange(1, isqrt(2*prec0*S[0, 0]) + 1)]
        vs_list.append(vector([0]*nrows))
        F = self.fourier_expansion()
        h = O(t ** prec)
        log_f = h
        a_plus_c = -1
        excluded_vectors = set()
        corrector = 1
        if nrows == 1:
            rb_zero = rb.gens()[0]
        for v in vs_list:
            g = S_inv * v
            v_norm = g * v / 2
            g_frac = [frac(y) for y in g]
            a_plus_c = -1
            while a_plus_c <= prec / d:
                a_plus_c += 1
                for c in srange(val, a_plus_c + 1):
                    a = a_plus_c - c
                    a_times_c = a * c
                    if val <= a_times_c - v_norm < prec0:
                        big_v = vector([a] + list(v) + [c])
                        if tuple(big_v) not in excluded_vectors:
                            if v and not (a or c):
                                j = next(j for j, v_j in enumerate(v) if v_j)
                                if v[j] > 0:
                                    v = -v
                            n = a_times_c - v_norm
                            big_tuple = tuple(g_frac + [n])
                            try:
                                exponent = coeffs[big_tuple]
                                if exponent:
                                    if nrows > 1:
                                        m = rb.monomial(*d*v)
                                    else:
                                        m = rb_zero ** (d * v[0])
                                    if (a or c) and c >= 0:
                                        u = t ** (d * a_plus_c) * x **( d * (c - a))
                                        if v:
                                            log_f += exponent * log(1 - u * (m + ~m - u) + h)
                                        else:
                                            log_f += exponent * log(1 - u + h)
                                    elif n:
                                        p = rpoly(1)
                                        bound = isqrt(val / n) + 1
                                        for k in range(1, bound):
                                            try:
                                                exponent_k = coeffs[tuple([frac(y) for y in k * g] + [n * (k * k)])]
                                                p *= (1 - t0 ** k) ** exponent_k
                                                excluded_vectors.add(tuple(k * big_v))
                                            except KeyError:
                                                break
                                        if c >= 0:
                                            corrector *= p.subs({t0 : m})
                                        else:
                                            deg_p = p.degree()
                                            if v:
                                                s1, s2 = 0, 0
                                                for j, p_j in enumerate(list(p)):
                                                    f = p_j * t ** (d * (a_plus_c * j - c * deg_p)) * x ** (d * (a * j + c * (deg_p - j)))
                                                    s1 += f * m ** (d * j)
                                                    s2 += f * (~m) ** (d * j)
                                                corrector *= (h + s1 * s2)
                                                weyl_v[0] += 2 * c * d * deg_p
                                            else:
                                                corrector *= (h + sum([p * t ** (d * (a_plus_c * j - c * deg_p)) * x ** (d * (a * j + c * (deg_p - j))) for j, p in enumerate(list(p))]))
                                                weyl_v[0] += c * d * deg_p
                            except KeyError:
                                if n > prec:
                                    prec = d * a_plus_c
                                    h = O(t ** prec)
                                    log_f += h
                                pass
        if nrows > 1:
            weyl_vector_term = (t ** (weyl_v[0] + weyl_v[-1])) * (x ** (weyl_v[0] - weyl_v[-1])) * rb.monomial(*weyl_v[1:-1])
        else:
            weyl_vector_term = (t ** (weyl_v[0] + weyl_v[-1])) * (x ** (weyl_v[0] - weyl_v[-1])) * rb_zero ** weyl_v[1]
        try:
            return OrthogonalModularForm(weight, S, exp(log_f) * r(corrector) * weyl_vector_term, scale = d, weylvec = weyl_v / d)
        except TypeError:
            raise RuntimeError('I caught a TypeError. This probably means you are trying to compute a Borcherds product that is not holomorphic.')

def jacobian(X):
    r"""
    Compute the Jacobian (Rankin--Cohen--Ibukiyama) operator.

    INPUT:
    - ``X`` -- a list [F_1, ..., F_N] of N orthogonal modular forms for the same Gram matrix, where N = 3 + (number of self's gram matrix rows)

    OUTPUT: OrthogonalModularForm. (If F_1, ..., F_N have weights k_1, ..., k_N then the result has weight k_1 + ... + k_N + N - 1.)

    EXAMPLES::

        sage: from weilrep import *
        sage: jacobian([ParamodularForms(1).eisenstein_series(k, 7) for k in [4, 6, 10, 12]]) / (-589927441461779261030400000/2354734631251) #funny multiple
        (r^-1 - r)*q^3*s^2 + (-r^-1 + r)*q^2*s^3 + (-r^-3 - 69*r^-1 + 69*r + r^3)*q^4*s^2 + (r^-3 + 69*r^-1 - 69*r - r^3)*q^2*s^4 + O(q, s)^7
    """
    N = len(X)
    Xref = X[0]
    nvars = Xref.nvars()
    f = Xref.true_fourier_expansion()
    t, = f.parent().gens()
    rb_x = f.base_ring()
    x, = rb_x.gens()
    r_list = rb_x.base_ring().gens()
    if not N == nvars + 1:
        raise ValueError('The Jacobian requires %d modular forms.'%(nvars + 1))
    k = N - 1
    v = vector([0] * (N - 1))
    r_deriv = [[] for _ in r_list]
    t_deriv = []
    x_deriv = []
    u = []
    for y in X:
        f = y.true_fourier_expansion()
        t_deriv.append(t * f.derivative())
        if nvars > 1:
            x_deriv.append(f.map_coefficients(lambda a: x * a.derivative()))
            if nvars > 2:
                for i, r in enumerate(r_list):
                    r_deriv[i].append(f.map_coefficients(lambda a: rb_x([r * y.derivative(r) for y in list(a)]) * (x ** (a.polynomial_construction()[1]) )))
        y_k = y.weight()
        k += y_k
        v += y.weyl_vector()
        u.append(y_k * f)
    S = Xref.gram_matrix()
    L = [u, t_deriv]
    if nvars > 1:
        L.append(x_deriv)
        if nvars > 2:
            L.extend(r_deriv)
    if nvars > S.nrows():
        return OrthogonalModularForm(k, S, matrix(L).determinant(), scale = 1, weylvec = v)
    from .lorentz import OrthogonalModularFormLorentzian
    return OrthogonalModularFormLorentzian(k, S, matrix(L).determinant(), scale = 1, weylvec = v, qexp_representation = Xref.qexp_representation())

def omf_matrix(X):
    r"""
    Convert the Fourier coefficients of a list of orthogonal modular forms X into a matrix.
    """
    if not X:
        return matrix([])
    nrows = X[0].nvars()
    prec = min(x.precision() for x in X)
    Xcoeffs = [x.coefficients(prec = prec) for x in X]
    Xitems = [set(xcoeffs.keys()) for xcoeffs in Xcoeffs]
    Xitems = list(Xitems[0].union(*Xitems[1:]))
    lenXitems = len(Xitems)
    L = [[0]*lenXitems for _ in X]
    M = []
    for i, x in enumerate(X):
        L = [0]*lenXitems
        for j, g in enumerate(Xitems):
            try:
                L[j] = Xcoeffs[i][g]
            except KeyError:
                pass
        M.append(L)
    return matrix(M)

def omf_rank(X):
    r"""
    Compute the rank of the space spanned by the list of orthogonal modular forms X.
    """
    return omf_matrix(X).rank()

def omf_relations(X):
    r"""
    Compute all linear relations among the list of orthogonal modular forms X.
    """
    return omf_matrix(X).kernel()