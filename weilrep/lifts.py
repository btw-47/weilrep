r"""

Additive and multiplicative theta lifts

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

from copy import copy

import cypari2

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli, GCD
from sage.arith.srange import srange
from sage.calculus.var import var
from sage.functions.other import ceil, factorial, frac, sqrt
from sage.geometry.cone import Cone
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import matrix
from sage.misc.functional import denominator, isqrt
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector
from sage.rings.fraction_field import FractionField
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.number_field.number_field_base import NumberField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing, LaurentPolynomialRing_generic
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RealField_class

from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis


try:
    from sage.rings.complex_mpfr import ComplexField_class
except ModuleNotFoundError:
    from sage.rings.complex_field import ComplexField_class

pari = cypari2.Pari()
PariError = cypari2.PariError


class OrthogonalModularForms(object):
    r"""
    This class represents spaces of modular forms on the type IV domain attached to a lattice of the form L + II_{2,2}, where L is positive-definite.

    INPUT: an OrthogonalModularForms instance is constructed by calling ``OrthogonalModularForms(S)``, where:
    - ``S`` -- a Gram matrix for the lattice; or
    - ``S`` -- a quadratic form; or
    - ``S`` -- the WeilRep instance WeilRep(S)

    NOTE: this takes positive-definite lattices and appends two unimodular hyperbolic planes to them to get lattices of signature (n, 2). Lattices that are not split by two unimodular hyperbolic planes will be bumped to the class OrthogonalModularFormsLorentzian from the .lorentz.py file.
    """

    def __init__(self, w, **kwargs):
        try:
            S = w.gram_matrix()
        except AttributeError:
            w = WeilRep(w)
            S = w.gram_matrix()
        if w.is_lorentzian() or w.is_lorentzian_plus_II():
            from .lorentz import OrthogonalModularFormsLorentzian
            self.__class__ = OrthogonalModularFormsLorentzian
        elif w.is_positive_definite():
            from .positive_definite import OrthogonalModularFormsPositiveDefinite
            self.__class__ = OrthogonalModularFormsPositiveDefinite
        else:
            raise ValueError('Invalid signature in OrthogonalModularForms')
        unitary = kwargs.pop('unitary', 0)
        if unitary:
            self.__weilrep = w
        else:
            try:
                self.__weilrep = w.trace_form()
            except AttributeError:
                self.__weilrep = w
        self.__gram_matrix = S

    def __repr__(self):
        w = self.weilrep()
        s = 'Orthogonal modular forms associated to the Gram matrix\n%s' % self.__gram_matrix
        if w.is_positive_definite():
            return s + ' + 2U'
        elif w.is_lorentzian():
            return s + ' + U'
        return s

    def __add__(self, other):
        from .lorentz import RescaledHyperbolicPlane
        w = self.weilrep()
        if isinstance(other, RescaledHyperbolicPlane):
            return OrthogonalModularForms(w + other)
        return NotImplemented
    __radd__ = __add__

    def gram_matrix(self):
        r"""
        Return self's Gram matrix.
        """
        return self.__gram_matrix

    def nrows(self):
        r"""
        Return the diimension of self's Gram matrix.
        """
        return Integer(self.gram_matrix().nrows())

    def weilrep(self):
        r"""
        Return self's Weil representation.
        """
        return self.__weilrep

    # ## constructors of modular forms
    def eisenstein_series(self, k, prec, allow_small_weight=False):
        r"""
        Compute the Eisenstein series. (i.e. the theta lift of a vector-valued Eisenstein series)

        INPUT:
        - ``k`` -- the weight (an even integer)
        - ``prec`` -- the output precision

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: OrthogonalModularForms(diagonal_matrix([-2, 2, 2])).eisenstein_series(4, 5)
            1 + 480*t + (240*x^-2 + (2880*r_0^-1 + 7680 + 2880*r_0)*x^-1 + (240*r_0^-2 + 7680*r_0^-1 + 18720 + 7680*r_0 + 240*r_0^2) + (2880*r_0^-1 + 7680 + 2880*r_0)*x + 240*x^2)*t^2 + ((480*r_0^-2 + 15360*r_0^-1 + 28800 + 15360*r_0 + 480*r_0^2)*x^-2 + (15360*r_0^-2 + 76800*r_0^-1 + 92160 + 76800*r_0 + 15360*r_0^2)*x^-1 + (28800*r_0^-2 + 92160*r_0^-1 + 134400 + 92160*r_0 + 28800*r_0^2) + (15360*r_0^-2 + 76800*r_0^-1 + 92160 + 76800*r_0 + 15360*r_0^2)*x + (480*r_0^-2 + 15360*r_0^-1 + 28800 + 15360*r_0 + 480*r_0^2)*x^2)*t^3 + (2160*x^-4 + (7680*r_0^-2 + 44160*r_0^-1 + 61440 + 44160*r_0 + 7680*r_0^2)*x^-3 + (7680*r_0^-3 + 112320*r_0^-2 + 207360*r_0^-1 + 312960 + 207360*r_0 + 112320*r_0^2 + 7680*r_0^3)*x^-2 + (44160*r_0^-3 + 207360*r_0^-2 + 380160*r_0^-1 + 430080 + 380160*r_0 + 207360*r_0^2 + 44160*r_0^3)*x^-1 + (2160*r_0^-4 + 61440*r_0^-3 + 312960*r_0^-2 + 430080*r_0^-1 + 656160 + 430080*r_0 + 312960*r_0^2 + 61440*r_0^3 + 2160*r_0^4) + (44160*r_0^-3 + 207360*r_0^-2 + 380160*r_0^-1 + 430080 + 380160*r_0 + 207360*r_0^2 + 44160*r_0^3)*x + (7680*r_0^-3 + 112320*r_0^-2 + 207360*r_0^-1 + 312960 + 207360*r_0 + 112320*r_0^2 + 7680*r_0^3)*x^2 + (7680*r_0^-2 + 44160*r_0^-1 + 61440 + 44160*r_0 + 7680*r_0^2)*x^3 + 2160*x^4)*t^4 + O(t^5)
        """
        w = self.__weilrep
        try:
            return (-((k + k) / bernoulli(k)) * w.eisenstein_series(k + self.input_wt(), ceil(prec * prec / 4) + 1, allow_small_weight=allow_small_weight)).theta_lift(prec)
        except (TypeError, ValueError, ZeroDivisionError):
            raise ValueError('Invalid weight') from None

    def lifts_basis(self, k, prec, cusp_forms=True):
        r"""
        Compute a basis of the Maass Spezialschar of weight ``k`` up to precision ``prec``.

        This computes the theta lifts of a basis of cusp forms (or a basis of modular forms, if ``cusp_forms`` is set to False).

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- the precision of the output
        - ``cusp_forms`` -- boolean (default True). If True then we output only cusp forms.

        OUTPUT: list of OrthogonalModularForm's

        EXAMPLES::

            sage: from weilrep import *
            sage: ParamodularForms(N = 1).spezialschar(10, 5)
            [(r^-1 - 2 + r)*q*s + (-2*r^-2 - 16*r^-1 + 36 - 16*r - 2*r^2)*q^2*s + (-2*r^-2 - 16*r^-1 + 36 - 16*r - 2*r^2)*q*s^2 + (r^-3 + 36*r^-2 + 99*r^-1 - 272 + 99*r + 36*r^2 + r^3)*q^3*s + (-16*r^-3 + 240*r^-2 - 240*r^-1 + 32 - 240*r + 240*r^2 - 16*r^3)*q^2*s^2 + (r^-3 + 36*r^-2 + 99*r^-1 - 272 + 99*r + 36*r^2 + r^3)*q*s^3 + O(q, s)^5]
        """
        S = self.gram_matrix()
        w = self.weilrep()
        if cusp_forms:
            X = w.cusp_forms_basis(k + self.input_wt(), ceil(prec * prec / 4) + 1)
        else:
            X = w.modular_forms_basis(k + self.input_wt(), ceil(prec * prec / 4) + 1)
        try:
            return X[0].theta_lift(prec, _L=X)
        except TypeError:
            return [x.theta_lift(prec) for x in X]
        except IndexError:
            return []

    def spezialschar(self, *args, **kwargs):
        if args:
            return self.lifts_basis(*args, **kwargs)
        return Spezialschar(self)
    maass_space = spezialschar

    # ## methods for Borcherds products

    def _borcherds_product_polyhedron(self, pole_order, prec, verbose=False):
        r"""
        Construct a polyhedron representing a cone of Heegner divisors.

        For internal use in the methods borcherds_input_basis() and borcherds_input_Qbasis().

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
        wt = self.input_wt()
        w = self.weilrep()
        rds = w.rds()
        norm_dict = w.norm_dict()
        X = w.nearly_holomorphic_modular_forms_basis(wt, pole_order, prec, verbose=verbose)
        N = len([g for g in rds if not norm_dict[tuple(g)]])
        v_list = w.coefficient_vector_exponents(0, 1, starting_from=-pole_order, include_vectors=True)
        exp_list = [v[1] for v in v_list]
        v_list = [vector(v[0]) for v in v_list]
        positive = [None] * len(exp_list)
        zero = vector([0] * (len(exp_list) + 1))
        M = matrix([x.coefficient_vector(starting_from=-pole_order, ending_with=0)[:-N] for x in X])
        vs = M.transpose().kernel().basis()
        for i, n in enumerate(exp_list):
            ieq = copy(zero)
            ieq[i + 1] = 1
            for j, m in enumerate(exp_list[:i]):
                N = sqrt(m / n)
                if N in ZZ:
                    v1 = v_list[i]
                    v2 = v_list[j]
                    ieq[j + 1] = denominator(v1 * N - v2) == 1 or denominator(v1 * N + v2) == 1
            positive[i] = ieq
        p = Polyhedron(ieqs=positive, eqns=[vector([0] + list(v)) for v in vs])
        return M, p, X

    def borcherds_input_basis(self, pole_order, prec, verbose=False):
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
        wt = self.input_wt()
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose=verbose)
        if verbose:
            print('I will now try to find a Hilbert basis.')
        try:
            b = matrix(Cone(p).Hilbert_basis())
            try:
                u = M.solve_left(b)
                Y = [v * X for v in u.rows()]
                Y.sort(key=lambda x: x.fourier_expansion()[0][2][0])
                Y = WeilRepModularFormsBasis(wt, Y, w)
            except ValueError:
                Y = WeilRepModularFormsBasis(wt, [], w)
        except IndexError:
            Y = WeilRepModularFormsBasis(wt, [], w)
        if wt >= 0:
            X = w.basis_vanishing_to_order(wt, max(0, -pole_order), prec)
            X = WeilRepModularFormsBasis(wt, list(X), w)
            if X:
                X.extend(Y)
                return X
        return Y

    def borcherds_input_Qbasis(self, pole_order, prec, verbose=False):
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
        wt = self.input_wt()
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose=verbose)
        try:
            b = matrix(Cone(p).rays())
            if verbose:
                print('I will now try to find Borcherds product inputs.')
            try:
                u = M.solve_left(b)
                Y = [v * X for v in u.rows()]
                Y.sort(key=lambda x: x.fourier_expansion()[0][2][0])
                Y = WeilRepModularFormsBasis(wt, Y, w)
            except ValueError:
                Y = WeilRepModularFormsBasis(wt, [], w)
        except IndexError:
            Y = WeilRepModularFormsBasis(wt, [], w)
        if wt >= 0:
            X = w.basis_vanishing_to_order(wt, max(0, -pole_order), prec)
            X = WeilRepModularFormsBasis(wt, list(X), w)
            if X:
                X.extend(Y)
                return X
        return Y


class Spezialschar(object):
    r"""
    The Spezialschar of theta lifts.
    """

    def __init__(self, m):
        self.__m = m

    def __repr__(self):
        return 'Spezialschar in ' + str(self.__m)

    def basis(self, k, prec):
        return self.__m.lifts_basis(k, prec, cusp_forms=False)

    modular_forms_basis = basis

    def cusp_forms_basis(self, k, prec):
        return self.__m.lifts_basis(k, prec)

    def __contains__(self, f):
        if f.weilrep() != self.__m.weilrep():
            return False
        return f.is_lift()


class OrthogonalModularForm(object):
    r"""
    This class represents modular forms on type IV domains.

    INPUT: Orthogonal modular forms are constructed by calling ``OrthogonalModularForm(k, w, f, scale, weylvec)``, where:
    - ``k`` -- the weight
    - ``w`` -- WeilRep of the underlying lattice.
    - ``f`` -- the Fourier expansion. This is a power series in the variable t over a Laurent polynomial ring in the variable x over a base ring of Laurent polynomials in the variables r_0,...,r_d
    - ``scale`` -- a natural number. If scale != 1 then all exponents in the Fourier expansion should be divided by `scale`. (This is done in the __repr__ method)
    - ``weylvec`` -- a ``Weyl vector``. For modular forms that are not constructed as Borcherds products this vector should be zero.
    """
    def __init__(self, k, w, f, scale, weylvec, pullback_function=NotImplemented, qexp_representation=None, ppcoeffs=None):
        s = qexp_representation
        self.__fourier_expansion = f
        self.__qexp_representation = s
        self.__scale = scale
        self.__valuation = f.valuation()
        self.__weight = k
        self.__weilrep = w
        self.__weylvec = weylvec
        self.__pullback = pullback_function
        if ppcoeffs:
            self.__ppcoeffs = ppcoeffs
        # change class
        if w.is_lorentzian() or w.is_lorentzian_plus_II():
            from .lorentz import OrthogonalModularFormLorentzian
            self.__class__ = OrthogonalModularFormLorentzian
        elif w.is_positive_definite():
            from .positive_definite import OrthogonalModularFormPositiveDefinite
            self.__class__ = OrthogonalModularFormPositiveDefinite
            if s is None:
                self.__qexp_representation = 'PD+II'
        if s:
            if len(s) == 3:
                if s[0] == 'hermite':
                    from .hermitian import HermitianModularForm
                    self.__class__ = HermitianModularForm
                    HermitianModularForm.__init__(self, s[1], s[2])
                elif s[0] == 'hilbert':
                    from .hilbert import HilbertModularForm
                    self.__class__ = HilbertModularForm
                    HilbertModularForm.__init__(self, s[1], s[2])
            elif s == 'siegel':
                from .paramodular import ParamodularForm
                self.__class__ = ParamodularForm
            elif s == 'unitary':
                pass
                # from .unitary import UnitaryModularForm
                # self.__class__ = UnitaryModularForm

    # ## basic attributes

    def base_ring(self):
        r"""
        Returns self's base ring.
        """
        f = self.__fourier_expansion
        r = f.base_ring()
        try:
            return r.base_ring()
        except AttributeError:
            return r

    def _base_ring_is_laurent_polynomial_ring(self):
        try:
            return self.__brilpr
        except AttributeError:
            r = self.base_ring()
            self.__brilpr = isinstance(r, LaurentPolynomialRing_generic) or isinstance(r, NumberField) or isinstance(r, ComplexField_class) or isinstance(r, RealField_class)  # are we missing anything?
            return self.__brilpr

    def __bool__(self):
        return bool(self.true_fourier_expansion())

    def divisor(self):
        r"""
        Compute the divisor of a Borcherds product.

        The output is a function that accepts positive-norm vectors 'v' in the dual lattice L' as input, and outputs an integer (the order of vanishing of self on the rational-quadratic divisor v^{\perp}).
        """
        try:
            d = self._principal_part_coefficients()
            S = self.gram_matrix()
            m = QQ(-min(x[-1] for x in d.keys()))

            def a(v):
                Sv = S * v
                n = GCD(Sv)
                v, Sv = v / n, Sv / n
                x = v * Sv / 2
                if x <= 0:
                    raise ValueError('Not a positive norm vector.')
                return sum(d[tuple(list(map(frac, v * b)) + [-x * (b * b)])] for b in srange(1, isqrt(m / x) + 1))
            return a
        except AttributeError:
            raise NotImplementedError from None

    def gram_matrix(self):
        r"""
        Return our Gram matrix.
        """
        return self.weilrep().gram_matrix()

    def inverse(self):
        f = self.true_fourier_expansion()
        try:
            g = ~f
        except ValueError:
            g = ~self._laurent_to_fraction().true_fourier_expansion()
        return OrthogonalModularForm(-self.__weight, self.__weilrep, g, self.__scale, -self.__weylvec, qexp_representation=self.__qexp_representation)
    __invert__ = inverse

    def _laurent_to_fraction(self):
        r"""
        Convert the Laurent polynomials in 'r_0,...,r_n' to quotients.
        """
        r = self.base_ring()
        if self._base_ring_is_laurent_polynomial_ring():
            s = FractionField(PolynomialRing(QQ, list(var('r_%d' % i) for i in range(r.ngens()))))
            rx = LaurentPolynomialRing(s, 'x')
            rx.inject_variables(verbose=False)
            rt = PowerSeriesRing(rx, 't')
            f = self.true_fourier_expansion()
            f = rt([x.map_coefficients(lambda y: y, s) for x in f.padded_list()]).add_bigoh(f.prec())
            try:
                ppcoeffs = self.__ppcoeffs
            except AttributeError:
                ppcoeffs = None
            return OrthogonalModularForm(self.__weight, self.__weilrep, f, self.__scale, self.__weylvec, qexp_representation=self.__qexp_representation, ppcoeffs=ppcoeffs)
        return self

    def precision(self):
        r"""
        Return our precision.
        """
        return Integer(self.true_fourier_expansion().prec()) / self.scale()
    prec = precision

    def _principal_part_coefficients(self):
        return self.__ppcoeffs

    def pullback(self, *args, **kwargs):
        return self.__pullback(*args, **kwargs)

    def qexp_representation(self):
        return self.__qexp_representation

    def reduce_precision(self, new_prec):
        r"""
        Reduce precision to 'new_prec'.
        """
        try:
            ppcoeffs = self.__ppcoeffs
        except AttributeError:
            ppcoeffs = None
        return OrthogonalModularForm(self.__weight, self.__weilrep, self.true_fourier_expansion().add_bigoh(new_prec), self.__scale, self.__weylvec, qexp_representation=self.__qexp_representation, ppcoeffs=ppcoeffs)

    def rescale(self, d):
        r"""
        Rescale self by "d". This should not change the output when self is printed.
        """
        nrows = self.nvars()
        w = self.weilrep()
        f = self.true_fourier_expansion()
        if nrows > 1:
            rb_x = f.base_ring()
            rb_x.inject_variables(verbose=False)
            x = rb_x.gens()[0]
            if nrows > 2:
                rbgens = rb_x.base_ring().gens()
                rescale_dict = {a: a ** d for a in rbgens}
                return OrthogonalModularForm(self.weight(), w, (f.map_coefficients(lambda y: (x ** (d * y.polynomial_construction()[1])) * rb_x([y.polynomial_construction()[0][i // d].subs(rescale_dict) if i % d == 0 else 0 for i in range(d * len(list(y)))]))).V(d), scale=self.scale() * d, weylvec=self.weyl_vector(), qexp_representation=self.qexp_representation())
            return OrthogonalModularForm(self.weight(), w, (f.map_coefficients(lambda p: p.subs({x: x ** d}))).V(d), scale=self.scale() * d, weylvec=self.weyl_vector(), qexp_representation=self.qexp_representation())
        return OrthogonalModularForm(self.weight(), w, f.V(d), scale=self.scale() * d, weylvec=self.weyl_vector(), qexp_representation=self.qexp_representation())

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

    def weilrep(self):
        r"""
        Return self's WeilRep.
        """
        return self.__weilrep

    def weyl_vector(self):
        r"""
        Return self's Weyl vector.
        """
        return self.__weylvec

    # ## methods to extract Fourier coefficients

    def coefficients(self, prec=+Infinity):
        r"""
        Return a dictionary of self's known Fourier coefficients.

        The input into the dictionary should be a tuple of the form (a, b, c_0, ..., c_n). The output will then be the Fourier coefficient of the monomial t^a x^b r_0^(c_0)...r_n^(c_n).
        """
        L = {}
        d = self.scale()
        nrows = self.nvars()
        f = self.true_fourier_expansion()
        d_prec = d * prec
        if d_prec < Infinity:
            constant_prec = max(10, d_prec)
        else:
            constant_prec = 10
        try:
            D = f.dict()
        except AttributeError:
            v = f.valuation()
            D = {(x + v): y for x, y in f.valuation_zero_part().dict().items()}
        for j_t, p in D.items():
            if j_t < d_prec:
                j_t = Integer(j_t)
                if nrows > 1:
                    try:
                        p_dict = p.dict()
                    except AttributeError:
                        p = LaurentSeriesRing(p.base_ring(), 'x')(p)
                        v = p.valuation()
                        p_dict = {(x + v): y for x, y in p.valuation_zero_part().dict().items()}
                    for j_x, h in p_dict.items():
                        j_x = Integer(j_x)
                        if nrows > 2:
                            if nrows > 3:
                                try:
                                    for j_r, y in h.dict().items():
                                        g = tuple([j_t / d, j_x / d] + list(vector(ZZ, j_r) / d))
                                        L[g] = y
                                except AttributeError:
                                    hn, hd = h.numerator(), h.denominator()
                                    r = hd.parent()
                                    if hd.constant_coefficient():
                                        s = PowerSeriesRing(r.base_ring(), list(var('r_%d' % i) for i in range(nrows - 2)), default_prec=constant_prec)
                                        h = s(h).add_bigoh(constant_prec)
                                    else:
                                        s = LaurentPolynomialRing(r.base_ring(), list(var('r_%d' % i) for i in range(nrows - 2)))
                                        h = s(hn) / s(hd)
                                    for j_r, y in h.dict().items():
                                        g = tuple([j_t / d, j_x / d] + list(vector(ZZ, j_r) / d))
                                        L[g] = y
                            else:
                                try:
                                    for j_r, y in h.dict().items():
                                        g = tuple([j_t / d, j_x / d, Integer(j_r) / d])
                                        L[g] = y
                                except AttributeError:
                                    hn, hd = h.numerator(), h.denominator()
                                    r = hd.parent()
                                    if hd.constant_coefficient():
                                        s = PowerSeriesRing(r.base_ring(), 'r_0', default_prec=constant_prec)
                                        h = s(h).add_bigoh(constant_prec)
                                        try:
                                            for j_r, y in h.dict().items():
                                                g = tuple([j_t / d, j_x / d, Integer(j_r) / d])
                                                L[g] = y
                                        except AttributeError:
                                            pass
                                    else:
                                        s = LaurentSeriesRing(r.base_ring(), 'r_0')
                                        h = s(hn) * s(1 / hd)
                                        hval = h.valuation()
                                        h = h.valuation_zero_part()
                                        try:
                                            for j_r, y in h.dict().items():
                                                g = tuple([j_t / d, j_x / d, (Integer(j_r) + hval) / d])
                                                L[g] = y
                                        except AttributeError:
                                            pass
                        else:
                            g = tuple([j_t / d, j_x / d])
                            L[g] = h
                else:
                    g = tuple([j_t / d])
                    L[g] = p
        return L

    true_coefficients = coefficients

    def qs_coefficients(self, prec=+Infinity):
        r"""
        Return a dictionary of self's known 'qs' Fourier coefficients.

        The input into the dictionary should be a tuple of the form (a, b_0, ..., b_d, c). The output will then be the Fourier coefficient of the monomial q^a r_0^(b_0)...r_d^(b_d) s^c.

        EXAMPLES::

            sage: from weilrep import *
            sage: f = ParamodularForms(4).borcherds_input_by_weight(1/2, 10)[0].borcherds_lift()
            sage: f.qs_coefficients()[(1/8, -1/2, 1/8)]
            -1
        """
        L = {}
        d = self.scale()
        nrows = self.nvars()
        w = self.weilrep()
        if w.is_positive_definite() or w._is_positive_definite_plus_II():
            nrows -= 2
        elif w._is_positive_definite_plus_2II():
            nrows -= 4
        elif not self.has_fourier_jacobi_representation():
            raise NotImplementedError
        f = self.fourier_expansion()
        coeffs = f.coefficients()
        q, s = f.parent().gens()
        d_prec = d * prec
        for j, x in coeffs.items():
            a, c = [Integer(i) for i in j.exponents()[0]]
            if a + c < d_prec:
                try:
                    x_coeffs = x.coefficients()
                except AttributeError:
                    xn, xd = x.numerator(), x.denominator()
                    r = xd.parent()
                    if xd.constant_coefficient():
                        s = PowerSeriesRing(r.base_ring(), list(var('r_%d' % i) for i in range(nrows)))
                        x = s(xn) / s(xd)
                    else:
                        s = LaurentPolynomialRing(r.base_ring(), list(var('r_%d' % i) for i in range(nrows)))
                        x = s(xn) / s(xd)
                    x_coeffs = x.coefficients()
                if nrows > 1:
                    for i, y in enumerate(x.exponents()):
                        g = tuple([a / d] + list(vector(ZZ, y) / d) + [c / d])
                        L[g] = x_coeffs[i]
                else:
                    for i, y in enumerate(x.exponents()):
                        g = a / d, Integer(y) / d, c / d
                        L[g] = x_coeffs[i]
        return L

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

    # ## arithmetic operations

    def __add__(self, other):
        r"""
        Add modular forms, rescaling if necessary.
        """
        if not other:
            return self
        if self._base_ring_is_laurent_polynomial_ring() + other._base_ring_is_laurent_polynomial_ring() == 1:
            return self._laurent_to_fraction() + other._laurent_to_fraction()
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

        def p(*args, **kwargs):
            return self.pullback(*args, **kwargs) + other.pullback(*args, **kwargs)
        if not self_scale == other_scale:
            new_scale = lcm(self_scale, other_scale)
            X1 = self.rescale(new_scale // self_scale)
            X2 = other.rescale(new_scale // other_scale)
            return OrthogonalModularForm(self.__weight, self.__weilrep, X1.true_fourier_expansion() + X2.true_fourier_expansion(), scale=new_scale, weylvec=self_v, pullback_function=p, qexp_representation=self.qexp_representation())
        return OrthogonalModularForm(self.__weight, self.__weilrep, self.true_fourier_expansion() + other.true_fourier_expansion(), scale=self_scale, weylvec=self_v, pullback_function=p, qexp_representation=self.qexp_representation())

    __radd__ = __add__

    def __sub__(self, other):
        r"""
        Subtract modular forms, rescaling if necessary.
        """
        if not other:
            return self
        if self._base_ring_is_laurent_polynomial_ring() + other._base_ring_is_laurent_polynomial_ring() == 1:
            return self._laurent_to_fraction() - other._laurent_to_fraction()
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

        def p(*args, **kwargs):
            return self.pullback(*args, **kwargs) - other.pullback(*args, **kwargs)
        if not self_scale == other_scale:
            new_scale = lcm(self_scale, other_scale)
            X1 = self.rescale(new_scale // self_scale)
            X2 = other.rescale(new_scale // other_scale)
            return OrthogonalModularForm(self.__weight, self.__weilrep, X1.true_fourier_expansion() - X2.true_fourier_expansion(), scale=new_scale, weylvec=self_v, pullback_function=p, qexp_representation=self.qexp_representation())
        return OrthogonalModularForm(self.__weight, self.__weilrep, self.true_fourier_expansion() - other.true_fourier_expansion(), scale=self_scale, weylvec=self_v, pullback_function=p, qexp_representation=self.qexp_representation())

    def __neg__(self):
        def p(*args, **kwargs):
            return -self.pullback(*args, **kwargs)
        return OrthogonalModularForm(self.__weight, self.__weilrep, -self.true_fourier_expansion(), scale=self.scale(), weylvec=self.weyl_vector(), pullback_function=p, qexp_representation=self.qexp_representation())

    def __mul__(self, other):
        r"""
        Multiply modular forms, rescaling if necessary.
        """
        if isinstance(other, OrthogonalModularForm):
            if self._base_ring_is_laurent_polynomial_ring() + other._base_ring_is_laurent_polynomial_ring() == 1:
                return self._laurent_to_fraction() * other._laurent_to_fraction()
            if not self.gram_matrix() == other.gram_matrix():
                raise ValueError('Incompatible Gram matrices')
            self_scale = self.scale()
            other_scale = other.scale()
            if self_scale != 1 or other_scale != 1:
                new_scale = lcm(self.scale(), other.scale())
                X1 = self.rescale(new_scale // self_scale)
                X2 = other.rescale(new_scale // other_scale)
            else:
                new_scale = 1
                X1 = self
                X2 = other
            f1 = X1.true_fourier_expansion()
            f2 = X2.true_fourier_expansion()
            f = f1 * f2
            if f1.valuation() < 0 or f2.valuation() < 0 and f.valuation() >= 0:
                r = PowerSeriesRing(f.base_ring(), 't')
                f = r(f)

            def p(*args, **kwargs):
                return self.pullback(*args, **kwargs) * other.pullback(*args, **kwargs)
            return OrthogonalModularForm(self.__weight + other.weight(), self.__weilrep, f, scale=new_scale, weylvec=self.weyl_vector() + other.weyl_vector(), pullback_function=p, qexp_representation=self.qexp_representation())
        else:
            def p(*args, **kwargs):
                return other * self.pullback(*args, **kwargs)
            return OrthogonalModularForm(self.weight(), self.__weilrep, self.true_fourier_expansion() * other, scale=self.scale(), weylvec=self.weyl_vector(), pullback_function=p, qexp_representation=self.qexp_representation())

    __rmul__ = __mul__

    def __div__(self, other):
        r"""
        Divide modular forms, rescaling if necessary.
        """
        if isinstance(other, OrthogonalModularForm):
            return self.__mul__(other.inverse())
        else:
            def p(*args, **kwargs):
                return self.pullback(*args, **kwargs) / other
            return OrthogonalModularForm(self.weight(), self.__weilrep, self.true_fourier_expansion() / other, scale=self.scale(), weylvec=self.weyl_vector(), pullback_function=p, qexp_representation=self.qexp_representation())

    __truediv__ = __div__

    def __rdiv__(self, other):
        return self.inverse().__mul__(other)

    __rtruediv__ = __rdiv__

    def __eq__(self, other):
        if self._base_ring_is_laurent_polynomial_ring() + other._base_ring_is_laurent_polynomial_ring() == 1:
            return self._laurent_to_fraction() == other._laurent_to_fraction()
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
        if other not in ZZ:
            raise ValueError('Not a valid exponent')

        def p(*args, **kwargs):
            return self.pullback(*args, **kwargs) ** other
        return OrthogonalModularForm(other * self.weight(), self.__weilrep,
                                     self.true_fourier_expansion() ** other,
                                     scale=self.scale(),
                                     weylvec=other * self.weyl_vector(),
                                     pullback_function=p,
                                     qexp_representation=self.qexp_representation())

    def n(self):
        d = self.true_coefficients()
        f = self.true_fourier_expansion()
        t, = f.parent().gens()
        n = self.nvars()
        if n > 1:
            rb, x = f.base_ring().objgen()
            if n > 2:
                r = rb.base_ring()
                f = sum(a.n() * x**g[1] * r.monomial(*g[2:]) * t**g[0] for g, a in d.items()).add_bigoh(f.prec())
            else:
                f = sum(a.n() * x**g[1] * t**g[0] for g, a in d.items()).add_bigoh(f.prec())
        else:
            f = sum(a.n() * t**g[0] for g, a in d.items()).add_bigoh(f.prec())
        return OrthogonalModularForm(self.weight(), self.__weilrep, f, scale=self.scale(), weylvec=self.weyl_vector(), qexp_representation=self.qexp_representation())


def orthogonal_eisenstein_series(k, S, prec, w=None):
    r"""
    Computes the "orthogonal Eisenstein series" as the theta-lift of the vector-valued Eisenstein series E_{k, 0} (if it exists). We renormalize such that the constant term is 1.

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
    if w:
        return OrthogonalModularForms(w).eisenstein_series(k, prec)
    return OrthogonalModularForms(S).eisenstein_series(k, prec)


def jacobian(*X):
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
    from .unitary import unitary_jacobian, UnitaryModularForm
    N = len(X)
    if N == 1:
        X = X[0]
        N = len(X)
    if not all(x._base_ring_is_laurent_polynomial_ring() for x in X):
        X = [x._laurent_to_fraction() for x in X]
    Xref = X[0]
    if isinstance(Xref, UnitaryModularForm):
        return unitary_jacobian(X)
    nvars = Xref.nvars()
    f = Xref.true_fourier_expansion()
    t, = f.parent().gens()
    rb_x = f.base_ring()
    x, = rb_x.gens()
    r_list = rb_x.base_ring().gens()
    if N != nvars + 1:
        raise ValueError('The Jacobian requires %d modular forms.' % (nvars + 1))
    k = N - 1
    v = vector([0] * nvars)
    r_deriv = [[] for _ in r_list]
    t_deriv = []
    x_deriv = []
    u = []
    S = Xref.gram_matrix()
    new_scale = lcm(x.scale() for x in X)
    for y in X:
        if y.gram_matrix() != S:
            raise ValueError('These forms do not have the same Gram matrix.')
        f = y.rescale(new_scale // y.scale()).true_fourier_expansion()
        t_deriv.append(t * f.derivative())
        if nvars > 1:
            x_deriv.append(f.map_coefficients(lambda a: x * a.derivative()))
            if nvars > 2:
                for i, r in enumerate(r_list):
                    r_deriv[i].append(f.map_coefficients(lambda a: rb_x([r * y.derivative(r) for y in list(a)]) * (x ** (a.polynomial_construction()[1]))))
        y_k = y.weight()
        k += y_k
        v += y.weyl_vector()
        u.append(y_k * f)
    L = [u, t_deriv]
    if nvars > 1:
        L.append(x_deriv)
        if nvars > 2:
            L.extend(r_deriv)
    return OrthogonalModularForm(k, Xref.weilrep(), matrix(L).determinant(), scale=new_scale, weylvec=v, qexp_representation=Xref.qexp_representation())


def omf_matrix(*X):
    r"""
    Convert the Fourier coefficients of a list of orthogonal modular forms X into a matrix.

    Used in omf_pivots; omf_rank; omf_relations
    """
    if not X:
        return matrix([])
    N = len(X)
    if N == 1:
        try:
            Y = X[0]
            Xref = Y[0]
            X = Y
        except IndexError:
            Xref = X[0]
    else:
        Xref = X[0]
    k = Xref.weight()
    prec = min(x.precision() for x in X)
    Xcoeffs = [x.true_coefficients(prec=prec) for x in X]
    Xitems = [set(xcoeffs.keys()) for xcoeffs in Xcoeffs]
    Xitems = list(Xitems[0].union(*Xitems[1:]))
    lenXitems = len(Xitems)
    M = []
    check_wt = True
    for i, x in enumerate(X):
        if check_wt and x.weight() != k:
            print('Warning: these forms do not have the same weight!')
            # in this case the result is probably meaningless, but we'll do it anyway
            check_wt = False
        L = [0] * lenXitems
        for j, g in enumerate(Xitems):
            try:
                L[j] = Xcoeffs[i][g]
            except KeyError:
                pass
        M.append(L)
    return matrix(M)


def omf_pivots(*X):
    r"""
    Compute a set of pivot indices in the list of orthogonal modular forms X.
    """
    return list(omf_matrix(*X).transpose().pivots())


def omf_rank(*X):
    r"""
    Compute the rank of the space spanned by the list of orthogonal modular forms X.

    WARNING: we only check the rank up to the *minimal precision of all elements of X*. For best results let X be a list of forms with the same (sufficiently high) precision!
    """
    return omf_matrix(*X).rank()


def _omf_relations(*X):
    r"""
    Compute all linear relations among the list of orthogonal modular forms X.
    This should no longer be called directly. Use weilrep_misc.relations instead.

    WARNING: we only check the relations up to the *minimal precision of all elements of X*. For best results let X be a list of forms with the same (sufficiently high) precision!
    """
    return omf_matrix(*X).kernel()


def _laplacian(f):
    r"""
    Apply the Laplace operator.

    WARNING: the Laplace operator does not act on modular forms! It is only used to define Rankin--Cohen brackets.
    """
    from .positive_definite import _pos_def_laplacian
    from .lorentz import _lorentz_laplacian
    s = f.qexp_representation()
    if s == 'PD+II' or s == 'siegel' or s == 'hermite':
        return _pos_def_laplacian(f)
    return _lorentz_laplacian(f)


def _omf_rankin_cohen(n, *x):
    r"""
    Compute Rankin--Cohen brackets.

    The n-th Rankin--Cohen bracket is a differential operator that takes a set of modular forms f_1,...,f_r of weights k_1,...,k_r for the orthogonal group
    of a common lattice and produces a modular form of weight k_1+...+k_r + 2*n.

    INPUT:
    - ``n`` -- a nonnegative integer
    - ``*x`` -- modular forms for the orthogonal group of a common lattice; or, a list of modular forms

    OUTPUT: OrthogonalModularForm

    REFERENCE: Choie, Kim - J. Number Theory 82(1), 140--163 (2000)
    """
    if len(x) == 1:
        x = x[0]
    try:
        xref = x[0]
    except TypeError:
        raise ValueError('The Rankin--Cohen bracket requires at least 2 modular forms') from None
    xlen = len(x)

    def list_sums_to(N, list_len):
        # lists of nonnegative integers of length list_len whose sum is N
        if list_len == 1:
            return [[N]]
        a = [[[i] + y for y in list_sums_to(N - i, list_len - 1)] for i in range(N + 1)]
        return [a for b in a for a in b]
    l = xref.nvars()
    k = [f.weight() for f in x]
    beta = sum(k) + 1 - l / 2

    def local_factorial(n):
        if n in ZZ:
            return factorial(n)
        return ZZ(2 * n).multifactorial(2)

    def A(u):
        p = u[-1]
        c = (-1)**p * local_factorial(beta + 2 * n - p - 2) / (factorial(p) * local_factorial(beta + 2 * n - 2))
        return QQ(c / prod(local_factorial(h) * local_factorial(h + k[i] - l / 2) for i, h in enumerate(u[:-1])))
    lap = [[f] for f in x]
    for i in range(n):
        for y in lap:
            y.append(_laplacian(y[-1]))
    U = list_sums_to(n, xlen + 1)
    L = []
    for c in U:
        h = prod(lap[i][j] for i, j in enumerate(c[:-1]))
        for j in range(c[-1]):
            h = _laplacian(h)
        L.append(A(c) * h)
    return sum(L)
