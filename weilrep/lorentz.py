r"""

Additive and multiplicative theta lifts for Lorentzian lattices

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

import cypari2
pari = cypari2.Pari()
PariError = cypari2.PariError

from .weilrep import WeilRep
from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis

from copy import copy, deepcopy
from re import sub

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli, GCD, is_prime, is_square
from sage.arith.srange import srange
from sage.calculus.var import var
from sage.functions.other import ceil, floor, frac, sqrt
from sage.geometry.cone import Cone
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import matrix
from sage.matrix.special import identity_matrix
from sage.misc.functional import denominator, isqrt
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.big_oh import O
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ

sage_one_half = Integer(1) / Integer(2)
sage_three_half = Integer(3) / Integer(2)


class OrthogonalModularFormsLorentzian(object):
    r"""
    This class represents orthogonal modular forms for a Lorentzian lattice. (If the lattice is L and H is the hyperbolic plane then these are automorphic forms on the symmetric domain for L+H.)

    Can be called simply with OrthogonalModularForms(w), where
    - ``w`` -- a WeilRep instance, or a Gram matrix
    """

    def __init__(self, w):
        try:
            S = w.gram_matrix()
        except AttributeError:
            w = WeilRep(w)
            S = w.gram_matrix()
        self.__weilrep = w
        self.__gram_matrix = S

    def __repr__(self):
        return 'Orthogonal modular forms associated to the quadratic form \n%s + U'%self.__gram_matrix

    def gram_matrix(self):
        r"""
        Return self's Gram matrix.
        """
        return self.__gram_matrix

    def weilrep(self):
        r"""
        Return self's Weil representation.
        """
        return self.__weilrep

    #construct modular forms

    def eisenstein_series(self, k, prec):
        r"""
        Compute the Eisenstein series. (i.e. the theta lift of a vector-valued Eisenstein series)

        INPUT:
        - ``k`` -- the weight (an even integer)
        - ``prec`` -- the output precision

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: OrthogonalModularForms(diagonal_matrix([2, 2, -2])).eisenstein_series(4, 5)
            1 + 480*t + (240*x^-2 + (2880*r_0^-1 + 7680 + 2880*r_0)*x^-1 + (240*r_0^-2 + 7680*r_0^-1 + 18720 + 7680*r_0 + 240*r_0^2) + (2880*r_0^-1 + 7680 + 2880*r_0)*x + 240*x^2)*t^2 + ((480*r_0^-2 + 15360*r_0^-1 + 28800 + 15360*r_0 + 480*r_0^2)*x^-2 + (15360*r_0^-2 + 76800*r_0^-1 + 92160 + 76800*r_0 + 15360*r_0^2)*x^-1 + (28800*r_0^-2 + 92160*r_0^-1 + 134400 + 92160*r_0 + 28800*r_0^2) + (15360*r_0^-2 + 76800*r_0^-1 + 92160 + 76800*r_0 + 15360*r_0^2)*x + (480*r_0^-2 + 15360*r_0^-1 + 28800 + 15360*r_0 + 480*r_0^2)*x^2)*t^3 + (2160*x^-4 + (7680*r_0^-2 + 44160*r_0^-1 + 61440 + 44160*r_0 + 7680*r_0^2)*x^-3 + (7680*r_0^-3 + 112320*r_0^-2 + 207360*r_0^-1 + 312960 + 207360*r_0 + 112320*r_0^2 + 7680*r_0^3)*x^-2 + (44160*r_0^-3 + 207360*r_0^-2 + 380160*r_0^-1 + 430080 + 380160*r_0 + 207360*r_0^2 + 44160*r_0^3)*x^-1 + (2160*r_0^-4 + 61440*r_0^-3 + 312960*r_0^-2 + 430080*r_0^-1 + 656160 + 430080*r_0 + 312960*r_0^2 + 61440*r_0^3 + 2160*r_0^4) + (44160*r_0^-3 + 207360*r_0^-2 + 380160*r_0^-1 + 430080 + 380160*r_0 + 207360*r_0^2 + 44160*r_0^3)*x + (7680*r_0^-3 + 112320*r_0^-2 + 207360*r_0^-1 + 312960 + 207360*r_0 + 112320*r_0^2 + 7680*r_0^3)*x^2 + (7680*r_0^-2 + 44160*r_0^-1 + 61440 + 44160*r_0 + 7680*r_0^2)*x^3 + 2160*x^4)*t^4 + O(t^5)
        """
        w = self.__weilrep
        nrows = ZZ(self.__gram_matrix.nrows())
        try:
            return (-((k + k) / bernoulli(k)) * w.eisenstein_series(k + 1 - nrows/2, ceil(prec * prec / 4) + 1)).theta_lift(prec)
        except ValueError:
            raise ValueError('Invalid weight')

    def lifts_basis(self, k, prec):
        r"""
        This computes the theta lifts of a basis of cusp forms.

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- the precision of the output

        OUTPUT: list of OrthogonalModularFormLorentzian's

        EXAMPLES::

            sage: from weilrep import *
            sage: OrthogonalModularForms(diagonal_matrix([2, 2, -2])).lifts_basis(6, 5)
            [t + ((-2*r_0^-1 - 8 - 2*r_0)*x^-1 + (-8*r_0^-1 + 16 - 8*r_0) + (-2*r_0^-1 - 8 - 2*r_0)*x)*t^2 + ((r_0^-2 + 16*r_0^-1 + 20 + 16*r_0 + r_0^2)*x^-2 + (16*r_0^-2 - 32 + 16*r_0^2)*x^-1 + (20*r_0^-2 - 32*r_0^-1 + 168 - 32*r_0 + 20*r_0^2) + (16*r_0^-2 - 32 + 16*r_0^2)*x + (r_0^-2 + 16*r_0^-1 + 20 + 16*r_0 + r_0^2)*x^2)*t^3 + ((-8*r_0^-2 - 36*r_0^-1 - 36*r_0 - 8*r_0^2)*x^-3 + (-8*r_0^-3 - 32*r_0^-2 + 104*r_0^-1 - 128 + 104*r_0 - 32*r_0^2 - 8*r_0^3)*x^-2 + (-36*r_0^-3 + 104*r_0^-2 - 392*r_0^-1 - 392*r_0 + 104*r_0^2 - 36*r_0^3)*x^-1 + (-128*r_0^-2 + 256 - 128*r_0^2) + (-36*r_0^-3 + 104*r_0^-2 - 392*r_0^-1 - 392*r_0 + 104*r_0^2 - 36*r_0^3)*x + (-8*r_0^-3 - 32*r_0^-2 + 104*r_0^-1 - 128 + 104*r_0 - 32*r_0^2 - 8*r_0^3)*x^2 + (-8*r_0^-2 - 36*r_0^-1 - 36*r_0 - 8*r_0^2)*x^3)*t^4 + O(t^5)]
        """
        S = self.__gram_matrix
        nrows = S.nrows()
        w = self.__weilrep
        X = w.cusp_forms_basis(k + 1 - nrows/2, ceil(prec * prec / 4) + 1)
        return [x.theta_lift(prec) for x in X]

    ## methods for borcherds products

    def _borcherds_product_polyhedron(self, pole_order, prec, verbose = False):
        r"""
        Construct a polyhedron representing a cone of Heegner divisors. For internal use in the methods borcherds_input_basis() and borcherds_input_Qbasis().

        INPUT:
        - ``pole_order`` -- pole order
        - ``prec`` -- precision

        OUTPUT: a tuple consisting of an integral matrix M, a Polyhedron p, and a WeilRepModularFormsBasis X
        """
        S = self.gram_matrix()
        wt = 1 - Integer(S.nrows())/2
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
        Compute a basis of input functions into the Borcherds lift.

        This method computes a list of Borcherds lift inputs F_0, ..., F_d which is a basis in the following sense: it is minimal with the property that every modular form with pole order at most pole_order whose Borcherds lift is holomorphic can be expressed in the form (k_0 F_0 + ... + k_d F_d) where k_i are nonnegative integers.

        WARNING: this can take a long time, and the output list might be longer than you expect!!

        INPUT:
        - ``pole_order`` -- positive number (does not need to be an integer)
        - ``prec`` -- precision of the output

        OUTPUT: WeilRepModularFormsBasis

        EXAMPLES::

            sage: from weilrep import *
            sage: OrthogonalModularForms(II(1)).borcherds_input_basis(1, 5)
            [(0, 0), 1 + O(q^5)]
            ------------------------------------------------------------
            [(0, 0), q^-1 + 196884*q + 21493760*q^2 + 864299970*q^3 + 20245856256*q^4 + O(q^5)]
        """
        S = self.gram_matrix()
        w = self.weilrep()
        wt = 1 - Integer(S.nrows())/2
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose = verbose)
        if verbose:
            print('I will now try to find a Hilbert basis.')
        try:
            b = matrix(Cone(p).Hilbert_basis())
            try:
                u = M.solve_left(b)
                Y = [v * X for v in u.rows()]
                Y.sort(key = lambda x: x.fourier_expansion()[0][2][0])
                Y = WeilRepModularFormsBasis(wt, Y, w)
            except ValueError:
                Y = WeilRepModularFormsBasis(wt, [], w)
        except IndexError:
            Y = WeilRepModularFormsBasis(wt, [], w)
        if wt >= 0:
            X = deepcopy(w.basis_vanishing_to_order(wt, max(0, -pole_order), prec))
            if X:
                X.extend(Y)
                return X
        return Y

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
            sage: OrthogonalModularForms(II(1)).borcherds_input_Qbasis(1, 5)
            [(0, 0), 1 + O(q^5)]
            ------------------------------------------------------------
            [(0, 0), q^-1 + 196884*q + 21493760*q^2 + 864299970*q^3 + 20245856256*q^4 + O(q^5)]
        """
        S = self.gram_matrix()
        w = self.weilrep()
        wt = 1 - Integer(S.nrows())/2
        M, p, X = self._borcherds_product_polyhedron(pole_order, prec, verbose = verbose)
        try:
            b = matrix(Cone(p).rays())
            if verbose:
                print('I will now try to find Borcherds product inputs.')
            try:
                u = M.solve_left(b)
                Y = [v * X for v in u.rows()]
                Y.sort(key = lambda x: x.fourier_expansion()[0][2][0])
                Y = WeilRepModularFormsBasis(wt, Y, w)
            except ValueError:
                Y = WeilRepModularFormsBasis(wt, [], w)
        except IndexError:
            Y = WeilRepModularFormsBasis(wt, [], w)
        if wt >= 0:
            X = X = deepcopy(w.basis_vanishing_to_order(wt, max(0, -pole_order), prec))
            if X:
                X.extend(Y)
                return X
        return Y

class OrthogonalModularFormLorentzian:
    r"""
    This class represents modular forms on the type IV domain attached to a lattice of the form L + II_{1, 1}, where L is Lorentzian.
    """

    def __init__(self, k, S, f, scale = 1, weylvec = None, qexp_representation = None):
        self.__weight = k
        self.__gram_matrix = S
        self.__fourier_expansion = f
        self.__precision = f.prec()
        self.__scale = scale
        self.__valuation = f.valuation()
        if weylvec is None:
            self.__weylvec = vector([0] * S.nrows())
        else:
            self.__weylvec = weylvec
        try:
            n = len(qexp_representation)
            if n == 2:
                self.__qexp_representation = qexp_representation[0]
                if self.__qexp_representation == 'hilbert':
                    from .hilbert import HilbertModularForm
                    self.__class__ = HilbertModularForm
                    self._HilbertModularForm__base_field = qexp_representation[1]
            else:
                self.__qexp_representation = qexp_representation
        except TypeError:
            self.__qexp_representation = qexp_representation

    def __repr__(self):
        r"""
        Various string representations, depending on the parameter 'qexp_representation'
        """
        try:
            return self.__string
        except AttributeError:
            d = self.__scale
            h = self.__fourier_expansion
            hprec = h.prec()
            def m(obj):
                m1, m2 = obj.span()
                obj_s = obj.string[m1 : m2]
                x = obj_s[0]
                if x == '^':
                    u = ZZ(obj_s[1:])/d
                    if u.is_integer():
                        if u == 1:
                            return ''
                        return '^%d'%u
                    return '^(%s)'%u
                return (x, obj_s)[x == '_'] + '^(%s)'%(1/d)
            _m = lambda s: sub(r'\^-?\d+|(?<!O\(|\, )(\_\d+|q|s|t|x)(?!\^)', m, s)
            if self.__qexp_representation == 'PD+II':
                S = self.__gram_matrix
                try:
                    f = self.__q_s_exp
                except AttributeError:
                    hval = h.valuation()
                    qs = True
                    try:
                        try:
                            q, s = PowerSeriesRing(self.base_ring(), ('q', 's')).gens()
                            self.__q_s_exp = O(q ** hprec) + sum([(q ** (ZZ(i - n) / 2)) * (s ** (ZZ(i + n) / 2)) * p.coefficients()[j] for i, p in enumerate(h.list()) for j, n in enumerate(p.exponents()) ])
                        except ValueError:
                            self.__q_s_exp = O(q ** (2 * hprec)) + sum([(q ** ((i - n))) * (s ** ((i + n))) * p.coefficients()[j] for i, p in enumerate(h.list()) for j, n in enumerate(p.exponents()) ])
                            d *= 2
                            hprec *= 2
                    except NotImplementedError:
                        rs, s = LaurentSeriesRing(self.base_ring(), 's').objgen()
                        q, = LaurentSeriesRing(rs, 'q').gens()
                        qs = False
                        try:
                            self.__q_s_exp = O(s ** hprec) + O(q ** hprec) + sum([(q ** (ZZ(i + hval - n) / 2)) * (s ** (ZZ(i + hval + n) / 2)) * p.coefficients()[j] for i, p in enumerate(h.list()) for j, n in enumerate(p.exponents()) ])
                        except ValueError:
                            self.__q_s_exp = O(q ** (2 * hprec)) + sum([(q ** ((i + hval - n))) * (s ** ((i + hval + n))) * p.coefficients()[j] for i, p in enumerate(h.list()) for j, n in enumerate(p.exponents()) ])
                            d *= 2
                            hprec *= 2
                    f = self.__q_s_exp
                self.__string = str(f)
                if not qs:
                    self.__string += ' + O(s^%s)'%hprec
                self.__string = self.__string.replace('((', '(')
                self.__string = self.__string.replace('))', ')')
                if d != 1:
                    self.__string = _m(self.__string)
                if S.nrows() == 3:
                    self.__string = self.__string.replace('r_0', 'r')
                return self.__string
            elif self.__qexp_representation == 'shimura':
                s = str(h).replace('t', 'q')
                if d == 1:
                    self.__string = s
                    return s
                else:
                    self.__string = _m(s)
                    return self.__string
            else:
                if d == 1:
                    self.__string = str(h)
                else:
                    self.__string = _m(str(h))
                return self.__string

    def base_ring(self):
        f = self.__fourier_expansion
        try:
            return f.base_ring()
        except AttributeError:
            return self.base_ring()

    def gram_matrix(self):
        r"""
        Return our Gram matrix.
        """
        return self.__gram_matrix

    def precision(self):
        r"""
        Return our precision.
        """
        return self.__precision

    def qexp_representation(self):
        r"""
        Return our qexp representation.
        """
        return self.__qexp_representation

    def rescale(self, d):
        r"""
        Rescale self by "d". This should not change the output when self is printed.
        """
        S = self.gram_matrix()
        nrows = S.nrows()
        f = self.true_fourier_expansion()
        if nrows > 1:
            rb_x = f.base_ring()
            x = rb_x.gens()[0]
            if nrows > 2:
                rbgens = rb_x.base_ring().gens()
                rescale_dict = {a : a ** d for a in rbgens}
                return OrthogonalModularFormLorentzian(self.weight(), S, (f.map_coefficients(lambda y: (x ** (d * y.polynomial_construction()[1])) * rb_x([p.subs(rescale_dict) for p in list(y)]).subs({x : x ** d}))).V(d), scale = self.scale() * d, weylvec = self.weyl_vector(), qexp_representation = self.qexp_representation())
            return OrthogonalModularFormLorentzian(self.weight(), S, (f.map_coefficients(lambda p: p.subs({x : x ** d}))).V(d), scale = self.scale() * d, weylvec = self.weyl_vector(), qexp_representation = self.qexp_representation())
        return OrthogonalModularFormLorentzian(self.weight(), S, f.V(d), scale = self.scale() * d, weylvec = self.weyl_vector(), qexp_representation = self.qexp_representation())

    def scale(self):
        r"""
        Return self's scale.
        """
        return self.__scale

    def true_fourier_expansion(self):
        r"""
        Return self's Fourier expansion (as a power series).
        """
        return self.__fourier_expansion

    def weight(self):
        r"""
        Return self's weight.
        """
        return self.__weight

    def weyl_vector(self):
        r"""
        Return self's Weyl vector (or the zero vector)
        """
        return self.__weylvec

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
            return OrthogonalModularFormLorentzian(self.__weight, self.__gram_matrix, X1.true_fourier_expansion() + X2.true_fourier_expansion(), scale = new_scale, weylvec = self_v, qexp_representation = self.__qexp_representation)
        return OrthogonalModularFormLorentzian(self.__weight, self.__gram_matrix, self.true_fourier_expansion() + other.true_fourier_expansion(), scale = self_scale, weylvec = self_v, qexp_representation = self.__qexp_representation)

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
            return OrthogonalModularFormLorentzian(self.__weight, self.__gram_matrix, X1.true_fourier_expansion() - X2.true_fourier_expansion(), scale = new_scale, weylvec = self_v, qexp_representation = self.__qexp_representation)
        return OrthogonalModularFormLorentzian(self.__weight, self.__gram_matrix, self.true_fourier_expansion() - other.true_fourier_expansion(), scale = self_scale, weylvec = self_v, qexp_representation = self.__qexp_representation)

    def __neg__(self):
        return OrthogonalModularFormLorentzian(self.__weight, self.__gram_matrix, -self.true_fourier_expansion(), scale = self.scale(), weylvec = self.weyl_vector(), qexp_representation = self.__qexp_representation)

    def __mul__(self, other):
        r"""
        Multiply modular forms, rescaling if necessary.
        """
        if isinstance(other, OrthogonalModularFormLorentzian):
            if not self.gram_matrix() == other.gram_matrix():
                raise ValueError('Incompatible Gram matrices')
            self_scale = self.scale()
            other_scale = other.scale()
            if self_scale != 1 or other_scale != 1:
                new_scale = lcm(self.scale(), other.scale())
                X1 = self.rescale(new_scale // self_scale)
                X2 = other.rescale(new_scale // other_scale)
                return OrthogonalModularFormLorentzian(self.weight() + other.weight(), self.__gram_matrix, X1.true_fourier_expansion() * X2.true_fourier_expansion(), scale = new_scale, weylvec = self.weyl_vector() + other.weyl_vector(), qexp_representation = self.__qexp_representation)
            return OrthogonalModularFormLorentzian(self.weight() + other.weight(), self.__gram_matrix, self.true_fourier_expansion() * other.true_fourier_expansion(), scale = 1, weylvec = self.weyl_vector() + other.weyl_vector(), qexp_representation = self.__qexp_representation)
        elif other in QQ:
            return OrthogonalModularFormLorentzian(self.weight(), self.__gram_matrix, self.true_fourier_expansion() * other, scale = self.scale(), weylvec = self.weyl_vector(), qexp_representation = self.__qexp_representation)

    __rmul__ = __mul__

    def __div__(self, other):
        r"""
        Divide modular forms, rescaling if necessary. (This is usually not a good idea.)
        """
        if isinstance(other, OrthogonalModularFormLorentzian):
            if not self.gram_matrix() == other.gram_matrix():
                raise ValueError('Incompatible Gram matrices')
            self_scale = self.scale()
            other_scale = other.scale()
            if self_scale != 1 or other_scale != 1:
                new_scale = lcm(self.scale(), other.scale())
                X1 = self.rescale(new_scale // self_scale)
                X2 = other.rescale(new_scale // other_scale)
                return OrthogonalModularFormLorentzian(self.weight() - other.weight(), self.__gram_matrix, X1.true_fourier_expansion() / X2.true_fourier_expansion(), scale = new_scale, weylvec = self.weyl_vector() - other.weyl_vector(), qexp_representation = self.__qexp_representation)
            return OrthogonalModularFormLorentzian(self.weight() - other.weight(), self.__gram_matrix, self.true_fourier_expansion() / other.true_fourier_expansion(), scale = 1, weylvec = self.weyl_vector() - other.weyl_vector(), qexp_representation = self.__qexp_representation)
        elif other in QQ:
            return OrthogonalModularFormLorentzian(self.weight(), self.__gram_matrix, self.true_fourier_expansion() / other, scale = self.scale(), weylvec = self.weyl_vector(), qexp_representation = self.__qexp_representation)

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
        return OrthogonalModularFormLorentzian(other * self.weight(), self.__gram_matrix, self.true_fourier_expansion() ** other, scale=self.scale(), weylvec = other * self.weyl_vector(), qexp_representation = self.__qexp_representation)

class WeilRepLorentzian(WeilRep):
    def __init__(self, S, lift_qexp_representation = None):
        #S should be a Lorentzian lattice in which the bottom-right entry is negative
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self.lift_qexp_representation = lift_qexp_representation

    def change_of_basis_matrix(self):
        try:
            return self.__change_of_basis_matrix
        except AttributeError:
            S = self.gram_matrix()
            b = S.rows()[-1]
            b_perp = matrix(b).transpose().kernel().basis()
            e0 = vector([0] * S.nrows())
            e0[-1] = 1
            self.__change_of_basis_matrix = matrix(b_perp + [e0])
            return self.__change_of_basis_matrix

    def is_lorentzian(self):
        return True

    def orthogonalized_gram_matrix(self):
        try:
            return self.__orthogonalized_gram_matrix
        except AttributeError:
            S = self.gram_matrix()
            a = self.change_of_basis_matrix()
            self.__orthogonalized_gram_matrix = a * S * a.transpose()
            return self.__orthogonalized_gram_matrix


class RescaledHyperbolicPlane(WeilRepLorentzian):
    r"""
    Represents the rescaled hyperbolic plane (i.e. Z^2, with quadratic form (x, y) -> Nxy for some N).

    The main use of this is to provide certain orthogonal modular forms with Fourier--Jacobi expansions. You can add a RescaledHyperbolicPlane to a positive-definite WeilRep to produce input forms whose lifts are given as Fourier--Jacobi series instead of the default 't', 'x', 'r_0,...' For example:

    w = WeilRep(matrix([[2]]))
    w = w + II(3)
    """

    def __init__(self, N):
        self.__N = N
        S = matrix([[0, N], [N, -(N + N)]])
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self.lift_qexp_representation = 'PD+II'

    def N(self):
        return self.__N

def II(N): #short constructor for rescaled hyperbolic planes
    return RescaledHyperbolicPlane(N)

class WeilRepModularFormLorentzian(WeilRepModularForm):

    def __init__(self, k, f, w):
        self._WeilRepModularForm__weight = k
        self._WeilRepModularForm__fourier_expansions = f
        self._WeilRepModularForm__weilrep = w
        self._WeilRepModularForm__gram_matrix = w.gram_matrix()

    def theta_lift(self, prec = None):
        r"""
        Compute the (additive) theta lift.

        This computes the additive theta lift (e.g. Shimura lift; Doi--Naganuma lift; etc) of the given vector-valued modular form.

        INPUT:
        - ``prec`` -- max precision (default None). (This is limited by the precision of the input. If prec is None then we compute as much as possible.)

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: WeilRep(matrix([[2, 1, 0], [1, 2, 0], [0, 0, -2]])).cusp_forms_basis(11/2, 5)[0].theta_lift()
            t + ((-6*r_0^-1 - 6)*x^-1 + (-6*r_0^-1 + 12 - 6*r_0) + (-6 - 6*r_0)*x)*t^2 + ((15*r_0^-2 + 24*r_0^-1 + 15)*x^-2 + (24*r_0^-2 - 24*r_0^-1 - 24 + 24*r_0)*x^-1 + (15*r_0^-2 - 24*r_0^-1 + 162 - 24*r_0 + 15*r_0^2) + (24*r_0^-1 - 24 - 24*r_0 + 24*r_0^2)*x + (15 + 24*r_0 + 15*r_0^2)*x^2)*t^3 + O(t^4)
        """
        prec0 = self.precision()
        val = self.valuation()
        if val < 0:
            raise ValueError('Nonholomorphic input function in theta lift')
        if prec is None:
            prec = isqrt(4 * prec0)
        else:
            prec = min(prec, isqrt(4 * prec0))
        wt = self.weight()
        coeffs = self.coefficients()
        w = self.weilrep()
        S = w.gram_matrix()
        s_0 = w.orthogonalized_gram_matrix()
        if self.is_symmetric() == 1:
            eps = 1
        else:
            eps = -1
        nrows = Integer(s_0.nrows())
        k = wt + nrows/2 - 1
        if nrows > 1:
            if nrows > 2:
                rb = LaurentPolynomialRing(QQ, list(var('r_%d' % i) for i in range(nrows - 2)))
            else:
                rb = QQ
            rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        else:
            rb_x = QQ
            x = 1
        t, = PowerSeriesRing(rb_x, 't').gens()
        a = w.change_of_basis_matrix()
        a_tr = a.transpose()
        scale = a.determinant()
        b_norm = s_0[-1,-1]
        s_0 = s_0[:-1,:-1]
        s_0inv = s_0.inverse()
        new_prec = ceil(prec * prec * scale / (-4 * b_norm) + prec)
        if nrows >= 3:
            v_matrix = _, _, vs_matrix = pari(s_0inv).qfminim(new_prec, flag = 2)
            vs_list = vs_matrix.sage().columns()
            rb0 = rb.gens()[0]
        elif nrows == 2:
            vs_list = [vector([n]) for n in srange(1, isqrt(2 * new_prec * s_0[0, 0]))]
        else:
            vs_list = []
        lift = O(t ** prec)
        if k % 2 == 0:
            try:
                c = coeffs[tuple([0]*(S.nrows() + 1))]
                if c:
                    lift -= c * bernoulli(k) / (k + k)
            except KeyError:
                pass
        for v in vs_list:
            sv = s_0inv * v
            j = 1
            while j < prec:
                prec_j = prec//j + 1
                v_big = vector(list(sv) + [-j / b_norm])
                z = a_tr * v_big
                norm_z = z * S * z / 2
                try:
                    c = coeffs[tuple([frac(y) for y in z] + [-norm_z] )]
                    m = x ** v[0]
                    if nrows >= 3:
                        if nrows >= 4:
                            m *= rb.monomial(*v[1:])
                        else:
                            m *= rb0 ** v[1]
                    lift += c * sum([n ** (k - 1) * (m ** n + eps * m ** (-n)) * t ** (n * j) for n in srange(1, prec_j)])
                except KeyError:
                    if -norm_z >= self.precision():
                        prec = j
                        break
                    pass
                except KeyError:
                    if -norm_z >= self.precision():
                        prec= j
                        break
                    pass
                j += 1
        for j in range(1, prec):
            v_big = vector([0]*(nrows - 1) + [-j / b_norm])
            z = a_tr * v_big
            norm_z = z * S * z / 2
            try:
                c = coeffs[tuple([frac(y) for y in z] + [-norm_z] )]
                lift += c * sum([n ** (k - 1) * t ** (n * j) for n in srange(1, prec//j + 1)])
            except KeyError:
                if -norm_z >= self.precision():
                    prec= j
                    break
                pass
        return OrthogonalModularFormLorentzian(k, S, lift + O(t ** prec), scale = 1, qexp_representation = w.lift_qexp_representation)

    def weyl_vector(self, _x = True):
        r"""
        Compute the Weyl vector for the Borcherds lift.

        NOTE: this uses a recursive algorithm to compute the Weyl vector instead of a closed formula. This can be slow so it would be better to use something like Theorem 10.4 of [Borcherds1998]

        WARNING: Do not call this with the keyword _x = False! you can run into errors or infinite loops! it is for internal use only!

        OUTPUT: a vector

        """
        w = self.weilrep()
        S = w.gram_matrix()
        nrows = Integer(S.nrows())
        #suppose v is a primitive vector. extend_vector computes a matrix M in GL_n(Z) whose left column is v. If v has a denominator then we divide it off first.
        extend_vector = lambda v: matrix(ZZ, matrix([v]).transpose().echelon_form(transformation = True)[1].inverse())
        if nrows > 1:
            N = nrows - 1
            u = identity_matrix(nrows)
            u[0, 0], u[0, N], u[N, 0], u[N, N] = 0, 1, 1, 0
            if _x:
                a = w.change_of_basis_matrix().transpose()
                S = a.transpose() * S * a
                X_small = self.reduce_precision(2).conjugate(a)
                X = X_small.conjugate(u)
            else:
                X_small = copy(self)
                X = copy(X_small)
            v = vector([0] + list(X.theta_contraction().weyl_vector(_x = False)))
            if not _x:
                X_small = X_small.conjugate(u)
                S = u * S * u
            norm = S[0, 0] // 2
            u = -S[-1, -1] // 2
            m = X.valuation(exact = True)
            prime = GCD(norm, u) == 1 and not (norm.is_square() and u.is_square())
            a = vector([-1] + [0] * (N - 1) + [isqrt(4 * norm * (1 / (4 * u) - m))])
            norm = ZZ(a * S * a // 2)
            while (norm >= 0 or is_square(-norm)) or (prime and not is_prime(-norm)):
                a[-1] += 1
                norm = ZZ(a * S * a // 2)
            u = extend_vector(a)
            X = X_small.conjugate(u)
            while N > 0:
                X = X.theta_contraction()
                N -= 1
            h = X.weyl_vector()
            v[0] = h[0] - v[1:] * a[1:]
            return v
        else:
            N = -ZZ(S[0, 0]/2)
            val = self.valuation(exact = True)
            q, = PowerSeriesRing(QQ, 'q').gens()
            if N == 1 or N.is_prime():
                f = self.fourier_expansion()
                if val < 0:
                    L = []
                    ds = []
                    for i, x in enumerate(f):
                        if x[2].valuation() <= 0:
                            ds.append(x[0])
                            L.append(i)
                    _ds = [x[0] for i, x in enumerate(f) if x[2].valuation() <= 0]
                    indices = [None] * len(_ds)
                    e = w.dual().eisenstein_series(sage_three_half, max(1, 1 - ceil(val)), allow_small_weight = True, components = (_ds, indices)).fourier_expansion()
                    s = sum([(f[j][2] * e[i][2] * q ** (floor(f[j][1])))[0] for i, j in enumerate(L)])
                else:
                    s = f[0][2][0]
                return vector([s * (1 + N) / 24])
            i = 0
            L = []
            v = [None, None]
            h = copy(v)
            mult = max(1, ceil(1/4 - val))
            j = 1
            s = isqrt(N) 
            s = s * s == N
            sqr = False
            if s:
                if N % 2:
                    if -val < 10:
                        X = WeilRepModularForm(-sage_one_half, matrix([[4]]), [((0,), 0, 4 + 16*q + 56*q**2 + 160*q**3 + 400*q**4 + 928*q**5 + 2016*q**6 + 4160*q**7 + 8248*q**8 + 15792*q**9 + O(q**10)), ((QQ(1)/4,), QQ(-1)/8, 1 - q + q**2 - 2*q**3 + 3*q**4 - 4*q**5 + 5*q**6 - 7*q**7 + 10*q**8 - 13*q**9 + 16*q**10 + O(q**11)), ((QQ(1)/2,), QQ(-1)/2, -8*q - 32*q**2 - 96*q**3 - 256*q**4 - 616*q**5 - 1376*q**6 - 2912*q**7 - 5888*q**8 - 11456*q**9 - 21600*q**10 + O(q**11)), ((QQ(3)/4,), QQ(-1)/8, 1 - q + q**2 - 2*q**3 + 3*q**4 - 4*q**5 + 5*q**6 - 7*q**7 + 10*q**8 - 13*q**9 + 16*q**10 + O(q**11))])
                    else:
                        X = WeilRep(matrix([[4]])).nearly_holomorphic_modular_forms_basis(-sage_one_half, 1, ceil(-val))[0]
                    p_0 = 2
                elif N % 3:
                    if -val < 10:
                        X = WeilRepModularForm(-sage_one_half, matrix([[6]]), [((0,), 0, 2 + 4*q + 12*q**2 + 24*q**3 + 52*q**4 + 96*q**5 + 180*q**6 + 312*q**7 + 540*q**8 + 892*q**9 + O(q**10)), ((QQ(1)/6,), QQ(-1)/12, 1 + 2*q + 5*q**2 + 12*q**3 + 24*q**4 + 46*q**5 + 85*q**6 + 150*q**7 + 257*q**8 + 430*q**9 + 701*q**10 + O(q**11)), ((QQ(1)/3,), QQ(-1)/3, -2*q - 4*q**2 - 10*q**3 - 20*q**4 - 40*q**5 - 72*q**6 - 132*q**7 - 224*q**8 - 380*q**9 - 620*q**10 + O(q**11)), ((QQ(1)/2,), QQ(-3)/4, -2*q - 6*q**2 - 14*q**3 - 30*q**4 - 60*q**5 - 114*q**6 - 206*q**7 - 360*q**8 - 612*q**9 - 1014*q**10 + O(q**11)), ((QQ(2)/3,), QQ(-1)/3, -2*q - 4*q**2 - 10*q**3 - 20*q**4 - 40*q**5 - 72*q**6 - 132*q**7 - 224*q**8 - 380*q**9 - 620*q**10 + O(q**11)), ((QQ(5)/6,), QQ(-1)/12, 1 + 2*q + 5*q**2 + 12*q**3 + 24*q**4 + 46*q**5 + 85*q**6 + 150*q**7 + 257*q**8 + 430*q**9 + 701*q**10 + O(q**11))])
                    else:
                        X = WeilRep(matrix([[6]])).nearly_holomorphic_modular_forms_basis(-sage_one_half, 1, ceil(-val))[0]
                    p_0 = 3
                else:
                    sqr = True
            if sqr or not s:
                if -val < 10:
                    X = WeilRepModularForm(-sage_one_half, matrix([[2]]), [((0,), 0, 10 + 108*q + 808*q**2 + 4016*q**3 + 16524*q**4 + 58640*q**5 + 188304*q**6 + 556416*q**7 + 1541096*q**8 + 4038780*q**9 + O(q**10)), ((QQ(1)/2,), QQ(-1)/4, 1 - 64*q - 513*q**2 - 2752*q**3 - 11775*q**4 - 43200*q**5 - 141826*q**6 - 427264*q**7 - 1201149*q**8 - 3189120*q**9 - 8067588*q**10 + O(q**11))])
                else:
                    X = WeilRep(matrix([[2]])).nearly_holomorphic_modular_forms_basis(-sage_one_half, 1, ceil(-val))[0]
                p_0 = 1
            X = self * X
            m = X.valuation(exact = True)
            scale = isqrt(2 / N - 4 * p_0 * m)
            while i < 2:
                jN = j * j * N
                for p in range(jN - j * j * scale - 1, jN):
                    d = jN - p
                    f = isqrt(d // p_0)
                    if f * f * p_0 == d and (sqr or is_prime(p)):
                        v[i] = vector([j, f])
                        L.append(extend_vector(v[i]))
                        i += 1
                    if i == 2:
                        break
                j += 1
            for j, a in enumerate(L):
                x = X.conjugate(a).theta_contraction()
                h[j] = QQ(x.weyl_vector()[0])
            return vector([(matrix(v).solve_right(vector(h)) * Integer(p_0) / 12)[0]])

    def borcherds_lift(self, prec = None):
        r"""
        Compute the Borcherds lift.

        INPUT:
        - ``prec`` -- precision (optional). The precision of the output is limited by the precision of the input. However if ``prec`` is given then the output precision will not exceed ``prec``.

        OUTPUT: OrthogonalModularForm of weight equal to (1/2) of self's constant term.

        EXAMPLES::

            sage: from weilrep import *
            sage: WeilRep(matrix([[-10]])).theta_series(50).borcherds_lift()
            q^(1/4) - q^(5/4) - q^(9/4) + q^(25/4) + 2*q^(29/4) - 2*q^(41/4) + q^(45/4) - q^(49/4) + O(q^(57/4))

            sage: from weilrep import *
            sage: x = var('x')
            sage: K.<sqrt5> = NumberField(x * x - 5)
            sage: HMF(K).borcherds_input_Qbasis(1, 5)[0].borcherds_lift()
            q1^(-1/10*sqrt5 + 1/2)*q2^(1/10*sqrt5 + 1/2) - 1*q1^(1/10*sqrt5 + 1/2)*q2^(-1/10*sqrt5 + 1/2) - 1*q1^(-2/5*sqrt5 + 1)*q2^(2/5*sqrt5 + 1) - 10*q1^(-1/5*sqrt5 + 1)*q2^(1/5*sqrt5 + 1) + 10*q1^(1/5*sqrt5 + 1)*q2^(-1/5*sqrt5 + 1) + q1^(2/5*sqrt5 + 1)*q2^(-2/5*sqrt5 + 1) + 120*q1^(-3/10*sqrt5 + 3/2)*q2^(3/10*sqrt5 + 3/2) - 108*q1^(-1/10*sqrt5 + 3/2)*q2^(1/10*sqrt5 + 3/2) + 108*q1^(1/10*sqrt5 + 3/2)*q2^(-1/10*sqrt5 + 3/2) - 120*q1^(3/10*sqrt5 + 3/2)*q2^(-3/10*sqrt5 + 3/2) + 10*q1^(-4/5*sqrt5 + 2)*q2^(4/5*sqrt5 + 2) - 108*q1^(-3/5*sqrt5 + 2)*q2^(3/5*sqrt5 + 2) - 156*q1^(-2/5*sqrt5 + 2)*q2^(2/5*sqrt5 + 2) - 140*q1^(-1/5*sqrt5 + 2)*q2^(1/5*sqrt5 + 2) + 140*q1^(1/5*sqrt5 + 2)*q2^(-1/5*sqrt5 + 2) + 156*q1^(2/5*sqrt5 + 2)*q2^(-2/5*sqrt5 + 2) + 108*q1^(3/5*sqrt5 + 2)*q2^(-3/5*sqrt5 + 2) - 10*q1^(4/5*sqrt5 + 2)*q2^(-4/5*sqrt5 + 2) + O(q1, q2)^5

        """
        prec0 = self.precision()
        val = self.valuation()
        prec0val = prec0 - val
        if prec is None:
            prec = isqrt(4 * (prec0+val))
        else:
            prec = min(prec, isqrt(4 * (prec0+val)))
        wt = self.weight()
        coeffs = self.coefficients()
        w = self.weilrep()
        S = w.gram_matrix()
        s_0 = w.orthogonalized_gram_matrix()
        nrows = Integer(S.nrows())
        try:
            k = coeffs[tuple([0] * (nrows + 1))] / 2
        except KeyError:
            k = 0
        if nrows > 1:
            if nrows > 2:
                rb = LaurentPolynomialRing(QQ, list(var('r_%d' % i) for i in range(nrows - 2)))
            else:
                rb = QQ
            rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        else:
            rb_x = QQ
            x = 1
        r, t = PowerSeriesRing(rb_x, 't').objgen()
        weyl_vector = self.weyl_vector()
        d = denominator(weyl_vector)
        weyl_vector *= d
        a = w.change_of_basis_matrix()
        a_tr = a.transpose()
        scale = a.determinant()
        b_norm = s_0[-1, -1]
        s_0 = s_0[:-1,:-1]
        s_0inv = s_0.inverse()
        new_prec = ceil(prec * (prec * scale/ (-4 * b_norm) + 1))
        if nrows >= 3:
            v_matrix = _, _, vs_matrix = pari(s_0inv).qfminim(new_prec, flag = 2)
            vs_list = vs_matrix.sage().columns()
            rb0 = rb.gens()[0]
        elif nrows == 2:
            vs_list = [vector([n]) for n in range(1, isqrt(2 * new_prec * s_0[0, 0]))]
        else:
            vs_list = []
        h = O(t ** prec)
        f = 1 + h
        val = self.valuation(exact = True)
        corrector = r(1)
        excluded_vectors = set([])
        rpoly, tpoly = PolynomialRing(QQ, 'tpoly').objgen()
        for v in vs_list:
            sv = s_0inv * v
            v *= d
            m = x**v[0]
            if nrows >= 3:
                if nrows >= 4:
                    m *= rb.monomial(*v[1:])
                else:
                    m *= rb0**v[1]
            for j in range(1, prec):
                u = 0
                v_big = vector(list(sv) + [-j / b_norm])
                z = a_tr * v_big
                norm_z = z * S * z / 2
                try:
                    c = coeffs[tuple([frac(y) for y in z] + [-norm_z] )]
                    u = c
                    f *= (1 - t**j * (m + ~m) + t**(j + j) + h) ** c
                except KeyError:
                    if -norm_z >= prec0:
                        prec = j
                        h = O(t ** j)
                        break
                    pass
            if nrows > 1 and tuple(v) not in excluded_vectors:
                s = next(s for s in sv if s)
                if s < 0:
                    v *= -1
                v_big = vector(list(sv) + [0])
                z = a_tr * v_big
                norm_z = z * S * z / 2
                try:
                    c = coeffs[tuple([frac(y) for y in z] + [-norm_z] )]
                    if c > 0:
                        f *= (1 - m + h) ** c
                    else:
                        p = (1 - tpoly) ** c
                        for j in range(2, isqrt(-val / norm_z) + 1):
                            try:
                                excluded_vectors.add(tuple(j * v))
                                c_new = coeffs[tuple([frac(j * y) for y in z] + [-j * j * norm_z] )]
                                p *= (1 - tpoly**j) ** c_new
                            except KeyError:
                                pass
                        f *= rb_x(rpoly(p).subs({tpoly:m}))
                except KeyError:
                    pass
        v = a.rows()[-1]
        norm_v = v * S * v / 2
        for j in range(1, prec):
            jb = -j/b_norm
            norm_z = -norm_v * jb * jb
            try:
                c = coeffs[tuple([frac(jb * y) for y in v] + [norm_z])]
                f *= (1 - t ** j + h) ** c
            except KeyError:
                if norm_z >= prec0:
                    prec = j
                    h = O(t ** j)
                pass
        weyl_monomial = x ** (weyl_vector[0])
        if nrows >= 3:
            if nrows >= 4:
                weyl_monomial *= rb.monomial(*weyl_vector[1:-1])
            else:
                weyl_monomial *= rb0**weyl_vector[1]
        return OrthogonalModularFormLorentzian(k, S, f.V(d) * weyl_monomial * (t ** weyl_vector[-1]), scale = d, weylvec = weyl_vector, qexp_representation = w.lift_qexp_representation)