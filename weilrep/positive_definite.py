r"""

Additive and multiplicative theta lifts for positive-definite lattices

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

import math

import cypari2
pari = cypari2.Pari()
PariError = cypari2.PariError

from copy import copy, deepcopy
from re import sub

from sage.arith.functions import lcm
from sage.arith.misc import bernoulli, divisors, GCD, is_prime, is_square
from sage.arith.srange import srange
from sage.calculus.var import var
from sage.combinat.combinat import bernoulli_polynomial
from sage.functions.log import exp, log
from sage.functions.other import binomial, ceil, floor, frac, sqrt
from sage.geometry.cone import Cone
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import matrix
from sage.matrix.special import block_diagonal_matrix, block_matrix, identity_matrix
from sage.misc.functional import denominator, isqrt
from sage.modular.arithgroup.congroup_gamma0 import Gamma0_constructor
from sage.modular.modform.constructor import ModularForms
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.rings.big_oh import O
from sage.rings.fraction_field import FractionField
from sage.rings.infinity import Infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR

sage_one_half = Integer(1) / Integer(2)
sage_three_half = Integer(3) / Integer(2)

from .jacobi_forms_class import JacobiForm, JacobiForms, JacobiFormWithCharacter
from .lifts import OrthogonalModularForm, OrthogonalModularForms
from .weilrep import WeilRep
from .weilrep_modular_forms_class import EtaCharacterPower, WeilRepModularForm, WeilRepModularFormWithCharacter, WeilRepModularFormsBasis



class OrthogonalModularFormsPositiveDefinite(OrthogonalModularForms):
    r"""
    Represents spaces of Orthogonal modular forms for positive-definite lattices (or rather lattices of the form 2U + K, K positive definite)

    Compared to more general lattices (in lifts.py and lorentz.py) this provides Fourier--Jacobi expansions and better ways to construct Borcherds products.
    """

    def input_wt(self):
        r"""
        Nearly-holomorphic modular forms of this weight lift to Borcherds products.
        """
        return 1 - self.nvars() / 2

    def nvars(self):
        r"""
        The number of variables appearing in the Fourier expansion of a modular form.
        """
        return self.nrows() + 2

    def modular_form_from_fourier_jacobi_expansion(self, fj):
        r"""
        Recover an orthogonal modular form from its Fourier--Jacobi coefficients.

        WARNING: we do not check whether the Fourier--Jacobi coefficients actually correspond to a modular form!

        INPUT:
        - ``fj`` -- a list of Jacobi forms

        OUTPUT: OrthogonalModularForm

        EXAMPLES::
            sage: from weilrep import *
            sage: f = jacobi_eisenstein_series(4, 1, 10)
            sage: m = ParamodularForms(1)
            sage: m.modular_form_from_fourier_jacobi_expansion([f.hecke_V(N) for N in range(5)])
            1/240 + q + s + 9*q^2 + (r^-2 + 56*r^-1 + 126 + 56*r + r^2)*q*s + 9*s^2 + 28*q^3 + (126*r^-2 + 576*r^-1 + 756 + 576*r + 126*r^2)*q^2*s + (126*r^-2 + 576*r^-1 + 756 + 576*r + 126*r^2)*q*s^2 + 28*s^3 + 73*q^4 + (56*r^-3 + 756*r^-2 + 1512*r^-1 + 2072 + 1512*r + 756*r^2 + 56*r^3)*q^3*s + (9*r^-4 + 576*r^-3 + 2520*r^-2 + 4032*r^-1 + 5166 + 4032*r + 2520*r^2 + 576*r^3 + 9*r^4)*q^2*s^2 + (56*r^-3 + 756*r^-2 + 1512*r^-1 + 2072 + 1512*r + 756*r^2 + 56*r^3)*q*s^3 + 73*s^4 + O(q, s)^5
        """
        if len(fj) < 2:
            raise ValueError('Too few coefficients')
        k = fj[0].weight()
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
        return OrthogonalModularForm(k, self.weilrep(), f, 1, vector([0] * (nrows + 2)))

    def borcherds_input_by_weight(self, k, prec, pole_order = None, verbose = False, **kwargs):
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
            pole_order = (2 * k / abs(eisenstein_bound)) ** (1 / (1 - wt))
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
        p = Polyhedron(ieqs = positive, eqns = [r] + [vector([0] + list(v)) for v in vs], **kwargs)
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

class OrthogonalModularFormPositiveDefinite(OrthogonalModularForm):

    def __repr__(self):
        r"""
        Print self's Fourier expansion. If 'scale' > 1 then we divide all exponents by 'scale' in order to print modular forms with characters. If the Gram matrix is 1x1 then we pretend to use the variable 'r' instead of 'r_0'.
        """
        try:
            return self.__string
        except AttributeError:
            s = str(self.fourier_expansion())
            d = self.scale()
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

    def nvars(self):
        return 2 + Integer(self.gram_matrix().nrows())

    ## Fourier series and Fourier--Jacobi series

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
            h = self._OrthogonalModularForm__fourier_expansion
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
            vfrac = frac(v[0])
            if vfrac + vfrac and vfrac + vfrac != 1:
                raise NotImplemented('Invalid character')
            scale = denominator(vector(v[1:-1]))
            prec = self.precision()
            L = [O(q ** (prec - ceil(vfrac) - n)) for n in range(prec)]
            coeffs = self.qs_coefficients()
            for x, y in coeffs.items():
                a = x[0]
                c = x[2]
                b = [scale * x for x in x[1:-1]]
                if nrows > 1:
                    u = rb.monomial(*b)
                else:
                    u = z**(b[0])
                L[c - vfrac] += (q ** ZZ(a - vfrac)) * u * y
            if vfrac:
                d = bool(vfrac)
                chi = EtaCharacterPower(24 * v[0])
                self.__fourier_jacobi = [JacobiFormWithCharacter(k, (n  + d - vfrac) * S, j, qshift = vfrac, character = chi, w_scale = scale) for n, j in enumerate(L)]
            else:
                self.__fourier_jacobi = [JacobiForm(k, n * S, j) for n, j in enumerate(L)]
            return self.__fourier_jacobi
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

    def is_lift(self):
        r"""
        Return True if self's (known) Fourier coefficients satisfy the Maass relations, otherwise False

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
            if not bool(f):
                return False
            return all(f.hecke_V(N) == x for N, x in enumerate(X) if N)
        except NotImplementedError:
            return False

    def _add_II(self):
        from .lorentz import OrthogonalModularFormLorentzian, II
        w = self.weilrep() + II(Integer(1))
        return OrthogonalModularFormLorentzian(self.weight(), w, self.true_fourier_expansion(), scale = self.scale(), weylvec = self.weyl_vector(), qexp_representation = 'PD+II')

    ## other methods

    def phi(self):
        r"""
        Apply the Siegel Phi operator.

        The Siegel Phi operator sets s->0 in self's Fourier--Jacobi expansion. The result is an OrthogonalModularForm on a lattice of signature (2, 1).

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: w = WeilRep(matrix([[2, 1], [1, 2]]))
            sage: m = OrthogonalModularForms(w)
            sage: m.eisenstein_series(4, 5).phi()
            1 + 240*q + 2160*q^2 + 6720*q^3 + 17520*q^4 + O(q^5)

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[4]]))
            sage: X = m.borcherds_input_Qbasis(1, 10)
            sage: X[2].borcherds_lift().phi()
            -q + 24*q^2 - 252*q^3 + 1472*q^4 - 4830*q^5 + O(q^6)
        """
        f = self.true_fourier_expansion()
        R = PowerSeriesRing(QQ, 't')
        prec = f.prec()
        f = R([f[j][j] for j in range(prec)]).O(prec)
        from .lorentz import WeilRepLorentzian, OrthogonalModularFormLorentzian
        S = matrix([[-2]])
        return OrthogonalModularFormLorentzian(self.weight() / 2, WeilRepLorentzian(S), f, scale = self.scale(), weylvec = vector([self.weyl_vector()[0]]), qexp_representation = 'shimura')

    def pullback(self, *v):
        if not v:
            return self
        v_ref = v[0]
        S = self.gram_matrix()
        if len(v_ref) > S.nrows():
            return self._add_II().pullback( *v )
        f = self.true_fourier_expansion()
        r = f.base_ring().base_ring()
        z = r.gens()
        z_new = z[:len(v)]
        r_new = LaurentPolynomialRing(r.base_ring(), z_new)
        s = LaurentPolynomialRing(r_new, 'x')
        s = PowerSeriesRing(s, 't')
        A = matrix(v)
        if len(z_new) > 1:
            a = A.columns()
            d = {x: r_new.monomial(*a[i]) for i, x in enumerate(z)}
        else:
            z0, = r_new.gens()
            d = {x: z0**A[0][i] for i, x in enumerate(z)}
        S = A * self.gram_matrix() * A.transpose()
        u = self.weyl_vector()
        u = vector([u[0]] + list(A * u[1:-1]) + [u[-1]])
        return OrthogonalModularForm(self.weight(), WeilRep(S), s(f.map_coefficients(lambda y: y.map_coefficients(lambda z: z.subs(d)))), scale = self.scale(), weylvec = u)

    def witt(self):
        r"""
        Apply the Witt operator.

        The Witt operator sets all r_i to 1 in self's Fourier--Jacobi expansion. The result is an OrthogonalModularForm on a lattice of signature (2, 2).

        OUTPUT: OrthogonalModularFormLorentzian

        EXAMPLES::

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[2]]))
            sage: X = m.borcherds_input_Qbasis(1, 10)
            sage: X[1].borcherds_lift().witt()
            2*q^(5/2)*s^(3/2) - 2*q^(3/2)*s^(5/2) - 120*q^(7/2)*s^(3/2) + 120*q^(3/2)*s^(7/2) + 3420*q^(9/2)*s^(3/2) - 389988*q^(7/2)*s^(5/2) + 389988*q^(5/2)*s^(7/2) - 3420*q^(3/2)*s^(9/2) - 61360*q^(11/2)*s^(3/2) - 19505280*q^(9/2)*s^(5/2) + 19505280*q^(5/2)*s^(9/2) + 61360*q^(3/2)*s^(11/2) + 773490*q^(13/2)*s^(3/2) + 180216090*q^(11/2)*s^(5/2) + 1837196280*q^(9/2)*s^(7/2) - 1837196280*q^(7/2)*s^(9/2) - 180216090*q^(5/2)*s^(11/2) - 773490*q^(3/2)*s^(13/2) + O(q, s)^9

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(matrix([[2, 1], [1, 2]]))
            sage: m.eisenstein_series(4, 5).witt()
            1 + 240*q + 240*s + 2160*q^2 + 57600*q*s + 2160*s^2 + 6720*q^3 + 518400*q^2*s + 518400*q*s^2 + 6720*s^3 + 17520*q^4 + 1612800*q^3*s + 4665600*q^2*s^2 + 1612800*q*s^3 + 17520*s^4 + O(q, s)^5
        """
        a = lambda x: sum(x[1] for x in x.dict().items())
        def b(x):
            u, v = x.polynomial_construction()
            return u.map_coefficients(a) * (x.parent().gens()[0]**v)
        f = self.true_fourier_expansion().map_coefficients(b)
        rb, x = LaurentPolynomialRing(QQ, 'x').objgen()
        r, t = PowerSeriesRing(rb, 't').objgen()
        f = r(f)
        from .lorentz import WeilRepLorentzian, OrthogonalModularFormLorentzian
        S = matrix([[-2, 1], [1, 0]])
        return OrthogonalModularFormLorentzian(self.weight(), WeilRepLorentzian(S), f, scale = self.scale(), weylvec = vector([self.weyl_vector()[0], self.weyl_vector()[-1]]), qexp_representation = 'PD+II')

class WeilRepPositiveDefinite(WeilRep):
    def __init__(self, lift_qexp_representation = None):
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        if not lift_qexp_representation:
            lift_qexp_representation = 'PD+II'
        self.lift_qexp_representation = lift_qexp_representation

    def __add__(self, other, _flag = None):
        r"""
        Tensor product of Weil representations.

        If 'other' is a rescaled hyperbolic plane then we rearrange it so that 'other' goes in the first and last coordinates.
        """
        from .weilrep import WeilRep
        from .lorentz import RescaledHyperbolicPlane
        p = self.is_positive_definite()
        p2 = self._is_positive_definite_plus_II()
        try:
            if not _flag and isinstance(other, RescaledHyperbolicPlane):
                S = self.gram_matrix()
                n = S.nrows()
                N = other._N()
                S_new = matrix(ZZ, n + 2)
                for i in range(n):
                    for j in range(n):
                        S_new[i + 1, j + 1] = S[i, j]
                S_new[0, -1], S_new[-1, 0] = N, N
                if p:
                    return WeilRepPositiveDefinitePlusII(S_new, S, N, lift_qexp_representation = self.lift_qexp_representation)
                elif p2:
                    N1 = self._N()
                    return WeilRepPositiveDefinitePlus2II(S_new, self._pos_def_gram_matrix(), N1, N, lift_qexp_representation = self.lift_qexp_representation)
        except AttributeError:
            pass
        if isinstance(other, WeilRep):
            return WeilRep(block_diagonal_matrix([self.gram_matrix(), other.gram_matrix()], subdivide = False))
        return NotImplemented
    __radd__ = __add__

    def is_lorentzian(self):
        return False

    def is_lorentzian_plus_II(self):
        return False

    def is_positive_definite(self):
        return True

    def jacobi_forms(self):
        return JacobiForms(self.gram_matrix(), weilrep = self)

    def _pos_def_gram_matrix(self):
        return self.gram_matrix()



class WeilRepModularFormPositiveDefinite(WeilRepModularForm):
    r"""
    subclass of WeilRepModularForms for positive definite lattices. This adds methods for constructing Jacobi forms and theta lifts.
    """
    def __init__(self, k, f, w):
        self._WeilRepModularForm__weight = k
        self._WeilRepModularForm__fourier_expansions = f
        self._WeilRepModularForm__weilrep = w
        self._WeilRepModularForm__gram_matrix = w.gram_matrix()

    def convert_to_II(self):
        r"""
        Theta lifts for II(2, 2)

        The empty matrix is technically positive-definite but needs to be dealt with using the methods from the file lorentz.py
        Therefore we replace self with a modular form for the WeilRep II(1)
        """
        w = self.weilrep()
        if w.gram_matrix():
            raise ValueError
        from .lorentz import II
        w = II(1)
        X = [(vector([0, 0]), 0, self.fourier_expansion()[0][2])]
        return WeilRepModularForm(self.weight(), w.gram_matrix(), X, weilrep = w)

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
        K = self.base_ring()
        if e:
            Rb = LaurentPolynomialRing(K,list(var('w_%d' % i) for i in range(e) ))
        else:
            Rb = K
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
        elif e == 1:
            w, = Rb.gens()
            m = S[0,0] #twice the index
            eps = 2*self.is_symmetric()-1
            jf = [X[0][2][i] + sum(X[r%m][2][ceil(i - r*r / (2*m))]*(w**r + eps * w**QQ(-r)) for r in range(1,isqrt(2*(i-val)*m)+1)) for i in range(val, prec)]
        else:
            return JacobiForm(self.weight(), S, self.fourier_expansion()[0][2], weilrep = self.weilrep(), modform = self)
        return JacobiForm(self.weight() + e/2, S, q ** val * R(jf) + O(q**prec), weilrep = self.weilrep(), modform = self)

    def _weight_one_theta_lift_constant_term(self):
        r"""
        Compute the constant term in the additive theta lift to weight 1.

        This should not be called directly.

        The additive theta lift does not map cusp forms to cusp forms when the target has weight 1. (for subgroups of SL_2 this means weight 2). We try to compute the missing constant term here.

        NOTE: theta lifts of weight 1 only exist for lattices of signature (2, n) with n <= 4. They do not exist at all for positive-definite lattices by Skoruppa's theorem.

        OUTPUT: a rational number
        """
        if not self:
            return 0
        w = self.weilrep()
        if w.is_positive_definite():
            return 0
        extra_plane = w._is_positive_definite_plus_2II()
        nrows = w.gram_matrix().nrows()
        nrows2 = nrows - 2*extra_plane
        a = identity_matrix(nrows2)
        a[-1, 0] = -1
        if extra_plane:
            from .lorentz import II
            z = matrix([[1]])
            a = block_diagonal_matrix([z, a, z])
            b = matrix(nrows)
            b[0, 0], b[-1, 1] = 1, 1
            for i in range(nrows2):
                b[i + 1, (i + 2)%nrows] = 1
            a = a * b
            nrows = nrows2
        x = self.conjugate(a)
        while nrows > 1:
            x = x.theta_contraction()
            nrows -= 1
        n = x.gram_matrix()[-1, -1]
        N = 1
        if extra_plane:
            N = x.gram_matrix()[0, 1]
            A = matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            x = x.conjugate(A, w = WeilRep(matrix([[n]])) + II(N))
        x = x.theta_lift(constant_term_weight_one = False)
        f = x.fourier_expansion()
        m = ModularForms(Gamma0_constructor(-n*N // 2), 2, prec=x.precision()).echelon_basis()
        f -= sum(z.qexp() * f[z.qexp().valuation()] for z in m[1:])
        i = f.exponents()[0]
        return f[i] / m[0].qexp()[i]

    def theta_lift(self, prec=Infinity):
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

            sage: from weilrep import *
            sage: w = WeilRep([[2]])
            sage: (w + II(4)).modular_forms_basis(1/2, 15)[0].theta_lift()
            -1/4 - q - s - q^2 + (-r^-1 - 2 - r)*q*s - s^2 + (-2*r^-1 - 2*r)*q^2*s + (-2*r^-1 - 2*r)*q*s^2 - q^4 + (-r^-2 - 2 - r^2)*q^2*s^2 - s^4 + (-2)*q^5 + (-r^-2 - 2 - r^2)*q^4*s + (-r^-2 - 2 - r^2)*q*s^4 + (-2)*s^5 + (-2*r^-2 - 2*r^-1 - 2*r - 2*r^2)*q^5*s + (-2*r^-2 - 2*r^2)*q^4*s^2 + (-2*r^-2 - 2*r^2)*q^2*s^4 + (-2*r^-2 - 2*r^-1 - 2*r - 2*r^2)*q*s^5 + (-2*r^-3 - 2*r^-1 - 2*r - 2*r^3)*q^5*s^2 + (-2*r^-3 - 2*r^-1 - 2*r - 2*r^3)*q^2*s^5 + O(q, s)^8

            sage: from weilrep import *
            sage: m = OrthogonalModularForms(II(3))
            sage: L3 = m.lifts_basis(3, 10, cusp_forms = False)
            sage: L3[0]
            -q - 3*q^2 + 9*q*s - 9*q^3 + 27*q^2*s - 27*q*s^2 - 13*q^4 + 81*q^3*s - 81*q^2*s^2 + 9*q*s^3 - 24*q^5 + 117*q^4*s - 243*q^3*s^2 + 27*q^2*s^3 + 117*q*s^4 - 27*q^6 + 216*q^5*s - 351*q^4*s^2 + 81*q^3*s^3 + 351*q^2*s^4 - 216*q*s^5 - 50*q^7 + 243*q^6*s - 648*q^5*s^2 + 117*q^4*s^3 + 1053*q^3*s^4 - 648*q^2*s^5 - 27*q*s^6 - 51*q^8 + 450*q^7*s - 729*q^6*s^2 + 216*q^5*s^3 + 1521*q^4*s^4 - 1944*q^3*s^5 - 81*q^2*s^6 + 450*q*s^7 - 81*q^9 + 459*q^8*s - 1350*q^7*s^2 + 243*q^6*s^3 + 2808*q^5*s^4 - 2808*q^4*s^5 - 243*q^3*s^6 + 1350*q^2*s^7 - 459*q*s^8 + O(q, s)^10
        """
        prec0 = self.precision() + 1
        min_prec = isqrt(4 * prec0 + 4)
        prec = min(prec, min_prec)
        w = self.weilrep()
        S = w._pos_def_gram_matrix()
        coeffs = self.coefficients()
        S_inv = S.inverse()
        try:
            N2 = w._N2()
            if N2 == 1:
                K = QQ
                zeta = 1
            elif N2 == 2:
                K = QQ
                zeta = -1
            else:
                K, zeta = CyclotomicField(N2, var('mu%d'%N2)).objgen()
        except AttributeError:
            K = QQ
            N2 = 1
            zeta = 1
        if S:
            rb = LaurentPolynomialRing(K, list(var('r_%d' % i) for i in range(S.nrows()) ))
            z = rb.gens()[0]
        else:
            rb = K
        rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        t, = PowerSeriesRing(rb_x, 't').gens()
        k = self.weight()
        nrows = ZZ(S.nrows())
        wt = k + nrows / 2
        try:
            N = w._N()
        except AttributeError:
            N = 1
        if wt <= 0:
            return NotImplemented
        if nrows > 1:
            _, _, vs_matrix = pari(S_inv).qfminim(prec0 + prec0 + 1, flag = 2)
            vs_list = vs_matrix.sage().columns()
        elif nrows == 1:
            vs_list = [vector([n]) for n in range(1, isqrt(2 * prec0*S[0, 0]) + 1)]
        else:
            vs_list = []
        f = O(t ** prec)
        C = 0
        if wt == 1:
            C = self._weight_one_theta_lift_constant_term()
        bool_1 = w._is_positive_definite_plus_II()
        bool_2 = w._is_positive_definite_plus_2II()
        if bool_1:
            def update(g_n, sum_coeff):
                g_n = tuple([frac(c / (d*N))] + list(g_n)[:-1] + [frac(-a / (d*N)), g_n[-1]])
                sum_coeff += coeffs[g_n] * d_wt
                return sum_coeff
        elif bool_2:
            def update(g_n, sum_coeff):
                g_n_copy = list(g_n)[:-1]
                for i in srange(N2):
                    g_n = tuple([i / N2, frac(c/(d*N))] + g_n_copy + [frac(-a/(d*N)), 0, g_n[-1]])
                    sum_coeff += coeffs[g_n] * zeta ** ((i*d) % N2) * d_wt
                return sum_coeff
        else:
            def update(g_n, sum_coeff):
                sum_coeff += coeffs[g_n] * d_wt
                return sum_coeff
        try:
            N = w._N()
        except AttributeError:
            N = 1
        if wt % 2 == 0:
            if bool_2:
                y, = PolynomialRing(QQ, 'y').gens()
                bp = bernoulli_polynomial(y, wt)
                for i in srange(N2):
                    zeta_i = zeta ** i
                    c = coeffs[tuple([i / N2] + [0]*(nrows + 4))]
                    if c:
                        f -= c * sum([bp(j / N2) * (zeta_i ** j) for j in srange(1, N2 + 1)])
                f *= (N2 ** (wt - 1)) / (wt + wt)
            else:
                f_0 = coeffs[tuple([0]*(nrows + 1 + 2*Integer(bool_1)))]
                if f_0:
                    f -= f_0 * bernoulli(wt) / (wt + wt)
        for v in vs_list:
            j = next(j for j, w in enumerate(v) if w)
            g = S_inv * v
            v_norm = v * g / 2
            sqrt_v_norm = ceil(sqrt(v_norm))
            if nrows > 1:
                v_monomial = rb.monomial(*v)
            elif nrows == 1:
                v_monomial = z**v[0]
            else:
                v_monomial = 1
            a = 1
            while a < prec * N:
                c = 1
                while c < (prec- a) * N:
                    a_plus_c = a + c
                    n = Integer(a * c) / N - v_norm
                    if n >= 0:
                        sum_coeff_1 = 0
                        sum_coeff_2 = 0
                        for d in divisors(GCD([a, c] + list(v))):
                            d_wt = d ** (wt - 1)
                            g_n = tuple([frac(y) for y in g / d] + [n / (d * d)])
                            sum_coeff_1 = update(g_n, sum_coeff_1)
                            g_n_2 = tuple([frac(-y) for y in g/d] + [n / (d*d)])
                            sum_coeff_2 = update(g_n_2, sum_coeff_2)
                        f += t**(a_plus_c) * x**(c - a) * (sum_coeff_1 * v_monomial + sum_coeff_2 * ~v_monomial)
                    c += 1
                a += 1
        #now take b = zero vector
        for a in range(N * prec):
            for c in range(N * (prec - a)):
                n = Integer(a * c)/N
                a_plus_c = a + c
                if a_plus_c:
                    sum_coeff = 0
                    g = GCD(a, c)
                    try:
                        L = divisors(g)
                    except ValueError:
                        if a:
                            L = divisors(a)
                        else:
                            L = divisors(c)
                    for d in L:
                        d_wt = d ** (wt - 1)
                        g_n = tuple([0]*nrows + [n / (d * d)])
                        sum_coeff = update(g_n, sum_coeff)
                    if sum_coeff:
                        f += sum_coeff * t**(a_plus_c) * x ** (c - a)
        try:
            h = self.weilrep().lift_qexp_representation
        except(AttributeError, IndexError):
            h = None
        if wt % 2 and bool_2 and N2 >= 3:
            f /= sum(zeta**i - zeta**(-i) for i in range(1, (N2 + 1)//2))
        return OrthogonalModularForm(wt, self.weilrep(), f + C, scale = 1, weylvec = vector([0] * (nrows + 2)), qexp_representation = h)
    additive_lift = theta_lift
    gritsenko_lift = theta_lift
    maass_lift = theta_lift

    def _singular_theta_lift(self, prec = None):
        S = self.gram_matrix()
        nrows = Integer(S.nrows())
        k = self.weight() + nrows/2
        if k < 2:
            return NotImplemented
        #eulerian polynomials
        def a(n, m):
            return sum((-1)**k * binomial(n + 1, k) * (m + 1 - k)**n for k in range(m + 1))
        rpoly, t = PolynomialRing(QQ, 't').objgen()
        def A(n) :
            if n == 0:
                return 1
            return rpoly([a(n, k) for k in range(n)])
        E = t * A(k - 1)
        det_S = S.determinant()
        prec0 = self.precision()
        val = self.valuation()
        eps = self.is_symmetric()
        if eps == 0:
            eps = -1
        prec0val = prec0 - val
        if prec is None:
            prec = isqrt(4 * (prec0+val))
        else:
            prec = min(prec, isqrt(4 * (prec0+val)))
        S_inv = self.inverse_gram_matrix()
        coeffs = self.coefficients()
        rb = LaurentPolynomialRing(QQ, list(var('r_%d' % i) for i in range(nrows)) )
        frb = FractionField(rb)
        rb_x, x = LaurentPolynomialRing(frb, 'x').objgen()
        r, t = PowerSeriesRing(rb_x, 't', prec).objgen()
        rpoly, t0 = PolynomialRing(QQ, 't0').objgen()
        ds_dict = self.weilrep().ds_dict()
        if nrows > 1:
            _, _, vs_matrix = pari(S_inv).qfminim(prec0val + prec0val + 1, flag = 2)
            vs_list = vs_matrix.sage().columns()
        else:
            vs_list = [vector([n]) for n in range(1, isqrt(2*prec0*S[0, 0]) + 1)]
        vs_list.append(vector([0]*nrows))
        F = self.fourier_expansion()
        h = O(t ** prec)
        f = h
        if k % 2 == 0:
            try:
                f -= coeffs[tuple([0]*(nrows + 1))] *  bernoulli(k) / (2 * k)
            except KeyError:
                pass
        if nrows == 1:
            rb_zero = rb.gens()[0]
        for v in vs_list:
            g = S_inv * v
            v_norm = g * v / 2
            g_frac = map(frac, g)
            a_plus_c = -1
            while a_plus_c <= prec:
                a_plus_c += 1
                for c in srange(val, a_plus_c + 1):
                    a = a_plus_c - c
                    a_times_c = a * c
                    n = a_times_c - v_norm
                    if val <= n < prec0:
                        big_v = vector([a] + list(v) + [c])
                        if v and not (a or c):
                            j = next(j for j, v_j in enumerate(v) if v_j)
                            if v[j] > 0:
                                v = -v
                        big_tuple = tuple(list(g_frac) + [n])
                        try:
                            C = coeffs[big_tuple]
                            if C:
                                if nrows > 1:
                                    m = rb.monomial(*v)
                                else:
                                    m = rb_zero ** v[0]
                                if (a or c) and c >= 0:
                                    u = t**a_plus_c * x**(c - a)
                                    if v:
                                        f += C * (E.subs({t : u*m}) * (1 - u * m + h)**(-k) + eps * E.subs({t : u * ~m}) * (1 - u * ~m + h)**(-k))
                                    else:
                                        f += C  * E.subs({t : u}) * (1 - u + h)**(-k)
                                elif n and v:
                                    f += C * E.subs({t : m}) * frb(1 - m)**(-k)
                        except KeyError:
                            if n > prec:
                                prec = a_plus_c
                                h = O(t ** prec)
                                f += h
                            pass
        try:
            h = self.weilrep().lift_qexp_representation
        except(AttributeError, IndexError, TypeError):
            h = None
        return OrthogonalModularForm(k, self.weilrep(), f, scale = 1, weylvec = vector([0] * (nrows + 2)), qexp_representation = h)

    def weyl_vector(self):
        r"""
        Compute the Weyl vector in the Borcherds lift.
        """
        w = self.weilrep()
        K = w._pos_def_gram_matrix()
        nrows = K.nrows()
        bool_1 = w._is_positive_definite_plus_2II()
        bool_2 = w._is_positive_definite_plus_II()
        bool_3 = bool_1 or bool_2
        if bool_1:
            X = self.reduce_lattice(z = vector([1] + [0]*(nrows + 3)), z_prime = vector([0]*(nrows + 3) + [Integer(1)/w._N2()]))
            coeff = X.principal_part_coefficients()
            X = X.reduce_lattice(z = vector([1] + [0]*(nrows + 1)), z_prime = vector([0]*(nrows + 1) + [Integer(1)/w._N()]))
            nrows += 2
        elif bool_2:
            coeff = self.principal_part_coefficients()
            X = self.reduce_lattice(z = vector([1] + [0]*(nrows + 1)), z_prime = vector([0]*(nrows + 1) + [Integer(1)/w._N()]))
            nrows += 2
        else:
            X = self
            coeff = self.principal_part_coefficients()
        try:
            N = w._N()
        except AttributeError:
            N = Integer(1)
        K_inv = K.inverse()
        val = self.valuation()
        if X:
            theta_K = WeilRep(-K).theta_series(1 - val)
        else:
            theta_K = WeilRep(-K).zero(K.nrows()/2, 1-val)
        rho = vector([Integer(0)] * K.nrows())
        rho_z = Integer(0)
        negative = lambda v: next(s for s in v if s) < 0
        if K:
            try:
                _, _, vs_matrix = pari(K_inv).qfminim(1 - val, flag=2)
                vs_list = vs_matrix.sage().columns()
            except PariError:
                vs_list = [vector([n]) for n in srange(1, isqrt(2 * K[0, 0] * (-val)) + 1)]
        else:
            vs_list = []
        for v in vs_list:
            y = list(map(frac, K_inv * v))
            if negative(v):
                v *= -1
            v_norm = -v * K_inv * v / 2
            if bool_3:
                rho += coeff[tuple([0] + y + [0, v_norm])] * v
                for i in srange(N):
                    j = i / N
                    c = coeff[tuple([j] + y + [0, v_norm])]
                    rho_z += c * (j * (j - 1) + Integer(1) / 6) / 2
            else:
                c = coeff[tuple(y + [v_norm])]
                rho += c * v
                rho_z += c * Integer(1) / 12
        for i in srange(N):
            j = i / N
            c = coeff[tuple([j] + [0]*nrows)]
            rho_z += c * (j * (j - 1) + Integer(1) / 6) / 4
        e2 = eisenstein_series_qexp(2, 1 - val)
        rho_z_prime = -((X &theta_K) * e2)[0]
        return vector([N*rho_z] + list(rho/2) + [rho_z_prime])

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

            sage: from weilrep import *
            sage: w = WeilRep([[2, -1], [-1, 2]]) + II(3)
            sage: m = OrthogonalModularForms(w)
            sage: X = m.borcherds_input_Qbasis(1, 5)
            sage: X[0].borcherds_lift()
            (r_0 - r_1 - r_0*r_1^-1 + r_0^-1*r_1 + r_1^-1 - r_0^-1)*q*s + (-6*r_0 + 6*r_1 + 6*r_0*r_1^-1 - 6*r_0^-1*r_1 - 6*r_1^-1 + 6*r_0^-1)*q^2*s + (-6*r_0 + 6*r_1 + 6*r_0*r_1^-1 - 6*r_0^-1*r_1 - 6*r_1^-1 + 6*r_0^-1)*q*s^2 + (9*r_0 - 9*r_1 - 9*r_0*r_1^-1 + 9*r_0^-1*r_1 + 9*r_1^-1 - 9*r_0^-1)*q^3*s + (3*r_0^2 - 3*r_1^2 + 12*r_0 - 12*r_1 - 3*r_0^2*r_1^-2 - 12*r_0*r_1^-1 + 12*r_0^-1*r_1 + 3*r_0^-2*r_1^2 + 12*r_1^-1 - 12*r_0^-1 + 3*r_1^-2 - 3*r_0^-2)*q^2*s^2 + (9*r_0 - 9*r_1 - 9*r_0*r_1^-1 + 9*r_0^-1*r_1 + 9*r_1^-1 - 9*r_0^-1)*q*s^3 + (-r_0^2 + r_1^2 + 12*r_0 - 12*r_1 + r_0^2*r_1^-2 - 12*r_0*r_1^-1 + 12*r_0^-1*r_1 - r_0^-2*r_1^2 + 12*r_1^-1 - 12*r_0^-1 - r_1^-2 + r_0^-2)*q^4*s + (-9*r_0^2 + 9*r_1^2 + 18*r_0 - 18*r_1 + 9*r_0^2*r_1^-2 - 18*r_0*r_1^-1 + 18*r_0^-1*r_1 - 9*r_0^-2*r_1^2 + 18*r_1^-1 - 18*r_0^-1 - 9*r_1^-2 + 9*r_0^-2)*q^3*s^2 + (-9*r_0^2 + 9*r_1^2 + 18*r_0 - 18*r_1 + 9*r_0^2*r_1^-2 - 18*r_0*r_1^-1 + 18*r_0^-1*r_1 - 9*r_0^-2*r_1^2 + 18*r_1^-1 - 18*r_0^-1 - 9*r_1^-2 + 9*r_0^-2)*q^2*s^3 + (-r_0^2 + r_1^2 + 12*r_0 - 12*r_1 + r_0^2*r_1^-2 - 12*r_0*r_1^-1 + 12*r_0^-1*r_1 - r_0^-2*r_1^2 + 12*r_1^-1 - 12*r_0^-1 - r_1^-2 + r_0^-2)*q*s^4 + O(q, s)^6
        """
        prec0 = self.precision()
        val = self.valuation()
        prec0val = prec0 - val
        if prec is None:
            prec = isqrt(4 * (prec0+val))
        else:
            prec = min(prec, isqrt(4 * (prec0+val)))
        weilrep = self.weilrep()
        S = weilrep._pos_def_gram_matrix()
        S_inv = S.inverse()
        det_S = S.determinant()
        nrows = Integer(S.nrows())
        if not self.weight() == -nrows/2:
            raise ValueError('Incorrect input weight')
        w = self.weyl_vector()
        d = ZZ(denominator(w))
        weyl_v = d * w
        prec *= d
        coeffs = self.coefficients()
        weight = coeffs[tuple([0]*(self.gram_matrix().nrows() + 1))] / 2
        try:
            N2 = weilrep._N2()
            if N2 == 1:
                K = QQ
                zeta = 1
            elif N2 == 2:
                K = QQ
                zeta = -1
            else:
                K, zeta = CyclotomicField(N2, var('mu%d'%N2)).objgen()
        except AttributeError:
            K = QQ
            N2 = 1
            zeta = 1
        if S:
            rb = LaurentPolynomialRing(K, list(var('r_%d' % i) for i in range(S.nrows()) ))
            rb_zero = rb.gens()[0]
        else:
            rb = K
        try:
            N = weilrep._N()
        except AttributeError:
            N = 1
        prec0 *= N
        prec0val *= N
        bool_1 = weilrep._is_positive_definite_plus_II()
        bool_2 = weilrep._is_positive_definite_plus_2II()
        rb_x, x = LaurentPolynomialRing(rb, 'x').objgen()
        r, t = PowerSeriesRing(rb_x, 't', prec).objgen()
        rpoly, t0 = PolynomialRing(K, 't0').objgen()
        ds_dict = weilrep.ds_dict()
        if nrows > 1:
            _, _, vs_matrix = pari(S_inv).qfminim(prec0val + prec0val + 1, flag = 2)
            vs_list = vs_matrix.sage().columns()
        elif nrows == 1:
            vs_list = [vector([n]) for n in range(1, isqrt(2*prec0*S[0, 0]) + 1)]
        else:
            vs_list = []
        vs_list.append(vector([0]*nrows))
        F = self.fourier_expansion()
        h = O(t ** prec)
        log_f = h
        a_plus_c = -1
        excluded_vectors = set()
        corrector = 1
        Nval = N * val
        excluded_vectors_2 = set()
        def update_aux(a, c, n, _g, mu, log_f, i=0):
            exponent = coeffs[_g]
            if exponent:
                if nrows > 1:
                    m = rb.monomial(*d*v)
                elif nrows == 1:
                    m = rb_zero ** (d * v[0])
                else:
                    m = 1
                if (a or c) and c >= 0:
                    u = t ** (d * a_plus_c) * x **( d * (c - a))
                    log_f += exponent * log(1 - mu * u * m + h)
                elif n and big_v not in excluded_vectors_2:
                    nonlocal corrector
                    p = rpoly(1)
                    bound = isqrt(val / n) + 1
                    for k in range(1, bound):
                        if bool_2:
                            for i in range(N2):
                                exponent_k = coeffs[tuple([i / N2] + [frac(y) for y in k * vector(_g[1:-1])] + [n * k * k])]
                                if exponent_k:
                                    p *= (1 - (zeta**i * t0) ** k) ** exponent_k
                        else:
                            exponent_k = coeffs[tuple([frac(y) for y in k * vector(_g[:-1])] + [n * k * k])]
                            if exponent_k:
                                p *= (1 - t0**k)**exponent_k
                        excluded_vectors.add(tuple(k * y for y in big_v))
                        excluded_vectors_2.add(big_v)
                    if c >= 0:
                        corrector *= p.subs({t0 : m})
                    elif p != 1:
                        deg_p = p.degree()
                        corrector *= (h + sum(p * t ** (d * (a_plus_c * j - c * deg_p)) * x ** (d * (a * j + c * (deg_p - j))) * m ** (d*j) for j, p in enumerate(list(p))))
                        weyl_v[0] += c * d * deg_p
            return log_f
        if bool_1:
            def update(a, c, n, _g, log_f):
                _g = tuple([frac(c / N)] + _g + [frac(-a / N), n])
                return update_aux(a, c, n, _g, 1, log_f)
        elif bool_2:
            def update(a, c, n, _g, log_f):
                mu = 1
                for i in srange(N2):
                    _g0 = tuple([i / N2, frac(c / N)] + _g + [frac(-a / N), 0, n])
                    log_f = update_aux(a, c, n, _g0, mu, log_f, i=i)
                    mu *= zeta
                return log_f
        else:
            def update(a, c, n, _g, log_f):
                return update_aux(a, c, n, tuple(_g + [n]), 1, log_f)
        for v in vs_list:
            g = S_inv * v
            v_norm = g * v / 2
            g_frac = [frac(-y) for y in g]
            g_frac_2 = [frac(y) for y in g]
            a_plus_c = Nval - 1
            if any(v):
                j = next(j for j, v_j in enumerate(v) if v_j)
                if v[j] > 0:
                    v = -v
            while a_plus_c <= prec / d:
                a_plus_c += 1
                for c in srange(Nval, a_plus_c + 1):
                    a = a_plus_c - c
                    a_times_c = a * c
                    n = a_times_c / N - v_norm
                    if val <= n < prec0 and (c >= 0 or a > 0):
                        big_v = tuple([a] + list(v) + [c])
                        if big_v not in excluded_vectors:
                            log_f = update(a, c, n, g_frac, log_f)
                            if any(v) and (a or c):
                                v = -v
                                big_v, tmp = tuple([a] + list(v) + [c]), big_v
                                if big_v not in excluded_vectors:
                                    log_f = update(a, c, n, g_frac_2, log_f)
                                v = -v
                                big_v = tmp
        if nrows > 1:
            weyl_vector_term = (t ** (weyl_v[0] + weyl_v[-1])) * (x ** (weyl_v[0] - weyl_v[-1])) * rb.monomial(*weyl_v[1:-1])
            weyl_vector_term_inverse = (t ** -(weyl_v[0] + weyl_v[-1])) * (x ** -(weyl_v[0] - weyl_v[-1])) * rb.monomial(*(-weyl_v[1:-1]))
        elif nrows == 1:
            weyl_vector_term = (t ** (weyl_v[0] + weyl_v[-1])) * (x ** (weyl_v[0] - weyl_v[-1])) * rb_zero ** weyl_v[1]
            weyl_vector_term_inverse = (t ** -(weyl_v[0] + weyl_v[-1])) * (x ** -(weyl_v[0] - weyl_v[-1])) * rb_zero ** -weyl_v[1]
        else:
            weyl_vector_term = (t ** (weyl_v[0] + weyl_v[-1])) * (x ** (weyl_v[0] - weyl_v[-1]))
            weyl_vector_term_inverse = (t ** -(weyl_v[0] + weyl_v[-1])) * (x ** -(weyl_v[0] - weyl_v[-1]))
        try:
            h = self.weilrep().lift_qexp_representation
        except(AttributeError, IndexError, TypeError):
            h = None
        try:
            f = exp(log_f)
            if bool_2 and N2 > 1:
                C = Integer(1)
                for i in srange(1, N2 // 2):
                    c = Integer(coeffs[tuple([i / N2] + [0] * (nrows + 4))])
                    C *= (1 - zeta**i)**c
                c = Integer(coeffs[tuple([Integer(1) / 2] + [0] * (nrows + 4))])
                C *= Integer(2)**Integer(c / 2)
                f *= C
            X = OrthogonalModularForm(weight, self.weilrep(), f * r(corrector) * weyl_vector_term, scale = d, weylvec = weyl_v / d, qexp_representation = h)
            try:
                X._OrthogonalModularForm__inverse = f**(-1) * weyl_vector_term_inverse / FractionField(rb)(rb(corrector))
            except (TypeError, ValueError):
                pass
            return X
        except TypeError:
            raise RuntimeError('I caught a TypeError. This probably means you are trying to compute a Borcherds product that is not holomorphic.')

class WeilRepModularFormPositiveDefiniteWithCharacter(WeilRepModularFormWithCharacter, WeilRepModularFormPositiveDefinite):
    r"""
    Adds the Jacobi form corresponding to a vector-valued modular form with additional character for a positive-definite lattice.
    """
    def jacobi_form(self, *args, **kwargs):
        from .weilrep_modular_forms_class import smf_eta
        chi = self.character()
        k = chi._k()
        psi = smf_eta() ** (24 - k)
        f = (self.__mul__(psi)).jacobi_form(*args, **kwargs)
        return f / psi

class WeilRepPositiveDefinitePlusII(WeilRepPositiveDefinite):

    def __init__(self, S, pos_def_S, N, lift_qexp_representation = None):
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self.lift_qexp_representation = lift_qexp_representation
        self.__positive_definite_gram_matrix = pos_def_S
        self.__N = N

    def _N(self):
        return self.__N

    def is_lorentzian(self):
        return True

    def is_positive_definite(self):
        return False

    def _is_positive_definite_plus_II(self):
        return True

    def _is_positive_definite_plus_2II(self):
        return False

    def _lifts_have_fourier_jacobi_expansion(self):
        return True

    def nvars(self):
        return Integer(self.gram_matrix().nrows())

    def _lorentz_gram_matrix(self):
        return self.gram_matrix()

    def _pos_def_gram_matrix(self):
        return self.__positive_definite_gram_matrix

class WeilRepPositiveDefinitePlus2II(WeilRepPositiveDefinite):

    def __init__(self, S, pos_def_S, N1, N2, lift_qexp_representation = None):
        #S should be a Lorentzian lattice in which the bottom-right entry is negative!!
        self._WeilRep__gram_matrix = S
        self._WeilRep__quadratic_form = QuadraticForm(S)
        self._WeilRep__eisenstein = {}
        self._WeilRep__cusp_forms_basis = {}
        self._WeilRep__modular_forms_basis = {}
        self.lift_qexp_representation = lift_qexp_representation
        self.__positive_definite_gram_matrix = pos_def_S
        self.__N = N1
        self.__N2 = N2

    def _N(self):
        return self.__N

    def _N2(self):
        return self.__N2

    def is_positive_definite(self):
        return False

    def is_lorentzian_plus_II(self):
        return True

    def _is_positive_definite_plus_II(self):
        return False

    def _is_positive_definite_plus_2II(self):
        return True

    def _lifts_have_fourier_jacobi_expansion(self):
        return True

    def _lorentz_gram_matrix(self):
        return self.gram_matrix()[1:-1, 1:-1]

    def nvars(self):
        return Integer(self.gram_matrix().nrows()) - 2

    def _pos_def_gram_matrix(self):
        return self.__positive_definite_gram_matrix