r"""

Sage code for Fourier expansions of vector-valued modular forms

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

from sage.modular.modform.element import is_ModularFormElement
from sage.structure.element import is_Matrix

class WeilRepModularForm(object):

    r"""
    The WeilRepModularForm class represents vector-valued modular forms which transform with the dual Weil representation.

    INPUT:

    A WeilRepModularForm is constructed by calling WeilRepModularForm(k, S, X), where
    - ``k`` -- a weight (half integer)
    - ``S`` -- a symmetric integral matrix with even diagonal and nonzero determinant (this is not checked)
    - ``X`` -- a list of lists: X = [[g_0, n_0, f_0], [g_1, n_1, f_1], ...] where each element [g, n, f] consists of
        - ``g`` -- a vector of length = size of S
        - ``n`` -- a rational number for which n + g*S*g/2 is integral
        - ``f`` -- a power series with rational Fourier coefficients
        The vectors g are assumed to be sorted in the order they appear in WeilRep(S).ds(). (We do not check this!)
    - ``weilrep`` -- optional (default None). This should be set if the VVMF was created by a WeilRep instance.

    OUTPUT: WeilRepModularForm(k, S, [[g, n, f], ...]) represents a vector-valued modular form of weight k for the Weil representation attached to the Gram matrix S, whose Fourier expansion takes the form \sum_g q^n f(q) e_g.

    """

    def __init__(self, weight, gram_matrix, fourier_expansions, weilrep = None):
        self.__weight = weight
        self.__gram_matrix = gram_matrix
        self.__fourier_expansions = fourier_expansions
        if weilrep:
            self.__weilrep = weilrep
            if weilrep.is_positive_definite():
                self.__class__ = WeilRepModularFormPositiveDefinite #in the 'lifts.sage' file.
            elif weilrep.is_lorentzian():
                self.__class__ = WeilRepModularFormLorentzian #in the 'lorentzian.sage' file.

    def __repr__(self): #represent as a list of pairs (g, f) where g is in the discriminant group and f is a q-series with fractional exponents
        try:
            return self.__qexp_string
        except AttributeError:
            X = self.__fourier_expansions
            def a(x):
                def b(y):
                    y = y.string[slice(*y.span())]
                    if y[0] != 'q':
                        return '%sq^(%s) '%([y[:-1]+'*',''][y == '1 '], x)
                    try:
                        return 'q^(%s)'%(QQ(y[2:]) + x)
                    except TypeError:
                        return 'q^(%s)'%(QQ(1 + x))
                return b
            self.__qexp_string = '\n'.join(['[%s, %s]'%(x[0], sub(r'(q(\^-?\d+)?)|((?<!\^)\d+\s)', a(x[1]), str(x[2]))) if x[1] else '[%s, %s]'%(x[0], x[2]) for x in X])
            return self.__qexp_string

    def _latex_(self):
        X = self.fourier_expansion()
        def a(obj):
            y = obj.string[slice(*obj.span())]
            if y[0] != 'q' and y != '*':
                if x[1]:
                    return '%sq^{%s} '%([y[:-1]+'*',''][y == '1 '], x[1])
                return y
            try:
                return 'q^{%s}'%(QQ(y[2:]) + x[1])
            except TypeError:
                if y == '*':
                    return ''
                return 'q^{%s}'%(1 + x[1])
        return '&' + ' + &'.join(['\\left(%s\\right)\\mathfrak{e}_{%s}\\\\'%(sub(r'q(\^-?\d+)?|\*|((?<!\^)\d+\s)', a, str(x[2])), x[0]) for x in X])[:-2]

    ## basic attributes

    def denominator(self):
        r"""
        Return the denominator of self's Fourier coefficients.
        """
        sturm_bound = self.weight() / 12
        val = self.valuation()
        return denominator(self.coefficient_vector(starting_from = val, ending_with = max(1 / 24, sturm_bound)))

    def fourier_expansion(self):
        r"""
        Return the Fourier expansion.
        """
        return self.__fourier_expansions

    def gram_matrix(self):
        r"""
        Return the Gram matrix.
        """
        return self.__gram_matrix

    def inverse_gram_matrix(self):
        r"""
        Return the inverse Gram matrix.
        """
        try:
            return self.__inverse_gram_matrix
        except AttributeError:
            self.__inverse_gram_matrix = self.__gram_matrix.inverse()
            return self.__inverse_gram_matrix

    def __nonzero__(self):
        return any(x[2] for x in self.fourier_expansion())

    __bool__ = __nonzero__

    def weight(self):
        r"""
        Return the weight
        """
        return self.__weight

    def weilrep(self):
        r"""
        Returns the Weil Representation that produced this modular form.
        """
        try:
            return self.__weilrep
        except AttributeError:
            self.__weilrep = WeilRep(self.gram_matrix())
            return self.__weilrep

    def precision(self):
        r"""
        Returns the precision to which our Fourier expansion is given (rounded down).
        """
        try:
            return self.__precision
        except AttributeError:
            X = self.fourier_expansion()
            self.__precision = min([floor(x[2].prec() + x[1]) for x in X])
            return self.__precision

    def reduce_precision(self, prec, in_place = False):
        r"""
        Reduce self's precision.
        """
        prec = floor(prec)
        R.<q> = PowerSeriesRing(QQ)
        X = [(x[0], x[1], x[2] + O(q**(prec - floor(x[1])))) for x in self.__fourier_expansions]
        if in_place:
            self.__fourier_expansions = X
            self.__precision = prec
        else:
            return WeilRepModularForm(self.__weight, self.__gram_matrix, X, weilrep = self.weilrep())

    def principal_part(self):
        r"""
        Return the principal part of self's Fourier expansion as a string.

        EXAMPLES::

            sage: X = WeilRep(matrix([[2, 1],[1, 4]])).nearly_holomorphic_modular_forms_basis(-1, 1, 3)
            sage: X[1].principal_part()
            '28*e_(0, 0) + -5*q^(-1/7)e_(1/7, 5/7) + -5*q^(-1/7)e_(6/7, 2/7) + q^(-4/7)e_(5/7, 4/7) + q^(-4/7)e_(2/7, 3/7)'
        """
        coeffs = self.coefficients()
        w = self.weilrep()
        ds_dict = w.ds_dict()
        norm_dict = w.norm_dict()
        ds = w.ds()
        sorted_ds = sorted(ds, key = lambda x: -norm_dict[tuple(x)])
        val = self.valuation()
        len_ds = len(ds)
        L = [None]*(len_ds * val)
        try:
            C0 = coeffs[tuple([0] * (len(ds[0]) + 1))]
        except KeyError:
            C0 = 0
        s = str(C0)+'*e_%s'%ds[0]
        for i, g in enumerate(sorted_ds):
            j = norm_dict[tuple(g)]
            for n in range(1 - val):
                if j or n:
                    try:
                        C = coeffs[tuple(list(g) + [j - n])]
                        if C != 1:
                            s += ' + %s*q^(%s)e_%s'%(C, (j - n), g)
                        else:
                            s += ' + q^(%s)e_%s'%((j - n), g)
                    except KeyError:
                        pass
        return s

    def valuation(self, exact = False):
        r"""
        Return the lowest exponent in our Fourier expansion with a nonzero coefficient (rounded down).

        INPUT:
        - ``exact`` -- boolean (default False). If True then we do not round down.
        """
        try:
            if exact:
                return self.__exact_valuation
            return self.__valuation
        except AttributeError:
            X = self.fourier_expansion()
            try:
                self.__exact_valuation = min([x[2].valuation() + x[1] for x in X if x[2]])
            except ValueError: #probably trying to take valuation of 0
                self.__exact_valuation = 0 #for want of a better value
            self.__valuation = floor(self.__exact_valuation)
            if exact:
                return self.__exact_valuation
            return self.__valuation

    def is_symmetric(self):
        r"""
        Determines whether the components f_{\gamma} in our Fourier expansion satisfy f_{\gamma} = f_{-\gamma} or f_{\gamma} = -f_{\gamma}.
        This can be read off the weight.
        """
        try:
            return self.__is_symmetric
        except AttributeError:
            self.__is_symmetric = [1,None,0,None][(ZZ(2*self.weight()) + self.weilrep().signature()) % 4]
            return self.__is_symmetric

    def coefficient_vector(self, starting_from=None, ending_with=None, G = None, set_v = None):
        r"""
        Return self's Fourier coefficients as a vector.

        INPUT:
        - ``starting_from`` -- the minimal exponent whose coefficient is included in the vector (default self's valuation)
        - ``ending_with`` -- the maximal exponent whose coefficient is included in the vector (default self's precision)
        - ``set_v`` -- vector (default None). If a vector v is given then we *set* the coefficient vector of self to v. (this should only be used internally)

        OUTPUT: a vector of rational numbers

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).eisenstein_series(3, 5).coefficient_vector()
            (1, 27, 72, 216, 270, 459, 720, 1080, 936, 1350)

            sage: w = WeilRep(matrix([[2,1],[1,2]]))
            sage: w.cusp_forms_basis(9,5)[0].coefficient_vector()
            (0, 1, -6, -10, 90, 8, -540, 310, 1488, -1750)

        """
        if not set_v is None:
            self.__coefficient_vector = set_v
            return None
        if not (starting_from or ending_with):
            try:
                return self.__coefficient_vector
            except:
                pass
        elif (not starting_from) and (ending_with == self.weight() / 12):
            try:
                return self.__coefficient_vector_sturm_bound
            except:
                pass
        if G is None:
            w = self.weilrep()
            G = w.rds()
        symm = self.is_symmetric()
        prec = self.precision()
        X = [x for x in self.fourier_expansion()]
        X.sort(key = lambda x: x[1])
        Y = []
        if ending_with is None:
            ending_with = prec + 1
        if ending_with > prec + 1:
            raise ValueError('Insufficient precision')
        if starting_from is None:
            starting_from = self.valuation()
        for n in range(floor(starting_from),ceil(ending_with)+1):
            for x in X:
                if starting_from <= n + x[1] <= ending_with:
                    if (x[0] in G) and (symm or x[0].denominator() > 2):
                        try:
                            Y.append(x[2][n])
                        except:
                            pass
        v = vector(Y)
        if not (starting_from or ending_with):
            self.__coefficient_vector = v
        elif (not starting_from) and (ending_with == self.weight() / 12):
            self.__coefficient_vector_sturm_bound = v
        return v

    def coefficients(self):#returns a dictionary of self's Fourier coefficients
        return {tuple(list(x[0])+[n+x[1]]):x[2][n] for x in self.fourier_expansion() for n in x[2].exponents()}

    def components(self):
        r"""
        Return the components of our Fourier expansion as a dictionary.

        NOTE: this requires the component vectors to be passed as tuples and to be reduced mod ZZ, i.e. g = (g_0,...,g_d) with 0 <= g_i < 1

        EXAMPLES::

            sage: w = WeilRep(-matrix([[2,1],[1,2]])).theta_series(5)
            sage: w.components()[0,0]
            1 + 6*q + 6*q^3 + 6*q^4 + O(q^5)

        """
        return {tuple(x[0]):x[2] for x in self.fourier_expansion()}

    ## arithmetic operations

    def __add__(self, other):
        if not other:
            return self
        if not self.gram_matrix() == other.gram_matrix():
            raise ValueError('Incompatible Gram matrices')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        X = self.fourier_expansion()
        Y = other.fourier_expansion()
        X_plus_Y = WeilRepModularForm(self.weight(), self.gram_matrix(), [(x[0],x[1],x[2]+Y[i][2]) for i,x in enumerate(X)], weilrep = self.weilrep())
        try: #keep coefficient vectors if we've done this already
            v2 = other.__dict__['_WeilRepModularForm__coefficient_vector']
            v1 = self.__coefficient_vector
            X_plus_Y.coefficient_vector(setv = v1 + v2)
        finally:
            return X_plus_Y

    __radd__ = __add__

    def __neg__(self):
        X = self.fourier_expansion()
        neg_X = WeilRepModularForm(self.weight(), self.gram_matrix(), [(x[0],x[1],-x[2]) for x in X], weilrep = self.weilrep())
        try: #keep coefficient vectors
            negX.coefficient_vector(setv = -self.__coefficient_vector)
        finally:
            return neg_X

    def __sub__(self, other):
        if not other:
            return self
        if not self.gram_matrix() == other.gram_matrix():
            raise ValueError('Incompatible Gram matrices')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        X = self.fourier_expansion()
        Y = other.fourier_expansion()
        X_minus_Y = WeilRepModularForm(self.weight(), self.gram_matrix(), [(x[0],x[1],x[2]-Y[i][2]) for i,x in enumerate(X)], weilrep = self.weilrep())
        try: #keep coefficient vectors
            v2 = other.__dict__['_WeilRepModularForm__coefficient_vector']
            v1 = self.__coefficient_vector
            X_minus_Y.coefficient_vector(setv = v1 - v2)
        finally:
            return X_minus_Y

    def __mul__(self, other): #tensor product!
        r"""
        Tensor multiplication of WeilRepModularForms.

        If ``other`` is a WeilRepModularForm then multiplication should be interpreted as the tensor product. This corresponds to a modular form for the Weil representation attached to the direct sum of the underlying lattices. Otherwise we multiply componentwise

        EXAMPLES::

            sage: w1 = WeilRep(matrix([[2,1],[1,2]]))
            sage: w2 = WeilRep(matrix([[-4]]))
            sage: e1 = w1.eisenstein_series(3, 5)
            sage: theta = w2.theta_series(5)
            sage: e1 * theta
            [(0, 0, 0), 1 + 72*q + 272*q^2 + 864*q^3 + 1476*q^4 + O(q^5)]
            [(1/3, 1/3, 3/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^(139/24))]
            [(2/3, 2/3, 1/2), 54*q^(7/6) + 432*q^(13/6) + 918*q^(19/6) + 2160*q^(25/6) + 2754*q^(31/6) + O(q^(37/6))]
            [(0, 0, 1/4), q^(1/8) + 73*q^(9/8) + 342*q^(17/8) + 991*q^(25/8) + 1728*q^(33/8) + O(q^(41/8))]
            [(1/3, 1/3, 0), 27*q^(2/3) + 216*q^(5/3) + 513*q^(8/3) + 1512*q^(11/3) + 2268*q^(14/3) + O(q^(17/3))]
            [(2/3, 2/3, 3/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^(139/24))]
            [(0, 0, 1/2), 2*q^(1/2) + 144*q^(3/2) + 540*q^(5/2) + 1440*q^(7/2) + 1874*q^(9/2) + O(q^(11/2))]
            [(1/3, 1/3, 1/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^(139/24))]
            [(2/3, 2/3, 0), 27*q^(2/3) + 216*q^(5/3) + 513*q^(8/3) + 1512*q^(11/3) + 2268*q^(14/3) + O(q^(17/3))]
            [(0, 0, 3/4), q^(1/8) + 73*q^(9/8) + 342*q^(17/8) + 991*q^(25/8) + 1728*q^(33/8) + O(q^(41/8))]
            [(1/3, 1/3, 1/2), 54*q^(7/6) + 432*q^(13/6) + 918*q^(19/6) + 2160*q^(25/6) + 2754*q^(31/6) + O(q^(37/6))]
            [(2/3, 2/3, 1/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^(139/24))]

        """

        if isinstance(other, WeilRepModularForm):
            S1 = self.gram_matrix()
            S2 = other.gram_matrix()
            if not S2:
                return WeilRepModularForm(self.weight() + other.weight(), S1, [(x[0], x[1], x[2]*other.fourier_expansion()[0][2]) for x in self.fourier_expansion()], weilrep = self.weilrep())
            else:
                S = block_diagonal_matrix([S1,S2])
                w = WeilRep(S)
                _ds_dict = w.ds_dict()
                X = [None]*w.discriminant()
                R.<q> = PowerSeriesRing(QQ)
                for x1 in self.fourier_expansion():
                    u1 = x1[1]
                    for x2 in other.fourier_expansion():
                        u2 = x2[1]
                        g = tuple(list(x1[0]) + list(x2[0]))
                        i = _ds_dict[g]
                        s = u1+u2
                        f = x1[2] * x2[2]
                        if s <= -1:
                            X[i] = vector(g), s+1, f/q
                        else:
                            X[i] = vector(g), s, f
                return WeilRepModularForm(self.__weight+other.weight(), S, X, weilrep = w)
        elif is_ModularFormElement(other):
            if not other.level() == 1:
                raise NotImplementedError
            X = self.fourier_expansion()
            return WeilRepModularForm(self.__weight + other.weight(), self.gram_matrix(), [(x[0], x[1], x[2]*other.qexp()) for x in X], weilrep = self.weilrep())
        elif other in QQ:
            X = self.fourier_expansion()
            X_times_other = WeilRepModularForm(self.__weight, self.gram_matrix(), [(x[0], x[1], x[2]*other) for x in X], weilrep = self.weilrep())
            try:
                v = X.__coefficient_vector
                X_times_other.coefficient_vector(set_v = v * other)
            finally:
                return X_times_other
        else:
            raise TypeError('Cannot multiply these objects')

    __rmul__ = __mul__

    def __truediv__(self, other):
        X = self.fourier_expansion()
        if is_ModularFormElement(other):
            if not other.level() == 1:
                raise NotImplementedError
            return WeilRepModularForm(self.weight() - other.weight(), self.gram_matrix(), [(x[0], x[1], x[2]/other.qexp()) for x in X], weilrep = self.weilrep())
        elif other in QQ:
            X_div_other = WeilRepModularForm(self.weight(), self.gram_matrix(), [(x[0], x[1], x[2]/other) for x in X], weilrep = self.weilrep())
            try:
                v = self.__coefficient_vector
                X_div_other.coefficient_vector(set_v = v / other)
            finally:
                return WeilRepModularForm(self.weight(), self.gram_matrix(), [(x[0],x[1],x[2]/other) for x in X], weilrep = self.weilrep())
        else:
            raise TypeError('Cannot divide these objects')

    __div__ = __truediv__

    def __and__(self, other):
        r"""
        Apply a trace map.

        The & operator takes two modular forms for Weil representations that are duals of one another and traces them to a scalar-valued modular form.

        OUTPUT: a power series in 'q'

        EXAMPLE::

            sage: e1 = WeilRep(matrix([[2, 1], [1, 2]])).eisenstein_series(3, 5)
            sage: e2 = WeilRep(matrix([[-2, -1], [-1, -2]])).eisenstein_series(5, 5)
            sage: e1 & e2
            1 + 480*q + 61920*q^2 + 1050240*q^3 + 7926240*q^4 + O(q^5)
        """
        if isinstance(other, WeilRepModularForm):
            w = self.weilrep()
            minus_w = other.weilrep()
            if not w.dual() == minus_w:
                raise NotImplementedError
            f1 = self.fourier_expansion()
            f2 = other.fourier_expansion()
            dsdict = w.ds_dict()
            r.<q> = LaurentSeriesRing(QQ)
            h = 0 + O(q ** self.precision())
            for g, o, x in f2:
                j = dsdict[tuple(g)]
                g1, o1, y = f1[j]
                h += r(q^(o + o1) * x * y)
            return h
        raise NotImplementedError

    def __eq__(self,other):
        if isinstance(other,WeilRepModularForm):
            return self.fourier_expansion() == other.fourier_expansion()
        return False

    def __pow__(self, other):
        if other in ZZ and other >= 1:
            if other == 1:
                return self
            elif other == 2:
                return self * self
            else:
                nhalf = other // 2
                return (self ** nhalf) * (self ** (other - nhalf))
        else:
            raise NotImplementedError

    ## other operations

    def bol(self):
        r"""
        Apply the Bol operator.

        This applies the operator (d / dtau)^(1-k), where k is self's weight.

        NOTE: this is defined only when the weight is an integer <= 1.

        EXAMPLES::

            sage: w = WeilRep(matrix([[2,1],[1,2]]))
            sage: j = w.nearly_holomorphic_modular_forms_basis(-1, 1, 5)[0]
            sage: j.bol()
            [(0, 0), 378*q + 14256*q^2 + 200232*q^3 + 1776384*q^4 + O(q^5)]
            [(2/3, 2/3), 1/9*q^(-1/3) - 328/9*q^(2/3) - 22300/9*q^(5/3) - 132992/3*q^(8/3) - 1336324/3*q^(11/3) - 28989968/9*q^(14/3) + O(q^(17/3))]
            [(1/3, 1/3), 1/9*q^(-1/3) - 328/9*q^(2/3) - 22300/9*q^(5/3) - 132992/3*q^(8/3) - 1336324/3*q^(11/3) - 28989968/9*q^(14/3) + O(q^(17/3))]

        """
        k = self.weight()
        if k > 1 or not k.is_integer():
            raise ValueError('Invalid weight')
        X = self.fourier_expansion()
        R.<q> = PowerSeriesRing(QQ)
        X_new = [None]*len(X)
        prec = self.precision()
        for j, x in enumerate(X):
            val = x[2].valuation()
            X_new[j] = x[0], x[1], (q ** val) * R([ y * (i + x[1] + val) ** (1-k) for i, y in enumerate(x[2])]) + O(q ** (prec - floor(x[1])))
        return WeilRepModularForm(2 - k, self.gram_matrix(), X_new, weilrep = self.weilrep())

    def conjugate(self,A):
        r"""
        Conjugate modular forms by integral matrices.

        Suppose f(tau) is a modular form for the Weil representation attached to the Gram matrix S. This produces a modular form for the Gram matrix A.transpose() * S * A with the same Fourier expansion but different component vectors. If A is not invertible over ZZ then the result is an ``oldform``.

        INPUT:
        - ``A`` -- a square integral matrix with nonzero determinant

        EXAMPLES::

            sage: w = WeilRep(matrix([[-2,0],[0,-2]]))
            sage: w.theta_series(5).conjugate(matrix([[1,1],[0,1]]))
            [(0, 0), 1 + 4*q + 4*q^2 + 4*q^4 + O(q^5)]
            [(0, 1/2), 4*q^(1/2) + 8*q^(5/2) + 4*q^(9/2) + O(q^(11/2))]
            [(1/2, 1/2), 2*q^(1/4) + 4*q^(5/4) + 2*q^(9/4) + 4*q^(13/4) + 4*q^(17/4) + O(q^(21/4))]
            [(1/2, 0), 2*q^(1/4) + 4*q^(5/4) + 2*q^(9/4) + 4*q^(13/4) + 4*q^(17/4) + O(q^(21/4))]

            sage: w = WeilRep(matrix([[-2]]))
            sage: w.theta_series(5).conjugate(matrix([[2]]))
            [(0), 1 + 2*q + 2*q^4 + O(q^5)]
            [(7/8), O(q^(81/16))]
            [(3/4), 2*q^(1/4) + 2*q^(9/4) + O(q^(21/4))]
            [(5/8), O(q^(89/16))]
            [(1/2), 1 + 2*q + 2*q^4 + O(q^5)]
            [(3/8), O(q^(89/16))]
            [(1/4), 2*q^(1/4) + 2*q^(9/4) + O(q^(21/4))]
            [(1/8), O(q^(81/16))]
        """
        R.<q> = PowerSeriesRing(QQ)
        X = self.fourier_expansion()
        S = self.gram_matrix()
        prec = self.precision()
        S_conj = A.transpose()*S*A
        _ds_dict = self.weilrep().ds_dict()
        w_conj = WeilRep(S_conj)
        ds_conj = w_conj.ds()
        Y = [None] * len(ds_conj)
        for j, g in enumerate(ds_conj):
            g_old = tuple(frac(x) for x in A*g)
            try:
                i = _ds_dict[g_old]
                x = X[i]
                Y[j] = g, x[1], x[2]
            except:
                offset = -frac(g*S_conj*g/2)
                prec_g = prec - floor(offset)
                Y[j] = g, offset, O(q ** prec_g)
        return WeilRepModularForm(self.weight(), S_conj, Y, weilrep = w_conj)

    def hecke_P(self, N):
        r"""
        Apply the Nth Hecke projection map.

        This is the Hecke P_N operator of [BCJ]. It is a trace map on modular forms from WeilRep(N^2 * S) to WeilRep(S)
        """
        S = self.gram_matrix()
        S_new = matrix(ZZ, S / (N^2))
        nrows = S.nrows()
        symm = self.is_symmetric()
        w = self.weilrep()
        w_new = WeilRep(S_new)
        ds_dict = w_new.ds_dict()
        ds_new = w_new.ds()
        ds = w.ds()
        f = self.fourier_expansion()
        X = [None] * len(ds_new)
        multiplier = N^(-nrows)
        for i, g in enumerate(ds):
            g_new = [frac(N * x) for x in g]
            try:
                j = ds_dict[tuple(g_new)]
                if X[j] is None:
                    X[j] = [vector(g_new), f[i][1], f[i][2]]
                else:
                    X[j][2] += f[i][2]
            except KeyError:
                pass
        return WeilRepModularForm(self.weight(), S_new, X, w_new) * multiplier

    def hecke_T(self, N):
        r"""
        Apply the Nth Hecke operator where N is coprime to the level.

        REFERENCE: This uses the formula of section 2.6 of Ajouz's thesis.

        WARNING: if self has a unimodular lattice then hecke_T(N) corresponds to the classical/scalar Hecke operator of index N^2.
        If self has a rank 0 underlying lattice (with matrix([]) as Gram matrix) then hecke_V(N) corresponds to the classical/scalar Hecke operator of index N.

        INPUT:
        - ``N`` -- a natural number coprime to the level of our lattice

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[2, 0], [0, -2]]))
            sage: f = w.eisenstein_series(4, 50)
            sage: f.hecke_T(3)
            [(0, 0), 757 + 96896*q + 763056*q^2 + 2713088*q^3 + 6286128*q^4 + O(q^5)]
            [(1/2, 0), 42392*q^(3/4) + 520816*q^(7/4) + 2016648*q^(11/4) + 5341392*q^(15/4) + 10386040*q^(19/4) + O(q^(23/4))]
            [(0, 1/2), 1514*q^(1/4) + 190764*q^(5/4) + 1146098*q^(9/4) + 3327772*q^(13/4) + 7439796*q^(17/4) + O(q^(21/4))]
            [(1/2, 1/2), 84784*q + 872064*q^2 + 2373952*q^3 + 6976512*q^4 + O(q^5)]
        """
        w = self.weilrep()
        l = w.level()
        S = self.gram_matrix()
        if gcd(l, N) != 1:
            raise ValueError('hecke_T() only takes indices coprime to the level of the discriminant form.')
        nrows = S.nrows()
        eps = nrows - w.signature()
        if eps % 4 == 2:
            eps = -1
        else:
            eps = 1
        if nrows % 4 == 2 or nrows % 4 == 3:
            eps = -eps
        k = self.weight()
        k1 = floor(k - 1)
        T = self.fourier_expansion()
        prec = self.precision() // (N^2)
        q = T[0][2].parent().0
        F = [[t[0], t[1], O(q ** (prec - floor(t[1])))] for t in T]
        val = self.valuation() * N * N
        ds_dict = w.ds_dict()
        ds = w.ds()
        D = len(ds)
        odd_rank = nrows % 2
        D *= (eps * (1 + odd_rank))
        N_sqr_l = N * N * l
        indices = w.rds(indices = True)
        symm = self.is_symmetric()
        if not symm:
            symm = -1
        def rho(n, a):
            if a == 1:
                return 1
            if odd_rank:
                nl = ZZ(n * l)
                if (nl * N^2) % a:
                    return 0
                g1 = gcd(a, nl)
                f = isqrt(g1)
                fsqr = f * f
                if fsqr - g1:
                    return 0
                a_f = a / fsqr
                n_f = n / fsqr
                return f * kronecker_symbol(ZZ(n_f * D), a_f)
            else:#?? this shouldn't get called anyway
                return kronecker_symbol(D, a)
        for i, g in enumerate(ds):
            if indices[i] is None:
                offset = F[i][1]
                for a in divisors(N * N):
                    if True:
                        Na_inv = N * a.inverse_mod(l)
                        b = (N / a)^2
                        a_pow = a^k1
                        a_sqr = a * a
                        j = ds_dict[tuple(frac(Na_inv * x) for x in g)]
                        u = F[j][1]
                        for n0 in range(val, prec - floor(offset)):
                            n = n0 + offset
                            if odd_rank:
                                if (N_sqr_l * n) % a_sqr == 0:
                                    try:
                                        F[i][2] += a_pow * rho(-n, a) * T[j][2][ZZ(b * n - u)] * q^(n0)
                                    except IndexError:
                                        if n0 > 0:
                                            F[i][2] += O(q^n0)
                                            break
                            else:
                                nl = ZZ(l * n)
                                if nl % a == 0:
                                    try:
                                        F[i][2] += a_pow * kronecker_symbol(D, a) * T[j][2][ZZ(b * n - u)] * q^(n0)
                                    except IndexError:
                                        if n0 > 0:
                                            F[i][2] += O(q^n0)
                                            break
            else:
                F[i][2] = symm * F[indices[i]][2]
        return WeilRepModularForm(k, S, F, w)

    def hecke_U(self, N):
        r"""
        Apply the index-raising Hecke operator U_N.

        This is the same as conjugating by N times the identity matrix.
        """
        return self.conjugate(N * identity_matrix(self.gram_matrix().nrows()))

    def hecke_V(self, N):
        r"""
        Apply the index-raising Hecke operator V_N.

        This is the Eichler--Zagier V_N operator applied to vector-valued modular forms on lattices of arbitrary signature instead of Jacobi forms.

        INPUT:
        - ``N`` -- a natural number or 0

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[2, 0], [0, -2]]))
            sage: f = w.eisenstein_series(4, 10)
            sage: f.hecke_V(3)
            [(0, 0), 82 + 3584*q + 28224*q^2 + O(q^3)]
            [(1/6, 0), 2664*q^(11/12) + 24336*q^(23/12) + 86688*q^(35/12) + O(q^(47/12))]
            [(1/3, 0), 1008*q^(2/3) + 16128*q^(5/3) + 66672*q^(8/3) + O(q^(11/3))]
            [(1/2, 0), 56*q^(1/4) + 7056*q^(5/4) + 45416*q^(9/4) + O(q^(13/4))]
            [(2/3, 0), 1008*q^(2/3) + 16128*q^(5/3) + 66672*q^(8/3) + O(q^(11/3))]
            [(5/6, 0), 2664*q^(11/12) + 24336*q^(23/12) + 86688*q^(35/12) + O(q^(47/12))]
            [(0, 5/6), 2*q^(1/12) + 4396*q^(13/12) + 31502*q^(25/12) + O(q^(37/12))]
            [(1/6, 5/6), 3136*q + 32256*q^2 + O(q^3)]
            [(1/3, 5/6), 1514*q^(3/4) + 19264*q^(7/4) + 74592*q^(11/4) + O(q^(15/4))]
            [(1/2, 5/6), 112*q^(1/3) + 9216*q^(4/3) + 38528*q^(7/3) + O(q^(10/3))]
            [(2/3, 5/6), 1514*q^(3/4) + 19264*q^(7/4) + 74592*q^(11/4) + O(q^(15/4))]
            [(5/6, 5/6), 3136*q + 32256*q^2 + O(q^3)]
            [(0, 2/3), 128*q^(1/3) + 8304*q^(4/3) + 44032*q^(7/3) + O(q^(10/3))]
            [(1/6, 2/3), 56*q^(1/4) + 7056*q^(5/4) + 40880*q^(9/4) + O(q^(13/4))]
            [(1/3, 2/3), 1 + 3584*q + 28224*q^2 + O(q^3)]
            [(1/2, 2/3), 688*q^(7/12) + 13720*q^(19/12) + 59584*q^(31/12) + O(q^(43/12))]
            [(2/3, 2/3), 1 + 3584*q + 28224*q^2 + O(q^3)]
            [(5/6, 2/3), 56*q^(1/4) + 7056*q^(5/4) + 40880*q^(9/4) + O(q^(13/4))]
            [(0, 1/2), 1676*q^(3/4) + 19264*q^(7/4) + 74592*q^(11/4) + O(q^(15/4))]
            [(1/6, 1/2), 1152*q^(2/3) + 14112*q^(5/3) + 73728*q^(8/3) + O(q^(11/3))]
            [(1/3, 1/2), 252*q^(5/12) + 9828*q^(17/12) + 48780*q^(29/12) + O(q^(41/12))]
            [(1/2, 1/2), 3136*q + 32256*q^2 + O(q^3)]
            [(2/3, 1/2), 252*q^(5/12) + 9828*q^(17/12) + 48780*q^(29/12) + O(q^(41/12))]
            [(5/6, 1/2), 1152*q^(2/3) + 14112*q^(5/3) + 73728*q^(8/3) + O(q^(11/3))]
            [(0, 1/3), 128*q^(1/3) + 8304*q^(4/3) + 44032*q^(7/3) + O(q^(10/3))]
            [(1/6, 1/3), 56*q^(1/4) + 7056*q^(5/4) + 40880*q^(9/4) + O(q^(13/4))]
            [(1/3, 1/3), 1 + 3584*q + 28224*q^2 + O(q^3)]
            [(1/2, 1/3), 688*q^(7/12) + 13720*q^(19/12) + 59584*q^(31/12) + O(q^(43/12))]
            [(2/3, 1/3), 1 + 3584*q + 28224*q^2 + O(q^3)]
            [(5/6, 1/3), 56*q^(1/4) + 7056*q^(5/4) + 40880*q^(9/4) + O(q^(13/4))]
            [(0, 1/6), 2*q^(1/12) + 4396*q^(13/12) + 31502*q^(25/12) + O(q^(37/12))]
            [(1/6, 1/6), 3136*q + 32256*q^2 + O(q^3)]
            [(1/3, 1/6), 1514*q^(3/4) + 19264*q^(7/4) + 74592*q^(11/4) + O(q^(15/4))]
            [(1/2, 1/6), 112*q^(1/3) + 9216*q^(4/3) + 38528*q^(7/3) + O(q^(10/3))]
            [(2/3, 1/6), 1514*q^(3/4) + 19264*q^(7/4) + 74592*q^(11/4) + O(q^(15/4))]
            [(5/6, 1/6), 3136*q + 32256*q^2 + O(q^3)]
        """
        k = self.weight()
        S = self.gram_matrix()
        k_1 = k + S.nrows()/2 - 1
        symm = self.is_symmetric()
        if symm:
            eps = 1
        else:
            eps = -1
        if N == 0:
            k_0 = k_1 + 1
            if symm:
                return self.fourier_expansion()[0][2][0] * smf(k_1, eisenstein_series_qexp(k_0, self.precision()))
            q = self.fourier_expansion()[0][2].base_ring().0
            return smf(k_0, 0 + O(q ** self.precision()))
        r.<q> = PowerSeriesRing(QQ)
        prec = self.precision()
        w = self.weilrep()
        ds_dict = w.ds_dict()
        ds = w.ds()
        X = self.fourier_expansion()
        big_w = w(N)
        big_ds = big_w.ds()
        indices = big_w.rds(indices = True)
        new_prec = prec // N
        Y = [None] * len(big_ds)
        for j, g in enumerate(big_ds):
            r_val = g*N*S*g/2
            big_offset = -frac(r_val)
            if indices[j] is None:
                Y[j] = [g, big_offset, O(q^(new_prec + ceil(-big_offset)))]
                for a in divisors(N):
                    d = N//a
                    a_pow = a^k_1
                    try:
                        i = ds_dict[tuple(frac(d * x) for x in g)]
                        offset = X[i][1]
                        n = big_offset
                        while n < (prec + offset) * (a / d):
                            if (n + r_val) % a == 0:
                                Y[j][2] += X[i][2][ZZ(d * n / a - offset)] * q^(ceil(n)) * a_pow
                            n += 1
                    except KeyError:
                        pass
            else:
                i = indices[j]
                Y[j] = [g, big_offset, eps * Y[i][2]]
        return WeilRepModularForm(self.weight(), N*S, Y )

    def reduce_lattice(self, z = None, z_prime = None, zeta = None):
        r"""
        Compute self's image under lattice reduction.

        This implements the lattice-reduction map from isotropic lattices of signature (b^+, b^-) to signature (b^+ - 1, b^- - 1). In Borcherds' notation ([B], Chapter 5) this takes the form F_M as input and outputs the form F_K.

        NOTE: If it is possible to choose zeta with <z, zeta> = 1 then this method yields a smaller-rank lattice with an equivalent discriminant form and it preserves Fourier coefficients. (This is always possible if we have an isotropic lattice of square-free discriminant.) Otherwise if L is the original lattice and K is the result lattice then |L'/L| = N^2 * |K'/K| where <z, zeta> = N.

        INPUT:
        - ``z`` -- a primitive norm-zero vector. If this is not given then we try to compute such a vector using PARI qfsolve(), and raise a ValueError if this does not exist (i.e. the lattice is anisotropic; this can only happen if the lattice is definite, or indefinite of rank less than 5).
        - ``z_prime`` -- a vector in the dual lattice with <z, z_prime> = 1. If this is not given then we compute it.
        - ``zeta`` -- a lattice vector for which <z, zeta> = N is minimal among all <z, x> for x in the lattice. If this is not given then we compute it.

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[2, 0], [0, -2]]))
            sage: f = w.eisenstein_series(4, 10)
            sage: f.reduce_lattice()
            [(), 1 + 240*q + 2160*q^2 + 6720*q^3 + 17520*q^4 + 30240*q^5 + 60480*q^6 + 82560*q^7 + 140400*q^8 + 181680*q^9 + O(q^10)]
        """
        w = self.weilrep()
        s = w.gram_matrix()
        if z is None:
            z = pari(s).qfsolve().sage()
        try:
            try:
                z = z.columns()[0]
            except AttributeError:
                pass
            _ = len(z)
            z = vector(z)
        except TypeError:
            raise ValueError('This lattice is anisotropic!')
        sz = s * z
        if z_prime is None or zeta is None:
            def xgcd_v(x): #xgcd for more than two arguments
                if len(x) > 1:
                    g, a = xgcd_v(x[:-1])
                    if g == 1:
                        return g, vector(list(a) + [0])
                    new_g, s, t = xgcd(g, x[-1])
                    return new_g, vector(list(a * s) + [t])
                return x[0], vector([1])
            if zeta is None:
                _, zeta = xgcd_v(sz)
            if z_prime is None:
                _, sz_prime = xgcd_v(z)
                z_prime = s.inverse() * sz_prime
            else:
                sz_prime = s * z_prime
        szeta = s * zeta
        n = sz * zeta
        zeta_norm = zeta * szeta
        z_prime_norm = z_prime * sz_prime
        k = matrix(matrix([sz, sz_prime]).transpose().integer_kernel().basis())
        try:
            k_k = k * s * k.transpose()
        except TypeError:
            k_k = matrix([])
        w_k = WeilRep(k_k)
        ds_k_dict = w_k.ds_dict()
        ds_k = w_k.ds()
        ds = w.ds()
        zeta_K = zeta - n * (z_prime - z_prime_norm * z) - (szeta * z_prime)* z
        Y = [None] * len(ds_k)
        X = self.fourier_expansion()
        prec = self.precision()
        r.<q> = PowerSeriesRing(QQ)
        for i, g in enumerate(ds):
            if not ZZ(g * sz) % n:
                g_k = g - (g * sz) * (z_prime - z_prime_norm * z) - (g * sz_prime) * z
                pg = g_k - (g * sz) * zeta_K / n
                try:
                    pg = vector(map(frac, k.solve_left(pg)))
                except ValueError:
                    pg = vector([])
                j = ds_k_dict[tuple(pg)]
                if Y[j] is None:
                    Y[j] = [pg, -frac(pg * k_k * pg / 2), X[i][2]]
                else:
                    Y[j][2] += X[i][2]
        for j, g in enumerate(ds_k):
            if Y[j] is None:
                o = -frac(g * k_k * g / 2)
                Y[j] = g, o, O(q^(prec - floor(o)))
        return WeilRepModularForm(self.weight(), k_k, Y, w_k)

    def serre_derivative(self, normalize_constant_term = False):
        r"""
        Compute the Serre derivative.

        This returns the WeilRepModularForm
        f'(z) / (2*pi*I) - k * E_2(z) * f(z)
        where f(z) is self; E_2(z) is the quasi-modular Eisenstein series of weight 2; and where k is self's weight.

        EXAMPLES::

            sage: w = WeilRep(matrix([[-2]]))
            sage: w.theta_series(5).serre_derivative()
            [(0), -1/24 + 35/12*q + 5*q^2 + 10*q^3 + 275/12*q^4 + O(q^5)]
            [(1/2), 5/12*q^(1/4) + 2*q^(5/4) + 125/12*q^(9/4) + 10*q^(13/4) + 20*q^(17/4) + O(q^(21/4))]

            sage: WeilRep(matrix([[-8]])).zero(1/2, 5).serre_derivative()
            [(0), O(q^5)]
            [(7/8), O(q^(81/16))]
            [(3/4), O(q^(21/4))]
            [(5/8), O(q^(89/16))]
            [(1/2), O(q^5)]
            [(3/8), O(q^(89/16))]
            [(1/4), O(q^(21/4))]
            [(1/8), O(q^(81/16))]

        """

        X = self.fourier_expansion()
        k = self.weight()
        prec = self.precision()
        X = [(x[0], x[1], serre_derivative_on_q_series(x[2],x[1],k,prec)) for x in X]
        if normalize_constant_term:
            a = X[0][2][0]
            if a != 0:
                return WeilRepModularForm(self.weight() + 2, self.gram_matrix(), X, weilrep = self.weilrep()) / a
        return WeilRepModularForm(self.weight() + 2, self.gram_matrix(), X, weilrep = self.weilrep())

    def symmetrized(self, b):
        r"""
        Compute the symmetrization of self over an isotropic subgroup of the finite quadratic module.

        INPUT:
        - ``b`` -- an integer-norm vector in self's discriminant group.

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: f = WeilRep(matrix([[-8]])).eisenstein_series(5/2, 5)
            sage: f.symmetrized(vector([1/2]))
            [(0), 1 - 70*q - 120*q^2 - 240*q^3 - 550*q^4 + O(q^5)]
            [(7/8), O(q^(81/16))]
            [(3/4), -10*q^(1/4) - 48*q^(5/4) - 250*q^(9/4) - 240*q^(13/4) - 480*q^(17/4) + O(q^(21/4))]
            [(5/8), O(q^(89/16))]
            [(1/2), 1 - 70*q - 120*q^2 - 240*q^3 - 550*q^4 + O(q^5)]
            [(3/8), O(q^(89/16))]
            [(1/4), -10*q^(1/4) - 48*q^(5/4) - 250*q^(9/4) - 240*q^(13/4) - 480*q^(17/4) + O(q^(21/4))]
            [(1/8), O(q^(81/16))]
        """
        d_b = denominator(b)
        if d_b == 1:
            return self
        R.<q> = PowerSeriesRing(QQ)
        S = self.__gram_matrix
        X = self.components()
        w = self.weilrep()
        ds = w.ds()
        symm = self.is_symmetric
        if symm:
            eps = 1
        else:
            eps = -1
        indices = w.rds(indices = True)
        norm_list = w.norm_list()
        Y = [None] * len(ds)
        prec = self.precision()
        for i, g in enumerate(ds):
            if indices[i] is None:
                g_b = frac(g * S * b)
                if g_b:
                    Y[i] = g, norm_list[i], O(q ** (prec - floor(norm_list[i])))
                else:
                    f = sum(X[tuple(frac(x) for x in g + j * b)] for j in range(d_b))
                    Y[i] = g, norm_list[i], f
            else:
                Y[i] = g, norm_list[i], eps * Y[indices[i]][2]
        return WeilRepModularForm(self.__weight, S, Y, weilrep = self.weilrep())

    def theta_contraction(self, odd = False, components = None, weilrep = None):
        r"""
        Compute the theta-contraction of self.

        This computes the theta-contraction to the Weil representation of the Gram matrix given by the upper (d-1)*(d-1) block of self's Gram matrix. (For this to be well-defined, the (d, d)- entry of the Gram matrix must satisfy a certain positivity condition!) This is essentially a product of self with a unary theta function, twisted such that the result transforms by the correct action of Mp_2(Z).

        See also section 3 of [Ma]. (S. Ma - Quasi-Pullback of Borcherds products)

        INPUT:
        - ``odd`` -- boolean (default False); if True, then we instead compute the theta-contraction as a product with a theta function of weight 3/2 (not 1/2). This is useful for constructing antisymmetric modular forms. (e.g. it is used in the .pssd() method of the WeilRep class)
        - ``components`` -- a list consisting of a subset of the discriminant group and indices among them which should be ignored. (default None) If None then we use the full discriminant group.
        - ``weilrep`` -- WeilRep (default None) assigns a WeilRep class to the output

        EXAMPLES::

            sage: w = WeilRep(matrix([[2]]))
            sage: w.eisenstein_series(7/2, 5).theta_contraction()
            [(), 1 + 240*q + 2160*q^2 + 6720*q^3 + 17520*q^4 + O(q^5)]

            sage: w = WeilRep(matrix([[-2,1],[1,2]]))
            sage: w.eisenstein_series(6, 5).theta_contraction()
            [(0), 1 - 25570/67*q - 1147320/67*q^2 - 10675440/67*q^3 - 52070050/67*q^4 + O(q^5)]
            [(1/2), -10/67*q^(1/4) - 84816/67*q^(5/4) - 2229850/67*q^(9/4) - 16356240/67*q^(13/4) - 73579680/67*q^(17/4) + O(q^(21/4))]

         """
        symm = self.is_symmetric()
        prec = self.precision()
        R.<q> = PowerSeriesRing(QQ)
        big_S = self.gram_matrix()
        val = self.valuation()
        big_e = big_S.nrows()
        e = big_e - 1
        S = big_S[:e,:e]
        try:
            Sb = vector(big_S[:e,e])
            b = S.inverse()*Sb
        except ValueError:#i.e. S is size 0x0. surprisingly everything but these two lines continues to work
            Sb = vector([])
            b = vector([])
        m = (big_S[e,e] - b*Sb)/2
        X = self.fourier_expansion()
        g_list = []
        S_indices = []
        bound = 3 + 2*isqrt(m * (prec - val))
        if components:
            _ds, _indices = components
        else:
            if not weilrep:
                weilrep = WeilRep(S)
            _indices = weilrep.rds(indices = True)
            _ds = weilrep.ds()
        big_ds_dict = {tuple(X[i][0]) : i for i in range(len(X))}
        b_denom = b.denominator()
        bm2 = ZZ(2*m*b_denom)
        Y = [None] * len(_ds)
        eps = (2 * (odd != symm) - 1)
        if val < 0:
            def map_to_R(f):
                try:
                    return R(f)
                except TypeError:
                    return R(0)
        for i, g in enumerate(_ds):
            offset = frac(g*S*g/2)
            prec_g = prec + ceil(offset)
            theta_twist = [[0]*prec_g for j in range(bm2)]
            gSb = frac(g*S*b)
            if (odd == symm) and g.denominator() <= 2:#component is zero
                Y[i] = g, -offset, O(q ** prec_g)
            elif _indices[i] is None:
                r_i = -1
                g_ind = []
                r_square = (bound + 1 + gSb)^2 / (4*m) + offset
                old_offset = 0
                big_offset_ind = []
                for r in range(-bound, bound+1):
                    r_i += 1
                    r_shift = r - gSb
                    if r_i < bm2:
                        i_m = r_i
                        g_new = list(g - b * r_shift/(2 * m)) + [r_shift/(2 * m)]
                        g_new = tuple([frac(x) for x in g_new])
                        j = big_ds_dict[g_new]
                        g_ind.append(j)
                        big_offset_ind.append(X[j][1])
                    else:
                        i_m = r_i % bm2
                        j = g_ind[i_m]
                    new_offset = big_offset_ind[i_m]
                    r_square += (new_offset - old_offset) + (2*r_shift - 1) / (4*m)
                    old_offset = new_offset
                    if r_square < prec_g:
                        if odd:
                            theta_twist[i_m][r_square] += r_shift
                        else:
                            theta_twist[i_m][r_square] += 1
                    elif r > 0:
                        break
                if val >= 0:
                    Y[i] = g, -offset, sum([R(theta_twist[j]) * X[g_ind[j]][2] for j in range(min(bm2, len(g_ind)))])+O(q ** prec_g)
                else:
                    Y[i] = g, -offset, q^(val) * (sum([R(theta_twist[j]) * map_to_R(q^(-val) * X[g_ind[j]][2]) for j in range(min(bm2, len(g_ind))) if theta_twist[j]])+O(q ** (prec_g - val)))
            else:
                Y[i] = g, -offset, eps * Y[_indices[i]][2]
        return WeilRepModularForm(self.weight() + 1/2 + odd, S, Y, weilrep = weilrep)

def smf(weight, f):
    r"""
    Construct WeilRepModularForms for the empty matrix from q-series.

    INPUT:
    - ``weight`` -- a weight (which should be an even integer)
    - ``f`` -- a power series in the variable 'q' (which should represent a modular form of weight 'weight' and level 1)

    OUTPUT: WeilRepModularForm

    EXAMPLES::

        sage: smf(12, delta_qexp(10))
        [(), q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 - 6048*q^6 - 16744*q^7 + 84480*q^8 - 113643*q^9 + O(q^10)]

    """
    return WeilRepModularForm(weight, matrix([]), [[vector([]), 0, f]])

class WeilRepModularFormsBasis:
    r"""
    The WeilRepModularFormsBasis class represents bases of vector-valued modular forms.
    """

    def __init__(self, weight, basis, weilrep):
        self.__weight = weight
        self.__basis = basis
        self.__weilrep = weilrep

    def __repr__(self):
        r"""
        Print the output with a line of hyphens as delimiter.
        """
        X = self.__basis
        if X:
            s = '\n' + '-'*60 + '\n'
            return s.join([x.__repr__() for x in self.__basis])
        return '[]'

    def append(self, other):
        r"""
        Append a WeilRepModularForm to self.
        """
        if other._WeilRepModularForm__weight == self.weight():
            self.__basis.append(other)
        else:
            raise ValueError('I have weight %s and you are trying to append a modular form of weight %s.' %(self.weight(), other._WeilRepModularForm__weight))

    def echelonize(self, save_pivots = False, starting_from = 0, ending_with = None, integer = False):
        r"""
        Reduce self to echelon form in place.

        INPUT:
        - ``save_pivots`` -- if True then return the pivot columns. (Otherwise we return None)
        - ``starting_from`` -- (default 0) the index at which we start looking at Fourier coefficients
        - ``ending_with`` -- (default None) if given then it should be the index at which we stop looking at Fourier coefficients.
        - ``integer`` -- (default False) if True then we assume all Fourier coefficients are integers. This is faster.
        """
        if ending_with is None:
            ending_with = self.__weight / 12
        if integer:
            m = matrix(ZZ, [v.coefficient_vector(starting_from = starting_from, ending_with = ending_with) for v in self.__basis])
            a, b = m.echelon_form(transformation = True)
            a_rows = a.rows()
            self.__basis = [self * v for i, v in enumerate(b.rows()) if a_rows[i]]
        else:
            m = matrix([v.coefficient_vector(starting_from = starting_from, ending_with = ending_with) for v in self.__basis]).extended_echelon_form(subdivide = True, proof = False)
            b = m.subdivision(0, 1)
            self.__basis = [self * v for v in b.rows()]
        if save_pivots:
            if not integer:
                a = m.subdivision(0, 0)
            pivots = [next(j for j, w in enumerate(v) if w) for v in a.rows()]
            return pivots

    def __eq__(self, other):
        if not len(self) == len(other):
            return False
        return all(x == other[i] for i, x in enumerate(self))

    def extend(self, other):
        r"""
        Extend self by another WeilRepModularFormsBasis
        """
        try:
            self.__basis.extend(other.list())
        except:
            self.__basis.extend(other)

    def __getitem__(self, n):
        return self.__basis[n]

    def __getslice__(self, i, j):
        return WeilRepModularFormsBasis(self.__weight, self.__basis[i:j], self.__weilrep)

    def gram_matrix(self):
        return self.__weilrep.gram_matrix()

    def is_symmetric(self):
        return self.__weilrep.is_symmetric_weight(self.__weight)

    def __iter__(self):
        for x in self.__basis:
            yield x

    def jacobi_forms(self):
        r"""
        Return a list of the Jacobi forms associated to all elements of self.

        If the Gram matrix is positive-definite (this is not checked!!) then this returns a list of Jacobi forms whose theta-decompositions are the vector valued modular forms that we started with.

        OUTPUT: a list of JacobiForm's
        """
        X = [x.fourier_expansion() for x in self.__basis]
        if not X:
            return []
        S = self.gram_matrix()
        prec = self.precision()
        val = self.valuation()
        e = S.nrows()
        Rb = LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(e) ))
        R.<q> = PowerSeriesRing(Rb,prec)
        if e > 1:
            precval = prec - val
            _ds_dict = self.weilrep().ds_dict()
            jf = [[Rb(0)]*precval for _ in self.__basis]
            Q = QuadraticForm(S)
            if not Q.is_positive_definite():
                raise ValueError('Index is not positive definite')
            S_inv = S.inverse()
            k = self.weight() + e/2
            try:
                _, _, vs_matrix = pari(S_inv).qfminim(precval + precval + 1, flag = 2)
                vs_list = vs_matrix.sage().columns()
                symm = self.is_symmetric()
                symm = 1 if symm else -1
                for v in vs_list:
                    wv = Rb.monomial(*v)
                    wv_symm = wv + (symm * (wv ** (-1)))
                    r = S_inv * v
                    r_norm = v*r / 2
                    i_start = ceil(r_norm)
                    j = _ds_dict[tuple(frac(x) for x in r)]
                    f = [x[j][2] for x in X]
                    m = ceil(i_start + val - r_norm)
                    for i in range(i_start, precval):
                        for ell, h in enumerate(f):
                            jf[ell][i] += wv_symm * h[m]
                        m += 1
                f = [x[0][2] for x in X]#deal with v=0 separately
                for i in range(precval):
                    for ell, h in enumerate(f):
                        jf[ell][i] += h[ceil(val) + i]
                return [JacobiForm(k, S, q^val * R(x) + O(q ** prec), weilrep = self.__weilrep, modform = self[i]) for i, x in enumerate(jf)]
            except PariError:
                pass
            lvl = Q.level()
            S_adj = lvl*S_inv
            vs = QuadraticForm(S_adj).short_vector_list_up_to_length(lvl * precval, up_to_sign_flag = True)
            for n in range(len(vs)):
                r_norm = n/lvl
                i_start = ceil(r_norm)
                for v in vs[n]:
                    r = S_inv*v
                    rfrac = tuple(frac(r[i]) for i in range(e))
                    wv = Rb.monomial(*v)
                    if v:
                        wv += symm * (wv ** (-1))
                    j = _ds_dict[rfrac]
                    f = [x[j][2] for x in X]
                    m = ceil(i_start + val - r_norm)
                    for i in range(i_start,prec):
                        for ell, h in enumerate(f):
                            jf[ell][i] += wv*h[m]
                        m += 1
            return [JacobiForm(k, S, q^val*R(x)+O(q^prec), weilrep = self.__weilrep, modform = self[i]) for i, x in enumerate(jf)]
        else:
            w = Rb.0
            m = S[0,0] #twice the index
            if self.is_symmetric():
                eps = 1
            else:
                eps = -1
            jf = [[None]*(prec - val) for _ in self.__basis]
            for i in range(prec - val):
                for j, x in enumerate(X):
                    jf[j][i] = x[0][2][i + val]
                    for r in range(1, isqrt(2 * i * m) + 1):
                        wr = (w ** r + eps * (w ** (-r)))
                        jf[j][i] += x[r%m][2][ceil(i + val - r^2 / (2*m))]*wr
            k = self.weight() + 1/2
            return [JacobiForm(k, S, q^val * R(x) + O(q^prec), weilrep = self.__weilrep, modform = self[i]) for i, x in enumerate(jf)]

    def __len__(self):
        return len(self.__basis)

    def list(self):
        return self.__basis

    def __mul__(self, v):
        return sum([self.__basis[i] * w for i, w in enumerate(v)])

    def precision(self):
        return min(x.precision() for x in self.__basis)

    def rank(self, starting_from = 0):
        ending_with = self.__weight / 12
        m = matrix(v.coefficient_vector(starting_from = starting_from, ending_with = ending_with) for v in self.__basis)
        return m.rank()

    def reverse(self):
        self.__basis.reverse()

    __rmul__ = __mul__

    def shrink(self, starting_from = 0, ending_with = None):
        if ending_with is None:
            ending_with = self.__weight / 12
        m = matrix(v.coefficient_vector(starting_from = starting_from, ending_with = ending_with) for v in self.__basis)
        self.__basis = [self[j] for j in m.pivot_rows()]

    def theta(self, odd = False, weilrep = None):
        r"""
        Compute the theta-contraction of all of self's WeilRepModularForm's at the same time.
        """
        big_S = self.gram_matrix()
        big_e = big_S.nrows()
        e = big_e - 1
        S = big_S[:e,:e]
        k = self.weight() + 1/2 + odd
        if not weilrep:
            weilrep = WeilRep(S)
        if not self.__basis:
            return WeilRepModularFormsBasis(k, [], weilrep)
        symm = self.is_symmetric()
        prec = self.precision()
        R.<q> = PowerSeriesRing(QQ)
        try:
            Sb = vector(big_S[:e,e])
            b = S.inverse()*Sb
        except ValueError:
            Sb = vector([])
            b = vector([])
        m = (big_S[e,e] - b*Sb)/2
        X = [x.fourier_expansion() for x in self.__basis]
        X_ref = X[0]#reference
        g_list = []
        S_indices = []
        bound = 3 + 2*isqrt(m * (prec - self.valuation()))
        _ds = weilrep.ds()
        _indices = weilrep.rds(indices = True)
        big_ds_dict = {tuple(x[0]) : i for i, x in enumerate(X_ref)}
        b_denom = b.denominator()
        bm2 = ZZ(2*m*b_denom)
        Y = [None] * len(_ds)
        Y = [copy(Y) for _ in range(self.__len__())]
        eps = odd != symm
        eps = eps + eps - 1
        for i, g in enumerate(_ds):
            offset = frac(g*S*g/2)
            prec_g = prec + ceil(offset)
            theta_twist = [[0]*prec_g for j in range(bm2)]
            gSb = frac(g*S*b)
            if (odd == symm) and g.denominator() <= 2:#component is zero
                t = g, -offset, O(q ** prec_g)
                for y in Y:
                    y[i] = t
            elif _indices[i] is None:
                r_i = -1
                g_ind = []
                r_square = (bound + 1 + gSb)^2 / (4*m) + offset
                old_offset = 0
                big_offset_ind = []
                for r in range(-bound, bound+1):
                    r_i += 1
                    r_shift = r - gSb
                    if r_i < bm2:
                        i_m = r_i
                        y = r_shift / (2*m)
                        g_new = list(g - b * y) + [y]
                        g_new = tuple([frac(x) for x in g_new])
                        j = big_ds_dict[g_new]
                        g_ind.append(j)
                        big_offset_ind.append(X_ref[j][1])
                    else:
                        i_m = r_i % bm2
                        j = g_ind[i_m]
                    new_offset = big_offset_ind[i_m]
                    r_square += (new_offset - old_offset) + (2*r_shift - 1) / (4*m)
                    old_offset = new_offset
                    if r_square < prec_g:
                        if odd:
                            theta_twist[i_m][r_square] += r_shift
                        else:
                            theta_twist[i_m][r_square] += 1
                    elif r > 0:
                        break
                for iy, y in enumerate(Y):
                    y[i] = g, -offset, sum([R(theta_twist[j]) * X[iy][g_ind[j]][2] for j in range(min(bm2, len(g_ind)))])+O(q ** prec_g)
            else:
                index = _indices[i]
                for y in Y:
                    y[i] = g, -offset, eps * y[_indices[i]][2]
        return WeilRepModularFormsBasis(k, [WeilRepModularForm(k, S, y, weilrep = weilrep) for y in Y], weilrep = weilrep)

    def valuation(self):
        return min(x.valuation() for x in self.__basis)

    def weight(self):
        return self.__weight

    def weilrep(self):
        return self.__weilrep