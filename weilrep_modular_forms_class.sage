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

    def __repr__(self):
        try:
            return self.__qexp_string
        except:
            def offset_qseries_string(offset,f): #display f with exponents shifted by a rational number.
                s = ''
                j = -1
                start = f.valuation()
                if start == Infinity:
                    start = 0
                for i in range(start,f.prec()):
                    i_offset = i + offset
                    if f[i]:
                        j += 1
                        sgn = [' + ', ' - '][f[i] < 0]
                        if j > 0:
                            s += sgn
                        elif f[i] < 0:
                            s += '-'
                        if abs(f[i]) !=1 or not i_offset:
                            s += str(abs(f[i]))
                        if i_offset:
                            if abs(f[i]) != 1:
                                s += '*'
                            s += 'q'
                            if i_offset != 1:
                                s += '^' + ['(' + str(i_offset) + ')', str(i_offset)][i_offset.is_integer()]
                if j >= 0:
                    s += ' + '
                return s +  'O(q^' + str(floor(f.prec() + offset)) + ')'
            X = self.fourier_expansion()
            self.__qexp_string = '\n'.join(['['+str(x[0])+', ' + offset_qseries_string(x[1],x[2]) + ']' for x in X]) #display all fourier expansions with offset, each on a separate line
            return self.__qexp_string

    ## basic attributes

    def weight(self):
        return self.__weight

    def gram_matrix(self):
        return self.__gram_matrix

    def fourier_expansion(self):
        return self.__fourier_expansions

    def __nonzero__(self):
        return any(x[2] for x in self.__fourier_expansions)

    __bool__ = __nonzero__

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

    def reduce_precision(self, prec, in_place = True):
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

    def valuation(self, round_down = True):
        r"""
        Returns the lowest exponent in our Fourier expansion with a nonzero coefficient (rounded down).
        """
        try:
            if round_down:
                return self.__valuation
            return self.__true_valuation
        except AttributeError:
            X = self.fourier_expansion()
            try:
                self.__true_valuation = min([x[2].valuation() + x[1] for x in X if x[2]])
                self.__valuation = floor(self.__true_valuation)
            except ValueError: #probably trying to take valuation of 0
                self.__valuation = 0 #for want of a better value
            if round_down:
                return self.__valuation
            return self.__true_valuation

    def is_symmetric(self):
        r"""
        Determines whether the components f_{\gamma} in our Fourier expansion satisfy f_{\gamma} = f_{-\gamma} or f_{\gamma} = -f_{\gamma}.
        This can be read off the weight.
        """
        try:
            return self.__is_symmetric
        except AttributeError:
            self.__is_symmetric = [1,None,0,None][(ZZ(2*self.weight()) + self.signature()) % 4]
            return self.__is_symmetric


    def rds(self, indices = False):
        return self.weilrep().rds(indices = indices)

    def ds(self):
        return self.weilrep().ds()

    def signature(self):
        return self.weilrep().signature()

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
            G = self.rds()
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
            [(1/3, 1/3, 3/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^5)]
            [(2/3, 2/3, 1/2), 54*q^(7/6) + 432*q^(13/6) + 918*q^(19/6) + 2160*q^(25/6) + 2754*q^(31/6) + O(q^6)]
            [(0, 0, 1/4), q^(1/8) + 73*q^(9/8) + 342*q^(17/8) + 991*q^(25/8) + 1728*q^(33/8) + O(q^5)]
            [(1/3, 1/3, 0), 27*q^(2/3) + 216*q^(5/3) + 513*q^(8/3) + 1512*q^(11/3) + 2268*q^(14/3) + O(q^5)]
            [(2/3, 2/3, 3/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^5)]
            [(0, 0, 1/2), 2*q^(1/2) + 144*q^(3/2) + 540*q^(5/2) + 1440*q^(7/2) + 1874*q^(9/2) + O(q^5)]
            [(1/3, 1/3, 1/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^5)]
            [(2/3, 2/3, 0), 27*q^(2/3) + 216*q^(5/3) + 513*q^(8/3) + 1512*q^(11/3) + 2268*q^(14/3) + O(q^5)]
            [(0, 0, 3/4), q^(1/8) + 73*q^(9/8) + 342*q^(17/8) + 991*q^(25/8) + 1728*q^(33/8) + O(q^5)]
            [(1/3, 1/3, 1/2), 54*q^(7/6) + 432*q^(13/6) + 918*q^(19/6) + 2160*q^(25/6) + 2754*q^(31/6) + O(q^6)]
            [(2/3, 2/3, 1/4), 27*q^(19/24) + 243*q^(43/24) + 675*q^(67/24) + 1566*q^(91/24) + 2646*q^(115/24) + O(q^5)]

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
            [(2/3, 2/3), 1/9*q^(-1/3) - 328/9*q^(2/3) - 22300/9*q^(5/3) - 132992/3*q^(8/3) - 1336324/3*q^(11/3) - 28989968/9*q^(14/3) + O(q^5)]
            [(1/3, 1/3), 1/9*q^(-1/3) - 328/9*q^(2/3) - 22300/9*q^(5/3) - 132992/3*q^(8/3) - 1336324/3*q^(11/3) - 28989968/9*q^(14/3) + O(q^5)]

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
            [(0, 1/2), 4*q^(1/2) + 8*q^(5/2) + 4*q^(9/2) + O(q^5)]
            [(1/2, 1/2), 2*q^(1/4) + 4*q^(5/4) + 2*q^(9/4) + 4*q^(13/4) + 4*q^(17/4) + O(q^5)]
            [(1/2, 0), 2*q^(1/4) + 4*q^(5/4) + 2*q^(9/4) + 4*q^(13/4) + 4*q^(17/4) + O(q^5)]

            sage: w = WeilRep(matrix([[-2]]))
            sage: w.theta_series(5).conjugate(matrix([[2]]))
            [(0), 1 + 2*q + 2*q^4 + O(q^5)]
            [(7/8), O(q^5)]
            [(3/4), 2*q^(1/4) + 2*q^(9/4) + O(q^5)]
            [(5/8), O(q^5)]
            [(1/2), 1 + 2*q + 2*q^4 + O(q^5)]
            [(3/8), O(q^5)]
            [(1/4), 2*q^(1/4) + 2*q^(9/4) + O(q^5)]
            [(1/8), O(q^5)]
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
                Y[j] = g, -frac(g*S_conj*g/2), O(q ** prec_g)
        return WeilRepModularForm(self.weight(), S_conj, Y, weilrep = w_conj)

    def jacobi_form(self):
        r"""
        Return the Jacobi form associated to self.

        If the Gram matrix is positive-definite (this is not checked!!) then this returns the Jacobi form whose theta-decomposition is the vector valued modular form that we started with.

        OUTPUT: a JacobiForm

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).eisenstein_series(3, 3).jacobi_form()
            1 + (w_0^2*w_1 + w_0*w_1^2 + 27*w_0*w_1 + 27*w_0 + 27*w_1 + w_0*w_1^-1 + 72 + w_0^-1*w_1 + 27*w_1^-1 + 27*w_0^-1 + 27*w_0^-1*w_1^-1 + w_0^-1*w_1^-2 + w_0^-2*w_1^-1)*q + (27*w_0^2*w_1^2 + 72*w_0^2*w_1 + 72*w_0*w_1^2 + 27*w_0^2 + 216*w_0*w_1 + 27*w_1^2 + 216*w_0 + 216*w_1 + 72*w_0*w_1^-1 + 270 + 72*w_0^-1*w_1 + 216*w_1^-1 + 216*w_0^-1 + 27*w_1^-2 + 216*w_0^-1*w_1^-1 + 27*w_0^-2 + 72*w_0^-1*w_1^-2 + 72*w_0^-2*w_1^-1 + 27*w_0^-2*w_1^-2)*q^2 + O(q^3)
        """
        X = self.fourier_expansion()
        S = self.gram_matrix()
        prec = self.precision()
        val = self.valuation()
        e = S.nrows()
        Rb = LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(e) ))
        R.<q> = PowerSeriesRing(Rb,prec)
        if e > 1:
            _ds_dict = self.weilrep().ds_dict()
            jf = [Rb(0)]*(prec-val)
            Q = QuadraticForm(S)
            if not Q.is_positive_definite():
                raise ValueError('Index is not positive definite')
            S_inv = S.inverse()
            precval = prec - val
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
                        jf[i] += (wv + symm * (wv ** (-1))) * f[m]
                        m += 1
                f = X[0][2]#deal with v=0 separately
                for i in range(precval):
                    jf[i] += f[ceil(val) + i]
                return JacobiForm(self.weight() + e/2, S, q^val * R(jf) + O(q ** prec), weilrep = self.weilrep())
            except PariError: #oops!
                pass
            lvl = Q.level()
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
            return JacobiForm(self.weight()+e/2, S, q^val*R(jf)+O(q^prec), weilrep = self.weilrep())
        else:
            w = Rb.0
            m = S[0,0] #twice the index
            eps = 2*self.is_symmetric()-1
            jf = [X[0][2][i] + sum(X[r%m][2][ceil(i - r^2 / (2*m))]*(w^r + eps/w^r) for r in range(1,isqrt(2*(i-val)*m)+1)) for i in range(val, prec)]
            return JacobiForm(self.weight()+1/2, S, q^val * R(jf) + O(q^prec), weilrep = self.weilrep(), modform = self)

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
            [(1/2), 5/12*q^(1/4) + 2*q^(5/4) + 125/12*q^(9/4) + 10*q^(13/4) + 20*q^(17/4) + O(q^5)]

            sage: WeilRep(matrix([[-8]])).zero(1/2, 5).serre_derivative()
            [(0), O(q^5)]
            [(7/8), O(q^5)]
            [(3/4), O(q^5)]
            [(5/8), O(q^5)]
            [(1/2), O(q^5)]
            [(3/8), O(q^5)]
            [(1/4), O(q^5)]
            [(1/8), O(q^5)]

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
        """
        d_b = denominator(b)
        if d_b == 1:
            return self
        R.<q> = PowerSeriesRing(QQ)
        S = self.__gram_matrix
        X = self.components()
        ds = self.ds()
        symm = self.is_symmetric
        if symm:
            eps = 1
        else:
            eps = -1
        indices = self.rds(indices = True)
        norm_list = self.weilrep().norm_list()
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
            [(1/2), -10/67*q^(1/4) - 84816/67*q^(5/4) - 2229850/67*q^(9/4) - 16356240/67*q^(13/4) - 73579680/67*q^(17/4) + O(q^5)]

         """
        symm = self.is_symmetric()
        prec = self.precision()
        R.<q> = PowerSeriesRing(QQ)
        big_S = self.gram_matrix()
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
        bound = 3 + 2*isqrt(m * (prec - self.valuation()))
        if components:
            _ds, _indices = components
        else:
            if not weilrep:
                weilrep = WeilRep(S)
            _ds = weilrep.ds()
            _indices = weilrep.rds(indices = True)
            #[_ds,_reduced_indices] = reduced_discriminant_group(S, reduced_list = True)
        big_ds_dict = {tuple(X[i][0]) : i for i in range(len(X))}
        b_denom = b.denominator()
        bm2 = ZZ(2*m*b_denom)
        Y = [None] * len(_ds)
        eps = (2 * (odd != symm) - 1)
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
                Y[i] = g, -offset, sum([R(theta_twist[j]) * X[g_ind[j]][2] for j in range(min(bm2, len(g_ind)))])+O(q ** prec_g)
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

    The main purpose of this class is to print lists of modular forms with a line of hyphens as delimiter.
    """

    def append(self, other):
        r"""
        Append a WeilRepModularForm to self.
        """
        self.__basis.append(other)

    def echelonize(self, save_pivots = False, starting_from = None, ending_with = None):
        r"""
        Reduce self to echelon form in place.

        INPUT:
        - ``save_pivots`` -- if True then return the pivot columns. (Otherwise we return None)
        - ``starting_from`` -- (default 0) the index at which we start looking at Fourier coefficients
        - ``ending_with`` -- (default None) if given then it should be the index at which we stop looking at Fourier coefficients.
        """
        if ending_with is None:
            ending_with = self.__weight / 12
        if starting_from is None:
            starting_from = self.valuation()
        m = matrix([v.coefficient_vector(starting_from = starting_from, ending_with = ending_with) for v in self.__basis]).extended_echelon_form(subdivide = True)
        b = m.subdivision(0, 1)
        self.__basis = [self * v for v in b.rows()]
        if save_pivots:
            a = m.subdivision(0, 0)
            return [next(j for j, w in enumerate(v) if w) for v in a.rows()]

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

    def __init__(self, weight, basis, weilrep):
        self.__weight = weight
        self.__basis = basis
        self.__weilrep = weilrep

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
                return [JacobiForm(k, S, q^val * R(x) + O(q ** prec), weilrep = self.__weilrep, modform = x) for x in jf]
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
            return [JacobiForm(k, S, q^val*R(x)+O(q^prec), weilrep = self.__weilrep, modform = x) for x in jf]
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
            return [JacobiForm(k, S, q^val * R(x) + O(q^prec), weilrep = self.__weilrep, modform = x) for x in jf]

    def __len__(self):
        return len(self.__basis)

    def list(self):
        return self.__basis

    def __mul__(self, v):
        return sum([self.__basis[i] * w for i, w in enumerate(v)])

    def precision(self):
        return min(x.precision() for x in self.__basis)

    def __repr__(self):
        X = self.__basis
        if X:
            s = '\n' + '-'*60 + '\n'
            return s.join([x.__repr__() for x in self.__basis])
        return '[]'

    def reverse(self):
        self.__basis.reverse()

    __rmul__ = __mul__

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