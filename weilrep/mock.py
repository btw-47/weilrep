r"""

Sage code for vector-valued quasimodular forms and mock modular forms

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

import cmath
import math

from re import sub
from scipy.special import gamma, hyperu

from sage.functions.generalized import sgn
from sage.functions.other import ceil, factorial, floor, sqrt
from sage.misc.cachefunc import cached_method
from sage.misc.latex import latex
from sage.modules.free_module_element import vector
from sage.plot.complex_plot import complex_plot
from sage.rings.all import CC
from sage.rings.big_oh import O
from sage.rings.infinity import Infinity, SignError
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi

from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis, WeilRepModularFormWithCharacter

class WeilRepQuasiModularForm(WeilRepModularForm):
    r"""
    Class for quasimodular forms.

    A quasimodular form is the holomorphic part of an almost-holomorphic modular form i.e. a function f that satisfies modular transformations and has the expansion
    f(\tau) = f_0(\tau) + f_1(\tau) (4\pi y)^(-1) + ... + f_r(\tau) (4 \pi y)^(-r)

    A WeilRepQuasiModularForm instance is constructed using
    WeilRepQuasiModularForm(k, S, list_f)
    by providing
    - ``k `` -- the weight
    - ``S`` -- the gram matrix
    - ``list_f`` -- a list of WeilRepModularForm's f_r, ..., f_1, f_0

    This represents the form f_0(\tau) + f_1(\tau) (4\pi y)^(-1) + ... + f_r(\tau) (4 \pi y)^(-r).

    A WeilRepQuasiModularForm instance acts as the WeilRepModularForm f_0 (which is a quasimodular form).
    """

    def __init__(self, k, S, terms, weilrep=None):
        l = len(terms)
        try:
            j = next(j for j, x in enumerate(terms) if x)
        except StopIteration:
            j = l
        f = terms[-1].fourier_expansion()
        if weilrep is None:
            weilrep = terms[-1].weilrep()
        super().__init__(k, S, f, weilrep=weilrep)
        l -= 1
        if j < l:
            self.__class__ = WeilRepQuasiModularForm
            self.__terms = terms[j:]
            self.__depth = l - j
        else:
            self.__class__ = WeilRepModularForm

    def __add__(self, other):
        r"""
        Add quasimodular forms.
        """
        if not other:
            return self
        S1 = self.gram_matrix()
        S2 = other.gram_matrix()
        if S1 != S2:
            raise ValueError('Incompatible Gram matrices')
        if self.weight() != other.weight():
            raise ValueError('Incompatible weights')
        d1 = self.depth()
        d2 = other.depth()
        i = d1 <= d2
        t1 = self._terms()
        t2 = other._terms()
        j = d2-d1
        if i:
            X = t2[:j] + [t1[i]+t2[i+j] for i in range(d1 + 1)]
        else:
            X = t1[:-j] + [t1[i-j] + t2[i] for i in range(d2 + 1)]
        return WeilRepQuasiModularForm(self.weight(), S1, X, weilrep = self.weilrep())

    __radd__ = __add__

    def __sub__(self, other):
        r"""
        Subtract quasimodular forms.
        """
        if not other:
            return self
        S1 = self.gram_matrix()
        S2 = other.gram_matrix()
        if S1 != S2:
            raise ValueError('Incompatible Gram matrices')
        if self.weight() != other.weight():
            raise ValueError('Incompatible weights')
        d1 = self.depth()
        d2 = other.depth()
        i = d1 <= d2
        t1 = self._terms()
        t2 = other._terms()
        j = d2-d1
        if i:
            X = [-x for x in t2[:j]] + [t1[i]-t2[i+j] for i in range(d1 + 1)]
        else:
            X = t1[:-j] + [t1[i-j] - t2[i] for i in range(d2 + 1)]
        return WeilRepQuasiModularForm(self.weight(), S1, X, weilrep = self.weilrep())

    def __neg__(self):
        r"""
        Negative of quasimodular form
        """
        return WeilRepQuasiModularForm(self.weight(), self.gram_matrix(), [-x for x in self._terms()], weilrep=self.weilrep())

    def __mul__(self, other, w = None):
        r"""
        (Tensor) Product of quasimodular forms
        """
        if other in CC:
            return WeilRepQuasiModularForm(self.weight(), self.gram_matrix(), [other*x for x in self._terms()], weilrep=self.weilrep())
        k = self.weight() + other.weight()
        if w is None:
            w1 = self.weilrep()
            w2 = other.weilrep()
            w = w1 + w2
        t1 = self._terms()
        t2 = other._terms()
        r = self.depth()
        s = other.depth()
        rs = r + s
        j = k - rs - rs
        p = self.precision()
        X = [w.zero(i + i + j, p) for i in range(rs + 1)]
        for i in range(r + 1):
            for j in range(s + 1):
                X[i + j] += t1[i]*t2[j]
        return WeilRepQuasiModularForm(k, w.gram_matrix(), X, weilrep=w)

    __rmul__ = __mul__

    def __div__(self, other):
        r"""
        Division of quasimodular forms.
        """
        try:
            if other.is_modular():
                return WeilRepQuasiModularForm(self.weight() - other.weight(), self.gram_matrix(), [x/other for x in self._terms()], weilrep = self.weilrep())
            else:
                return NotImplemented
        except AttributeError:
            pass
        if other in CC:
            return WeilRepQuasiModularForm(self.weight(), self.gram_matrix(), [x/other for x in self._terms()], weilrep=self.weilrep())
        return NotImplemented

    __truediv__ = __div__

    def __rdiv__(self, other):
        raise NotImplementedError

    __rtruediv__ = __rdiv__

    def __invert__(self):
        raise NotImplementedError

    def __pow__(self, N):
        r"""
        Tensor power of quasimodular form with itself.
        """
        if N == 1:
            return self
        elif N > 1:
            Nhalf = floor(N / 2)
            return (self ** Nhalf) * (self ** (N - Nhalf))
        raise ValueError('Invalid exponent')

    def depth(self):
        r"""
        Computes the "depth" of the quasimodular form.
        """
        try:
            return self.__depth
        except AttributeError:
            return 0

    def _terms(self):
        r"""
        Returns self's Taylor expansion in (4pi y)^(-1) as a list of WeilRepModularForm's.

        This should not be called directly because the terms in self's Taylor expansion are not modular forms (although we treat them as if they were).
        """
        try:
            return self.__terms
        except AttributeError:
            self.__class__ = WeilRepModularForm
            return [self]

    def is_modular(self):
        return False

    def completion(self):
        r"""
        Return self's completion to an almost-holomorphic modular form.

        EXAMPLES::
            sage: from weilrep import WeilRep
            sage: w = WeilRep([])
            sage: e2 = w.eisenstein_series(2, 5)
            sage: e2.completion()
            Almost holomorphic modular form f_0 + f_1 * (4 pi y)^(-1), where:
            f_0 =
            1 - 24*q - 72*q^2 - 96*q^3 - 168*q^4 + O(q^5)
            --------------------------------------------------------------------------------
            f_1 =
            -12 + O(q^5)

            sage: from weilrep import WeilRep
            sage: w = WeilRep(matrix([[-2, 0], [0, -2]]))
            sage: r.<x, y> = PolynomialRing(ZZ)
            sage: f = w.theta_series(5, P = x^2)
            sage: f.completion()
            Almost holomorphic modular form f_0 + f_1 * (4 pi y)^(-1), where:
            f_0 =
            [(0, 0), 2*q + 4*q^2 + 8*q^4 + O(q^5)]
            [(1/2, 0), 1/2*q^(1/4) + q^(5/4) + 9/2*q^(9/4) + 9*q^(13/4) + q^(17/4) + O(q^(21/4))]
            [(0, 1/2), 4*q^(5/4) + 4*q^(13/4) + 16*q^(17/4) + O(q^(21/4))]
            [(1/2, 1/2), q^(1/2) + 10*q^(5/2) + 9*q^(9/2) + O(q^(11/2))]
            --------------------------------------------------------------------------------
            f_1 =
            [(0, 0), -1/2 - 2*q - 2*q^2 - 2*q^4 + O(q^5)]
            [(1/2, 0), -q^(1/4) - 2*q^(5/4) - q^(9/4) - 2*q^(13/4) - 2*q^(17/4) + O(q^(21/4))]
            [(0, 1/2), -q^(1/4) - 2*q^(5/4) - q^(9/4) - 2*q^(13/4) - 2*q^(17/4) + O(q^(21/4))]
            [(1/2, 1/2), -2*q^(1/2) - 4*q^(5/2) - 2*q^(9/2) + O(q^(11/2))]
        """
        X = [self]
        r = self.depth()
        j = 1
        for i in range(r):
            j *= (i + 1)
            X.append(X[-1].shift() / j)
        return WeilRepAlmostHolomorphicModularForm(self.weight(), self.gram_matrix(), X, weilrep = self.weilrep(), character = self.character())

    def derivative(self):
        r"""
        Compute self's derivative. This is a quasimodular form whose depth is (usually) one greater than self.

        EXAMPLES::
            sage: from weilrep import WeilRep
            sage: w = WeilRep([])
            sage: e2 = w.eisenstein_series(2, 5)
            sage: e2.derivative()
            -24*q - 144*q^2 - 288*q^3 - 672*q^4 + O(q^5)
        """
        t = self._terms()
        S = self.gram_matrix()
        w = self.weilrep()
        R, q = self.fourier_expansion()[0][2].parent().objgen()
        def a(offset, f):
            if not f:
                return O(q ** f.prec())
            val = f.valuation()
            prec = f.prec()
            return (q**val * R([(i + offset) * f[i] for i in range(val, prec)])).add_bigoh(prec - floor(offset))
        chi = self.character()
        if chi:
            def d(X):
                k = X.weight()
                Y = [(x[0], x[1], a(x[1], x[2])) for x in X.fourier_expansion()]
                return WeilRepModularFormWithCharacter(k + 2, S, Y, weilrep = w, character = chi)
        else:
            def d(X):
                k = X.weight()
                Y = [(x[0], x[1], a(x[1], x[2])) for x in X.fourier_expansion()]
                return WeilRepModularForm(k + 2, S, Y, w)
        r = self.depth()
        k = self.weight()
        X = [(r - k) * t[0]] + [(r - k - j) * t[j] + d(t[j-1]) for j in range(1, r+1)] + [d(t[-1])]
        return WeilRepQuasiModularForm(k + 2, self.gram_matrix(), X, weilrep = self.weilrep())

    def character(self):
        return self.__terms[0].character()

    def serre_derivative(self):
        from weilrep import WeilRep
        return self.derivative() - (QQ(self.weight()) / 12) * (self * WeilRep([]).eisenstein_series(2, self.precision()))

    def raising_operator(self):
        return NotImplemented

    def lowering_operator(self):
        return NotImplemented

    def shift(self):
        r"""
        Apply the "shift operator" on quasimodular forms. This is a quasimodular form whose depth is one less than self.

        EXAMPLES::
            sage: from weilrep import WeilRep
            sage: w = WeilRep([])
            sage: e2 = w.eisenstein_series(2, 5)
            sage: e2.shift()
            -12 + O(q^5)
        """
        r = self.depth()
        f = [1]*(r + 1)
        k = r - 1
        for j in range(1, r + 1):
            f[k] = j * f[k + 1]
            k -= 1
        X = self._terms()
        return WeilRepQuasiModularForm(self.weight() - 2, self.gram_matrix(), [f[i]*y for i, y in enumerate(X[:-1])], weilrep=self.weilrep())

    def reduce_precision(self, prec, in_place = False):
        r"""
        Reduce self's precision.

        Overwrite the reduce_precision() method for holomorphic modular forms to return a quasimodular form, all of whose shifts precisions are lowered.
        """
        if in_place:
            for x in self.__terms:
                x.reduce_precision(prec, in_place = True)
        else:
            return WeilRepQuasiModularForm(self.weight(), self.gram_matrix(), [x.reduce_precision(prec) for x in self._terms()], weilrep = self.weilrep())

    def hecke_P(self, N):
        return WeilRepQuasiModularForm(self.weight(), self.gram_matrix() / (N * N), [x.hecke_P(N) for x in self._terms()], weilrep = self.weilrep())

    def hecke_T(self, N):
        return WeilRepQuasiModularForm(self.weight(), self.gram_matrix(), [x.hecke_T(N) for x in self._terms()], weilrep = self.weilrep())

    def hecke_U(self, N):
        return WeilRepQuasiModularForm(self.weight(), self.gram_matrix() * N * N, [x.hecke_U(N) for x in self._terms()])

    def hecke_V(self, N):
        return WeilRepQuasiModularForm(self.weight(), self.gram_matrix() * N, [x.hecke_V(N) for x in self._terms()])

    def bol(self):
        k = self.weight()
        try:
            k = Integer(k)
        except TypeError:
            raise TypeError('Invalid weight') from None
        if k == 1:
            return self
        elif k == 0:
            return self.derivative()
        return (self.derivative()).bol().derivative()

    def theta_contraction(self):
        X = WeilRepModularFormsBasis(self.weight(), self._terms(), self.weilrep()).theta()
        return WeilRepQuasiModularForm(X.weight(), X.gram_matrix(), list(X), X.weilrep())

class WeilRepAlmostHolomorphicModularForm:
    r"""
    Provide a custom string representation for the 'completion' of a quasimodular form. Also implements arithmetic operations and the Maass raising operator.

    Currently the only way to construct a WeilRepAlmostHolomorphicModularForm is as the modular completion of a quasimodular form.
    """
    def __init__(self, k, S, X, weilrep=None, character = None):
        self.__weight = k
        self.__gram_matrix = S
        self.__list = X
        if weilrep:
            self.__weilrep = weilrep
        self.__depth = X[0].depth()
        if character:
            self.__character = character
            self.__list = [WeilRepModularFormWithCharacter(x.weight(), x.gram_matrix(), x.fourier_expansion(), weilrep = self.weilrep(), character = character) for x in X]

    def __repr__(self):
        try:
            return self.__string
        except AttributeError:
            r = self.__depth
            X = self.__list
            s = 'Almost holomorphic modular form f_0'
            t = str(X[0])
            h = '\n'+'-'*80
            for i, x in enumerate(X):
                if i >= 1:
                    s, t = s + ' + f_%d * (4 pi y)^(-%d)'%(i, i), t + h + '\nf_%d =\n%s'%(i, str(x))
            x = s + ', where:\nf_0 =\n' + t
            self.__string = x
            return x

    def __getitem__(self, n):
        return self.__list[n]

    def __iter__(self):
        for x in self.__list:
            yield x

    def weight(self):
        return self.__weight

    def gram_matrix(self):
        return self.__gram_matrix

    def depth(self):
        return self.__depth

    def precision(self):
        return self[0].precision()

    def valuation(self):
        return self[0].valuation()

    def coefficient_vector(self):
        return self[0].coefficient_vector()

    def weilrep(self):
        try:
            return self.__weilrep
        except AttributeError:
            from .weilrep import WeilRep
            self.__weilrep = WeilRep(self.__gram_matrix)
            return self.__weilrep

    ## lazy arithmetic ##

    def __add__(self, other):
        if not other:
            return self
        try:
            return (self[0] + other[0]).completion()
        except AttributeError:
            return (self[0] + other).completion()

    __radd__ = __add__

    def __sub__(self, other):
        if not other:
            return self
        try:
            return (self[0] - other[0]).completion()
        except AttributeError:
            return (self[0] - other).completion()

    def __mul__(self, other, **kwargs):
        try:
            if other in QQ:
                return (other * self[0]).completion()
            try:
                return (self[0].__mul__(other[0], **kwargs)).completion()
            except AttributeError:
                return (self[0] * other).completion()
        except TypeError:
            print('self:', self)

    __rmul__ = __mul__

    def __neg__(self):
        return (-self[0]).completion()

    def __div__(self, n):
        return (self[0] / n).completion()
    __truediv__ = __div__

    def __pow__(self, n):
        return (self[0]**n).completion()

    def derivative(self):
        return NotImplemented

    def character(self):
        return self.__character

    def __getattr__(self, x):
        try:
            h = self[0].__getattribute__(x)
            def a(*args, **kwargs):
                return h(*args, **kwargs).completion()
            return a
        except AttributeError as e:
            raise e

    def raising_operator(self):
        return self[0].derivative().completion()

    def shift(self):
        return self[0].shift().completion()

    lowering_operator = shift

    def holomorphic_part(self):
        return self[0]

    ## evaluate at points. partially copied from holomorphic case

    def __call__(self, z, q = False, funct = None, cayley = False):
        if funct is None:
            funct = self.__call__
        if q:
            return self.__call__(cmath.log(z) / complex(0.0, 2 * math.pi), funct = funct)
        elif cayley:
            return self.__call__(complex(0.0, 1.0) * (1 + z) / (1 - z), funct = funct)
        if 0 < abs(z) < 1:
            z = -1 / z
            return (z ** self.weight()) * (self.weilrep()._evaluate(0, -1, 1, 0) * funct(z))
        else:
            try:
                y = z.imag()
            except TypeError:
                y = z.imag
            if y <= 0:
                raise ValueError('Not in the upper half plane.')
            elif y < 0.5:
                try:
                    x = z.real()
                except TypeError:
                    x = z.real
                if abs(x) > 0.5:
                    try:
                        f = x.round()
                    except AttributeError:
                        f = round(x)
                    return self.weilrep()._evaluate(1, f, 0, 1) * funct(z - f)
                return (z ** self.weight()) * (self.weilrep()._evaluate(0, -1, 1, 0) * funct(z))
        four_pi_y_inv = 1 / (4 * math.pi * y)
        return sum(x.__call__(z) * four_pi_y_inv ** j for j, x in enumerate(self))

    @cached_method
    def _cached_call(self, z, isotherm = False, **kwargs):
        _ = kwargs.pop('funct', None)
        s = self.__call__(z, funct = self._cached_call, **kwargs)
        if isotherm:
            v = [0] * len(s)
            for i, x in enumerate(s):
                if x == 0.0:
                    v[i] = x
                else:
                    c = abs(x)
                    v[i] = x * 2 * math.frexp(c)[0] / c
            return v
        return s

    def plot(self, x_range = [-1, 1], y_range = [0.01, 2], isotherm = True, **kwargs):
        r"""
        Plot self on the upper half-plane.

        See the plot() method from weilrep_modular_forms_class.py
        """
        if isotherm and 'plot_points' not in kwargs:
            kwargs['plot_points'] = 150
        function = kwargs.pop('function', None)
        if function is not None:
            f = lambda z: function(self._cached_call(z, isotherm = isotherm))
            self._cached_call.clear_cache()
            return complex_plot(f, x_range, y_range, **kwargs)
        f = lambda i: lambda z: self._cached_call(z, isotherm = isotherm)[i]
        L = []
        rds = self.weilrep().rds(indices = True)
        ds = self.weilrep().ds()
        for i, x in enumerate(rds):
            if x is None and self[i]:
                L.append(complex_plot(f(i), x_range, y_range, **kwargs))
                print('Component %s:'%ds[i])
                L[-1].show()
        self._cached_call.clear_cache()
        return L

    def plot_cayley(self, **kwargs):
        kwargs['_cayley'] = True
        return self.plot_q(**kwargs)

    def plot_q(self, isotherm = True, show = True, **kwargs):
        r"""
        Plot self on the unit disc.

        See the plot_q() method from weilrep_modular_forms_class.py
        """
        if 'figsize' not in kwargs:
            kwargs['figsize'] = [6, 6]
        if isotherm and 'plot_points' not in kwargs:
            kwargs['plot_points'] = 150
        function = kwargs.pop('function', None)
        cayley = kwargs.pop('_cayley', None)
        if cayley:
            q, cayley = False, True
        else:
            q, cayley = True, False
        if function is not None:
            f = lambda z: function(self._cached_call(z, q = q, cayley = cayley, isotherm = isotherm)) if abs(z) < 1 else Infinity
            self._cached_call.clear_cache()
            return complex_plot(f, [-1, 1], [-1, 1], **kwargs)
        f = lambda i: (lambda z: self._cached_call(z, q = q, cayley = cayley, isotherm = isotherm)[i] if abs(z) < 1 else Infinity)
        L = []
        rds = self.weilrep().rds(indices = True)
        ds = self.weilrep().ds()
        for i, x in enumerate(rds):
            if x is None and self[i]:
                L.append(complex_plot(f(i), [-1, 1], [-1, 1], **kwargs))
                if show:
                    print('Component %s:'%ds[i])
                    L[-1].show()
        self._cached_call.clear_cache()
        return L


class WeilRepMixedModularForm(object):

    def __init__(self, y_power, weight, gram_matrix, fourier_expansions, weilrep = None):
        self.__weight = weight
        self.__gram_matrix = gram_matrix
        self.__fourier_expansions = fourier_expansions
        if weilrep is None:
            from .weilrep import WeilRep
            weilrep = WeilRep(gram_matrix)
        self.__weilrep = weilrep
        self.__y_power = y_power

    def __repr__(self): #represent as a list of pairs (g, f) where g is in the discriminant group and f is a q-series with fractional exponents
        try:
            return self.__qexp_string
        except AttributeError:
            r = r'((?<!\w)q(bar)?(?!\w)(\^-?\d+)?)|((?<!\^)\d+\s)'
            X = self.__fourier_expansions
            def a(x1, x2):
                def b(y):
                    y = y.string[slice(*y.span())]
                    if y[0] != 'q':
                        return '%sq^(%s)*qbar^(%s) '%([y[:-1]+'*',''][y == '1 '], x1, x2)
                    try:
                        return 'q^(%s)'%(QQ(y[2:]) + x)
                    except TypeError:
                        return 'q^(%s)'%(1 + x)
                return b
            if self.weilrep():
                s = '\n'.join(['[%s, %s]'%(x[0], sub(r, a(x[1], x[2]), str(x[3]))) if x[1] else '[%s, %s]'%(x[0], x[2]) for x in X])
            else:
                s = str(X[0][2])
            self.__qexp_string = s
            return s


class WeilRepMockModularForm(WeilRepModularForm):
    r"""
    Class for mock modular forms.
    """

    def __init__(self, k, S, X, shadow, **kwargs):
        WeilRepModularForm.__init__(self, k, S, X, weilrep = kwargs.pop('weilrep', None))
        self.__class__ = WeilRepMockModularForm
        self.__shadow = shadow
        self.__multiplier = kwargs.pop('multiplier', Integer(1))
        try:
            self.__multiplier_n = kwargs.pop('multiplier_n', self.__multiplier.n())
        except AttributeError:
            self.__multiplier_n = Integer(self.__multiplier).n()
        self.__shadow_multiplier = kwargs.pop('shadow_multiplier', Integer(1))
        try:
            self.__shadow_multiplier_n = kwargs.pop('shadow_multiplier_n', self.__shadow_multiplier.n())
        except AttributeError:
            self.__shadow_multiplier_n = Integer(self.__shadow_multiplier).n()

    def __repr__(self):
        m = self.__multiplier
        if m != 1:
            return '%s times\n'%m + super().__repr__()
        return super().__repr__()


    def shadow(self):
        return self.__shadow

    def shadow_multiplier(self):
        return self.__shadow_multiplier

    def __add__(self, other):
        f = super().__add__(other)
        s = self.shadow()
        if isinstance(other, WeilRepMockModularForm):
            s += other.shadow()
        return WeilRepMockModularForm(self.weight(), self.gram_matrix(), f.fourier_expansion(), s, weilrep = self.weilrep())

    def __sub__(self, other):
        f = super().__sub__(other)
        s = self.shadow()
        if isinstance(other, WeilRepMockModularForm):
            s -= other.shadow()
        return WeilRepMockModularForm(self.weight(), self.gram_matrix(), f.fourier_expansion(), s, weilrep = self.weilrep())

    def __neg__(self):
        return WeilRepMockModularForm(self.weight(), self.gram_matrix(), super().__neg__().fourier_expansion(), -self.shadow(), weilrep = self.weilrep())

    def __mul__(self, other):
        if other in QQ:
            f = super().__mul__(other)
            return WeilRepMockModularForm(self.weight(), self.gram_matrix(), f.fourier_expansion(), self.shadow() * other, weilrep = self.weilrep())
        raise NotImplementedError

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.__mul__(~other)
    __div__ = __truediv__

    def derivative(self):
        raise NotImplementedError

    def completion(self):
        return WeilRepWeakMaassForm(self.weight(), self.gram_matrix(), self.fourier_expansion(), self.shadow(), multiplier = self.__multiplier, multiplier_n = self.__multiplier_n, shadow_multiplier = self.shadow_multiplier(), shadow_multiplier_n = self.__shadow_multiplier_n, weilrep = self.weilrep(), holomorphic_part = self)

    def n(self):
        return WeilRepMockModularForm(self.weight(), self.gram_matrix(), super().n().__mul__(self.__multiplier_n).fourier_expansion(), self.shadow().n(), weilrep = self.weilrep())

    def bol(self):
        X = super().bol()
        Y = self.shadow()
        Z = X.fourier_expansion()
        for i, z in enumerate(Z):
            if not z[1]:
                Z[i] = z[0], 0, z[2] + Y[tuple(z[0])][0]
        return WeilRepModularForm(X.weight(), X.gram_matrix(), Z, weilrep = self.weilrep())

    def flip(self): ##does not work correctly??
        k = self.weight()
        try:
            k = ZZ(k)
        except TypeError:
            return NotImplemented
        u = factorial(-k)
        return WeilRepMockModularForm(k, self.gram_matrix(), (u * self.shadow()).fourier_expansion(), self.bol() / u, weilrep = self.weilrep() )

    def is_modular(self):
        return False

    def is_quasimodular(self):
        return False

    def reduce_precision(self, prec, **kwargs):
        f = super().reduce_precision(prec)
        g = self.shadow().reduce_precision(prec)
        return WeilRepMockModularForm(self.weight(), self.gram_matrix(), f.fourier_expansion(), g, multiplier = self.__multiplier, shadow_multipler = self.__shadow_multiplier)

    def theta_contraction(self):
        raise NotImplementedError

    def hecke_P(self, N):
        raise NotImplementedError
        #X = super().hecke_P(N)
        #return WeilRepMockModularForm(self.weight(), X.gram_matrix(), X.fourier_expansion(), self.shadow().hecke_P(N), weilrep = X.weilrep(), multiplier = self.__multiplier, shadow_multiplier = self.__shadow_multiplier)

    def hecke_T(self, N):
        raise NotImplementedError
        #X = super().hecke_T(N)
        #return WeilRepMockModularForm(self.weight(), X.gram_matrix(), X.fourier_expansion(), self.shadow().hecke_T(N), weilrep = X.weilrep(), multiplier = self.__multiplier, shadow_multiplier = self.__shadow_multiplier)

    def hecke_U(self, N):
        raise NotImplementedError
        #X = super().hecke_U(N)
        #return WeilRepMockModularForm(self.weight(), X.gram_matrix(), X.fourier_expansion(), self.shadow().hecke_U(N), weilrep = X.weilrep(), multiplier = self.__multiplier, shadow_multiplier = self.__shadow_multiplier)

    def hecke_V(self, N):
        raise NotImplementedError
        #X = super().hecke_V(N)
        #return WeilRepMockModularForm(self.weight(), X.gram_matrix(), X.fourier_expansion(), self.shadow().hecke_V(N), weilrep = X.weilrep(), multiplier = self.__multiplier, shadow_multiplier = self.__shadow_multiplier * sqrt(self.weilrep().discriminant()))


class WeilRepWeakMaassForm(WeilRepModularForm):
    r"""
    Class for weak Maass forms.
    """

    def __init__(self, k, S, X, shadow, **kwargs):
        WeilRepModularForm.__init__(self, k, S, X, weilrep = kwargs.pop('weilrep', None))
        self.__class__ = WeilRepWeakMaassForm
        self.__shadow = shadow
        self.__multiplier = kwargs.pop('multiplier', Integer(1))
        try:
            self.__multiplier_n = kwargs.pop('multiplier_n', self.__multiplier.n())
        except AttributeError:
            self.__multiplier_n = Integer(self.__multiplier).n()
        self.__shadow_multiplier = kwargs.pop('shadow_multiplier', Integer(1))
        try:
            self.__shadow_multiplier_n = kwargs.pop('shadow_multiplier_n', self.__shadow_multiplier.n())
        except AttributeError:
            self.__shadow_multiplier_n = Integer(self.__shadow_multiplier).n()

    def __repr__(self):
        s = self.__shadow_multiplier
        s = ' %s times'%s if s != 1 else ''
        t = self.__multiplier
        t = ' %s times'%t if t != 1 else ''
        w = '' if self.is_maass_form() else 'weak '
        return 'Harmonic %sMaass form with holomorphic part%s\n%s\nand shadow%s\n%s'%(w, t, super().__repr__(), s, self.__shadow.__repr__())

    def __call__(self, z, q = False, funct = None):
        if funct is None:
            funct = self.__call__
        if q:
            return self.__call__(cmath.log(z) / complex(0.0, 2 * math.pi), funct = funct)
        try:
            y = z.imag()
        except TypeError:
            y = z.imag
        if 0 < abs(z) < 1:
            z = -1 / z
            return (z ** self.weight()) * (self.weilrep()._evaluate(0, -1, 1, 0) * funct(z))
        else:
            try:
                y = z.imag()
            except TypeError:
                y = z.imag
            if y <= 0:
                raise ValueError('Not in the upper half plane.')
            elif y < 0.5:
                try:
                    x = z.real()
                except TypeError:
                    x = z.real
                if abs(x) > 0.5:
                    try:
                        f = x.round()
                    except AttributeError:
                        f = round(x)
                    return self.weilrep()._evaluate(1, f, 0, 1) * funct(z - f)
                return (z ** self.weight()) * (self.weilrep()._evaluate(0, -1, 1, 0) * funct(z))
        two_pi = 2 * math.pi
        exp = cmath.exp
        try:
            zbar = z.conjugate()
        except TypeError:
            zbar = z.conjugate
        f = self.holomorphic_part().n().__call__(z)
        four_pi_y = 2 * two_pi * y
        k = self.weight()
        psi = lambda n: hyperu(k, k, four_pi_y * n) * (2 * two_pi * n) ** (k - 1)
        d = self.xi().coefficients()
        dsdict = self.weilrep().ds_dict()
        normlist= self.weilrep().norm_list()
        z = complex(0.0, 0.0)
        def correction():
            s = vector([z for _ in f])
            for g, a in d.items():
                i = dsdict[tuple(g[:-1])]
                n = g[-1]
                if n:
                    s[i] += psi(n) * exp(-complex(0.0, two_pi) * zbar * n) * a
                else:
                    s[i] += y ** (1 - k) * a / (k - 1)
            return s
        try:
            s = correction()
        except TypeError:
            y = y.n()
            four_pi_y = 2 * two_pi * y
            psi = lambda n: hyperu(k, k, four_pi_y * n) * (2 * two_pi * n) ** (k - 1)
            s = correction()
        a = f - s * self.__shadow_multiplier_n
        return a

    @cached_method
    def _cached_call(self, z, q = False, isotherm = False, f = None):
        s = self.__call__(z, q = q, funct = self._cached_call)
        if f is not None:
            s = f(s)
            if isotherm:
                if s == 0.0:
                    return s
                else:
                    c = abs(s)
                    return 2 * s * math.frexp(c)[0] / c
        elif isotherm:
            v = [0] * len(s)
            for i, x in enumerate(s):
                if x == 0.0:
                    v[i] = x
                else:
                    c = abs(x)
                    v[i] = 2 * x * math.frexp(c)[0] / c
            return v
        return s

    def _latex_(self):
        return self.__repr__(_flag = 'latex')

    def holomorphic_part(self):
        try:
            return self.__holomorphic_part
        except AttributeError:
            self.__holomorphic_part = WeilRepMockModularForm(self.weight(), self.gram_matrix(), self.fourier_expansion(), self.xi(), multiplier = self.__multiplier, shadow_multiplier = self.__shadow_multiplier, weilrep = self.weilrep())
            return self.__holomorphic_part

    def __add__(self, other):
        f = self.holomorphic_part()
        g = other.holomorphic_part()
        N = self.shadow_multiplier()
        return WeilRepWeakMaassForm(self.weight(), self.gram_matrix(), (f + g).fourier_expansion(), self.shadow() + other.shadow() * (other.shadow_multiplier() / N), shadow_multiplier = N, weilrep = self.weilrep())

    def __sub__(self, other):
        f = self.holomorphic_part()
        g = other.holomorphic_part()
        N = self.shadow_multiplier()
        return WeilRepWeakMaassForm(self.weight(), self.gram_matrix(), (f - g).fourier_expansion(), self.shadow() - other.shadow() * (other.shadow_multiplier() / N), shadow_multiplier = N, weilrep = self.weilrep())

    def __mul__(self, N):
        if N in QQ:
            return WeilRepWeakMaassForm(self.weight(), self.gram_matrix(), [[x[0], x[1], x[2]*N] for x in self.fourier_expansion()], self.shadow() * N, shadow_multiplier = self.shadow_multiplier(), weilrep = self.weilrep())
        raise NotImplementedError

    __rmul__ = __mul__

    def __div__(self, N):
        return self.__mul__(1 / N)

    __truediv__ = __div__

    def __neg__(self):
        return self.__mul__(-1)

    def __pow__(self, N):
        raise NotImplementedError

    def xi(self):
        return self.__shadow
    shadow = xi

    def shadow_multiplier(self):
        return self.__shadow_multiplier

    def is_maass_form(self):
        try:
            return self.__is_maass_form
        except AttributeError:
            self.__is_maass_form = self.holomorphic_part().is_holomorphic() and self.shadow().is_holomorphic()
            return self.__is_maass_form

    def n(self):
        return WeilRepWeakMaassForm(self.weight(), self.gram_matrix(), (super().n() * self.__multiplier_n).fourier_expansion(), self.shadow().n() * self.__shadow_multiplier_n, weilrep = self.weilrep())

    def reduce_precision(self, prec):
        return self.holomorphic_part().reduce_precision(prec).completion()

    def hecke_P(self, N):
        return self.holomorphic_part().hecke_P(N).completion()

    def hecke_T(self, N):
        return self.holomorphic_part().hecke_T(N).completion()

    def hecke_U(self, N):
        return self.holomorphic_part().hecke_U(N).completion()

    def hecke_V(self, N):
        return self.holomorphic_part().hecke_V(N).completion()

    def theta_contraction(self):
        raise NotImplementedError