from copy import copy


from sage.rings.all import CC
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ

from .weilrep import WeilRep
from .weilrep_modular_forms_class import smf_eisenstein_series, WeilRepModularFormsBasis


class FourierJacobiSeries:
    r"""
    Formal Fourier-Jacobi series.

    ** INCOMPLETE **
    """
    def __init__(self, X):
        self.__weight = X[0].weight()
        self.__gram_matrix = X[1].gram_matrix()
        self.__coefficients = [None]*len(X)
        self.__coefficients[0] = X[0]
        for i, x in enumerate(X[1:]):
            f = copy(x)
            f.flag = 'jacobi_form'
            self.__coefficients[i + 1] = f

    def __repr__(self):
        return 'Formal Fourier-Jacobi series with coefficients\n%s' % ('\n%s\n' % ('-' * 80)).join(map(str, self.fourier_jacobi()))

    def fourier_jacobi(self):
        return self.__coefficients

    def gram_matrix(self):
        return self.__gram_matrix

    def weight(self):
        return self.__weight

    def __add__(self, other):
        X = self.fourier_jacobi()
        Y = other.fourier_jacobi()
        return FourierJacobiSeries([x+Y[i] for i, x in enumerate(X[:min(len(X), len(Y))])])
    __radd__ = __add__

    def __div__(self, other):
        return FourierJacobiSeries([x / other for x in self.fourier_jacobi()])
    __truediv__ = __div__

    def __eq__(self, other):
        X = self.fourier_jacobi()
        Y = other.fourier_jacobi()
        return all(x == Y[i] for i, x in enumerate(X[:min(len(X), len(Y))]))

    def __mul__(self, other):
        X = self.fourier_jacobi()
        if other in CC:
            return FourierJacobiSeries([x*other for x in X])
        Y = other.fourier_jacobi()
        return FourierJacobiSeries([sum(X[k] * Y[n-k] for k in range(n + 1)) for n in range(min(len(X), len(Y)))])
    __rmul__ = __mul__

    def __neg__(self):
        return FourierJacobiSeries([-x for x in self.fourier_jacobi()])

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

    def __sub__(self, other):
        X = self.fourier_jacobi()
        Y = other.fourier_jacobi()
        return FourierJacobiSeries([x-Y[i] for i, x in enumerate(X[:min(len(X), len(Y))])])

    def is_lift(self):
        try:
            X = self.fourier_jacobi()
            f = X[1]
            if not bool(f):
                return False
            return all(f.hecke_V(N) == x for N, x in enumerate(X))
        except NotImplementedError:
            return False


def formal_lift(f, prec):
    r"""
    Gritsenko lift.

    INPUT:
    - ``f`` -- a *vector-valued modular form* (which is meant to be the theta-decomposition of the Jacobi form to be lifted)
    - ``prec`` -- precision

    Instead of calling this directly you should use
    f.formal_lift()
    """
    if not f.is_holomorphic():
        return NotImplemented
    S = f.gram_matrix()
    N = S.nrows()
    k = f.weight() + Integer(N) / 2
    C = f.fourier_expansion()[0][2][0]
    if C:
        X = [smf_eisenstein_series(k, prec, normalization='linear') * C]
    else:
        X = [WeilRep([]).zero(k, prec)]
    X.append(f)
    for i in range(2, prec):
        X.append(f.hecke_V(i))
    return FourierJacobiSeries(X)


def _fj_relations(*X):
    r"""
    Compute linear relations among a list of formal Fourier-Jacobi series.
    """
    if len(X) == 1:
        X = X[0]
    k = X[0].weight()
    X = [x.fourier_jacobi() for x in X]
    prec = min(map(len, X))
    Z = [WeilRepModularFormsBasis(k, [x[i] for x in X], X[0][i].weilrep()) for i in range(prec)]
    V = Z[0].relations()
    i = 1
    while V and i < prec:
        V = V.intersection(Z[i].relations())
        i += 1
    return V
