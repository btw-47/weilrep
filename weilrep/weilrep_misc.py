r"""


Miscellaneous functions


"""

import cmath
import math
from bisect import bisect

from sage.arith.misc import dedekind_sum, divisors, fundamental_discriminant, is_square, kronecker, prime_divisors
from sage.arith.srange import srange
from sage.calculus.var import var
from sage.combinat.subset import Subsets
from sage.functions.generalized import sgn
from sage.functions.other import ceil, factorial, floor, frac, sqrt
from sage.functions.transcendental import zeta
from sage.matrix.constructor import matrix
from sage.misc.functional import denominator, isqrt
from sage.misc.misc_c import prod
from sage.modular.dirichlet import kronecker_character
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.modular.modform.vm_basis import delta_qexp
from sage.modules.free_module import FreeModule
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.quadratic_forms.special_values import quadratic_L_function__exact
from sage.rings.big_oh import O
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import QuadraticField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.symbolic.function import BuiltinFunction

sage_one_eighth = Integer(1) / Integer(8)
sage_three_half = Integer(3) / Integer(2)


def update_echelon_form_with(X, basis, basis_vectors, pivots, rank, sturm_bound):
    r"""
    Updates a basis in echelon form with a new object. Usually used for JacobiForm's

    INPUT:
    - ``X`` -- a WeilRepModularForm or JacobiForm
    - ``basis`` -- a list of WeilRepModularForm's or JacobiForm's
    - ``basis_vectors`` -- a list of coefficient vectors corresponding to basis
    - ``pivots`` -- a list of pivot indices in the basis_vectors
    - ``rank`` -- current rank (= len(basis))
    - ``sturm_bound`` -- a Sturm bound that applies to the coefficient vector of X

    OUTPUT: an updated tuple (basis, basis_vectors, pivots, rank)
    """
    v = X.coefficient_vector(starting_from = 0, ending_with = sturm_bound, correct = False)
    for j, vec in enumerate(basis_vectors):
        X -= v[pivots[j]]*basis[j]
        v -= v[pivots[j]]*vec
    for j, vj in enumerate(v):
        if vj:
            X = X / vj
            v /= vj
            for i, basis_i in enumerate(basis_vectors):
                wj = basis_i[j]
                basis_vectors[i] = basis_i - wj*v
                basis[i] -= wj*X
            new_index = bisect(pivots, j)
            pivots.insert(new_index, j)
            basis_vectors.insert(new_index, v)
            basis.insert(new_index, X)
            rank += 1
            return basis, basis_vectors, pivots, rank
    return basis, basis_vectors, pivots, rank

def gegenbauer_polynomial(N, s):
    r"""
    Two-variable Gegenbauer polynomials.
    """
    x, y = PolynomialRing(QQ, ['x', 'y']).gens()
    return factorial(N) * sum( prod(srange(s + ceil(N / 2), s + N - k)) / (factorial(k) * factorial(N - (k + k))) * x**(N - (k + k)) * (-y)**k for k in range(N // 2 + 1) )

def multilinear_gegenbauer_polynomial(n, s, vectors, S):
    r"""
    Multilinear Gegenbauer polynomials.
    """
    P = gegenbauer_polynomial(n, s)
    N = S.nrows()
    R = PolynomialRing(QQ, list(var('z_%d' % i) for i in range(N + 1) ))
    gens = R.gens()
    mu = vector(gens[:-1])
    n = gens[-1]
    v = vectors[0]
    if all(x == v for x in vectors[1:]):
        return P(mu * S * v, n * (v * S * v) / 2)
    a = Subsets(range(len(vectors)))
    h = 0
    for s in a:
        v = sum(vectors[i] for i in s)
        if v:
            Sv = S * v
        else:
            v = vector([0]*S.nrows())
            Sv = vector([0] * S.nrows())
        h += (-1) ** len(s) * P(mu * Sv, n * (v * Sv)/2)
    return h / factorial(len(vectors))

## linear relations

def relations(*x):
    r"""
    Compute linear relations.

    This computes the vector space of linear relations among the input 'x'. This can be vector-valued modular forms, Jacobi forms, or modular forms on orthogonal groups.

    INPUT:

    - ``x`` -- one or more (or a list of) JacobiForm's; or WeilRepModularForm's; or OrthogonalModularForm's.

    OUTPUT: a Free Module

    EXAMPLES::

        sage: from weilrep import *
        sage: relations(JacobiForms(1).basis(10, 5) + [jacobi_eisenstein_series(4, 1, 5) * smf_eisenstein_series(6), jacobi_eisenstein_series(6, 1, 5) * smf_eisenstein_series(4)])
        Vector space of degree 4 and dimension 2 over Rational Field
        Basis matrix:
        [     1      0 -11/18  -7/18]
        [     0      1 -1/144  1/144]

        sage: from weilrep import *
        sage: e4, e6, e10, e14 = [ParamodularForms(1).eisenstein_series(k, 6) for k in [4, 6, 10, 14]]
        sage: relations(e4*e4*e6, e4*e10, e14)
        Vector space of degree 3 and dimension 1 over Rational Field
        Basis matrix:
        [                 1 -47200892/12330549  34870343/12330549]

        sage: from weilrep import *
        sage: relations(WeilRep([[2, 1], [1, 2]]).modular_forms_basis(15, 10))
        Vector space of degree 3 and dimension 0 over Rational Field
        Basis matrix:
        []
    """
    from .amf import AlgebraicModularForm, _amf_relations
    from .fourier_jacobi import FourierJacobiSeries, _fj_relations
    from .jacobi_forms_class import JacobiForm, _jf_relations
    from .jacobi_lvl import JacobiFormWithLevel, _jf_relations_lvl
    from .lifts import OrthogonalModularForm, _omf_relations
    from .unitary import UnitaryModularForm, _umf_relations
    from .weilrep_modular_forms_class import WeilRepModularForm, WeilRepModularFormsBasis
    x_ref = x[0]
    if isinstance(x_ref, list):
        x = x_ref
        try:
            x_ref = x[0]
        except IndexError:
            return FreeModule(QQ, 0)
    if isinstance(x_ref, WeilRepModularForm):
        k = x_ref.weight()
        w = x_ref.weilrep()
        if any(y.weight() != k or y.weilrep() != w for y in x[1:]):
            raise ValueError('Incompatible modular forms.')
        X = WeilRepModularFormsBasis(x_ref.weight(), x, x_ref.weilrep())
        return X.relations()
    elif isinstance(x_ref, WeilRepModularFormsBasis):
        return x_ref.relations()
    elif isinstance(x_ref, JacobiForm):
        return _jf_relations(x)
    elif isinstance(x_ref, JacobiFormWithLevel):
        return _jf_relations_lvl(x)
    elif isinstance(x_ref, OrthogonalModularForm):
        return _omf_relations(x)
    elif isinstance(x_ref, UnitaryModularForm):
        return _umf_relations(x)
    elif isinstance(x_ref, FourierJacobiSeries):
        return _fj_relations(x)
    elif isinstance(x_ref, AlgebraicModularForm):
        return _amf_relations(x)
    return NotImplemented

## theta blocks

def weight_two_basis_from_theta_blocks(N, prec, dim, jacobiforms = None, verbose = False):
    r"""
    Look for theta blocks of weight two and given index among the infinite families of weight two theta blocks associated to the root systems A_4, B_2+G_2, A_1+B_3, A_1+C_3

    This is not meant to be called directly. Use JacobiForms(N).basis(2) with the optional parameter try_theta_blocks = True.

    INPUT:
    - ``N`` -- the index
    - ``prec`` -- precision
    - ``dim`` -- the dimension of the space of Jacobi forms.
    - ``jacobiforms`` -- a JacobiForms instance for this index. (If none is provided then we construct one now.)
    - ``verbose`` -- boolean (default False); if True then we comment on the computations.

    OUTPUT: a list of JacobiForm's
    """
    from .jacobi_forms_class import theta_block
    if not jacobiforms:
        jacobiforms = JacobiForms(N)
    rank = 0
    #four families of weight two theta blocks from root lattices
    thetablockQ_1 = QuadraticForm(matrix([[4,3,2,1],[3,6,4,2],[2,4,6,3],[1,2,3,4]]))
    thetablockQ_2 = QuadraticForm(matrix([[24,12,0,0],[12,8,0,0],[0,0,12,6],[0,0,6,6]]))
    thetablockQ_3 = QuadraticForm(matrix([[4,0,0,0],[0,20,10,20],[0,10,10,20],[0,20,20,60]]))
    thetablockQ_4 = QuadraticForm(matrix([[4,0,0,0],[0,8,8,4],[0,8,16,8],[0,4,8,6]]))
    thetablock_tuple = thetablockQ_1, thetablockQ_2, thetablockQ_3, thetablockQ_4
    thetablock_1 = lambda a, b, c, d, prec: theta_block([a, a+b, a+b+c, a+b+c+d, b, b+c, b+c+d, c, c+d, d], -6, prec, jacobiforms = jacobiforms)
    thetablock_2 = lambda a, b, c, d, prec: theta_block([a, 3*a+b, 3*a+b+b, a+a+b, a+b, b, c+c, c+c+d, 2*(c+d), d], -6, prec, jacobiforms = jacobiforms)
    thetablock_3 = lambda a, b, c, d, prec: theta_block([a+a, b+b, b+b+c, 2*(b+c+d+d), b+b+c+d+d, b+b+c+4*d, c, c+d+d, c+4*d, d+d], -6, prec, jacobiforms = jacobiforms)
    thetablock_4 = lambda a, b, c, d, prec: theta_block([a+a, b, b+b+c+c+d, b+c, b+c+c+d, b+c+d, c, c+c+d, c+d, d], -6, prec, jacobiforms = jacobiforms)
    thetablocks = thetablock_1, thetablock_2, thetablock_3, thetablock_4
    basis = []
    basis_vectors = []
    pivots = []
    div_N = divisors(N)
    div_N.reverse()
    for i, Q in enumerate(thetablock_tuple):
        v_list = Q.short_vector_list_up_to_length(N + 1, up_to_sign_flag = True)
        if verbose:
            print('I am looking through the theta block family of the root system %s.' %['A_4','B_2+G_2','A_1+B_3','A_1+C_3'][i])
        for _d in div_N:
            prec_d = prec * (N // _d)
            for v in v_list[_d]:
                a, b, c, d = v
                try:
                    j = thetablocks[i](a, b, c, d, prec_d).hecke_V(N // _d)
                    old_rank = 0 + rank
                    basis, basis_vectors, pivots, rank = update_echelon_form_with(j, basis, basis_vectors, pivots, rank, sage_one_eighth)
                    if verbose and old_rank < rank:
                        if i == 0:
                            L = [abs(x) for x in (a, a+b, a+b+c, a+b+c+d, b, b+c, b+c+d, c, c+d, d)]
                        elif i == 1:
                            L = [abs(x) for x in (a, 3*a+b, 3*a+b+b, a+a+b, a+b, b, c+c, c+c+d, 2*(c+d), d)]
                        elif i == 2:
                            L = [abs(x) for x in (a+a, b+b, b+b+c, 2*(b+c+d+d), b+b+c+d+d, b+b+c+4*d, c, c+d+d, c+4*d, d+d)]
                        else:
                            L = [abs(x) for x in (a+a, b, b+b+c+c+d, b+c, b+c+c+d, b+c+d, c, c+c+d, c+d, d)]
                        L.sort()
                        print('I found the theta block Th_' + str(L) + ' / eta^6.')
                    if rank == dim:
                        return basis
                except ValueError:
                    pass
    if verbose:
        print('I did not find enough theta blocks. Time to try something else.')
    return jacobiforms.basis(2, prec, try_theta_blocks = False, verbose = verbose)

def weight_three_basis_from_theta_blocks(N, prec, dim, jacobiforms = None, verbose = False):
    r"""
    Look for theta blocks of weight three and given index among the infinite families of weight three theta blocks associated to the root systems B_3, C_3, A_2+A_3, 3A_2, 2A_1 + A_2 + B_2, 3A_1 + A_3, A_6, A_1+D_5.

    This is not meant to be called directly. Use JacobiForms(N).basis(3) with the optional parameter try_theta_blocks = True.

    INPUT:
    - ``N`` -- the index
    - ``prec`` -- precision
    - ``dim`` -- the dimension of the space of Jacobi forms.
    - ``jacobiforms`` -- a JacobiForms instance for this index. (If none is provided then we construct one now.)
    - ``verbose`` -- boolean (default False); if True then we comment on the computations.

    OUTPUT: a list of JacobiForm's
    """
    from .jacobi_forms_class import theta_block
    if not jacobiforms:
        jacobiforms = JacobiForms(N)
    rank = 0
    thetablockQ_1 = QuadraticForm(matrix([[20,10,20],[10,10,20],[20,20,60]])) #B3
    thetablockQ_2 = QuadraticForm(matrix([[8,8,4],[8,16,8],[4,8,6]])) #C3
    thetablockQ_3 = QuadraticForm(matrix([[2,1,0,0,0],[1,2,0,0,0],[0,0,12,4,4],[0,0,4,4,4],[0,0,4,4,12]])) #A2 + A3
    thetablockQ_4 = QuadraticForm(matrix([[2,1,0,0,0,0],[1,2,0,0,0,0],[0,0,2,1,0,0],[0,0,1,2,0,0],[0,0,0,0,2,1],[0,0,0,0,1,2]])) #3 A2
    thetablockQ_5 = QuadraticForm(matrix([[4,0,0,0,0,0],[0,4,0,0,0,0],[0,0,2,1,0,0],[0,0,1,2,0,0],[0,0,0,0,6,6],[0,0,0,0,6,12]])) #2A1 + A2 + B2
    thetablockQ_6 = QuadraticForm(matrix([[4,0,0,0,0,0],[0,4,0,0,0,0],[0,0,4,0,0,0],[0,0,0,12,4,4],[0,0,0,4,4,4],[0,0,0,4,4,12]])) #3A1 + A3
    thetablockQ_7 = QuadraticForm(matrix([[6,5,4,3,2,1],[5,10,8,6,4,2],[4,8,12,9,6,3],[3,6,9,12,8,4],[2,4,6,8,10,5],[1,2,3,4,5,6]])) #A6
    thetablockQ_8 = QuadraticForm(matrix([[4,0,0,0,0,0],[0,10,6,12,8,4],[0,6,10,12,8,4],[0,12,12,24,16,8],[0,8,8,16,16,8],[0,4,4,8,8,8]])) #A1 + D5
    thetablock_tuple = thetablockQ_1, thetablockQ_2, thetablockQ_3, thetablockQ_4, thetablockQ_5, thetablockQ_6, thetablockQ_7, thetablockQ_8
    args1 = lambda b, c, d : [b+b, b+b+c, 2*(b+c+d+d), b+b+c+d+d, b+b+c+4*d, c, c+d+d, c+4*d, d+d]
    args2 = lambda b, c, d : [b, b+b+c+c+d, b+c, b+c+c+d, b+c+d, c, c+c+d, c+d, d]
    args3 = lambda a, b, d, e, f : [a, a+b, b, d+d, e, d+d+e, f+f, e+f+f, d+d+e+f+f]
    args4 = lambda a, b, c, d, e, f : [a, a+b, b, c, c+d, d, e, e+f, f]
    args5 = lambda a, b, c, d, e, f : [a+a, b+b, c, c+d, d, e, f+f, e+f+f, e+e+f+f]
    args6 = lambda a, b, c, d, e, f : [a+a, b+b, c+c, d+d, e, d+d+e, f+f, e+f+f, d+d+e+f+f]
    args7 = lambda a, b, c, d, e, f : [a, a+b, a+b+c, a+b+c+d, a+b+c+d+e, a+b+c+d+e+f, b, b+c, b+c+d, b+c+d+e, b+c+d+e+f, c, c+d, c+d+e, c+d+e+f, d, d+e, d+e+f, e, e+f, f]
    args8 = lambda a, b, c, d, e, f : [a+a, b, c, d, b + d, c + d, b + c + d, e, d + e, b + d + e, c + d + e, b + c + d + e, b + c + 2*d + e, f, e + f, d + e + f, b + d + e + f, c + d + e + f, b + c + d + e + f, b + c + 2*d + e + f, b + c + 2*d + 2*e + f]
    args_tuple = args1, args2, args3, args4, args5, args6, args7, args8
    thetablock_1 = lambda b, c, d, prec: theta_block(args1(b, c, d), -3, prec, jacobiforms = jacobiforms)
    thetablock_2 = lambda b, c, d, prec: theta_block(args2(b, c, d), -3, prec, jacobiforms = jacobiforms)
    thetablock_3 = lambda a, b, d, e, f, prec: theta_block(args3(a, b, d, e, f), -3, prec, jacobiforms = jacobiforms)
    thetablock_4 = lambda a, b, c, d, e, f, prec: theta_block(args4(a, b, c, d, e, f), -3, prec, jacobiforms = jacobiforms)
    thetablock_5 = lambda a, b, c, d, e, f, prec: theta_block(args5(a, b, c, d, e, f), -3, prec, jacobiforms = jacobiforms)
    thetablock_6 = lambda a, b, c, d, e, f, prec: theta_block(args6(a, b, c, d, e, f), -3, prec, jacobiforms = jacobiforms)
    thetablock_7 = lambda a, b, c, d, e, f, prec: theta_block(args7(a, b, c, d, e, f), -15, prec, jacobiforms = jacobiforms)
    thetablock_8 = lambda a, b, c, d, e, f, prec: theta_block(args8(a, b, c, d, e, f), -15, prec, jacobiforms = jacobiforms)
    thetablocks = thetablock_1, thetablock_2, thetablock_3, thetablock_4, thetablock_5, thetablock_6, thetablock_7, thetablock_8
    basis = []
    basis_vectors = []
    pivots = []
    div_N = divisors(N)
    div_N.reverse()
    for i, Q in enumerate(thetablock_tuple):
        v_list = Q.short_vector_list_up_to_length(N + 1, up_to_sign_flag = True)
        if verbose:
            print('I am looking through the theta block family of the root system %s.' %['B_3','C_3','A_2+A_3','3A_2','2A_1+A_2+B_2','3A_1 + A_3','A_6','A_1+D_5'][i])
        for _d in div_N:
            prec_d = prec * (N // _d)
            for v in v_list[_d]:
                if all(a for a in args_tuple[i](*v)):
                    try:
                        j = thetablocks[i](*(list(v) + [prec])).hecke_V(N // _d)
                        old_rank = 0 + rank
                        basis, basis_vectors, pivots, rank = update_echelon_form_with(j, basis, basis_vectors, pivots, rank, sage_one_eighth)
                        if verbose and old_rank < rank:
                            L = [abs(x) for x in args_tuple[i](*v)]
                            L.sort()
                            if _d == N:
                                print('I found the theta block Th_' + str(L) + [' / eta^3.', ' / eta^15.'][i >= 6])
                            else:
                                print('I applied the Hecke operator V_%d to the theta block Th_'%(N // _d) + str(L) + [' / eta^3.', ' / eta^15.'][i >= 6])
                        if rank == dim:
                            return basis
                    except ValueError:
                        pass
    if verbose:
        print('I did not find enough theta blocks. Time to try something else.')
    return jacobiforms.basis(2, prec, try_theta_blocks = False, verbose = verbose)



class QuadraticLFunction(BuiltinFunction):
    r"""
    Symbolic quadratic L-functions.

    L(s, D) represents the Dirichlet series \sum_{n = 1}^{\infty} \chi_D(n) n^{-s}, where D is a discriminant (i.e. 0 or 1 mod 4) and \chi_D(n) is the kronecker symbol (D / n)
    """
    def __init__(self):
        super().__init__('L', nargs=2, latex_name = 'L')

    def _eval_(self, x, D): #symbolic value
        D = Integer(D)
        if D % 4 > 1:
            raise ValueError('Not a discriminant')
        f = D.squarefree_part()
        if f % 4 > 1 and not D % 4:
            f *= 4
        m = D // f
        if f != D:
            L = QuadraticLFunction()
            return prod(1 - p**(-x) * kronecker(f, p) for p in prime_divisors(m)) * L(x, f)
        if D == 1:
            return prod(1 - p**(-x) * kronecker(f, p) for p in prime_divisors(m)) * zeta(x)
        try:
            return quadratic_L_function__exact(x, D)
        except TypeError:
            pass

    def _evalf_(self, x, D, **kwargs): #numerical value
        D = Integer(D)
        if D % 4 > 1:
            raise ValueError('Not a discriminant')
        f = D.squarefree_part()
        if f % 4 > 1 and not D % 4:
            f *= 4
        m = D // f
        if f != 1:
            s = kronecker_character(f).lfunction(algorithm='lcalc').value(x).real()
        else:
            s = zeta(x).n()
        if m != 1:
            return prod(1 - p**(-x) * kronecker(f, p) for p in prime_divisors(m)) * s
        return s

    def residue(self, D): #residue at s=1 if there is a pole
        D = Integer(D)
        f = D.squarefree_part()
        if f != 1:
            return 0
        return prod((p - 1) / p for p in prime_divisors(D))
