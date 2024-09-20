r"""
Auxiliary code for vector-valued Eisenstein series

AUTHORS:

- Brandon Williams

"""

# ****************************************************************************
#       Copyright (C) 2020-2023 Brandon Williams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from sage.all import cached_function
from sage.arith.misc import factor, fundamental_discriminant, GCD, is_prime, kronecker_symbol, valuation
from sage.functions.other import frac
from sage.functions.transcendental import zeta
from sage.matrix.constructor import matrix
from sage.matrix.special import block_diagonal_matrix, identity_matrix
from sage.misc.functional import denominator, isqrt, log
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector
from sage.quadratic_forms.quadratic_form import QuadraticForm, DiagonalQuadraticForm
from sage.quadratic_forms.special_values import quadratic_L_function__exact
from sage.rings.big_oh import O
from sage.rings.fast_arith import prime_range
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import IntegerRing, ZZ
from sage.rings.monomials import monomials
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ



## odd primes ##

@cached_function
def iard(a, r, d, p, t, m):
    r"""
    Compute the helper function I_a(r,d).

    This computes special values of the helper function I_a(r,d) from [CKW] following Table 2 of loc. cit.

    INPUT:
    - ``a`` -- an integer
    - ``r`` -- an integer
    - ``d`` -- an integer
    - ``p`` -- an odd prime
    - ``t`` -- a rational number
    - ``m`` -- an integer

    OUTPUT: a rational number

    EXAMPLES::

        sage: iard(2, 3, 5, 3, 1/3, 1)
        136/243
    """
    if r % 2 == 0:
        iard = 1 - m
        if a % p == 0:
            iard *= (1 + m * t) * (1 - 1/p)
        else:
            iard *= (1 - 1/p + m * (1 - t/p))
    else:
        if a % p == 0:
            iard = (1 - t * p ** QQ(-r)) * (1 - 1/p)
        else:
            iard = 1 - 1/p - (p ** QQ(-r)) * (1 - t/p) + (p ** QQ((-r) // 2)) * kronecker_symbol(-a * d * (-1)**(r // 2), p) * (t - 1)
    return iard


def igusa_zetas(Q,L,c,p,t):
    r"""
    Compute values of the Igusa (local) zeta function for quadratic polynomials over Z_p.

    This computes a special value of the Igusa (local) zeta functions attached to the polynomials P_j(x) = Q(x) + L(x) + c_j for a list of integers c = [c_1,c_2,...].

    INPUT:
    - ``Q`` -- a quadratic form
    - ``L`` -- a vector
    - ``c`` -- a list of integers
    - ``p`` -- an odd prime
    - ``t`` -- a rational number

    OUTPUT:
    vector of rational numbers

    ALGORITHM: we use Theorem 2.1 of [CKW].

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1],[1,2]]))
        sage: L = vector([1,2])
        sage: c = [1,2,3,4,5]
        sage: p = 3
        sage: igusa_zetas(Q,L,c,p,1/9)
        (26/27, 494/729, 10/27, 26/27, 13130/19683)
    """
    quads, lins_gcd, const = isospectral_normal_form(Q, L, p)
    c = vector((const + cj/2) for cj in c)
    try:
        c_max_val = max([cj.valuation(p) for cj in c if cj])
    except:
        c_max_val = 0
    u = denominator(c/2)
    c *= u
    lins_gcd *= u
    Z = vector([0]*len(c))
    d_old = [1]
    r_old = [0]
    for i, a in enumerate(quads):
        a *= u
        p_i = a.valuation(p)
        try:
            d_old[p_i] *= a / (p ** p_i)
            r_old[p_i] += 1
        except:
            d_old.extend([1] * (p_i + 1 - len(r_old)))
            r_old.extend([0] * (p_i + 1 - len(r_old)))
            d_old[p_i] = a // (p ** p_i)
            r_old[p_i] = 1
    d = []
    r = []
    p_power = []
    if lins_gcd:
        max_range = max([len(d_old), lins_gcd.valuation(p), c_max_val + 1])
    else:
        max_range = max(len(d_old), c_max_val + 1)
    for k in range(max_range+1):
        if k >= len(d_old):
            d_old.append(1)
            r_old.append(0)
        for ell in range(k, max_range+1):
            if ell > len(d) - 1:
                d.append(1)
                r.append(0)
                p_power.append(1)
            if (ell % 2 == k % 2):
                d[ell] = d[ell] * d_old[k]
                r[ell] = r[ell] + r_old[k]
            if ell != k:
                p_power[ell] /= (p ** r[k])
    Z_vec = []
    m = [1 if x % 2 else (p ** QQ(-x//2)) * kronecker_symbol((-1)**(x//2) * d[i],p) for i, x in enumerate(r)]
    if lins_gcd:
        lamda = lins_gcd.valuation(p)
        f0 = sum((t ** i) * iard(0, r[i], d[i], p, t, m[i]) * p_power[i] for i in range(lamda)) + ((t ** lamda) * p_power[lamda]) * (1 - 1 / p)
        for cj in c:
            kappa = cj.valuation(p)
            if lamda <= kappa:
                Z_vec.append(f0)
            else:
                f1 = 0
                t_j = 1/t
                for j in range(kappa+1):
                    t_j *= t
                    f1 += t_j * iard(cj, r[j], d[j], p, t, m[j]) * p_power[j]
                    cj /= p
                Z_vec.append(t_j * (1 - t/p) * p_power[kappa + 1] + f1)
        return vector(Z_vec)
    else:
        w = len(d_old) - 1
        r_old_sum = QQ(sum(r_old))
        f0 = sum((t ** i) * iard(0, r[i], d[i], p, t, m[i]) * p_power[i] for i in range(w-1))
        try:
            f0 = f0 + 1/(1 - (t * t) / (p ** r_old_sum)) * (t ** (w-1)) * ( iard(0, r[w-1], d[w-1], p, t, m[w-1]) * p_power[w-1] + t * iard(0, r[w], d[w], p, t, m[w]) * p_power[w] )
        except ZeroDivisionError:
            t0, = PolynomialRing(QQ, 't0').gens()
            f1 = ~(1 - (t0 ** 2) / (p ** r_old_sum)) * (t0 ** (w-1)) * ( iard(0, r[w-1], d[w-1], p, t0, m[w-1]) * p_power[w-1] + t0 * iard(0, r[w], d[w], p, t0, m[w]) * p_power[w] )
            f0 = f0 + f1(t)
        for cj in c:
            if cj:
                kappa = cj.valuation(p)
                f1 = 0
                t_j = 1/t
                for j in range(kappa + 1):
                    t_j *= t
                    f1 += t_j * iard(cj, r[j], d[j], p, t, m[j]) * p_power[j]
                    cj /= p
                Z_vec.append(f1 + t_j * (1 - t/p) * p_power[kappa+1])
            else:
                Z_vec.append(f0)
        return vector(Z_vec)

def isospectral_normal_form(Q, L, p):
    r"""
    Computes an isospectral normal form of the quadratic polynomial Q + L modulo the odd prime p, where Q is a quadratic form and L is a linear form.

    An ``isospectral normal form`` is a quadratic polynomial of the form
        `P(x_1,...,x_n) = a_1 x_1^2 + ... + a_(n-1) x_(n-1)^2 + b_n x_n + c`
    with the following property: for every integer N, the number of zeros of the equations
        `Q(x_1,...,x_n) + L(x_1,...,x_n) = N`
    and
        `P(x_1,...,x_n) = N`

    modulo any power of p are the same.

    ALGORITHM: We use Section 4.9 of [CKW]. Essentially this is repeated use of either Hensel's lemma or completing the square.

    INPUT:

    - ``Q`` -- a quadratic form
    - ``L`` a vector
    - ``p`` an odd prime

    OUTPUT: a tuple consisting of
    - ``quads`` -- a list of the coefficients [a_1,...,a_(n-1)] of the quadratic part
    - ``lins_gcd`` -- the linear coefficient b_n
    - ``const`` -- the constant term c

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1],[1,2]]))
        sage: L = vector([1,2])
        sage: p = 3
        sage: isospectral_normal_form(Q,L,p)
        ([1, 3], 0, -1/4)

    """
    D, P = local_normal_form_with_change_vars(Q.matrix(),p)
    L = P * L
    linear_gcd = 0
    const = 0
    quads = []
    for j in range(Q.dim()):
        a = D[j, j]
        b = L[j] / 2
        if b.valuation(p) < a.valuation(p):
            linear_gcd = GCD(linear_gcd, b)
        else:
            quads.append(a)
            const = const - b * b / (4 * a)
    return quads, linear_gcd, const

@cached_function
def local_normal_form_with_change_vars(S,p):
    r"""
    Diagonalize the quadratic form with Gram matrix S over Q_p, p != 2.

    This is copied from the function local_normal_form() (specialized to odd primes). The only difference is that it keeps track of a change-of-basis matrix to the local normal form.

    INPUT:
    - ``S`` -- a symmetric integral matrix with even diagonal and nonzero determinant
    - ``p`` -- an odd prime

    OUTPUT: a tuple D, P where
    - ``D`` -- a diagonal quadratic form over ZZ
    - ``P`` -- an integral matrix with determinant coprime to p such that P * S * P.transpose() = D.matrix()

    NOTE: in general P is not invertible over ZZ; only over the p-adic numbers!

    EXAMPLES::

        sage: local_normal_form_with_change_vars(matrix([[2,1],[1,2]]),3)
        (
        Quadratic form in 2 variables over Integer Ring with coefficients: 
        [ 1 0 ]                                                            
        [ * 3 ]                                                            ,
        <BLANKLINE>
        [ 1  0]
        [-1  2]
        )
    """
    #
    Q = QuadraticForm(S)
    I = list(range(Q.dim()))
    M = identity_matrix(QQ,Q.dim())
    Q_Jordan = DiagonalQuadraticForm(ZZ,[])
    while Q.dim() > 0:
        n = Q.dim()
        (min_i, min_j) = Q.find_entry_with_minimal_scale_at_prime(p)
        if min_i == min_j:
            Q.swap_variables(0, min_i, in_place = True)
            M.swap_rows(I[0],I[min_i])
        else:
            min_val = valuation(Q[min_i, min_j], p)
            Q.swap_variables(0, min_i, in_place = True)
            Q.swap_variables(1, min_j, in_place = True)
            M.swap_rows(I[0],I[min_i])
            M.swap_rows(I[1],I[min_j])
            Q.add_symmetric(1, 0, 1, in_place = True)
            M.add_multiple_of_row(I[0],I[1],1)
        a = 2 * Q[0,0]
        for j in range(1, n):
            b = Q[0, j]
            g = GCD(a, b)
            Q.multiply_variable(a//g, j, in_place = True)
            Q.add_symmetric(-b//g, j, 0, in_place = True)
            M.rescale_row(I[j],a/g)
            M.add_multiple_of_row(I[j],I[0],-b//g)
        Q_Jordan = Q_Jordan + Q.extract_variables(range(1))
        I.remove(I[0])
        Q = Q.extract_variables(range(1, n))
    return Q_Jordan, M


@cached_function
def twonf_with_change_vars(Q):
    r"""
    Compute the Jordan normal form and a change-of-basis matrix for the quadratic form Q at the prime p=2.

    This is copied from the built-in function local_normal_form() (specialized to p = 2). The only difference is that it keeps track of a change-of-basis matrix associated to the local normal form.

    WARNING: the change-of-basis matrix P computed here does not put the quadratic form in Jordan normal form! (This is impossible over the integers.) For our purposes it is enough that the result is 2-adically ``close enough``. In the example below (Q = A_3 lattice) we obtain the Gram matrix P * (Q_Jordan).matrix() * P.transpose() = matrix([[2,1,0],[1,2,-3],[0,-3,114]]), which equals Q.matrix() = matrix([[2,1,0],[1,2,1],[0,1,2]]) only mod 4.

    INPUT:
    - ``Q`` -- matrix

    OUTPUT:
    - ``Q_Jordan`` -- the Jordan normal form of Q (also a quadratic form)
    - ``P`` -- an integral matrix for which P * Q_Jordan.matrix() * P.transpose() = Q.matrix() modulo the largest power of 2 dividing the discriminant of Q.

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1,0],[1,2,1],[0,1,2]]))
        sage: twonf_with_change_vars(Q)
        (
        Quadratic form in 3 variables over Integer Ring with coefficients: 
        [ 1 1 0 ]                                                          
        [ * 1 0 ]                                                          
        [ * * 6 ]                                                          ,
        <BLANKLINE>
        [ 1  0  0]
        [ 0  1  0]
        [ 1 -2  3]
        )
    """

    #basically copied from the built-in function local_normal_form(), specialized to p=2, but it also computes a change-of-variables to the local normal form. it skips the "Cassels' proof" step. this is accounted for later.
    Q = QuadraticForm(Q)
    I = list(range(Q.dim()))
    P = identity_matrix(QQ,Q.dim())
    Q_Jordan = DiagonalQuadraticForm(ZZ,[])
    while Q.dim() > 0:
        n = Q.dim()
        (min_i,min_j) = Q.find_entry_with_minimal_scale_at_prime(2)
        if min_i == min_j:
            min_val = valuation(2 * Q[min_i,min_j],2)
        else:
            min_val = valuation(Q[min_i, min_j],2)
        if (min_i == min_j):
            block_size = 1
            Q.swap_variables(0,min_i,in_place = True)
            P.swap_rows(I[0],I[min_i])
        else:
            Q.swap_variables(0, min_i, in_place = True)
            Q.swap_variables(1, min_j, in_place = True)
            P.swap_rows(I[0],I[min_i])
            P.swap_rows(I[1],I[min_j])
            block_size = 2
        min_scale = 2 ** (min_val)
        if (block_size == 1):
            a = 2 * Q[0,0]
            for j in range(block_size, n):
                b = Q[0, j]
                g = GCD(a, b)
                Q.multiply_variable(ZZ(a/g), j, in_place = True)
                Q.add_symmetric(ZZ(-b/g), j, 0, in_place = True)
                P.rescale_row(I[j],a/g)
                P.add_multiple_of_row(I[j],I[0],-b/g)
            NewQ = QuadraticForm(matrix([a]))
        else:
            a1 = 2*Q[0,0]
            a2 = Q[0,1]
            b2 = 2*Q[1,1]
            big_det = (a1*b2 - a2*a2)
            two_scale= ZZ(min_scale * min_scale)
            for j in range(block_size,n):
                a = Q[0,j]
                b = Q[1,j]
                Q.multiply_variable(big_det,j,in_place = True)
                Q.add_symmetric(ZZ(-(a*b2 - b*a2)),j,0,in_place = True)
                Q.add_symmetric(ZZ(-(-a*a2 + b*a1)),j,1,in_place = True)
                Q.divide_variable(two_scale,j,in_place = True)
                P.rescale_row(I[j],big_det)
                P.add_multiple_of_row(I[j],I[0],-a*b2+b*a2)
                P.add_multiple_of_row(I[j],I[1],a*a2-b*a1)
                P.rescale_row(I[j],~two_scale)
        Q_Jordan = Q_Jordan + Q.extract_variables(range(block_size))
        for j in range(block_size):
            I.remove(I[0])
        Q = Q.extract_variables(range(block_size, n))
    return Q_Jordan, P

def twoadic_isospectral_normal_form(Q,L):
    r"""
    Computes an isospectral normal form of the quadratic polynomial Q + L modulo the  pime p=2, where Q is a quadratic form and L is a linear form.

    An ``isospectral normal form`` is a quadratic polynomial of the form
        `P(x_1,...,x_n) = a_1 Q_1(x) + ... + a_(n-1) Q_(n-1)(x) + b_n x_n + c`
    where Q_i(x) is a unimodular quadratic form in at most two variables and with the following property: for every integer N, the number of zeros of the equations
        `Q(x_1,...,x_n) + L(x_1,...,x_n) = N`
    and
        `P(x_1,...,x_n) = N`

    modulo any power of 2 are the same.

    ALGORITHM: We use Section 4.9 of [CKW].

    INPUT:

    - ``Q`` -- a quadratic form
    - ``L`` a vector

    OUTPUT: a tuple consisting of
    - ``New_Q`` -- a quadratic form: the quadratic part of the isospectral normal form at p=2
    - ``lins_gcd`` -- the linear coefficient b_n
    - ``const`` -- the constant term c

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[4,2],[2,4]]))
        sage: L = vector([2,0])
        sage: twoadic_isospectral_normal_form(Q,L)
        (Quadratic form in 0 variables over Integer Ring with coefficients: 
        , 2, 0)

    """
    D, P = twonf_with_change_vars(Q.matrix())
    L = P * L
    linear_term_gcd = 0
    NewQ = DiagonalQuadraticForm(ZZ, [])
    const = 0
    j = 0
    while j < Q.dim():
        if j < Q.dim() - 1 and D[j, j + 1] != 0:
            ell = L[j]
            m = L[j + 1]
            a = 2*D[j, j]
            b = 2*D[j, j + 1]
            c = 2*D[j + 1, j + 1]
            if min(ell.valuation(2),m.valuation(2)) >= b.valuation(2):
                NewQ += D.extract_variables([j,j+1])
                const = const + (c * ell * ell - b * ell * m + a * m * m) / (b * b - 4 * a * c) #complete the square
            else:
                linear_term_gcd = GCD(linear_term_gcd, GCD(ell, m))
            j += 2
        else:
            a = 2 * D[j,j]
            b = L[j]
            a_val = a.valuation(2)
            b_val = b.valuation(2)
            if b_val < a_val:
                linear_term_gcd = GCD(linear_term_gcd, b)
            elif b_val == a_val:
                linear_term_gcd = GCD(linear_term_gcd, b + b)
            else:
                NewQ += D.extract_variables([j])
                const = const - b * b / (4 * a)
            j += 1
    return NewQ, linear_term_gcd, const

@cached_function
def twoadic_classify(Q):
    r"""
    Classify unimodular quadratic forms over Z_2 up to isometric isomorphism.

    Every unimodular quadratic form over Z_2 is equivalent to a direct sum of at most two rank-one forms Sq_a(x) = a*x^2 (where a = 1,3,5,7), at most one elliptic plane E(x,y) = 2*(x^2 + xy + y^2), and some number of hyperbolic planes H(x,y) = 2*x*y. Given an integer quadratic form Q which is unimodular at 2, we compute this decomposition here.

    ALGORITHM: repeated use of the addition rules of Table 4 of [CKW].

    NOTE: we use a definition of ``unimodular`` which allows certain lattices of even level e.g. the square forms Sq_a themselves

    INPUT:
    - ``Q`` -- an integer quadratic form which is unimodular at 2

    OUTPUT:
    - ``squares`` -- a list of integer from 1,3,5,7 (representing the square forms that make up Q)
    - ``ell`` -- 0 if we do not take E(x,y) as a direct summand in Q and 1 otherwise
    - ``hyp`` -- natural number r, the number of copies of H(x,y) in Q.

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1],[1,2]]))
        sage: twoadic_classify(Q)
        ([], 1, 0)

        sage: Q = QuadraticForm(matrix([[2,1],[1,4]]))
        sage: twoadic_classify(Q)
        ([], 0, 1)

        sage: Q = QuadraticForm(matrix([[6]]))
        sage: twoadic_classify(Q)
        ([3], 0, 0)

    """
    J = twonf_with_change_vars(Q.matrix())
    J = J[0].matrix()
    squares = []
    ell = 0
    hyp = 0
    k = 0
    while k < J.nrows():
        try:
            if J[k,k+1]:
                if J[k, k] * J[k+1, k+1] % 8:
                    ell += 1
                else:
                    hyp += 1
                k += 2
            else:
                squares.extend([(J[k,k] % 16) / 2, (J[k+1,k+1] % 16) / 2])
                k += 2
        except:
            squares.append((J[k,k] % 16) / 2)
            k += 1
    while len(squares) > 2:
        x = squares[:3]
        squares = squares[3:]
        sum_x = sum(x) % 8
        prod_x = prod(x) % 8
        squares.append(sum_x)
        if (sum_x + prod_x) % 8:
            ell += 1
        else:
            hyp += 1
    hyp += ell
    ell %= 2
    return squares, ell, hyp-ell

@cached_function
def twoadic_jordan_blocks(Q):
    r"""
    Decompose a quadratic form into Jordan blocks.

    This decomposes the quadratic form Q in the form \bigoplus 2^i Q_i, where Q_i are integral quadratic forms of rank either 1 or 2 that are unimodular at the prime 2.

    INPUT:
    - ``Q`` -- a quadratic form which is assumed to be in Jordan normal form

    OUTPUT: a list of tuples (val, jordan) where
    - ``val`` -- a valuation i implying that the quadratic form ``jordan`` is scaled by 2^i as a summand of Q;
    - ``jordan`` -- an integral quadratic form which is unimodular at p=2 and of rank either 1 or 2

    WARNING: We do not check whether Q is actually in Jordan normal form.

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1,0],[1,2,0],[0,0,12]]))
        sage: twoadic_jordan_blocks(Q)
        [(0, Quadratic form in 2 variables over Integer Ring with coefficients: 
        [ 1 1 ]
        [ * 1 ]), (2, Quadratic form in 1 variables over Integer Ring with coefficients: 
        [ 3 ])]

    """
    Q1 = QuadraticForm(Q.matrix())
    jordans = []
    valuations = []
    i = 0
    n = Q1.dim()
    while i < n:
        if (i == n-1) or (not Q1[i,i+1]):
            scale = valuation(Q1[i,i], 2) + 1
            Q_i = QuadraticForm(ZZ,Q1.extract_variables([i]).matrix() / 2 ** (scale - 1))
            if valuations and scale == valuations[-1]:
                jordans[-1] += Q_i
            else:
                valuations.append(scale)
                jordans.append(Q_i)
            i += 1
        else:
            scale = valuation(Q1[i,i+1], 2)
            Q_i = QuadraticForm(ZZ,Q1.extract_variables([i,i+1]).matrix() / 2 ** (scale))
            if valuations and scale == valuations[-1]:
                jordans[-1] += Q_i
            else:
                valuations.append(scale)
                jordans.append(Q_i)
            i += 2
    return list(zip(valuations, jordans))

def ig_v(a,u,j,t):
    r"""
    Convert p-adic generating functions to Igusa (local) zeta functions.

    This evaluates the map Ig() of [CKW] on the p-adic generating functions z^(a_j + 2^(min(u,j)) * ZZ_2) for several values of a_j at once. We use section 3.5 of [CKW].

    INPUT:
    - ``a`` -- a list [a_1,a_2,...] of integers
    - ``u`` -- a natural number, or +Infinity
    - ``j`` -- a natural number (in practice, j=0,1,2,3)
    - ``t`` -- a rational number

    OUTPUT:
    a vector of rational numbers.

    EXAMPLES::

        sage: ig_v([0,1,2,3,4],+Infinity,3, 1/8)
        (1/1024, 15/16, 15/128, 15/16, 15/1024)

    """
    #evaluate the map Ig() of [CKW] on the p-adic generating functions z^(a_j + 2^(min(u,j)) * Z_2) for several values at once. Input a = [a_1,a_2,...]
    a_vals = [x.valuation(2) for x in a]
    j = min(u,j)
    f0 = (t ** j) / 2
    f = tuple((t ** i) * (1 - t/2) for i in range(j))
    return vector(0 if i<0 else f[i] if i < j else f0 for i in a_vals)

def ig_v_from_valuations(a_vals,u,j,t):
    r"""
    Convert p-adic generating functions to Igusa (local) zeta functions.

    This is the same as ig_v(a,u,j) with a = [2^(a_vals[0]),2^(a_vals[1]),...]. The difference is that we assume that the numbers a_j have already been converted to their 2-adic valuations.

    INPUT:
    - ``a`` -- a list [a_1,a_2,...] of valuations 0,1,...,+Infinity
    - ``u`` -- a natural number or +Infinity
    - ``j`` -- a natural number (in practice, j=0,1,2,3)

    OUTPUT:
    a vector of rational numbers.

    EXAMPLES::

        sage: ig_v_from_valuations([+Infinity,0,1,0,2],+Infinity,3,1/8)
        (1/1024, 15/16, 15/128, 15/16, 15/1024)

    """
    #same as ig_v, but the values a = [a_1,a_2,...] have already been converted to their 2adic valuations: a_vals = [v_2(a_1),v_2(a_2),...]
    j = min(u,j)
    f0 = (t ** j) / 2
    f = tuple((t ** i) * (1 - t/2) for i in range(j))
    return vector(0 if i<0 else f[i] if i < j else f0 for i in a_vals)


def hat_hq_v(a,u,Q0,t): #Helper function labeled H_1(a,u,Q0) in section 12 of [W]
    r"""
    Compute the helper functions H_1 from section 12 of [W] (denoted hat HQ in Appendix B of [CKW]).

    INPUT:
    - ``a`` -- a vector (a_1,a_2,...) of integers
    - ``u`` -- a natural number or +Infinity
    - ``Q0`` -- the result of twoadic_classify(Q) for a unimodular quadratic form Q
    - ``t`` -- a rational number

    OUTPUT: a vector of rational numbers.

    EXAMPLES::

        sage: hat_hq_v(vector([0,1,2,3,4]),+Infinity,[[],1,0],1/16)
        (3/128, 93/128, 3/128, 93/128, 3/128)

    """
    r = len(Q0[0]) + 2*(Q0[1] + Q0[2])
    two_r = 2 ** QQ(-r)
    a_vals = [QQ(valuation(x, 2)) if x else +Infinity for x in a]
    if Q0[0]:
        return ig_v_from_valuations(a_vals, u, 0,t) - two_r * ig_v_from_valuations(a_vals, u, 1,t)
    else:
        return (1 - two_r) * ig_v_from_valuations(a_vals,u,1,t)

def tilde_hq_v(a,u,Q0,t): #Helper function labeled H_2(a,u,Q0) in section 12 of [W]
    r"""
    Compute the helper functions H_2 from section 12 of [W] (denoted tilde HQ in Appendix B of [CKW])

    INPUT:
    - ``a`` -- a vector (a_1,a_2,...) of integers
    - ``u`` -- a natural number or +Infinity
    - ``Q0`` -- the result of twoadic_classify(Q) for a unimodular quadratic form Q
    - ``t`` -- a rational number

    OUTPUT: a vector of rational numbers.

    EXAMPLES::

        sage: tilde_hq_v(vector([0,1,2,3,4]),+Infinity,[[],1,0],1/16)
        (93/2048, 93/128, 3/2048, 93/128, 93/2048)

    """
    eps = (-1)**Q0[1]
    j = len(Q0[0])
    r = j + 2*(Q0[1] + Q0[2])
    a_vals = [QQ(valuation(x, 2)) if x else +Infinity for x in a]
    eps_two_r = eps * (2 ** QQ((-r) // 2))
    if not j:
        return (1 - eps_two_r) * (ig_v_from_valuations(a_vals,u,1,t) + eps_two_r * ig_v_from_valuations(a_vals,u,2,t))
    elif j == 1:
        b = Q0[0][0]
        b = vector([b]*len(a))
        return (1 - 2*eps_two_r)*ig_v_from_valuations(a_vals,u,0,t) - (2 ** QQ(-r)) * ig_v_from_valuations(a_vals,u,2,t) + eps_two_r * (ig_v_from_valuations(a_vals,u,2,t) + ig_v(a+b,u,2,t))
    else:
        b = Q0[0][0]
        c = Q0[0][1]
        if (b+c)%4:
            b = vector([b]*len(a))
            return (1 - 2*eps_two_r)*ig_v_from_valuations(a_vals,u,0,t) + eps_two_r * ig_v_from_valuations(a_vals,u,1,t) + eps_two_r * ig_v(a+b,u,2,t) - (2 ** QQ(-r)) * ig_v_from_valuations(a_vals,u,2,t)
        else:
            return ig_v_from_valuations(a_vals,u,0,t) - eps_two_r * ig_v_from_valuations(a_vals,u,1,t) + (eps_two_r - (2 ** QQ(-r))) * ig_v_from_valuations(a_vals,u,2,t)

def tilde_hq_diff_v(a,u,Q0,t): #Helper function labeled H_3(a,u,Q0) in section 12 of [W]
    r"""
    Compute the helper functions H_3 from section 12 of [W] (denoted (HQ-tilde HQ) in Appendix B of [CKW])

    INPUT:
    - ``a`` -- a vector (a_1,a_2,...) of integers
    - ``u`` -- a natural number or +Infinity
    - ``Q0`` -- the result of twoadic_classify(Q) for a unimodular quadratic form Q
    - ``t`` -- a rational number

    OUTPUT: a vector of rational numbers

    EXAMPLES::

        sage: tilde_hq_diff_v(vector([0,1,2,3,4]),+Infinity,[[1,3],1,0], 1/16)
        (15/131072, 0, 0, 0, -15/131072)

    """
    if not Q0[0]:
        return vector([0]*len(a))
    r = QQ(len(Q0[0]) + 2*(Q0[1] + Q0[2]))
    two_r = 2 ** (-r)
    if len(Q0[0]) == 1:
        b = Q0[0][0]
        b = vector([b]*len(a))
        ab_vals = [QQ(valuation(x, 2)) if x else +Infinity for x in a+b]
        return two_r* (ig_v_from_valuations(ab_vals, u, 3,t) - ig_v_from_valuations(ab_vals, u, 2,t))
    else:
        [b,c] = Q0[0]
        a_vals = [QQ(valuation(x, 2)) if x else +Infinity for x in a]
        if (b+c)%4:
            bc = vector([b+c]*len(a))
            return -2*two_r * ig_v_from_valuations(a_vals, u, 1,t) + two_r * (ig_v_from_valuations(a_vals, u, 2,t) + ig_v(a+bc, u, 3,t))
        else:
            eps = (-1) ** ((b+c)/4)
            return eps * two_r * (ig_v_from_valuations(a_vals, u, 3,t) - ig_v_from_valuations(a_vals, u, 2,t))

def iaqqq_v(a,u,U0,U1,U2,t):
    r"""
    Compute the helper functions I_a^u(U0,U1,U2) from section 12 of [W].

    This computes the helper functions I_a^u(U0,U1,U2) which appear in section 12 of [W]. These are used in the generalization of Theorem 2.3 of [CKW] to quadratic polynomials with arbitrary constant term.

    INPUT:
    - ``a`` -- a vector (a_1,a_2,...) of integers
    - ``u`` -- a natural number or +Infinity
    - ``U0,U1,U2`` - integer quadratic forms that are unimodular at the prime 2
    - ``t`` -- a rational number

    OUTPUT: a vector of rational numbers

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1],[1,2]]))
        sage: iaqqq_v(vector([0,1,2,3,4]),+Infinity,Q,Q,Q,1/4)
        (21/128, 21/32, 3/128, 21/32, 21/128)

    """
    Q0 = twoadic_classify(U0)
    Q1 = twoadic_classify(U1)
    Q2 = twoadic_classify(U2)
    if Q1[0] and Q2[0]:
        return hat_hq_v(a, u, Q0, t)
    elif Q2[0]:
        return tilde_hq_v(a, u, Q0, t)
    eps1 = (-1) ** Q1[1]
    j = len(Q1[0])
    r1 = j + 2 * (Q1[1] + Q1[2])
    two_r = eps1 * (2 ** QQ((-r1) // 2))
    if not Q1[0]:
        return tilde_hq_v(a, u, Q0, t) + two_r * tilde_hq_diff_v(a, u, Q0, t)
    elif len(Q1[0]) == 1:
        c = vector([Q1[0][0]]*len(a))
        return hat_hq_v(a, u, Q0, t) + two_r * (tilde_hq_diff_v(a, u, Q0, t) + tilde_hq_diff_v(a + 2*c, u, Q0, t))
    else:
        c, d = Q1[0]
        if (c + d) % 4:
            c = vector([c]*len(a))
            return hat_hq_v(a, u, Q0, t) + two_r * tilde_hq_diff_v(a + 2*c, u, Q0, t)
        else:
            return hat_hq_v(a, u, Q0, t) + two_r * tilde_hq_diff_v(a, u, Q0, t)

def ig(a,u,j,t):
    r"""
    Convert p-adic generating functions to Igusa (local) zeta functions.

    This evaluates the map Ig() of [CKW] on the p-adic generating function z^(a + 2^(min(u,j)) * ZZ_2). We use section 3.5 of [CKW]. Non-vector version of ig_v().

    INPUT:
    - ``a`` -- an integer
    - ``u`` -- a natural number, or +Infinity
    - ``j`` -- a natural number (in practice, j=0,1,2,3)
    - ``t`` -- a rational number

    OUTPUT:
    a rational number

    EXAMPLES::

        sage: ig(0,+Infinity,3,1/4)
        1/128

    """
    j = min(u,j)
    if a:
        aval = QQ(valuation(a, 2))
        if aval < j:
            return (t ** aval) * (1 - t/2)
    return (t ** j) / 2

def hat_hq(a,u,Q0,t): #H1
    r"""
    Compute the helper function H_1 from section 12 of [W] (denoted hat HQ in Appendix B of [CKW]).

    Non-vector version of hat_hq_v().

    INPUT:
    - ``a`` -- an integer
    - ``u`` -- a natural number or +Infinity
    - ``Q0`` -- the result of twoadic_classify(Q) for a unimodular quadratic form Q
    - ``t`` -- a rational number

    OUTPUT: a rational number

    EXAMPLES::

        sage: hat_hq(0,+Infinity,[[],1,0],1/8)
        3/64

    """
    r = len(Q0[0]) + 2*(Q0[1] + Q0[2])
    two_r = 2 ** QQ(-r)
    if Q0[0]:
        return ig(a, u, 0, t) - two_r*ig(a, u, 1, t)
    else:
        return (1 - two_r)*ig(a, u, 1, t)

def tilde_hq(a,u,Q0,t): #H2
    r"""
    Compute the helper function H_2 from section 12 of [W] (denoted tilde HQ in Appendix B of [CKW])

    Non-vector version of tilde_hq_v().

    INPUT:
    - ``a`` -- an integer
    - ``u`` -- a natural number or +Infinity
    - ``Q0`` -- the result of twoadic_classify(Q) for a unimodular quadratic form Q
    - ``t`` -- a rational number

    OUTPUT: a rational number

    EXAMPLES::

        sage: tilde_hq(0,+Infinity,[[],1,0],1/8)
        45/512

    """
    eps = (-1) ** Q0[1]
    j = len(Q0[0])
    r = j + 2 * (Q0[1] + Q0[2])
    two_r = eps * (2 ** QQ((-r) // 2))
    if not j:
        return (1 - two_r)*(ig(a,u,1,t) + two_r*ig(a,u,2,t))
    elif j == 1:
        b = Q0[0][0]
        return (1 - 2 * two_r)*ig(a,u,0,t) - (2 ** QQ(-r)) * ig(a,u,2,t) + two_r * (ig(a,u,2,t) + ig(a+b,u,2,t))
    else:
        b, c = Q0[0]
        if (b+c)%4:
            return (1 - 2 * two_r)*ig(a,u,0,t) + two_r * (ig(a,u,1,t) + ig(a+b,u,2,t)) - (2 ** QQ(-r)) * ig(a,u,2,t)
        else:
            return ig(a,u,0,t) - two_r * ig(a,u,1,t) + (two_r - (2 ** QQ(-r))) * ig(a,u,2,t)

def tilde_hq_diff(a,u,Q0,t): #H3
    r"""
    Compute the helper function H_3 from section 12 of [W] (denoted (HQ-tilde HQ) in Appendix B of [CKW])

    Non-vector version of tilde_hq_diff_v()

    INPUT:
    - ``a`` -- an integer
    - ``u`` -- a natural number or +Infinity
    - ``Q0`` -- the result of twoadic_classify(Q) for a unimodular quadratic form Q
    - ``t`` -- a rational number

    OUTPUT: a rational number

    EXAMPLES::

        sage: tilde_hq_diff(0,+Infinity,[[1,3],1,0],1/8)
        7/16384

    """
    if not Q0[0]:
        return 0
    eps = (-1) ** Q0[1]%2
    j = len(Q0[0])
    r = QQ(j + 2*(Q0[1] + Q0[2]))
    two_r =  2 ** (-r)
    if j == 1:
        b = Q0[0][0]
        return two_r*(ig(a+b,u,3,t) - ig(a+b,u,2,t))
    else:
        b, c = Q0[0]
        if (b + c) % 4:
            return two_r * (ig(a, u, 2, t) + ig(a+b+c, u, 3, t) - 2 * ig(a, u, 1, t))
        else:
            return ((-1) ** ((b+c)/4)) * two_r * (ig(a, u, 3, t) - ig(a, u, 2, t))

@cached_function
def iaqqq(a,u,U0,U1,U2,t):
    r"""
    Compute the helper functions I_a^u(U0,U1,U2) from section 12 of [W].

    This computes the helper functions I_a^u(U0,U1,U2) which appear in section 12 of [W]. These are used in the generalization of Theorem 2.3 of [CKW] to quadratic polynomials with arbitrary constant term. Non-vector version of iaqqq_v()

    INPUT:
    - ``a`` -- an integer
    - ``u`` -- a natural number or +Infinity
    - ``U0,U1,U2`` -- integer quadratic forms that are unimodular at the prime 2
    - ``t`` -- a rational number

    OUTPUT: a rational number

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,1],[1,2]]))
        sage: iaqqq(0,+Infinity,Q,Q,Q,1/16)
        93/2048

    """
    Q0 = twoadic_classify(U0)
    Q1 = twoadic_classify(U1)
    Q2 = twoadic_classify(U2)
    if Q1[0] and Q2[0]:
        return hat_hq(a,u,Q0,t)
    elif Q2[0]:
        return tilde_hq(a,u,Q0,t)
    eps1 = (-1) ** Q1[1]
    j = len(Q1[0])
    r1 = (j + 2*(Q1[1] + Q1[2]))
    two_r1 = eps1 * (2 ** QQ((-r1) // 2))
    if not j:
        return tilde_hq(a,u,Q0,t) + two_r1 * tilde_hq_diff(a,u,Q0,t)
    elif j == 1:
        c = Q1[0][0]
        return hat_hq(a,u,Q0,t) + two_r1 * (tilde_hq_diff(a,u,Q0,t) + tilde_hq_diff(a+2*c,u,Q0,t))
    else:
        c = Q1[0][0]
        d = Q1[0][1]
        if (c+d)%4:
            return hat_hq(a,u,Q0,t) + two_r1 * tilde_hq_diff(a+2*c,u,Q0,t)
        else:
            return hat_hq(a,u,Q0,t) + two_r1 * tilde_hq_diff(a,u,Q0,t)

def twoadic_igusa_zetas(Q,L,c_list, t): #compute several igusa zeta functions at the same time
    r"""
    Compute special values of Igusa (local) zeta functions of quadratic polynomials at the prime p=2.

    This is the analogue of the function igusa_zetas() for the prime p=2. It computes the Igusa (local) zeta functions attached to the polynomials P_j(x) = Q(x) + L(x) + c_j for a list of integers c_list = [c_1,c_2,...].

    INPUT:
    - ``Q`` -- a quadratic form
    - ``L`` -- a vector
    - ``c_list`` -- a list of integers
    - ``t`` -- a rational number

    OUTPUT:
    a vector of rational numbers

    ALGORITHM: We use Chapter 5 of [CKW].

    EXAMPLES::

        sage: Q = QuadraticForm(matrix([[2,0],[0,4]]))
        sage: L = vector([0,0])
        sage: twoadic_igusa_zetas(Q,L,[0,1,2,3,4],1/8)
        (1/16, 15/16, 2055/32768, 15/16, 32775/524288)

    """
    big_r = Q.dim()
    new_Q, lins_gcd, const = twoadic_isospectral_normal_form(Q,L)
    c = vector(QQ(const + n) for n in c_list)
    Z = vector([0] * len(c))
    jordan_blocks = twoadic_jordan_blocks(new_Q)
    Qs = []
    vb = ZZ(lins_gcd).valuation(2)
    vc_s = tuple(valuation(n, 2) for n in c)
    try:
        vc_nonzero = max(cval for cval in vc_s if cval != +Infinity)
    except:
        vc_nonzero = 0
    vc = vc_nonzero if (0 not in c) else +Infinity
    vc_min = min(vc_s)
    if jordan_blocks:
        maxrange = jordan_blocks[-1][0] + 1
    else:
        maxrange = 2
    if lins_gcd:
        maxrange = max(maxrange,vb+1)
    maxrange = max(2, maxrange, vc_nonzero+3)
    jordans = [DiagonalQuadraticForm(ZZ,[])] * maxrange
    for k in range(len(jordan_blocks)):
        jordans[jordan_blocks[k][0]] += jordan_blocks[k][1]
    rs = [0] * maxrange
    zero_form = DiagonalQuadraticForm(ZZ,[])
    sq_form = DiagonalQuadraticForm(ZZ,[1])
    q_powers = []
    for k in range(maxrange):
        Qs.append(DiagonalQuadraticForm(ZZ,[]))
        ell = k
        while ell >= 0:
            Qs[k] += jordans[ell]
            rs[k] += ZZ(jordans[ell].dim())
            ell = ell - 2
        if k:
            q_powers.append(q_powers[k-1] * (2 ** (rs[k-1])))
        else:
            q_powers.append(1)
    if vb == Infinity:
        w = max(len(Qs)-1,1)
        bound = min(w, vc_nonzero + 1)
        Z = sum((t ** i) * iaqqq_v(c / (2 ** i), Infinity, Qs[i], Qs[i+1], jordans[i+2], t)/q_powers[i] for i in range(bound))
        u = []
        for j, cj in enumerate(c):
            cj_val = vc_s[j]
            if cj:
                u.append((t ** bound) * sum((t ** i) * iaqqq((cj >> i), Infinity, Qs[i], Qs[i+1], jordans[i+2], t) / q_powers[i] for i in range(bound, cj_val+1)) + (1 - t/2) * (t ** cj_val) / q_powers[cj_val + 1])
            else:
                u.append(((t ** (w-1)) * iaqqq(0, Infinity, Qs[w-1], Qs[w], zero_form ,t) / q_powers[w-1] + (t ** w) * iaqqq(0, Infinity, Qs[w], Qs[w-1], zero_form,t) / q_powers[w]) / (1 - t * t * (2 ** QQ(-big_r))))
        return Z + vector(u)
    else:
        Z = sum((t ** i) * iaqqq_v(c/(2 ** i), Infinity, Qs[i], Qs[i+1], jordans[i+2],t) / q_powers[i] for i in range(vb-2))
        if vb >= 1:
            Z += (t ** (vb-1)) * iaqqq_v(c/(2 ** (vb-1)),2,Qs[vb-1],sq_form,sq_form,t) / q_powers[vb-1]
            if vb >= 2:
                Z += (t ** (vb-2)) * iaqqq_v(c/(2 ** (vb-2)),2,Qs[vb-2],Qs[vb-1],sq_form,t) / q_powers[vb-2]
        u = []
        for c, val_cj in enumerate(vc_s):
            if val_cj >= vb:
                u.append((t ** vb) / (2 * q_powers[vb]))
            else:
                u.append((t ** val_cj) * (1 - t/2) / q_powers[val_cj+1])
        return Z + vector(u)

def L_values(L, c, S, p, k, t = None): #the Euler factors in the Eisenstein series, vector version
    r"""
    Compute the Euler factors in the Eisenstein series.

    This computes the Euler factors L_{\gamma, n}(k, p) which appear in the Eisenstein series, as in Theorem 4.6 of [BK].

    NOTE: We do not use the algorithm suggested in [BK].

    NOTE: The output is actually renormalized to (1 - p^(d/2 - k)) * L_{\gamma, n}(k, p) where d is the lattice rank.

    NOTE: if t is not None then the variable 'k' is never used

    INPUT:
    - ``L`` -- an integer vector of size equal to the dimension of S
    - ``c`` -- a list of integers
    - ``S`` -- the Gram matrix of a nondegenerate integral quadratic form
    - ``p`` -- a prime
    - ``k`` -- a half-integer for which 2*k + signature(S) = 0 mod 4. (This will be the weight of the Eisenstein series.)

    OUTPUT: a vector of rational numbers. If L = 2*g*S and c = [c_1,c_2,...] where c_i = 2n_i + g*S*g then the result is the list [L_{g, n_1}(k, p), L_{g, n_2}(k, p), ...] of [BK] L-values.

    EXAMPLES::

        sage: S = matrix([[2,1],[1,2]])
        sage: L = vector([0, 0])
        sage: c = [0,1,2,3,4]
        sage: L_values(L, c, S, 3, 7)
        (1, 730/729, 728/729, 531442/531441, 730/729)

    """

    if not c:
        return []
    if t is None:
        t = p ** (1 + ZZ(S.nrows())/2 - k)
    if t == 1:#kluge. can it be fixed??
        t = p
        S = block_diagonal_matrix([S, matrix([[0,1],[1,0]])])
        L = vector(list(L) + [0,0])
    Q = QuadraticForm(S)
    one_v = vector([1 - t/p]*len(c))
    if p == 2:
        try:
            return (one_v - twoadic_igusa_zetas(Q,L,c,t)) / (1 - t)
        except ZeroDivisionError:#should only happen in weight one
            t0, = PolynomialRing(QQ, 't0').gens()
            fv = (one_v - twoadic_igusa_zetas(Q,L,c,t0)) / (1 - t0)
            return vector(f(t) for f in fv)
    else:
        try:
            return (one_v - t*igusa_zetas(Q,L,c,p,t)) / (1 - t) #for technical reasons the igusa zetas here are multiplied by 't' and the igusa zetas at p=2 are not!
        except ZeroDivisionError:#should only happen in weight one
            t0, = PolynomialRing(QQ, 't0').gens()
            fv = (one_v - t0*igusa_zetas(Q,L,c,p,t0)) / (1 - t0)
            return vector(f(t) for f in fv)

def L_value_deriv(L, c, S, p, k, t = None):
    r"""
    ...
    """
    r, y = PolynomialRing(QQ, 't').objgen()
    x = L_values(L, c, S, p, k, t = y)
    N = ZZ(S.nrows())
    x = [-y * x.derivative() for x in x]
    if t is None:
        t = p ** (1 + N/2 - k)
        x = [x(t) for x in x]
    return x

@cached_function
def quadratic_L_function__cached(k,D):
    return quadratic_L_function__exact(k,D)

@cached_function
def quadratic_L_function__corrector(k,D): #quadratic_L_function__exact() includes the Euler factors for primes dividing D.
    d = fundamental_discriminant(D)
    return prod((1 - kronecker_symbol(d,p)*(p**(-k))) for p in ZZ(D).prime_factors() if d%p)

def quadratic_L_function__correct(k,D): #fix quadratic_L_function
    return quadratic_L_function__cached(k,D)*quadratic_L_function__corrector(k,D)
