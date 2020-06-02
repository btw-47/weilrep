r"""

Sage code for Jacobi forms

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

class JacobiForms:
    r"""
    The JacobiForms class represents the graded module of Jacobi forms for some lattice index.

    INPUT:

    A JacobiForms instance is constructed by calling JacobiForms(m), where:
    - ``m`` -- a positive-definite Gram matrix (symmetric, integral, even diagonal); OR:
    - ``m`` -- a natural number (not 0)

    """
    def __init__(self, index_matrix, weilrep = None):
        if index_matrix in ZZ:
            self.__index_matrix = matrix([[2*index_matrix]])
        else:
            self.__index_matrix = index_matrix
        if weilrep:
            self.__weilrep = weilrep

    def index(self):
        r"""
        Return self's index (as a scalar if one elliptic variable).

        EXAMPLES::

            sage: JacobiForms(1).index()
            1

            sage: JacobiForms(matrix([[2, 1],[1, 2]])).index()
            [2 1]
            [1 2]
        """
        S = self.index_matrix()
        if S.nrows() == 1:
            return ZZ(S[0,0]/2)
        else:
            return S

    def index_matrix(self):
        r"""
        Return self's index as a matrix.

        EXAMPLES::

            sage: JacobiForms(1).index_matrix()
            [2]
        """
        return self.__index_matrix

    def nvars(self):
        r"""
        Return the number of elliptic variables.

        EXAMPLES::

            sage: JacobiForms(1).nvars()
            1
        """
        try:
            return self.__nvars
        except:
            self.__nvars = self.index_matrix().nrows()
            return self.__nvars

    def theta_decomposition(self):
        r"""
        Return the WeilRep associated to this Gram matrix.

        EXAMPLES::

            sage: JacobiForms(1).theta_decomposition()
            Weil representation associated to the Gram matrix
            [2]
        """
        try:
            return self.__weilrep
        except AttributeError:
            self.__weilrep = WeilRep(self.index_matrix())
            return self.__weilrep

    weilrep = theta_decomposition

    ##construction of Jacobi Forms associated to this index

    def eisenstein_series(self, k, prec, allow_small_weight = False):
        r"""
        Compute the Jacobi Eisenstein series of weight k.

        INPUT:
        - ``k`` -- the weight
        - ``prec`` -- the precision of the Fourier expansion

        EXAMPLES::

            sage: JacobiForms(1).eisenstein_series(4, 5)
            1 + (w_0^-2 + 56*w_0^-1 + 126 + 56*w_0 + w_0^2)*q + (126*w_0^-2 + 576*w_0^-1 + 756 + 576*w_0 + 126*w_0^2)*q^2 + (56*w_0^-3 + 756*w_0^-2 + 1512*w_0^-1 + 2072 + 1512*w_0 + 756*w_0^2 + 56*w_0^3)*q^3 + (w_0^-4 + 576*w_0^-3 + 2072*w_0^-2 + 4032*w_0^-1 + 4158 + 4032*w_0 + 2072*w_0^2 + 576*w_0^3 + w_0^4)*q^4 + O(q^5)

        """

        return self.weilrep().eisenstein_series(k - self.nvars()/2, prec, allow_small_weight = allow_small_weight).jacobi_form()

    def eisenstein_newform(self, k, b, prec, allow_small_weight = False):
        r"""
        Compute certain newform Jacobi Eisenstein series of weight k.

        WARNING: this is experimental and also slow!!

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- a vector, or possibly (if self's index is an integer) an integer
        - ``prec`` -- the precision of the Fourier expansion

        """
        if b in ZZ:
            m = self.index()
            f = m.squarefree_part()
            f0 = isqrt(m / f)
            b = vector([b / f0])
        return self.weilrep().eisenstein_newform(k - self.nvars()/2, b, prec, allow_small_weight = allow_small_weight).jacobi_form()

    def eisenstein_oldform(self, k, b, prec, allow_small_weight = False):
        r"""
        Compute certain oldform Jacobi Eisenstein series of weight k.

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- a vector, or possibly (if self's index is an integer) an integer
        - ``prec`` -- the precision of the Fourier expansion

        """
        if b in ZZ:
            m = self.index()
            f = m.squarefree_part()
            f0 = isqrt(m / f)
            b = vector([b / f0])
        return self.weilrep().eisenstein_oldform(k - self.nvars()/2, b, prec, allow_small_weight = allow_small_weight).jacobi_form()

    ## dimensions associated to this index

    def cusp_forms_dimension(self, weight):
        r"""
        Dimension of Jacobi cusp forms.

        This computes the dimension of the space of Jacobi cusp forms of self's index and the given weight.

        EXAMPLES::

            sage: JacobiForms(1).cusp_forms_dimension(10)
            1

            sage: JacobiForms(13).cusp_forms_dimension(3)
            1
        """
        if self.nvars() == 1 and weight == 2:
            dim = self.weilrep().cusp_forms_dimension(3/2, force_Riemann_Roch = True)
            N = self.index()
            sqrtN = isqrt(N)
            return dim + (len(divisors(N)) + N.is_square())/2
        if weight <= 1:
            return 0
        return self.weilrep().cusp_forms_dimension(weight - self.nvars()/2)

    def dimension(self, weight):
        r"""
        Dimension of Jacobi forms.

        EXAMPLES::

            sage: JacobiForms(2).dimension(11)
            1

            sage: JacobiForms(4).dimension(8)
            3

            sage: JacobiForms(37).dimension(2)
            1

            sage: JacobiForms(49).dimension(2)
            2
        """
        if self.nvars() == 1 and weight == 2:
            dim = self.weilrep().modular_forms_dimension(3/2, force_Riemann_Roch = True)
            N = self.index()
            sqrtN = isqrt(N)
            return dim + len([d for d in divisors(N) if d <= sqrtN and N % (d * d)])
        if weight <= 1:
            return 0
        return self.weilrep().modular_forms_dimension(weight - self.nvars()/2)

    jacobi_forms_dimension = dimension

    def rank(self):
        r"""
        Compute the rank of the Jacobi forms of this index as a graded module over the ring of scalar modular forms.

        ALGORITHM: it's the determinant of self's index (matrix)
        """
        return self.index_matrix().determinant()

    ## bases of spaces associated to this index

    def cusp_forms_basis(self, weight, prec = 0, try_theta_blocks = None, verbose = False):
        r"""
        Compute a basis of Jacobi cusp forms.

        This computes a basis of Jacobi cusp forms of the given weight up to precision "prec". If "prec" is not given then a Sturm bound is used.

        INPUT:
         - ``weight`` -- the weight
         - ``prec`` -- the precision of the Fourier expansions (with respect to the variable 'q')
         - ``verbose`` -- boolean (default False); if True then we add commentary throughout the computation

        OUTPUT: a list of JacobiForm's

        EXAMPLES::

            sage: JacobiForms(1).cusp_forms_basis(10, 5)
            [(w_0^-1 - 2 + w_0)*q + (-2*w_0^-2 - 16*w_0^-1 + 36 - 16*w_0 - 2*w_0^2)*q^2 + (w_0^-3 + 36*w_0^-2 + 99*w_0^-1 - 272 + 99*w_0 + 36*w_0^2 + w_0^3)*q^3 + (-16*w_0^-3 - 272*w_0^-2 - 240*w_0^-1 + 1056 - 240*w_0 - 272*w_0^2 - 16*w_0^3)*q^4 + O(q^5)]

            sage: JacobiForms(2).cusp_forms_basis(11, 5)
            [(-w_0^-1 + w_0)*q + (w_0^-3 + 21*w_0^-1 - 21*w_0 - w_0^3)*q^2 + (-21*w_0^-3 - 189*w_0^-1 + 189*w_0 + 21*w_0^3)*q^3 + (-w_0^-5 + 189*w_0^-3 + 910*w_0^-1 - 910*w_0 - 189*w_0^3 + w_0^5)*q^4 + O(q^5)]

        """
        if self.nvars() == 1 and weight <= 3:
            dim = self.cusp_forms_dimension(weight)
            if not dim:
                return []
            jacobi_dim = self.dimension(weight)
            if verbose:
                print('I need to find %d Jacobi cusp form' %dim + ['s.', '.'][dim == 1])
            if try_theta_blocks is None:
                try_theta_blocks = (dim < 3) and (jacobi_dim == dim)#arbitrary?
            if try_theta_blocks and (jacobi_dim == dim):
                if verbose:
                    print('I will look for theta blocks of weight %d.'%weight)
                if weight == 2:
                    return weight_two_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms = self, verbose = verbose)
                else:
                    return weight_three_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms = self, verbose = verbose)
        wt = weight - self.nvars() / 2
        if verbose:
            print('I will look for the corresponding vector-valued cusp forms of weight %s.' %wt)
            print('-'*60)
        L = self.weilrep().cusp_forms_basis(wt, prec, verbose = verbose)
        if not L:
            return []
        if verbose:
            print('-'*60)
            print('I will now convert these cusp forms to Jacobi forms.')
        if len(L) > 1:
            return L.jacobi_forms()
        return [L[0].jacobi_form()]

    def jacobi_forms_basis(self, weight, prec = 0, try_theta_blocks = None, verbose = False):
        r"""
        Compute a basis of Jacobi forms.

        This computes a basis of Jacobi forms of the given weight up to precision "prec". If "prec" is not given then a Sturm bound is used.

        NOTE: can also be called simply with "basis()"

        INPUT:
         - ``weight`` -- the weight
         - ``prec`` -- the precision of the Fourier expansions (with respect to the variable 'q')
         - ``try_theta_blocks`` -- if True then in weight 2 or 3 we first try to find theta blocks which span the space.
         - ``verbose`` -- boolean (default False); if True then we add commentary throughout the computation

        OUTPUT: a list of JacobiForm's

        EXAMPLES::

            sage: JacobiForms(1).jacobi_forms_basis(10, 5)
            [1 + (w_0^-2 - 266 + w_0^2)*q + (-266*w_0^-2 - 26752*w_0^-1 - 81396 - 26752*w_0 - 266*w_0^2)*q^2 + (-81396*w_0^-2 - 1225728*w_0^-1 - 2582328 - 1225728*w_0 - 81396*w_0^2)*q^3 + (w_0^-4 - 26752*w_0^-3 - 2582328*w_0^-2 - 17211264*w_0^-1 - 29700762 - 17211264*w_0 - 2582328*w_0^2 - 26752*w_0^3 + w_0^4)*q^4 + O(q^5), (w_0^-1 - 2 + w_0)*q + (-2*w_0^-2 - 16*w_0^-1 + 36 - 16*w_0 - 2*w_0^2)*q^2 + (w_0^-3 + 36*w_0^-2 + 99*w_0^-1 - 272 + 99*w_0 + 36*w_0^2 + w_0^3)*q^3 + (-16*w_0^-3 - 272*w_0^-2 - 240*w_0^-1 + 1056 - 240*w_0 - 272*w_0^2 - 16*w_0^3)*q^4 + O(q^5)]

        """
        if self.nvars() == 1 and weight <= 3:
            if weight <= 1:
                return []
            dim = self.dimension(weight)
            if not dim:
                return []
            if verbose:
                print('I need to find %d Jacobi form' %dim + ['s.', '.'][dim == 1])
            if try_theta_blocks is None:
                try_theta_blocks = (dim < 3)#arbitrary?
            if try_theta_blocks:
                if verbose:
                    print('I will look for theta blocks of weight %d.'%weight)
                if weight == 2:
                    return weight_two_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms = self, verbose = verbose)
                else:
                    return weight_three_basis_from_theta_blocks(self.index(), prec, dim, jacobiforms = self, verbose = verbose)
        wt = weight - self.nvars() / 2
        if verbose:
            print('I will look for the corresponding vector-valued modular forms of weight %s.' %wt)
            print('-'*60)
        L = self.weilrep().modular_forms_basis(wt, prec, verbose = verbose)
        if not L:
            return []
        if verbose:
            print('-'*60)
            print('I will now convert these modular forms to Jacobi forms.')
        if len(L) > 1:
            return L.jacobi_forms()
        return [L[0].jacobi_form()]

    basis = jacobi_forms_basis

    def weak_forms_basis(self, weight, prec = 0, verbose = False, convert_to_Jacobi_forms = True):
        r"""
        Compute a basis of weak Jacobi forms.

        A weak Jacobi form is a holomorphic function f(tau, z) satisfying the transformations of Jacobi forms and with a Fourier expansion of the form
        f(tau, z) = \sum_{n=0}^{\infty} \sum_r c(n,r) q^n \zeta^r
        i.e. with the usual restriction on 'n' but without the restriction on 'r'.

        This computes a basis of weak Jacobi forms of the given weight up to precision "prec". If "prec" is not given then a Sturm bound is used.

        INPUT:
        - ``weight`` -- the weight
        - ``prec`` -- the precision of the Fourier expansion (with respect to the variable 'q')
        - ``verbose`` -- boolean (default False); if True then we add commentary throughout the computation
        - ``convert_to_Jacobi_forms`` -- boolean (default True); if True then we convert the computed vector-valued modular forms to Jacobi forms.

        OUTPUT: a list of JacobiForm's

        EXAMPLES::

            sage: JacobiForms(3).weak_forms_basis(0, 5)
            [(w_0^-1 + 2 + w_0) + (-2*w_0^-3 - 2*w_0^-2 + 2*w_0^-1 + 4 + 2*w_0 - 2*w_0^2 - 2*w_0^3)*q + (w_0^-5 - 2*w_0^-4 - 6*w_0^-3 - 4*w_0^-2 + 5*w_0^-1 + 12 + 5*w_0 - 4*w_0^2 - 6*w_0^3 - 2*w_0^4 + w_0^5)*q^2 + (2*w_0^-6 + 2*w_0^-5 - 4*w_0^-4 - 14*w_0^-3 - 10*w_0^-2 + 12*w_0^-1 + 24 + 12*w_0 - 10*w_0^2 - 14*w_0^3 - 4*w_0^4 + 2*w_0^5 + 2*w_0^6)*q^3 + (w_0^-7 + 4*w_0^-6 + 5*w_0^-5 - 10*w_0^-4 - 30*w_0^-3 - 20*w_0^-2 + 24*w_0^-1 + 52 + 24*w_0 - 20*w_0^2 - 30*w_0^3 - 10*w_0^4 + 5*w_0^5 + 4*w_0^6 + w_0^7)*q^4 + O(q^5), (w_0^-2 + 14 + w_0^2) + (w_0^-4 + 40*w_0^-3 - 76*w_0^-2 - 168*w_0^-1 + 406 - 168*w_0 - 76*w_0^2 + 40*w_0^3 + w_0^4)*q + (-76*w_0^-4 + 768*w_0^-3 - 1048*w_0^-2 - 1792*w_0^-1 + 4296 - 1792*w_0 - 1048*w_0^2 + 768*w_0^3 - 76*w_0^4)*q^2 + (14*w_0^-6 - 168*w_0^-5 - 1048*w_0^-4 + 7192*w_0^-3 - 7998*w_0^-2 - 12656*w_0^-1 + 29328 - 12656*w_0 - 7998*w_0^2 + 7192*w_0^3 - 1048*w_0^4 - 168*w_0^5 + 14*w_0^6)*q^3 + (406*w_0^-6 - 1792*w_0^-5 - 7998*w_0^-4 + 45312*w_0^-3 - 45558*w_0^-2 - 68096*w_0^-1 + 155452 - 68096*w_0 - 45558*w_0^2 + 45312*w_0^3 - 7998*w_0^4 - 1792*w_0^5 + 406*w_0^6)*q^4 + O(q^5), (w_0^-3 + 34 + w_0^3) + (-186*w_0^-3 + 2430*w_0^-2 - 8262*w_0^-1 + 12036 - 8262*w_0 + 2430*w_0^2 - 186*w_0^3)*q + (2430*w_0^-4 - 35307*w_0^-3 + 175932*w_0^-2 - 425493*w_0^-1 + 564876 - 425493*w_0 + 175932*w_0^2 - 35307*w_0^3 + 2430*w_0^4)*q^2 + (34*w_0^-6 - 8262*w_0^-5 + 175932*w_0^-4 - 1281814*w_0^-3 + 4623318*w_0^-2 - 9567396*w_0^-1 + 12116376 - 9567396*w_0 + 4623318*w_0^2 - 1281814*w_0^3 + 175932*w_0^4 - 8262*w_0^5 + 34*w_0^6)*q^3 + (12036*w_0^-6 - 425493*w_0^-5 + 4623318*w_0^-4 - 24202674*w_0^-3 + 72869868*w_0^-2 - 137425977*w_0^-1 + 169097844 - 137425977*w_0 + 72869868*w_0^2 - 24202674*w_0^3 + 4623318*w_0^4 - 425493*w_0^5 + 12036*w_0^6)*q^4 + O(q^5)]

            sage: JacobiForms(matrix([[2,1],[1,2]])).weak_forms_basis(-3, 5)
            [(-w_0*w_1 + w_0 + w_1 - w_1^-1 - w_0^-1 + w_0^-1*w_1^-1) + (w_0^2*w_1^2 - w_0^2 - 8*w_0*w_1 - w_1^2 + 8*w_0 + 8*w_1 - 8*w_1^-1 - 8*w_0^-1 + w_1^-2 + 8*w_0^-1*w_1^-1 + w_0^-2 - w_0^-2*w_1^-2)*q + (-w_0^3*w_1^2 - w_0^2*w_1^3 + w_0^3*w_1 + 8*w_0^2*w_1^2 + w_0*w_1^3 - 8*w_0^2 - 44*w_0*w_1 - 8*w_1^2 + w_0^2*w_1^-1 + 44*w_0 + 44*w_1 + w_0^-1*w_1^2 - w_0*w_1^-2 - 44*w_1^-1 - 44*w_0^-1 - w_0^-2*w_1 + 8*w_1^-2 + 44*w_0^-1*w_1^-1 + 8*w_0^-2 - w_0^-1*w_1^-3 - 8*w_0^-2*w_1^-2 - w_0^-3*w_1^-1 + w_0^-2*w_1^-3 + w_0^-3*w_1^-2)*q^2 + (-8*w_0^3*w_1^2 - 8*w_0^2*w_1^3 + 8*w_0^3*w_1 + 44*w_0^2*w_1^2 + 8*w_0*w_1^3 - 44*w_0^2 - 192*w_0*w_1 - 44*w_1^2 + 8*w_0^2*w_1^-1 + 192*w_0 + 192*w_1 + 8*w_0^-1*w_1^2 - 8*w_0*w_1^-2 - 192*w_1^-1 - 192*w_0^-1 - 8*w_0^-2*w_1 + 44*w_1^-2 + 192*w_0^-1*w_1^-1 + 44*w_0^-2 - 8*w_0^-1*w_1^-3 - 44*w_0^-2*w_1^-2 - 8*w_0^-3*w_1^-1 + 8*w_0^-2*w_1^-3 + 8*w_0^-3*w_1^-2)*q^3 + (w_0^4*w_1^3 + w_0^3*w_1^4 - w_0^4*w_1 - 44*w_0^3*w_1^2 - 44*w_0^2*w_1^3 - w_0*w_1^4 + 44*w_0^3*w_1 + 192*w_0^2*w_1^2 + 44*w_0*w_1^3 - w_0^3*w_1^-1 - 192*w_0^2 - 726*w_0*w_1 - 192*w_1^2 - w_0^-1*w_1^3 + 44*w_0^2*w_1^-1 + 726*w_0 + 726*w_1 + 44*w_0^-1*w_1^2 - 44*w_0*w_1^-2 - 726*w_1^-1 - 726*w_0^-1 - 44*w_0^-2*w_1 + w_0*w_1^-3 + 192*w_1^-2 + 726*w_0^-1*w_1^-1 + 192*w_0^-2 + w_0^-3*w_1 - 44*w_0^-1*w_1^-3 - 192*w_0^-2*w_1^-2 - 44*w_0^-3*w_1^-1 + w_0^-1*w_1^-4 + 44*w_0^-2*w_1^-3 + 44*w_0^-3*w_1^-2 + w_0^-4*w_1^-1 - w_0^-3*w_1^-4 - w_0^-4*w_1^-3)*q^4 + O(q^5)]

        """
        S = self.index_matrix()
        w = self.weilrep()
        indices = w.rds(indices = True)
        if verbose:
            print('I am looking for weak Jacobi forms of weight %d.' %weight)
        #N = self.longest_short_vector_norm()
        svn = self.short_vector_norms_by_component()
        N = max(svn)
        if verbose:
            print('I will compute nearly-holomorphic modular forms with a pole in infinity of order at most %s.'%N)
            print('-'*60)
        k = weight - S.nrows() / 2
        L = w.nearly_holomorphic_modular_forms_basis(k, N, prec, verbose = verbose)
        if not L:
            return []
        X = WeilRepModularFormsBasis(k, [x for x in L if all(f[2].valuation() + f[1] >= -svn[i] for i, f in enumerate(x.fourier_expansion()) if indices[i] is None)], w)
        if not convert_to_Jacobi_forms:
            return X
        if verbose:
            print('-'*60)
            print('%d of these nearly-holomorphic forms appear to arise as theta decompositions of Jacobi forms.'%len(X))
            print('I am converting these modular forms to Jacobi forms.')
        jf = WeilRepModularFormsBasis(k, [x for x in L if all(f[2].valuation() + f[1] >= -svn[i] for i, f in enumerate(x.fourier_expansion()) if indices[i] is None)], w).jacobi_forms()
        if verbose:
            print('I found %d nearly-holomorphic modular forms.'%len(jf))
            print('I will now check whether any of these modular forms occur as theta decompositions of weak Jacobi forms.')
        return jf
        #return [jf for jf in jacobi_forms if not any(jf.fourier_expansion()[-1-i] for i in range(ceil(N)))]

    ## other:

    def short_vector_norms_by_component(self):
        r"""
        Computes the expression max( min( Q(x): x in ZZ^N + g): g in self.ds())

        NOTE: used in weak_forms_basis() to determine which nearly-holomorphic modular forms might product weak Jacobi forms

        TODO: is there a better way to do this?? for now we use PARI qfminim() to find some short vectors

        EXAMPLES::

            sage: JacobiForms(matrix([[2,1],[1,2]])).longest_short_vector_norm()
            1/3

        """
        try:
            return self.__short_vector_norms
        except AttributeError:
            pass
        w = self.weilrep()
        ds_dict = w.ds_dict()
        rds = w.rds()
        indices = w.rds(indices = True)
        S = w.gram_matrix()
        N_triv = max(ceil(g*S*g/2) for g in rds) #a trivial upper bound for longest_short_vector_norm(). next we'll use PARI qfminim() to lower this
        found_vectors = [None if (x is None) else (-1) for x in indices]
        found_vectors[0] = 0
        S_inv = S.inverse()
        try:
            _, _, vs_matrix = pari(S_inv).qfminim(N_triv + 1, flag = 2)
            vs_list = vs_matrix.sage().columns()
            for v in vs_list:
                r = S_inv * v
                j = ds_dict[tuple(frac(x) for x in r)]
                if indices[j]:
                    j = indices[j]
                if found_vectors[j] is None:
                    found_vectors[j] = v * r / 2
                else:
                    found_vectors[j] = min(v * r/2, found_vectors[j])
            #return max(found_vectors)
            self.__short_vector_norms = found_vectors
            return found_vectors
        except PariError:
            lvl = w.level()
            S_adj = S_inv * lvl
            vs = QuadraticForm(S_adj).short_vector_list_up_to_length(lvl * N_triv + 1)
            for n in range(len(vs)):
                r_norm = n / lvl
                for v in vs[n]:
                    r = S_inv * v
                    j = ds_dict[tuple(frac(x) for x in r)]
                if indices[j]:
                    j = indices[j]
                if found_vectors[j] is None:
                    found_vectors[j] = v * r / 2
                if all(x is not None for x in found_vectors):
                    #return max(found_vectors)
                    self.__short_vectors_norms = found_vectors
                    return found_vectors

class JacobiForm:
    r"""
    The JacobiForm class represents Jacobi forms.

    INPUT: JacobiForm instances are constructed by calling

    JacobiForm(k, S, f)

    where:

    - ``k`` -- the weight (integer)
    - ``S`` -- the index: an integer or a Gram matrix
    - ``f`` -- the Fourier expansion. This is a power series in the variable 'q' over the base ring of Laurent polynomials in the variables 'w_0, ..., w_d' over QQ.
    - ``weilrep`` -- a WeilRep instance attached to this Jacobi form (default None)

    """
    def __init__(self, weight, index_matrix, fourier_expansion, modform = None, weilrep = None):
        self.__weight = weight
        self.__index_matrix = index_matrix
        self.__fourier_expansion = fourier_expansion
        if weilrep:
            self.__weilrep = weilrep
        if modform:
            self.__theta = modform

    def __repr__(self):
        return str(self.fourier_expansion())

    ## basic attributes

    def base_ring(self):
        r"""
        Laurent polynomial ring representing self's elliptic variables.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).base_ring()
            Univariate Laurent Polynomial Ring in w_0 over Rational Field
        """
        return LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(self.index_matrix().nrows())))

    def coefficient_vector(self, starting_from = None, ending_with = None, correct = True):
        r"""
        Return self's non-redundant Fourier coefficients c(n, r) as a vector sorted by increasing value of n - r^2 / (4m) (or its appropriate generalization to matrix index)

        This simply returns the coefficient vector of self's theta decomposition.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).coefficient_vector()
            (1, 56, 126, 576, 756, 1512, 2072, 4032, 4158, 5544)

        """
        try:
            return self.__coefficient_vector
        except:
            v = self.theta_decomposition(correct = correct).coefficient_vector(starting_from = starting_from, ending_with = ending_with)
            if correct:
                self.__coefficient_vector = v
            return v

    def fourier_expansion(self):
        r"""
        Return self's Fourier expansion.

        This is a Power series in the variable 'q' with coefficients in a ring of Laurent polynomials over QQ in variables w_0, ... w_d.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).fourier_expansion()
            1 + (w_0^-2 + 56*w_0^-1 + 126 + 56*w_0 + w_0^2)*q + (126*w_0^-2 + 576*w_0^-1 + 756 + 576*w_0 + 126*w_0^2)*q^2 + (56*w_0^-3 + 756*w_0^-2 + 1512*w_0^-1 + 2072 + 1512*w_0 + 756*w_0^2 + 56*w_0^3)*q^3 + (w_0^-4 + 576*w_0^-3 + 2072*w_0^-2 + 4032*w_0^-1 + 4158 + 4032*w_0 + 2072*w_0^2 + 576*w_0^3 + w_0^4)*q^4 + O(q^5)
        """
        return self.__fourier_expansion

    qexp = fourier_expansion

    def index(self):
        r"""
        Return self's index.

        If the index is a rank one matrix then return self's index as a scalar instead.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).index()
            1
        """
        S = self.index_matrix()
        e = S.nrows()
        if e == 1:
            return ZZ(S[0][0]/2)
        else:
            return S

    def index_matrix(self):
        r"""
        Return self's index as a matrix.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).index_matrix()
            [2]
        """
        return self.__index_matrix

    def modform(self):
        r"""
        Try to return self's theta decomposition without computing.
        """
        return self.__theta

    def precision(self):
        r"""
        Return self's precision (with respect to the variable 'q').

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).precision()
            5
        """
        try:
            return self.__precision
        except:
            self.__precision = self.fourier_expansion().prec()
            return self.__precision
    prec = precision

    def q_coefficients(self):
        r"""
        Return self's Fourier coefficients with respect to the variable 'q'.

        OUTPUT: a list of Laurent polynomials

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).q_coefficients()
            [1, w_0^-2 + 56*w_0^-1 + 126 + 56*w_0 + w_0^2, 126*w_0^-2 + 576*w_0^-1 + 756 + 576*w_0 + 126*w_0^2, 56*w_0^-3 + 756*w_0^-2 + 1512*w_0^-1 + 2072 + 1512*w_0 + 756*w_0^2 + 56*w_0^3, w_0^-4 + 576*w_0^-3 + 2072*w_0^-2 + 4032*w_0^-1 + 4158 + 4032*w_0 + 2072*w_0^2 + 576*w_0^3 + w_0^4]
        """
        return list(self.fourier_expansion())

    def theta_decomposition(self, correct = True):
        r"""
        Return self's theta decomposition.

        OUTPUT: a WeilRepModularForm

        WARNING: passing to JacobiForm and back to WeilRepModularForm can incur a severe precision loss! we try to avoid this by caching the original modular form

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).theta_decomposition()
            [(0), 1 + 126*q + 756*q^2 + 2072*q^3 + 4158*q^4 + O(q^5)]
            [(1/2), 56*q^(3/4) + 576*q^(7/4) + 1512*q^(11/4) + 4032*q^(15/4) + 5544*q^(19/4) + O(q^5)]

            sage: (jacobi_eisenstein_series(4, 1, 5)^2).theta_decomposition() #we lose precision here! it's unavoidable I guess
            [(0), 1 + 252*q + 23662*q^2 + O(q^3)]
            [(1/4), 112*q^(7/8) + 15376*q^(15/8) + 248112*q^(23/8) + O(q^3)]
            [(1/2), 2*q^(1/2) + 3640*q^(3/2) + 99288*q^(5/2) + O(q^3)]
            [(3/4), 112*q^(7/8) + 15376*q^(15/8) + 248112*q^(23/8) + O(q^3)]
        """
        try:
            return self.__theta
        except:
            if correct:
                N = JacobiForms(self.index()).longest_short_vector_norm()
            else:
                N = 0
            f = self.fourier_expansion()
            S = self.index_matrix()
            w = self.weilrep()
            e = S.nrows()
            prec = f.prec() - ceil(N)
            val = f.valuation()
            S_inv = S.inverse()
            ds_dict = w.ds_dict()
            ds = w.ds()
            n_list = w.norm_list()
            R.<q> = PowerSeriesRing(QQ)
            L = [[g, n_list[i], O(q ** (prec - 1 - floor(n_list[i])))] for i, g in enumerate(ds)]
            lower_bounds = [None]*len(ds)
            for i in range(val, prec):
                h = f[i]
                h_coeffs = h.coefficients()
                for k, v in enumerate(h.exponents()):
                    if e == 1:
                        r = vector(S_inv[0, 0]*v)
                    else:
                        r = S_inv*vector(v)
                    r_frac = tuple(frac(r[j]) for j in range(e))
                    j = ds_dict[r_frac]
                    exponent = ceil(i - r * S * r/2)
                    if (lower_bounds[j] is None) or (exponent > lower_bounds[j]):
                        lower_bounds[j] = exponent
                        L[j][2] += h_coeffs[k] * q^(exponent)
            if correct:
                self.__theta = WeilRepModularForm(self.weight() - e/2, S, L, weilrep = self.weilrep())
                return self.__theta
            return WeilRepModularForm(self.weight() - e/2, S, L, weilrep = self.weilrep())

    def valuation(self):
        r"""
        Return self's valuation (with respect to the variable 'q').

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).valuation()
            0
        """
        try:
            return self.__valuation
        except:
            self.__valuation = self.fourier_expansion().valuation()
            return self.__valuation

    def weight(self):
        r"""
        Return self's weight.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).weight()
            4
        """
        return self.__weight

    def weilrep(self):
        r"""
        Return self's WeilRep instance.

        If self was created without a WeilRep instance then we create one here.

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).weilrep()
            Weil representation associated to the Gram matrix
            [2]
        """
        try:
            return self.__weilrep
        except AttributeError:
            self.__weilrep = WeilRep(self.index_matrix())
            return self.__weilrep

    ## arithmetic operations

    def __add__(self, other):
        r"""
        Addition of Jacobi forms. Undefined unless both have the same weight and index.
        """
        if not other:
            return self
        if not isinstance(other, JacobiForm):
            raise TypeError('Cannot add these objects')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        if not self.index_matrix() == other.index_matrix():
            raise ValueError('Incompatible indices')
        try:
            modform = self.modform() + other.modform()
        except AttributeError:
            modform = None
        return JacobiForm(self.weight(), self.index_matrix(), self.fourier_expansion() + other.fourier_expansion(), modform = modform, weilrep = self.weilrep())

    __radd__ = __add__

    def __sub__(self, other):
        r"""
        Subtraction of Jacobi forms. Undefined unless both have the same weight and index.
        """
        if not other:
            return self
        if not isinstance(other, JacobiForm):
            raise TypeError('Cannot subtract these objects')
        if not self.weight() == other.weight():
            raise ValueError('Incompatible weights')
        if not self.index_matrix() == other.index_matrix():
            raise ValueError('Incompatible indices')
        try:
            modform = self.modform() - other.modform()
        except AttributeError:
            modform = None
        return JacobiForm(self.weight(), self.index_matrix(), self.fourier_expansion() - other.fourier_expansion(), modform = modform, weilrep = self.weilrep())

    def __neg__(self):
        r"""
        Return the negative of self.
        """
        try:
            modform = -self.modform()
        except AttributeError:
            modform = None
        return JacobiForm(self.weight, self.index_matrix, -self.fourier_expansion, modform = modform, weilrep = self.weilrep())

    def __mul__(self, other):
        r"""
        Multiplication of Jacobi forms. Undefined unless both have index of the same rank.
        """
        if isinstance(other, JacobiForm):
            S1 = self.index_matrix()
            S2 = other.index_matrix()
            if not S1.nrows() == S2.nrows():
                raise ValueError('Incompatible indices')
            return JacobiForm(self.weight() + other.weight(), S1+S2, self.fourier_expansion() * other.fourier_expansion())
        elif is_ModularFormElement(other):
            try:
                modform = self.modform() * smf(other.weight(), other.qexp())
            except AttributeError:
                modform = None
            return JacobiForm(self.weight() + other.weight(), self.index_matrix(), self.qexp() * other.qexp(), modform = modform, weilrep = self.weilrep())
        elif other in QQ:
            try:
                modform = self.modform() * other
            except AttributeError:
                modform = None
            return JacobiForm(self.weight(), self.index_matrix(), self.fourier_expansion() * other, modform = modform, weilrep = self.weilrep())
        else:
            raise TypeError('Cannot multiply these objects')

    __rmul__ = __mul__

    def __truediv__(self, other):
        r"""
        Division of Jacobi forms. Undefined unless both have index of the same rank.
        """
        if isinstance(other, JacobiForm):
            S1 = self.index_matrix()
            S2 = other.index_matrix()
            if not S1.nrows() == S2.nrows():
                raise ValueError('Incompatible indices')
            return JacobiForm(self.weight() - other.weight(), S1-S2, self.fourier_expansion() / other.fourier_expansion())
        elif is_ModularFormElement(other):
            try:
                modform = self.modform() / smf(-other.weight(), ~other.qexp())
            except AttributeError:
                modform = None
            return JacobiForm(self.weight() - other.weight(), self.index_matrix(), self.fourier_expansion() / other, modform = modform)
        elif other in QQ:
            try:
                modform = self.modform() / other
            except AttributeError:
                modform = None
            return JacobiForm(self.weight(), self.index_matrix(), self.fourier_expansion() / other, modform = modform, weilrep = self.weilrep())
        else:
            raise TypeError('Cannot divide these objects')

    __div__ = __truediv__

    def __pow__(self, other): ## "tensor product" i.e. multiply together with separate abelian variables!
        r"""
        Outer product of Jacobi forms.

        If f(tau, z_1,...,z_m) and g(tau, z_1,...,z_n) are our Jacobi forms then this produces the Jacobi form

        f ** g(tau, z_1,...,z_(m+n)) = f(tau, z_1,...,z_m) * g(tau, z_(m+1), ..., z_(m+n)).

        EXAMPLES::

            sage: J4 = jacobi_eisenstein_series(4, 2, 5)
            sage: J6 = jacobi_eisenstein_series(6, 1, 5)
            sage: J4 ** J6
            1 + (14*w_0^2 + w_1^2 + 64*w_0 - 88*w_1 - 246 - 88*w_1^-1 + 64*w_0^-1 + w_1^-2 + 14*w_0^-2)*q + (w_0^4 + 14*w_0^2*w_1^2 + 64*w_0^3 - 1232*w_0^2*w_1 + 64*w_0*w_1^2 - 4340*w_0^2 - 5632*w_0*w_1 - 246*w_1^2 - 1232*w_0^2*w_1^-1 - 20672*w_0 - 11616*w_1 + 64*w_0^-1*w_1^2 + 14*w_0^2*w_1^-2 - 5632*w_0*w_1^-1 - 34670 - 5632*w_0^-1*w_1 + 14*w_0^-2*w_1^2 + 64*w_0*w_1^-2 - 11616*w_1^-1 - 20672*w_0^-1 - 1232*w_0^-2*w_1 - 246*w_1^-2 - 5632*w_0^-1*w_1^-1 - 4340*w_0^-2 + 64*w_0^-1*w_1^-2 - 1232*w_0^-2*w_1^-1 + 64*w_0^-3 + 14*w_0^-2*w_1^-2 + w_0^-4)*q^2 + (w_0^4*w_1^2 - 88*w_0^4*w_1 + 64*w_0^3*w_1^2 - 246*w_0^4 - 5632*w_0^3*w_1 - 4340*w_0^2*w_1^2 - 88*w_0^4*w_1^-1 - 20672*w_0^3 - 83776*w_0^2*w_1 - 20672*w_0*w_1^2 - 88*w_1^3 + w_0^4*w_1^-2 - 5632*w_0^3*w_1^-1 - 196896*w_0^2 - 309760*w_0*w_1 - 34670*w_1^2 + 64*w_0^3*w_1^-2 - 83776*w_0^2*w_1^-1 - 628032*w_0 - 435928*w_1 - 20672*w_0^-1*w_1^2 - 4340*w_0^2*w_1^-2 - 309760*w_0*w_1^-1 - 866700 - 309760*w_0^-1*w_1 - 4340*w_0^-2*w_1^2 - 20672*w_0*w_1^-2 - 435928*w_1^-1 - 628032*w_0^-1 - 83776*w_0^-2*w_1 + 64*w_0^-3*w_1^2 - 34670*w_1^-2 - 309760*w_0^-1*w_1^-1 - 196896*w_0^-2 - 5632*w_0^-3*w_1 + w_0^-4*w_1^2 - 88*w_1^-3 - 20672*w_0^-1*w_1^-2 - 83776*w_0^-2*w_1^-1 - 20672*w_0^-3 - 88*w_0^-4*w_1 - 4340*w_0^-2*w_1^-2 - 5632*w_0^-3*w_1^-1 - 246*w_0^-4 + 64*w_0^-3*w_1^-2 - 88*w_0^-4*w_1^-1 + w_0^-4*w_1^-2)*q^3 + (-246*w_0^4*w_1^2 + 64*w_0^5 - 11616*w_0^4*w_1 - 20672*w_0^3*w_1^2 - 1232*w_0^2*w_1^3 - 34670*w_0^4 - 309760*w_0^3*w_1 - 196896*w_0^2*w_1^2 - 5632*w_0*w_1^3 + w_1^4 - 11616*w_0^4*w_1^-1 - 628032*w_0^3 - 1685040*w_0^2*w_1 - 628032*w_0*w_1^2 - 11616*w_1^3 - 246*w_0^4*w_1^-2 - 309760*w_0^3*w_1^-1 - 3033280*w_0^2 - 3969024*w_0*w_1 - 866700*w_1^2 - 5632*w_0^-1*w_1^3 - 20672*w_0^3*w_1^-2 - 1685040*w_0^2*w_1^-1 - 6790912*w_0 - 5239264*w_1 - 628032*w_0^-1*w_1^2 - 1232*w_0^-2*w_1^3 - 196896*w_0^2*w_1^-2 - 3969024*w_0*w_1^-1 - 8820030 - 3969024*w_0^-1*w_1 - 196896*w_0^-2*w_1^2 - 1232*w_0^2*w_1^-3 - 628032*w_0*w_1^-2 - 5239264*w_1^-1 - 6790912*w_0^-1 - 1685040*w_0^-2*w_1 - 20672*w_0^-3*w_1^2 - 5632*w_0*w_1^-3 - 866700*w_1^-2 - 3969024*w_0^-1*w_1^-1 - 3033280*w_0^-2 - 309760*w_0^-3*w_1 - 246*w_0^-4*w_1^2 - 11616*w_1^-3 - 628032*w_0^-1*w_1^-2 - 1685040*w_0^-2*w_1^-1 - 628032*w_0^-3 - 11616*w_0^-4*w_1 + w_1^-4 - 5632*w_0^-1*w_1^-3 - 196896*w_0^-2*w_1^-2 - 309760*w_0^-3*w_1^-1 - 34670*w_0^-4 - 1232*w_0^-2*w_1^-3 - 20672*w_0^-3*w_1^-2 - 11616*w_0^-4*w_1^-1 + 64*w_0^-5 - 246*w_0^-4*w_1^-2)*q^4 + O(q^5)

        """
        if other in ZZ:
            return JacobiForm(self.weight() * other, self.index_matrix() * other, self.fourier_expansion()^other)
        elif not isinstance(other, JacobiForm):
            raise ValueError('Cannot multiply these objects')
        S1 = self.index_matrix()
        S2 = other.index_matrix()
        bigS = block_diagonal_matrix([S1,S2])
        Rb = LaurentPolynomialRing(QQ, list(var('w_%d' % i) for i in range(bigS.nrows()) ))
        R.<q> = PowerSeriesRing(Rb)
        e1 = S1.nrows()
        e2 = S2.nrows()
        f = R(other.fourier_expansion())
        val = other.valuation()
        jf = [Rb(f[i]).subs({Rb('w_%d'%j):Rb('w_%d'%(j+e1)) for j in range(e2)}) for i in range(f.valuation(),f.prec())]
        return JacobiForm(self.weight() + other.weight(), bigS, q^val * R(self.fourier_expansion())*R(jf) + O(q^(other.precision())))

    def __eq__(self, other):
        return self.qexp() == other.qexp()


    ## other operations

    def hecke_U(self, N):
        r"""
        Apply the Nth Hecke U-operator.

        NOTE: same as pullback(self, A) where A = N * identity_matrix

        INPUT:
        - ``N`` -- a natural number

        OUTPUT: JacobiForm of the same weight and of index N^2 * self.index()

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).hecke_U(3)
            1 + (w_0^-6 + 56*w_0^-3 + 126 + 56*w_0^3 + w_0^6)*q + (126*w_0^-6 + 576*w_0^-3 + 756 + 576*w_0^3 + 126*w_0^6)*q^2 + (56*w_0^-9 + 756*w_0^-6 + 1512*w_0^-3 + 2072 + 1512*w_0^3 + 756*w_0^6 + 56*w_0^9)*q^3 + (w_0^-12 + 576*w_0^-9 + 2072*w_0^-6 + 4032*w_0^-3 + 4158 + 4032*w_0^3 + 2072*w_0^6 + 576*w_0^9 + w_0^12)*q^4 + O(q^5)

        """
        S = self.index_matrix()
        e = S.nrows()
        Rb = self.base_ring()
        F = self.q_coefficients()
        R.<q> = PowerSeriesRing(Rb)
        f = []
        val = self.valuation()
        f = [g.subs({Rb('w_%d'%j):Rb('w_%d'%j)^N for j in range(e)}) for g in F]
        return JacobiForm(self.weight, N^2 * S, q^(val) * R(f) + O(q^self.precision()))

    def hecke_V(self, N):
        r"""
        Apply the Nth Hecke V-operator.

        NOTE: if self has precision 'prec' then self.hecke_V(N) has precision floor(prec / N)

        INPUT:
        - ``N`` -- a natural number

        OUTPUT: JacobiForm of the same weight and of index N * self.index()

        EXAMPLES::

            sage: jacobi_eisenstein_series(4, 1, 5).hecke_V(3)
            28 + (56*w_0^-3 + 756*w_0^-2 + 1512*w_0^-1 + 2072 + 1512*w_0 + 756*w_0^2 + 56*w_0^3)*q + O(q^2)

        """
        if N == 1:
            return self
        elif N < 1:
            raise NotImplementedError
        S = self.index_matrix()
        e = S.nrows()
        k_1 = self.weight() - 1
        Rb = self.base_ring()
        F = self.q_coefficients()
        R.<q> = PowerSeriesRing(Rb)
        val = min(self.valuation(), 0)
        max_prec = - (self.precision()// -N)
        f = O(q^(max_prec + val))
        for a in divisors(N):
            d = ZZ(N/a)
            sub_a = {Rb('w_%d'%j):Rb('w_%d'%j)^a for j in range(e)}
            f += sum([a^(k_1) * q^(a * (i + val)) * F[d*(i + val) - val].subs(sub_a) for i in range(max(0, ceil(val * (1/d - 1))), min(-((len(F) + (1-d) * val) // -d), max_prec))])
        return JacobiForm(k_1+1, N * S, f)
    
    def pullback(self, A):#return self evaluated at tau, A*z
        r"""
        Apply a linear map to self's elliptic variables.

        INPUT:
        - ``A`` -- a matrix in which the number of columns equals self's number of variables.

        OUTPUT: the JacobiForm f(tau, A*z) if self is the JacobiForm f(tau, z)

        EXAMPLES::

            sage: j = jacobi_eisenstein_series(4, matrix([[2,1],[1,4]]), 5)
            sage: j.pullback(matrix([[1, 2]]))
            1 + (7*w_0^-5 + 8*w_0^-4 + 21*w_0^-3 + 35*w_0^-2 + 28*w_0^-1 + 42 + 28*w_0 + 35*w_0^2 + 21*w_0^3 + 8*w_0^4 + 7*w_0^5)*q + (w_0^-9 + 7*w_0^-8 + 35*w_0^-7 + 56*w_0^-6 + 106*w_0^-5 + 147*w_0^-4 + 182*w_0^-3 + 182*w_0^-2 + 252*w_0^-1 + 224 + 252*w_0 + 182*w_0^2 + 182*w_0^3 + 147*w_0^4 + 106*w_0^5 + 56*w_0^6 + 35*w_0^7 + 7*w_0^8 + w_0^9)*q^2 + (21*w_0^-10 + 49*w_0^-9 + 126*w_0^-8 + 168*w_0^-7 + 294*w_0^-6 + 315*w_0^-5 + 462*w_0^-4 + 469*w_0^-3 + 609*w_0^-2 + 567*w_0^-1 + 560 + 567*w_0 + 609*w_0^2 + 469*w_0^3 + 462*w_0^4 + 315*w_0^5 + 294*w_0^6 + 168*w_0^7 + 126*w_0^8 + 49*w_0^9 + 21*w_0^10)*q^3 + (w_0^-13 + 21*w_0^-12 + 70*w_0^-11 + 154*w_0^-10 + 315*w_0^-9 + 400*w_0^-8 + 623*w_0^-7 + 756*w_0^-6 + 952*w_0^-5 + 966*w_0^-4 + 1281*w_0^-3 + 1162*w_0^-2 + 1366*w_0^-1 + 1386 + 1366*w_0 + 1162*w_0^2 + 1281*w_0^3 + 966*w_0^4 + 952*w_0^5 + 756*w_0^6 + 623*w_0^7 + 400*w_0^8 + 315*w_0^9 + 154*w_0^10 + 70*w_0^11 + 21*w_0^12 + w_0^13)*q^4 + O(q^5)

        """
        f = self.fourier_expansion()
        S = self.index_matrix()
        e = S.nrows()
        new_e = A.nrows()
        Rb = LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(e)))
        Rb_new = LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(new_e)))
        R.<q> = PowerSeriesRing(Rb_new)
        val = f.valuation()
        prec = f.prec()
        if new_e > 1:
            sub_R = {Rb('w_%d'%j):Rb_new.monomial(A.columns()[j]) for j in range(e)}
        else:
            sub_R = {Rb('w_%d'%j):Rb_new.0^A[0,j] for j in range(e)}
        jf_new = [f[i].subs(sub_R) for i in range(val, prec)]
        return JacobiForm(self.weight(), A*S*A.transpose(), q^val * R(jf_new) + O(q^f.prec()))

    def substitute_zero(self, indices):
        r"""
        Set some of our elliptic variables equal to zero.

        This evaluates the Jacobi form f(tau, z_0, ..., z_d) at z_i = 0 for some subset of indices i in {0,...,d}.

        NOTE: this is a special case of self.pullback(A) for a particular choice of A

        INPUT:
        - ``indices`` -- a list of integers between 0 and d = self.nvars()

        OUTPUT: JacobiForm

        EXAMPLES::

            sage: j = jacobi_eisenstein_series(4, matrix([[2, 1],[1, 4]]), 5)
            sage: j.substitute_zero([0])
            1 + (14*w_0^2 + 64*w_0 + 84 + 64*w_0^-1 + 14*w_0^-2)*q + (w_0^4 + 64*w_0^3 + 280*w_0^2 + 448*w_0 + 574 + 448*w_0^-1 + 280*w_0^-2 + 64*w_0^-3 + w_0^-4)*q^2 + (84*w_0^4 + 448*w_0^3 + 840*w_0^2 + 1344*w_0 + 1288 + 1344*w_0^-1 + 840*w_0^-2 + 448*w_0^-3 + 84*w_0^-4)*q^3 + (64*w_0^5 + 574*w_0^4 + 1344*w_0^3 + 2368*w_0^2 + 2688*w_0 + 3444 + 2688*w_0^-1 + 2368*w_0^-2 + 1344*w_0^-3 + 574*w_0^-4 + 64*w_0^-5)*q^4 + O(q^5)

        """
        f = self.fourier_expansion()
        S = self.index_matrix()
        e = S.nrows()
        Rb = LaurentPolynomialRing(QQ,list(var('w_%d' % i) for i in range(e)))
        R.<q> = PowerSeriesRing(Rb)
        val = f.valuation()
        prec = f.prec()
        L = [k for k in range(e) if k not in indices]
        d = {Rb('w_%d'%j):1 for j in indices}
        d.update({Rb('w_%d'%k):Rb('w_%d'%i) for i,k in enumerate(L)})
        jf_sub = R([f[i].subs(d) for i in range(val, prec)])
        return JacobiForm(self.weight(), S[L,L], q^val * R(jf_sub)+O(q^prec))

def jacobi_eisenstein_series(k, m, prec, allow_small_weight = False):
    r"""
    Compute the Jacobi Eisenstein series.
    """
    return JacobiForms(m).eisenstein_series(k, prec, allow_small_weight = allow_small_weight)

def theta_block(a, n, prec):#theta block corresponding to a=[a1,...,ar] and eta^n
    r"""
    Compute theta blocks.

    This takes a list of nonzero integers [a_1,...,a_r] and an integer n as input and produces the theta block

    theta(tau, a_1 z) * ... * theta(tau, a_r z) * eta(tau)^n

    if the result is a Jacobi form without character. (If the theta block has a character then a ValueError is raised instead.) The a_1,...,a_r do not need to be distinct.

    INPUT:
    - ``a`` -- a list [a_1,...,a_r] of nonzero integers (repeats are allowed)
    - ``n`` -- an integer
    - ``prec`` -- precision

    OUTPUT: JacobiForm
    """
    from collections import Counter
    try:
        qval = ZZ((3*len(a) + n) / 24) #extra exponent of q
        wval = ZZ(sum(a) / 2) #extra exponent of w
    except:
        raise ValueError('Nontrivial character')
    if any(not a for a in a):
        raise ValueError('List "a" contains zeros')
    prec -=qval
    weight = (len(a) + n) // 2
    a0 = Counter(a)
    a0_list = list(a0)
    Rb.<w_0> = LaurentPolynomialRing(QQ)
    R.<q> = PowerSeriesRing(Rb,prec)
    thetas = [O(q^prec)]*len(a0_list) #compute theta functions. we don't generally spend much time on this part so I won't bother caching them
    eta = O(q^prec)
    bound = isqrt(prec + prec + 1/4) + 1
    eps = 1 - 2 * (bound % 2)
    for i in range(-bound, bound):
        eps = -eps
        for j, _a in enumerate(a0_list):
            thetas[j] += eps * w_0^(_a*i) * q^binomial(i+1,2)
        j = i * (3 * i + 1) / 2
        if j < bound:
            eta += eps * q^ZZ(j)
    return JacobiForm(weight, matrix([[sum([a^2 for a in a])]]), prod([theta ** a0[a0_list[i]] for i,theta in enumerate(thetas)]) * (eta ** n) * q^qval * w_0^wval)