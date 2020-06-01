r"""

Sage code for spaces of vector-valued modular forms

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

from bisect import bisect
from itertools import product
load('eisenstein_series.sage')
load('jacobi_forms_class.sage')
load('weilrep_misc.sage')
load('weilrep_modular_forms_class.sage')


class WeilRep(object):
    r"""
    The WeilRep class represents the module of vector-valued modular forms which transform with the dual Weil representation.

    INPUT:

    A WeilRep instance is constructed by calling WeilRep(S), where
    - ``S`` -- a symmetric integral matrix with even diagonal and nonzero determinant (this is not checked), OR
    - ``S`` -- a nondegenerate quadratic form

    OUTPUT: WeilRep
    """

    def __init__(self, S):
        if isinstance(S.parent(), MatrixSpace):
            self.__gram_matrix = S
            self.__quadratic_form = QuadraticForm(S)
        elif isinstance(S, QuadraticForm):
            self.__quadratic_form = S
            self.__gram_matrix = S.matrix()
        else:
            raise TypeError('Invalid input')

    def __repr__(self):
        #when printed:
        return 'Weil representation associated to the Gram matrix\n%s' % (self.gram_matrix())

    ## basic attributes

    def gram_matrix(self):
        return self.__gram_matrix

    def quadratic_form(self):
        return self.__quadratic_form

    def signature(self):
        r"""
        Return the signature of the underlying quadratic form.
        """
        try:
            return self.__signature
        except AttributeError:
            self.__signature = self.quadratic_form().signature() % 8
            return self.__signature

    def discriminant(self):
        r"""
        Return the discriminant (without sign) of the underlying quadratic form.
        """
        try:
            return self.__discriminant
        except AttributeError:
            try:
                self.__discriminant = ZZ(len(self.__ds))
                return self.__discriminant
            except AttributeError:
                self.__discriminant = abs(self.gram_matrix().determinant())
                return self.__discriminant

    def is_symmetric_weight(self, weight):
        r"""
        Computes whether the given weight is symmetric.

        INPUT:
        - ``weight`` -- a half-integer

        OUTPUT: 1, if all modular forms of this weight are symmetric. 0, if all modular forms of this weight are antisymmetric. None, otherwise.

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,4]])).is_symmetric_weight(4)
            0

        """
        return [1,None,0,None][(ZZ(2*weight) + self.signature()) % 4]

    ## methods for the discriminant form

    def coefficient_vector_exponents(self, prec, symm, starting_from = 0, include_vectors = False):
        r"""
        Interpret the ``coefficient vectors`` of modular forms for this Weil representation.

        INPUT:
        - ``prec`` -- the precision
        - ``symm`` -- 0 if we work with anti-symmetric modular forms and 1 otherwise
        - ``starting_from`` -- valuation of the modular form (default 0)
        - ``include_vectors`` -- a boolean (default False)

        OUTPUT: if include_vectors = False then this outputs a list of exponents [n_0, n_1, ...] such that if a modular form of this representation has ``coefficient vector`` (c_0, c_1, ...) (see weilrep_modular_forms_class.sage) then its Fourier expansion takes the form c_0 q^(n_0) e_(g_0) + c_1 q^(n_1) e_(g_1) + ... If include_vectors = True then we output the list of lists [[g_0,n_0], [g_1,n_1]...] instead.

        NOTE: we only take one representative from each pair {g, -g}. If symm = 0 then we also exclude g for which g = -g.

        NOTE: if include_vectors = True then the vectors are given as tuples!

        EXAMPLES::

            sage:WeilRep(matrix([[2,1],[1,2]])).coefficient_vector_exponents(3,1)
            [0, 2/3, 1, 5/3, 2, 8/3]

            sage:WeilRep(matrix([[2,1],[1,2]])).coefficient_vector_exponents(3,0)
            [2/3, 5/3, 8/3]

            sage:WeilRep(matrix([[2,1],[1,4]])).coefficient_vector_exponents(3,1,include_vectors = True)
            [[(0, 0), 0], [(5/7, 4/7), 3/7], [(4/7, 6/7), 5/7], [(1/7, 5/7), 6/7], [(0, 0), 1], [(5/7, 4/7), 10/7], [(4/7, 6/7), 12/7], [(1/7, 5/7), 13/7], [(0, 0), 2], [(5/7, 4/7), 17/7],   [(4/7, 6/7), 19/7], [(1/7, 5/7), 20/7]]

        """
        if starting_from == 0:
            try:
                return [self.__coefficient_vector_exponents[symm], self.__coefficient_vector_exponents_including_vectors[symm]][include_vectors]
            except AttributeError:
                pass
        G = self.sorted_rds()
        n_dict = self.norm_dict()
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        for n in range(floor(starting_from),prec+1):
            for g in G:
                true_n = n_dict[g] + n
                if starting_from <= true_n < prec:
                    X1.append(true_n)
                    Y1.append([g,true_n])
                    if 2 % vector(g).denominator():
                        X2.append(true_n)
                        Y2.append([g,true_n])
        if starting_from == 0:
            self.__coefficient_vector_exponents = [X2,X1]
            self.__coefficient_vector_exponents_including_vectors = [Y2,Y1]
        return [[X2,X1][symm], [Y2,Y1][symm]][include_vectors]

    def ds(self):
        r"""
        Compute representatives of the discriminant group of the underlying quadratic form.

        OUTPUT: a list of vectors which represent the cosets of S^(-1)*Z^N modulo Z^N, where S is the Gram matrix.

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).ds()
            [(0, 0), (2/3, 2/3), (1/3, 1/3)]
            """
        try:
            return self.__ds
        except AttributeError:
            if not self.gram_matrix():
                self.__ds = [vector([])]
            else:
                D, U, V = self.gram_matrix().smith_form()
                L = [vector(range(D[k, k])) / D[k, k] for k in range(D.nrows())]
                self.__ds = [vector(frac(x) for x in V * vector(r)) for r in product(*L)]
            return self.__ds

    def ds_dict(self):
        r"""
        Compute the discriminant group of the underlying quadratic form as a dictionary.

        OUTPUT: a dictionary whose keys are the elements of the discriminant group (as tuples) and whose values are their index in self.ds()

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,4]])).ds_dict() == {(2/7, 3/7): 4, (5/7, 4/7): 3, (0, 0): 0, (1/7, 5/7): 2, (3/7, 1/7): 6, (4/7, 6/7): 1, (6/7, 2/7): 5}
            True

        """
        try:
            return self.__ds_dict
        except:
            _ds = [tuple(g) for g in self.ds()]
            self.__ds_dict = dict(zip(_ds, range(len(_ds))))
            return self.__ds_dict

    def dual(self):
        r"""
        Compute the dual representation.

        This is simply the Weil representation obtained by multiplying the underlying quadratic form by (-1).

        OUTPUT: a WeilRep instance

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).dual()
            Weil representation associated to the Gram matrix
            [-2 -1]
            [-1 -2]

        """
        return weilrep(-self.gram_matrix())

    def embiggen(self, b, m):
        S = self.gram_matrix()
        tilde_b = b*S
        shift_m = m + b*tilde_b/2
        tilde_b = matrix(tilde_b)
        return WeilRep(block_matrix(ZZ,[[S,tilde_b.transpose()],[tilde_b,2*shift_m]]))

    def level(self):
        try:
            return self.__level
        except:
            self.__level = self.quadratic_form().level()
            return self.__level

    def norm_dict(self):
        r"""
        Compute the values of the quadratic form Q on the discriminant group as a dictionary.

        OUTPUT: a dictionary whose keys are the elements of the discriminant group (as tuples) and whose values are the *minus* norms -Q(x) in QQ/ZZ (represented by a rational number: -1 < -Q(x) <= 0)

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).norm_dict() == {(2/3, 2/3): -1/3, (0, 0): 0, (1/3, 1/3): -1/3}
            True

        """
        try:
            return self.__norm_dict
        except:
            _ds = self.ds()
            S = self.gram_matrix()
            self.__norm_dict = {tuple(g):-frac(g*S*g/2) for g in _ds}
            return self.__norm_dict

    def norm_list(self):
        r"""
        Compute the values of the quadratic form Q on the discriminant group as a list.

        OUTPUT: a list whose values are the *minus* norms -Q(x) in QQ/ZZ (represented by a rational number: -1 < -Q(x) <= 0) where x runs through the result of self.ds()

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).norm_list()
            [0, -1/3, -1/3]

        """
        try:
            return self.__norm_list
        except:
            _ds = self.ds()
            S = self.gram_matrix()
            self.__norm_list = [-frac(g*S*g/2) for g in _ds]
            self.__norm_dict = dict(zip([tuple(g) for g in _ds], self.__norm_list)) #create a norm_dict() while we're here
            return self.__norm_list

    def rds(self, indices = False):
        r"""
        Reduce the representatives of the discriminant group modulo equivalence g ~ -g

        OUTPUT:
        - If indices = False then output a sublist of discriminant_group(S) containing exactly one element from each pair {g,-g}. 
        - If indices = True then output a list X of indices defined as follows. Let ds = self.ds(). Then: X[j] = i if j > i and ds[j] = -ds[i] mod Z^N, and X[j] = None if no such i exists.

        NOTE: as we run through the discriminant group we also store the denominators of the vectors as a list

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).rds()
            [(0, 0), (2/3, 2/3)]

            sage: WeilRep(matrix([[2,1],[1,2]])).rds(indices = True)
            [None, None, 1]

        """
        try:
            return [self.__rds, self.__rds_indices][indices]
        except AttributeError:
            if not self.gram_matrix():
                self.__rds = [vector([])]
                self.__rds_indices = [None]
                self.__ds_denominators_list = [1]
                self.__rds_denominators_list = [1]
                self.__order_two_in_ds_list = [1]
                self.__order_two_in_rds_list = [1]
            else:
                L = []
                G = self.ds()
                X = [None]*len(G)
                X2 = [None]*len(G)
                Y = [0]*len(G)
                Z = []
                order_two_ds = [0]*len(G)
                order_two_rds = []
                for i, g in enumerate(G):
                    u = vector(frac(-x) for x in g)
                    dg = denominator(g)
                    Y[i] = dg
                    order_two_ds[i] = (2 % dg == 0)
                    if u in L:
                        X[i] = G.index(u)
                    else:
                        L.append(g)
                        Z.append(dg)
                        order_two_rds.append(order_two_ds[i])
                self.__rds = L
                self.__rds_indices = X
                self.__ds_denominators_list = Y
                self.__rds_denominators_list = Z
                self.__order_two_in_ds_list = order_two_ds
                self.__order_two_in_rds_list = order_two_rds
            return [self.__rds, self.__rds_indices][indices]

    def sorted_rds(self):
        r"""
        Computes a copy of the reduced discriminant group self.rds(), sorted by the norm dictionary norm_dict().

        OUTPUT: a list of *tuples*

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,4]])).sorted_rds()
            [(5/7, 4/7), (4/7, 6/7), (1/7, 5/7), (0, 0)]
        """
        S = self.gram_matrix()
        try:
            return self.__sorted_rds
        except AttributeError:
            self.__rds = self.rds()
            n_dict = self.norm_dict()
            G = [tuple(g) for g in self.__rds]
            G.sort(key = lambda g: n_dict[g])
            self.__sorted_rds = G
            return G


    ## constructors of modular forms for this representation. See also weilrep_modular_forms_class.sage
    
    def bb_lift(self, mf):
        r"""
        Construct vector-valued modular forms of prime level via the Bruinier--Bundschuh lift.

        NOTE: this works *only* when self has odd prime discriminant. (In particular the lattice rank must be even.)

        INPUT:
        - ``mf`` -- a modular form of level equal to self's discriminant, with the quadratic character and lying in the correct plus/minus space

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[2,1],[1,2]]))
            sage: mf = ModularForms(Gamma1(3), 3, prec = 20).basis()[0]
            sage: w.bb_lift(mf)
            [(0, 0), 1 + 72*q + 270*q^2 + 720*q^3 + 936*q^4 + 2160*q^5 + O(q^6)]
            [(2/3, 2/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + O(q^6)]
            [(1/3, 1/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + O(q^6)]

        """
        if not is_ModularFormElement(mf):
            raise TypeError('The Bruinier-Bundschuh lift takes modular forms as input')
        p = mf.level()
        if not p.is_prime() and p != 2 and self.discriminant() == p:
            raise TypeError('The Bruinier-Bundschuh lift takes modular forms of odd prime level as input')
        if not mf.character() == DirichletGroup(p)[ZZ((p-1)/2)]:
            raise TypeError('Invalid character')
        mfq = mf.qexp()
        R, q = mfq.parent().objgen()
        mf_coeffs = mfq.padded_list()
        prec = len(mf_coeffs)//p
        ds = self.ds()
        norm_list = self.norm_list()
        Y = [None]*len(ds)
        for i, g in enumerate(ds):
            offset = norm_list[i]
            if not g:
                f = R(mf_coeffs[::p]) + O(q ** prec)
                zero = not f
                Y[i] = g, offset, f
            else:
                u = q*R(mf_coeffs[p+ZZ(p*offset)::p])/2 + O(q^(prec+1))
                if not (u or zero):#are we checking this too often?? how else do we tell whether a modform lies in the +/- subspace??
                    raise ValueError('This modular form does not lie in the correct plus/minus subspace')
                Y[i] = g, offset, u
        return WeilRepModularForm(mf.weight(), self.gram_matrix(), Y, weilrep = self)

    def eisenstein_series(self, k, prec, allow_small_weight = False, components = None):
        r"""
        Constuct Eisenstein series attached to the vector e_0.

        This constructs the Eisenstein series E_(k,0) of weight k and constant term e_0 with Fourier expansion up to precision `prec`.

        INPUT:
        - ``k`` -- a weight (half-integer, and such that 2k + signature = 0 mod 4). also ``k`` can be a list of weights (then we produce a list of Eisenstein series).
        - ``prec`` -- precision
        - ``allow_small_weight`` -- a boolean (default False). If True then we compute the Eisenstein series in weights less than or equal to 2 (where it may not be a true modular form)
        - ``components`` -- optional parameter (default None). A sublist L of self's discriminant group and a list of indices (e.g. [None]*len(L) ) can be passed here as a tuple.

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: WeilRep(matrix([[2,1],[1,2]])).eisenstein_series(3,20)
            [(0, 0), 1 + 72*q + 270*q^2 + 720*q^3 + 936*q^4 + 2160*q^5 + 2214*q^6 + 3600*q^7 + 4590*q^8 + 6552*q^9 + 5184*q^10 + 10800*q^11 + 9360*q^12 + 12240*q^13 + 13500*q^14 + 17712*q^15 + 14760*q^16 + 25920*q^17 + 19710*q^18 + 26064*q^19 + O(q^20)]
            [(2/3,  2/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + 2808*q^(20/3) + 4752*q^(23/3) + 4590*q^(26/3) + 7560*q^(29/3) + 7371*q^(32/3) + 10800*q^(35/3) + 9774*q^(38/3) + 15120*q^(41/3) + 14040*q^(44/3) + 19872*q^(47/3) + 16227*q^(50/3) + 25272*q^(53/3) + 22950*q^(56/3) + 31320*q^(59/3) + O(q^20)]
            [(1/3, 1/3), 27*q^(2/3) + 216*q^(5/3) + 459*q^(8/3) + 1080*q^(11/3) + 1350*q^(14/3) + 2592*q^(17/3) + 2808*q^(20/3) + 4752*q^(23/3) + 4590*q^(26/3) + 7560*q^(29/3) + 7371*q^(32/3) + 10800*q^(35/3) + 9774*q^(38/3) + 15120*q^(41/3) + 14040*q^(44/3) + 19872*q^(47/3) + 16227*q^(50/3) + 25272*q^(53/3) + 22950*q^(56/3) + 31320*q^(59/3) + O(q^20)]

            sage: WeilRep(matrix([[2,0],[0,6]])).eisenstein_series(5,5)
            [(0, 0), 1 - 1280/11*q - 20910/11*q^2 - 104960/11*q^3 - 329040/11*q^4 + O(q^5)]
            [(0, 1/6), -915/11*q^(11/12) - 1590*q^(23/12) - 93678/11*q^(35/12) - 304980/11*q^(47/12) - 757335/11*q^(59/12) + O(q^5)]
            [(0, 1/3), -255/11*q^(2/3) - 9984/11*q^(5/3) - 65775/11*q^(8/3) - 234240/11*q^(11/3) - 612510/11*q^(14/3) + O(q^5)]
            [(0, 1/2), -5/11*q^(1/4) - 3198/11*q^(5/4) - 33215/11*q^(9/4) - 142810/11*q^(13/4) - 428040/11*q^(17/4) + O(q^5)]
            [(0, 2/3), -255/11*q^(2/3) - 9984/11*q^(5/3) - 65775/11*q^(8/3) - 234240/11*q^(11/3) - 612510/11*q^(14/3) + O(q^5)]
            [(0, 5/6), -915/11*q^(11/12) - 1590*q^(23/12) - 93678/11*q^(35/12) - 304980/11*q^(47/12) - 757335/11*q^(59/12) + O(q^5)]
            [(1/2, 0), -410/11*q^(3/4) - 12010/11*q^(7/4) - 75030/11*q^(11/4) - 255918/11*q^(15/4) - 651610/11*q^(19/4) + O(q^5)]
            [(1/2, 1/6), -240/11*q^(2/3) - 10608/11*q^(5/3) - 61440/11*q^(8/3) - 248880/11*q^(11/3) - 576480/11*q^(14/3) + O(q^5)]
            [(1/2, 1/3), -39/11*q^(5/12) - 5220/11*q^(17/12) - 44205/11*q^(29/12) - 176610/11*q^(41/12) - 493155/11*q^(53/12) + O(q^5)]
            [(1/2, 1/2), -1360/11*q - 19680/11*q^2 - 111520/11*q^3 - 307200/11*q^4 + O(q^5)]
            [(1/2, 2/3), -39/11*q^(5/12) - 5220/11*q^(17/12) - 44205/11*q^(29/12) - 176610/11*q^(41/12) - 493155/11*q^(53/12) + O(q^5)]
            [(1/2, 5/6), -240/11*q^(2/3) - 10608/11*q^(5/3) - 61440/11*q^(8/3) - 248880/11*q^(11/3) - 576480/11*q^(14/3) + O(q^5)]

            sage: WeilRep(matrix([[0,0,2],[0,-4,0],[2,0,0]])).eisenstein_series(5/2,5)
            [(0, 0, 0), 1 - 8*q - 102*q^2 - 48*q^3 - 184*q^4 + O(q^5)]
            [(0, 3/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^5)]
            [(0, 1/2, 0), -14*q^(1/2) - 16*q^(3/2) - 80*q^(5/2) - 64*q^(7/2) - 350*q^(9/2) + O(q^5)]
            [(0, 1/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^5)]
            [(0, 0, 1/2), -16*q - 64*q^2 - 96*q^3 - 128*q^4 + O(q^5)]
            [(0, 3/4, 1/2), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^5)]
            [(0, 1/2, 1/2), -8*q^(1/2) - 32*q^(3/2) - 64*q^(5/2) - 128*q^(7/2) - 200*q^(9/2) + O(q^5)]
            [(0, 1/4, 1/2), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^5)]
            [(1/2, 0, 0), -16*q - 64*q^2 - 96*q^3 - 128*q^4 + O(q^5)]
            [(1/2, 3/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^5)]
            [(1/2, 1/2, 0), -8*q^(1/2) - 32*q^(3/2) - 64*q^(5/2) - 128*q^(7/2) - 200*q^(9/2) + O(q^5)]
            [(1/2, 1/4, 0), -q^(1/8) - 25*q^(9/8) - 48*q^(17/8) - 121*q^(25/8) - 144*q^(33/8) + O(q^5)]
            [(1/2, 0, 1/2), -8*q^(1/2) - 32*q^(3/2) - 64*q^(5/2) - 128*q^(7/2) - 200*q^(9/2) + O(q^5)]
            [(1/2, 3/4, 1/2), -8*q^(5/8) - 40*q^(13/8) - 80*q^(21/8) - 120*q^(29/8) - 200*q^(37/8) + O(q^5)]
            [(1/2, 1/2, 1/2), -16*q - 64*q^2 - 96*q^3 - 128*q^4 + O(q^5)]
            [(1/2, 1/4, 1/2), -8*q^(5/8) - 40*q^(13/8) - 80*q^(21/8) - 120*q^(29/8) - 200*q^(37/8) + O(q^5)]

            sage: WeilRep(matrix([])).eisenstein_series(2, 10, allow_small_weight = True)
            [(), 1 - 24*q - 72*q^2 - 96*q^3 - 168*q^4 - 144*q^5 - 288*q^6 - 192*q^7 - 360*q^8 - 312*q^9 + O(q^10)]

        """
        #check input
        prec = ceil(prec)
        k_is_list = type(k) is list
        if not k_is_list:
            if prec <= 0:
                raise ValueError('Precision must be at least 0')
            if k <= 2 and not allow_small_weight and (k < 2 or self.discriminant().is_squarefree()):
                raise ValueError('Weight must be at least 5/2')
            if not self.is_symmetric_weight(k):
                raise ValueError('Invalid weight')
            try:#did we do this already?
                old_prec, e = self.__eisenstein
                if old_prec >= prec:
                    return e.reduce_precision(old_prec, in_place = False)
                assert False
            except:
                pass
        #setup
        R.<q> = PowerSeriesRing(QQ)
        S = self.gram_matrix()
        if components:
            _ds, _indices = components
            _norm_list = [-frac(g*S*g/2) for g in _ds]
        else:
            _ds = self.ds()
            _indices = self.rds(indices = True)
            _norm_list = self.norm_list()
        dets = self.discriminant()
        eps = (-1) ** (self.signature() in range(3, 7))
        S_rows_gcds = [GCD(x) for x in S.rows()]
        S_rows_sums = sum(S)
        level = self.level()
        L_half_s = (matrix(_ds) * S).rows() #fix?
        _dim = S.nrows()
        precomputed_lists = {}
        dets_primes = dets.prime_factors()
        X = [None] * len(_ds)
        if k_is_list:
            len_k = len(k)
            X = [copy(X) for _ in range(len_k)]
        #guess which Lvalues we have to look at. (this is always enough but sometimes its too many)
        def eisenstein_series_create_lists(g):
            d_gamma = denominator(g)
            d_gamma_squared = d_gamma * d_gamma
            old_modulus = 2 * d_gamma_squared * dets
            mod_value = old_modulus * _norm
            gcd_mm = GCD(old_modulus, mod_value)
            modulus = ZZ(old_modulus / gcd_mm)
            mod_value = mod_value / gcd_mm
            m = mod_value + modulus * modified_prec
            prime_list_1 = prime_range(2, m)
            prime_list_1.extend([p for p in dets_primes if p >= m])
            little_n_list = [mod_value / modulus + j for j in range(1, modified_prec)]
            prime_list = []
            n_lists = []
            removable_primes = []
            for p in prime_list_1:
                if level % p == 0:
                    if p != 2 and any(L_half[i]%p and not S_rows_gcds[i]%p for i in range(_dim)):# and (p != 2 or any(L_half[i]%2 and not S[i,i]%4 for i in range(_dim))):#???
                        removable_primes.append(p)
                    else:
                        prime_list.append(p)
                        n_lists.append([little_n_list,list(range(modified_prec - 1)),[],[]])
                else:
                    p_sqr = p * p
                    index_list_p = []
                    index_list_p_ii = []
                    n_list_p = []
                    n_list_p_ii = []
                    mod_value_mod_p = (modulus.inverse_mod(p_sqr) * mod_value) % p_sqr
                    i0 = p_sqr - mod_value_mod_p
                    i = p - (mod_value_mod_p % p)
                    while 1:#find exponents divisible by p
                        try:
                            N = little_n_list[i-1]
                            if (i-i0) % p_sqr:
                                n_list_p_ii.append(N)
                                index_list_p_ii.append(i-1)
                            else:
                                n_list_p.append(N)
                                index_list_p.append(i-1)
                            i += p
                        except:
                            break
                    if n_list_p or n_list_p_ii:
                        prime_list.append(p)
                        n_lists.append([n_list_p, index_list_p, n_list_p_ii, index_list_p_ii])
            return [little_n_list, n_lists, old_modulus, prime_list, removable_primes]
        #odd rank:
        if _dim % 2:
            if k_is_list:
                k_shift_list = [ZZ(j - 1/2) for j in k]
                two_k_shift_list = [j + j for j in k_shift_list]
                front_multiplier_list = [eps / zeta(1 - j) for j in two_k_shift_list]
                k_shift = k_shift_list[0]
            else:
                k_shift = ZZ(k - 1/2)
                two_k_shift = k_shift + k_shift
                front_multiplier = eps / zeta(1 - two_k_shift)
            for i_g, g in enumerate(_ds):
                _norm = _norm_list[i_g]
                if _indices[i_g] is None: #have we computed the negative component yet?
                    L_half = L_half_s[i_g]
                    L = L_half + L_half
                    modified_prec = prec - floor(_norm)
                    little_n_list, n_lists, old_modulus, prime_list, removable_primes = eisenstein_series_create_lists(g)
                    gSg = g * S * g
                    if k_is_list:
                        RPoly.<t> = PolynomialRing(QQ)
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, 0, t=t) for i_p, p in enumerate(prime_list)]
                    else:
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, k) for i_p, p in enumerate(prime_list)]
                    try:#some things depend only on the exponents not the component-vector "g"
                        D_list, main_term_list = precomputed_lists[_norm]
                    except:
                        if k_is_list:
                            main_term_list = [[0] for _ in range(len_k)]
                            D_list = [0]
                            for i_n, n in enumerate(little_n_list):
                                D = ((-1) ** k_shift) * old_modulus
                                for p, e in factor(n):
                                    if p==2 or (dets % p == 0):
                                        D *= (p ** e)
                                    else:
                                        D *= (p ** (e % 2))
                                little_D = abs(fundamental_discriminant(D))
                                sqrt_factor = sqrt(2 * n  * dets / little_D)
                                correct_L_function_list = [quadratic_L_function__corrector(k_shift, D) * quadratic_L_function__cached(1 - k_shift, D) for k_shift in k_shift_list]
                                D_list.append(D)
                                for j, k_shift in enumerate(k_shift_list):
                                    main_term_list[j].append(correct_L_function_list[j] * ((4 * n / little_D) ** k_shift) / sqrt_factor)
                        else:
                            main_term_list = [0]
                            D_list = [0]
                            for i_n, n in enumerate(little_n_list):
                                D = ((-1) ** k_shift) * old_modulus
                                for p, e in factor(n):
                                    if p==2 or (dets % p == 0):
                                        D *= (p ** e)
                                    else:
                                        D *= (p ** (e % 2))
                                little_D = abs(fundamental_discriminant(D))
                                sqrt_factor = sqrt(2 * n  * dets / little_D)
                                correct_L_function = quadratic_L_function__corrector(k_shift, D) * quadratic_L_function__cached(1 - k_shift, D)
                                D_list.append(D)
                                main_term_list.append(correct_L_function * ((4 * n / little_D) ** k_shift) / sqrt_factor)
                    local_factor_list = [1] * (modified_prec)
                    if k_is_list:
                        local_factor_list = [copy(local_factor_list) for _ in range(len_k)]
                        for i, p in enumerate(prime_list):
                            p = prime_list[i]
                            p_e_power = p ** ((1 + _dim) // 2)
                            for index_k, k_shift in enumerate(k_shift_list):
                                p_k_shift = p ** (-k_shift)
                                p_k_shift_squared = p_k_shift * p_k_shift
                                p1_mult = 1 / (1 + p_k_shift)
                                p2_mult = (1 + p_k_shift) / (1 - p_k_shift_squared)
                                p_pow = p_k_shift * p_e_power
                                p_tuple = (p2_mult, ~(1 - p_k_shift_squared), p1_mult)
                                for j in range(len(n_lists[i][0])):
                                    index_n = n_lists[i][1][j] + 1
                                    D = D_list[index_n]
                                    local_factor_list[index_k][index_n] *= Lvalue_list[i][j](p_pow) * p_tuple[kronecker_symbol(D, p) + 1]
                    else:
                        for i, p in enumerate(prime_list):
                            p = prime_list[i]
                            p_k_shift = p ** (-k_shift)
                            p_k_shift_squared = p_k_shift * p_k_shift
                            p1_mult = 1 / (1 + p_k_shift)
                            p2_mult = (1 + p_k_shift) / (1 - p_k_shift_squared)
                            p_tuple = (p2_mult, ~(1 - p_k_shift_squared), p1_mult)
                            for j in range(len(n_lists[i][0])):
                                index_n = n_lists[i][1][j] + 1
                                D = D_list[index_n]
                                local_factor_list[index_n] *= Lvalue_list[i][j] * p_tuple[kronecker_symbol(D, p) + 1]
                    E = (old_modulus == dets + dets) + O(q ** modified_prec)
                    if k_is_list:
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, E + front_multiplier_list[i] * (prod(1 / (1 - p ** (-two_k_shift_list[i])) for p in removable_primes)) * R([local_factor_list[i][j] * main_term_list[i][j] for j in range(modified_prec)])
                    else:
                        if (k == 1) and gSg == 0:
                            E = E - weight_one_zero_val(g,S)
                        X[i_g] = g, _norm, E + front_multiplier * (prod(1 / (1 - p ** (-two_k_shift)) for p in removable_primes)) * R([local_factor_list[j] * main_term_list[j] for j in range(modified_prec)])
                    precomputed_lists[_norm] = D_list, main_term_list
                else:
                    if k_is_list:
                        ind_g = _indices[i_g]
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, X[i][ind_g][2]
                    else:
                        X[i_g] = g, _norm, X[_indices[i_g]][2]
            if k_is_list:
                return [WeilRepModularForm(k[i], S, X[i], weilrep = self) for i in range(len_k)]
            else:
                e = WeilRepModularForm(k, S, X, weilrep = self)
                self.__eisenstein = prec, e
                return e
        #even rank
        else:
            if k_is_list:
                D = ((-1) ** k[0]) * dets
                littleD = fundamental_discriminant(D)
                sqrt_factor = QQ(2 / isqrt(abs(littleD * dets)))
                multiplier_list = [eps * (littleD ** k_i) * sqrt_factor / (quadratic_L_function__corrector(k_i, D) * quadratic_L_function__cached(1 - k_i, littleD)) for i, k_i in enumerate(k)]
            else:
                D = ((-1) ** k) * dets
                littleD = fundamental_discriminant(D)
                corrector = 1 / quadratic_L_function__corrector(k, D)
                sqrt_factor = QQ(2/isqrt(abs(littleD * dets)))
                multiplier = QQ(eps * corrector * (littleD ** k) * sqrt_factor / quadratic_L_function__cached(1 - k, littleD))
            for i_g, g in enumerate(_ds):
                _norm = _norm_list[i_g]
                modified_prec = prec - floor(_norm)
                if _indices[i_g] is None: #have we computed the negative component yet?
                    L_half = L_half_s[i_g]
                    L = L_half + L_half
                    _norm = _norm_list[i_g]
                    little_n_list, n_lists, old_modulus, prime_list, removable_primes = eisenstein_series_create_lists(g)
                    gSg = g * S * g
                    if k_is_list:
                        RPoly.<t> = PolynomialRing(QQ)
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, 0, t=t) for i_p, p in enumerate(prime_list)]
                    else:
                        Lvalue_list = [L_values(L, [2*n + gSg for n in n_lists[i_p][0]], S, p, k) for i_p, p in enumerate(prime_list)]
                    try:
                        main_term_list = precomputed_lists[_norm]#some things depend only on the exponents not the component-vector "g"
                    except:
                        if k_is_list:
                            main_term_list = [[0] for _ in range(len_k)]
                            for i, k_i in enumerate(k):
                                main_term_list[i].extend([n ** (k_i - 1) for n in little_n_list])
                        else:
                            main_term_list = [0]
                            main_term_list.extend([n ** (k-1) for n in little_n_list])
                    local_factor_list = [1] * (modified_prec)
                    if k_is_list:
                        local_factor_list = [copy(local_factor_list) for _ in range(len_k)]
                        for i, p in enumerate(prime_list):
                            kron = kronecker_symbol(D, p)
                            p_pow_e = p ** (1 + _dim//2)
                            for index_k, k_i in enumerate(k):
                                p_k = p ** (-k_i)
                                kron_p = kron * p_k
                                p_pow = p_pow_e * p_k
                                p_factor = 1 + (p * kron_p)
                                quot = 1 / (1 - kron_p)
                                for j in range(len(n_lists[i][0])):
                                    index_n = n_lists[i][1][j] + 1
                                    local_factor_list[index_k][index_n] *= (Lvalue_list[i][j](p_pow) * quot)
                                for j in range(len(n_lists[i][2])):
                                    index_n = n_lists[i][3][j] + 1
                                    local_factor_list[index_k][index_n] *= p_factor
                    else:
                        local_factor_list = [1] * (modified_prec)
                        for i, p in enumerate(prime_list):
                            kron_p = kronecker_symbol(D, p) * (p ** (-k))
                            p_factor = 1 + (p * kron_p)
                            quot = 1 / (1 - kron_p)
                            for j in range(len(n_lists[i][0])):
                                index_n = n_lists[i][1][j] + 1
                                local_factor_list[index_n] *= (Lvalue_list[i][j] * quot)
                            for j in range(len(n_lists[i][2])):#p is bad but not too bad at N=local_factor_list[index_n]
                                index_n = n_lists[i][3][j] + 1
                                local_factor_list[index_n] *= p_factor
                    E = (old_modulus == dets + dets) + O(q ** modified_prec)
                    if k_is_list:
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, E + multiplier_list[i] * R([local_factor_list[i][j] * main_term_list[i][j] for j in range(modified_prec)])
                    else:
                        if (k == 1) and _norm == 0:
                            E = E - weight_one_zero_val(g,S)
                        X[i_g] = g, _norm, E + multiplier * R([local_factor_list[j] * main_term_list[j] for j in range(modified_prec)])
                    precomputed_lists[_norm] = main_term_list
                else:
                    if k_is_list:
                        index = _indices[i_g]
                        for i in range(len_k):
                            X[i][i_g] = g, _norm, X[i][index][2]
                    else:
                        X[i_g] = g, _norm, X[_indices[i_g]][2]
            if k_is_list:
                return [WeilRepModularForm(k[i], S, X[i], weilrep = self) for i in range(len_k)]
            else:
                e = WeilRepModularForm(k, S, X, weilrep = self)
                self.__eisenstein = prec, e
                return e

    def eisenstein_newform(self, k, b, prec, allow_small_weight = False, print_exact = False):
        ## WARNING!! This is not fully tested and very likely has bugs!! it is also slow! ##

        r"""
        Compute Eisenstein newforms.

        This computes the sum:
        \sum_{\chi} E_{k, \beta, \chi}
        over all primitive Dirichlet characters \chi modulo the denominator of \beta.

        ALGORITHM: We use the formula of Theorem 1.4 of [Sch] for the Eisenstein series with character E_{k, \beta, \chi}. Their Fourier coefficients are computed numerically. We recover the (rational) power series using known bounds on the denominators of the Fourier coefficients.

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- a vector in self's discriminant group
        - ``prec`` -- the precision
        - ``allow_small_weight`` -- boolean (default False); if True then we do not check whether the weight is > 2
        - ``print_exact`` -- boolean (default False); if True then print some debugging information

        EXAMPLES::

            sage: w = WeilRep(matrix([[18]]))
            sage: w.eisenstein_newform(5/2, vector([1/3]), 5)
            [(0), O(q^5)]
            [(1/18), 72*q^(35/36) + 120*q^(71/36) + 396*q^(107/36) + 384*q^(143/36) + 804*q^(179/36) + O(q^5)]
            [(1/9), -54*q^(8/9) - 156*q^(17/9) - 276*q^(26/9) - 504*q^(35/9) - 660*q^(44/9) + O(q^5)]
            [(1/6), -4*q^(3/4) + 24*q^(7/4) - 36*q^(11/4) - 48*q^(15/4) + 168*q^(19/4) + O(q^5)]
            [(2/9), 24*q^(5/9) + 108*q^(14/9) + 264*q^(23/9) + 438*q^(32/9) + 540*q^(41/9) + O(q^5)]
            [(5/18), -12*q^(11/36) - 72*q^(47/36) - 276*q^(83/36) - 264*q^(119/36) - 672*q^(155/36) + O(q^5)]
            [(1/3), 2 - 12*q + 18*q^2 + 28*q^3 - 108*q^4 + O(q^5)]
            [(7/18), 24*q^(23/36) + 156*q^(59/36) + 192*q^(95/36) + 516*q^(131/36) + 480*q^(167/36) + O(q^5)]
            [(4/9), -6*q^(2/9) - 84*q^(11/9) - 216*q^(20/9) - 312*q^(29/9) - 528*q^(38/9) + O(q^5)]
            [(1/2), O(q^5)]
            [(5/9), 6*q^(2/9) + 84*q^(11/9) + 216*q^(20/9) + 312*q^(29/9) + 528*q^(38/9) + O(q^5)]
            [(11/18), -24*q^(23/36) - 156*q^(59/36) - 192*q^(95/36) - 516*q^(131/36) - 480*q^(167/36) + O(q^5)]
            [(2/3), -2 + 12*q - 18*q^2 - 28*q^3 + 108*q^4 + O(q^5)]
            [(13/18), 12*q^(11/36) + 72*q^(47/36) + 276*q^(83/36) + 264*q^(119/36) + 672*q^(155/36) + O(q^5)]
            [(7/9), -24*q^(5/9) - 108*q^(14/9) - 264*q^(23/9) - 438*q^(32/9) - 540*q^(41/9) + O(q^5)]
            [(5/6), 4*q^(3/4) - 24*q^(7/4) + 36*q^(11/4) + 48*q^(15/4) - 168*q^(19/4) + O(q^5)]
            [(8/9), 54*q^(8/9) + 156*q^(17/9) + 276*q^(26/9) + 504*q^(35/9) + 660*q^(44/9) + O(q^5)]
            [(17/18), -72*q^(35/36) - 120*q^(71/36) - 396*q^(107/36) - 384*q^(143/36) - 804*q^(179/36) + O(q^5)]
        """

        def l_value(k, psi0):
            psi = psi0.primitive_character()
            f_psi = psi.conductor()
            l_value_bar = CC(-psi.bar().bernoulli(k) / k)
            delta = psi.is_odd()
            L_val = l_value_bar * (2 * RRpi / f_psi)^k * CC(psi.gauss_sum_numerical(prec = lazy_bound)) / (2 * CC(I)^delta * RR(cos(math.pi * (k - delta) / 2)) * RR(k).gamma())
            for p in prime_factors(psi0.modulus() / f_psi):
                L_val = L_val * (1 - psi(p) / p^k)
            return L_val
        def lazy_l_value(g, n, S, p, k, u = None):
            L = 2 * g * S
            c = g * S * g + 2*n
            return L_values(2 * g * S, [c], S, p, k, t = u)[0]
        #check input
        prec = ceil(prec)
        if prec <= 0:
            raise ValueError('Precision must be at least 0')
        if k <= 2 and not allow_small_weight and (k < 2 or self.discriminant().is_squarefree()):
            raise ValueError('Weight must be at least 5/2')
        symm = self.is_symmetric_weight(k)
        if symm is None:
            raise ValueError('Invalid weight')
        #setup
        S = self.gram_matrix()
        e = S.nrows()
        if e % 2:
            denom = lcm([2*k-1, self.discriminant(), max(bernoulli(ZZ(2*k - 1)).numerator(), 1)])
            for p in self.discriminant().prime_divisors():
                denom = lcm(denom, p ** (ZZ(2*k - 1)) - 1)
        else:
            denom = lcm([k, self.discriminant(), max(bernoulli(k).numerator(), 1)])
            for p in self.discriminant().prime_divisors():
                denom = lcm(denom, p ** k - 1)
        if print_exact:
            print('denominator: ', denom)
        lazy_bound = max(4 * ceil(log(prec + 2) * (k - 1) + log(denom) + log(60) + log(k+1)), 53)
        RR = RealField(lazy_bound)
        CC = RR.complex_field()
        N_b = denominator(b)
        N_b_factor = factor(N_b)
        chi_list = [chi for chi in DirichletGroup(N_b) if chi.is_primitive() and chi.is_even() == symm]
        if not chi_list:
            raise ValueError('Not a newform')
        chi_decompositions = [chi.decomposition() for chi in chi_list]
        chi_gauss_sums = [chi.gauss_sum_numerical(prec = lazy_bound) for chi in chi_list]
        ds = self.ds()
        indices = self.rds(indices = True)
        discr = self.discriminant()
        sqrt_discr = sqrt(RR(discr))
        RRpi = RR.pi()
        RRgamma_k = RR(k).gamma()
        first_factor = (RR(2) ** (k+1)) * RRpi**k * CC(I) ** (k + self.signature() / 2) / (sqrt_discr * RRgamma_k)
        norm_list = self.norm_list()
        RPoly.<t> = PolynomialRing(QQ)
        X = []
        R.<q> = PowerSeriesRing(QQ)
        for i, g in enumerate(ds):
            if indices[i] is None:
                offset = norm_list[i]
                prec_g = prec - floor(offset)
                coeff_list = []
                for n_ceil in range(1, prec_g):
                    n = n_ceil + offset
                    N_bSg = ZZ(N_b * (b * S * g))
                    gcd_b_gb = gcd(N_b, N_bSg)
                    N_g = prod([p^d for (p, d) in factor(N_b) if gcd_b_gb % p == 0])
                    N_g_prime = N_b // N_g
                    D_g = DirichletGroup(N_g)
                    D_g_prime = DirichletGroup(N_g_prime)
                    chi_g_list = [prod([D_g(psi) for psi in chi_decompositions[i] if N_g % psi.modulus() == 0]) for i in range(len(chi_list))]
                    L_s = [[D_g_prime(psi) for psi in chi_decompositions[i] if N_g % psi.modulus()] for i in range(len(chi_list))]
                    chi_g_prime_list = [prod(L) if L else lambda x:1 for L in L_s]
                    front_factor = first_factor * RR(n) ** (k-1)
                    eps_factors = [chi_gauss_sums[i] * chi_g_prime(N_bSg)^(-1) / N_g for i, chi_g_prime in enumerate(chi_g_prime_list)]
                    D = ZZ(2 * N_g * N_g * n * S.determinant())
                    bad_primes = (D).prime_divisors()
                    if e % 2 == 1:
                        D0 = fundamental_discriminant(D * (-1)^((e+1)/2))
                    else:
                        D0 = fundamental_discriminant((-1)^(e/2) * S.determinant())
                    main_terms = [RR(1)]*len(chi_list)
                    chi0 = kronecker_character(D0)
                    for p in bad_primes:
                        main_term_L_val = lazy_l_value(g, n, S, p, k, u = t)
                        p_power = p ** (1 + e/2 - k)
                        Euler_factors = [CC(main_term_L_val(chi(p)*p_power)) if chi(p) else 1 for chi in chi_list]
                        chi0_p = CC(chi0(p))
                        if e % 2:
                            p_pow_2 = RR(p ** ZZ(1 / 2 - k))
                            for i in range(len(chi_list)):
                                chi_p = CC(chi_list[i](p))
                                main_terms[i] *= ( (1 - chi_p * chi0_p * p_pow_2) * Euler_factors[i] / (1 - (chi_p * p_pow_2) ** 2))#.n()
                        else:
                            for i in range(len(chi_list)):
                                main_terms[i] *= Euler_factors[i] / (1 - chi_list[i](p) * chi0_p * (p ** (-k)))
                    if e % 2:
                        for i, chi in enumerate(chi_list):
                            G = DirichletGroup(lcm(chi.modulus(), chi0.modulus()))
                            main_terms[i] *= CC(l_value(k - 1/2, G(chi)*G(chi0)))
                            main_terms[i] /= CC(l_value(2*k - 1, chi * chi))
                    else:
                        for i, chi in enumerate(chi_list):
                            G = DirichletGroup(lcm(chi.modulus(), chi0.modulus()))
                            main_terms[i] /= CC(l_value(k, G(chi)*G(chi0)))
                    finite_parts = [1 for _ in chi_list]
                    for p in prime_factors(gcd_b_gb):
                        p_sum = 0
                        p_power_N_b = p ** (N_b.valuation(p))
                        vp_g = gcd_b_gb.valuation(p)
                        w_p = 1 + 2 * (2*N_b * N_g * n).valuation(p)
                        N_g_over_p = N_g / (p ** N_g.valuation(p))
                        Dp_prime = DirichletGroup(N_g_over_p)
                        Dp = DirichletGroup(p ** N_g.valuation(p))
                        L_s = [[Dp_prime(psi) for psi in chi_decomposition if not N_g_over_p % psi.modulus()] for chi_decomposition in chi_decompositions]
                        chi_p_prime_list = [prod(L) if L else lambda x:1 for L in L_s]
                        chi_p_list = [prod([Dp(psi) for psi in chi_decomposition if not psi.modulus() % p]) for chi_decomposition in chi_decompositions]
                        s = vector([0] * len(chi_list))
                        for alpha in range(vp_g, w_p + 1):
                            p_alpha = p ** alpha
                            p_pow_list = [CC(chi_p_prime(p_alpha)) * (p_alpha ** (1 - e/2 - k)) for chi_p_prime in chi_p_prime_list]
                            p_e_alpha = (p_alpha ** e) / p
                            s_alpha = vector([0]*len(chi_list))
                            for v in range(p_power_N_b):
                                new_g = g - v * (N_b // p_power_N_b) * b
                                for u in range(p_power_N_b):
                                    if u % p and not (u * p_alpha - N_bSg) % p_power_N_b:
                                        LvalueSeries = RPoly(lazy_l_value(new_g, n + p_alpha * v * u / p_power_N_b , S, p, k, u = t))
                                        s_alpha += CC(p_e_alpha * (p * LvalueSeries[alpha] - LvalueSeries[alpha - 1])) * vector(CC(chi_p.bar()(u)) for chi_p in chi_p_list)
                            s = s + vector(p_pow_list[i] * s_alpha[i] for i in range(len(p_pow_list)))
                        for i in range(len(finite_parts)):
                            finite_parts[i] *= s[i]
                    coeff = (front_factor * sum(eps_factors[i] * main_terms[i] * finite_parts[i] for i in range(len(chi_list))))
                    coeff_numerator = RR(coeff.real()) * denom
                    if print_exact:
                        print(g, n, 'exact: ', coeff_numerator, front_factor.n(), eps_factors[0].n(), main_terms[0].n(), finite_parts[0].n())
                    coeff = round(coeff_numerator) / denom
                    coeff_list.append(coeff)
                f = q * R(coeff_list) + O(q ** prec_g)
                const_term = 0
                for n in range(N_b):
                    if denominator(g - n * b) == 1:
                        const_term = const_term + 2 * sum(chi(n).n() for chi in chi_list)
                const_term = round(const_term.real() * denom) / denom
                f = f + const_term
                X.append([g, offset, f])
            else:
                X.append([g, norm_list[indices[i]], (2 * symm - 1) * X[indices[i]][2]])
        return WeilRepModularForm(k, S, X, weilrep = self)

    def eisenstein_oldform(self, k, b, prec, allow_small_weight = False):
        r"""
        Compute certain Eisenstein oldforms.

        This computes the sum over E_{k, \beta} where \beta runs through all multiples of ``b`` in self's discriminant group.

        INPUT:
        - ``k`` -- the weight
        - ``b`` -- an element of self's discriminant group with integral norm
        - ``prec`` -- precision

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[8]]))
            sage: w.eisenstein_oldform(7/2, vector([1/2]), 5)
            [(0), 1 + 126*q + 756*q^2 + 2072*q^3 + 4158*q^4 + O(q^5)]
            [(1/8), O(q^5)]
            [(1/4), 56*q^(3/4) + 576*q^(7/4) + 1512*q^(11/4) + 4032*q^(15/4) + 5544*q^(19/4) + O(q^5)]
            [(3/8), O(q^5)]
            [(1/2), 1 + 126*q + 756*q^2 + 2072*q^3 + 4158*q^4 + O(q^5)]
            [(5/8), O(q^5)]
            [(3/4), 56*q^(3/4) + 576*q^(7/4) + 1512*q^(11/4) + 4032*q^(15/4) + 5544*q^(19/4) + O(q^5)]
            [(7/8), O(q^5)]
        """
        k_is_list = type(k) is list
        E = self.eisenstein_series(k, prec, allow_small_weight = allow_small_weight)
        d_b = denominator(b)
        if d_b == 1:
            return E
        R.<q> = PowerSeriesRing(QQ)
        S = self.gram_matrix()
        X = E.components()
        ds = self.ds()
        norm_list = self.norm_list()
        Y = [None] * len(ds)
        for i, g in enumerate(ds):
            g_b = frac(g * S * b)
            if g_b:
                Y[i] = g, norm_list[i], O(q ** (prec - floor(norm_list[i])))
            else:
                f = sum(X[tuple(frac(x) for x in g + j * b)] for j in range(d_b))
                Y[i] = g, norm_list[i], f
        return WeilRepModularForm(k, S, Y, weilrep = self)

    def eisenstein_series_shadow(self, prec):
        r"""
        Compute the shadow of the weight two Eisenstein series (if it exists).

        INPUT:
        - ``prec`` -- precision

        OUTPUT: a WeilRepModularForm of weight 0

        NOTE: the precision is irrelevant because modular forms of weight 0 are constant.

        EXAMPLES::

            sage: WeilRep(matrix([[2,0],[0,-2]])).eisenstein_series_shadow(5)
            [(0, 0), 1 + O(q^5)]
            [(1/2, 0), O(q^5)]
            [(0, 1/2), O(q^5)]
            [(1/2, 1/2), 1 + O(q^5)]

            sage: w = WeilRep(matrix([[8,0],[0,-2]]))
            sage: w.eisenstein_series_shadow(5)
            [(0, 0), 1 + O(q^5)]
            [(1/8, 0), O(q^5)]
            [(1/4, 0), O(q^5)]
            [(3/8, 0), O(q^5)]
            [(1/2, 0), 1 + O(q^5)]
            [(5/8, 0), O(q^5)]
            [(3/4, 0), O(q^5)]
            [(7/8, 0), O(q^5)]
            [(0, 1/2), O(q^5)]
            [(1/8, 1/2), O(q^5)]
            [(1/4, 1/2), 1 + O(q^5)]
            [(3/8, 1/2), O(q^5)]
            [(1/2, 1/2), O(q^5)]
            [(5/8, 1/2), O(q^5)]
            [(3/4, 1/2), 1 + O(q^5)]
            [(7/8, 1/2), O(q^5)]

        """
        prec = ceil(prec)
        if not self.is_symmetric_weight(0):
            raise NotImplementedError
        _ds = self.ds()
        _indices = self.rds(indices = True)
        _dets = self.discriminant()
        n_list = self.norm_list()
        R.<q> = PowerSeriesRing(QQ)
        o_q_prec = O(q ** prec)
        o_q_prec_plus_one = O(q ** (prec + 1))
        try:
            A_sqrt = ZZ(sqrt(_dets))
        except:
            return self.zero(0)
        S = self.gram_matrix()
        _nrows = S.nrows()
        bad_primes = (2*_dets).prime_factors()
        X = [None] * len(_ds)
        RPoly.<t> = PolynomialRing(QQ)
        for i, g in enumerate(_ds):
            L =  g * S
            c = L * g
            L = L + L
            if _indices[i]:
                x = X[_indices[i]]
                X[i] = g, x[1], x[2]
            else:
                offset = n_list[i]
                if offset:
                    X[i] = g, offset, o_q_prec_plus_one
                else:
                    #print(g, prod(L_values(L, [0], -S, p, 2, t=t)[0] / (1 + 1/p) for p in bad_primes))
                    X[i] = g, 0, o_q_prec + prod(L_values(L, [-c], -S, p, 2)[0] / (1 + 1/p) for p in bad_primes)
        return WeilRepModularForm(0, S, X, weilrep = self)


    def pss(self, weight, b, m, prec, weilrep = None):
        r"""
        Compute Poincare square series.

        These are obtained by theta-contraction of Eisenstein series attached to other lattices.

        INPUT:
        - ``weight`` -- the weight (a half-integer) which is at least 5/2
        - ``b`` -- a vector for which b*S is integral (where S is our Gram matrix)
        - ``m`` -- a rational number for which m + b*S*b/2 is a positive integer
        - ``prec`` -- precision (natural number)

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: WeilRep(matrix([[2,0],[0,-2]])).pss(6,vector([1/2,1/2]),1,5)
            [(0, 0), 1 - 8008/31*q - 8184*q^2 - 1935520/31*q^3 - 262392*q^4 + O(q^5)]
            [(1/2, 0), -1918/31*q^(3/4) - 130460/31*q^(7/4) - 1246938/31*q^(11/4) - 5912724/31*q^(15/4) - 19187894/31*q^(19/4) + O(q^5)]
            [(0, 1/2), -11/62*q^(1/4) - 24105/31*q^(5/4) - 919487/62*q^(9/4) - 2878469/31*q^(13/4) - 11002563/31*q^(17/4) + O(q^5)]
            [(1/2, 1/2), -7616/31*q - 8448*q^2 - 1876736/31*q^3 - 270336*q^4 + O(q^5)]

        """
        if weight < 5/2:
            raise NotImplementedError
        S = self.gram_matrix()
        if not weilrep:
            if S:
                tilde_b = b*S
                shift_m = m + b*tilde_b/2
                tilde_b = matrix(tilde_b)
                S_new = block_matrix(ZZ,[[S,tilde_b.transpose()],[tilde_b,2*shift_m]])
            else:
                S_new = matrix(ZZ, [[2*m]])
            w = WeilRep(S_new)
        else:
            w = weilrep
        new_k = weight - 1/2
        _components = [self.ds(), self.rds(indices = True)]
        X = w.eisenstein_series(new_k, prec, allow_small_weight = True).theta_contraction(components = _components)
        if weight > 5/2:
            return X
        elif weight == 5/2:#result might be wrong so lets fix it
            dets = w.discriminant()
            try:
                epsilon = QQ(24*(-1)^((1 + self.signature())/4) / sqrt(abs(dets))) #maybe we will need this number
            except TypeError:
                return X #result was ok
            R.<q> = PowerSeriesRing(QQ)
            theta = w.eisenstein_series_shadow(prec+1).theta_contraction(components = _components).fourier_expansion()
            Y = X.fourier_expansion()
            Z = [None] * len(theta)
            for i in range(len(theta)):
                offset = theta[i][1]
                theta_f = list(theta[i][2])
                Z[i] = Y[i][0], Y[i][1], Y[i][2] - epsilon * sum((n + offset) * theta_f[n] * (q ** n) for n in range(1, len(theta_f)) if theta_f[n])
                #Y[i][2] -= epsilon * sum((n + offset)*theta_f[n]*q^n for n in range(1, len(theta_f)) if theta_f[n])#fix it by adding derivative of theta-contraction of the weight 2 Eisenstein series' shadow, multiplied by epsilon
            return WeilRepModularForm(weight, S, Z, weilrep = self)


    def pssd(self, weight, b, m, prec, weilrep = None):
        r"""
        Compute antisymmetric modular forms.

        These are obtained by theta-contraction of Eisenstein series attached to other lattices.

        INPUT:
        - ``weight`` -- the weight (a half-integer) which is at least 7/2
        - ``b`` -- a vector for which b*S is integral (where S is our Gram matrix)
        - ``m`` -- a rational number for which m + b*S*b/2 is a positive integer
        - ``prec`` -- precision (natural number)

        NOTE: if b has order 2 in our discriminant group then this is zero!

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: W = WeilRep(matrix([[0,0,3],[0,-2,0],[3,0,0]]))
            sage: W.pssd(7/2, vector([0,1/2,1/3]), 1/4, 5)
            [(0, 0, 0), O(q^5)]
            [(0, 1/2, 1/3), 1/2*q^(1/4) - 3*q^(5/4) + 9/2*q^(9/4) + 6*q^(13/4) - 21*q^(17/4) + O(q^5)]
            [(0, 0, 2/3), q - 6*q^2 + 9*q^3 + 10*q^4 + O(q^5)]
            [(0, 1/2, 0), O(q^5)]
            [(0, 0, 1/3), -q + 6*q^2 - 9*q^3 - 10*q^4 + O(q^5)]
            [(0, 1/2, 2/3), -1/2*q^(1/4) + 3*q^(5/4) - 9/2*q^(9/4) - 6*q^(13/4) + 21*q^(17/4) + O(q^5)]
            [(1/3, 0, 0), q - 6*q^2 + 9*q^3 + 10*q^4 + O(q^5)]
            [(1/3, 1/2, 1/3), O(q^5)]
            [(1/3, 0, 2/3), q^(1/3) - 6*q^(4/3) + 10*q^(7/3) + 4*q^(10/3) - 20*q^(13/3) + O(q^5)]
            [(1/3, 1/2, 0), -1/2*q^(1/4) + 3*q^(5/4) - 9/2*q^(9/4) - 6*q^(13/4) + 21*q^(17/4) + O(q^5)]
            [(1/3, 0, 1/3), O(q^5)]
            [(1/3, 1/2, 2/3), -q^(7/12) + 5*q^(19/12) - 3*q^(31/12) - 19*q^(43/12) + 20*q^(55/12) + O(q^5)]
            [(2/3, 0, 0), -q + 6*q^2 - 9*q^3 - 10*q^4 + O(q^5)]
            [(2/3, 1/2, 1/3), q^(7/12) - 5*q^(19/12) + 3*q^(31/12) + 19*q^(43/12) - 20*q^(55/12) + O(q^5)]
            [(2/3, 0, 2/3), O(q^5)]
            [(2/3, 1/2, 0), 1/2*q^(1/4) - 3*q^(5/4) + 9/2*q^(9/4) + 6*q^(13/4) - 21*q^(17/4) + O(q^5)]
            [(2/3, 0, 1/3), -q^(1/3) + 6*q^(4/3) - 10*q^(7/3) - 4*q^(10/3) + 20*q^(13/3) + O(q^5)]
            [(2/3, 1/2, 2/3), O(q^5)]

        """
        if weight < 7/2:
            raise NotImplementedError
        if not weilrep:
            S = self.gram_matrix()
            if S:
                tilde_b = b*S
                shift_m = m + b*tilde_b/2
                tilde_b = matrix(tilde_b)
                S_new = block_matrix(ZZ,[[S,tilde_b.transpose()],[tilde_b,2*shift_m]])
            else:
                S_new = matrix(ZZ, [[2*m]])
            w = WeilRep(S_new)
        else:
            w = weilrep
        new_k = weight - 3/2
        X = w.eisenstein_series(new_k, prec, allow_small_weight = True).theta_contraction(odd = True, weilrep = self)
        if weight > 7/2:
            return X
        else:#result might be wrong so lets fix it
            try:
                epsilon = QQ(8*(-1)^((1 + self.signature())/4) / sqrt(self.discriminant()))#factor has to be 8 here, not 24 like in pss()
            except TypeError:
                return X#result was ok
            R.<q> = PowerSeriesRing(QQ)
            theta = w.eisenstein_shadow(prec+1).theta_contraction(odd = True).fourier_expansion()#this is a weight 3/2 theta
            Y = X.fourier_expansion()
            Z = [None] * len(theta)
            for i in range(len(theta)):
                offset = theta[i][1]
                theta_f = list(theta[i][2])
                #Y[i][2] -= epsilon*sum((n + offset)*theta_f[n]*q^n for n in range(1, len(theta_f)) if theta_f[n])#add derivative of theta
                Z[i] = Y[i][0], Y[i][1], Y[i][2] - epsilon * sum((n + offset) * theta_f[n] * (q ** n) for n in range(1, len(theta_f)) if theta_f[n])
            return WeilRepModularForm(weight, self.gram_matrix(), Z, weilrep = self)

    def pss_double(self, weight, b, m, prec):#to be used whe weight is even and >= 3
        r"""
        Compute the double theta-contraction of an Eisenstein series attached to a lattice of self's rank + 2

        (Usually this is slower.)

        INPUT:
        - ``weight`` -- the weight (a half-integer) which is at least 3
        - ``b`` -- a vector for which b*S is integral (where S is our Gram matrix) OR a matrix of size (2 x S.nrows())
        - ``m`` -- a rational number for which m + b*S*b/2 is a positive integer OR a symmetric rational matrix of size (2x2) for which 2*m + b*S*b.transpose() is a positive-definite integral matrix with even diagonal
        - ``prec`` -- precision (natural number)

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[4]]))
            sage: w.pss_double(7/2, vector([1/2]), 1/2, 5)
            [(0), 1 + 84*q + 574*q^2 + 1288*q^3 + 3444*q^4 + O(q^5)]
            [(1/4), 64*q^(7/8) + 448*q^(15/8) + 1344*q^(23/8) + 2688*q^(31/8) + 4928*q^(39/8) + O(q^5)]
            [(1/2), 14*q^(1/2) + 280*q^(3/2) + 840*q^(5/2) + 2368*q^(7/2) + 3542*q^(9/2) + O(q^5)]
            [(3/4), 64*q^(7/8) + 448*q^(15/8) + 1344*q^(23/8) + 2688*q^(31/8) + 4928*q^(39/8) + O(q^5)]
        """
        if weight < 3:
            raise NotImplementedError
        S = self.gram_matrix()
        try:
            _is_vector = b.is_vector() #???
            tilde_b = b*S
            shift_m = m + b*tilde_b/2
            tilde_b = matrix(tilde_b)
            zero_v = matrix([0] * S.nrows())
            S_new = block_matrix(ZZ,[[S, tilde_b.transpose(), zero_v.transpose()],[tilde_b, 2*shift_m, 0],[zero_v, 0, 2]])
        except AttributeError:
            tilde_b = b*S
            two_shift_m = 2*m + tilde_b * b.transpose()
            S_new = block_matrix(ZZ, [[S, tilde_b.transpose()], [tilde_b, two_shift_m]])
        new_k = weight - 1
        w = WeilRep(S_new)
        _components = [self.ds(), self.rds(indices = True)]
        X = w.eisenstein_series(weight - 1, prec, allow_small_weight = True).theta_contraction().theta_contraction(components = _components)
        if weight > 3:
            return X
        else:
            raise NotImplementedError #to be fixed
            R.<q> = PowerSeriesRing(QQ)
            theta = w.eisenstein_series_shadow(prec+1).theta_contraction().theta_contraction(components = _components).fourier_expansion()
            Y = X.fourier_expansion()
            Z = [None] * len(theta)
            for i in range(len(theta)):
                offset = theta[i][1]
                theta_f = list(theta[i][2])
                Z[i] = Y[i][0], Y[i][1], Y[i][2] - epsilon * sum((n + offset) * theta_f[n] * (q ** n) for n in range(1, len(theta_f)) if theta_f[n])
        return WeilRepModularForm(weight, S, Z, weilrep = self)

    def recover_modular_form_from_coefficient_vector(self, k, coefficient_vector, prec, starting_from = 0):
        r"""
        Recover a WeilRepModularForm for this representation from its coefficient vector.

        INPUT:
        - ``k`` -- the weight of the modular form
        - ``coefficient_vector`` -- a vector of coefficients
        - ``prec`` -- precision
        - ``starting_from`` -- the exponent at which the vector of coefficients begins (default 0)

        OUTPUT: WeilRepModularForm

        EXAMPLES::

            sage: w = WeilRep(matrix([[2,0],[0,-2]]))
            sage: e4 = w.eisenstein_series(4, 5)
            sage: v = e4.coefficient_vector()
            sage: e4 == w.recover_modular_form_from_coefficient_vector(4, v, 5)
            True

        """
        R.<q> = PowerSeriesRing(QQ)
        symm = self.is_symmetric_weight(k)
        Y = self.coefficient_vector_exponents(prec, symm = symm, starting_from = starting_from, include_vectors = True)
        eps = 2 * symm - 1
        _ds = self.ds()
        _ds_dict = self.ds_dict()
        _indices = self.rds(indices = True)
        _norm_list = self.norm_list()
        X = [None]*len(_ds)
        for i, c in enumerate(coefficient_vector):
            g, n = Y[i]
            j = _ds_dict[g]
            if X[j]:
                X[j][2] += c * q^(ceil(n))
            else:
                X[j] = [vector(g), _norm_list[j], O(q^(prec - floor(_norm_list[j]))) + c * q^(ceil(n))]
            minus_g = tuple(frac(-x) for x in g)
            if minus_g != g:
                j2 = _ds_dict[minus_g]
                if X[j2]:
                    X[j2][2] += eps * c * q^(ceil(n))
                else:
                    X[j2] = [vector(minus_g), _norm_list[j], O(q^(prec - floor(_norm_list[j]))) + eps * c * q^(ceil(n))]
        for i, g in enumerate(_ds):
            if X[i] is None:
                X[i] = g, _norm_list[i], O(q^(prec - floor(_norm_list[i])))
            else:
                X[i] = tuple(X[i])
        return WeilRepModularForm(k, self.gram_matrix(), X, weilrep = self)

    def theta_series(self, prec, P = None, test_P = True):
        r"""
        Construct vector-valued theta series.

        This computes the theta series \sum_x P(x) q^(-Q(x)) e_x, where Q is a negative-definite quadratic form, P is a harmonic homogeneous polynomial and x runs through the *dual* lattice of Q.

        NOTE: We take *negative-definite* quadratic forms because the Weil representation here is the *dual* of the theta representation. This is somewhat unfortunate, but the convention here is more natural for working with Jacobi forms.

        ALGORITHM: try to apply PARI qfminim on the inverse of the Gram matrix. if this fails then we rescale the inverse of the Gram matrix by the level of Q to obtain something integral. this seems to be necessary to get the nontrivial components of the theta series. (The e_0 component is simply the result of (-Q).theta_series())

        INPUT:
        - ``prec`` -- the precision
        - ``P`` -- a polynomial which is homogeneous and is harmonic with respect to the underlying quadratic form
        - ``test_P`` -- a boolean (default True). If False then we do not test whether P is homogeneous and harmonic. (If P is not harmonic then the theta series is only a quasi-modular form!)

        OUTPUT: VVMF

        EXAMPLES::

            sage: WeilRep(-matrix([[2,1],[1,4]])).theta_series(5)
            [(0, 0), 1 + 2*q + 4*q^2 + 6*q^4 + O(q^5)]
            [(3/7, 1/7), 2*q^(2/7) + q^(9/7) + 5*q^(16/7) + 2*q^(23/7) + O(q^5)]
            [(6/7, 2/7), q^(1/7) + 4*q^(8/7) + 4*q^(22/7) + 2*q^(29/7) + O(q^5)]
            [(2/7, 3/7), 3*q^(4/7) + 2*q^(11/7) + 2*q^(18/7) + q^(25/7) + 6*q^(32/7) + O(q^5)]
            [(5/7, 4/7), 3*q^(4/7) + 2*q^(11/7) + 2*q^(18/7) + q^(25/7) + 6*q^(32/7) + O(q^5)]
            [(1/7, 5/7), q^(1/7) + 4*q^(8/7) + 4*q^(22/7) + 2*q^(29/7) + O(q^5)]
            [(4/7, 6/7), 2*q^(2/7) + q^(9/7) + 5*q^(16/7) + 2*q^(23/7) + O(q^5)]

            sage: R.<x,y> = PolynomialRing(QQ)
            sage: P = x^2 - 2*y^2
            sage: w = WeilRep(matrix([[-2,-1],[-1,-4]]))
            sage: w.theta_series(10, P = P)
            [(0, 0), 2*q - 6*q^2 + 10*q^4 - 14*q^7 - 6*q^8 + 18*q^9 + O(q^10)]
            [(3/7, 1/7), 3/7*q^(2/7) - 9/7*q^(9/7) + 11/7*q^(16/7) - 18/7*q^(23/7) + 38/7*q^(37/7) + 30/7*q^(44/7) - 162/7*q^(58/7) + O(q^10)]
            [(6/7, 2/7), -1/7*q^(1/7) + 3/7*q^(8/7) - 18/7*q^(22/7) + 54/7*q^(29/7) - 45/7*q^(36/7) - 58/7*q^(43/7) + 75/7*q^(50/7) + 13*q^(64/7) + O(q^10)]
            [(2/7, 3/7), -5/7*q^(4/7) + 6/7*q^(11/7) + 27/7*q^(18/7) - 25/7*q^(25/7) - 45/7*q^(32/7) + 54/7*q^(46/7) + 6/7*q^(53/7) + 118/7*q^(67/7) + O(q^10)]
            [(5/7, 4/7), -5/7*q^(4/7) + 6/7*q^(11/7) + 27/7*q^(18/7) - 25/7*q^(25/7) - 45/7*q^(32/7) + 54/7*q^(46/7) + 6/7*q^(53/7) + 118/7*q^(67/7) + O(q^10)]
            [(1/7, 5/7), -1/7*q^(1/7) + 3/7*q^(8/7) - 18/7*q^(22/7) + 54/7*q^(29/7) - 45/7*q^(36/7) - 58/7*q^(43/7) + 75/7*q^(50/7) + 13*q^(64/7) + O(q^10)]
            [(4/7, 6/7), 3/7*q^(2/7) - 9/7*q^(9/7) + 11/7*q^(16/7) - 18/7*q^(23/7) + 38/7*q^(37/7) + 30/7*q^(44/7) - 162/7*q^(58/7) + O(q^10)]

        """
        Q = self.__quadratic_form
        if not Q.is_negative_definite():
            raise ValueError('Not a negative-definite lattice')
        R.<q> = PowerSeriesRing(QQ)
        _ds = self.ds()
        _ds_dict = self.ds_dict()
        n_dict = self.norm_dict()
        if P == 0:
            return self.zero(prec = prec)
        S_inv = -self.gram_matrix().inverse()
        deg_P = 0
        if P and test_P:#test whether P is OK
            deg_P = P.degree()
            if len(P.variables()) != Q.dim():
                raise ValueError('The number of variables in P does not equal the lattice rank')
            if not P.is_homogeneous():
                raise ValueError('Not a homogeneous polynomial')
            u = vector(P.gradient())*S_inv
            if sum(x.derivative(P.variables()[i]) for i, x in enumerate(u)):
                raise ValueError('Not a harmonic polynomial')
        else:
            P = lambda x: 1
        try:
            _, _, vs_matrix = pari(S_inv).qfminim(prec + prec + 1, flag=2)
            vs_list = vs_matrix.sage().columns()
            X = [[g, n_dict[tuple(g)], O(q^(prec - floor(n_dict[tuple(g)])))] for g in _ds]
            for v in vs_list:
                g = S_inv * v
                P_val = P(list(g))
                v_norm_with_offset = ceil(v*g/2)
                list_g = [frac(x) for x in g]
                g = tuple(list_g)
                j1 = _ds_dict[g]
                X[j1][2] += P_val * q^(v_norm_with_offset)
                if v:
                    minus_g = tuple([frac(-x) for x in g])
                    j2 = _ds_dict[minus_g]
                    X[j2][2] += (-1)^deg_P * P_val * q^(v_norm_with_offset)
            X[0][2] += P([0]*S_inv.nrows())
        except PariError: #when we are not allowed to use pari's qfminim with flag=2 for some reason. the code below is a little slower
            level = Q.level()
            Q_adj = QuadraticForm(level * S_inv)
            vs_list = Q_adj.short_vector_list_up_to_length(level*prec)
            X = [[g, n_dict[tuple(g)], O(q^(prec - floor(n_dict[tuple(g)])))] for g in _ds]
            for i in range(len(vs_list)):
                v_norm_offset = ceil(i/level)
                vs = vs_list[i]
                for v in vs:
                    S_inv_v = -S_inv*v
                    v_frac = tuple(frac(S_inv_v[l]) for l in range(len(S_inv_v)))
                    j = _ds_dict[v_frac]
                    X[j][2] += P(list(S_inv_v))*q^v_norm_offset
        return WeilRepModularForm(Q.dim()/2 + deg_P, self.gram_matrix(), X, weilrep = self)

    def zero(self, weight = 0, prec = 20):
        r"""
        Construct a WeilRepModularForm of weight 'weight' and precision 'prec' which is identically zero.

        EXAMPLES::

            sage: w = WeilRep(matrix([[2,1],[1,2]]))
            sage: w.zero(3, 5)
            [(0, 0), O(q^5)]
            [(2/3, 2/3), O(q^5)]
            [(1/3, 1/3), O(q^5)]

        """

        n_list = self.norm_list()
        _ds = self.ds()
        R.<q> = PowerSeriesRing(QQ)
        o_q = O(q ** prec)
        o_q_2 = O(q ** (prec + 1))
        X = [[g, n_list[i], [o_q_2,o_q][n_list[i] == 0]] for i, g in enumerate(_ds)]
        return WeilRepModularForm(weight, self.gram_matrix(), X, weilrep = self)

    ## dimensions of spaces of modular forms associated to this representation

    def rank(self, symm):
        r"""
        Compute the rank of self's modular forms as a module over the ring M_*(SL_2(Z)) of scalar-valued modular forms.

        INPUT:
        - ``symm`` -- boolean: if True then we consider the module of symmetric forms, if False then antisymmetric forms

        OUTPUT: a natural number

        EXAMPLES::

            sage: w = WeilRep(matrix([[4,0],[0,4]]))
            sage: w.rank(0), w.rank(1)
            (6, 10)

        """

        try:
            return [self.__antisymm_rank, self.__symm_rank][symm]
        except AttributeError:
            rds_list = self.rds()
            self.__symm_rank = len(self.rds())
            self.__antisymm_rank = self.__symm_rank - sum(self.__order_two_in_rds_list)
            return [self.__antisymm_rank, self.__symm_rank][symm]

    def cusp_forms_dimension(self, weight, eta_twist = 0, force_Riemann_Roch = False, do_not_round = False):
        r"""
        Compute the dimension of spaces of cusp forms.

        This computes the dimension of the space of cusp forms using Riemann-Roch. The formula is valid in weights > 2. In weights 1/2,1,3/2,2 we compute a --basis-- of the space and take its length. (This is slow!)

        INPUT:
        - ``weight`` -- the weight; a half-integer
        - ``eta_twist`` -- an integer (default 0). This computes instead the dimension of cusp forms after twisting the Weil representation by this power of the eta multiplier
        - ``force_Riemann_Roch`` -- a boolean (default False). If True then we produce the output of the Riemann-Roch formula, regardless of whether this represents the dimension
        - ``do_not_round`` -- a boolean (default False). If True then do not convert the output to an integer. (This is probably only useful for debugging.)

        OUTPUT: the dimension of the space of cusp forms of the given weight and eta twist as an integer (unless specified otherwise)

        NOTE: the first time we compute any dimension we have to compute some Gauss sums. After this it should be faster.

        EXAMPLES::

            sage:WeilRep(matrix([[2,1],[1,2]])).cusp_forms_dimension(11)
            2

            sage:WeilRep(matrix([[4]])).cusp_forms_dimension(21/2)
            1
        """
        eta_twist %= 24
        symm = self.is_symmetric_weight(weight - eta_twist/2)
        if weight <= 0 or symm is None:
            return 0 #should this be an error instead?
        elif weight > 2 or (weight == 2 and (eta_twist != 0 or not self.discriminant().is_squarefree())) or force_Riemann_Roch:
            eps = 2 * symm - 1
            modforms_rank = self.rank(symm)
            pi_i = complex(0.0, math.pi)
            sig = self.signature()
            try:
                gauss_sum_1 = self.__gauss_sum_1
                gauss_sum_2 = self.__gauss_sum_2
                gauss_sum_3 = self.__gauss_sum_3
                alpha_T = self.__alpha_T[eta_twist]
                if eps == -1:
                    alpha_T -= self.__alpha_T_order_two[eta_twist]
                sqrt_A = math.sqrt(self.discriminant())
                count_isotropic_vectors = self.__count_isotropic_vectors
                count_isotropic_vectors_of_order_two = self.__count_isotropic_vectors_of_order_two
            except AttributeError:
                S = self.gram_matrix()
                rds_grp = self.rds()
                order_two_indices = self.__order_two_in_rds_list
                sqrt_A = math.sqrt(self.discriminant())
                count_isotropic_vectors = [0]*24
                count_isotropic_vectors_of_order_two = [0]*24
                self.__alpha_T = vector([0]*24)
                self.__alpha_T_order_two = vector([0]*24)
                gauss_sum_1 = exp(pi_i * sig / 4) * sqrt_A
                self.__gauss_sum_1 = gauss_sum_1
                gauss_sum_2 = 0.0
                gauss_sum_3 = 0.0
                for i,g in enumerate(rds_grp):
                    gsg = (g*S*g)/2
                    multiplier = 2 - order_two_indices[i]
                    v = vector([frac(-gsg + N/24) for N in range(24)]) #fix this?
                    for j in range(len(v)):
                        if v[j] == 0:
                            count_isotropic_vectors[j] += 1
                            if order_two_indices[i]:
                                count_isotropic_vectors_of_order_two[j] += 1
                    self.__alpha_T += v
                    if order_two_indices[i]:
                        self.__alpha_T_order_two += v
                    pi_i_gsg = pi_i * gsg.n()
                    gauss_sum_2 += multiplier * exp(4*pi_i_gsg)
                    gauss_sum_3 += multiplier * exp(-6*pi_i_gsg)
                self.__gauss_sum_2 = gauss_sum_2
                self.__gauss_sum_3 = gauss_sum_3
                self.__count_isotropic_vectors = count_isotropic_vectors
                self.__count_isotropic_vectors_of_order_two = count_isotropic_vectors_of_order_two
                alpha_T = self.__alpha_T[eta_twist]
                alpha_T_order_two = self.__alpha_T_order_two[eta_twist]
                if not symm:
                    alpha_T = alpha_T - alpha_T_order_two
            g2 = gauss_sum_2.real if symm else gauss_sum_2.imag
            result_dim = modforms_rank * (weight + 5)/12 +  (exp(pi_i * (2*weight + sig + 1 - eps - eta_twist)/4) * g2).real / (4*sqrt_A) - alpha_T -  (exp(pi_i * (3*sig - 2*eta_twist + 4 * weight - 10)/12) * (gauss_sum_1 + eps * gauss_sum_3)).real / (3 * math.sqrt(3) * sqrt_A) - count_isotropic_vectors[eta_twist] + (1 - symm) * count_isotropic_vectors_of_order_two[eta_twist]
            if do_not_round:
                return result_dim
            else:
                return ZZ(round(result_dim))
        else:#not good
            if eta_twist != 0:
                raise ValueError('Not yet implemented')
            return len(self.cusp_forms_basis(weight))

    def modular_forms_dimension(self, weight, eta_twist = 0, force_Riemann_Roch = False, do_not_round = False):
        r"""
        Compute the dimension of spaces of modular forms.

        This computes the dimension of the space of modular forms using Riemann-Roch. The formula is valid in weights >= 2. In weights 0,1/2,1,3/2 we compute a *basis* of the space and then take its length. (This is slow!)

        INPUT:
        - ``weight`` -- the weight; a half-integer
        - ``eta_twist`` -- an integer (default 0). This computes instead the dimension of modular forms after twisting the Weil representation by this power of the eta multiplier
        - ``force_Riemann_Roch`` -- a boolean (default False). If True then we produce the output of the Riemann-Roch formula, regardless of whether this represents the dimension
        - ``do_not_round`` -- a boolean (default False). If True then do not convert the output to an integer. (This is probably only useful for debugging.)

        OUTPUT: the dimension of the space of modular forms of the given weight and eta twist as an integer (unless specified otherwise)

        NOTE: the first time we compute any dimension, a lot of Gauss sums have to be computed. After this it should be fast.

        EXAMPLES::

            sage:WeilRep(matrix([[2,1],[1,2]])).modular_forms_dimension(11)
            3

            sage:WeilRep(matrix([[4]])).modular_forms_dimension(21/2)
            2

            sage:WeilRep(matrix([[4]])).modular_forms_dimension(17/2, 2)
            2

        """
        eta_twist %= 24
        if weight >= 2 or force_Riemann_Roch:
            symm = self.is_symmetric_weight(weight - eta_twist / 2)
            cusp_dim = self.cusp_forms_dimension(weight, eta_twist, force_Riemann_Roch = True, do_not_round = do_not_round)
            return cusp_dim + self.__count_isotropic_vectors[eta_twist] + (symm - 1) * self.__count_isotropic_vectors_of_order_two[eta_twist]
        elif weight < 0:
            return 0
        else:
            if eta_twist != 0:
                raise ValueError('Not yet implemented')
            return len(self.modular_forms_basis(weight))

    ## bases of spaces associated to this representation

    def _eisenstein_packet(self, k, prec, dim = None, include_E = False):#packet of cusp forms that can be computed using only Eisenstein series
        j = floor((k - 1) / 2)
        if not dim:
            dim = j
        else:
            dim = min(dim, j)
        k_list = [k - (j + j) for j in range(dim)]
        E = self.eisenstein_series(k_list, prec)
        if include_E:
            X = [E[0]]
        else:
            X = []
        def repeated_serre_deriv(x, N):
            if N <= 0:
                return x
            return repeated_serre_deriv(x.serre_derivative(normalize_constant_term = True), N - 1)
        if len(k_list) > 1:
            e4 = eisenstein_series_qexp(4, prec, normalization='constant')
            if len(k_list) > 2:
                e6 = eisenstein_series_qexp(6, prec, normalization='constant')
        for c in range(dim // 3 + 1):
            if c:
                s6 = smf(6*c, e6 ** c)
            for b in range((dim - 3*c) // 2 + 1):
                if b:
                    s4 = smf(4*b, e4 ** b)
                j = dim - 3*c - 2*b
                for a in range(j):
                    p = a + 2*b + 3*c
                    if p:
                        u = E[a + 2*b + 3*c]
                        if b:
                            u = u * s4
                        if c:
                            u = u * s6
                        u = repeated_serre_deriv(u, a)
                        if include_E:
                            X.append(u)
                        else:
                            X.append(E[0] - u)
        b = WeilRepModularFormsBasis(k, X, self)
        if include_E:
            return b
        return E[0], b

    def cusp_forms_basis(self, k, prec=None, verbose = False, E = None, dim = None, save_pivots = False):#basis of cusp forms
        r"""
        Compute a basis of the space of cusp forms.

        ALGORITHM: If k is a symmetric weight, k >= 5/2, then we compute a basis from linear combinations of self's eisenstein_series() and pss(). If k is an antisymmetric weight, k >= 7/2, then we compute a basis from self's pssd(). Otherwise, we compute S_k as the intersection
        S_k(\rho^*) = E_4^(-1) * S_{k+4}(\rho^*) intersect E_6^(-1) * S_{k+6}(\rho^*). (This is slow!!)
        The basis is always converted to echelon form (i.e. a ``Victor Miller basis``)

        INPUT:
        - ``k`` -- the weight (half-integer)
        - ``prec`` -- precision (default None). If precision is not given then we use the Sturm bound.
        - ``verbose`` -- boolean (default False). If True then add comments throughout the computation.
        - ``E`` -- WeilRepModularForm (default None). If this is given then the computation assumes that E is the Eisenstein series of weight k.
        - ``dim`` -- (default None) If given then we stop computing after having found 'dim' vectors. (this is automatically minimized to the true dimension)
        - ``save_pivots`` -- boolean (default False) If True then we also output the pivots of each element of the basis' coefficient-vectors

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: W = WeilRep(matrix([[4,0],[0,4]]))
            sage: W.cusp_forms_basis(6, 5)
            [(0, 0), O(q^5)]
            [(1/4, 0), O(q^5)]
            [(1/2, 0), O(q^5)]
            [(3/4, 0), O(q^5)]
            [(0, 1/4), 6*q^(7/8) - 46*q^(15/8) + 114*q^(23/8) - 72*q^(31/8) + 42*q^(39/8) + O(q^5)]
            [(1/4, 1/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^5)]
            [(1/2, 1/4), q^(3/8) + 3*q^(11/8) - 75*q^(19/8) + 282*q^(27/8) - 276*q^(35/8) + O(q^5)]
            [(3/4, 1/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^5)]
            [(0, 1/2), O(q^5)]
            [(1/4, 1/2), O(q^5)]
            [(1/2, 1/2), O(q^5)]
            [(3/4, 1/2), O(q^5)]
            [(0, 3/4), -6*q^(7/8) + 46*q^(15/8) - 114*q^(23/8) + 72*q^(31/8) - 42*q^(39/8) + O(q^5)]
            [(1/4, 3/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^5)]
            [(1/2, 3/4), -q^(3/8) - 3*q^(11/8) + 75*q^(19/8) - 282*q^(27/8) + 276*q^(35/8) + O(q^5)]
            [(3/4, 3/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^5)]
            ------------------------------------------------------------
            [(0, 0), O(q^5)]
            [(1/4, 0), 6*q^(7/8) - 46*q^(15/8) + 114*q^(23/8) - 72*q^(31/8) + 42*q^(39/8) + O(q^5)]
            [(1/2, 0), O(q^5)]
            [(3/4, 0), -6*q^(7/8) + 46*q^(15/8) - 114*q^(23/8) + 72*q^(31/8) - 42*q^(39/8) + O(q^5)]
            [(0, 1/4), O(q^5)]
            [(1/4, 1/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^5)]
            [(1/2, 1/4), O(q^5)]
            [(3/4, 1/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^5)]
            [(0, 1/2), O(q^5)]
            [(1/4, 1/2), q^(3/8) + 3*q^(11/8) - 75*q^(19/8) + 282*q^(27/8) - 276*q^(35/8) + O(q^5)]
            [(1/2, 1/2), O(q^5)]
            [(3/4, 1/2), -q^(3/8) - 3*q^(11/8) + 75*q^(19/8) - 282*q^(27/8) + 276*q^(35/8) + O(q^5)]
            [(0, 3/4), O(q^5)]
            [(1/4, 3/4), -4*q^(3/4) + 24*q^(7/4) - 12*q^(11/4) - 184*q^(15/4) + 300*q^(19/4) + O(q^5)]
            [(1/2, 3/4), O(q^5)]
            [(3/4, 3/4), 4*q^(3/4) - 24*q^(7/4) + 12*q^(11/4) + 184*q^(15/4) - 300*q^(19/4) + O(q^5)]

            sage: W = WeilRep(matrix([[-12]]))
            sage: W.cusp_forms_basis(1/2, 20)
            [(0), O(q^20)]
            [(11/12), q^(1/24) - q^(25/24) - q^(49/24) + q^(121/24) + q^(169/24) - q^(289/24) - q^(361/24) + O(q^20)]
            [(5/6), O(q^20)]
            [(3/4), O(q^20)]
            [(2/3), O(q^20)]
            [(7/12), -q^(1/24) + q^(25/24) + q^(49/24) - q^(121/24) - q^(169/24) + q^(289/24) + q^(361/24) + O(q^20)]
            [(1/2), O(q^20)]
            [(5/12), -q^(1/24) + q^(25/24) + q^(49/24) - q^(121/24) - q^(169/24) + q^(289/24) + q^(361/24) + O(q^20)]
            [(1/3), O(q^20)]
            [(1/4), O(q^20)]
            [(1/6), O(q^20)]
            [(1/12), q^(1/24) - q^(25/24) - q^(49/24) + q^(121/24) + q^(169/24) - q^(289/24) - q^(361/24) + O(q^20)]
         """
        if k <= 0:
            return []
        S = self.gram_matrix()
        G = self.sorted_rds()
        if verbose:
            print('I am now looking for cusp forms for the Weil representation for the Gram matrix\n%s'%S)
        _norm_dict = self.norm_dict()
        symm = self.is_symmetric_weight(k)
        if symm is None:
            return [] #should this raise an error instead?
        if k > 2:
            true_dim = self.cusp_forms_dimension(k)
            if dim is None:
                dim = true_dim
            else:
                dim = min(dim, true_dim)
            if not dim:
                X = WeilRepModularFormsBasis(k, [], self)
                if save_pivots:
                    return X, []
                return X
            elif verbose:
                print('I need to find %d cusp forms of weight %s.' %(dim, k))
        sturm_bound = k / 12
        if not prec:
            prec = ceil(sturm_bound)
        else:
            prec = ceil(max(prec, sturm_bound))
        if k >= 7/2 or (k >= 5/2 and symm):
            if symm:
                if not E:
                    E = self.eisenstein_series(k, prec)
                    if verbose:
                        print('I computed the Eisenstein series of weight %s up to precision %s.' %(k, prec))
            rank = 0
            if symm and k >= 9/2:
                E, X = self._eisenstein_packet(k, prec, dim = dim+1)
                if verbose and X:
                    print('I computed a packet of %d cusp forms using Eisenstein series.'%len(X))
            elif symm and not E:
                E = self.eisenstein_series(k, prec)
                X = WeilRepModularFormsBasis(k, [], self)
                if verbose:
                    print('I computed the Eisenstein series of weight %s up to precision %s.' %(k, prec))
            else:
                X = WeilRepModularFormsBasis(k, [], self)
            #use PSS to finish spanning
            if verbose and rank > 0:
                print('I found %d cusp forms by subtracting the Eisenstein series away from Serre derivatives of Eisenstein series of lower weight.' %rank)
            m0 = 1
            skipped_indices = []
            failed_exponent = 0
            while rank < dim:
                while len(X) < dim:
                    for b_tuple in G:
                        old_rank = rank
                        b = vector(b_tuple)
                        if symm or b.denominator() > 2:
                            m = m0 + _norm_dict[b_tuple]
                            if m != failed_exponent or m >= sturm_bound:
                                dim_rank = dim - len(X)
                                if symm:
                                    if k in ZZ:
                                        w_new = self.embiggen(b, m)
                                        if k > 3 and dim_rank > 2:
                                            if verbose:
                                                print('-'*40)
                                            y = w_new.cusp_forms_basis(k - 1/2, prec, verbose = verbose, dim = dim_rank).theta()
                                            X.extend(y)
                                            if verbose:
                                                print('-'*40)
                                                print('I computed %d cusp forms using a basis of cusp forms from the index %s.'%(len(y), (b, m)))
                                                print('I am returning to the Gram matrix\n%s'%S)
                                        if len(X) < dim:
                                            X.append(E - self.pss(k, b, m, prec, weilrep = w_new))
                                            if verbose:
                                                print('I computed a Poincare square series of index %s.'%([b, m]))
                                    else:
                                        w_new = self.embiggen(b, m)
                                        if dim_rank > 1 and k > 9/2:
                                            _, x = w_new._eisenstein_packet(k - 1/2, prec, dim = dim_rank)
                                            X.extend(x.theta())
                                            if x and verbose:
                                                print('I computed a packet of %d cusp forms using the index %s.'%(len(x), (b, m)))
                                        if len(X) < dim:
                                            X.append(E - self.pss(k, b, m, prec, weilrep = w_new))
                                            if verbose:
                                                print('I computed a Poincare square series of index %s.'%([b, m]))
                                else:
                                    if k in ZZ:
                                        w_new = self.embiggen(b, m)
                                        if dim_rank > 1 and k > 4:
                                            if verbose:
                                                print('-'*40)
                                            y = w_new.cusp_forms_basis(k - 3/2, prec, verbose = verbose, dim = dim - len(X)).theta(odd = True)
                                            X.extend(y)
                                            if verbose:
                                                print('-'*40)
                                                print('I computed %d cusp forms using a basis of cusp forms from the index %s.'%(len(y), (b, m)))
                                                print('I am returning to the Gram matrix\n%s'%S)
                                        if len(X) < dim:
                                            X.append(self.pssd(k, b, m, prec, weilrep = w_new))
                                            if verbose:
                                                print('I computed a Poincare square series of index %s.'%([b, m]))
                                    else:
                                        dim_rank = dim-len(X)
                                        if dim_rank > 1 and k > 6:
                                            w_new = self.embiggen(b, m)
                                            y = w_new._eisenstein_packet(k - 3/2, prec, dim = dim_rank, include_E = true).theta(odd = True)
                                            X.extend(y)
                                            if verbose:
                                                print('I computed a packet of %d cusp forms using the index %s.'%(len(y), (b, m)))
                                        else:
                                            X.append(self.pssd(k, b, m, prec))
                                            if verbose:
                                                print('I computed a Poincare square series of index %s.'%([b, m]))
                                pivots = X.echelonize(save_pivots)
                                rank = len(X)
                                if rank >= dim:
                                    break
                                if rank == old_rank:
                                    failed_exponent = m
                                elif verbose:
                                    print('I have found %d out of %d cusp forms.'%(rank, dim))
                            else:
                                if verbose:
                                    print('I will skip the index %s.'%([b,m]))
                                skipped_indices.append([b, m])
                    m0 += 1
                    if m0 > prec:#this will probably never happen but lets be safe
                        for [b, m] in skipped_indices:
                            if symm:
                                X.append(E - self.pss(k, b, m, prec))
                            else:
                                X.append(self.pssd(k, b, m, prec))
                            pivots = X.echelonize(save_pivots)
                            rank = len(X)
                            if rank >= dim:
                                break
                        if rank < dim:
                            raise RuntimeError('Something went horribly wrong!')
                if rank < dim:
                    pivots = X.echelonize(save_pivots)
                    rank = len(X)
            if save_pivots:
                return X, pivots
            return X
        else:#slow
            p = self.discriminant()
            if symm and p.is_prime() and p != 2:
                if verbose:
                    print('The discriminant is prime so I can construct cusp forms via the Bruinier--Bundschuh lift.')
                chi = DirichletGroup(p)[(p-1)//2]
                cusp_forms = CuspForms(chi, k, prec = p*prec).echelon_basis()
                mod_sturm_bound = ceil(p * k / 12)
                sig = self.signature()
                eps = sig == 0 or sig == 6
                eps = 1 - 2 * eps
                m = matrix([[y for i, y in enumerate(x.coefficients(mod_sturm_bound)) if kronecker(i + 1, p) == eps] for x in cusp_forms])
                v_basis = m.kernel().basis()
                L = [sum([mf * v[i] for i, mf in enumerate(cusp_forms)]) for v in v_basis]
                L = [2*self.bb_lift(x) if x.valuation() % p else self.bb_lift(x) for x in L]
                X = WeilRepModularFormsBasis(k, L, self)
                pivots = X.echelonize(save_pivots)
                if save_pivots:
                    return X, pivots
                return X
            if verbose:
                print('I am going to compute the spaces of cusp forms of weights %s and %s.' %(k+4, k+6))
            e4 = smf(-4, ~eisenstein_series_qexp(4, prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6, prec))
            X1 = self.cusp_forms_basis(k + 4, prec, verbose = verbose)
            X2 = self.cusp_forms_basis(k + 6, prec, verbose = verbose)
            if verbose:
                print('I am now going to compute S_%s by intersecting the spaces E_4^(-1) * S_%s and E_6^(-1) * S_%s.' %(k, k +4, k +6))
            try:
                V1 = span((x * e4).coefficient_vector() for x in X1)
                V2 = span((x * e6).coefficient_vector() for x in X2)
                X = WeilRepModularFormsBasis(k, [self.recover_modular_form_from_coefficient_vector(k, v, prec) for v in V1.intersection(V2).basis()], self)
                pivots = X.echelonize(save_pivots)
                if save_pivots:
                    return X, pivots
                return X
            except AttributeError: #we SHOULD only get ``AttributeError: 'Objects_with_category' object has no attribute 'base_ring'`` when X1 or X2 is empty...
                return []

    def modular_forms_basis(self, weight, prec = 0, eisenstein = False, verbose = False):
        r"""
        Compute a basis of the space of modular forms.

        ALGORITHM: If k is a symmetric weight, k >= 5/2, then we compute a basis from linear combinations of self's eisenstein_series() and pss(). If k is an antisymmetric weight, k >= 7/2, then we compute a basis from self's pssd(). Otherwise, we compute M_k as the intersection
        M_k(\rho^*) = E_4^(-1) * M_{k+4}(\rho^*) intersect E_6^(-1) * M_{k+6}(\rho^*). (This is slow!!)
        Note: Eisenstein series at nonzero cusps are not implemented yet so when the Eisenstein space has dim > 1 we instead compute the image in S_{k+12} of Delta * M_k. (This is even slower!!)
        The basis is always converted to echelon form (i.e. a ``Victor Miller basis``)

        INPUT:
        - ``k`` -- the weight (half-integer)
        - ``prec`` -- precision (default None). If precision is not given then we use the Sturm bound.
        - ``eisenstein`` -- boolean (default False). If True and weight >= 5/2 then the first element in the output is always the Eisenstein series (i.e. we do not pass to echelon form).
        - ``verbose`` -- boolean (default False). If true then we add comments throughout the computation.

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: w = WeilRep(matrix([[2,0],[0,4]]))
            sage: w.modular_forms_basis(5, 10)
            [(0, 0), 1 - 150*q - 2270*q^2 - 11820*q^3 - 36750*q^4 - 89888*q^5 - 188380*q^6 - 344640*q^7 - 589230*q^8 - 954210*q^9 + O(q^10)]
            [(0, 1/4), -80*q^(7/8) - 1808*q^(15/8) - 9840*q^(23/8) - 32320*q^(31/8) - 82160*q^(39/8) - 171360*q^(47/8) - 320528*q^(55/8) - 559600*q^(63/8) - 891600*q^(71/8) - 1365920*q^(79/8) + O(q^10)]
            [(0, 1/2), -10*q^(1/2) - 740*q^(3/2) - 5568*q^(5/2) - 21760*q^(7/2) - 59390*q^(9/2) - 130980*q^(11/2) - 257600*q^(13/2) - 461056*q^(15/2) - 747540*q^(17/2) - 1166180*q^(19/2) + O(q^10)]
            [(0, 3/4), -80*q^(7/8) - 1808*q^(15/8) - 9840*q^(23/8) - 32320*q^(31/8) - 82160*q^(39/8) - 171360*q^(47/8) - 320528*q^(55/8) - 559600*q^(63/8) - 891600*q^(71/8) - 1365920*q^(79/8) + O(q^10)]
            [(1/2, 0), -40*q^(3/4) - 1440*q^(7/4) - 7720*q^(11/4) - 30496*q^(15/4) - 68520*q^(19/4) - 166880*q^(23/4) - 283600*q^(27/4) - 551040*q^(31/4) - 787200*q^(35/4) - 1396960*q^(39/4) + O(q^10)]
            [(1/2, 1/4), -24*q^(5/8) - 1000*q^(13/8) - 6880*q^(21/8) - 24840*q^(29/8) - 65880*q^(37/8) - 145352*q^(45/8) - 276600*q^(53/8) - 485960*q^(61/8) - 805280*q^(69/8) - 1233120*q^(77/8) + O(q^10)]
            [(1/2, 1/2), -368*q^(5/4) - 3520*q^(9/4) - 17040*q^(13/4) - 43840*q^(17/4) - 117440*q^(21/4) - 205440*q^(25/4) - 421840*q^(29/4) - 632000*q^(33/4) - 1117680*q^(37/4) + O(q^10)]
            [(1/2, 3/4), -24*q^(5/8) - 1000*q^(13/8) - 6880*q^(21/8) - 24840*q^(29/8) - 65880*q^(37/8) - 145352*q^(45/8) - 276600*q^(53/8) - 485960*q^(61/8) - 805280*q^(69/8) - 1233120*q^(77/8) + O(q^10)]
            ------------------------------------------------------------
            [(0, 0), -12*q + 56*q^2 - 72*q^3 + 80*q^4 - 352*q^5 + 336*q^6 + 704*q^7 - 1056*q^8 + 540*q^9 + O(q^10)]
            [(0, 1/4), 8*q^(7/8) - 24*q^(15/8) - 40*q^(23/8) + 160*q^(31/8) + 24*q^(39/8) - 272*q^(47/8) + 104*q^(55/8) - 360*q^(63/8) + 72*q^(71/8) + 1424*q^(79/8) + O(q^10)]
            [(0, 1/2), -2*q^(1/2) - 12*q^(3/2) + 112*q^(5/2) - 224*q^(7/2) + 90*q^(9/2) + 52*q^(11/2) - 112*q^(13/2) + 672*q^(15/2) - 452*q^(17/2) - 268*q^(19/2) + O(q^10)]
            [(0, 3/4), 8*q^(7/8) - 24*q^(15/8) - 40*q^(23/8) + 160*q^(31/8) + 24*q^(39/8) - 272*q^(47/8) + 104*q^(55/8) - 360*q^(63/8) + 72*q^(71/8) + 1424*q^(79/8) + O(q^10)]
            [(1/2, 0), 6*q^(3/4) - 16*q^(7/4) - 26*q^(11/4) + 48*q^(15/4) + 134*q^(19/4) + 80*q^(23/4) - 756*q^(27/4) - 320*q^(31/4) + 1920*q^(35/4) - 48*q^(39/4) + O(q^10)]
            [(1/2, 1/4), -4*q^(5/8) + 4*q^(13/8) + 48*q^(21/8) - 44*q^(29/8) - 228*q^(37/8) + 180*q^(45/8) + 492*q^(53/8) - 268*q^(61/8) - 240*q^(69/8) - 208*q^(77/8) + O(q^10)]
            [(1/2, 1/2), q^(1/4) + 8*q^(5/4) - 45*q^(9/4) - 8*q^(13/4) + 226*q^(17/4) - 96*q^(21/4) - 335*q^(25/4) + 88*q^(29/4) - 156*q^(33/4) + 456*q^(37/4) + O(q^10)]
            [(1/2, 3/4), -4*q^(5/8) + 4*q^(13/8) + 48*q^(21/8) - 44*q^(29/8) - 228*q^(37/8) + 180*q^(45/8) + 492*q^(53/8) - 268*q^(61/8) - 240*q^(69/8) - 208*q^(77/8) + O(q^10)]
        """
        symm = self.is_symmetric_weight(weight)
        _ds = self.ds()
        _indices = self.rds(indices = True)
        _norm_list = self.norm_list()
        sturm_bound = weight / 12
        prec = max(prec, ceil(sturm_bound))
        b_list = [i for i in range(len(_ds)) if not (_indices[i] or _norm_list[i]) and (self.__ds_denominators_list[i] < 5 or self.__ds_denominators_list[i] == 6)]
        if weight > 3 or (symm and weight > 2):
            dim1 = self.modular_forms_dimension(weight)
            dim2 = self.cusp_forms_dimension(weight)
            if verbose:
                print('I need to find %d modular forms of weight %s.' %(dim1, weight))
            if (symm and dim1 <= dim2 + len(b_list)):
                if verbose:
                    print('I found %d Eisenstein series.' %len(b_list))
                    if dim2 > 0:
                        print('I am now going to look for %d cusp forms of weight %s.' %(dim2, weight))
                L = WeilRepModularFormsBasis(weight, [self.eisenstein_oldform(weight, _ds[i], prec) for i in b_list], self)
                if eisenstein:
                    L.extend(self.cusp_forms_basis(weight, prec, verbose = verbose, E = L0))
                    return L
                else:
                    X = self.cusp_forms_basis(weight, prec, verbose = verbose, E = L[0])
                    X.extend(L)
                    X.echelonize()
                    return X
            elif dim1 == dim2:
                return self.cusp_forms_basis(weight, prec, verbose = verbose)
            else:
                pass
        p = self.discriminant()
        if symm and p.is_prime() and p != 2:
            if weight == 0:
                return []
            if verbose:
                print('The discriminant is prime so I can construct modular forms via the Bruinier--Bundschuh lift.')
            chi = DirichletGroup(p)[(p-1)//2]
            mod_forms = ModularForms(chi, weight, prec = p*prec).echelon_basis()
            mod_sturm_bound = p * ceil(weight / 12)
            sig = self.signature()
            if (sig == 0 or sig == 6):
                eps = -1
            else:
                eps = 1
            m = matrix([[y for i, y in enumerate(x.coefficients(mod_sturm_bound)) if kronecker(i + 1, p) == eps] for x in mod_forms])
            v_basis = m.kernel().basis()
            L = [sum([mf * v[i] for i, mf in enumerate(mod_forms)]) for v in v_basis]
            L = [2*self.bb_lift(x) if x.valuation() % p else self.bb_lift(x) for x in L]
            return WeilRepModularFormsBasis(weight, L, self)
        dim1 = self.modular_forms_dimension(weight+4)
        dim2 = self.cusp_forms_dimension(weight+4)
        if symm and (dim1 <= dim2 + len(b_list)):
            if verbose:
                print('I am going to compute the spaces of modular forms of weights %s and %s.' %(weight+4, weight+6))
            e4 = smf(-4, ~eisenstein_series_qexp(4,prec))
            e6 = smf(-6, ~eisenstein_series_qexp(6,prec))
            X1 = self.modular_forms_basis(weight+4,prec, verbose = verbose)
            X2 = self.modular_forms_basis(weight+6,prec, verbose = verbose)
            if verbose:
                print('I am now going to compute M_%s by intersecting the spaces E_4^(-1) * M_%s and E_6^(-1) * M_%s.' %(weight, weight +4, weight +6))
            try:
                V1 = span([(x * e4).coefficient_vector() for x in X1])
                V2 = span([(x * e6).coefficient_vector() for x in X2])
                V = (V1.intersection(V2)).echelonized_basis()
                return WeilRepModularFormsBasis(weight, [self.recover_modular_form_from_coefficient_vector(weight, v, prec) for v in V], self)
            except AttributeError:
                return []
        else:
            if verbose:
                print('I do not know how to find enough Eisenstein series. I am going to compute the image of M_%s under multiplication by Delta.' %weight)
            return self.nearly_holomorphic_modular_forms_basis(weight, 0, prec, inclusive = True, reverse = False, force_N_positive = True, verbose = verbose)

    def basis_vanishing_to_order(self, k, N=0, prec=0, inclusive = False,  inclusive_except_zero_component = False, keep_N = False, verbose = False):
        r"""
        Compute bases of modular forms that vanish to a specified order at infinity.

        ALGORITHM: We first try to reduce to lower weight by dividing by a power of the modular Delta function. Then compute the full cusp space and pick out the forms with given vanishing order. (This is easy because cusp_forms_basis computes an echelon form.)

        INPUT:
        - ``k`` -- the weight
        - ``N`` -- the minimum order of vanishing (default 0)
        - ``prec`` -- the precision (default 0); will be raised to at least the Sturm bound
        - ``inclusive`` -- boolean (default False); if True then we also exclude modular forms whose order of vanishing is *exactly* N
        - ``inclusive_except_zero_component`` -- boolean (default False); if True then we exclude modular forms any of whose components (except the zero component) has order of vanishing exactly N
        - ``keep_N`` -- boolean (default False); if True then we skip the first step of trying to reduce to lower weight (for internal use)

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: w = WeilRep(matrix([[-6]]))
            sage: w.basis_vanishing_to_order(21/2, 3/4, 10)
            [(0), -2*q + 24*q^2 - 56*q^3 - 588*q^4 + 4740*q^5 - 13680*q^6 + 8464*q^7 + 51768*q^8 - 137754*q^9 + O(q^10)]
            [(5/6), 3*q^(13/12) - 45*q^(25/12) + 255*q^(37/12) - 519*q^(49/12) - 879*q^(61/12) + 5916*q^(73/12) - 5610*q^(85/12) - 18123*q^(97/12) + 34017*q^(109/12) + O(q^10)]
            [(2/3), -6*q^(4/3) + 102*q^(7/3) - 720*q^(10/3) + 2568*q^(13/3) - 3876*q^(16/3) - 3246*q^(19/3) + 20904*q^(22/3) - 22440*q^(25/3) - 4692*q^(28/3) + O(q^10)]
            [(1/2), q^(3/4) - 10*q^(7/4) + 3*q^(11/4) + 360*q^(15/4) - 1783*q^(19/4) + 1716*q^(23/4) + 11286*q^(27/4) - 35466*q^(31/4) + 1080*q^(35/4) + 148662*q^(39/4) + O(q^10)]
            [(1/3), -6*q^(4/3) + 102*q^(7/3) - 720*q^(10/3) + 2568*q^(13/3) - 3876*q^(16/3) - 3246*q^(19/3) + 20904*q^(22/3) - 22440*q^(25/3) - 4692*q^(28/3) + O(q^10)]
            [(1/6), 3*q^(13/12) - 45*q^(25/12) + 255*q^(37/12) - 519*q^(49/12) - 879*q^(61/12) + 5916*q^(73/12) - 5610*q^(85/12) - 18123*q^(97/12) + 34017*q^(109/12) + O(q^10)]

            sage: w = WeilRep(matrix([[0,2],[2,0]]))
            sage: w.basis_vanishing_to_order(8, 1/2, 10)
            [(0, 0), 8*q - 64*q^2 + 96*q^3 + 512*q^4 - 1680*q^5 - 768*q^6 + 8128*q^7 - 4096*q^8 - 16344*q^9 + O(q^10)]
            [(0, 1/2), -8*q + 64*q^2 - 96*q^3 - 512*q^4 + 1680*q^5 + 768*q^6 - 8128*q^7 + 4096*q^8 + 16344*q^9 + O(q^10)]
            [(1/2, 0), -8*q + 64*q^2 - 96*q^3 - 512*q^4 + 1680*q^5 + 768*q^6 - 8128*q^7 + 4096*q^8 + 16344*q^9 + O(q^10)]
            [(1/2, 1/2), q^(1/2) + 12*q^(3/2) - 210*q^(5/2) + 1016*q^(7/2) - 2043*q^(9/2) + 1092*q^(11/2) + 1382*q^(13/2) - 2520*q^(15/2) + 14706*q^(17/2) - 39940*q^(19/2) + O(q^10)]
        """
        if verbose:
            print('I am now looking for modular forms of weight %s which vanish to order %s at infinity.' %(k, N))
        if inclusive and inclusive_except_zero_component:
            raise ValueError('At most one of "inclusive" and "inclusive_except_zero_component" may be true')
        symm = self.is_symmetric_weight(k)
        if symm is None:
            raise ValueError('Invalid weight')
        sturm_bound = k/12
        prec = ceil(max(prec,sturm_bound))
        if N > sturm_bound:
            return []
        elif N == 0:
            if inclusive:
                if verbose:
                    print('The vanishing condition is trivial so I am looking for all cusp forms.')
                return self.cusp_forms_basis(k, prec, verbose = verbose)
            else:
                if verbose:
                    print('The vanishing condition is trivial so I am looking for all modular forms.')
                return self.modular_forms_basis(k, prec, eisenstein = False, verbose = verbose)
        elif N >= 1 and not keep_N:
            frac_N = frac(N)
            floor_N = floor(N)
            computed_weight = k - 12*floor_N
            if computed_weight <= 2:
                frac_N += 1
                floor_N -= 1
            if frac_N < N:
                smf_delta_N = smf(12*floor_N, delta_qexp(prec)^floor_N)
                if verbose:
                    print('I am going to find a basis of modular forms of weight %s which vanish to order %s at infinity and multiply them by Delta^%d.' %(computed_weight, frac_N, floor_N))
                X = self.basis_vanishing_to_order(computed_weight, frac_N, prec, inclusive, verbose = verbose)
                return WeilRepModularFormsBasis(k, [x * smf_delta_N for x in X], self)
        cusp_forms, pivots = self.cusp_forms_basis(k, prec, verbose = verbose, save_pivots = True)
        Y = self.coefficient_vector_exponents(prec, symm, include_vectors = inclusive_except_zero_component)
        try:
            if inclusive:
                j = next(i for i in range(len(cusp_forms)) if Y[pivots[i]] > N)
            elif inclusive_except_zero_component:
                j = next(i for i in range(len(cusp_forms)) if (Y[0][pivots[i]] >= N) and (Y[0][pivots[i]] > N or not Y[1][pivots[i]]))
            else:
                j = next(i for i in range(len(cusp_forms)) if Y[pivots[i]] >= N)
        except:
            return []
        Z = cusp_forms[j:]
        if type(Z) is list:
            return WeilRepModularFormsBasis(k, Z, self)
        return Z

    def nearly_holomorphic_modular_forms_basis(self, k, pole_order, prec = 0, inclusive = True, reverse = True, force_N_positive = False, verbose = False):
        r"""
        Computes a basis of nearly holomorphic modular forms.

        A nearly-holomorphic modular form is a function f : H -> C[L' / L] that is holomorphic on H and meromorphic at cusps and satisfies the usual transformations. In other words they are allowed to have a finite principal part.

        ALGORITHM: compute spaces of modular forms of higher weight and divide by Delta.

        INPUT:
        - ``k`` -- the weight
        - ``pole_order`` -- the worst pole the modular forms are allowed to have
        - ``prec`` -- precision (default 0); will be raised at least to a Sturm bound
        - ``inclusive`` -- boolean (default True); if True then we allow forms with pole order *exactly* pole_order
        - ``reverse`` -- boolean (default True); if True then output forms in reverse echelon order
        - ``force_N_positive`` -- boolean (default False); if True then we always divide by Delta at least once in the computation. (for internal use)

        OUTPUT: a list of WeilRepModularForms

        EXAMPLES::

            sage: w = WeilRep(matrix([[0,2],[2,0]]))
            sage: w.nearly_holomorphic_modular_forms_basis(0, 1, 10)
            [(0, 0), O(q^10)]
            [(0, 1/2), 1 + O(q^10)]
            [(1/2, 0), -1 + O(q^10)]
            [(1/2, 1/2), O(q^10)]
            ------------------------------------------------------------
            [(0, 0), 1 + O(q^10)]
            [(0, 1/2), O(q^10)]
            [(1/2, 0), 1 + O(q^10)]
            [(1/2, 1/2), O(q^10)]
            ------------------------------------------------------------
            [(0, 0), 2048*q + 49152*q^2 + 614400*q^3 + 5373952*q^4 + 37122048*q^5 + 216072192*q^6 + 1102430208*q^7 + 5061476352*q^8 + 21301241856*q^9 + O(q^10)]
            [(0, 1/2), -2048*q - 49152*q^2 - 614400*q^3 - 5373952*q^4 - 37122048*q^5 - 216072192*q^6 - 1102430208*q^7 - 5061476352*q^8 - 21301241856*q^9 + O(q^10)]
            [(1/2, 0), -24 - 2048*q - 49152*q^2 - 614400*q^3 - 5373952*q^4 - 37122048*q^5 - 216072192*q^6 - 1102430208*q^7 - 5061476352*q^8 - 21301241856*q^9 + O(q^10)]
            [(1/2, 1/2), q^(-1/2) + 276*q^(1/2) + 11202*q^(3/2) + 184024*q^(5/2) + 1881471*q^(7/2) + 14478180*q^(9/2) + 91231550*q^(11/2) + 495248952*q^(13/2) + 2390434947*q^(15/2) + 10487167336*q^(17/2) + 42481784514*q^(19/2) + O(q^10)]
            ------------------------------------------------------------
            [(0, 0), 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(0, 1/2), -98304*q - 10747904*q^2 - 432144384*q^3 - 10122952704*q^4 - 166601228288*q^5 - 2126011957248*q^6 - 22328496095232*q^7 - 200745446014976*q^8 - 1588220107653120*q^9 + O(q^10)]
            [(1/2, 0), q^-1 - 24 + 98580*q + 10745856*q^2 + 432155586*q^3 + 10122903552*q^4 + 166601412312*q^5 + 2126011342848*q^6 + 22328497976703*q^7 + 200745440641024*q^8 + 1588220122131300*q^9 + O(q^10)]
            [(1/2, 1/2), -4096*q^(1/2) - 1228800*q^(3/2) - 74244096*q^(5/2) - 2204860416*q^(7/2) - 42602483712*q^(9/2) - 611708977152*q^(11/2) - 7039930359808*q^(13/2) - 68131864608768*q^(15/2) - 572940027371520*q^(17/2) - 4286110556078080*q^(19/2) + O(q^10)]
            ------------------------------------------------------------
            [(0, 0), 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(0, 1/2), q^-1 + 98580*q + 10745856*q^2 + 432155586*q^3 + 10122903552*q^4 + 166601412312*q^5 + 2126011342848*q^6 + 22328497976703*q^7 + 200745440641024*q^8 + 1588220122131300*q^9 + O(q^10)]
            [(1/2, 0), -24 - 98304*q - 10747904*q^2 - 432144384*q^3 - 10122952704*q^4 - 166601228288*q^5 - 2126011957248*q^6 - 22328496095232*q^7 - 200745446014976*q^8 - 1588220107653120*q^9 + O(q^10)]
            [(1/2, 1/2), -4096*q^(1/2) - 1228800*q^(3/2) - 74244096*q^(5/2) - 2204860416*q^(7/2) - 42602483712*q^(9/2) - 611708977152*q^(11/2) - 7039930359808*q^(13/2) - 68131864608768*q^(15/2) - 572940027371520*q^(17/2) - 4286110556078080*q^(19/2) + O(q^10)]
            ------------------------------------------------------------
            [(0, 0), q^-1 + 98580*q + 10745856*q^2 + 432155586*q^3 + 10122903552*q^4 + 166601412312*q^5 + 2126011342848*q^6 + 22328497976703*q^7 + 200745440641024*q^8 + 1588220122131300*q^9 + O(q^10)]
            [(0, 1/2), 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(1/2, 0), 24 + 98304*q + 10747904*q^2 + 432144384*q^3 + 10122952704*q^4 + 166601228288*q^5 + 2126011957248*q^6 + 22328496095232*q^7 + 200745446014976*q^8 + 1588220107653120*q^9 + O(q^10)]
            [(1/2, 1/2), 4096*q^(1/2) + 1228800*q^(3/2) + 74244096*q^(5/2) + 2204860416*q^(7/2) + 42602483712*q^(9/2) + 611708977152*q^(11/2) + 7039930359808*q^(13/2) + 68131864608768*q^(15/2) + 572940027371520*q^(17/2) + 4286110556078080*q^(19/2) + O(q^10)]

            sage: w = WeilRep(matrix([[2,1],[1,2]]))
            sage: w.nearly_holomorphic_modular_forms_basis(0, 1, 10)
            [(0, 0), O(q^10)]
            [(2/3, 2/3), q^(-1/3) + 248*q^(2/3) + 4124*q^(5/3) + 34752*q^(8/3) + 213126*q^(11/3) + 1057504*q^(14/3) + 4530744*q^(17/3) + 17333248*q^(20/3) + 60655377*q^(23/3) + 197230000*q^(26/3) + 603096260*q^(29/3) + O(q^10)]
            [(1/3, 1/3), -q^(-1/3) - 248*q^(2/3) - 4124*q^(5/3) - 34752*q^(8/3) - 213126*q^(11/3) - 1057504*q^(14/3) - 4530744*q^(17/3) - 17333248*q^(20/3) - 60655377*q^(23/3) - 197230000*q^(26/3) - 603096260*q^(29/3) + O(q^10)]

            sage: w = WeilRep(matrix([[4]]))
            sage: w.nearly_holomorphic_modular_forms_basis(5/2, 1, 7)
            [(0), O(q^7)]
            [(1/4), q^(-1/8) + 243*q^(7/8) + 2889*q^(15/8) + 15382*q^(23/8) + 62451*q^(31/8) + 203148*q^(39/8) + 593021*q^(47/8) + 1551069*q^(55/8) + O(q^7)]
            [(1/2), O(q^7)]
            [(3/4), -q^(-1/8) - 243*q^(7/8) - 2889*q^(15/8) - 15382*q^(23/8) - 62451*q^(31/8) - 203148*q^(39/8) - 593021*q^(47/8) - 1551069*q^(55/8) + O(q^7)]
        """
        if verbose:
            print('I am now looking for modular forms of weight %s which are holomorphic on H and have a pole of order at most %s in infinity.' %(k, pole_order))
        sturm_bound = k/12
        if sturm_bound > prec:
            raise ValueError('Low precision')
        dual_sturm_bound = 1/6 - sturm_bound
        symm = self.is_symmetric_weight(k)
        if pole_order >= dual_sturm_bound + 2:
            if verbose:
                print('The pole order is large so I will compute modular forms with a smaller pole order and multiply them by the j-invariant.')
            j_order = floor(pole_order - dual_sturm_bound - 1)
            new_pole_order = pole_order - j_order
            X = self.nearly_holomorphic_modular_forms_basis(k, new_pole_order, prec = prec + j_order + 1, inclusive = inclusive, reverse = reverse, force_N_positive = force_N_positive, verbose = verbose)
            j = j_invariant_qexp(prec + j_order + 1) - 744
            j = [smf(0, j^n) for n in range(1, j_order + 1)]
            Y = copy(X)
            Y.extend([x * j[n] for n in range(j_order) for x in X])
            for y in Y:
                y.reduce_precision(prec)
            Y.echelonize(starting_from = -pole_order)
            if reverse:
                Y.reverse()
            return Y
        ceil_pole_order = ceil(pole_order)
        computed_weight = k + 12*ceil_pole_order
        N = ceil_pole_order
        while computed_weight < 7/2 or (symm and computed_weight < 5/2):
            computed_weight += 12
            N += 1
        if force_N_positive and N <= pole_order:
            N += 1
            computed_weight += 12
        prec = ceil(max(prec, (computed_weight / 12)))
        if verbose:
            print('I am going to compute modular forms of weight %s which vanish in infinity to order %s and divide them by Delta^%d.' %(computed_weight, N - pole_order, N))
        X = self.basis_vanishing_to_order(computed_weight, N - pole_order, prec + N, not inclusive, keep_N = True, verbose = verbose)
        delta_power = smf(-12 * N, ~(delta_qexp(prec + N + 1) ** N))
        Y = WeilRepModularFormsBasis(k, [x * delta_power for x in X], self)
        Y.echelonize(starting_from = -N, ending_with = sturm_bound)
        if reverse:
            Y.reverse()
        return Y

    weakly_holomorphic_modular_forms_basis = nearly_holomorphic_modular_forms_basis

    def borcherds_obstructions(self, weight, prec, reverse = True, verbose = False):
        r"""
        Compute a basis of the Borcherds obstruction space.

        EXAMPLES::

            sage: w = WeilRep(matrix([[-8]]))
            sage: w.borcherds_obstructions(5/2, 5)
            [(0), 1 - 24*q - 72*q^2 - 96*q^3 - 358*q^4 + O(q^5)]
            [(7/8), -1/2*q^(1/16) - 24*q^(17/16) - 72*q^(33/16) - 337/2*q^(49/16) - 192*q^(65/16) + O(q^5)]
            [(3/4), -5*q^(1/4) - 24*q^(5/4) - 125*q^(9/4) - 120*q^(13/4) - 240*q^(17/4) + O(q^5)]
            [(5/8), -25/2*q^(9/16) - 121/2*q^(25/16) - 96*q^(41/16) - 168*q^(57/16) - 264*q^(73/16) + O(q^5)]
            [(1/2), -46*q - 48*q^2 - 144*q^3 - 192*q^4 + O(q^5)]
            [(3/8), -25/2*q^(9/16) - 121/2*q^(25/16) - 96*q^(41/16) - 168*q^(57/16) - 264*q^(73/16) + O(q^5)]
            [(1/4), -5*q^(1/4) - 24*q^(5/4) - 125*q^(9/4) - 120*q^(13/4) - 240*q^(17/4) + O(q^5)]
            [(1/8), -1/2*q^(1/16) - 24*q^(17/16) - 72*q^(33/16) - 337/2*q^(49/16) - 192*q^(65/16) + O(q^5)]
        """
        prec = ceil(prec)
        if weight > 2 or (weight == 2 and not self.discriminant().is_squarefree()):
            if verbose:
                print('I am looking for obstructions to Borcherds products of weight %s.' %weight)
            E = self.eisenstein_series(weight, prec)
            if verbose:
                print('I computed the Eisenstein series and will now compute cusp forms.')
            L = [E]
            L.extend(self.cusp_forms_basis(weight, prec, verbose = verbose))
            return WeilRepModularFormsBasis(weight, L, self)
        else:
            if verbose:
                print('I am going to compute the obstruction spaces in weights %s and %s.' %(weight+4, weight+6))
            e4 = smf(-4,eisenstein_series_qexp(4,prec)^(-1))
            e6 = smf(-6,eisenstein_series_qexp(6,prec)^(-1))
            X1 = self.borcherds_obstructions(weight+4, prec, verbose = verbose)
            X2 = self.borcherds_obstructions(weight+6, prec, verbose = verbose)
            try:
                V1 = span([(x * e4).coefficient_vector() for x in X1])
                V2 = span([(x * e6).coefficient_vector() for x in X2])
                V = (V1.intersection(V2)).echelonized_basis()
                Y = [reconstruct_VVMF_from_coefficient_vector(weight, S, v, 0, prec) for v in V]
                if reverse:
                    Y.reverse()
                return WeilRepModularFormsBasis(weight, Y, self)
            except AttributeError:
                return []

weilrep = WeilRep