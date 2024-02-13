import numpy, math
import numpy as np
import nutils.topology
from nutils.expression_v2 import Namespace
from nutils import function, mesh, solver, numeric, export, types, sparse
from nutils import element, function, evaluable, _util as util, parallel, numeric, cache, transform, transformseq, warnings, types, points, sparse
from nutils._util import single_or_multiple
from functools import cached_property
from nutils.elementseq import References
from nutils.pointsseq import PointsSequence
from nutils.sample import Sample
import tools_nutils
import time
import scipy



def BernGramMatrix(p):
    # given splines degree p, return inverted Grammian Matrix for the Bernstein polynomials of degree p
    G = numpy.zeros((p+1,p+1))
    for j in range(p+1):
        for k in range(p+1):
            intersum = 0
            for i in range(min(j,k)+1):
                intersum += ( 2 * i + 1) * math.comb( p - i, p - j ) * math.comb( p - i , p - k ) * math.comb( p + i + 1, p - j) * math.comb( p + i + 1, p - k )
            G[j,k] = (-1)**(j+k) * intersum / (2 * math.comb( p, j) * math.comb(p, k))
            # G[ii,jj] = 1/(2 * p + 1) * math.comb(p, ii) * math.comb(p, jj) / math.comb(2*p, ii + jj)
    return G


def pinv_right(mat):
    mat = numpy.array(mat)
    return  mat.transpose().__matmul__(np.linalg.inv(mat.__matmul__(mat.transpose())))

def CalcC_BP_inv(BernBasis):
    C_BP = numpy.array(BernBasis.get_coefficients(0))

    active_col = [any(C_BP[:, i] != 0) for i in range(C_BP.shape[1])]
    active_row = [i for i in range(C_BP.shape[0])]
    # if C_BP_inv is None:
    C_BP_inv = numpy.zeros(C_BP.shape).transpose()
    C_BP_inv[numpy.ix_(active_col, active_row)] = numpy.linalg.inv(C_BP[:, active_col])

    return C_BP_inv


def Coef_THB_Bern_element_list(ielem_list, THBBasis, BernBasis, pers_data, THB_support, THB_dofs):
    t0 = time.time()
    ielem_list = numpy.array(ielem_list)



    THB_coefTest = THB_dofs[ielem_list[0]]
    for i, ielem in enumerate(ielem_list):
        if i == 0: continue
        THB_coefTest = numpy.union1d(THB_coefTest, THB_dofs[ielem])


    # THB_coef = THBBasis.get_dofs(ielem_list)
    THB_coef = THB_coefTest
    # print(THB_coef)
    Bern_coef = BernBasis.get_dofs(ielem_list)




    # THB_coef_reverse = { glob_ind : loc_ind  for loc_ind, glob_ind in enumerate(THB_coef)}
    # Bern_coef_reverse = { glob_ind : loc_ind  for loc_ind, glob_ind in enumerate(Bern_coef)}

    THB_coef_reverse = numpy.zeros(len(THBBasis), dtype=int)
    Bern_coef_reverse = numpy.zeros(len(BernBasis), dtype=int)
    for loc_ind, glob_ind in enumerate(THB_coef): THB_coef_reverse[glob_ind] = loc_ind
    for loc_ind, glob_ind in enumerate(Bern_coef): Bern_coef_reverse[glob_ind] = loc_ind


    J_THB = len(THB_coef)
    J_Bern = len(Bern_coef)

    C = numpy.zeros((J_THB, J_Bern))

    time0 = [0,0,0,0,0, time.time() - t0]

    for ielem in ielem_list:
        t0 = time.time()
        subC = numpy.matmul(THBBasis.get_coefficients(ielem), pers_data),
        # glob_Bern_ind = BernBasis.get_dofs(ielem)
        # glob_THB_ind = THB_dofs[ielem]

        time0[0] += time.time() - t0
        t0 = time.time()
        # glob_THB_ind = THBBasis.get_dofs(ielem)
        glob_THB_ind = THB_dofs[ielem]
        time0[1] += time.time() - t0
        t0 = time.time()
        glob_Bern_ind = BernBasis.get_dofs(ielem)
        time0[2] += time.time() - t0



        # find local indices
        t0 = time.time()
        loc_THB_ind = [THB_coef_reverse[index] for index in glob_THB_ind]
        loc_Bern_ind = [Bern_coef_reverse[index] for index in glob_Bern_ind]
        time0[3] += time.time() - t0

        # C[np.ix_(loc_THB_ind, loc_Bern_ind)] = subC * ElemSize[ielem]
        t0 = time.time()
        C[numpy.ix_(loc_THB_ind, loc_Bern_ind)] = subC
        time0[4] += time.time() - t0


    return C, THB_coef, Bern_coef, time0

def GenerateProjMatrices(self, THB_basis, bern_basis):

    C_list = [[] for i in range(self.proj_count)]
    Cind_list = [[] for i in range(self.proj_count)]
    Bind_list = [[] for i in range(self.proj_count)]
    w_list = [[] for i in range(self.proj_count)]

    w_full = numpy.zeros(len(THB_basis))

    # allocate memory for internal mapping. Needs to be implemented better
    pers_data = CalcC_BP_inv(bern_basis)


    t0 = time.time()
    THB_basis = self.topology.basis('th-spline', degree=self.degree)
    print(f'   basis calc : {time.time() - t0:.6}')

    print(f'   elems : {THB_basis.nelems}')
    print(f'   dofs  : {THB_basis.ndofs}')

    t0 = time.time()
    # THB_support, THB_dofs = generate_basis_data(THB_basis)
    THB_dofs = [THB_basis.get_dofs(ielem) for ielem in range(THB_basis.nelems)]
    # THB_support, THB_dofs = generate_basis_data_integral_form(self, THB_basis)
    print(f'   support calc : {time.time() - t0:.6}')

    t0 = time.time()
    # THB_support, THB_dofs = generate_basis_data(THB_basis)
    test = [THB_basis.get_coefficients(ielem) for ielem in range(THB_basis.nelems)]
    # THB_support, THB_dofs = generate_basis_data_integral_form(self, THB_basis)
    print(f'   coeffs calc : {time.time() - t0:.6}')

    timeData = [0,0,0,0,0,0]
    for proj in range(self.proj_count):
        n = self.proj_elem[proj]
        n = numpy.sort(n)

        t0 = time.time()
        C, Cind, Bind, time0 = Coef_THB_Bern_element_list(n, THB_basis, bern_basis, pers_data, None, THB_dofs)

        timeData = [timeData[i] + time0[i] for i in range(5)]


        C_list[proj] = C
        Cind_list[proj] = Cind
        Bind_list[proj] = Bind

        w = C.sum(axis=1)

        w_full[Cind] += w
        w_list[proj] = w

    print(timeData)

    for proj in range(self.proj_count):
        indices = Cind_list[proj]
        w_list[proj] = w_list[proj] / w_full[indices]


    return C_list, Cind_list, Bind_list, w_list

def ToArrayElementwise(A_sparse, n):
    indices, values, shape = sparse.extract(A_sparse)

    def MaskElement(n, shape):
        selectElem = False * numpy.ndarray((shape[0]), dtype=bool)
        selectElem[n] = True
        return [ selectElem ] + [True + False * numpy.ndarray((shape[i+1]), dtype=bool) for i in range(len(shape)-1)]

    A_spars_elem = sparse.take(A_sparse, MaskElement(n, shape))

    def toarrayElement(A_sparse_elem):
        indices, values, shape = sparse.extract(A_sparse_elem)

        loc_indices = [ numpy.unique(indices[i+1]) for i in range(len(shape)-1) ]
        loc_inv_indices = [0*numpy.ndarray(shape[i+1], dtype=int) for i in range(len(shape) - 1)]

        # print(shape[1:])
        # loc_inv_indices = np.ndarray(shape[1:], dtype=np.int64)
        for i in range(len(shape)-1):
            for global_index in indices[i + 1]:
                loc_inv_indices[i][global_index] = numpy.where(loc_indices[i] == global_index)[0][0]

        subshape = [len(loc_indices[i]) for i in range(len(loc_indices))]

        subindices = [0 * indices[i+1] for i in range(len(shape) - 1)]

        for i in range(len(shape) - 1):
            for index_count, global_index in enumerate(indices[i + 1]):
                subindices[i][index_count] = loc_inv_indices[i][global_index]



        retval = numpy.zeros(tuple(subshape), values.dtype)
        numpy.add.at(retval, tuple(subindices), values)
        return retval, loc_indices

    A_elem, loc_indices = toarrayElement(A_spars_elem)
    return A_elem, loc_indices


def project_bern(pbox, func, arguments):
    t0 = time.time()
    pbox.GenerateProjectionElement()
    print(f"Generate Proj elem: {time.time() - t0}")
    t0 = time.time()
    bern_basis = pbox.basis('discont',  degree = pbox.degree)
    print(f"Generate dis basis: {time.time() - t0}")
    t0 = time.time()
    THB_basis = pbox.basis('th-spline', degree = pbox.degree)
    print(f"Generate THB basis: {time.time() - t0}")



    t0 = time.time()
    proj_mat = GenerateProjMatrices(pbox, THB_basis, bern_basis)
    print(f"Generate Proj Mat:  {time.time() - t0}")

    t0 = time.time()
    bern_coef, args = project_element_bern(pbox, func, arguments, basis = bern_basis)
    print(f"loc proj bernstein: {time.time() - t0}")


    args["phi"] = numpy.zeros(len(THB_basis))

    t0 = time.time()
    for proj in range(pbox.proj_count):
        # if proj != 2: continue
        # print(proj)
        # print(pbox.proj_elem[proj])
        n = pbox.proj_elem[proj]
        # print(n)
        n = numpy.sort(n)
        # loop over elements and get sub matrices

        C = proj_mat[0][proj]
        indices = proj_mat[1][proj]
        bern_indices = proj_mat[2][proj]
        w = proj_mat[3][proj]

        # b_vec, b_ind = ToArrayElementwise(bern_coef, n)
        J = numpy.prod(pbox.degree + 1)
        b_vec = numpy.zeros(J * len(n))
        for i, ielem in enumerate(n):
            b_vec[i * J: i*J + J] = args['bern'][ielem * J:ielem*J + J]

        b_vec = args['bern'][bern_indices]

        THB_Coef = numpy.linalg.lstsq(C.transpose(), b_vec, rcond=1e-9)[0]

        args['phi'][indices] += w * THB_Coef
    print(f"THB proj:          {time.time() - t0}")

    return args

def project_element_bern(pbox, fun, arguments, basis = None):
    t0 = time.time()

    # degree = max(degree)
    degree = pbox.degree
    J = math.prod([d + 1 for d in degree])
    G = [ 2 * BernGramMatrix(d) for d in degree]
    Pmat = numpy.kron(G[0], G[1])

    ns = Namespace()
    ns.x = pbox.geometry

    # ns.BernBasis = pbox.arb_basis_discontinuous(pbox.degree)
    if basis is None:
        basis = pbox.basis('discont', degree=pbox.degree)

    # print(basis._arg_coeffs.value)
    # print(basis._coeffs[0])

    # basis._arg_coeffs.value = numpy.matmul(Pmat, basis._arg_coeffs.value)
    # basis._arg_coeffs.value = 2 * basis._arg_coeffs.value
    # print(basis._arg_coeffs.value)
    # print(Pmat)
    bAlt = basis * fun
    proj = function.outer(basis)
    Jacob = function.J(pbox.geometry)

    print(f"   Setup :{time.time() - t0}")
    t0 = time.time()


    b_sparse, Pmat_sparse = tools_nutils.integrate_elementwise_sparse(pbox.topology, [bAlt * Jacob, proj * Jacob] , degree=(max(degree) * 2 + 1), arguments=arguments)
    print(f"   Integrate :{time.time() - t0}")
    t0 = time.time()

    # b_projected = b_sparse.copy()

    # print(b_projected[0]['value'])
    indices, values, shape = sparse.extract(b_sparse)
    indices, values, shape = sparse.extract(b_sparse)

    # print(Pmat.shape)
    print(f"   Setup 2 :{time.time() - t0}")
    t0 = time.time()
    for n in range(shape[0]):
        values[n * J:n * J + J] = numpy.matmul(Pmat, values[n * J:n * J + J])
        # values[n * J:n * J + J] = values[n * J:n * J + J]
    print(f"   Projection :{time.time() - t0}")
    t0 = time.time()
    for n in range(shape[0]):
        values[n * J:n * J + J] = values[n * J:n * J + J] / pbox.elem_size(n)
    print(f"   elem Scaling :{time.time() - t0}")

    b_sparse['value'] = values
    args = {"bern" : values}

    return b_sparse, args

def project_THB(pbox, func, arguments):
    pbox.GenerateProjectionElement()
    THB_basis = pbox.basis('th-spline', degree=pbox.degree)

    ns = Namespace()
    ns.x = pbox.geometry

    A = function.outer(THB_basis, THB_basis)
    b = THB_basis * func
    w = THB_basis
    J = function.J(pbox.geometry)

    A_sparse, b_sparse, w_sparse = tools_nutils.integrate_elementwise_sparse(pbox.topology, [A * J, b * J, w * J],
                                                         degree=(max(pbox.degree) * 2 + 1), arguments=arguments)

    args = {"phi" : numpy.zeros(len(THB_basis))}
    weights = numpy.zeros(len(THB_basis))

    for proj in range(pbox.proj_count):
        n = pbox.proj_elem[proj]

        A_vec, A_ind = ToArrayElementwise(A_sparse, n)
        b_vec, b_ind = ToArrayElementwise(b_sparse, n)
        w_vec, w_ind = ToArrayElementwise(w_sparse, n)

        args['phi'][b_ind] += w_vec * numpy.linalg.solve(A_vec, b_vec)
        weights[w_ind] += w_vec

    args['phi'] = args['phi'] / weights

    return args

def project_global(pbox, func, arguments):
    THB_basis = pbox.basis('th-spline', degree=pbox.degree)

    ns = Namespace()
    ns.x = pbox.geometry

    A = function.outer(THB_basis, THB_basis)
    b = THB_basis * func
    J = function.J(pbox.geometry)

    t0 = time.time()
    A, b = pbox.topology.integrate([A * J, b * J], degree=(max(pbox.degree) * 2 + 1), arguments=arguments)

    u = A.solve(b)
    print(f"took : {time.time() - t0}")
    args = {"phi": u}

    return args
