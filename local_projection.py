import numpy, math
import numpy as np
import nutils.topology
from nutils.expression_v2 import Namespace
from nutils import function, mesh, solver, numeric, export, types, sparse
from nutils import element, function, evaluable, _util as util, parallel, numeric, cache, transform, transformseq, warnings, types, points, sparse
from nutils.topology import Topology
from nutils._util import single_or_multiple
from functools import cached_property
from nutils.elementseq import References
from nutils.pointsseq import PointsSequence
from nutils.sample import Sample
import tools_nutils
import time
import scipy
import typing
from tools_nutils import UniformDiscontBasis



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

def project_onto_discontinuous_basis(
        topo: Topology,
        geom: function.Array,
        basis: typing.Union[function.DiscontBasis, UniformDiscontBasis],
        fun: function.Array,
        degree: int,
        arguments = {}) -> np.ndarray:
    '''Returns the projection coefficients of `fun` onto the discontinuous `basis`.

    Given a topology `topo`, a geometry `geom` for `topo` and a discontinuous
    basis `basis`, this function computes the projection coefficients of `fun`
    onto `basis`, using a gauss quadrature scheme for the exact integration of
    functions of degree `degree`. For an exact projection, `degree` must be the
    sum of the degree of the `basis` and of `fun`.

    This function does not verify that the supplied `basis` is discontinuous.

    This function is equivalent to, but faster than,

    ```
    from nutils import function, solver
    coeffs = solver.optimize(
        'coeffs',
        topo.integral((function.dotarg('coeffs', basis) - fun)**2 * function.J(geom), degree=degree),
        arguments=arguments,
    )
    ```
    '''

    # Create a sample on `topo` with gauss quadrature points for exact
    # integration of functions of degree `degree`.
    smpl = topo.sample('gauss', degree)

    # We use an evaluable loop to evaluate the projection for each element.
    # `ielem` is the element number, used a.o. to obtain the local basis
    # coefficients and dofs and the quadrature weights. `lower_args` is a
    # representation of the local elemement with index `ielem`, used to lower
    # `function.Array` to `evaluable.Array`.
    ielem = evaluable.loop_index('elems', smpl.nelems)
    lower_args = smpl.get_lower_args(ielem)

    # Define the approximate element integral `elem_int` using the quadrature
    # scheme from `smpl`, scaled with the geometry `geom`.
    weights = smpl.get_evaluable_weights(ielem) * function.jacobian(geom, topo.ndims).lower(lower_args)
    elem_int = lambda integrand: evaluable.einsum('A,AB->B', weights, integrand)

    # Obtain the local dofs and coefficients from `basis` at element `ielem`.
    dofs, basis_coeffs = basis.f_dofs_coeffs(ielem)

    # Sample the local basis in the local coordinates. The first axes of
    # `shapes` correspond to `weights`, the last axis has length `basis.ndofs`
    shapes = evaluable.Polyval(basis_coeffs, topo.f_coords.lower(lower_args))

    # Compute the local mass matrix and right hand side.
    mass = elem_int(evaluable.einsum('Ai,Aj->Aij', shapes, shapes))
    rhs = elem_int(evaluable.einsum('Ai,AB->AiB', shapes, fun.lower(lower_args)))

    # Solve the local least squares problem.
    local_proj_coeffs = evaluable.einsum('ij,jB->Bi', evaluable.inverse(mass), rhs)

    # Scatter the local projection coefficients to the global coefficients and
    # do this for every element in the topology.
    proj_coeffs = evaluable.Inflate(local_proj_coeffs, dofs, evaluable.asarray(basis.ndofs))
    proj_coeffs = evaluable.loop_sum(proj_coeffs, ielem)
    proj_coeffs = evaluable.Transpose.from_end(proj_coeffs, 0)

    # Evaluate.
    return sparse.toarray(evaluable.eval_sparse((proj_coeffs,), **arguments)[0])

def project_element_bern(pbox, fun, arguments, basis = None):
    t0 = time.time()
    if basis is None:
        basis = pbox.basis('discont',degree = pbox.degree)
    args = {"bern":project_onto_discontinuous_basis(topo = pbox.topology, geom = pbox.geometry, basis = basis, fun=fun, degree = max(pbox.degree) * 4 + 1)}
    print(f"   Projection :{time.time() - t0}")

    return None, args

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
