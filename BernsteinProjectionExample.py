from nutils import mesh, function, sparse
from nutils.expression_v2 import Namespace
from tools_nutils import arb_basis_discontinuous
import nutils_poly as poly
import numpy as np
from matplotlib import pyplot as plt


# setup domain and pick degrees (note that they can be different)
N_elems = 4
degree = np.array([3,2])
topology, geometry = mesh.rectilinear([np.linspace(0,1,N_elems + 1),np.linspace(0,1,N_elems + 1)])


ns = Namespace()
ns.x = geometry
ns.Pi = np.pi

# functions for discontinuous basis where the degree can be chosen arbitrary
# def _get_poly_coeffs(reference, degree):
#     p1 = reference.ref1.get_poly_coeffs('bernstein', degree=int(degree[0]))
#     p2 = reference.ref2.get_poly_coeffs('bernstein', degree=int(degree[1]))
#     plan = poly.MulPlan((poly.MulVar.Left, poly.MulVar.Right), degree[0], degree[1])
#     coeffs_LIST = [plan(p1[i, :], p2[j, :]) for i in range(p1.shape[0]) for j in range(p2.shape[0])]
#     return np.array(coeffs_LIST)
#
# def arb_basis_discontinuous(topology, degree):
#
#     # print(degree[0])
#     # print(topology.references[0].ref1.get_poly_coeffs('bernstein', degree=3))
#     coeffs = [_get_poly_coeffs(ref, degree) for ref in topology.references]
#
#     return function.DiscontBasis(coeffs, topology.f_index, topology.f_coords)

ns.basis = arb_basis_discontinuous(topology, degree)
ns.fun = 'sin( Pi x_0 ) cos( Pi x_1 )'

b = ns.basis * ns.fun
mass = function.outer(ns.basis, ns.basis)
Jacob = function.J(geometry)

# function that does the exact same as integrate_elementwise, but does not reduce to a final dense matrix by removing the sparsity over elements.
def integrate_elementwise_sparse(self, funcs, degree: int, asfunction: bool = False, ischeme: str = 'gauss',arguments=None):
    'element-wise integration'
    retvals = [retval for retval in self.sample(ischeme, degree).integrate_sparse(
        [function.kronecker(func, pos=self.f_index, length=len(self), axis=0) for func in funcs],
        arguments=arguments)]
    return retvals



# do integration, this returns a sparse matrix where the first dim described the element index, and the remaining indices are the global dofs
b_sparse, mass_sparse = integrate_elementwise_sparse(topology, [b * Jacob, mass * Jacob] , degree=(max(degree) * 4 + 1))
x = np.zeros(ns.basis.ndofs)

# the following function gets the local dense matrix associated for the element index ielem, it also returns the associated global dofs
def get_local_elem(sparse_data, ielem):
    indices, values, shape = sparse.extract(sparse_data)
    ndim = len(shape) - 1

    mask = np.where(indices[0] == ielem)
    global_index_list = [ indices[dim + 1][mask] for dim in range(ndim)]
    local_index_list  = [{global_index:local_index  for local_index, global_index in enumerate(np.unique(global_index_list[dim]))} for dim in range(ndim)]

    local_data = np.zeros([len(local_index_list[dim].keys()) for dim in range(ndim)])

    for i, val in enumerate(values[mask]):
        global_index = [global_index_list[dim][i] for dim in range(ndim)]
        local_index = tuple( [local_index_list[dim][index] for dim, index in enumerate(global_index)] )
        local_data[local_index] = val

    return local_data, [np.unique(global_index) for global_index in global_index_list ]



for ielem in range(len(topology)):
    # get local vector and matrices
    b_vec, global_indices = get_local_elem(b_sparse, ielem)
    mass_matrix, _ = get_local_elem(mass_sparse, ielem)

    # solve local projection and store at correct global dofs
    x[global_indices] = np.linalg.solve(mass_matrix, b_vec)


# plot solution and find maximal error
args = {"approx":x}
ns.add_field('approx', ns.basis)
bezier = topology.sample('bezier', 4 * max(degree))
x, approx, error = bezier.eval(['x_i','approx', 'approx - fun'] @ ns, **args)
print(f"max error over full domain : {max(error)}")
plt.tripcolor(x[:, 0], x[:, 1], bezier.tri, approx, shading='gouraud', rasterized=True)
plt.show()