from nutils import mesh, function, sparse, evaluable
from nutils.topology import Topology
from nutils.expression_v2 import Namespace
from tools_nutils import arb_basis_discontinuous, integrate_elementwise_sparse
import nutils_poly as poly
import numpy as np
from matplotlib import pyplot as plt
from unittest import TestCase
import time


def project_onto_discontinuous_basis(
        topo: Topology,
        geom: function.Array,
        basis: function.DiscontBasis,
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


class TestProjectOntoDiscontinuousBasis(TestCase):

    def test_1d(self):
        topo, geom = mesh.line(4)
        basis = topo.basis('discont', 1)
        np.testing.assert_almost_equal(
            project_onto_discontinuous_basis(topo, geom, basis, geom, 2),
            [
                0.0, 1.0, # element 0
                1.0, 2.0, # element 1
                2.0, 3.0, # element 2
                3.0, 4.0, # element 3
            ],
        )

    def test_2d(self):
        topo, geom = mesh.rectilinear([2, 2])
        basis = topo.basis('discont', 1)
        np.testing.assert_almost_equal(
            project_onto_discontinuous_basis(topo, geom, basis, geom[0], 2),
            [
                0.0, 0.0, 1.0, 1.0, # element 0
                0.0, 0.0, 1.0, 1.0, # element 1
                1.0, 1.0, 2.0, 2.0, # element 2
                1.0, 1.0, 2.0, 2.0, # element 3
            ]
        )
        np.testing.assert_almost_equal(
            project_onto_discontinuous_basis(topo, geom, basis, geom[1], 2),
            [
                0.0, 1.0, 0.0, 1.0, # element 0
                1.0, 2.0, 1.0, 2.0, # element 1
                0.0, 1.0, 0.0, 1.0, # element 2
                1.0, 2.0, 1.0, 2.0, # element 3
            ]
        )

    def test_1d_vector(self):
        topo, geom = mesh.line(4)
        basis = topo.basis('discont', 1)
        coeffs = project_onto_discontinuous_basis(topo, geom, basis, np.stack([geom, 2 * geom]), 2)
        np.testing.assert_almost_equal(
            project_onto_discontinuous_basis(topo, geom, basis, np.stack([geom, 2 * geom]), 2),
            [
                [0.0, 0.0], [1.0, 2.0], # element 0
                [1.0, 2.0], [2.0, 4.0], # element 1
                [2.0, 4.0], [3.0, 6.0], # element 2
                [3.0, 6.0], [4.0, 8.0], # element 3
            ],
        )


# setup domain and pick degrees (note that they can be different)
N_elems = 2
degree = np.array([1,1])
topology, geometry = mesh.rectilinear([np.linspace(0,1,N_elems + 1),np.linspace(0,1,N_elems + 1)])
topology = topology.refined_by([0])

print(topology)

ns = Namespace()
ns.x = geometry
ns.Pi = np.pi

ns.basis = arb_basis_discontinuous(topology, degree)
ns.fun = 'sin( Pi x_0 ) cos( Pi x_1 )'

t0 = time.time()

b = ns.basis * ns.fun
mass = function.outer(ns.basis, ns.basis)
Jacob = function.J(geometry)

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


print(f"Old :      {time.time() - t0:.6}")
t0 = time.time()
x = project_onto_discontinuous_basis(topology, geometry, ns.basis, ns.fun, max(degree) * 4 + 1)
print(f"Improved : {time.time() - t0:.6}")


print(x)

# plot solution and find maximal error
args = {"approx":x}
ns.add_field('approx', ns.basis)
bezier = topology.sample('bezier', 4 * max(degree))
x, approx, error = bezier.eval(['x_i','approx', 'approx - fun'] @ ns, **args)
print(f"max error over full domain : {max(error)}")
plt.tripcolor(x[:, 0], x[:, 1], bezier.tri, approx, shading='gouraud', rasterized=True)
plt.show()