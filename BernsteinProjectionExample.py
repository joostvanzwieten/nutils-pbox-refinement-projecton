from nutils import mesh, function, sparse, evaluable
from nutils.element import LineReference, TensorReference
from nutils.topology import Topology
from nutils.expression_v2 import Namespace
import nutils_poly as poly
import numpy as np
from matplotlib import pyplot as plt
from unittest import TestCase


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
N_elems = 4
degree = np.array([3,2])
topology, geometry = mesh.rectilinear([np.linspace(0,1,N_elems + 1),np.linspace(0,1,N_elems + 1)])


ns = Namespace()
ns.x = geometry
ns.Pi = np.pi

# functions for discontinuous basis where the degree can be chosen arbitrary
def _get_poly_coeffs(reference, degree):
    assert len(degree) == reference.ndims
    if len(degree) == 1:
        assert isinstance(reference, LineReference)
        return reference.get_poly_coeffs('bernstein', degree[0].__index__())
    else:
        assert isinstance(reference, TensorReference)
        ref1 = reference.ref1
        ref2 = reference.ref2
        p1 = _get_poly_coeffs(ref1, degree[:ref1.ndims])
        p2 = _get_poly_coeffs(ref2, degree[ref1.ndims:])
        coeffs = poly.mul_different_vars(p1[:,None], p2[None,:], ref1.ndims, ref2.ndims)
        return coeffs.reshape(-1, coeffs.shape[-1])

def arb_basis_discontinuous(topology, degree):

    # print(degree[0])
    # print(topology.references[0].ref1.get_poly_coeffs('bernstein', degree=3))
    if topology.references.isuniform:
        coeffs = [_get_poly_coeffs(topology.references[0], degree)] * len(topology)
    else:
        coeffs = [_get_poly_coeffs(ref, degree) for ref in topology.references]

    return function.DiscontBasis(coeffs, topology.f_index, topology.f_coords)

ns.basis = arb_basis_discontinuous(topology, degree)
ns.fun = 'sin( Pi x_0 ) cos( Pi x_1 )'

x = project_onto_discontinuous_basis(topology, geometry, ns.basis, ns.fun, max(degree) * 4 + 1)

# plot solution and find maximal error
args = {"approx":x}
ns.add_field('approx', ns.basis)
bezier = topology.sample('bezier', 4 * max(degree))
x, approx, error = bezier.eval(['x_i','approx', 'approx - fun'] @ ns, **args)
print(f"max error over full domain : {max(error)}")
plt.tripcolor(x[:, 0], x[:, 1], bezier.tri, approx, shading='gouraud', rasterized=True)
plt.show()