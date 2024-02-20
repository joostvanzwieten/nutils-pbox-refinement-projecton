import numpy as np
import nutils.types
from nutils import evaluable, function, mesh, sparse
from nutils.topology import Topology
import typing
from unittest import TestCase


def project_onto_basis_macroelemwise(
        topo: Topology,
        geom: function.Array,
        basis: typing.Union[function.Basis],
        weights: str,
        macroelems: typing.Sequence[np.ndarray],
        fun: function.Array,
        degree: int,
        arguments = {}) -> np.ndarray:
    '''Returns the macroelementwise weighted average projection coefficients of `fun` onto the `basis`.

    Given a topology `topo`, a geometry `geom` for `topo` a basis, a set of
    macroelements and a weighting method, this function computes the
    macroelementwise weighted average projection coefficients of `fun` onto
    `basis`, using a gauss quadrature scheme for the exact integration of
    functions of degree `degree`. For an exact projection, `degree` must be the
    sum of the degree of the `basis` and of `fun`.

    The parameter `macroelems` must be a sequence of macroelements, with
    each macroelement being an array of element indices, such that combined all
    element indices of `topo` are given exactly once and such that the basis
    functions that have support on a macroelement are linearly independent on
    that macroelement.

    The parameter `weigths` defines the weighting that is applied when
    averaging the projection coefficients. Supported are `'uniform'` and
    `'integral'`.
    '''

    concat_macroelems = np.concatenate(macroelems)
    unique_macroelems = np.unique(concat_macroelems)
    if len(unique_macroelems) != len(concat_macroelems):
        raise ValueError('duplicate element indices in parameter `macroelems`')
    if len(unique_macroelems) != len(topo) or not (unique_macroelems == np.arange(len(topo))).all():
        raise ValueError('`macroelems` is not a partition of the element indices of `topo`')

    # Collect all dofs that have support on a macroelement for all
    # macroelements in `macro_dofs` and a mapping of element index to
    # macroelement index in `elem_to_macro`.
    macro_dofs = []
    elem_to_macro = [None] * len(topo)
    for ielems in macroelems:
        dofs = [basis.get_dofs(ielem) for ielem in ielems]
        unique = np.unique(np.concatenate(dofs))
        macro_dofs.append(unique)
        for ielem, dofs in zip(ielems, dofs):
            elem_to_macro[ielem] = np.searchsorted(unique, dofs)
            assert (unique[elem_to_macro[ielem]] == dofs).all()

    # Create a sample on `topo` with gauss quadrature points for exact
    # integration of functions of degree `degree`.
    smpl = topo.sample('gauss', degree)

    # We use an evaluable loop to evaluate the projection for each macro
    # element. `imacro` is the macro element number.
    imacro = evaluable.loop_index('macro', len(macroelems))

    # For each macro element we define an integral by summing integrals over
    # the real elements that belong to the macro element. The summation is
    # implemented using `loop_sum`. The loop index `imacroelem` is the local
    # element index w.r.t. the macroelement. The index `ielem` is the mapping
    # of `imacroelem` to the elements w.r.t. `topo`.
    ielems = evaluable.Elemwise(tuple(map(nutils.types.arraydata, macroelems)), imacro, int)
    imacroelem = evaluable.loop_index('elem', ielems.shape[0])
    ielem = evaluable.Take(ielems, imacroelem)
    lower_args = smpl.get_lower_args(ielem)
    elem_int_weights = smpl.get_evaluable_weights(ielem) * function.jacobian(geom, topo.ndims).lower(lower_args)
    elem_int = lambda integrand: evaluable.einsum('A,AB->B', smpl.get_evaluable_weights(ielem) * function.jacobian(geom, topo.ndims).lower(lower_args), integrand)
    macro_int = lambda integrand: evaluable.loop_sum(elem_int(integrand), imacroelem)

    # Obtain the dofs at macroelement `imacro`, which we have prepared at the
    # beginning of this function.
    dofs = evaluable.Elemwise(tuple(map(nutils.types.arraydata, macro_dofs)), imacro, int)

    # Obtain the coefficients of `basis` at real element `ielem` ...
    _, basis_coeffs = basis.f_dofs_coeffs(ielem)
    # ... evaluate the polynomial at the quadrature points ...
    elem_to_macro = evaluable.Elemwise(tuple(map(nutils.types.arraydata, elem_to_macro)), ielem, int)
    shapes = evaluable.Polyval(basis_coeffs, topo.f_coords.lower(lower_args))
    # ... and insert zeros for the dofs of the macroelement for which that
    # basis has no support on the real element. This `inflate` triggers an
    # explicit inflation warning, which should be ignored.
    shapes = evaluable.Inflate(shapes, elem_to_macro, dofs.shape[0])

    # Compute the local mass matrix and right hand side.
    mass = macro_int(evaluable.einsum('Ai,Aj->Aij', shapes, shapes))
    rhs = macro_int(evaluable.einsum('Ai,AB->AiB', shapes, fun.lower(lower_args)))

    # Solve the local least squares problem.
    macro_proj_coeffs = evaluable.einsum('ij,jB->Bi', evaluable.inverse(mass), rhs)

    # Multiply the local projection coefficients with the weights for this
    # element.
    if weights == 'uniform':
        weights = evaluable.ones(shapes.shape[1:])
    elif weights == 'integral':
        weights = macro_int(shapes)
    else:
        raise ValueError(f'unknown weights method: {weights}, supported: uniform, integral')
    macro_proj_coeffs *= evaluable.prependaxes(weights, macro_proj_coeffs.shape[:-1])

    # Scatter the local projection coefficients to the global coefficients ...
    proj_coeffs = evaluable.Inflate(macro_proj_coeffs, dofs, evaluable.asarray(basis.ndofs))
    # ... sum over all elements ...
    proj_coeffs = evaluable.loop_sum(proj_coeffs, imacro)
    # ... and normalize by the sum of the weights.
    # FIXME: The `.eval()` should not be necessary, but causes an unnecessary
    # explicit inflation.
    summed_weights = evaluable.loop_sum(evaluable.Inflate(weights, dofs, evaluable.asarray(basis.ndofs)), imacro)
    proj_coeffs /= evaluable.prependaxes(summed_weights.eval(), proj_coeffs.shape[:-1])
    # `Inflate` scatters the last axis, but we want this axis to be the first
    # axis, so we move the last axis to the front.
    proj_coeffs = evaluable.Transpose.from_end(proj_coeffs, 0)

    # Evaluate.
    return sparse.toarray(evaluable.eval_sparse((proj_coeffs,), **arguments)[0])


class TestProjectOntoBasisMacroelemwise(TestCase):

    def test_1d(self):
        for weights in 'uniform', 'integral':
            with self.subTest(weights=weights):
                topo, geom = mesh.line(4)
                basis = topo.basis('spline', 1)
                np.testing.assert_almost_equal(
                    project_onto_basis_macroelemwise(
                        topo,
                        geom,
                        basis,
                        weights,
                        [[0], [1, 2], [3]],
                        geom,
                        2,
                    ),
                    [0, 1, 2, 3, 4],
                )

    def test_2d(self):
        for weights in 'uniform', 'integral':
            with self.subTest(weights=weights):
                topo, geom = mesh.rectilinear([2, 2])
                basis = topo.basis('spline', 1)
                np.testing.assert_almost_equal(
                    project_onto_basis_macroelemwise(
                        topo,
                        geom,
                        basis,
                        weights,
                        [[0], [1, 2], [3]],
                        geom[0],
                        2,
                    ),
                    [0, 0, 0, 1, 1, 1, 2, 2, 2],
                )
                np.testing.assert_almost_equal(
                    project_onto_basis_macroelemwise(
                        topo,
                        geom,
                        basis,
                        weights,
                        [[0], [1, 2], [3]],
                        geom[1],
                        2,
                    ),
                    [0, 1, 2, 0, 1, 2, 0, 1, 2],
                )

    def test_1d_vector(self):
        for weights in 'uniform', 'integral':
            with self.subTest(weights=weights):
                topo, geom = mesh.line(4)
                basis = topo.basis('spline', 1)
                np.testing.assert_almost_equal(
                    project_onto_basis_macroelemwise(
                        topo,
                        geom,
                        basis,
                        weights,
                        [[0], [1, 2], [3]],
                        np.stack([geom, 2 * geom]),
                        2,
                    ),
                    [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]],
                )
