from nutils import mesh, function, sparse, evaluable
from nutils.element import LineReference, TensorReference
from nutils.topology import Topology
from nutils.expression_v2 import Namespace
from tools_nutils import arb_basis_discontinuous, integrate_elementwise_sparse, UniformDiscontBasis
import nutils_poly as poly
import numpy as np
from matplotlib import pyplot as plt
from unittest import TestCase
import typing
import time
from PboxTopology import Pbox
from local_projection import project_onto_discontinuous_basis

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


