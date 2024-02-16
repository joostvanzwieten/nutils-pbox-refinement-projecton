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
import local_projection



# class TestProjectOntoDiscontinuousBasis(TestCase):
#
#     def test_1d(self):
#         topo, geom = mesh.line(4)
#         basis = topo.basis('discont', 1)
#         np.testing.assert_almost_equal(
#             project_onto_discontinuous_basis(topo, geom, basis, geom, 2),
#             [
#                 0.0, 1.0, # element 0
#                 1.0, 2.0, # element 1
#                 2.0, 3.0, # element 2
#                 3.0, 4.0, # element 3
#             ],
#         )
#
#     def test_2d(self):
#         topo, geom = mesh.rectilinear([2, 2])
#         basis = topo.basis('discont', 1)
#         np.testing.assert_almost_equal(
#             project_onto_discontinuous_basis(topo, geom, basis, geom[0], 2),
#             [
#                 0.0, 0.0, 1.0, 1.0, # element 0
#                 0.0, 0.0, 1.0, 1.0, # element 1
#                 1.0, 1.0, 2.0, 2.0, # element 2
#                 1.0, 1.0, 2.0, 2.0, # element 3
#             ]
#         )
#         np.testing.assert_almost_equal(
#             project_onto_discontinuous_basis(topo, geom, basis, geom[1], 2),
#             [
#                 0.0, 1.0, 0.0, 1.0, # element 0
#                 1.0, 2.0, 1.0, 2.0, # element 1
#                 0.0, 1.0, 0.0, 1.0, # element 2
#                 1.0, 2.0, 1.0, 2.0, # element 3
#             ]
#         )
#
#     def test_1d_vector(self):
#         topo, geom = mesh.line(4)
#         basis = topo.basis('discont', 1)
#         coeffs = project_onto_discontinuous_basis(topo, geom, basis, np.stack([geom, 2 * geom]), 2)
#         np.testing.assert_almost_equal(
#             project_onto_discontinuous_basis(topo, geom, basis, np.stack([geom, 2 * geom]), 2),
#             [
#                 [0.0, 0.0], [1.0, 2.0], # element 0
#                 [1.0, 2.0], [2.0, 4.0], # element 1
#                 [2.0, 4.0], [3.0, 6.0], # element 2
#                 [3.0, 6.0], [4.0, 8.0], # element 3
#             ],
#         )


# setup domain and pick degrees (note that they can be different)
N_elems = 2
degree = np.array([3,2])
# topology, geometry = mesh.rectilinear([np.linspace(0,1,N_elems + 1),np.linspace(0,1,N_elems + 1)])
# topology = topology.refined_by([0])

r_elems = (N_elems,N_elems)
# degree = (2,2)

pbox = Pbox(degree, r_elems)
print(pbox.pbox_active_indices_per_level)
pbox.refined_by_pbox([0])

print(pbox.pbox_active_indices_per_level)

ns = Namespace()
ns.x = pbox.geometry
ns.Pi = np.pi

ns.basis = arb_basis_discontinuous(pbox.topology, degree)
# ns.basis = pbox.basis('discont', degree=degree)
ns.fun = 'sin( 2 Pi x_0 ) cos( 2 Pi x_1 )'

t0 = time.time()
x = local_projection.project_onto_discontinuous_basis(pbox.topology, pbox.geometry, ns.basis, ns.fun, max(degree) * 4 + 1)
print(f"Old :      {time.time() - t0:.6}")
# t0 = time.time()
# x = project_onto_discontinuous_basis(topology, geometry, ns.basis, ns.fun, max(degree) * 4 + 1)
# print(f"Improved : {time.time() - t0:.6}")


print(x)
# quit()

# plot solution and find maximal error
args = {"approx":x}
ns.add_field('approx', ns.basis)
bezier = pbox.topology.sample('bezier', 4 * max(degree))
x, approx, error = bezier.eval(['x_i','approx', 'approx - fun'] @ ns, **args)
print(f"max error over full domain : {max(error)}")
plt.tripcolor(x[:, 0], x[:, 1], bezier.tri, error, shading='gouraud', rasterized=True)
plt.show()