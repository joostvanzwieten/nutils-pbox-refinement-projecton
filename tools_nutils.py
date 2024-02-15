import numpy, math
import nutils.topology
from nutils.expression_v2 import Namespace
from nutils import function, mesh, solver, numeric, export, types, sparse
from nutils import element, function, evaluable, _util as util, parallel, numeric, cache, transform, transformseq, warnings, types, points, sparse
from nutils.element import LineReference, TensorReference
from nutils._util import single_or_multiple
from functools import cached_property
from nutils.elementseq import References
from nutils.pointsseq import PointsSequence
from nutils.sample import Sample
import nutils_poly as poly


# contains code that are either somewhere in nutils that I have not yet found, or functions in nutils that have need to be extended for my use case.
def map_index_vec(index_list, counts):
    # reverse counts for simplicity and get cumprod
    index_list = numpy.array(index_list,dtype=int)
    counts = list(reversed(counts))
    counts.insert(0,1)
    CumProdCounts = list(reversed(numpy.cumprod(list(counts), dtype=int)))
    CumProdCounts.pop(0)

    # preallocate memory
    vec_list = numpy.zeros((index_list.size,len(counts)-1),dtype=int)

    # for cumprod counts, get the division as the j'th vector component
    # the remainder contains the remaining vector components
    index_altered = index_list
    for j in range(len(CumProdCounts)):
        vec_list[:,j], index_altered = numpy.divmod(index_altered, CumProdCounts[j])

    return vec_list


def map_vec_index(vec_list, counts):
    # create list of cumulative products to offset indices by
    cumProdCounts = numpy.cumprod(list(reversed(counts)),dtype=int)
    cumProdCounts = numpy.insert(cumProdCounts, 0, 1)

    # check vector index bounds
    for i, count in enumerate(counts):
        vec_list[ vec_list[:, i] < 0     , i] = -cumProdCounts[-1]
        vec_list[ vec_list[:, i] >= count, i] = -cumProdCounts[-1]

    # preallocate memory
    final_ind = numpy.zeros(vec_list.shape[0],dtype=int)

    # insert values indices, multiplied by offset counts
    for i, count in enumerate(reversed(cumProdCounts[:-1])):
        # print(i,count)
        final_ind = final_ind + count * vec_list[:,i]

    # mark non existent elements as -1
    final_ind[final_ind < 0] = -1

    return final_ind

def combvec(list):
    # input a list of n numpy lists
    # output a n x m array where all combinations are row vectors

    list1 = numpy.array([list.pop()]).T # get a sublist to add
    if not list:
        return list1 # if there are no remaining sub lists, return list1 as solution
    list2 = combvec(list) # else find the combinations of the remaining lists

    # find list sizes. 1D arrays require some clean up
    size1 = list1.shape
    size2 = list2.shape

    # preallocate
    output = numpy.zeros( (size1[0] * size2[0], size1[1] + size2[1]), dtype=int )

    # get combinations
    for i, elem in enumerate(list1):
        output[size2[0] * i:size2[0] * (i+1), -1 ] = elem
        output[size2[0] * i:size2[0] * (i+1), :-1] = list2

    return output


def integrate_elementwise_sparse(self, funcs, degree: int, asfunction: bool = False, ischeme: str = 'gauss',
                                 arguments=None):
    'element-wise integration'

    retvals = [retval for retval in self.sample(ischeme, degree).integrate_sparse(
        [function.kronecker(func, pos=self.f_index, length=len(self), axis=0) for func in funcs],
        arguments=arguments)]
    return retvals


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
        coeffs = poly.mul_different_vars(p1[:, None], p2[None, :], ref1.ndims, ref2.ndims)
        return coeffs.reshape(-1, coeffs.shape[-1])

def arb_basis_discontinuous(topology, degree):

    # print(degree[0])
    # print(topology.references[0].ref1.get_poly_coeffs('bernstein', degree=3))
    coeffs = [_get_poly_coeffs(ref, degree) for ref in topology.references]

    return function.DiscontBasis(coeffs, topology.f_index, topology.f_coords)