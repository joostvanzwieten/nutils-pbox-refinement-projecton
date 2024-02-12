# this is an extension for nutils that extends the hierarchicaltopology to use p-box
import time

import numpy as np
import nutils.topology
from nutils import function, mesh, solver, numeric, export, types, sparse
from nutils.expression_v2 import Namespace
import numpy

import local_projection
from local_projection import project_bern, project_THB, project_global
import tools_nutils
from typing import Any, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Sequence

_ArgDict = Mapping[str, numpy.ndarray]

from nutils import element, function, evaluable, _util as util, parallel, numeric, cache, transform, transformseq, warnings, types, points, sparse
from nutils._util import single_or_multiple
from functools import cached_property
from nutils.elementseq import References
from nutils.pointsseq import PointsSequence
from nutils.sample import Sample

from dataclasses import dataclass
from functools import reduce
from os import environ
from typing import Any, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Sequence

import itertools
import numpy
import math
import nutils_poly as poly
import operator
import treelog as log


def map_pboxVec_elemVec(pboxVec, degreeList):
    dim = pboxVec.shape[1]
    combinations = tools_nutils.combvec([range(degree) for degree in degreeList])
    offset = combinations.shape[0]

    output = numpy.zeros( (pboxVec.shape[0] * combinations.shape[0],dim))

    for i in range(pboxVec.shape[0]):
        index = i * offset
        output[index:index+offset,:] = degreeList * pboxVec[i,:] + combinations

    return output

def map_elemVec_pboxVec(elemVec, degreeList):
    if elemVec.shape[0] == 0:
        return elemVec
    pboxVec = numpy.zeros(elemVec.shape, dtype=int)
    for i, degree in enumerate(degreeList):
        pboxVec[:,i] = numpy.floor_divide(elemVec[:,i], degree)

    # quick and dirty check for uniqueness.
    I = numpy.zeros(pboxVec.shape[0],dtype=bool)
    I[0] = True
    for j in range(pboxVec.shape[0]-1):
        if any(pboxVec[j+1,:]-pboxVec[j,:]!=0):
            I[j+1] = True

    return pboxVec[I,:]

def refine_Vec(pboxVec):
    dim = pboxVec.shape[1]
    combinations = tools_nutils.combvec([[0, 1] for _ in range(dim)])
    offset = combinations.shape[0]

    output = numpy.zeros( (pboxVec.shape[0] * combinations.shape[0],dim), dtype=int)

    for i in range(pboxVec.shape[0]):
        index = i * offset
        output[index:index+offset,:] = 2 * pboxVec[i,:] + combinations

    return output

def whereList(elem_list, list):
    elem_list_count = len(elem_list)
    output = numpy.zeros(elem_list_count, dtype=int)
    elem_list = (numpy.sort(elem_list))

    # print(elem_list)
    # print(list)


    output_index = 0
    elem2check = elem_list[output_index]
    for i, elem in enumerate((list)):
        # print(elem, elem2check)
        if elem == elem2check:
            output[output_index] = i
            output_index += 1
            # print(output_index, elem_list_count)
            if output_index == elem_list_count:
                return output
            elem2check = elem_list[output_index]

    return output


class Pbox():

    def __init__(self, degree : tuple, pbox_count : tuple, MaxL = numpy.infty):
        self.refined = False
        self.degree = numpy.array(degree,dtype=int)
        self.pbox_count = numpy.array(pbox_count,dtype=int)
        self.elem_count = numpy.array([pbox_count[i] * degree[i] for i in range(len(degree))],dtype=int)

        self.topology, self.geometry = mesh.rectilinear([numpy.linspace(0, 1, pbox_count[i] * degree[i] + 1) for i in range(len(self.degree))])

        self.pbox_active_indices_per_level = tuple([numpy.arange( numpy.prod(pbox_count) )])
        self.pbox_refined_indices_per_level = tuple([])
        self.L = 1
        self.MaxL = MaxL
        self.pbox_offsets = numpy.array([0, numpy.prod(pbox_count)])

        self.ElemSize = []

        # self.GenerateProjectionElement()


    def refined_by(self, refine):
        self.topology = self.topology.refined_by(refine)
        return self.topology

    def basis(self, type, degree):
        if type == "discont":
            return tools_nutils.arb_basis_discontinuous(self.topology, degree)
        else:
            return self.topology.basis(name= type, degree = degree, continuity = 0)

    def refined_by_pbox(self, refine):
        # map refine to pboxes over every level
        # results in three arrays,
        # pbox_active_indices_per_level : the active p-boxes per level
        # pbox_refinened_indices_per_level : the p-boxes that are refinened to a lower level.
        # pbox_offsets : given active p-box array index, is used to map to correct level p-box
        #
        # together the active and refined p-boxes define the full refinement domain.

        # setup array
        refine = numpy.array(refine)

        # find splits and extend the indices_per_level lists
        splits = numpy.searchsorted(refine, self.pbox_offsets, side='left')
        active_indices_per_level = list(map(list, self.pbox_active_indices_per_level)) + [[]]
        refined_indices_per_level = list(map(list, self.pbox_refined_indices_per_level)) + [[]]


        print(splits)
        if len(splits)>self.MaxL + 1:
            splits = splits[0:self.MaxL + 1]
        if numpy.sum(splits) == 0:
            print(f"Elements to alter are beyond max level")
            return self.topology, False


        # looping over every level
        for ilevel, (start, stop) in enumerate(zip(splits[:-1], splits[1:])):
            # map active-p-box index to level index and add to refined-index-list
            pbox_ind_list_ilevel = tuple(map(active_indices_per_level[ilevel].pop, reversed(refine[start:stop] - self.pbox_offsets[ilevel])))
            refined_indices_per_level[ilevel].extend(pbox_ind_list_ilevel)

            # map p-box level index to refined indices
            pbox_vec_list_ilevel = tools_nutils.map_index_vec(pbox_ind_list_ilevel, self.pbox_count * 2 ** (ilevel) )
            pbox_vec_list_ilevel_refined = refine_Vec(pbox_vec_list_ilevel)
            pbox_index_list_ilevel_refined = tools_nutils.map_vec_index(pbox_vec_list_ilevel_refined, self.pbox_count * 2 ** (ilevel+1)  )

            # add new active levels to active-indices array
            active_indices_per_level[ilevel+1].extend( pbox_index_list_ilevel_refined )

        # make sure that last level array is non-empty
        if not active_indices_per_level[-1]:
            active_indices_per_level.pop()

        if not refined_indices_per_level[-1]:
            refined_indices_per_level.pop()

        # clean up
        self.pbox_active_indices_per_level = [numpy.unique(numpy.array(i, dtype=int)) for i in active_indices_per_level]
        self.pbox_refined_indices_per_level = [numpy.unique(numpy.array(i, dtype=int)) for i in refined_indices_per_level]
        self.pbox_offsets = numpy.cumsum([0, *map(len, active_indices_per_level)], dtype=int)
        self.L = len(self.pbox_active_indices_per_level)

        # update topology with nutils
        self.GenerateHierarchcialTopology()

        self.ElemSize = []

        # generate projection elements
        # self.GenerateProjectionElement()

        return self.topology, True


    def GenerateHierarchcialTopology(self):
        # updates the topology by transforming the p-box structure to element structure.

        # generate data structure
        element_indices_per_level = [[] for _ in self.pbox_active_indices_per_level]

        # loop over levels
        for ilevel, pbox_indices in enumerate(self.pbox_active_indices_per_level):
            # generate elements for each p-box
            pbox_vec_list = tools_nutils.map_index_vec(pbox_indices, self.pbox_count * (2 ** (ilevel)))
            elem_vec_list = map_pboxVec_elemVec(pbox_vec_list, self.degree)
            elem_ind_list = (tools_nutils.map_vec_index(elem_vec_list, self.elem_count * (2 ** (ilevel))))
            element_indices_per_level[ilevel] = numpy.unique(elem_ind_list)

        # create topology (code snippit taken from nutils)
        # need refined boolean for initial refinement
        if self.refined:
            self.topology = nutils.topology.HierarchicalTopology(self.topology.basetopo, ([numpy.unique(numpy.array(i, dtype=int)) for i in element_indices_per_level]) )
        else:
            self.refined  = True
            self.topology = nutils.topology.HierarchicalTopology(self.topology, ([numpy.unique(numpy.array(i, dtype=int)) for i in element_indices_per_level]))

    def find_well_behaved_border_p_box_level(self, level):
        refinement_domain = self.pbox_active_indices_per_level[level]
        if level < len(self.pbox_refined_indices_per_level):
            refinement_domain = numpy.append(refinement_domain,self.pbox_refined_indices_per_level[level])

        pbox_vec_list = tools_nutils.map_index_vec(refinement_domain, self.pbox_count * 2 ** level)

        well_behaved_border_level = []

        for pbox in range(len(refinement_domain)):  # loop over all pboxes
            # find pbox corner, which overlaps with previous level mesh. Use this corner to check the relevant edges
            corner_vec = pbox_vec_list[pbox, :] % 2
            corner_ind = corner_vec[0] + 2 * corner_vec[1]

            corner_normals = (-1 + 2 * corner_vec)
            indices = tools_nutils.combvec([numpy.array([0, offset]) for offset in corner_normals])[1:, :]
            # check relevant edges
            vec2check = pbox_vec_list[pbox, :] + indices
            ind2check = tools_nutils.map_vec_index(vec2check, self.pbox_count * (2 ** (level)))
            Neighbours = [element not in refinement_domain and element >= 0 for element in ind2check]

            if any(Neighbours):
                well_behaved_border_level.append(refinement_domain[pbox])

        well_behaved_border_level_nested = [[[element],] for element in well_behaved_border_level]

        # generate the resulting higher level p-boxes contained in these well-behaved border elements
        for i, pbox in enumerate(well_behaved_border_level_nested):
            pbox_vec_list = tools_nutils.map_index_vec(pbox[0], self.pbox_count * 2 ** level)
            for lvl in range(level+1,self.L):
                pbox_vec_list = refine_Vec(pbox_vec_list)
                pbox_ind_list = tools_nutils.map_vec_index(pbox_vec_list, self.pbox_count * 2 ** (lvl))
                well_behaved_border_level_nested[i].append(pbox_ind_list)


        return well_behaved_border_level, well_behaved_border_level_nested

    def map_elem_pbox(self, elem_global_ind):
        elem_level_ind = {l : [] for l in range(self.L)}
        for elem in elem_global_ind:
            elem_level_index, level = self.ElemLevelIndex(elem)
            elem_level_ind[level].append(elem_level_index)

        pbox_list = []

        for level in range(self.L):
            elemVec = tools_nutils.map_index_vec(elem_level_ind[level], self.elem_count * 2 ** ( level ) )
            pboxVec = map_elemVec_pboxVec(elemVec, self.degree)
            pbox_level_ind = tools_nutils.map_vec_index(pboxVec, self.pbox_count * 2 ** level)
            for pbox in pbox_level_ind:
                pbox_list.append(self.PboxGlobalIndex(pbox, level))

        return pbox_list

    def PboxGlobalIndex(self, pbox, level):
        return numpy.sum(self.pbox_active_indices_per_level[level]< pbox) + self.pbox_offsets[level]

    def PboxLevelIndex(self, pbox):
        level = numpy.sum(self.pbox_offsets <= pbox)-1
        return self.pbox_active_indices_per_level[level][pbox - self.pbox_offsets[level]], level

    def ElemGlobalIndex(self, elem, level):
        if self.refined:
            return numpy.sum(self.topology._indices_per_level[level]< elem) + self.topology._offsets[level]
        else:
            return elem

    def ElemLevelIndex(self, elem):
        if self.refined:
            level = numpy.sum(self.topology._offsets <= elem) - 1
            return self.topology._indices_per_level[level][elem - self.topology._offsets[level]], level
        else:
            return elem, 0

    def GenerateProjectionPbox(self):

        active_Pbox_proj_elem = [] # active Pboxes making up a projection element
        parent_Pbox_proj_elem = [] # the parent well-behaved border pbox

        def CoveredPbox(pbox, level):
            # check wether the pbox is conained in one of the active pbox proj elem
            for parent in parent_Pbox_proj_elem:
                parent_level = self.L - len(parent)
                if pbox in parent[level-parent_level]:
                    return True
            return False

        def CoveredPboxCount(pbox, level):
            # check wether the pbox is conained in one of the active pbox proj elem
            for i,parent in enumerate(parent_Pbox_proj_elem):
                parent_level = self.L - len(parent)
                if level < parent_level:
                    continue
                if pbox in parent[level-parent_level]:
                    # print("pbox",pbox)
                    # print("parent",parent)
                    # print("level",level)
                    # print("parent level",parent_level)
                    return True, i
            return False, -1

        for level in range(self.L):
            # get well-behaved-border-pboxes
            well_behaved_border_pbox_level, well_behaved_border_level_parent = self.find_well_behaved_border_p_box_level(level)

            # add all well-behaved border pboxes that are not already contained in a different active_pbox
            for i,wbb_pbox in enumerate(well_behaved_border_pbox_level):
                if not CoveredPbox(wbb_pbox,level):
                    if wbb_pbox in self.pbox_active_indices_per_level[level]:
                        active_Pbox_proj_elem.append([self.PboxGlobalIndex(wbb_pbox,level)])
                    else:
                        active_Pbox_proj_elem.append([])
                    parent_Pbox_proj_elem.append(well_behaved_border_level_parent[i])

        for i in range(self.pbox_offsets[-1]):
            pbox, level = self.PboxLevelIndex(i)
            Covered, loc =  CoveredPboxCount(pbox, level)
            if Covered and i not in active_Pbox_proj_elem[loc]:
                active_Pbox_proj_elem[loc].append(i)
            elif not any([i in pbox_proj_elem for pbox_proj_elem in active_Pbox_proj_elem]):
                active_Pbox_proj_elem.append([i])

        self.proj_pbox = active_Pbox_proj_elem

    def GenerateProjectionElement(self):
        t0 = time.time()
        self.GenerateProjectionPbox()
        print(f'projection pbox: {time.time() - t0}')

        t0 = time.time()
        offset = numpy.prod(self.degree)
        self.proj_elem = [numpy.zeros(offset * len(box), dtype = int) for box in self.proj_pbox]

        for i, proj_pbox in enumerate(self.proj_pbox):
            for j, pbox in enumerate(proj_pbox):
                pbox_index, level = self.PboxLevelIndex(pbox)
                pbox_vec = tools_nutils.map_index_vec(pbox_index, self.pbox_count * 2 ** level)
                elem_vec = map_pboxVec_elemVec(pbox_vec, self.degree)
                elem_index = tools_nutils.map_vec_index(elem_vec, self.elem_count * 2 ** level)
                elem = numpy.array(
                    [self.ElemGlobalIndex(elem_index_individual, level) for elem_index_individual in elem_index],
                    dtype=int)
                self.proj_elem[i][j * offset : j * offset + offset] = elem

        print(f'projection elem: {time.time() - t0}')

        self.proj_count = len(self.proj_elem)


    def find_elem_size(self):
        J = function.J(self.geometry)
        elemSizeSparse = tools_nutils.integrate_elementwise_sparse(self.topology, [J], degree=0)
        indices, elemSize, shape = sparse.extract(elemSizeSparse[0])
        self.ElemSize = elemSize

    def elem_size(self, elem):
        if len(self.ElemSize) == 0:
            self.find_elem_size()
        return self.ElemSize[elem]

    def project(self, func : function.Array, method = "bern", arguments: Optional[_ArgDict] = None ):

        if method == "bern":
            return project_bern(self, func, arguments)
        elif method == "THB":
            return project_THB(self, func, arguments)
        elif method == "global":
            return project_global(self, func, arguments)
        elif method == "bern_element":
            # for testing only
            _, args = local_projection.project_element_bern(self, func, arguments, basis=None)
            return args
        else:
            return None

