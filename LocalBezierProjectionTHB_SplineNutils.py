import numpy
import PboxTopology
from nutils import function, mesh, solver, numeric, export, types, sparse, evaluable
import nutils_poly as poly
from nutils.expression_v2 import Namespace
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from matplotlib import cm
import math, time
import scipy
import pickle
# from MHD_VMS_functions import *
# from MHD2DplottingFUNCTIONS import *
import Tools
import TestCases
from THB_spline_functions import *
import tools_nutils


def PlotData(topology, geometry, list_of_vars, ns, args, points, numPoints):
    # NumPoints = 20
    # xx, yy = np.meshgrid(np.linspace(0, 1, NumPoints), np.linspace(0, 1, NumPoints))
    xx_col = xx.reshape((points[0].size, 1), order='C')
    yy_col = yy.reshape((points[1].size, 1), order='C')

    datapoint = (np.concatenate((xx_col, yy_col), 1))
    sampleobj = topology.locate(geometry, datapoint, eps = 1e-10)

    vars = sampleobj.eval(list_of_vars @ ns, **args)


    vars_out = [ var.reshape((numPoints, numPoints)) for var in vars]

    return vars_out


def BottomRefTest2(r_elems, degree, func, ElemProj = "Bern"):
    if any([r % 2 for r in r_elems]):
        raise 'r_elem needs to be a multiple of two'

    print(f'solving with {ElemProj} for r = {r_elems[0]} and degree {degree[0]}')



    time0 = time.time()
    pbox = PboxTopology.Pbox(degree, r_elems)

    # get pbox indices of lower quadrant and refine
    refined_vec = tools_nutils.combvec([list(range(int(r/2))) for r in r_elems])
    refined_index = tools_nutils.map_vec_index(refined_vec, r_elems)

    pbox.refined_by_pbox(refined_index)

    ns = Namespace()
    ns.x = pbox.geometry
    ns.PI = math.pi

    ns.ProjFun = func

    args = pbox.project(ns.ProjFun, method = ElemProj)


    time3 = time.time()

    if ElemProj == "bern_element":
        ns.basis = pbox.basis('discont', degree=degree)
        args["phi"] = args["bern"]
    else:
        ns.basis = pbox.basis('th-spline', degree = degree)
    ns.add_field('phi', ns.basis)
    ns.errorFunc = 'phi - ProjFun'
    # args = {'phi' : coef}
    ns.define_for('x', gradient='D', normal='n', jacobians=('dV', 'dS'))
    error = math.sqrt(pbox.topology.integral('errorFunc errorFunc dV' @ ns, degree=max(degree)).eval(**args))

    time4 = time.time()
    return error, time4 - time0

def BottomRefTest(r_elems, degree, func, ElemProj = "Bern"):
    if any([r % 2 for r in r_elems]):
        raise 'r_elem needs to be a multiple of two'

    print(f'solving with {ElemProj} for r = {r_elems[0]} and degree {degree[0]}')



    time0 = time.time()
    pbox = PboxTopology.Pbox(degree, r_elems)

    # get pbox indices of lower quadrant and refine
    refined_vec = PboxTopology.combvec([list(range(int(r/2))) for r in r_elems])
    refined_index = PboxTopology.map_vec_index(refined_vec, r_elems)

    topology = pbox.refined_by_pbox(refined_index)

    time1 = time.time()

    ns = Namespace()
    ns.x = pbox.geometry
    ns.PI = math.pi

    ns.ProjFun = func
    # ns.ProjFun = 'sin(PI (1 - x_0)) sin(PI (1 - x_1)) sin(PI (1 - x_2))'

    # ns.basis = topology.basis('spline', degree = [px, py])
    ns.THBBasis = pbox.basis('th-spline', degree=degree)
    ns.BernBasis = pbox.basis('discont', degree=degree[0])

    J = function.J(pbox.geometry)

    Afun = function.outer(ns.THBBasis)
    bfun = ns.THBBasis * ns.ProjFun

    AAltfun = function.outer(ns.THBBasis, ns.BernBasis)
    bAltfun = ns.BernBasis * ns.ProjFun

    integrateDegree = max(degree)**2 + 1
    if ElemProj == "Bern":
        FunctionsIntegrate = [ns.THBBasis * J, AAltfun * J,bAltfun * J]
        w_sparse, A_sparse, b_sparse = PboxTopology.integrate_elementwise_sparse(pbox.topology, FunctionsIntegrate, degree=integrateDegree)
    elif ElemProj == "Glob":
        FunctionsIntegrate = [Afun * J, bfun * J, ns.THBBasis * J]
        A_sparse, b_sparse, w_sparse = PboxTopology.integrate_elementwise_sparse(pbox.topology, FunctionsIntegrate,degree=integrateDegree)
    else:
        FunctionsIntegrate = [Afun * J, bfun * J, ns.THBBasis * J]
        A_sparse, b_sparse, w_sparse = PboxTopology.integrate_elementwise_sparse(pbox.topology,FunctionsIntegrate,degree=integrateDegree)

    w_sparse_normalize = PboxTopology.sparse_normalize(w_sparse, 1)

    args = {}
    args['phi'] = np.zeros(ns.THBBasis.ndofs)

    # print(ns.THBBasis.nelems)
    if ElemProj != "Glob":
        ProjMatrices = [PboxTopology.ToArrayElementwise(A_sparse, n) for n in pbox.proj_elem]
        ProjVector   = [PboxTopology.ToArrayElementwise(b_sparse, n) for n in pbox.proj_elem]
        ProjWeight   = [PboxTopology.ToArrayElementwise(w_sparse_normalize, n) for n in pbox.proj_elem]
    else:
        n = list(range(sum([len(elem_list) for elem_list in pbox.proj_elem])))
        An, Aindicesn = PboxTopology.ToArrayElementwise(A_sparse, n)
        bn, bindicesn = PboxTopology.ToArrayElementwise(b_sparse, n)

    time2 = time.time()

    if ElemProj == "Glob":


        args['phi'] = np.linalg.solve(An, bn)

    else:
        for proj in range(pbox.proj_count):
            # print(proj)
            # print(pbox.proj_elem[proj])
            n = pbox.proj_elem[proj]

            An, Aindicesn = ProjMatrices[proj]
            bn, bindicesn = ProjVector[proj]
            wn, windicesn = ProjWeight[proj]

            # loop over elements and get sub matrices

            if ElemProj == "Bern":
                # wn, windicesn = ToArrayElementwise(w_sparse_normalize, n)
                # An, Aindicesn = ToArrayElementwise(A_sparse, n)
                # bn, bindicesn = ToArrayElementwise(b_sparse, n)

                ElemProjCoef = np.linalg.lstsq(An.transpose(), bn, rcond=1e-15)[0]

            else:  # use THB
                # An, Aindicesn = ToArrayElementwise(A_sparse, n)
                # bn, bindicesn = ToArrayElementwise(b_sparse, n)
                # wn, windicesn = ToArrayElementwise(w_sparse_normalize, n)

                ElemProjCoef = np.linalg.solve(An, bn)

            args['phi'][Aindicesn[0]] += wn * ElemProjCoef

    time3 = time.time()

    ns.add_field('phi', ns.THBBasis)
    ns.errorFunc = 'phi - ProjFun'
    # args = {'phi' : coef}
    ns.define_for('x', gradient='D', normal='n', jacobians=('dV', 'dS'))
    error = math.sqrt(topology.integral('errorFunc errorFunc dV' @ ns, degree=integrateDegree).eval(**args))

    time4 = time.time()
    return error, np.array([time1 - time0, time2 - time1,time3 - time2,time4 - time3])

# BottomRefTest((2,2),(2,2),'sin(PI (1 - x_0)) sin(PI (1 - x_1))')

def AccuracyTest():
    # maxNum = 32
    maxNum = 16

    r_list = [2*(r+1) for r in range(int(maxNum / 2 - 1))]
    # degreeList = [1,2,3,4,5]
    degreeList = [1,2,3]


    rDegreeList = [[ r  for r in r_list if r * degree < maxNum ] for degree in degreeList]
    hDegreeList = [[ 1/r/degree  for r in r_list if r * degree < maxNum ] for degree in degreeList]
    print(rDegreeList)
    print(hDegreeList)
    func = 'sin( 3.14 (1 - x_0) ) sin( 3.14 (1 - x_1) )'


    colorList = ['b','g','r','c','m','y']
    lineList = {"bern":"solid","global":"dotted","THB":"dashed","bern_element":"dashdot"}
    time_title = ["Creating, Refining and finding proj elem","Calculating Matrices","Projection","Error Calc"]

    ElemProjList = ["bern","bern_element","global","THB"]
    # ElemProjList = ["Bern"]


    error = {}
    timeList = {}
    for ProjMethod in ElemProjList:
        for d,degree in enumerate(degreeList):
            # output = [BottomRefTest((r,r),(degree,degree),func, ElemProj=ProjMethod) for r in rDegreeList[degree-1]]
            output = [BottomRefTest2((r,r),(degree,degree),func, ElemProj=ProjMethod) for r in rDegreeList[d]]
            error[ProjMethod + str(degree)] = [error_problem for error_problem, _ in output]
            timeList[ProjMethod + str(degree)] = [time_problem for _, time_problem in output]




    for key, data in error.items():
        degree = int(key[-1])
        print(degree, degreeList)
        d = numpy.where(degree == numpy.array(degreeList))[0][0]
        print(d)
        key = key[0:-1]
        plt.loglog(hDegreeList[d],data,linestyle = lineList[str(key)],color = colorList[d])
    plt.legend(error.keys())

    # fig, axis = plt.subplots(2,2)
    # for key, data in timeList.items():
    #     data = np.array(data)
    #     degree = int(key[-1])
    #     key = key[0:-1]
    #
    #     for i in range(data.shape[1]):
    #         i1,i2 = divmod(i,2)
    #         axis[i1][i2].loglog(hDegreeList[degree-1],data[:,i],linestyle = lineList[str(key)],color = colorList[degree-1])
    #         axis[i1][i2].title.set_text(time_title[i])
    #         axis[i1][i2].legend(timeList.keys())

    plt.legend(timeList.keys())
    plt.show()

# AccuracyTest()
# quit()

r_elems = (2, 2)
# r_elems = (4, 4)
degree  = (3, 3)

ElemProj = "Bern"

# topology, geometry = mesh.rectilinear([np.linspace(0, 1, r_elems * px + 1), np.linspace(0, 1, r_elems * py + 1)])
pbox = PboxTopology.Pbox(degree,r_elems)
topology = pbox.topology
geometry = pbox.geometry

ns = Namespace()
ns.x = geometry

# pbox_list = pbox.map_elem_pbox([0])
# pbox_list = pbox.map_elem_pbox([3,4])
# pbox_list = pbox.map_elem_pbox([7])
# print(pbox_list)

# ns.testBasis = pbox.basis('th-spline', degree = degree)
# ns.add_field('g', ns.testBasis)
#
# args = { 'g': numpy.random.random(len(ns.testBasis)) }
args = {}
NumPoints = 100
xx, yy = np.meshgrid(np.linspace(0, 1, NumPoints), np.linspace(0, 1, NumPoints))
# g_data = PlotData(topology, geometry, ['g'], ns, args, (xx, yy) ,NumPoints)
#
# fig, axs  = plt.subplots(1,1)
# axs.contourf(xx,yy,g_data[0])

# bezier = topology.sample('bezier', 2 * (degree[0]+1) * (degree[1]+1))
# x, phi =  bezier.eval(['x_i', 'g'] @ ns, **args)
# fig, axs = plt.subplots(1,1)
# plotFig = export.triplot(axs,x, phi, tri = bezier.tri, hull = bezier.hull)
# plt.colorbar(plotFig)



pbox.refined_by_pbox([0])
# pbox.refined_by_pbox([3,4])
# pbox.refined_by_pbox([7])

# ns = Namespace()
ns.x = geometry
ns.PI = math.pi

# ns.ProjFun = 'sin(PI (1 - x_0))'
ns.ProjFun = 'sin(PI (1 - x_0)) sin(PI (1 - x_1))'
# ns.ProjFun = 'sin(PI (1 - x_0)^2) sin(PI (1 - x_1)^2)'

ns.ProjFun = '1 - tanh( ( sqrt( ( 2 x_0 - 1)^2 + ( 2 x_1 - 1 )^2  ) - 0.3 ) / ( 0.05 sqrt(2) )  )'

# pbox.refined_by_pbox([0])

ns.THBBasis = pbox.basis('th-spline', degree = degree)

# print(ns.THBBasis._coeffs)
# print(ns.THBBasis._dofs_shape)
# print(ns.THBBasis._transforms_shape)



# print(ns.THBBasis.get_dofs(0))
# print(ns.THBBasis._arg_dofs.eval(_index=0))
# print(ns.THBBasis.f_dofs_coeffs(0))
# # print(ns.THBBasis.f_dofs_coeffs(0)[0].func._na.value)
# # print(ns.THBBasis.f_dofs_coeffs(0)[0].func._nb.value)
#
# ns.BernBasis = pbox.basis("discont", degree = degree)
# print(ns.THBBasis.get_dofs(0))

#
#
#
#
#
# #
# pbox.refined_by_pbox([0])
# args = pbox.project(ns.ProjFun)
#
# print(args)
#
# ns.add_field('phi', ns.THBBasis)
# phi_data = PlotData(pbox.topology, pbox.geometry, ['phi', 'phi - ProjFun'], ns, args, (xx, yy) ,NumPoints)
# fig, axs  = plt.subplots(1,1)
#
# bezier = pbox.topology.sample('bezier',3)
# X = bezier.eval(['x_i'] @ ns, **args)
#
#
# x = [X[0][:,i] for i in range(2)]
#
# export.plotlines_(axs, x, bezier.hull, colors='k', linewidths=.1, alpha=.5)
# # im = axs.tripcolor(x[:, 0], x[:, 1], bezier.tri, x, shading='gouraud', rasterized=True)
#
# im = axs.contourf(xx,yy,phi_data[1])
# plt.colorbar(im)
#
# print(pbox.proj_elem)
#
# # error = phi_data[0] - g_data[0]
# #
# # print(f'error mean : {numpy.mean(error, axis=(0,1))}')
#
#
# #
# # plotFig = export.triplot(axs,x, phi, tri = bezier.tri, hull = bezier.hull)
# # plt.colorbar(plotFig)
#
# # plt.tricontourf(x[:, 0], x[:, 1], bezier.tri, phi)
# plt.show()
#
# quit()

eps = numpy.infty

eps_target = 1e-6
pbox.MaxL = 9

iterCount = 0

TimeList = []
NumList = []

while eps > eps_target:
    iterCount += 1
    print(f"starting iteration: {iterCount}")
    print(f"number of elements {len(pbox.topology.references)}")
    t0 = time.time()
    # args = pbox.project(ns.ProjFun, method="global")
    args = pbox.project(ns.ProjFun, method="bern", arguments=args)
    t1 = time.time()
    print(f"projection took {t1 - t0}")
    TimeList.append(t1 - t0)
    NumList.append(sum([len(subList) for subList in pbox.pbox_active_indices_per_level]))

    ns.THBBasis = pbox.basis('th-spline', degree = degree)

    ns.add_field('phi', ns.THBBasis)
    # ns.add_field('bern',ns.BernBasis)
    ns.define_for('x', gradient='D', normal='n', jacobians=('dV', 'dS'))

    pbox.elem_size(0)
    element_error = numpy.sqrt( pbox.topology.integrate_elementwise(' (phi - ProjFun)^2 dV' @ ns,degree = 2 * max(degree), arguments = args) / pbox.ElemSize )
    # element_error = numpy.sqrt( pbox.topology.integrate_elementwise(' (phi - ProjFun)^2 ' @ ns,degree = 4 * max(degree), arguments = args) )


    eps = numpy.max(element_error )
    print(f'===============================================================')
    print(f'max element error {eps}')
    print(f'===============================================================')

    elements_to_refine = numpy.where(element_error > eps_target)
    pbox_to_refine = numpy.unique(pbox.map_elem_pbox(elements_to_refine[0]))

    # print(f"Adding pboxes {[pbox_to_refine]}")



    _, altered = pbox.refined_by_pbox(pbox_to_refine)
    # _, altered = pbox.refined_by_pbox(pbox_to_refine[numpy.random.random(len(pbox_to_refine)) > 0.75])
    # if iterCount > 1:
    #     break

    if not altered:
        break



plt.plot(NumList,TimeList)
plt.show()


ns.add_field('phi', ns.THBBasis)
phi_data = PlotData(pbox.topology, pbox.geometry, ['phi', 'phi - ProjFun'], ns, args, (xx, yy) ,NumPoints)
fig, axs  = plt.subplots(1,1)

bezier = pbox.topology.sample('bezier',3)
X = bezier.eval(['x_i'] @ ns, **args)


x = [X[0][:,i] for i in range(2)]

export.plotlines_(axs, x, bezier.hull, colors='k', linewidths=.1, alpha=.5)

axs.contourf(xx,yy,phi_data[1])

print(numpy.mean(phi_data[0],axis=(0,1)))
print(numpy.var(phi_data[0],axis=(0,1)))

# error = phi_data[0] - g_data[0]
#
# print(f'error mean : {numpy.mean(error, axis=(0,1))}')


#
# plotFig = export.triplot(axs,x, phi, tri = bezier.tri, hull = bezier.hull)
# plt.colorbar(plotFig)

# plt.tricontourf(x[:, 0], x[:, 1], bezier.tri, phi)
plt.show()

