import numpy as np
from nutils import function
def ProjectPrevSolutions(args, topo, ns, nsRef, geom, p, refargs):
    args['p'] = topo.project(ns.p, nsRef.Basis3, geom, degree = 2* p, arguments=refargs)
    args['pP'] = topo.project(ns.pP, nsRef.Basis3, geom, degree=2 * p, arguments=refargs)
    args['r'] = topo.project(ns.r, nsRef.Basis3, geom, degree=2 * p, arguments=refargs)
    args['u'] = topo.project(ns.u, nsRef.Basis2, geom, degree=2 * p, arguments=refargs)
    args['B'] = topo.project(ns.B, nsRef.Basis2, geom, degree=2 * p, arguments=refargs)
    args['E'] = topo.project(ns.E, nsRef.Basis1, geom, degree=2 * p, arguments=refargs)

    args['u0'] = args['u']
    args['B0'] = args['B']
    return args


def RefineTopology(ns, topo, geom, args):
    p = ns.degreep.eval()
    refbasis = topo.basis('h-spline', degree=[p, p])
    ns.add_field('vref', refbasis)
    res = topo.integral(' vref ( sum_i uP_i^2 ) dV' @ ns, degree=9 * p)
    indicator = res.derivative('vref').eval(**args)

    elem_volume = topo.integrate_elementwise(function.J(geom), degree=1)
    for i in range(indicator.size):
        supp = refbasis.get_support(i)
        indicator[i] = elem_volume[supp].sum()**2 * indicator[i]

    # print(indicator)
    # indicator[indicator < np.mean(indicator)] = 0
    # index = np.argsort(indicator)
    # sort_indicator = indicator[index]
    # cumsum_sort_indicator = np.cumsum(sort_indicator)
    # N = sum(cumsum_sort_indicator < 0.05 * cumsum_sort_indicator[-1])
    #
    # indicator_dorfler = 0*indicator
    # indicator_dorfler[index[0:N-1]] = 1

    supp = refbasis.get_support(indicator > 0.25 * np.mean(indicator))
    # supp = refbasis.get_support(indicator_dorfler == 1)

    return topo.refined_by(supp)