## example of get_dofs acting slow for THB-splines
import PboxTopology
import time

# in this script, the get_dofs function is surprisingly slow, especially compared to get_coeffs.

# the topology is generated from a pbox mesh. This is simply a higher level mesh, where every pbox contains p_i elements
# in every direction. Meaning that for the cubic case considered in this script, the pbox consists of 9 mesh elements.
# the PboxTopology class generates a pbox mesh, and from this, generates a nutils Hierarchical topology.

# the pbox code has been structered in a similar fashion as the hierarchical topology.

# in my code, I have extracted the following example and timings:
# THB basis generation: 5.68693 seconds
#    number of elems: 13077
#    number of dofs: 11970
# dofs per element list calc: 11.246 seconds
# coeffs per element list calc: 0.626274 seconds

# the pbox_refinements array consists of the pboxes refined at the various iterations steps.
pbox_refinements = [[0],
                    [0, 1, 2, 3, 4, 5, 6],
                    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27],
                    [ 2,  8,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                           27, 29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 47, 48, 59, 62, 63, 65,
                           66, 67, 73, 74, 75, 79, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 93,
                           94, 95, 96, 97, 98, 99],
                    [ 35,  40,  41,  42,  48,  49,  50,  51,  58,  59,  60,  61,  62,
                            66,  67,  68,  69,  70,  75,  76,  77,  78,  79,  85,  86,  87,
                            88,  89,  90,  91,  92,  93,  94,  95, 102, 103, 104, 105, 106,
                           107, 108, 109, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123,
                           124, 125, 130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144,
                           145, 174, 180, 185, 186, 190, 191, 192, 193, 195, 196, 197, 198,
                           202, 203, 204, 205, 206, 207, 208, 213, 214, 215, 216, 217, 218,
                           224, 225, 226, 227, 228, 229, 230, 235, 236, 237, 238, 239, 240,
                           241, 242, 248, 249, 250, 251, 252, 253, 254, 255, 256, 260, 261,
                           262, 263, 264, 265, 266, 267, 268, 269, 270],
                    [ 44,  75,  76, 127, 137, 138, 139, 140, 146, 147, 148, 149, 150,
                           152, 156, 157, 158, 159, 160, 161, 162, 169, 170, 171, 172, 173,
                           174, 175, 184, 185, 186, 187, 188, 189, 190, 191, 199, 200, 201,
                           202, 203, 204, 205, 206, 214, 215, 216, 217, 218, 219, 220, 221,
                           227, 228, 229, 230, 231, 232, 233, 234, 240, 241, 242, 243, 244,
                           245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 263,
                           264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276,
                           277, 278, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293,
                           294, 295, 296, 297, 298, 304, 305, 306, 307, 308, 309, 310, 311,
                           312, 313, 314, 315, 316, 317, 325, 326, 327, 328, 329, 330, 331,
                           332, 333, 334, 335, 336, 344, 345, 346, 347, 348, 349, 350, 351,
                           352, 353, 354, 355, 356, 357, 361, 363, 364, 365, 366, 367, 368,
                           369, 370, 372, 376, 377, 378, 379, 380, 381, 420, 424, 425, 430,
                           431, 436, 439, 440, 441, 446, 449, 456, 462, 463, 468, 469, 470,
                           480, 481, 482, 483, 484, 491, 492, 493, 494, 495, 496, 497, 502,
                           503, 504, 505, 506, 507, 508, 509, 510, 511, 516, 517, 518, 519,
                           520, 521, 522, 523, 528, 529, 530, 531, 532, 536, 537, 538, 539,
                           543, 544, 545, 546, 547, 552, 558, 559, 560, 561, 562, 568, 576,
                           577, 578, 579, 586, 593, 594, 595, 596, 597, 604, 615, 616, 617,
                           618, 619, 626, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640,
                           641, 642, 643, 644, 645, 646, 647, 648]]

# r_elems is the starting number of pboxes per dimension
r_elems = (2, 2)
# degree is the spline degree
degree  = (3, 3)
# the starting topology is thus a tensor mesh of 6 elements per dimension

# generate the pbox class
pbox = PboxTopology.Pbox(degree,r_elems)
pbox.MaxL = 9

# refine the pbox mesh via pbox_refinements (the implementation of this is inspired by the refinement code from nutils)
for refinements in pbox_refinements:
    pbox.refined_by_pbox(refinements)

# generate the THB-spline basis
t0 = time.time()
THB_basis = pbox.topology.basis('th-spline', degree = degree)
print(f'Creating THB basis     : {time.time() - t0:0.6} seconds')

# check elements counts and dofs counts
print(f'   number of elems     : {THB_basis.nelems}')
print(f'   number of dofs      : {THB_basis.ndofs}')

# get the dofs list
t0 = time.time()
dofs_element = [THB_basis.get_dofs(ielem) for ielem in range(THB_basis.nelems)]
print(f'generating dofs list   :  {time.time() - t0:0.6} seconds')

# get the coefficients list
t0 = time.time()
dofs_element = [THB_basis.get_coefficients(ielem) for ielem in range(THB_basis.nelems)]
print(f'generating coeffs list : {time.time() - t0:0.6} seconds')

# get the dofs list
t0 = time.time()
dofs_element = [THB_basis.get_dofs(ielem) for ielem in range(THB_basis.nelems)]
print(f'generating dofs list   :  {time.time() - t0:0.6} seconds')

# get the coefficients list
t0 = time.time()
dofs_element = [THB_basis.get_coefficients(ielem) for ielem in range(THB_basis.nelems)]
print(f'generating coeffs list : {time.time() - t0:0.6} seconds')




# on my laptop, this script results in the following timings:
# Creating THB basis     : 5.37516 seconds
#    number of elems     : 13077
#    number of dofs      : 11970
# generating dofs list   :  9.91507 seconds
# generating coeffs list : 0.565187 seconds

# python code used:
# python 11
# - packages
# Pillow	10.1.0
# appdirs	1.4.4
# bottombar	2.0.2
# contourpy	1.2.0
# cycler	0.12.1
# fonttools	4.45.1
# kiwisolver	1.4.5
# matplotlib	3.8.2
# numpy	1.26.4
# nutils	8.6
# nutils-poly	1.0.0
# packaging	23.2
# pip	23.3.1
# psutil	5.9.6
# pyparsing	3.1.1
# python-dateutil	2.8.2
# scipy	1.12.0
# setuptools	68.0.0
# six	1.16.0
# stringly	1.0b3
# treelog	1.0	1.0
# typing-extensions	4.8.0
# wheel	0.41.2