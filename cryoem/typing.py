'''
Provide indicative types for cryo-data
'''
from typing import TypeAlias, Literal
from numpy import ndarray

# geometric
Quat: TypeAlias = ndarray
'''Quaternion, 4-vector'''
Rot3D: TypeAlias = ndarray
'''Rotation Matrix, 3x3'''
Vec3D: TypeAlias = ndarray
'''3D Vector'''
Rot2D: TypeAlias = ndarray
'''2D Rotation Matrix'''
Projection: TypeAlias = ndarray 
'''3 x 2 `orthonormal` mapping'''

# distribution
Density: TypeAlias = ndarray
''' N x N (x N) array, one entry for each spatial point, e.g. a density'''
Voxels: TypeAlias = Density
'''Grid-like density in 3D, NxNxN array'''
Pixels: TypeAlias = Density
'''Grid-like density in 2D, NxN array'''
Grid: TypeAlias = ndarray[int]
''' Coordinates of a grid, shape (N,)*d + (d,), i.e. NxNxNx3 or NxNx2, cfr. `transposed meshgrid`'''
GridMask: TypeAlias = ndarray
'''flattened subset of a grid, N_Tx2 for a planar grid TODO perhaps implement with numpy.maskedarray'''

# interpolation
Kernel: TypeAlias = Literal['lanczos','sinc','linear','quad']
'''Kernel type for interpolation'''
