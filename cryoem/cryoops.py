'''
Utilities for interpolating projected densities.
'''

from typing import Literal, Iterable

import numpy as np
from numpy.fft import fftshift, ifft, ifftshift

from geom import rotmat3D_dir, memoize, rotmat2D, gencoords
from sincint import compute_interpolation_matrix

# Custom types
from geom import Vec3D, Rot2D
Projection = np.ndarray 
'''3 x 2 `orthonormal` mapping'''
Kernel = Literal['lanczos','sinc','linear','quad']
'''Kernel type for interpolation'''

@memoize
def to_xy(normal: Vec3D) -> Projection:
    '''projection onto perp plane, cached with memoize'''
    return rotmat3D_dir(normal)[:,:2]

@memoize
def rot_2d(angle: float) -> Rot2D:
    '''2D rotation over angle, cached with memoize'''
    return rotmat2D(np.require(angle,dtype=np.float32))

def compute_projection_matrix(projdirs: Iterable[Vec3D],
                              N: int,
                              kern: Kernel,
                              kernsize: int,
                              radius : float,
                              projdirtype: Literal['dirs','rots'] = 'dirs',
                              sym=None, 
                              onlyRs: bool = False, 
                              **kwargs):
    projdirs = np.require(projdirs,dtype=np.float32)

    match projdirtype:
        case 'dirs':
            projections = list(map(to_xy,projdirs))
        case 'rots':
            assert projdirs[0].shape == (3,2), f'Invalid rotations, shape is {projdirs[0].shape}, must be 3x2'
            projections = projdirs
        case _:
            raise ValueError('Unknown projdirtype, must be either dirs or rots')

    if sym is None:
        symRs = None
    else:
        symRs = np.stack([ np.require(R,dtype=np.float32).reshape((3,3)) for R in sym.get_rotations()],axis = 0)

    if onlyRs:
        return projections

    return compute_interpolation_matrix(projections,N,N,radius,kern,kernsize,symRs)

def compute_inplanerot_matrix(thetas: Iterable[float],
                              N: int,
                              kern,
                              kernsize: int,
                              radius: float,
                              N_src: int = None,
                              onlyRs: bool = False):
    
    N_src = N if N_src is None else N_src
    scale = N / N

    rotations = np.stack([scale * rot_2d(th) for th in thetas], axis = 0)

    if onlyRs:
        return rotations
    
    return compute_interpolation_matrix(rotations,N,N_src,radius,kern,kernsize,None)

def compute_shift_phases(pts: np.ndarray, N: int, rad: float) -> np.ndarray:
    xy = gencoords(N,2,rad)
    return np.exp(2.0j*np.pi/N * (pts @ xy.T))

def compute_premultiplier(N: int, kernel: Kernel, kernsize: int, scale: int = 512) -> np.ndarray:
    krange = N // 2
    koffset = krange * scale
    grid = np.linspace(-krange,krange,2*koffset)

    match kernel:
        case 'lanczos':
            a = kernsize/2
            spectrum = np.sinc(grid)*np.sinc(grid/a)*(np.abs(grid) <= a)
        case 'sinc': 
            a = kernsize/2.0
            spectrum = np.sinc(grid)*(np.abs(grid) <= a)
        case 'linear':
            assert kernsize == 2
            spectrum = np.maximum(0.0, 1 - np.abs(grid))
        case 'quad':
            assert kernsize == 3
            spectrum = (np.abs(grid) <= 0.5) * (1-2*grid**2) + ((np.abs(grid)<1)*(np.abs(grid)>0.5)) * 2 * (1-np.abs(grid))**2
        case _ :
            raise ValueError('Unknown kernel type')

    density = fftshift(ifft(ifftshift(spectrum))).real
    return 1.0/(N*density[koffset-krange:koffset+krange])
    

if __name__ == '__main__':
    
    kern = 'sinc'
    kernsize = 3
    
    N = 128
    
    pm1 = compute_premultiplier(N,kern,kernsize,512)
    pm2 = compute_premultiplier(N,kern,kernsize,8192)
    
    print(np.max(np.abs(pm1-pm2)))
    


