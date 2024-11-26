'''
Module with geometric utilities:
* conversions between rotational representations 
* generators of an Nd meshgrid
'''

from typing import Literal, Tuple, List
from functools import cache as memoize

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_array


# Custom types for readability
Quat = np.ndarray
'''Quaternion'''
Rot3D = np.ndarray
'''Rotation Matrix'''
Vec3D = np.ndarray
'''3D Vector'''
Rot2D = np.ndarray
'''2D Rotation Matrix'''

def rotmat2D(theta: float) -> Rot2D:
    '''2D rotation'''
    return np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])

def rotmat3D_to_quat(mat: Rot3D) -> Quat:
    '''quaternion from rotation matrix'''
    return Rotation.from_matrix(mat).as_quat(scalar_first=True)

def rotmat3D_quat(q: Quat) -> Rot3D:
    '''Rotation matrix from quaternion'''
    return Rotation.from_quat(q, scalar_first=True).as_matrix()

def rotmat3D_EA(phi: float, theta: float, psi: float = None) -> Rot3D:
    """
    Generates a rotation matrix from Z-Y-Z Euler angles. This rotation matrix
    maps from image coordinates (x,y,0) to view coordinates and should be
    consistent with JLRs code.
    """
    R = Rotation.from_euler('ZY',[phi,theta]) if psi is None else Rotation.from_euler('ZYZ',[phi,theta,psi])
    return R.as_matrix()

def rotmat3D_dir(normal: Vec3D) -> Rot3D:
    '''
    Rotate the normal plane to XY in a `minimal` way, 
    i.e. keeping new and old XY `as aligned as possible`.
    
    normal: 3D vector
    '''
    phi = np.arctan2(normal[1],normal[0])
    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    return Rotation.from_euler('ZYZ',[phi,theta,-phi]).as_matrix()

def rotmat3D_expmap(rotvec: Vec3D) -> Rot3D:
    '''rotation matric for rotation vector'''
    return Rotation.from_rotvec(rotvec).as_matrix()

def genDir(EAs: List[Tuple[float,float]]):
    """
    Generate the projection direction given the euler angles.  Since the image
    is in the x-y plane, the projection direction is given by R(EA)*z where 
    z = (0,0,1)
    """
    
    return np.stack([rotmat3D_EA(*EA)[:,2] for EA in EAs],axis=0)

def genEA(directions):
    """
    Generates euler angles from a vector direction
    p is a column vector in the direction that the new x-axis should point
    returns tuple (phi, theta, psi) with psi=0
    """
    assert directions.shape[-1] == 3
    theta = np.arctan2(np.linalg.norm(directions[...,:2],axis=-1),directions[...,2])
    phi = np.arctan2(directions[...,1],directions[...,0])
    
    return np.stack([phi,theta,np.zeros_like(theta)], axis=-1)


@memoize
def gencoords_base(N: int, d: int):
    '''
    Return a d-dim freqency base between -N/2 and N/2
    
    Parameters:
    -----------
    N: band width
    d: dimension
    '''
    frequencies = np.linspace(-N//2, N//2, N, endpoint = False)
    return np.stack(np.meshgrid(*[frequencies]*d),-1).reshape(-1,d)

@memoize
def gencoords(N: int, 
              d: int, 
              radius: float = None, 
              trunctype: Literal['circ','square'] = 'circ') -> np.ndarray[int]:
    """ generate coordinates of all points in an Nxnp..xN grid with d dimensions 
    coords in each dimension are [-N/2, N/2) 
    N should be even"""
    
    frequencies = gencoords_base(N,d) # (N^d, d) array

    if radius is None:
        return frequencies

    if trunctype == 'circ':
        r2 = np.sum(frequencies**2,axis=1)
        trunkmask = r2 < (radius*N/2.0)**2
    elif trunctype == 'square':
        r = np.max(np.abs(frequencies),axis=1)
        trunkmask = r < (radius*N/2.0)
        
    return frequencies[trunkmask,:]
  
def gentrunctofull(N: int, radius: float) -> coo_array:
    full = gencoords_base(N,2)
    mask = (np.sum(full**2,axis=-1) < (radius*N/2.0)**2).flatten()
    rows, = np.nonzero(mask)
    N_T = len(rows)
    cols = np.arange(N_T)
    data = np.ones((N_T,))
    return coo_array((data,(rows,cols)),(N**2,N_T))
