from typing import Tuple, Literal

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from scipy.sparse import csr_array

from geom import gencoords

# Custyom types
from geom import Rot3D, Vec3D
Density = np.ndarray
''' N x N x N array, one entry for each spatial point, e.g. a density'''

def compute_density_moments(density: Density, mu: Vec3D = None) -> Tuple[np.ndarray,np.ndarray]:
    '''calculate moments of the coordinate/frequency matrix TODO misleading function name'''
    # Regularize
    density = np.abs(density) # just to be sure, should always be the case
    density /= density.sum()

    # Reshape
    N = len(density)
    density = density.reshape((-1,1))
    coords = gencoords(N,3).reshape((-1,3))

    # Mean
    mu = (density * coords).sum(axis=0) if mu is None else mu # 3-vector

    # Covariance
    wave_function = np.sqrt(density) * (coords - mu) # (N^3,3) array
    covar = wave_function.T @ wave_function / N**3 # (3,3) array

    return mu, covar

def rotate_density(density: Density, R: Rot3D, t: float = 0.0, upsamp: float = 1.0) -> Density:
    assert len(density.shape) == 3

    N = density.shape[0]
    Nup = int(N*upsamp)
    coords = gencoords(Nup,3) / upsamp

    interp_coords = ((coords @ R.T) + t).transpose((3,0,1,2)) + N/2
    return map_coordinates(density, interp_coords, order=1)

def align_density(density: Density, upsamp: float = 1.0) -> Tuple[Density, Rot3D]:
    '''
    Rotate a density to its `proper frame` aligned with its covariance matrix.

    The X axis will be the least spread out direction, the Z axis the most.
    '''
    assert len(density.shape) == 3

    mu, covar = compute_density_moments(density)

    _,V = np.linalg.eigh(covar)
    assert np.linalg.det(V) > 0

    return rotate_density(density,V,mu,upsamp), V

def radial_histogram(density: Density, integer_radii: Density, fractional_radii: Density, max_radius: float) -> np.ndarray:
    '''count coordinates within an radial integer bin, weighted by their proximity to the target radius'''
    if np.iscomplexobj(density):
        real = radial_histogram(density.real,integer_radii, fractional_radii, max_radius)
        imag = radial_histogram(density.imag,integer_radii, fractional_radii, max_radius)
        return real + 1.0j * imag
    
    floor_contributions = np.bincount(integer_radii, 
                                        weights=(1-fractional_radii)*density.imag, 
                                        minlength=max_radius)[:max_radius]
    ceil_contributions = np.bincount(integer_radii + 1, 
                                        weights=fractional_radii*density.imag, 
                                        minlength=max_radius)[:max_radius]
    return floor_contributions + ceil_contributions
    
def rotational_average(density: Density, 
                       maxRadius: float = None, 
                       doexpand: bool = False, 
                       normalize: bool = True, 
                       return_cnt: bool = False) -> np.ndarray | Tuple[np.ndarray | Density, np.ndarray]:
    '''
    Returns occupation numbers of (integer) radial slices:
    * weighted by density [normalize: relative to unweighted | doexpand: return interpolated values for the full grid]
    * [return_cnt: unweighted occupation numbers]
    '''
    N = density.shape[0]
    D = len(density.shape)
    
    assert D >= 2, 'Cannot rotationally average a 1D array'

    pts = gencoords(N,D)
    radius_grid: Density = np.linalg.norm(pts,axis=-1)
    radius_grid_int = np.require(np.floor(radius_grid),dtype='uint32')
    radius_grid_frac = radius_grid - radius_grid_int

    if maxRadius is None:
        maxRadius = np.ceil(N/np.sqrt(D))

    if maxRadius < np.max(radius_grid_int)+2:
        valid_radius = radius_grid_int+1 < maxRadius
        radius_grid_int = radius_grid_int[valid_radius]
        radius_grid_frac = radius_grid_frac[valid_radius]
        density = density[valid_radius]
    
    raps = radial_histogram(density,radius_grid_int,radius_grid_frac,maxRadius)

    if normalize or return_cnt:
        cnt = radial_histogram(np.ones(density.shape),radius_grid_int,radius_grid_frac,maxRadius)

    if normalize:
        raps[cnt > 0] /= cnt[cnt > 0]

    if doexpand:
        raps = radial_expand(raps,radius_grid)
    
    return raps, cnt if return_cnt else raps

def radial_expand(radial_density: np.ndarray, radius_grid: Density, interp_order: int = 1) -> Density:
    '''1D interpolation of integer spaced values in radial_density by target values in radial_grid'''
    return map_coordinates(radial_density, radius_grid[None], order=interp_order, mode='nearest')

def resize_ndarray(density: Density, new_shape: np.ndarray[float], axes: np.ndarray[int]):
    '''reduce size of array by zooming out after blurring NOTE doesn't seem useful for upsampling, right?'''
    zoom_factors = [new_size / old_size if axis in axes else 1 
                    for old_size, new_size, axis in zip(density.shape, new_shape,range(42))]
    sigmas = [0.66 / zoom_factor if axis in axes else 0 for axis, zoom_factor in enumerate(zoom_factors)]

    blurred_density = gaussian_filter(density, sigma=sigmas, order=0, mode='constant')
    return zoom(blurred_density, zoom_factors, order=0)

def compute_fsc(VF1,VF2,maxrad,width=1.0,thresholds = [0.143,0.5]):
    assert VF1.shape == VF2.shape
    N = VF1.shape[0]
    
    r = np.sqrt(np.sum(gencoords(N,3).reshape((N,N,N,3))**2,axis=3))
    
    prev_rad = -np.inf
    fsc = []
    rads = []
    resInd = len(thresholds)*[None]
    for i,rad in enumerate(np.arange(1.5,maxrad*N/2.0,width)):
        cxyz = np.logical_and(r >= prev_rad,r < rad)
        cF1 = VF1[cxyz] 
        cF2 = VF2[cxyz]
        
        if len(cF1) == 0:
            break
        
        cCorr = np.vdot(cF1,cF2) / np.sqrt(np.vdot(cF1,cF1)*np.vdot(cF2,cF2))
        
        for j,thr in enumerate(thresholds):
            if cCorr < thr and resInd[j] is None:
                resInd[j] = i
        fsc.append(cCorr.real)
        rads.append(rad/(N/2.0))
        prev_rad = rad

    fsc = np.array(fsc)
    rads = np.array(rads)

    resolutions = []
    for rI,thr in zip(resInd,thresholds):
        if rI is None:
            resolutions.append(rads[-1])
        elif rI == 0:
            resolutions.append(np.inf)
        else:
            x = (thr - fsc[rI])/(fsc[rI-1] - fsc[rI])
            resolutions.append(x*rads[rI-1] + (1-x)*rads[rI])
    
    
    return rads, fsc, thresholds, resolutions


# So the key is to make sure that the image is zero at the nyquist frequency (index n/2)
# The interpolation idea is to assume that the actual function f(x,y) is band-limited i.e.
# made up of exactly the frequency components in the FFT. Since we are interpolating in frequency space, 
# The assumption is that in frequency space the signal F(wx,wy) is band-limited. 
# This means that it's fourier transform should have components less than the nyquist frequency.
# But the fourier transform of F(wx,wy) is ~f(x,y) since FFT and iFFT are same. So f(x,y) must be nonzero at the nyquist frequency (and preferrably even less than that) which means in image space, the n/2 row and n/2 column (and n/2 page). 
# since the image will be zero at the edges once some windowing (circular or hamming etc) is applied,
# we can just fftshift the image since translations do not change the FFT except by phase. This makes the nyquist components
# zero and everything is fine and dandy. Even linear iterpolation works then, except it leaves ghosting.
  
def getslices(V, SLOP) -> np.ndarray:
    return csr_array(SLOP) @ V.reshape(-1)
            
# 3D Densities
# ===============================================================================================    

def window( volume: Density, 
            func: Literal['hanning','hamming','circle','box'] = 'hanning', 
            params = [1.0]):
    """ applies a windowing function to the 3D volume v (inplace, as reference) """
    
    N = volume.shape[0]
    D = volume.ndim
    if any( [ d != N for d in list(volume.shape) ] ) or D != 3:
        raise Exception("Error: Volume is not Cube.")
    
    def apply_seperable_window (volume: Density, window_1d: np.ndarray[float]):
        volume *= window_1d[:,None,None] * window_1d[None,:,None] * window_1d
    
    match func:
        case "hanning":
            w = np.hanning(N)
            apply_seperable_window(volume,w)
        case 'hamming':
            w = np.hamming(N)
            apply_seperable_window(volume,w)
        case 'gaussian':
            raise NotImplementedError()
        case 'circle':
            coordinates = gencoords(N,3)
            r = params[0] * (N/2 - 1)
            volume[np.sum(coordinates**2 , -1)  < ( r ** 2 )] = 0.0
        case 'box':
            volume[:,0,0] = 0.0
            volume[0,:,0] = 0.0
            volume[0,0,:] = 0.0
        case _:
            raise Exception("Error: Window Type Not Supported")

def random_unit_vector(size = None) -> Vec3D:
    '''sample a sphericaly uniform unit vector'''
    vectors = np.random.multivariate_normal(mean = np.zeros(3), cov=np.eye(3), size = size)
    vectors /= np.linalg.norm(vectors,axis=-1,keepdims=True)
    return vectors

def generate_phantom_density(N: int, window_radius: float, sigma: float, num_blobs: int, seed: int = None) -> Density:
    '''sequentially add Gaussian noise around a centr-ish point in space to form a density'''
    if seed is not None:
        np.random.seed(seed)
    density = np.zeros((N,N,N),dtype=np.float32)

    coords = gencoords(N,3)
    inside_window = np.sum(coords**2,axis=-1) < window_radius ** 2

    def add_gaussian_noise(density: Density, center: Vec3D, sigma: float):
        coord_radii = np.linalg.norm(coords - center,axis=-1)
        inside = (coord_radii < 3*sigma) & inside_window
        density[inside] += np.exp(-0.5*(coord_radii[inside]/sigma**2))  

    def update_center(center: Vec3D, sigma: float) -> Vec3D:
        '''search new center within window by moving back a forth relative to a chosen direction'''
        delta_dir = random_unit_vector()
        center += 2.0 * sigma * delta_dir
        while (center_radius := np.linalg.norm(center)) > window_radius:
            center_dir = center/center_radius
            update_dir = 2 * delta_dir.dot(center_dir) * center_dir - delta_dir # flip delta about center, in their common plane
            center[:] = center_dir + (center_radius - window_radius) * update_dir

    moving_center = np.zeros((3,))
    blob_count = 0
    while (blob_count := blob_count + 1) <= num_blobs:
        inflated_sigma = sigma * np.exp(0.25 * np.random.randn())
        add_gaussian_noise(density, moving_center, inflated_sigma)
        update_center(moving_center, inflated_sigma)

    return density
