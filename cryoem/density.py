
from math import prod

import numpy as np
from numpy.fft import fftshift, ifftshift

# Custom types
from .typing import Density, Voxels, Projection, Grid, GridMask

def get_fft_module():
    try:
        import pyfftw
        from multiprocessing import cpu_count
        return pyfftw, cpu_count()
    except:
        print("ERROR LOADING FFTW! USING NUMPY")
        from numpy import fft
        return fft, 0

class FFT:
    module, threads = get_fft_module()

    @classmethod
    def using_fftw(cls) -> bool:
        return cls.threads > 0

    @classmethod
    def with_threads(cls, kwargs: dict) -> dict:
        if not cls.using_fftw():
            if 'threads' in kwargs:
                del kwargs['threads']
            return kwargs
        threads = kwargs.get('threads')
        kwargs['threads'] = cls.threads if threads is None else threads
        return kwargs

    @classmethod
    def fftn(cls,*args, **kwargs):
        cls.module.fft.__doc__
        return cls.module.fftn(*args, **cls.with_threads(kwargs))
    
    @classmethod
    def ifftn(cls,*args, **kwargs):
        cls.module.ifft.__doc__
        return cls.module.ifftn(*args, **cls.with_threads(kwargs))
    
    @classmethod
    def empty(cls, shape: tuple, dtype = np.float32):
        if cls.using_fftw():
            return cls.module.n_byte_align_empty(shape,32,dtype)
        return np.empty(shape,dtype)
    
    @classmethod
    def zeros(cls, shape: tuple, dtype = np.float32):
        if cls.using_fftw():
            res = cls.module.n_byte_align_empty(shape,32,dtype)
            res[:] = 0
            return res
        return np.zeros(shape,dtype)

def real_to_fspace(wave: Density ,axes: tuple = None, threads: int = None) -> Density:
    """ Convert real-space M to (unitary) Fourier space """

    spectral_wave = np.require(fftshift(FFT.fftn(fftshift(wave,axes),axes,threads=threads),axes), dtype=np.complex64)
    nrm = np.sqrt(wave.size if axes is None else prod(wave.shape[axis] for axis in axes))
    return spectral_wave / nrm

def fspace_to_real(spectral_wave: Density, axes: tuple = None, threads: int = None) -> Density:
    """ Convert unitary Fourier space fM to real space """

    ret = np.require(ifftshift(FFT.ifftn(ifftshift(spectral_wave,axes),axes,threads=threads),axes), dtype=np.float32)
    nrm = np.sqrt(spectral_wave.size if axes is None else prod(spectral_wave.shape[axis] for axis in axes))
    return ret / nrm

def conjugate(spectral_wave: Density) -> Density:
    res = np.flip(spectral_wave).conj()
    if len(spectral_wave) %2:
        return res
    return np.roll(res, axis=(0,1,2),shift=1)

def make_hermitian(spectral_wave: Density) -> Density:
    return (spectral_wave + conjugate(spectral_wave))/2

def check_hermitian(spectral_wave: Density) -> Density:
    return np.linalg.norm(np.absolute(spectral_wave-conjugate(spectral_wave)))

