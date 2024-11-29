'''
Module to generate sample snapshots from a dataset
'''

from typing import List, Tuple, Self
from dataclasses import dataclass, InitVar, field

from scipy.sparse import sparray
import numpy as np

from cryoem.cryoem import random_unit_vector
from cryoem.cryoio.ctf import compute_full_ctf
from cryoem.cryoops import compute_projection_matrix, compute_shift_phases
from cryoem.geom import genEA, rotmat3D_EA, gentrunctofull
from cryoem.density import fspace_to_real

# Custom types
from cryoem.cryoem import Density
Voxels = Density
'''Grid-like density in 3D'''
Pixels = Density
'''Grid-like density in 2D'''

def random_euler_angles() -> np.ndarray[float]:
    '''generate euler angles, spherically uniformly distributed + psi'''
    angles = genEA(random_unit_vector())
    angles[2] = 2*np.pi*np.random.rand()
    return angles

@dataclass
class Snapshot:
    '''
    An instance represent records of input-parameters to generated pictures form a given
    3D spectral field.
    '''

    # statics
    defocus_min = 10000
    defocus_max = 20000
    shift_sigma = 0
    bfactor = 50.0
    counter = 0
    projection_opts = dict( kern = 'lanczos',
                            kernsize = 6,
                            radius = 0.95,
                            projdirtype = 'rots')

    # init
    euler_angles: InitVar[np.ndarray] = None
    '''Euler angles, in radians'''

    # input data
    amp_contrast: float = 0.07
    '''1. Amplitude contrast, constant'''
    phi: float = None
    '''2. Euler angle phi, in degrees, random per record'''
    theta: float = None
    '''3. Euler angle theta, in degrees, random per record'''
    psi: float = None
    '''4. Euler angle psi, in degrees, random per record'''
    class_numer: int = 1
    astig_angle: int = field(default_factory=lambda: np.random.uniform(0,360))
    '''6. Defocus angle, randomly generated per record'''
    defocus_a: int = None
    '''7. Defocus U'''
    defocus_b: int = None
    '''8. Defocus V'''
    pixel_size: int = None
    '''9. Pixel size, required!'''
    image_name: str = field(default_factory=lambda:f"{Snapshot.count()}@/simulated_particles.mrcs")
    '''10. File name'''
    magnification: float = 10000.0
    origin_x: int = 0
    origin_y: int = 0
    phase_shift: float = 0.0
    spherical_abberation: float = 2.7
    voltage: float = 300
    '''16. Voltage in kV, constant'''

    # auxiliary vars, defined in post_init
    shift = None
    projection = None

    # output data
    spectral_wave_2d = None
    wave_2d = None
    wave_2d_ctf = None
    wave_2d_noisy = None
    
    @classmethod
    def count(cls) -> int:
        cls.counter += 1
        return cls.counter

    def __post_init__(self, euler_angles: np.ndarray = None):
        euler_angles = random_euler_angles() if euler_angles is None else euler_angles
        self.phi, self.theta, self.psi = euler_angles * 180 / np.pi
        self.projection = rotmat3D_EA(*euler_angles)[:,:2]
        self.shift = np.random.randn(2) * self.shift_sigma
        base_defocus = np.random.uniform(self.defocus_min,self.defocus_max)
        self.defocus_a,self.defocus_b = base_defocus + np.random.uniform(-500,500,size=(2,))

    @classmethod
    def embedding(cls, box_size: int) -> sparray:
        return gentrunctofull(box_size, cls.projection_opts['radius'])

    def project(self,spectral_wave_3d: Voxels, boxSize: int, embedding: sparray) -> Pixels:
        '''Apply CTFS'''
        slop = compute_projection_matrix([self.projection], boxSize, **self.projection_opts)
        phase_factors = compute_shift_phases(self.shift.reshape((1,2)), boxSize, self.projection_opts['radius'])[0]
        spectral_wave_2d = phase_factors * slop.dot(spectral_wave_3d.reshape((-1,)))
        return (embedding @ spectral_wave_2d).reshape((boxSize,boxSize))
    
    def ctf_distorted(self, spectral_wave_2d: Pixels) -> Pixels:
        '''Generate the CTF'''
        boxSize = len(spectral_wave_2d)
        ctfs = compute_full_ctf(None, boxSize, self.pixel_size, self.voltage, self.spherical_abberation, self.amp_contrast, self.defocus_a, self.defocus_b, np.radians(self.astig_angle), 1, self.bfactor)
        return fspace_to_real(ctfs.reshape((boxSize,boxSize)) * spectral_wave_2d)
    
    @classmethod
    def pixel_noise(cls, boxSize: int) -> Pixels:
        return np.require(np.random.randn(boxSize, boxSize)*10,dtype=np.float32)
    
    def generate_images(self,spectral_wave_3d: Voxels, boxSize: int, embedding: sparray) -> Self:
        '''Add relevant 2D images to this instance'''
        self.spectral_wave_2d = self.project(spectral_wave_3d, boxSize, embedding)
        self.wave_2d = fspace_to_real(self.spectral_wave_2d)
        self.wave_2d_ctf = self.ctf_distorted(self.spectral_wave_2d)
        self.wave_2d_noisy = self.wave_2d_ctf + self.pixel_noise(boxSize)
        return self
