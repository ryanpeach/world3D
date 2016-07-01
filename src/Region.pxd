from Matrix cimport *
from Data cimport *

cimport numpy as np

# Create Vectorized combination algorithm
cdef float layer_select(Region r0, Region r1, int x, int y, int z, np.ndarray M0, np.ndarray M1, float fill = ?)
        
cdef class Region:
    cpdef np.ndarray spaceM, angleM, layerM
    cpdef np.ndarray shape, center_ijk, center_uvw, center_xyz
    cpdef np.ndarray U, X
    
    cpdef rotate(self, np.ndarray rotation_ang, np.ndarray vector_ijk = ?)
    
    cpdef move(self, np.ndarray vector_xyz)
    cpdef place(self, np.ndarray location_xyz)
    cpdef flatten(self, int l = ?)
    cpdef np.ndarray ijk_uvw(self, int i, int j, int k)
    cpdef np.ndarray uvw_ijk(self, int u, int v, int w)
    cpdef np.ndarray uvw_xyz(self, int u, int v, int w)
    cpdef np.ndarray xyz_uvw(self, int x, int y, int z)
    cpdef np.ndarray ijk_xyz(self, int i, int j, int k)
    cpdef np.ndarray xyz_ijk(self, int x, int y, int z)

    cpdef tuple getCorners(self)
        
    cpdef valid(self, int i, int j, int k)
    
    cpdef np.ndarray getAngle(self)
        
    cpdef np.ndarray getAngleM(self)
        
    cpdef Region add(self, Region other)
    
    cpdef Region copy(self)
    
cpdef Region MakeRegion(np.ndarray spaceM, np.ndarray angleM, np.ndarray layerM = ?, 
                       np.ndarray location_xyz = ?, 
                       np.ndarray rotation_angs = ?, 
                       np.ndarray rotation_ijk = ?)