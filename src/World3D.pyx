from Region cimport *
from Data cimport *

cimport numpy as np

cdef Region Cube(int block_id, int dx, int dy, int dz,
                 np.ndarray block_ang = np.array([0,0,0]), int layer = 0,
                 np.ndarray location_xyz = np.array([0,0,0]),
                 np.ndarray rotation_angs = np.array([0,0,0]), np.ndarray rotation_ijk = None):
    cdef np.ndarray R = np.ones((dx, dy, dz, 3), dtype="float") * block_ang.reshape((1,1,1,3))  # Set the rotation matrix using r vector
    cdef np.ndarray L = np.full((dx, dy, dz), fill_value = layer)                               # Initialize layer with layer value
    cdef np.ndarray V = np.full((dx, dy, dz), fill_value = block_id)                            # Initialize the value with block_id
    return MakeRegion(V, R, L, location_xyz, rotation_angs, rotation_ijk)                       # Create the region