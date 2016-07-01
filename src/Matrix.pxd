cimport numpy as np

cdef np.ndarray toV3(np.ndarray V)
cdef np.ndarray to4x1(np.ndarray V, float fill_v = ?)
cdef np.ndarray to3x1(np.ndarray V)
cdef np.ndarray unitV(np.ndarray V)
cdef np.ndarray rotM(float ax, float ay, float az)
cdef np.ndarray transM(int dx, int dy, int dz)
cdef np.ndarray forward(int a, int b, int c, np.ndarray M, np.ndarray C)
cdef np.ndarray inverse(int d, int e, int f, np.ndarray M, np.ndarray C)