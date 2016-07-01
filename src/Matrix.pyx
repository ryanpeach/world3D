cdef np.ndarray toV3(np.ndarray V):
    cdef int dims = V.ndim
    if dims==2 and (V.shape[0] == 4 or V.shape[0] == 3) and V.shape[1] == 1:
        return np.array([V[0,0],V[1,0],V[2,0]], dtype=V.dtype)
    elif dims==1 and V.shape[0] == 4:
        return V[:3]
    elif dims==1 and V.shape[0] == 3:
        return V
    else:
        raise Exception("Input not the right shape.")
        
cdef np.ndarray to4x1(np.ndarray V, float fill_v = 0.):
    V = toV3(V)
    return np.array([[V[0]],[V[1]],[V[2]],[fill_v]], dtype="float")
    
cdef np.ndarray to3x1(np.ndarray V):
    V = toV3(V)
    return np.array([[V[0]],[V[1]],[V[2]]])
    
cdef np.ndarray unitV(np.ndarray V):
    V = toV3(V)
    return V / np.linalg.norm(V)
    
cdef np.ndarray rotM(float ax, float ay, float az):
    cdef float c1 = np.cos(ax), c2 = np.cos(ay), c3 = np.cos(az)
    cdef float s1 = np.sin(ax), s2 = np.sin(ay), s3 = np.sin(az)
    cdef np.ndarray R1 = np.array([[1,0,0,0]   ,[0,c1,s1,0],[0,s1,c1,0] ,[0,0,0,1]], dtype="float")
    cdef np.ndarray R2 = np.array([[c2,0,s2,0] ,[0,1,0,0]  ,[-s2,0,c2,0],[0,0,0,1]], dtype="float")
    cdef np.ndarray R3 = np.array([[c3,-s3,0,0],[s3,c3,0,0],[0,0,1,0]   ,[0,0,0,1]], dtype="float")
    return np.dot(np.dot(R1,R2),R3)
    
cdef np.ndarray transM(int dx, int dy, int dz):
    return np.array([[1,0,0,dx],[0,1,0,dy],[0,0,1,dz],[0,0,0,1]], dtype="int")

cdef np.ndarray forward(int a, int b, int c, np.ndarray M, np.ndarray C):
    cdef np.ndarray cV = to4x1(C,0)
    cdef np.ndarray V = to4x1(np.array([a,b,c]),1.) - cV
    cdef np.ndarray abcV = np.dot(M, cV)
    return toV3(abcV)
    
cdef np.ndarray inverse(int d, int e, int f, np.ndarray M, np.ndarray C):
    cdef np.ndarray cV = to4x1(C,0)
    cdef np.ndarray defV = to4x1(np.array([d,e,f]), 1.)
    cdef np.ndarray V = np.dot(np.inv(M), defV)
    cdef np.ndarray abcV = V + cV
    return toV3(abcV)