import cython
cimport numpy
from libc.math cimport sqrt

ctypedef numpy.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def custom_metric(numpy.ndarray[DTYPE_t, ndim=1] x1, numpy.ndarray[DTYPE_t, ndim=1] x2, int ncts, double ncate):
    cdef double euc = 0.0
    cdef double ham = 0.0
    cdef double d = 0.0
    cdef int i, j

    for i in range(0,ncts):
        d = x1[i] - x2[i]
        euc = euc + d*d
    euc = sqrt(euc)

    for j in range(ncts, int(ncts+ncate)):
        if abs(x1[j] - x2[j]) > 0.9999:
            ham = ham + 1
    ham = ham/ncate

    return (euc + ham)