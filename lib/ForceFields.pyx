import numpy as np
cimport numpy as np

cimport ForceFields
cimport cython

DTYPE = np.float32

cdef class Force:
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force):
        return 0

        
cdef class MuellerForce(Force):
    """
    2-dimensional potential defined in:
    Muller, K., and Brown, L.D. 1979. Location of saddle points and minimum energy paths by a constrained simplex
    optimization procedure. Theoret. Chem. Acta 53, 75-93.

    """

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force):
        cdef double x,y,fx,fy,b, xt, yt
        
        x = coords[0]
        y = coords[1]
        
        fx = 0
        fy = 0
        
        # 0
        b = -200.0*exp(-(x-1.0)**2 - 10.0*y**2)
        fx += -2.0*(x-1.0)*b
        fy += -20.0*y*b
        
        # 1
        b = -100.0*exp(-x**2 -10.0*(y-0.5)**2)
        fx += -2.0*x*b
        fy += -20.0*(y-0.5)*b
        
        #2
        xt = x+0.5
        yt = y-1.5
        b = -170.0*exp(-6.5*xt**2 + 11.0*xt*yt -6.5*yt**2)
        fx += (-13.0*xt + 11.0*yt)*b
        fy += (11.0*xt -13.0*yt)*b
        
        #3
        xt = x+1
        yt = y-1
        b = 15.0*exp(0.7*xt**2 + 0.6*xt*yt + 0.7*yt**2)
        fx += (1.4*xt + 0.6*yt)*b
        fy += (0.6*xt + 1.4*yt)*b 
        
        
        force[0] = -fx
        force[1] = -fy
        
        return 0
    
cdef class RuggedMuellerForce(Force):
    """
    Rugged version of the 2-dimensional potential defined in:
    Muller, K., and Brown, L.D. 1979. Location of saddle points and minimum energy paths by a constrained simplex
    optimization procedure. Theoret. Chem. Acta 53, 75-93.

    """

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force):
        cdef double x,y,fx,fy,b,xt,yt
        cdef double twopik,a

        twopik = 5.0*6.283185307179586
        a = 9.0

        x = coords[0]
        y = coords[1]
        
        fx = 0
        fy = 0
        
        # 0
        b = -200.0*exp(-(x-1.0)**2 - 10.0*y**2)
        fx += -2.0*(x-1.0)*b
        fy += -20.0*y*b
        
        # 1
        b = -100.0*exp(-x**2 -10.0*(y-0.5)**2)
        fx += -2.0*x*b
        fy += -20.0*(y-0.5)*b
        
        #2
        xt = x+0.5
        yt = y-1.5
        b = -170.0*exp(-6.5*xt**2 + 11.0*xt*yt -6.5*yt**2)
        fx += (-13.0*xt + 11.0*yt)*b
        fy += (11.0*xt -13.0*yt)*b
        
        #3
        xt = x+1
        yt = y-1
        b = 15.0*exp(0.7*xt**2 + 0.6*xt*yt + 0.7*yt**2)
        fx += (1.4*xt + 0.6*yt)*b
        fy += (0.6*xt + 1.4*yt)*b 
        
        # rugged terms
        fx += a*twopik*cos(twopik*x)*sin(twopik*y)
        fy += a*twopik*sin(twopik*x)*cos(twopik*y)
        
        force[0] = -fx
        force[1] = -fy
        
        return 0

cdef class DoubleWellForce(Force):
    '''
    Simple one-dimensional double-well potential
    '''
    cdef double _alpha, _beta, _delta, _v0

    def __init__(self, alpha=1, beta=0, delta=1, v0=5):
        self._alpha = alpha
        self._beta = beta
        self._delta = delta
        self._v0 = v0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int evaluate(self, np.ndarray[DTYPE_t,ndim=1] coords, np.ndarray[DTYPE_t,ndim=1] force):
        cdef double x,fx
        cdef double alpha, beta, delta, v0, x_over_alpha

        alpha = self._alpha
        beta = self._beta
        delta = self._delta
        v0 = self._v0
        
        x = coords[0]

        x_over_alpha = x / alpha

        fx = 4 * ( ( x_over_alpha**2 - delta ) * x/alpha**2 ) + (beta/alpha)
        fx = v0 * fx

        force[0] = -fx

        return 0

cdef class Dickson2dPeriodicForce(Force):
    """
    2D Periodic potential from Dickson, et al (2009) Nonequilibrium umbrella sampling in 
    spaces of many order parameters. J Chem Phys 130 074104
    """
    cdef double _alpha

    def __init__(self,alpha):
        self._alpha = alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int evaluate(self,np.ndarray[DTYPE_t,ndim=1] coords, np.ndarray[DTYPE_t,ndim=1] force):
        cdef double x,y,fx,fy
        cdef double gam,alpha,fext,twopi,s2piy,c2piy
        
        twopi = 6.283185307179586

        gam = 6.0
        alpha = self._alpha
        fext = 4.8

        x = coords[0]
        y = coords[1]

        s2piy = sin(twopi*y)
        c2piy = cos(twopi*y)
        fx = -gam*(s2piy - 2.0*x)
        fy = -0.5*twopi*(2.0*gam*x*c2piy + 2.0*alpha*s2piy - gam*s2piy*c2piy)
        #fx = -gam*(sin(twopi*y) - 2.0*x)
        #fy = 0.25*twopi*(-4.0*alpha*sin(twopi*y) - 4.0*gam*x*cos(twopi*y) + gam*sin(2.0*twopi*y))

        force[0] = -fx
        force[1] = -(fy-fext)

        return 0

cdef class Dickson2dPeriodicForce_revised(Force):
    """
    2D Periodic potential from Dickson, et al (2009) Nonequilibrium umbrella sampling in 
    spaces of many order parameters. J Chem Phys 130 074104

    ***Parameters are revised based on personal communication with Aaron Dinner 
    """
    cdef double _alpha

    def __init__(self,alpha):
        self._alpha = alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int evaluate(self,np.ndarray[DTYPE_t,ndim=1] coords, np.ndarray[DTYPE_t,ndim=1] force):
        cdef double x,y,fx,fy
        cdef double gam,alpha,fext,twopi,s2piy,c2piy
        
        twopi = 6.283185307179586

        gam = 2.25
        alpha = self._alpha
        fext = 1.8

        x = coords[0]
        y = coords[1]

        s2piy = sin(twopi*y)
        c2piy = cos(twopi*y)
        fx = -gam*(s2piy - 2.0*x)
        fy = -0.5*twopi*(2.0*gam*x*c2piy + 2.0*alpha*s2piy - gam*s2piy*c2piy)
        #fx = -gam*(sin(twopi*y) - 2.0*x)
        #fy = 0.25*twopi*(-4.0*alpha*sin(twopi*y) - 4.0*gam*x*cos(twopi*y) + gam*sin(2.0*twopi*y))

        force[0] = -fx
        force[1] = -(fy-fext)

        return 0

cdef class Dickson2dRingForce(Force):
    """
    2D Ring-shaped potential from Dickson, et al (2009) Separating forward and backward 
    pathways in nonequilibrium umbrellasampling. J Chem Phys 131 154104
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int evaluate(self,np.ndarray[DTYPE_t,ndim=1] coords, np.ndarray[DTYPE_t,ndim=1] force):
        cdef double x,y,fx,fy
        cdef double gamma,alpha,chi1,chi2,fext
        cdef double att,x2y2,A,B1,B2,r,invr

        alpha = 3.0
        gamma = 3.0
        chi1 = 2.25
        chi2 = 4.5

        fext = 7.2

        x = coords[0]
        y = coords[1]

        x2y2 = x**2 + y**2
        r = sqrt(x2y2)
        invr = 1.0/r
        A = 2.0*alpha*(1.0 - gamma*invr)

        att = atan2(y,x)
        B1 = chi1*sin(2.0*att)*invr*invr #/x2y2
        B2 = chi2*sin(4.0*att)*invr*invr #/x2y2

        fx = x*A - 2.0*y*B1 - 4.0*y*B2
        fy = y*A + 2.0*x*B1 + 4.0*x*B2

        force[0] = -(fx - fext*invr*sin(att))
        force[1] = -(fy + fext*invr*cos(att))

        return 0
