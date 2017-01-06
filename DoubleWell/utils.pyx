from __future__ import division

import numpy as np

# Utils for energy function:
#  V(x) = v0 * ( ( (x/alpha)**2 - delta )**2 + (beta/alpha)*x )
#       = v0 * ( (1/alpha**4)*x**4 - 2*(delta/alpha**2)*x**2 + (beta/alpha)*x + delta**2 )


# Return coefficients of energy function for given parameters
def get_coefs(alpha=1, beta=0, delta=1, v0=5):
    ret_arr = np.zeros(5)
    ret_arr[0] = 1/alpha**4
    ret_arr[1] = 0
    ret_arr[2] = -2 * delta/alpha**2
    ret_arr[3] = beta/alpha
    ret_arr[4] = delta**2
    ret_arr = v0 * ret_arr

    return ret_arr

def get_coefs_prime(alpha=1, beta=0, delta=1, v0=5):
    ret_arr = np.zeros(4)
    ret_arr[0] = 4/alpha**4
    ret_arr[1] = 0
    ret_arr[2] = -4 * delta/alpha**2
    ret_arr[3] = beta/alpha
    ret_arr = v0 * ret_arr

    return ret_arr

def V(x, alpha=1, beta=0, delta=1, v0=5):
    args = (alpha, beta, delta, v0)
    coefs = get_coefs(*args)

    return coefs[0]*x**4 + coefs[1]*x**3 + coefs[2]*x**2 + coefs[3]*x + coefs[4]

def V_prime(x, alpha=1, beta=0, delta=1, v0=5):
    args = (alpha, beta, delta, v0)
    coefs = get_coefs_prime(*args)

    return coefs[0]*x**3 + coefs[1]*x**2 + coefs[2]*x + coefs[3]


# Find bin boundaries between x0 and x1 such that the energy gradient within a bin
#    is no more than bin_grad (in kT).  fn is function; fn(x) is energy (in kT).
# 
#   x0 < x1 must be true
def find_bounds(x0, x1, fn, bin_grad=1.0):

    assert x0 < x1

    if np.abs(fn(x0) - fn(x1)) <= bin_grad:
        return np.array([x0, x1])

    else:
        xmid = (x0+x1) / 2

        bounds_lower = find_bounds(x0, xmid, V, bin_grad)
        bounds_upper = find_bounds(xmid, x1, V, bin_grad)

        return np.append( bounds_lower, bounds_upper )


# Given the point of an extrema (x_pt) for polynomial p (np.poly1d object),
#   Find the x points to the right and left of x_extrema where 
#   abs( fn(x_[left, right]) - fn(x_extrema) ) == bin_grad
def find_extrema_bounds(x_pt, p, bin_grad=1.0):
    v_pt = p(x_pt)
    shift_coefs = p.coeffs.copy()
    shift_coefs[-1] -= np.abs(v_pt - bin_grad)
    # The roots of p_shift are the xpts where the potential
    #    is 1kT below v_max
    p_shift = np.poly1d(shift_coefs)
    shift_roots = p_shift.roots
    # Find the xpts closest to x_pt on left and right
    left_points = shift_roots[shift_roots < x_pt]
    right_points = shift_roots[shift_roots > x_pt]

    left_idx = np.argmin( np.abs(left_points - x_pt) )
    right_idx = np.argmin( np.abs(right_points - x_pt) )

    x_pt_left = left_points[left_idx]
    x_pt_right = right_points[right_idx]

    return x_pt_left, x_pt_right

# Construct bins between two wells, where boundaries are
#   chosen over approximately 1 kT.
#   This function starts by adding boundaries on either side of the maximum of V, Vmax
#   (i.e. x where Vmax - V(x) ~= 1kT) and then moves towards the basins until a boundary
#   is placed that is <= 1kT above the basin
#
#   Returns an array of the bin boundaries in x
def construct_bin_bounds(alpha=1, beta=0, delta=1, v0=5, bin_grad=1.0):

    args = (alpha, beta, delta, v0)

    coefs = get_coefs(*args)

    p = np.poly1d(coefs)
    p_prime = np.poly1d(p.deriv())
    p_curve = np.poly1d(p_prime.deriv())

    # Positions of extrema of p
    x_extrema = p_prime.r
    # Curvature - i.e. is V a min or max at x_extrema[i]?
    x_curve = p_curve(x_extrema)

    x_maxima = x_extrema[np.where(x_curve < 0)]
    x_minima = x_extrema[np.where(x_curve > 0)]

    bounds = np.array([])

    for x_max in x_maxima:
        v_max = p(x_max)

        # Find the xpts to right and left of x_max
        #    where potential is 1 kT below v_max


        x_max_left, x_max_right = find_extrema_bounds(x_max, p, bin_grad)

        # Find the locations of the wells to the left and right of x_max

        x_min_left_points = x_minima[x_minima < x_max]
        x_min_right_points = x_minima[x_minima > x_max]

        left_min_idx = np.argmin( np.abs(x_min_left_points - x_max) )
        right_min_idx = np.argmin( np.abs(x_min_right_points - x_max) )

        x_min_left = x_min_left_points[left_min_idx]
        x_min_right = x_min_right_points[right_min_idx]

        # find points to the left and right of x_min's that are 1 kT above v(x_min)
        x_min_left = find_extrema_bounds(x_min_left, p)[1]
        x_min_right = find_extrema_bounds(x_min_right, p)[0]

        if x_min_left < x_max:
            left_bounds = find_bounds(x_min_left, x_max_left, p, bin_grad)
        else:
            left_bounds = np.array([])

        if x_min_right > x_max:
            right_bounds = find_bounds(x_max_right, x_min_right, p, bin_grad)


        bounds = np.append(bounds, left_bounds)
        bounds = np.append(bounds, right_bounds)

        bounds = np.unique(bounds)
        bounds.sort()